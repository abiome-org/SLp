"""Download raw sources into data/raw/<source>/ and record sha256 in data/raw/MANIFEST.tsv.

    uv run python -m slpbench.fetch                 # everything
    uv run python -m slpbench.fetch slkb biomart    # selected sources

Large figshare files download far faster with `python -m slpbench.pget URL DEST`.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

RAW = Path("data/raw")
PINS = Path("reference/raw_sha256sums.txt")


def pinned_hashes() -> dict[str, str]:
    if not PINS.exists():
        return {}
    return {name: digest for digest, name in (line.split("  ./", 1) for line in PINS.read_text().splitlines())}

# source -> list of (filename, url)
SOURCES: dict[str, list[tuple[str, str]]] = {
    "slkb": [
        ("SQL_Dumps.zip", "https://ndownloader.figshare.com/files/41055392"),
        ("README.md", "https://ndownloader.figshare.com/files/42103311"),
    ],
    "harle2025": [
        ("METADATA.tar.gz", "https://ndownloader.figshare.com/files/46763992"),
        ("DATA.tar.gz", "https://ndownloader.figshare.com/files/46763995"),
    ],
    "in4mer2024": [
        (f"Supp_table{n}.{ext}", f"https://ndownloader.figshare.com/files/{fid}")
        for n, ext, fid in [
            (1, "xlsx", 44435633), (2, "xlsx", 44435630), (3, "xlsx", 44435636),
            (4, "txt", 44435645), (5, "txt", 44435651), (6, "xlsx", 44436191),
            (7, "txt", 44435639), (8, "txt", 44435648), (9, "txt", 44435654),
        ]
    ],
    "fischer2015_dmel": [
        ("Interactions.rda", "https://raw.githubusercontent.com/bioc/DmelSGI/HEAD/data/Interactions.rda"),
        ("mainEffects.rda", "https://raw.githubusercontent.com/bioc/DmelSGI/HEAD/data/mainEffects.rda"),
    ],
    "heigwer2023_dmel": [
        ("interactions_stat_tested_bias_corrected.csv.gz", "https://ndownloader.figshare.com/files/31506566"),
    ],
    "horn2011_dmel": [
        ("nmeth1581_MOESM15.xls", "https://static-content.springer.com/esm/art%3A10.1038%2Fnmeth.1581/MediaObjects/41592_2011_BFnmeth1581_MOESM15_ESM.xls"),
    ],
    "frost2012_spombe": [
        ("mmc2_averaged.zip", "https://ars.els-cdn.com/content/image/1-s2.0-S0092867412005739-mmc2.zip"),
        ("mmc3_unaveraged.zip", "https://ars.els-cdn.com/content/image/1-s2.0-S0092867412005739-mmc3.zip"),
    ],
    "dualcrispri2025_spneumo": [
        ("mmc4.csv", "https://ars.els-cdn.com/content/image/1-s2.0-S2405471225002418-mmc4.csv"),
        ("mmc3_counts.csv", "https://ars.els-cdn.com/content/image/1-s2.0-S2405471225002418-mmc3.csv"),
    ],
    "costanzo2016_scer": [
        ("pairwise.zip", "https://thecellmap.org/costanzo2016/data_files/Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip"),
    ],
    "ryan2012_spombe": [
        ("mmc5_averaged.zip", "https://ars.els-cdn.com/content/image/1-s2.0-S1097276512004443-mmc5.zip"),
        ("mmc4_unaveraged.zip", "https://ars.els-cdn.com/content/image/1-s2.0-S1097276512004443-mmc4.zip"),
    ],
    "spidr2025": [
        ("MOESM5_pairs.csv", "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-025-08815-4/MediaObjects/41586_2025_8815_MOESM5_ESM.csv"),
        ("MOESM6_followup.xlsx", "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-025-08815-4/MediaObjects/41586_2025_8815_MOESM6_ESM.xlsx"),
        ("MOESM9_counts.txt", "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-025-08815-4/MediaObjects/41586_2025_8815_MOESM9_ESM.txt"),
    ],
    "harle2025_calls": [
        ("MOESM1_additional_file1.xlsx", "https://static-content.springer.com/esm/art%3A10.1186%2Fs13059-025-03737-w/MediaObjects/13059_2025_3737_MOESM1_ESM.xlsx"),
    ],
    "dede2020": [
        ("MOESM3.txt", "https://static-content.springer.com/esm/art%3A10.1186%2Fs13059-020-02173-2/MediaObjects/13059_2020_2173_MOESM3_ESM.txt"),
    ],
    "chymera2020": [
        ("MOESM10.xlsx", "https://static-content.springer.com/esm/art%3A10.1038%2Fs41587-020-0437-z/MediaObjects/41587_2020_437_MOESM10_ESM.xlsx"),
    ],
    "flister2025": [
        ("mmc3.xlsx", "https://ars.els-cdn.com/content/image/1-s2.0-S2211124725012835-mmc3.xlsx"),
        ("mmc6.xlsx", "https://ars.els-cdn.com/content/image/1-s2.0-S2211124725012835-mmc6.xlsx"),
        ("mmc7.xlsx", "https://ars.els-cdn.com/content/image/1-s2.0-S2211124725012835-mmc7.xlsx"),
    ],
    "chou2025": [
        ("chou2025.xlsx", "https://ndownloader.figshare.com/files/55382033"),
    ],
    "ryanlab2025_bench": [  # uniform zdLFC re-scoring of paralog screens (figshare 27868350)
        (n, f"https://ndownloader.figshare.com/files/{fid}") for n, fid in [
            ("ITO.csv", 52694552), ("DeDe_zdLFC.csv", 52694549), ("Thompson_zdLFC.csv", 52694558),
            ("ChymeraHAP1.csv", 52694546), ("ChymeraRPE1.csv", 52694561),
            ("Parrish_Hela.csv", 52694555), ("Parrish_PC9.csv", 52694564),
            ("combn_gene_calls.tsv", 52694945),
        ]
    ],
    "cellosaurus": [
        ("cellosaurus.txt", "https://ftp.expasy.org/databases/cellosaurus/cellosaurus.txt"),
    ],
    "depmap": [
        ("Model_24Q4.csv", "https://ndownloader.figshare.com/files/51065297"),
        ("CRISPRGeneEffect_24Q4.csv", "https://ndownloader.figshare.com/files/51064667"),
    ],
    "ids": [
        ("hgnc_complete_set.txt", "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"),
        ("SGD_features.tab", "https://downloads.yeastgenome.org/curation/chromosomal_feature/SGD_features.tab"),
        ("pombase_gene_IDs_names_products.tsv", "https://www.pombase.org/data/names_and_identifiers/gene_IDs_names_products.tsv"),
        ("fb_synonym.tsv.gz", "https://s3ftp.flybase.org/releases/current/precomputed_files/synonyms/fb_synonym_fb_2026_03.tsv.gz"),
    ],
    "orthology": [
        ("ORTHOLOGY-ALLIANCE_COMBINED.tsv.gz", "https://fms.alliancegenome.org/download/ORTHOLOGY-ALLIANCE_COMBINED.tsv.gz"),
        ("pombe-cerevisiae-orthologs.tsv", "https://www.pombase.org/data/orthologs/pombe-cerevisiae-orthologs.tsv"),
        ("pombe-human-orthologs.tsv", "https://www.pombase.org/data/orthologs/pombe-human-orthologs.tsv"),
        ("hcop_human_scerevisiae.txt.gz", "https://storage.googleapis.com/public-download-files/hcop/human_s.cerevisiae_hcop_fifteen_column.txt.gz"),
        ("hcop_human_spombe.txt.gz", "https://storage.googleapis.com/public-download-files/hcop/human_s.pombe_hcop_fifteen_column.txt.gz"),
    ],
    "paralogs": [
        ("ens111_human_SL.csv", "https://raw.githubusercontent.com/cancergenetics/paralog_seq_similarity/main/data/ens111_human_SL.csv"),
        ("ens111_yeast_SL.csv", "https://raw.githubusercontent.com/cancergenetics/paralog_seq_similarity/main/data/ens111_yeast_SL.csv"),
        ("ens111_human_similPerc.csv", "https://zenodo.org/api/records/14973633/files/ens111_human_similPerc.csv/content"),
        ("compara116_spombe_homologies.tsv.gz", "https://ftp.ensemblgenomes.ebi.ac.uk/pub/fungi/current/tsv/ensembl-compara/homologies/schizosaccharomyces_pombe/Compara.116.protein_default.homologies.tsv.gz"),
    ],
}


# Ensembl BioMart paralog queries (POST); written to data/raw/paralogs/ensembl_<ds>_paralogs.tsv
BIOMART_URL = "https://useast.ensembl.org/biomart/martservice"
BIOMART_DATASETS = ["hsapiens", "scerevisiae", "dmelanogaster"]


def biomart_query(ds: str) -> str:
    attrs = ["ensembl_gene_id", "external_gene_name", f"{ds}_paralog_ensembl_gene", f"{ds}_paralog_associated_gene_name",
             f"{ds}_paralog_orthology_type", f"{ds}_paralog_subtype", f"{ds}_paralog_perc_id", f"{ds}_paralog_perc_id_r1"]
    a = "".join(f'<Attribute name="{x}"/>' for x in attrs)
    return ('<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query><Query virtualSchemaName="default" formatter="TSV" '
            'header="1" uniqueRows="1" completionStamp="1" datasetConfigVersion="0.6">'
            f'<Dataset name="{ds}_gene_ensembl" interface="default"><Filter name="biotype" value="protein_coding"/>{a}'
            '</Dataset></Query>')


def fetch_biomart(force: bool = False) -> None:
    for ds in BIOMART_DATASETS:
        dest = RAW / "paralogs" / f"ensembl_{ds}_paralogs.tsv"
        if dest.exists() and not force:
            print(f"skip {dest}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"get  biomart {ds} -> {dest}", flush=True)
        subprocess.run(["curl", "-sSf", "-m", "900", "-o", str(dest), "--data-urlencode", f"query={biomart_query(ds)}",
                        BIOMART_URL], check=True)
        if "[success]" not in dest.read_text()[-200:]:
            raise RuntimeError(f"incomplete BioMart result for {ds}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(source: str, force: bool = False) -> None:
    out_dir = RAW / source
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = RAW / "MANIFEST.tsv"
    pins = pinned_hashes()
    for name, url in SOURCES[source]:
        dest = out_dir / name
        expected = pins.get(f"{source}/{name}")
        if dest.exists() and not force:
            if expected and sha256(dest) != expected:
                raise ValueError(f"existing raw file differs from pinned sha256: {dest}")
            print(f"skip {dest}")
            continue
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"get  {url} -> {dest}", flush=True)
        subprocess.run(
            ["curl", "-fL", "--retry", "5", "--retry-delay", "5", "-sS", "-A", "slpbench/0.1", "-o", str(tmp), url],
            check=True,
        )
        if expected and sha256(tmp) != expected:
            tmp.unlink()
            raise ValueError(f"download differs from pinned sha256: {dest}")
        tmp.rename(dest)
        with manifest.open("a") as m:
            m.write(f"{source}\t{name}\t{dest.stat().st_size}\t{sha256(dest)}\t{url}\t{time.strftime('%Y-%m-%d')}\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="*", default=list(SOURCES) + ["biomart"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    failed = []
    for s in args.sources:
        try:
            fetch_biomart(args.force) if s == "biomart" else fetch(s, args.force)
        except subprocess.CalledProcessError as e:
            print(f"FAILED {s}: {e}", file=sys.stderr)
            failed.append(s)
    if failed:
        sys.exit(f"failed: {failed}")


if __name__ == "__main__":
    main()
