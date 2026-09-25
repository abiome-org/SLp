"""Download inputs for the extra orthology/paralogy graph (families_extra).

Writes files under data/raw/orthology_extra/ and data/raw/ids/, then rewrites
reference/fetch/orthology.tsv (filename<TAB>url) and data/raw/orthology_extra/SHA256SUMS.
Usage: uv run python scripts/orthology_extra/fetch.py [--force]
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from slbench.fetch import BIOMART_URL, biomart_query

RAW = Path("data/raw")
ENS = "https://ftp.ensembl.org/pub/release-116/fasta"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&rettype=gbwithparts&retmode=text&id="

FILES = [
    # proteomes (one FASTA per species; longest isoform per gene is chosen at build time)
    ("orthology_extra/proteomes/human.pep.fa.gz", f"{ENS}/homo_sapiens/pep/Homo_sapiens.GRCh38.pep.all.fa.gz"),
    ("orthology_extra/proteomes/mmus.pep.fa.gz", f"{ENS}/mus_musculus/pep/Mus_musculus.GRCm39.pep.all.fa.gz"),
    ("orthology_extra/proteomes/cele.pep.fa.gz", f"{ENS}/caenorhabditis_elegans/pep/Caenorhabditis_elegans.WBcel235.pep.all.fa.gz"),
    ("orthology_extra/proteomes/dmel.pep.fa.gz", f"{ENS}/drosophila_melanogaster/pep/Drosophila_melanogaster.BDGP6.54.pep.all.fa.gz"),
    ("orthology_extra/proteomes/scer.pep.fa.gz", f"{ENS}/saccharomyces_cerevisiae/pep/Saccharomyces_cerevisiae.R64-1-1.pep.all.fa.gz"),
    ("orthology_extra/proteomes/spom.pep.fa.gz", "https://www.pombase.org/data/genome_sequence_and_features/feature_sequences/peptide.fa.gz"),
    ("orthology_extra/proteomes/calb.pep.fa.gz", "http://www.candidagenome.org/download/sequence/C_albicans_SC5314/Assembly22/current/C_albicans_SC5314_A22_current_default_protein.fasta.gz"),
    # bacterial genomes (GenBank with translations; locus tags are the canonical IDs)
    ("orthology_extra/proteomes/spne_D39V_CP027540.1.gb", EFETCH + "CP027540.1"),
    ("orthology_extra/proteomes/ecol_MG1655_U00096.3.gb", EFETCH + "U00096.3"),
    ("orthology_extra/proteomes/bsub_168_AL009126.3.gb", EFETCH + "AL009126.3"),
    ("orthology_extra/proteomes/mtub_H37Rv_AL123456.3.gb", EFETCH + "AL123456.3"),
    ("orthology_extra/proteomes/saur_NCTC8325_CP000253.1.gb", EFETCH + "CP000253.1"),
    # validation only: eggNOG 5.0 LUCA-level (taxid 1) orthologous groups
    ("orthology_extra/eggnog5_1_members.tsv.gz", "http://eggnog5.embl.de/download/eggnog_5.0/per_tax_level/1/1_members.tsv.gz"),
    # ID resolver tables
    ("ids/MRK_List2.rpt", "https://www.informatics.jax.org/downloads/reports/MRK_List2.rpt"),
    ("ids/c_elegans.WS298.geneIDs.txt.gz", "https://ftp.ebi.ac.uk/pub/databases/wormbase/releases/WS298/species/c_elegans/PRJNA13758/annotation/c_elegans.PRJNA13758.WS298.geneIDs.txt.gz"),
    ("ids/c_elegans.WS298.geneOtherIDs.txt.gz", "https://ftp.ebi.ac.uk/pub/databases/wormbase/releases/WS298/species/c_elegans/PRJNA13758/annotation/c_elegans.PRJNA13758.WS298.geneOtherIDs.txt.gz"),
    ("ids/C_albicans_SC5314_A22_chromosomal_feature.tab", "http://www.candidagenome.org/download/chromosomal_feature_files/C_albicans_SC5314/C_albicans_SC5314_A22_current_chromosomal_feature.tab"),
]
BIOMART = {"mmusculus": "orthology_extra/ensembl_mmusculus_paralogs.tsv",
           "celegans": "orthology_extra/ensembl_celegans_paralogs.tsv"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(force: bool = False) -> None:
    rows = []
    for rel, url in FILES:
        dest = RAW / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if force or not dest.exists() or dest.stat().st_size == 0:
            print("get", url, flush=True)
            subprocess.run(["curl", "-sSfL", "--retry", "3", "-m", "1800", "-o", str(dest), url], check=True)
        rows.append((rel, url))
    for ds, rel in BIOMART.items():
        dest = RAW / rel
        if force or not dest.exists():
            print("get biomart", ds, flush=True)
            subprocess.run(["curl", "-sSf", "-m", "1800", "-o", str(dest), "--data-urlencode",
                            f"query={biomart_query(ds)}", BIOMART_URL], check=True)
            if "[success]" not in dest.read_text()[-200:]:
                raise RuntimeError(f"incomplete BioMart result for {ds}")
        rows.append((rel, f"{BIOMART_URL} (POST, slbench.fetch.biomart_query('{ds}'))"))
    Path("reference/fetch").mkdir(parents=True, exist_ok=True)
    Path("reference/fetch/orthology.tsv").write_text("".join(f"{r}\t{u}\n" for r, u in rows))
    sums = "".join(f"{sha256(RAW / r)}  {r}\n" for r, _ in rows)
    (RAW / "orthology_extra/SHA256SUMS").write_text(sums)
    print(sums)


if __name__ == "__main__":
    main(force="--force" in sys.argv)
