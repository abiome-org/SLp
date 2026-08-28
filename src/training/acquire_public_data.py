"""Acquire high-value public perturbation datasets with checksum manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
GEO_POLICY = "https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html"


def geo(accession: str, files: list[str], description: str) -> dict[str, object]:
    prefix = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{accession[:-3]}nnn/{accession}/suppl"
    return {
        "accession": accession,
        "description": description,
        "policy": GEO_POLICY,
        "files": [(name, f"{prefix}/{name}") for name in files],
    }


GSE220974 = [
    "GSE220974_D1_protospacer_calls_per_cell.csv.gz",
    "GSE220974_D2_protospacer_calls_per_cell.csv.gz",
    "GSE220974_D3_protospacer_calls_per_cell.csv.gz",
    "GSE220974_K562_cell_metadata.csv.gz",
    "GSE220974_RNA_barcodes.tsv.gz",
    "GSE220974_RNA_features.tsv.gz",
    "GSE220974_RNA_matrix.mtx.gz",
    "GSE220974_S1Sa_protospacer_calls_per_cell.csv.gz",
    "GSE220974_S2Sp_protospacer_calls_per_cell.csv.gz",
    "GSE220974_S3SaSp_protospacer_calls_per_cell.csv.gz",
    "GSE220974_features.csv.gz",
    "GSE220974_gRNA_barcodes.tsv.gz",
    "GSE220974_gRNA_features.tsv.gz",
    "GSE220974_gRNA_matrix.mtx.gz",
]
GSE337988_PILOT = [
    "GSE337988_NGS6194_crispr_demux_index_map.csv.gz",
    "GSE337988_pilot_samplesheet.csv.gz",
    *[
        f"GSE337988_pilot_processed_objects_MOI_{moi}_{suffix}"
        for moi in ("0.1", "0.2", "0.5", "1.0", "3.0", "5.0")
        for suffix in ("assays.h5", "se.rds")
    ],
]
DATASETS = {
    "gse220974": geo(
        "GSE220974",
        GSE220974,
        "K562 CRISPRa/CRISPRi singles and pairwise Perturb-seq",
    ),
    "gse221321": geo(
        "GSE221321",
        ["GSE221321_RAW.tar"],
        "THP-1 compressed Perturb-seq over 598 immune-response genes",
    ),
    "gse337988_pilot": geo(
        "GSE337988",
        GSE337988_PILOT,
        "DLD1 CRISPRi Perturb-seq pilot across six multiplicities of infection",
    ),
    "gse278572": geo(
        "GSE278572",
        [
            "GSE278572_barcodes.tsv.gz",
            "GSE278572_features.tsv.gz",
            "GSE278572_matrix.mtx.gz",
            "GSE278572_protospacer_calls_per_cell.csv.gz",
        ],
        "Primary CD4 Treg/Teff Perturb-CITE-seq across donors and activation contexts",
    ),
    "gse213957": geo(
        "GSE213957",
        [
            "GSE213957_THP1-CaRPool-seq.metadata.tsv.gz",
            "GSE213957_THP1-CaRPool-seq_and_GSE213957_HEX293FTstabRNA.tar.gz",
            "GSE213957_THP1_Cas13d_combinatorial_pooledScreen.counts.csv.gz",
            "GSE213957_THP1_Cas13d_pooledScreen.counts.csv.gz",
        ],
        "THP-1 Cas13 CaRPool-seq single and combinatorial RNA perturbations",
    ),
    "gse200201": geo(
        "GSE200201",
        ["GSE200201_RAW.tar", "filelist.txt"],
        "MOLM13 single and combinatorial mSWI/SNF Perturb-seq",
    ),
    "gse208240": geo(
        "GSE208240",
        [
            "GSE208240_CRISPRi_perturbseq_sarscov2_filtered.tar.gz",
            "GSE208240_viral_references.tar.gz",
        ],
        "Calu-3 CRISPRi Perturb-seq with SARS-CoV-2 infection and bystander states",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def acquire(name: str, output: Path, verify_only: bool = False) -> dict[str, object]:
    dataset = DATASETS[name]
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for filename, url in dataset["files"]:
        destination = output / filename
        if not verify_only:
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--continue-at",
                    "-",
                    "--retry",
                    "6",
                    "--output",
                    str(destination),
                    url,
                ],
                check=True,
            )
        if not destination.is_file():
            raise FileNotFoundError(destination)
        files.append(
            {
                "path": filename,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "source_url": url,
            }
        )
    manifest = {
        "schema": "slp-public-source-manifest-v1",
        "dataset": name,
        "accession": dataset["accession"],
        "description": dataset["description"],
        "distribution_policy": dataset["policy"],
        "license_note": (
            "NCBI places no restrictions on use or distribution of GEO data, "
            "while warning that submitters may assert third-party rights."
        ),
        "files": files,
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", choices=tuple(DATASETS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        print(json.dumps({key: value["description"] for key, value in DATASETS.items()}, indent=2))
        return
    if args.dataset is None:
        parser.error("dataset is required unless --list is used")
    output = args.output or ROOT / "data/raw" / args.dataset
    print(json.dumps(acquire(args.dataset, output, args.verify_only), indent=2))


if __name__ == "__main__":
    main()
