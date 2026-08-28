"""Build a Calu-3 CRISPRi Perturb-seq pack from GEO GSE208240."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = "GSE208240_CRISPRi_perturbseq_sarscov2_filtered.tar.gz"
BASE = "data/sunshine/perturb_seq/sars_cov_2_geo_upload/CRISPRi_perturbseq_sarscov2_filtered"
MEMBERS = ("barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz", "cell_identities.csv")
KEEP_MATCH_TYPES = ("exact_match", "likely_match")
MAX_CARDINALITY = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_extracted(raw: Path) -> Path:
    extracted = raw / "extracted"
    expected = [f"{BASE}/{name}" for name in MEMBERS]
    missing = [name for name in expected if not (extracted / name).is_file()]
    if missing:
        extracted.mkdir(parents=True, exist_ok=True)
        with tarfile.open(raw / ARCHIVE) as archive:
            members = {member.name: member for member in archive.getmembers()}
            unavailable = set(missing) - set(members)
            if unavailable:
                raise ValueError(f"archive lacks expected members: {sorted(unavailable)}")
            for name in missing:
                archive.extract(members[name], extracted, filter="data")
    return extracted


def guide_doses(identity: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in str(identity).split(";"):
        target = token.split("_", 1)[0]
        if target.lower() != "non-targeting":
            counts[target] = counts.get(target, 0) + 1
    return counts


def element_genes(element: str) -> list[str]:
    body = element.split("_", 2)[2]
    return [] if body.startswith("non_targeting") else body.split("_")


def build(raw: Path, output: Path) -> dict[str, object]:
    raw = Path(raw)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    extracted = ensure_extracted(raw)
    base = extracted / BASE

    calls = pd.read_csv(base / "cell_identities.csv")
    barcodes = pd.read_csv(base / "barcodes.tsv.gz", sep="\t", header=None).iloc[:, 0].to_numpy(dtype=str)
    features = pd.read_csv(base / "features.tsv.gz", sep="\t", header=None)
    gene = features.iloc[:, 2].astype(str).eq("Gene Expression").to_numpy()
    feature_names = features.loc[gene, 1].to_numpy(dtype=str)

    calls = calls[calls["good_coverage"].astype(bool) & calls["match_type"].isin(KEEP_MATCH_TYPES)]
    calls = calls[calls["matched_library_element"].notna()]
    doses_per_cell = [guide_doses(identity) for identity in calls["guide_identity"]]
    designs = [element_genes(element) for element in calls["matched_library_element"].astype(str)]
    # Retain only cells whose observed guide targets reproduce the matched design exactly;
    # scrambled pairs and multiplets are ambiguous and excluded.
    consistent = np.asarray(
        [
            (sorted(doses) == sorted(set(design))) and (set(doses) == set(design)) and (bool(design) or not doses)
            for doses, design in zip(doses_per_cell, designs)
        ]
    )
    calls = calls[consistent]
    doses_per_cell = [doses for doses, keep in zip(doses_per_cell, consistent) if keep]
    calls["actions"] = [tuple(sorted(doses.items())) for doses in doses_per_cell]
    positions = pd.Index(barcodes).get_indexer(calls["cell_barcode"].astype(str))
    if np.any(positions < 0) or len(np.unique(positions)) != len(positions):
        raise ValueError("perturbation calls do not map one-to-one to retained barcodes")

    cardinality = np.asarray([len(row) for row in calls["actions"]])
    valid = cardinality <= MAX_CARDINALITY
    if not valid.all():
        calls = calls[valid]
        positions = positions[valid]
        cardinality = cardinality[valid]
        doses_per_cell = [doses for doses, keep in zip(doses_per_cell, valid) if keep]

    action_names = np.asarray(
        sorted(set().union(*(set(name for name, _ in row) for row in calls["actions"])))
    )
    action_id = {name: index for index, name in enumerate(action_names)}
    keys = np.asarray(["+".join(f"{name}@{dose}" for name, dose in row) for row in calls["actions"]])
    unique, group = np.unique(keys, return_inverse=True)
    counts = np.bincount(group, minlength=len(unique))
    supported = counts >= 4
    supported_group = np.flatnonzero(supported)

    matrix = mmread(base / "matrix.mtx.gz").tocsr()
    if matrix.shape != (len(features), len(barcodes)):
        raise ValueError("matrix axes do not match barcodes and features")
    expression = matrix[gene][:, positions].T.tocsr().astype("float32")
    library_size = np.asarray(expression.sum(axis=1)).ravel().clip(1)
    expression = expression.multiply((1e4 / library_size)[:, None]).tocsr()
    np.log1p(expression.data, out=expression.data)

    assignment = sparse.csr_matrix(
        (
            np.ones(len(group), dtype="float32"),
            (group, np.arange(len(group))),
        ),
        shape=(len(unique), len(group)),
    )
    means = (assignment[supported_group] @ expression).multiply(
        (1 / counts[supported_group])[:, None]
    ).toarray()
    supported_keys = [key for key in unique[supported_group]]
    control = np.asarray([key == "" for key in supported_keys])
    control_cells = int(counts[supported_group][control].sum())
    if control_cells < 32:
        raise ValueError("insufficient supported non-targeting control cells")
    control_mean = means[control].sum(axis=0) / control_cells
    outcome = ~control
    parsed = [parse_key(key) for key in unique[supported_group][outcome]]
    actions = np.full((len(parsed), MAX_CARDINALITY), -1, dtype="int32")
    doses = np.zeros((len(parsed), MAX_CARDINALITY), dtype="int8")
    for row, members in enumerate(parsed):
        actions[row, : len(members)] = [action_id[target] for target, _ in members]
        doses[row, : len(members)] = [dose for _, dose in members]
    modes = np.full(actions.shape, "", dtype="<U16")
    modes[actions >= 0] = "repression"

    artifact = output / "gse208240_calu3_crispri_pseudobulk_v1.npz"
    np.savez_compressed(
        artifact,
        actions=actions,
        action_modes=modes,
        action_doses=doses,
        action_names=action_names,
        target=(means[outcome] - control_mean).astype("float32"),
        target_semantics=np.asarray("perturbation_delta"),
        target_feature_name=feature_names,
        cardinality=(actions >= 0).sum(axis=1).astype("int8"),
        source_id=np.asarray("GSE208240"),
        context_id=np.asarray("CALU3"),
        experimental_condition_id=np.asarray("CRISPRi|duration=unreported|infection=unreported"),
        sample_id=np.asarray("single_library"),
        replicate_id=np.asarray("single_library"),
        cell_count=counts[supported_group][outcome].astype("int32"),
        observation_unit=np.asarray("pseudobulk"),
    )
    audit = {
        "schema": "slp-data-release-audit-v1",
        "release_id": "data/perturbseq/gse208240-calu3-crispri-pseudobulk-v1",
        "source": {
            "name": "CRISPRi Perturb-seq of SARS-CoV-2 host factors in Calu-3 cells (filtered public upload)",
            "accession": "GSE208240",
            "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE208240",
            "source_manifest_sha256": sha256(raw / "source_manifest.json"),
        },
        "license": {
            "id": "NCBI-GEO-PUBLIC-DATA",
            "evidence": "NCBI states that it places no restrictions on use or distribution of GEO data, while noting that submitters may assert rights.",
            "policy": "https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html",
        },
        "transformations": "Good-coverage cells with an exact or likely library-element match whose observed guide targets reproduce that design were grouped by gene-level action multiset; guide counts within a target are retained as dose. Supported groups with at least four cells were normalized per cell as log1p counts per 10,000, averaged, and differenced from the pooled non-targeting control mean.",
        "schema_description": "NPZ with up to eight CRISPRi actions and guide doses, full-gene expression perturbation deltas, Calu-3 context, and pseudobulk cell counts.",
        "population": "Calu-3 human lung adenocarcinoma cells",
        "endpoints": ["mean per-cell log1p(CP10K) expression change from pooled non-targeting controls"],
        "split_construction": "No train/test split is embedded. The hard generalization gate constructs deterministic folds downstream.",
        "exclusions": "Scrambled-pair calls without a matched library element and multiplets were excluded as ambiguous, as were matched cells whose observed guide targets disagree with the matched design. The filtered upload provides no per-cell infection annotation, so infected and bystander states are pooled and the condition is explicit as infection=unreported. Perturbation duration was not recovered from the deposit. Cells outside supported action groups were excluded.",
        "rows": len(actions),
        "rows_by_cardinality": {
            str(k): int(v)
            for k, v in zip(*np.unique((actions >= 0).sum(axis=1), return_counts=True))
        },
        "unique_action_targets": len(action_names),
        "expression_features": len(feature_names),
        "control_cells": control_cells,
        "retained_cells": int(counts[supported_group][outcome].sum()),
        "sl_labels_used": False,
        "files": [
            {"path": artifact.name, "bytes": artifact.stat().st_size, "sha256": sha256(artifact)}
        ],
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def parse_key(key: str) -> tuple[tuple[str, int], ...]:
    members = []
    for part in key.split("+"):
        name, dose = part.rsplit("@", 1)
        members.append((name, int(dose)))
    return tuple(members)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/gse208240")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/data/gse208240-calu3-crispri-pseudobulk-v1",
    )
    args = parser.parse_args()
    print(json.dumps(build(args.raw, args.output), indent=2))


if __name__ == "__main__":
    main()
