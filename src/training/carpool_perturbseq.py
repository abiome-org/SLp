"""Build a mode-aware THP-1 CaRPool-seq pack from GEO GSE213957."""

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
PREFIX = "THP1-CaRPool-seq_and_HEK293FTstabRNA.GEXGDO"
ARCHIVE = "GSE213957_THP1-CaRPool-seq_and_GSE213957_HEX293FTstabRNA.tar.gz"
SUFFIXES = ("barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_extracted(raw: Path) -> Path:
    extracted = raw / "extracted"
    expected = [f"{PREFIX}{lane}.{suffix}" for lane in range(1, 5) for suffix in SUFFIXES]
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


def parse_actions(pair: str) -> tuple[tuple[str, int], ...]:
    counts = Counter(name.upper() for name in pair.split("_") if name != "NT")
    return tuple(sorted(counts.items()))


def lane_rows(
    extracted: Path,
    lane: int,
    metadata: pd.DataFrame,
    action_id: dict[str, int],
):
    prefix = extracted / f"{PREFIX}{lane}"
    barcodes = pd.read_csv(
        f"{prefix}.barcodes.tsv.gz", sep="\t", header=None
    ).iloc[:, 0].to_numpy(dtype=str)
    features = pd.read_csv(
        f"{prefix}.features.tsv.gz", sep="\t", header=None
    )
    gene = features.iloc[:, 2].astype(str).eq("Gene Expression").to_numpy()
    feature_names = features.loc[gene, 1].to_numpy(dtype=str)
    matrix = mmread(f"{prefix}.matrix.mtx.gz").tocsr()
    if matrix.shape != (len(features), len(barcodes)):
        raise ValueError(f"lane {lane} matrix axes do not match")
    cell_id = np.asarray([f"L{lane}_{barcode}" for barcode in barcodes])
    keep = np.isin(cell_id, metadata.index)
    if not np.any(keep):
        raise ValueError(f"lane {lane} has no cells in source-curated metadata")
    selected_id = cell_id[keep]
    selected_metadata = metadata.loc[selected_id]
    expression = matrix[gene][:, keep].T.tocsr().astype("float32")
    library_size = np.asarray(expression.sum(axis=1)).ravel().clip(1)
    expression = expression.multiply((1e4 / library_size)[:, None]).tocsr()
    np.log1p(expression.data, out=expression.data)

    keys = np.asarray(
        [
            f"{sample}\0{pair}"
            for sample, pair in zip(selected_metadata["HTO"], selected_metadata["GenePair"])
        ]
    )
    unique, group = np.unique(keys, return_inverse=True)
    counts = np.bincount(group, minlength=len(unique))
    supported = counts >= 4
    supported_group = np.flatnonzero(supported)
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
    supported_keys = [unique[index].split("\0", 1) for index in supported_group]
    controls = {
        sample: means[index]
        for index, (sample, pair) in enumerate(supported_keys)
        if pair == "NT_NT"
    }
    if set(selected_metadata["HTO"]) - set(controls):
        raise ValueError(f"lane {lane} lacks a supported control for an HTO sample")
    outcome = np.asarray([pair != "NT_NT" for _, pair in supported_keys])
    outcome_keys = [key for key, include in zip(supported_keys, outcome) if include]
    parsed = [parse_actions(pair) for _, pair in outcome_keys]
    if any(not actions or len(actions) > 2 for actions in parsed):
        raise ValueError("unexpected CaRPool action cardinality")
    actions = np.full((len(parsed), 2), -1, dtype="int32")
    doses = np.zeros((len(parsed), 2), dtype="int8")
    for row, members in enumerate(parsed):
        actions[row, : len(members)] = [action_id[target] for target, _ in members]
        doses[row, : len(members)] = [dose for _, dose in members]
    modes = np.full(actions.shape, "", dtype="<U16")
    modes[actions >= 0] = "rna_knockdown"
    samples = np.asarray([sample for sample, _ in outcome_keys])
    targets = means[outcome] - np.stack([controls[sample] for sample in samples])
    return {
        "actions": actions,
        "action_modes": modes,
        "action_doses": doses,
        "target": targets.astype("float32"),
        "target_feature_name": feature_names,
        "sample_id": samples,
        "cell_count": counts[supported_group][outcome].astype("int32"),
        "replicate_id": np.repeat(f"10x_lane_{lane}", len(actions)),
        "audit": {
            "lane": lane,
            "matrix_sha256": sha256(Path(f"{prefix}.matrix.mtx.gz")),
            "source_metadata_cells": len(selected_id),
            "retained_pseudobulks": len(actions),
            "retained_cells": int(counts[supported_group][outcome].sum()),
        },
    }


def build(raw: Path, output: Path) -> dict[str, object]:
    raw = Path(raw)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = raw / "GSE213957_THP1-CaRPool-seq.metadata.tsv.gz"
    metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)
    action_names = np.asarray(
        sorted(
            set().union(
                *(set(name for name, _ in parse_actions(pair)) for pair in metadata["GenePair"])
            )
        )
    )
    action_id = {name: index for index, name in enumerate(action_names)}
    extracted = ensure_extracted(raw)
    rows = [lane_rows(extracted, lane, metadata, action_id) for lane in range(1, 5)]
    feature_names = rows[0]["target_feature_name"]
    if any(not np.array_equal(feature_names, row["target_feature_name"]) for row in rows[1:]):
        raise ValueError("CaRPool expression axes differ across lanes")
    actions = np.concatenate([row["actions"] for row in rows])
    artifact = output / "gse213957_thp1_carpool_pseudobulk_v1.npz"
    np.savez_compressed(
        artifact,
        actions=actions,
        action_modes=np.concatenate([row["action_modes"] for row in rows]),
        action_doses=np.concatenate([row["action_doses"] for row in rows]),
        action_names=action_names,
        target=np.concatenate([row["target"] for row in rows]),
        target_semantics=np.asarray("perturbation_delta"),
        target_feature_name=feature_names,
        cardinality=(actions >= 0).sum(axis=1).astype("int8"),
        source_id=np.asarray("GSE213957"),
        context_id=np.asarray("THP1"),
        experimental_condition_id=np.asarray("Cas13d|duration=unreported"),
        sample_id=np.concatenate([row["sample_id"] for row in rows]),
        replicate_id=np.concatenate([row["replicate_id"] for row in rows]),
        cell_count=np.concatenate([row["cell_count"] for row in rows]),
        observation_unit=np.asarray("pseudobulk"),
    )
    source_manifest = raw / "source_manifest.json"
    audit = {
        "schema": "slp-data-release-audit-v1",
        "release_id": "data/perturbseq/gse213957-thp1-carpool-pseudobulk-v1",
        "source": {
            "name": "Efficient combinatorial targeting of RNA transcripts in single cells with Cas13 RNA Perturb-seq",
            "accession": "GSE213957",
            "doi": "10.1038/s41587-022-01500-7",
            "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213957",
            "source_manifest_sha256": sha256(source_manifest) if source_manifest.exists() else None,
        },
        "license": {
            "id": "NCBI-GEO-PUBLIC-DATA",
            "evidence": "NCBI states that it places no restrictions on use or distribution of GEO data, while noting that submitters may assert rights.",
            "policy": "https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html",
        },
        "transformations": "Source-curated high-quality THP-1 cells were grouped by 10x lane, HTO sample, and gene pair. Groups supported by at least four cells were normalized per cell as log1p counts per 10,000, averaged, and differenced from the lane-and-HTO-matched NT_NT mean. Target order was made permutation invariant and repeated targets are represented as action dose.",
        "schema_description": "NPZ with one or two RNA-knockdown actions, per-action doses, full-gene expression perturbation deltas, THP-1 provenance, HTO sample, 10x replicate, and pseudobulk cell counts.",
        "population": "THP-1 acute monocytic leukemia cells in the source CaRPool-seq experiment",
        "endpoints": ["mean per-cell log1p(CP10K) expression change from lane-and-HTO-matched NT_NT control"],
        "split_construction": "No train/test split is embedded. The hard generalization gate constructs deterministic folds downstream.",
        "exclusions": "Cells absent from source-curated metadata, control-only outcomes, and lane/sample/action groups with fewer than four cells were excluded. HTO labels are retained as samples rather than assigned an undocumented biological meaning. Perturbation duration was not recovered from the GEO deposit and is explicit as unreported.",
        "rows": len(actions),
        "single_rows": int(((actions >= 0).sum(axis=1) == 1).sum()),
        "pair_rows": int(((actions >= 0).sum(axis=1) == 2).sum()),
        "unique_action_targets": len(action_names),
        "expression_features": len(feature_names),
        "sources": [row["audit"] for row in rows],
        "sl_labels_used": False,
        "files": [
            {"path": artifact.name, "bytes": artifact.stat().st_size, "sha256": sha256(artifact)}
        ],
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/gse213957")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/data/gse213957-thp1-carpool-pseudobulk-v1",
    )
    args = parser.parse_args()
    print(json.dumps(build(args.raw, args.output), indent=2))


if __name__ == "__main__":
    main()
