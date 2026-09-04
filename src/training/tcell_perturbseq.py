"""Build a primary-T-cell CRISPRi Perturb-seq pack from GEO GSE278572."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
MAX_CARDINALITY = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_actions(call: str) -> tuple[tuple[str, int], ...]:
    counts = Counter(
        target
        for token in str(call).split("|")
        if (target := token.split("_", 1)[0].upper()) != "NON-TARGETING"
    )
    return tuple(sorted(counts.items()))


def stream_selected_matrix(path: Path, column_lookup: np.ndarray, retained_count: int):
    """Read a genes x cells MatrixMarket file keeping only selected columns.

    The file is read in binary blocks that are split on explicit newline
    boundaries; buffered CSV chunking over a text stream can split lines and
    silently corrupt coordinates, which was observed on this deposit.
    """
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    malformed = 0
    with gzip.open(path, "rb") as handle:
        header = handle.readline()
        if not header.startswith(b"%%MatrixMarket"):
            raise ValueError(f"{path.name} is not a MatrixMarket file")
        while True:
            line = handle.readline()
            if not line.startswith(b"%"):
                break
        dimensions = line.split()
        rows, columns, entries = int(dimensions[0]), int(dimensions[1]), int(dimensions[2])
        if columns != len(column_lookup):
            raise ValueError("matrix column count does not match barcodes")
        read = 0
        buffer = b""
        while read < entries:
            block = handle.read(1 << 26)
            if not block:
                break
            data = buffer + block
            cut = data.rfind(b"\n")
            if cut == -1:
                buffer = data
                continue
            buffer = data[cut + 1 :]
            payload = data[: cut + 1]
            read += payload.count(b"\n")
            frame = pd.read_csv(
                io.BytesIO(payload),
                sep=" ",
                header=None,
                names=["row", "column", "value"],
                dtype={"row": np.int64, "column": np.int64, "value": np.float64},
            )
            good = (
                frame["row"].between(1, rows)
                & frame["column"].between(1, columns)
                & frame["value"].notna()
            ).to_numpy()
            malformed += int((~good).sum())
            frame = frame[good]
            selected = column_lookup[frame["column"].to_numpy(dtype="int64") - 1]
            keep = selected >= 0
            if keep.any():
                row_parts.append(frame["row"].to_numpy(dtype="int64")[keep] - 1)
                column_parts.append(selected[keep])
                value_parts.append(frame["value"].to_numpy(dtype="float32")[keep])
        if buffer.strip():
            raise ValueError("matrix ended with an unparsed partial line")
    if malformed:
        print(f"note: skipped {malformed} malformed matrix lines", flush=True)
    matrix = sparse.csr_matrix(
        (
            np.concatenate(value_parts),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ),
        shape=(rows, retained_count),
    )
    return matrix


def build(raw: Path, output: Path) -> dict[str, object]:
    raw = Path(raw)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    calls = pd.read_csv(raw / "GSE278572_protospacer_calls_per_cell.csv.gz")
    barcodes = pd.read_csv(
        raw / "GSE278572_barcodes.tsv.gz", sep="\t", header=None
    ).iloc[:, 0].to_numpy(dtype=str)
    features = pd.read_csv(raw / "GSE278572_features.tsv.gz", sep="\t", header=None)
    gene = features.iloc[:, 2].astype(str).eq("Gene Expression").to_numpy()
    feature_names = features.loc[gene, 1].to_numpy(dtype=str)

    calls["actions"] = [parse_actions(call) for call in calls["feature_call"].astype(str)]
    cardinality = np.asarray([len(row) for row in calls["actions"]])
    calls = calls[cardinality <= MAX_CARDINALITY]
    calls = calls.iloc[np.argsort(calls["cell_barcode"].astype(str).to_numpy(), kind="stable")]
    positions = pd.Index(barcodes).get_indexer(calls["cell_barcode"].astype(str))
    if np.any(positions < 0) or len(np.unique(positions)) != len(positions):
        raise ValueError("perturbation calls do not map one-to-one to retained barcodes")

    action_names = np.asarray(
        sorted(set().union(*(set(name for name, _ in row) for row in calls["actions"])))
    )
    action_id = {name: index for index, name in enumerate(action_names)}
    keys = np.asarray(["+".join(f"{name}@{dose}" for name, dose in row) for row in calls["actions"]])
    unique, group = np.unique(keys, return_inverse=True)
    counts = np.bincount(group, minlength=len(unique))
    supported = counts >= 4
    supported_group = np.flatnonzero(supported)

    column_lookup = np.full(len(barcodes), -1, dtype=np.int64)
    column_lookup[positions] = np.arange(len(positions))
    expression = stream_selected_matrix(
        raw / "GSE278572_matrix.mtx.gz", column_lookup, len(positions)
    )
    if expression.shape != (len(features), len(positions)):
        raise ValueError("expression axes do not match features and retained cells")
    expression = expression[gene].T.tocsr().astype("float32")
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
    control = np.asarray([key == "" for key in unique[supported_group]])
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

    artifact = output / "gse278572_primary_tcell_perturbseq_v1.npz"
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
        source_id=np.asarray("GSE278572"),
        context_id=np.asarray("PRIMARY_CD4_T_CELL"),
        experimental_condition_id=np.asarray("CRISPRi|duration=unreported|donor=unreported|activation=unreported"),
        sample_id=np.asarray("unreported"),
        replicate_id=np.asarray("unreported"),
        cell_count=counts[supported_group][outcome].astype("int32"),
        observation_unit=np.asarray("pseudobulk"),
    )
    audit = {
        "schema": "slp-data-release-audit-v1",
        "release_id": "data/perturbseq/gse278572-primary-tcell-perturbseq-v1",
        "source": {
            "name": "Primary CD4 T-cell Perturb-CITE-seq CRISPRi screen",
            "accession": "GSE278572",
            "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE278572",
            "source_manifest_sha256": sha256(raw / "source_manifest.json"),
        },
        "license": {
            "id": "NCBI-GEO-PUBLIC-DATA",
            "evidence": "NCBI states that it places no restrictions on use or distribution of GEO data, while noting that submitters may assert rights.",
            "policy": "https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html",
        },
        "transformations": "Per-cell protospacer calls were collapsed to gene-level action multisets (non-targeting tokens dropped, guide counts within a target retained as dose), and action sets supported by at least four cells were normalized per cell as log1p counts per 10,000, averaged, and differenced from the pooled all-non-targeting control mean. The 1.03-billion-entry matrix was streamed so only perturbation-called cells were materialized.",
        "schema_description": "NPZ with one or two CRISPRi actions and guide doses, full-gene expression perturbation deltas, primary-T-cell context, and pseudobulk cell counts.",
        "population": "Primary human CD4+ T cells; the deposit provides no per-cell donor or activation-state annotation, so those axes are explicit as unreported rather than inferred.",
        "endpoints": ["mean per-cell log1p(CP10K) expression change from pooled all-non-targeting controls"],
        "split_construction": "No train/test split is embedded. The hard generalization gate constructs deterministic folds downstream.",
        "exclusions": "Cells whose calls exceed two distinct target genes were excluded as multiplets, groups with fewer than four cells were excluded, and all-non-targeting cells define controls. Donor and activation contexts are unreported in the supplement and were not reconstructed from filenames.",
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
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/gse278572")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/data/gse278572-primary-tcell-perturbseq-v1",
    )
    args = parser.parse_args()
    print(json.dumps(build(args.raw, args.output), indent=2))


if __name__ == "__main__":
    main()
