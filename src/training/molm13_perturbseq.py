"""Build a MOLM13 single/combinatorial Perturb-seq pack from GSE200201."""

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
ARCHIVE = "GSE200201_RAW.tar"
EXPERIMENTS = tuple(
    (6021932 + index, 6021952 + index, f"single_{index}", "single")
    for index in range(1, 16)
) + tuple(
    (6021947 + index, 6021967 + index, f"combo_{index}", "combo")
    for index in range(1, 4)
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def names(expression_gsm: int, call_gsm: int, experiment: str) -> dict[str, str]:
    base = f"GSM{expression_gsm}_MOLM13_RNAseq_{experiment}"
    return {
        "cells": f"{base}.cells.tsv.gz",
        "genes": f"{base}.genes.tsv.gz",
        "matrix": f"{base}.matrix.mtx.gz",
        "calls": f"GSM{call_gsm}_MOLM13_RNAseq_{experiment}.protospacer_calls.csv.gz",
    }


def ensure_extracted(raw: Path) -> Path:
    extracted = raw / "extracted"
    expected = [
        name
        for expression_gsm, call_gsm, experiment, _ in EXPERIMENTS
        for name in names(expression_gsm, call_gsm, experiment).values()
    ]
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


def guide_target(token: str) -> str | None:
    if token.upper().startswith("NTC"):
        return None
    target, separator, guide = token.rpartition("_")
    if not separator or not guide:
        raise ValueError(f"unrecognized source guide name {token!r}")
    return target.upper()


def parse_actions(call: str) -> tuple[tuple[str, int], ...]:
    counts = Counter(
        target
        for token in call.split("|")
        if (target := guide_target(token)) is not None
    )
    return tuple(sorted(counts.items()))


def load_calls(path: Path) -> pd.DataFrame:
    calls = pd.read_csv(path)
    calls["actions"] = [parse_actions(value) for value in calls["feature_call"].astype(str)]
    calls["action_key"] = [
        "+".join(f"{name}@{dose}" for name, dose in row) for row in calls["actions"]
    ]
    return calls


def experiment_rows(
    extracted: Path,
    expression_gsm: int,
    call_gsm: int,
    experiment: str,
    library: str,
    action_id: dict[str, int],
):
    paths = {
        key: extracted / value
        for key, value in names(expression_gsm, call_gsm, experiment).items()
    }
    barcodes = pd.read_csv(paths["cells"], sep="\t", header=None).iloc[:, 0].to_numpy(dtype=str)
    features = pd.read_csv(paths["genes"], sep="\t", header=None)
    gene = features.iloc[:, 2].astype(str).eq("Gene Expression").to_numpy()
    feature_names = features.loc[gene, 1].to_numpy(dtype=str)
    calls = load_calls(paths["calls"])
    positions = pd.Index(barcodes).get_indexer(calls["cell_barcode"].astype(str))
    if np.any(positions < 0) or len(np.unique(positions)) != len(positions):
        raise ValueError(f"{experiment} perturbation calls do not map one-to-one to cells")
    matrix = mmread(paths["matrix"]).tocsr()
    if matrix.shape != (len(features), len(barcodes)):
        raise ValueError(f"{experiment} matrix axes do not match")
    expression = matrix[gene][:, positions].T.tocsr().astype("float32")
    library_size = np.asarray(expression.sum(axis=1)).ravel().clip(1)
    expression = expression.multiply((1e4 / library_size)[:, None]).tocsr()
    np.log1p(expression.data, out=expression.data)

    cardinality = np.asarray([len(row) for row in calls["actions"]])
    valid = cardinality <= 8
    control = valid & (cardinality == 0)
    if control.sum() < 4:
        raise ValueError(f"{experiment} lacks at least four non-targeting control cells")
    control_mean = np.asarray(expression[control].mean(axis=0)).ravel()
    outcome = valid & (cardinality >= 1)
    keys, group = np.unique(calls.loc[outcome, "action_key"].astype(str), return_inverse=True)
    counts = np.bincount(group, minlength=len(keys))
    supported = counts >= 4
    supported_group = np.flatnonzero(supported)
    assignment = sparse.csr_matrix(
        (
            np.ones(len(group), dtype="float32"),
            (group, np.arange(len(group))),
        ),
        shape=(len(keys), len(group)),
    )
    means = (assignment[supported_group] @ expression[outcome]).multiply(
        (1 / counts[supported_group])[:, None]
    ).toarray()
    call_by_key = {
        key: actions
        for key, actions in zip(calls.loc[outcome, "action_key"], calls.loc[outcome, "actions"])
    }
    parsed = [call_by_key[key] for key in keys[supported_group]]
    actions = np.full((len(parsed), 8), -1, dtype="int32")
    doses = np.zeros((len(parsed), 8), dtype="int8")
    for row, members in enumerate(parsed):
        actions[row, : len(members)] = [action_id[target] for target, _ in members]
        doses[row, : len(members)] = [dose for _, dose in members]
    modes = np.full(actions.shape, "", dtype="<U10")
    modes[actions >= 0] = "knockout"
    return {
        "actions": actions,
        "action_modes": modes,
        "action_doses": doses,
        "target": (means - control_mean).astype("float32"),
        "target_feature_name": feature_names,
        "replicate_id": np.repeat(experiment, len(actions)),
        "condition": np.repeat(f"Cas9_knockout|duration=unreported|library={library}", len(actions)),
        "cell_count": counts[supported_group].astype("int32"),
        "audit": {
            "experiment": experiment,
            "library": library,
            "assigned_cells": len(calls),
            "control_cells": int(control.sum()),
            "retained_pseudobulks": len(actions),
            "retained_cells": int(counts[supported_group].sum()),
            "matrix_sha256": sha256(paths["matrix"]),
            "calls_sha256": sha256(paths["calls"]),
        },
    }


def build(raw: Path, output: Path) -> dict[str, object]:
    raw = Path(raw)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    extracted = ensure_extracted(raw)
    all_calls = [
        load_calls(extracted / names(expression_gsm, call_gsm, experiment)["calls"])
        for expression_gsm, call_gsm, experiment, _ in EXPERIMENTS
    ]
    action_names = np.asarray(
        sorted(
            {
                target
                for calls in all_calls
                for actions in calls["actions"]
                for target, _ in actions
            }
        )
    )
    action_id = {name: index for index, name in enumerate(action_names)}
    rows = [
        experiment_rows(extracted, *experiment, action_id)
        for experiment in EXPERIMENTS
    ]
    feature_names = rows[0]["target_feature_name"]
    if any(not np.array_equal(feature_names, row["target_feature_name"]) for row in rows[1:]):
        raise ValueError("MOLM13 expression axes differ across experiments")
    actions = np.concatenate([row["actions"] for row in rows])
    artifact = output / "gse200201_molm13_multi_action_pseudobulk_v1.npz"
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
        source_id=np.asarray("GSE200201"),
        context_id=np.asarray("MOLM13"),
        experimental_condition_id=np.concatenate([row["condition"] for row in rows]),
        replicate_id=np.concatenate([row["replicate_id"] for row in rows]),
        cell_count=np.concatenate([row["cell_count"] for row in rows]),
        observation_unit=np.asarray("pseudobulk"),
    )
    source_manifest = raw / "source_manifest.json"
    audit = {
        "schema": "slp-data-release-audit-v1",
        "release_id": "data/perturbseq/gse200201-molm13-multi-action-pseudobulk-v1",
        "source": {
            "name": "Structural and functional properties of mSWI/SNF chromatin remodeling complexes revealed through single-cell perturbation and genomic profiling",
            "accession": "GSE200201",
            "doi": "10.1016/j.molcel.2023.03.013",
            "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE200201",
            "source_manifest_sha256": sha256(source_manifest) if source_manifest.exists() else None,
        },
        "license": {
            "id": "NCBI-GEO-PUBLIC-DATA",
            "evidence": "NCBI states that it places no restrictions on use or distribution of GEO data, while noting that submitters may assert rights.",
            "policy": "https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html",
        },
        "transformations": "Source protospacer calls were collapsed to gene-level action multisets, retaining target dose and cardinalities through eight. Per experiment, supported action sets with at least four cells were normalized per cell as log1p counts per 10,000, averaged, and differenced from the same-experiment all-NTC mean.",
        "schema_description": "NPZ with up to eight knockout actions and guide doses, full-gene expression perturbation deltas, MOLM13 context, single/combinatorial library condition, experiment replicate, and pseudobulk cell counts.",
        "population": "MOLM13 acute myeloid leukemia cells",
        "endpoints": ["mean per-cell log1p(CP10K) expression change from experiment-matched all-NTC control"],
        "split_construction": "No train/test split is embedded. The hard generalization gate constructs deterministic folds downstream.",
        "exclusions": "All-NTC cells define controls, action sets supported by fewer than four cells and cells above cardinality eight were excluded, and SHARE-seq rows were not mixed into the expression endpoint. Perturbation duration was not recovered from the GEO deposit and is explicit as unreported.",
        "rows": len(actions),
        "rows_by_cardinality": {
            str(card): int(((actions >= 0).sum(axis=1) == card).sum())
            for card in np.unique((actions >= 0).sum(axis=1))
        },
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
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/gse200201")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/data/gse200201-molm13-multi-action-pseudobulk-v1",
    )
    args = parser.parse_args()
    print(json.dumps(build(args.raw, args.output), indent=2))


if __name__ == "__main__":
    main()
