#!/usr/bin/env python3
"""Assemble fitting-only human population views for the shared world model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/derived/slp11-joint-world-populations-v1"
FEATURE_DIM = 577
TAXON = 9606
SCHEMA = "slp.joint-world-population-corpus/v1"

SOURCES = {
    "k562": {
        "moments": ROOT / "data/derived/slp11-human-k562-essential-fitting-action-moments-v1/fitting-action-moments.npz",
        "control": ROOT / "data/derived/slp11-human-k562-essential-count-control/reconstruction-train-nt-gem-v1/gem-control-reference.npz",
        "static": ROOT / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1/k562-essential-count-static577.npz",
        "roster": ROOT / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1/roster-index.npz",
        "context": "replogle-2022-k562-essential-crispri-day6",
    },
    "rpe1": {
        "moments": ROOT / "data/derived/slp11-human-rpe1-essential-raw-cells-v1/fitting-action-moments.npz",
        "control": ROOT / "data/derived/slp11-human-rpe1-essential-count-control/reconstruction-train-nt-gem-v1/gem-control-reference.npz",
        "static": ROOT / "data/derived/slp11-human-rpe1-essential-count-static/ensembl116-esm8m-shared-go-v1/rpe1-essential-count-static577.npz",
        "roster": ROOT / "data/derived/slp11-human-rpe1-essential-count-static/ensembl116-esm8m-shared-go-v1/roster-index.npz",
        "context": "replogle-2022-rpe1-essential-crispri-day7",
    },
}
NORMAN = {
    "development": ROOT / "data/derived/slp11-norman-author-normalized-v2/norman-2019-author-normalized-development-v2.npz",
    "static": ROOT / "data/derived/slp11-norman-static/ensembl116-goa2022-fixed-basis-v1/norman-extended-static-esm-go-features.npz",
    "loader": ROOT / "modules/slp-1-1-compositional-state-v1/data.py",
}
PINS = {
    SOURCES["k562"]["moments"]: "a1f44a15a42c5b56e4ce897fde6ebba97298fc296105c6c870ee0e740331694e",
    SOURCES["k562"]["control"]: "c72d28e9eb6633fa237b11e0c16258d875eadaacf31e5b8b3def862150b36d13",
    SOURCES["k562"]["static"]: "6706f8867adedef8822897bc275ea90680584f84afd24771e4beb3c8ecf07659",
    SOURCES["k562"]["roster"]: "f2ee702a0714ca7f11f4fd2aa96f4c1825617c0e4f2bcdac42135cd0ba938d7b",
    SOURCES["rpe1"]["moments"]: "d15def86aead06b0bc75ab63c77513735ec7c57d65012bff72f3947bc654895c",
    SOURCES["rpe1"]["control"]: "c0c2eab217d00f9555b6ab5725cd2c49f56b1ecdf34b7af47f303eee9d1b8e20",
    SOURCES["rpe1"]["static"]: "621e1e9f0dffc740ef42382b1b2898f629edd5037e8a02d411e8d30e815ed816",
    SOURCES["rpe1"]["roster"]: "b9e1b169c2be4ac756e94f465009dc5bef80d06bc0652950c3cf6916d26d1e56",
    NORMAN["development"]: "ab81e7ed07d7f111b3dfc964cece28a2db7de0dcf5975f6ff1a3bc2db0be683e",
    NORMAN["static"]: "7b3d78af66f013e2d1df3a3f98924707ed111bc795757753e82a5e8f495408b5",
}


class JointWorldDataError(ValueError):
    """Raised when a source violates the fitting-only corpus contract."""


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def scalar(value: str) -> np.ndarray:
    return np.asarray(value)


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable output differs: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _feature_lookup(static: dict[str, np.ndarray]) -> tuple[dict[str, int], np.ndarray]:
    ids = static["entity_id"].astype(str)
    features = np.asarray(static["feature_values"], dtype=np.float32)
    taxa = np.asarray(static["entity_taxon"], dtype=np.int64)
    if (
        len(ids) != len(set(ids.tolist()))
        or features.shape != (len(ids), FEATURE_DIM)
        or not np.array_equal(taxa, np.full(len(ids), TAXON, dtype=np.int64))
        or not np.isfinite(features).all()
    ):
        raise JointWorldDataError("static577 feature contract mismatch")
    return {gene: row for row, gene in enumerate(ids)}, features


def load_crispri_source(name: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    spec = SOURCES[name]
    moments, control, static, roster = (
        load_npz(spec[key]) for key in ("moments", "control", "static", "roster")
    )
    actions = moments["action_ids"].astype(str)
    queries = moments["query_ids"].astype(str)
    if not np.array_equal(actions, roster["fitting_action_ids"].astype(str)):
        raise JointWorldDataError(f"{name} fitting action roster mismatch")
    if not all(np.array_equal(queries, value["query_ids"].astype(str)) for value in (control, roster)):
        raise JointWorldDataError(f"{name} native query roster mismatch")
    if not np.array_equal(moments["gem_group"], control["gem_group"]):
        raise JointWorldDataError(f"{name} GEM roster mismatch")
    if str(moments["source_sha256"].item()) != str(control["source_sha256"].item()) or str(
        moments["routing_sha256"].item()
    ) != str(control["routing_sha256"].item()):
        raise JointWorldDataError(f"{name} moment/control lineage mismatch")
    cell_count = np.asarray(moments["cell_count"], dtype=np.int64)
    gem_cell_count = np.asarray(moments["gem_cell_count"], dtype=np.int64)
    cp10k_sum = np.asarray(moments["cp10k_sum"], dtype=np.float64)
    basal_rate = np.asarray(control["basal_rate"], dtype=np.float32)
    basal_mask = np.asarray(control["basal_mask"], dtype=np.bool_)
    if (
        cp10k_sum.shape != (len(actions), len(queries))
        or gem_cell_count.shape != (len(actions), len(basal_rate))
        or not np.array_equal(gem_cell_count.sum(1), cell_count)
        or np.any(cell_count <= 0)
        or np.any(cp10k_sum < 0)
        or not np.isfinite(cp10k_sum).all()
        or basal_rate.shape != basal_mask.shape
        or basal_rate.shape[1] != len(queries)
    ):
        raise JointWorldDataError(f"{name} population moment shape mismatch")
    lookup, features = _feature_lookup(static)
    if any(gene not in lookup for gene in (*actions, *queries)):
        raise JointWorldDataError(f"{name} static coverage incomplete")
    action_features = features[[lookup[gene] for gene in actions]]
    query_features = features[[lookup[gene] for gene in queries]]
    mean = (cp10k_sum / cell_count[:, None]).astype(np.float32)
    weights = gem_cell_count / cell_count[:, None]
    matched_basal = np.log1p(weights @ basal_rate).astype(np.float32)
    padded_action_features = np.zeros((len(actions), 2, FEATURE_DIM), dtype=np.float32)
    padded_action_features[:, 0] = action_features
    action_mask = np.zeros((len(actions), 2), dtype=np.bool_)
    action_mask[:, 0] = True
    action_offsets = np.arange(len(actions) + 1, dtype=np.int64)
    arrays = {
        "schema": scalar("slp.joint-world-crispri-populations/v1"),
        "source_id": scalar(name),
        "context_id": scalar(str(spec["context"])),
        "intervention_mode": scalar("CRISPRi"),
        "assay": scalar("single-cell-RNA-count-population-mean"),
        "ncbi_taxon": np.asarray(TAXON, dtype=np.int64),
        "feature_dim": np.asarray(FEATURE_DIM, dtype=np.int64),
        "action_roster_ids": actions,
        "action_roster_features": action_features,
        "action_ids": actions,
        "action_offsets": action_offsets,
        "query_ids": queries,
        "action_features": padded_action_features,
        "action_mask": action_mask,
        "query_features": query_features,
        "target_cp10k_mean": mean,
        "target_ln1p_mean": np.log1p(mean).astype(np.float32),
        "targets": np.log1p(mean).astype(np.float32),
        "target_observed": np.ones(mean.shape, dtype=np.bool_),
        "observed": np.ones(mean.shape, dtype=np.bool_),
        "target_cell_count": cell_count,
        "gem_group": moments["gem_group"],
        "gem_cell_count": gem_cell_count,
        "control_basal_cp10k_rate": basal_rate,
        "control_basal_observed": basal_mask,
        "control_cell_count": np.asarray(control["control_num_cells"], dtype=np.int64),
        "action_matched_basal_ln1p_mean": matched_basal,
        "basal": matched_basal,
        "target_units": scalar("ln1p(population-mean-per-cell-CP10k)"),
        "uncertainty_available": np.asarray(False),
        "cell_count_role": scalar("measurement-exposure-only-not-model-mean-input"),
        "rate_definition": moments["rate_definition"],
    }
    return arrays, {gene: action_features[row] for row, gene in enumerate(actions)}


def _load_composition_module():
    spec = importlib.util.spec_from_file_location("slp11_joint_composition_data", NORMAN["loader"])
    if spec is None or spec.loader is None:
        raise ImportError(NORMAN["loader"])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_norman_source() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    module = _load_composition_module()
    data = module.load_compositional_data(NORMAN["development"], NORMAN["static"])
    # Read only exposure/routing and core-control arrays here. The composition
    # loader separately retains quantitative targets for split_train rows only.
    with np.load(NORMAN["development"], allow_pickle=False) as archive:
        raw = {
            name: archive[name].copy()
            for name in (
                "num_cells_filtered",
                "control_targets",
                "control_observed",
                "control_num_cells_filtered",
            )
        }
    static = load_npz(NORMAN["static"])
    lookup, features = _feature_lookup(static)
    if any(gene not in lookup for gene in data.query_ids):
        raise JointWorldDataError("Norman native queries lack static features")
    offsets = np.zeros(len(data.canonical_actions) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(actions) for actions in data.canonical_actions])
    flattened = np.asarray([gene for actions in data.canonical_actions for gene in actions])
    padded_action_features = np.zeros(
        (len(data.canonical_actions), 2, FEATURE_DIM), dtype=np.float32
    )
    valid = data.action_feature_index >= 0
    padded_action_features[valid] = data.gene_features[data.action_feature_index[valid]]
    basal = np.zeros_like(data.y, dtype=np.float32)
    arrays = {
        "schema": scalar("slp.joint-world-crispra-compositions/v1"),
        "source_id": scalar("norman"),
        "context_id": scalar("norman-2019-k562-crispra-day5"),
        "intervention_mode": scalar("CRISPRa"),
        "assay": scalar("single-cell-RNA-control-z-population-mean"),
        "ncbi_taxon": np.asarray(TAXON, dtype=np.int64),
        "feature_dim": np.asarray(FEATURE_DIM, dtype=np.int64),
        "action_roster_ids": data.action_ids,
        "action_roster_features": data.gene_features,
        "action_ids": flattened,
        "action_offsets": offsets,
        "action_feature_index": data.action_feature_index,
        "action_features": padded_action_features,
        "action_mask": data.action_mask,
        "query_ids": data.query_ids,
        "query_features": features[[lookup[gene] for gene in data.query_ids]],
        "target_control_z_mean": data.y,
        "targets": data.y,
        "target_observed": data.observed,
        "observed": data.observed,
        "target_cell_count": np.asarray(
            [sum(raw["num_cells_filtered"][list(rows)]) for rows in data.source_record_indices],
            dtype=np.int64,
        ),
        "control_baseline_control_z": np.zeros(len(data.query_ids), dtype=np.float32),
        "basal": basal,
        "control_target_control_z": np.asarray(raw["control_targets"], dtype=np.float32),
        "control_observed": np.asarray(raw["control_observed"], dtype=np.bool_),
        "control_cell_count": np.asarray(raw["control_num_cells_filtered"], dtype=np.int64),
        "single_rows": data.single_rows,
        "combination_rows": data.combination_rows,
        "combination_single_rows": data.combination_single_rows,
        "combination_common_query_mask": data.combination_common_query_mask,
        "combination_fold": data.combination_fold,
        "uncertainty_available": np.asarray(False),
        "cell_count_role": scalar("measurement-exposure-only-not-model-mean-input"),
        "target_value_space": scalar(data.target_value_space),
        "target_units": scalar(data.target_value_space),
        "canonical_aggregation": scalar(data.aggregation),
    }
    return arrays, {gene: data.gene_features[row] for row, gene in enumerate(data.action_ids)}


def global_feature_normalization(
    sources: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Feature packs are immutable source inputs. Some Norman rows predate the
    # count-pack basis and differ materially for the same ENSG, so source-qualify
    # rows rather than silently selecting one representation.
    qualified = sorted(
        (f"{source}|{gene}", np.asarray(value, dtype=np.float32))
        for source, values in sources.items()
        for gene, value in values.items()
    )
    roster = np.asarray([name for name, _ in qualified])
    matrix = np.stack([value for _, value in qualified]).astype(np.float64)
    mean = matrix.mean(0)
    scale = matrix.std(0)
    scale[scale <= 1e-5] = 1.0
    return roster, mean.astype(np.float32), scale.astype(np.float32)


def prepare(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    for path, expected in PINS.items():
        if sha256(path) != expected:
            raise JointWorldDataError(f"pinned input changed: {path}")
    k562, k562_features = load_crispri_source("k562")
    rpe1, rpe1_features = load_crispri_source("rpe1")
    norman, norman_features = load_norman_source()
    normalization_ids, feature_mean, feature_scale = global_feature_normalization(
        {"k562": k562_features, "rpe1": rpe1_features, "norman": norman_features}
    )
    for arrays in (k562, rpe1, norman):
        arrays["feature_normalization_action_ids"] = normalization_ids
        arrays["feature_mean"] = feature_mean
        arrays["feature_scale"] = feature_scale
        arrays["feature_normalization_scope"] = scalar(
            "source-qualified fitting intervention rows pooled across all three shards"
        )
    payloads = {name: deterministic_npz(arrays) for name, arrays in (("k562", k562), ("rpe1", rpe1), ("norman", norman))}
    outputs = {}
    for name, payload in payloads.items():
        path = output / f"{name}.npz"
        write_new(path, payload)
        outputs[name] = {
            "path": path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest = {
        "schema": SCHEMA,
        "status": "fitting-only-development-corpus-not-omf-admitted",
        "sources": {
            "k562": {"mode": "CRISPRi", "populations": 1443, "queries": 8563, "gemContexts": 48},
            "rpe1": {"mode": "CRISPRi", "populations": 1666, "queries": 8749, "gemContexts": 56},
            "norman": {"mode": "CRISPRa", "canonicalSingles": 71, "canonicalCombinations": 59, "queries": 7226, "controlPseudobulks": 20},
        },
        "normalization": {
            "features": "mean/population-SD over source-qualified fitting intervention rows pooled across sources; scale one when SD<=1e-5; immutable source feature revisions are preserved",
            "crispriTargets": "CP10k population mean and ln1p of that mean, native query axes",
            "normanTargets": "source-supplied core-control population z-score means, equal-construct canonical aggregation",
        },
        "uncertainty": "No within-action second moments exist in the fitting inputs. Shards expose cell counts and GEM composition as measurement exposure; uncertainty_available is false and no variance is invented.",
        "accessBoundary": {
            "crispriFittingMoments": True,
            "normanOriginalFittingRows": True,
            "normanOriginalValidationOutcomes": False,
            "protectedTestOutcomes": False,
            "benchmarkData": False,
            "nativeQueryAxesMerged": False,
        },
        "inputs": {
            str(path.relative_to(ROOT)).replace("\\", "/"): {"sha256": digest}
            for path, digest in PINS.items()
        },
        "sourceCode": {
            "assembler": {"path": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(Path(__file__).resolve())},
            "normanLoader": {"path": str(NORMAN["loader"].relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(NORMAN["loader"])},
        },
        "outputs": outputs,
    }
    write_new(output / "manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return result


def main() -> int:
    args = parser().parse_args()
    report = prepare(args.output)
    print(json.dumps({"output": str(args.output), "outputs": report["outputs"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
