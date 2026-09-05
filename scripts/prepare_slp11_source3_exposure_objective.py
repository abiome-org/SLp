#!/usr/bin/env python3
"""Prepare the fixed source3 exposure-precision objective resource."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/derived/slp11-human-gwps-fixed-panel-context-v1/replogle-k562-rpe1-gwps-complete-panel-development-v2-fixed-control-context.npz"
VARIANCE = ROOT / "results/slp11-transition/human-gwps-fixed-context-minimal-control-physical-state128-response32-seed731-v1/model/exposure-uncertainty.npz"
VARIANCE_SOURCE = ROOT / "modules/slp-1-1-world-transition-v1/exposure_uncertainty.py"
REFERENCE = ROOT / "results/slp11-transition/human-source3-bp-neural-mean-pair-seed731-v2-finalization-v1/bp128-present/reference.npz"
WEIGHTING = ROOT / "results/slp11-transition/human-source3-bp-neural-mean-pair-seed731-v2-finalization-v1/source/objective_weighting.py"
FIXED_MODEL = ROOT / "results/slp11-transition/human-source3-bp-fixed-response-basis-seed731-v2/model.safetensors"
FIXED_REPORT = ROOT / "results/slp11-transition/human-source3-bp-fixed-response-basis-seed731-v2/report.json"
HELPER = ROOT / "modules/slp-1-1-exposure-objective-v1/exposure_objective.py"
DERIVED = ROOT / "data/derived/slp11-human-exposure-objective/source3-mean-oof-control-variance-v1"
RESULT = ROOT / "results/slp11-transition/human-source3-exposure-objective-preparation-v1"

PINS = {
    DATA: "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c",
    VARIANCE: "9cf5f4a5352dccaa7cb3d6c84e2123b16b190220a1ef9e03c933a887be6c81dd",
    VARIANCE_SOURCE: "afafb9f1614264fa80a86ec5861be24cce6f45e0123685cfed13277d865151bd",
    REFERENCE: "aea0c407fcc0e3199b2926eb9f639cc8fe4d8f3abd5eb3dd1055abc065c0dfef",
    WEIGHTING: "2f54e3a3e6ef4e84b4d7ca63d62fd38bd0751a1f7e8aaf4769f9a2c505352c38",
    FIXED_MODEL: "d073e4d66bb498dbbc2048f656b90da069318ff8a736860c03436e37a58cc693",
}
SCALE_FLOOR = 0.05


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    for path, expected in PINS.items():
        if sha256(path) != expected:
            raise ValueError(f"pinned source drift: {path}")
    objective = load(HELPER, "_slp11_exposure_objective_prepare")
    weighting = load(WEIGHTING, "_slp11_exposure_objective_weighting")
    with np.load(DATA, allow_pickle=False) as archive:
        train = archive["split_train"]
        contexts = archive["context_index"][train].astype(np.int64)
        action_ids = archive["action_ids"][train].astype(str)
        num_cells = archive["num_cells_filtered"][train].astype(np.float64)
        observed = archive["observed"][train].astype(bool)
        query_ids = archive["query_ids"].astype(str)
        context_ids = archive["context_ids"].astype(str)
        role = str(archive["num_cells_role"].item())
    if role != "likelihood-only-measurement-precision-not-predictor":
        raise ValueError("cell-count role drift")
    with np.load(VARIANCE, allow_pickle=False) as archive:
        tau = archive["mean_biological_variance"].astype(np.float64)
        sigma = archive["mean_sampling_variance"].astype(np.float64)
        from_controls = archive["mean_sampling_from_controls"].astype(bool)
        residual_counts = archive["mean_residual_counts"].astype(np.int64)
        control_counts = archive["mean_control_counts"].astype(np.int64)
    if tau.shape != (3, 7036) or not from_controls.all() or not np.array_equal(residual_counts[:, 0], [1522, 1759, 7438]) or not np.array_equal(control_counts[:, 0], [97, 113, 514]):
        raise ValueError("exposure component identifiability or shape drift")
    with np.load(REFERENCE, allow_pickle=False) as archive:
        old_scale = archive["objective_query_scale"].astype(np.float64)
        if not np.array_equal(query_ids, archive["query_ids"].astype(str)) or not np.array_equal(context_ids, archive["context_ids"].astype(str)):
            raise ValueError("reference query/context identity drift")
    base_weight = weighting.training_row_weights(contexts, action_ids, objective=weighting.EQUAL_CONTEXT_GENE_V1)
    raw_precision = objective.exposure_precision(num_cells, contexts, tau, sigma, scale_floor=SCALE_FLOOR)
    normalization = objective.match_global_precision(raw_precision, old_scale, contexts, observed, base_weight)
    precision = raw_precision * normalization.multiplier
    DERIVED.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        DERIVED / "precision-components.npz",
        schema=np.asarray("slp.source3-fixed-exposure-precision/v1"),
        query_ids=query_ids,
        context_ids=context_ids,
        biological_variance=tau,
        sampling_variance=sigma,
        sampling_from_controls=from_controls,
        residual_counts=residual_counts,
        control_counts=control_counts,
        scale_floor=np.asarray(SCALE_FLOOR, dtype=np.float64),
        global_precision_multiplier=np.asarray(normalization.multiplier, dtype=np.float64),
        old_weighted_mean_precision=np.asarray(normalization.old_weighted_mean_precision, dtype=np.float64),
        new_weighted_mean_precision_before=np.asarray(normalization.new_weighted_mean_precision_before, dtype=np.float64),
        new_weighted_mean_precision_after=np.asarray(normalization.new_weighted_mean_precision_after, dtype=np.float64),
    )
    resource_hash = sha256(DERIVED / "precision-components.npz")
    audit = {
        "schema": "slp.source3-exposure-objective-preparation/v1",
        "status": "prepared-only-no-training",
        "varianceLaw": "max(tau2[context,query] + sigma2[context,query] / num_cells, 0.05^2)",
        "precision": "global_multiplier / variance",
        "globalNormalization": "single fitting-snapshot scalar makes equal-context/equal-gene weighted mean precision equal the old fitting-query-SD objective weighted mean precision; never minibatch renormalized",
        "normalization": normalization.__dict__,
        "componentProvenance": "tau2 from source3 fitting-only grouped-OOF mean residuals after removing core-control sigma2/n; sigma2 from core controls in each context; all components marked identifiable",
        "shape": [3, 7036],
        "contextIds": context_ids.tolist(),
        "queryIdsSha256": hashlib.sha256(("\n".join(query_ids) + "\n").encode()).hexdigest(),
        "fittingRows": len(train),
        "cellCountUse": "loss precision only; excluded from mean, state, action, query, and context inputs",
        "matchedExperiment": {
            "candidate": "same fixed-query rank128 BP128+presence model, initialization, 12000 batches, seed731, row order, optimizer, and final-only validation as the frozen fixed-basis arm; replace only objective precision",
            "initialization": "repeat the pinned seed731 initializer; do not warm-start from the old final model",
            "control": {"path": str(FIXED_MODEL.relative_to(ROOT)), "sha256": PINS[FIXED_MODEL]},
            "fixedRule": "Use the already frozen fixed-basis development rule; compare candidate with the old fixed-basis final checkpoint in every source context, without an intermediate validation selection.",
        },
        "inputs": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in [DATA, VARIANCE, VARIANCE_SOURCE, REFERENCE, WEIGHTING, FIXED_MODEL, FIXED_REPORT, HELPER]},
        "resource": {"path": str((DERIVED / "precision-components.npz").relative_to(ROOT)).replace("\\", "/"), "sha256": resource_hash},
        "testOrBenchmarkAccessed": False,
    }
    write_json(DERIVED / "manifest.json", audit)
    RESULT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(HELPER, RESULT / "exposure_objective.py")
    write_json(RESULT / "protocol.json", audit)
    write_json(RESULT / "report.json", {**audit, "helperSha256": sha256(RESULT / "exposure_objective.py"), "manifestSha256": sha256(DERIVED / "manifest.json")})
    print(json.dumps({"resourceSha256": resource_hash, "manifestSha256": sha256(DERIVED / "manifest.json"), "reportSha256": sha256(RESULT / "report.json"), "normalization": normalization.__dict__}, indent=2))


if __name__ == "__main__":
    main()
