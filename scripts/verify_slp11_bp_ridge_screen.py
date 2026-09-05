#!/usr/bin/env python3
"""Verify frozen BP ridge models without reading molecular outcomes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def load_runner(path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("slp11_bp_ridge_frozen", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen BP ridge runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact
    protocol = json.loads((artifact / "protocol.json").read_text(encoding="utf-8"))
    runner_path = artifact / "source" / "run_slp11_bp_ridge_screen.py"
    runner = load_runner(runner_path)
    inputs = protocol["inputs"]
    for item in inputs.values():
        if runner.sha256_file(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"pinned input hash mismatch: {item['path']}")
    with np.load(inputs["development"]["path"], allow_pickle=False) as data:
        validation = data["split_validation"].astype(np.int64)
        action_ids = data["action_ids"].astype(str)
        context_index = data["context_index"].astype(np.int64)
        record_ids = data["record_ids"].astype(str)
        query_ids = data["query_ids"].astype(str)
    with np.load(inputs["physical"]["path"], allow_pickle=False) as pack:
        physical_ids = pack["entity_id"].astype(str)
        physical = pack["feature_values"].astype(np.float32)
    with np.load(inputs["bp"]["path"], allow_pickle=False) as pack:
        bp_ids = pack["entity_id"].astype(str)
        bp = pack["feature_values"].astype(np.float32)
        present = pack["annotation_present"].astype(np.float32)
    physical_index = {gene: index for index, gene in enumerate(physical_ids)}
    bp_index = {gene: index for index, gene in enumerate(bp_ids)}
    validation_actions = action_ids[validation]
    physical_rows = np.stack([physical[physical_index[gene]] for gene in validation_actions])
    bp_rows = np.stack([bp[bp_index[gene]] for gene in validation_actions])
    present_rows = np.asarray([present[bp_index[gene]] for gene in validation_actions])[:, None]
    values = {
        "physical1156": physical_rows,
        "physical1156_bp128_present1": np.concatenate((physical_rows, bp_rows, present_rows), axis=1),
    }
    with np.load(artifact / "development-predictions.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["record_ids"].astype(str), record_ids[validation]):
            raise RuntimeError("prediction record identity mismatch")
        if not np.array_equal(saved["action_ids"].astype(str), validation_actions):
            raise RuntimeError("prediction action identity mismatch")
        if not np.array_equal(saved["context_index"].astype(np.int64), context_index[validation]):
            raise RuntimeError("prediction context identity mismatch")
        if not np.array_equal(saved["query_ids"].astype(str), query_ids):
            raise RuntimeError("prediction query identity mismatch")
        saved_means = {arm: saved[arm] for arm in values}
    errors: dict[str, float] = {}
    for arm, matrix in values.items():
        for context in range(3):
            local = np.flatnonzero(context_index[validation] == context)
            model_path = artifact / f"model-{arm}-context-{context}.npz"
            with np.load(model_path, allow_pickle=False) as model:
                if str(model["feature_block"]) != arm or not np.array_equal(model["query_ids"].astype(str), query_ids):
                    raise RuntimeError("saved model identity mismatch")
                state = {key: model[key] for key in ("feature_mean", "feature_scale", "target_mean", "eigenvalues", "eigenvectors", "rhs")}
                reloaded = runner.predict_state(state, matrix[local], str(model["selected_alpha"]))
            error = float(np.max(np.abs(reloaded - saved_means[arm][local])))
            errors[f"{arm}/context{context}"] = error
            if error > 1e-5:
                raise RuntimeError(f"model reload mismatch: {arm}/context{context} {error}")
    verifier_copy = artifact / "source" / Path(__file__).name
    if not verifier_copy.exists():
        shutil.copyfile(Path(__file__), verifier_copy)
    report = {
        "schema": "slp.bp-ridge-source-three-verification/v1",
        "status": "pass",
        "targetValuesRead": False,
        "checks": {
            "allPinnedInputHashes": True,
            "savedPredictionIdentityExact": True,
            "staticFeatureIdentityExact": True,
            "allSixModelsReload": True,
            "maximumAbsoluteErrorAtMost1e-5": True,
        },
        "maximumAbsoluteErrorByModel": errors,
        "predictionsSha256": runner.sha256_file(artifact / "development-predictions.npz"),
        "runnerSha256": runner.sha256_file(runner_path),
        "verifierSha256": runner.sha256_file(verifier_copy),
    }
    (artifact / "verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
