#!/usr/bin/env python3
"""Verify frozen BP-augmented Nyström models without reading target values."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def load_wrapper(path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("slp11_bp_nystrom_frozen", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen BP Nyström wrapper")
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
    wrapper_path = artifact / "source" / "run_slp11_bp_nystrom_screen.py"
    wrapper = load_wrapper(wrapper_path)
    helper = wrapper.load_helper(artifact / "source" / "run_slp11_nystrom_rbf_baseline.py")
    inputs = protocol["inputs"]
    for item in inputs.values():
        if helper.sha256_file(Path(item["path"])) != item["sha256"]:
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
    actions = action_ids[validation]
    physical_rows = np.stack([physical[physical_index[gene]] for gene in actions])
    bp_rows = np.stack([bp[bp_index[gene]] for gene in actions])
    present_rows = np.asarray([present[bp_index[gene]] for gene in actions])[:, None]
    features = np.concatenate((physical_rows, bp_rows, present_rows), axis=1)
    with np.load(artifact / "development-predictions.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["record_ids"].astype(str), record_ids[validation]):
            raise RuntimeError("record identity mismatch")
        if not np.array_equal(saved["action_ids"].astype(str), actions):
            raise RuntimeError("action identity mismatch")
        if not np.array_equal(saved["context_index"].astype(np.int64), context_index[validation]):
            raise RuntimeError("context identity mismatch")
        if not np.array_equal(saved["query_ids"].astype(str), query_ids):
            raise RuntimeError("query identity mismatch")
        saved_mean = saved["mean"]
    errors: dict[str, float] = {}
    for context in range(3):
        local = np.flatnonzero(context_index[validation] == context)
        with np.load(artifact / f"model-context-{context}.npz", allow_pickle=False) as model:
            kernel = helper.NystromMap(
                feature_mean=model["feature_mean"],
                feature_scale=model["feature_scale"],
                bandwidth=float(model["bandwidth"]),
                landmark_ids=tuple(model["landmark_ids"].astype(str)),
                landmarks=model["standardized_landmarks"],
                kernel_basis=model["kernel_basis"],
                eigenvalues=model["kernel_eigenvalues"],
            )
            mapped = kernel.transform(features[local])
            rotated = (mapped - model["ridge_feature_mean"]) @ model["ridge_eigenvectors"]
            prediction = model["target_mean"] + (
                rotated / (model["ridge_eigenvalues"] + float(str(model["selected_alpha"])))
            ) @ model["ridge_rhs"]
        error = float(np.max(np.abs(prediction - saved_mean[local])))
        errors[f"context{context}"] = error
        if error > 1e-4:
            raise RuntimeError(f"reload error context{context}: {error}")
    verifier_copy = artifact / "source" / Path(__file__).name
    if not verifier_copy.exists():
        shutil.copyfile(Path(__file__), verifier_copy)
    report = {
        "schema": "slp.bp-nystrom-source-three-verification/v1",
        "status": "pass",
        "targetValuesRead": False,
        "checks": {
            "allPinnedInputHashes": True,
            "savedIdentityExact": True,
            "staticFeatureAlignmentExact": True,
            "allModelsReload": True,
            "maximumAbsoluteErrorAtMost1e-4": True,
        },
        "maximumAbsoluteErrorByContext": errors,
        "predictionsSha256": helper.sha256_file(artifact / "development-predictions.npz"),
        "wrapperSha256": helper.sha256_file(wrapper_path),
        "helperSha256": helper.sha256_file(artifact / "source" / "run_slp11_nystrom_rbf_baseline.py"),
        "verifierSha256": helper.sha256_file(verifier_copy),
    }
    (artifact / "verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
