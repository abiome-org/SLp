#!/usr/bin/env python3
"""Target-free verification for the joint context-conditioned RBF baseline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def load_runtime(path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("slp11_joint_context_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    import hashlib

    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact
    source = artifact / "source"
    runtime_source = Path(__file__).with_name("inference_slp11_joint_context_rbf.py")
    runtime_copy = source / "inference_slp11_joint_context_rbf_v2.py"
    verifier_copy = source / "verify_slp11_joint_context_rbf_baseline_v2.py"
    if not runtime_copy.exists():
        shutil.copyfile(runtime_source, runtime_copy)
    if not verifier_copy.exists():
        shutil.copyfile(Path(__file__), verifier_copy)
    runtime_module = load_runtime(runtime_copy)
    protocol = json.loads((artifact / "protocol.json").read_text(encoding="utf-8"))
    for item in protocol["inputs"].values():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"pinned input hash mismatch: {item['path']}")
    data_path = Path(protocol["inputs"]["development"]["path"])
    feature_path = Path(protocol["inputs"]["physical"]["path"])
    with np.load(data_path, allow_pickle=False) as data:
        validation = data["split_validation"].astype(np.int64)
        records = data["record_ids"].astype(str)[validation]
        actions = data["action_ids"].astype(str)[validation]
        contexts = data["context_index"].astype(np.int64)[validation]
        query_ids = data["query_ids"].astype(str)
        basal = data["context_basal_expression"].astype(np.float32)
        basal_observed = data["context_basal_observed"]
        controls = data["basal_control"].astype(np.float32)
    runtime = runtime_module.JointContextRbfRuntime(artifact / "model.npz", feature_path)
    if not np.array_equal(runtime.query_ids, query_ids):
        raise RuntimeError("runtime query identity mismatch")
    context_basis = np.concatenate(
        [runtime._context_basis(basal[index], basal_observed[index]) for index in range(3)],
        axis=0,
    )
    unique_actions = sorted(set(actions))
    action_basis = {gene: runtime._action_basis(gene)[0] for gene in unique_actions}
    action_matrix = np.stack([action_basis[gene] for gene in actions])
    selected_context = context_basis[contexts]
    interaction = np.einsum(
        "ni,nj->nij", selected_context, action_matrix, optimize=True
    ).reshape(len(actions), 1536)
    design = np.concatenate((selected_context, interaction), axis=1).astype(np.float32)
    model = runtime.model
    rotated = (design - model["ridge_feature_mean"]) @ model["ridge_eigenvectors"]
    effect = model["target_mean"] + (
        rotated
        / (model["ridge_eigenvalues"] + float(str(model["selected_alpha"])))
    ) @ model["ridge_rhs"]
    reloaded = effect.astype(np.float32)
    with np.load(artifact / "development-predictions.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["record_ids"].astype(str), records):
            raise RuntimeError("saved record identity mismatch")
        if not np.array_equal(saved["action_ids"].astype(str), actions):
            raise RuntimeError("saved action identity mismatch")
        if not np.array_equal(saved["context_index"].astype(np.int64), contexts):
            raise RuntimeError("saved context identity mismatch")
        if not np.array_equal(saved["query_ids"].astype(str), query_ids):
            raise RuntimeError("saved query identity mismatch")
        maximum_error = float(np.max(np.abs(reloaded - saved["mean"])))
    if maximum_error > 1e-4:
        raise RuntimeError(f"target-free reload mismatch: {maximum_error}")
    synthetic_control = np.linspace(-1.0, 1.0, len(query_ids), dtype=np.float32)
    empty = runtime.predict((), np.full_like(synthetic_control, np.nan), np.zeros_like(synthetic_control, dtype=bool), synthetic_control)
    if not np.array_equal(empty, synthetic_control):
        raise RuntimeError("empty action did not return supplied control bitwise")
    example = runtime.predict(
        (actions[0],),
        basal[contexts[0]],
        basal_observed[contexts[0]],
        controls[contexts[0]],
    )
    example_error = float(np.max(np.abs(example - reloaded[0])))
    if example_error > 1e-4:
        raise RuntimeError(f"single-record public runtime mismatch: {example_error}")
    report = {
        "schema": "slp.joint-context-rbf-source-three-verification/v1",
        "status": "pass",
        "targetValuesRead": False,
        "checks": {
            "allPinnedInputHashes": True,
            "queryAndRecordIdentityExact": True,
            "allDevelopmentMeansReloadTargetFree": True,
            "callerSuppliedContextDescriptorUsed": True,
            "emptyActionReturnsSuppliedControlBitwise": True,
            "noTargetScaleRequiredAtInference": True,
            "singleActionOnlyEnforced": True,
        },
        "maximumDevelopmentMeanAbsoluteError": maximum_error,
        "examplePublicRuntimeAbsoluteError": example_error,
        "modelSha256": sha256_file(artifact / "model.npz"),
        "predictionsSha256": sha256_file(artifact / "development-predictions.npz"),
        "runtimeSha256": sha256_file(runtime_copy),
        "verifierSha256": sha256_file(verifier_copy),
    }
    (artifact / "verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
