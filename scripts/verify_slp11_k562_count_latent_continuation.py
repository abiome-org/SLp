"""Isolated CPU replay for a frozen two-arm count-latent continuation."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_module(path: Path, expected: str):
    if sha256(path) != expected:
        raise ValueError("inference source checksum mismatch")
    spec = importlib.util.spec_from_file_location("isolated_count_continuation", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def main(artifact: Path) -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    manifest = json.loads((artifact / "artifact-manifest.json").read_text(encoding="utf-8"))
    inference = load_module(
        artifact / "source/inference.py", manifest["sha256"]["source/inference.py"]
    )
    with np.load(artifact / "target-free-probe.npz", allow_pickle=False) as values:
        probe = {key: np.asarray(values[key]) for key in values.files}
    result = {
        "schema": "slp.k562-count-latent-continuation-isolated-cpu-replay/v1",
        "device": "cpu",
        "separateProcess": True,
        "relativeCp10kTolerance": 1e-6,
        "absoluteLog1pTolerance": 1e-6,
        "arms": {},
    }
    passes = True
    for arm in ("count-only", "mean-aux"):
        predictor = inference.Predictor(artifact, arm, device="cpu")
        first = predictor.predict(probe["raw_action_features"], probe["gem_weights"])
        repeated = predictor.predict(probe["raw_action_features"], probe["gem_weights"])
        empty = predictor.predict(
            probe["raw_action_features"],
            probe["gem_weights"],
            action_mask=np.zeros((len(probe["raw_action_features"]), 1), np.bool_),
        )
        expected = probe[f"{arm}_mean_cp10k"]
        difference = np.abs(first["mean_cp10k"].astype(np.float64) - expected.astype(np.float64))
        relative = difference / np.maximum(np.abs(expected), 1.0)
        log_difference = np.abs(
            first["mean_log1p_cp10k"].astype(np.float64)
            - np.log1p(expected.astype(np.float64))
        )
        empty_difference = np.abs(
            empty["mean_cp10k"].astype(np.float64)
            - probe[f"{arm}_empty_cp10k"].astype(np.float64)
        )
        weight_values = np.asarray(probe["gem_weights"], dtype=np.float64)
        weight_values /= weight_values.sum(1, keepdims=True)
        weights = torch.as_tensor(weight_values, dtype=torch.float64)
        basal = torch.as_tensor(predictor.reference["basal_rate"], dtype=torch.float32)
        expected_empty = (basal[None] * weights[..., None]).sum(1).numpy()
        empty_identity_exact = bool(np.array_equal(empty["mean_cp10k"], expected_empty))
        arm_passes = bool(
            np.max(relative) <= 1e-6
            and np.max(log_difference) <= 1e-6
            and np.max(empty_difference) <= 1e-6
            and np.array_equal(first["mean_cp10k"], repeated["mean_cp10k"])
            and empty_identity_exact
        )
        passes &= arm_passes
        result["arms"][arm] = {
            "maximumAbsoluteCp10kDifference": float(np.max(difference)),
            "maximumRelativeCp10kDifferenceWithUnitFloor": float(np.max(relative)),
            "maximumAbsoluteLog1pDifference": float(np.max(log_difference)),
            "maximumAbsoluteEmptyDifference": float(np.max(empty_difference)),
            "repeatedMeanBitExact": bool(
                np.array_equal(first["mean_cp10k"], repeated["mean_cp10k"])
            ),
            "emptyMeanExact": empty_identity_exact,
            "passes": arm_passes,
        }
    result["passes"] = passes
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_slp11_k562_count_latent_continuation.py ARTIFACT")
    main(Path(sys.argv[1]).resolve())
