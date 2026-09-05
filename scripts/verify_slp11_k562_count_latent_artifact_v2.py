#!/usr/bin/env python3
"""Isolated CPU replay for the frozen K562 count-latent artifact."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load(path: Path):
    spec = importlib.util.spec_from_file_location("count_latent_inference_v2", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify(artifact: Path, inference_path: Path) -> dict[str, object]:
    inference = load(inference_path)
    predictor = inference.Predictor(artifact, device="cpu")
    with np.load(artifact / "target-free-probe.npz", allow_pickle=False) as source:
        probe = {name: np.asarray(source[name]) for name in source.files}
    with np.load(artifact / "reference.npz", allow_pickle=False) as source:
        basal = np.asarray(source["basal_rate"])
    first = predictor.predict(
        probe["raw_action_features"], probe["gem_group_weights"],
        action_mask=probe["action_mask"], chunk_size=1024,
    )
    second = predictor.predict(
        probe["raw_action_features"], probe["gem_group_weights"],
        action_mask=probe["action_mask"], chunk_size=1024,
    )
    cp_difference = np.abs(first["mean_cp10k"] - probe["expected_cp10k"])
    relative = cp_difference / np.maximum(np.abs(probe["expected_cp10k"]), 1.0)
    log_difference = np.abs(
        first["mean_log1p_cp10k"] - probe["expected_log1p_cp10k"]
    )
    result = {
        "schema": "slp.k562-count-latent-isolated-cpu-replay/v2",
        "maximumAbsoluteCp10kDifference": float(cp_difference.max()),
        "maximumRelativeCp10kDifferenceWithUnitFloor": float(relative.max()),
        "maximumAbsoluteLog1pDifference": float(log_difference.max()),
        "relativeCp10kTolerance": 1e-6,
        "absoluteLog1pTolerance": 1e-6,
        "repeatedMeanBitExact": bool(np.array_equal(
            first["mean_cp10k"], second["mean_cp10k"]
        )),
        "emptyMeanExact": bool(np.array_equal(first["mean_cp10k"][2], basal[1])),
        "separateProcess": True,
        "device": "cpu",
    }
    result["passes"] = bool(
        result["maximumRelativeCp10kDifferenceWithUnitFloor"] <= 1e-6
        and result["maximumAbsoluteLog1pDifference"] <= 1e-6
        and result["repeatedMeanBitExact"]
        and result["emptyMeanExact"]
    )
    if not result["passes"]:
        raise RuntimeError("isolated CPU replay failed the amended numerical gate")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("inference", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.artifact.resolve(), args.inference.resolve()), sort_keys=True))
