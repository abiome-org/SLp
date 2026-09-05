#!/usr/bin/env python3
"""Verify a frozen Frangieh cell-state artifact in a fresh process."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load_inference(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_cell_state_inference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen inference source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify(artifact: Path) -> dict[str, object]:
    inference = load_inference(artifact / "source/inference.py")
    frozen = inference.FrozenCellState(artifact, device="cpu")
    with np.load(artifact / "probe-input.npz", allow_pickle=False) as archive:
        inputs = {name: archive[name] for name in archive.files}
    with np.load(artifact / "probe-expected.npz", allow_pickle=False) as archive:
        expected = {name: archive[name] for name in archive.files}

    state = frozen.encode(inputs["rna"], inputs["protein"])
    empty = frozen.forecast(
        np.zeros((3, 1156), dtype=np.float32),
        inputs["context_index"],
        has_action=np.zeros(3, dtype=bool),
    )
    forecast = frozen.forecast(
        inputs["action_features"], inputs["context_index"],
        has_action=np.ones(3, dtype=bool),
    )
    differences = {
        "state": float(np.max(np.abs(state - expected["state"]))),
        "emptyRna": float(np.max(np.abs(empty["rna"] - expected["empty_rna"]))),
        "emptyProtein": float(np.max(np.abs(empty["protein"] - expected["empty_protein"]))),
        "forecastRna": float(np.max(np.abs(forecast["rna"] - expected["forecast_rna"]))),
        "forecastProtein": float(np.max(np.abs(forecast["protein"] - expected["forecast_protein"]))),
        "forecastStateDelta": float(np.max(np.abs(
            forecast["state_delta"] - expected["forecast_state_delta"]
        ))),
    }
    if max(differences.values()) > 1e-5:
        raise ValueError(f"portable replay mismatch: {differences}")
    for head in ("rna", "protein"):
        controls = frozen.reference[f"{head}_controls"][inputs["context_index"]]
        if not np.array_equal(empty[head], controls):
            raise ValueError(f"empty identity failed for {head}")
    return {
        "passed": True,
        "tolerance": 1e-5,
        "maxAbsoluteDifferences": differences,
        "emptyIdentityExact": True,
        "device": "cpu",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.artifact.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
