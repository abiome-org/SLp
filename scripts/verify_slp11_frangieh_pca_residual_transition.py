#!/usr/bin/env python3
"""Fresh-process verification for a frozen residual PCA transition artifact."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse


def load(path: Path):
    spec = importlib.util.spec_from_file_location("portable_pca_residual_inference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load portable inference")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify(artifact: Path) -> dict[str, object]:
    inference = load(artifact / "source/inference.py")
    frozen = inference.FrozenResidualPcaTransition(artifact, device="cpu")
    control_state = frozen.pca.pca.encode(
        sparse.csr_matrix(frozen.pca.rna_controls), frozen.pca.protein_controls
    ).astype(np.float32)
    encoder_difference = float(np.max(np.abs(control_state - frozen.reference["control_state"])))
    if encoder_difference > 1e-6:
        raise ValueError(f"portable PCA encoder mismatch: {encoder_difference}")
    with np.load(artifact / "probe-input.npz", allow_pickle=False) as archive:
        inputs = {name: archive[name] for name in archive.files}
    with np.load(artifact / "probe-expected.npz", allow_pickle=False) as archive:
        expected = {name: archive[name] for name in archive.files}
    forecast = frozen.forecast(inputs["action_features"], inputs["context_index"])
    empty = frozen.forecast(
        inputs["action_features"], inputs["context_index"],
        has_action=np.zeros(len(inputs["context_index"]), dtype=bool),
    )
    differences = {
        head: float(np.max(np.abs(forecast[head] - expected[head])))
        for head in ("rna", "protein", "state_delta")
    }
    if max(differences.values()) > 1e-5:
        raise ValueError(f"portable replay mismatch: {differences}")
    for head in ("rna", "protein"):
        controls = getattr(frozen.pca, f"{head}_controls")[inputs["context_index"]]
        if not np.array_equal(empty[head], controls):
            raise ValueError(f"empty identity failed: {head}")
    if np.count_nonzero(empty["state_delta"]):
        raise ValueError("empty state delta is not zero")
    return {
        "passed": True, "device": "cpu", "tolerance": 1e-5,
        "maxAbsoluteDifferences": differences, "emptyIdentityExact": True,
        "pcaEncoderControlStateMaximumDifference": encoder_difference,
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("artifact", type=Path)
    print(json.dumps(verify(parser.parse_args().artifact.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
