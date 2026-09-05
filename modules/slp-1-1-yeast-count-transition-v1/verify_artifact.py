"""Fresh-process target-free verification for a yeast transition artifact."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    artifact = Path(sys.argv[1])
    manifest = json.loads((artifact / "artifact-manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if sha256(artifact / name) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    path = artifact / "source/inference.py"
    spec = importlib.util.spec_from_file_location(
        "slp11_yeast_artifact_inference", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load packaged inference")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    predictor = module.Predictor(artifact)
    with np.load(artifact / "target-free-probe.npz", allow_pickle=False) as probe:
        batch_index = probe["batch_index"]
        actual = predictor.predict(probe["raw_action_features"], batch_index)
        drift = float(np.max(np.abs(actual - probe["expected_mean"])))
        empty = predictor.predict_empty(batch_index)
    with np.load(artifact / "reference.npz", allow_pickle=False) as reference:
        exact_empty = bool(
            np.array_equal(empty, reference["control_mean"][batch_index])
        )
    if drift > 1e-5 or not exact_empty:
        raise ValueError("portable replay or empty identity failed")
    print(json.dumps({"maximumMeanDrift": drift, "emptyMeanBitExact": exact_empty}))


if __name__ == "__main__":
    main()
