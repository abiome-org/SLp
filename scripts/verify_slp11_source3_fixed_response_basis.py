#!/usr/bin/env python3
"""Fresh-process target-free verifier for a fixed-response-basis artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path):
    spec = importlib.util.spec_from_file_location("fixed_basis_portable_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def predict(model, reference, raw_actions, contexts):
    normalized = (raw_actions - reference["feature_mean"]) / reference["feature_std"]
    query = (reference["query_features"] - reference["query_feature_mean"]) / reference[
        "query_feature_std"
    ]
    selected = reference["context_query_indices"]
    with torch.no_grad():
        return model(
            torch.as_tensor(normalized),
            torch.as_tensor(reference["fixed_query_coordinates"]),
            torch.as_tensor(reference["control_mean"][contexts]),
            torch.as_tensor(reference["delta_amplitude"]),
            torch.as_tensor(reference["objective_query_scale"][contexts]),
            torch.as_tensor(query[selected]),
            torch.as_tensor(reference["context_values"][contexts]),
            torch.as_tensor(reference["context_mask"][contexts], dtype=torch.bool),
        )


def verify(root: Path) -> dict[str, object]:
    manifest = json.loads((root / "artifact-manifest.json").read_text())
    for relative, digest in manifest["sha256"].items():
        if sha256(root / relative) != digest:
            raise ValueError(f"artifact hash mismatch: {relative}")
    module = load(root / "source/transition_model.py")
    with np.load(root / "reference.npz", allow_pickle=False) as archive:
        reference = {name: archive[name] for name in archive.files}
    model = module.FixedQueryTransition(
        module.Config(
            len(reference["feature_mean"]),
            len(reference["query_feature_mean"]),
            state_dim=int(reference["state_dim"]),
            hidden_dim=int(reference["hidden_dim"]),
            dropout=float(reference["dropout"]),
        )
    )
    model.load_state_dict(load_file(root / "model.safetensors"))
    model.eval()
    with np.load(root / "target-free-probe.npz", allow_pickle=False) as probe:
        result = predict(model, reference, probe["raw_action_features"], probe["context_index"])
        difference = float(np.max(np.abs(result["mean"].numpy() - probe["expected_mean"])))
        contexts = sorted(set(probe["context_index"].tolist()))
        rows = len(probe["context_index"])
    empty = np.empty((3, 0, len(reference["feature_mean"])), dtype=np.float32)
    empty_result = predict(model, reference, empty, np.arange(3, dtype=np.int64))
    audit = {
        "probeRows": rows,
        "probeContexts": contexts,
        "probeMaximumAbsoluteDifference": difference,
        "probeMeanWithin1e5": difference <= 1e-5,
        "emptyMeanBitExact": bool(
            np.array_equal(empty_result["mean"].numpy(), reference["control_mean"])
        ),
        "emptyDeltaNonzero": int(torch.count_nonzero(empty_result["delta"])),
        "sourceModelReferenceOnly": True,
    }
    if not (
        audit["probeRows"] == 6
        and audit["probeContexts"] == [0, 1, 2]
        and audit["probeMeanWithin1e5"]
        and audit["emptyMeanBitExact"]
        and audit["emptyDeltaNonzero"] == 0
    ):
        raise RuntimeError(f"portable verification failed: {audit}")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.artifact.resolve(strict=True)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
