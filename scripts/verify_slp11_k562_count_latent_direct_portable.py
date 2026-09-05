"""Append-only target-free direct-versus-portable continuation replay."""
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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def direct_prediction(predictor, raw: np.ndarray, weights: np.ndarray) -> np.ndarray:
    ref = predictor.reference
    safe = np.clip(
        (raw[:, None, :] - ref["feature_mean"]) / ref["feature_scale"],
        -float(ref["feature_clip"]),
        float(ref["feature_clip"]),
    ).astype(np.float32)
    mixture = np.asarray(weights, dtype=np.float64)
    mixture /= mixture.sum(1, keepdims=True)
    groups = len(ref["gem_group_ids"])
    query = torch.as_tensor(ref["query_features"])
    basal = torch.as_tensor(ref["basal_rate"])
    mask = torch.ones_like(basal, dtype=torch.bool)
    context = predictor.model.encode_context(query, basal, mask)
    prior = predictor.model.prior_from_context(
        torch.as_tensor(np.repeat(safe, groups, axis=0)),
        torch.ones((len(raw) * groups, 1), dtype=torch.bool),
        context.repeat(len(raw), 1),
    )
    pieces = []
    for left in range(0, len(query), 1024):
        local = basal[:, left : left + 1024]
        local = local[None].expand(len(raw), -1, -1).reshape(len(raw) * groups, -1)
        rate = predictor.model.population_mean(
            prior, query[left : left + 1024], local
        ).reshape(len(raw), groups, -1)
        pieces.append(
            (rate * torch.as_tensor(mixture[..., None])).sum(1).cpu().numpy()
        )
    return np.concatenate(pieces, axis=1)


def main(artifact: Path) -> None:
    output = artifact / "postrun-direct-vs-portable-replay.json"
    if output.exists():
        raise FileExistsError("immutable direct-versus-portable replay exists")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    manifest = json.loads((artifact / "artifact-manifest.json").read_text(encoding="utf-8"))
    inference_path = artifact / "source/inference.py"
    if sha256(inference_path) != manifest["sha256"]["source/inference.py"]:
        raise ValueError("frozen inference hash mismatch")
    inference = load_module(inference_path, "postrun_count_continuation_inference")
    with np.load(artifact / "target-free-probe.npz", allow_pickle=False) as values:
        raw = np.asarray(values["raw_action_features"])
        weights = np.asarray(values["gem_weights"])
    report = {
        "schema": "slp.k562-count-latent-direct-portable-replay/v1",
        "sourceSha256": sha256(Path(__file__).resolve()),
        "protocolSha256": sha256(artifact / "protocol.json"),
        "artifactManifestSha256": sha256(artifact / "artifact-manifest.json"),
        "targetFreeProbeSha256": sha256(artifact / "target-free-probe.npz"),
        "device": "cpu",
        "relativeCp10kTolerance": 1e-6,
        "absoluteLog1pTolerance": 1e-6,
        "arms": {},
        "developmentRead": False,
        "testRead": False,
    }
    passes = True
    for arm in ("count-only", "mean-aux"):
        predictor = inference.Predictor(artifact, arm, device="cpu")
        portable = predictor.predict(raw, weights)["mean_cp10k"].astype(np.float64)
        direct = direct_prediction(predictor, raw, weights).astype(np.float64)
        difference = np.abs(portable - direct)
        relative = difference / np.maximum(np.abs(direct), 1.0)
        log_difference = np.abs(np.log1p(portable) - np.log1p(direct))
        arm_passes = bool(
            np.max(relative) <= 1e-6 and np.max(log_difference) <= 1e-6
        )
        passes &= arm_passes
        report["arms"][arm] = {
            "maximumAbsoluteCp10kDifference": float(np.max(difference)),
            "maximumRelativeCp10kDifferenceWithUnitFloor": float(np.max(relative)),
            "maximumAbsoluteLog1pDifference": float(np.max(log_difference)),
            "passes": arm_passes,
        }
    report["passes"] = passes
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_slp11_k562_count_latent_direct_portable.py ARTIFACT")
    main(Path(sys.argv[1]).resolve())
