"""Compare every context in a joint-world bundle across two CPU runtimes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_manifest(model: Path):
    manifest_path = model / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = set(manifest["files"])
    actual = {p.relative_to(model).as_posix() for p in model.rglob("*")
              if p.is_file() and p != manifest_path}
    if actual != expected:
        raise ValueError(f"export manifest inventory differs: missing={expected-actual}, extra={actual-expected}")
    for relative, expected_hash in manifest["files"].items():
        if _sha256(model / relative) != expected_hash:
            raise ValueError(f"export manifest hash mismatch: {relative}")
    return {"manifestSha256": _sha256(manifest_path), "verifiedFiles": len(expected)}


def _load_inference(model: Path):
    sys.dont_write_bytecode = True
    path = model / "inference.py"
    sys.path.insert(0, str(model))
    spec = importlib.util.spec_from_file_location("portable_joint_world_inference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _requests(bundle, context: str):
    """Five target-free requests, including empty, single and double actions."""
    q = bundle.query_ids(context)
    f = bundle.config.feature_dim
    scale = float(bundle.settings["contexts"][context]["response_scale"])
    phase = (sum(map(ord, context)) % 17) / 17.0
    action_a = bundle.feature_mean + bundle.feature_scale * np.sin(np.arange(f) * .017 + phase)
    action_b = bundle.feature_mean + bundle.feature_scale * np.cos(np.arange(f) * .013 + phase)
    actions = np.zeros((5, 2, f), np.float32)
    actions[2, 0] = action_a
    actions[3, 0] = action_b
    actions[4] = np.stack([action_a, action_b])
    mask = np.zeros((5, 2), bool)
    mask[2:, 0] = True
    mask[4] = True
    basal = np.zeros((5, len(q)), np.float64)
    observed = basal.copy()
    shift = scale * .01 * np.sin(np.arange(len(q)) * .031 + phase)
    observed[1] += shift
    observed[3] += shift
    adapter = bundle.adapters[context]
    kwargs = {}
    if "control_context_values" in adapter:
        kwargs["control_context_values"] = np.broadcast_to(adapter["control_context_values"], basal.shape).copy()
        kwargs["control_context_mask"] = np.broadcast_to(adapter["control_context_mask"], basal.shape).copy()
    return actions, mask, basal, observed, kwargs


def _run(model: Path, checkpoint: str, output: Path):
    inference = _load_inference(model)
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    bundle = inference.JointWorldBundle(model, checkpoint, "cpu")
    arrays = {}
    for context in sorted(bundle.settings["contexts"]):
        actions, mask, basal, observed, kwargs = _requests(bundle, context)
        prediction = bundle.predict(context, actions, mask, basal, observed=observed,
                                    batch_size=4, query_chunk=512, **kwargs)
        if not np.array_equal(prediction[:2], observed[:2]):
            raise AssertionError(f"{context}: empty actions did not preserve observed state exactly")
        arrays[f"{context}_query_ids"] = bundle.query_ids(context)
        arrays[f"{context}_supported"] = bundle.supported_query_mask(context)
        arrays[f"{context}_prediction"] = prediction
        arrays[f"{context}_empty_observed"] = observed[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    return f"/mnt/{drive}/" + "/".join(resolved.parts[1:])


def _compare(reference: Path, replay: Path):
    metrics = {}
    with np.load(reference, allow_pickle=False) as left, np.load(replay, allow_pickle=False) as right:
        if set(left.files) != set(right.files):
            raise ValueError("runtime replay keys differ")
        contexts = sorted(key.removesuffix("_prediction") for key in left.files
                          if key.endswith("_prediction"))
        if not contexts:
            raise ValueError("runtime replay contains no context predictions")
        for context in contexts:
            for suffix in ("query_ids", "supported"):
                key = f"{context}_{suffix}"
                if not np.array_equal(left[key], right[key]):
                    raise ValueError(f"{key} differs across runtimes")
            empty_key = f"{context}_empty_observed"
            empty_delta = (np.asarray(right[empty_key], np.float64)
                           - np.asarray(left[empty_key], np.float64))
            empty_drift = float(np.max(np.abs(empty_delta)))
            if empty_drift > 1e-12:
                raise ValueError(f"{empty_key} drift exceeds tolerance: {empty_drift}")
            key = f"{context}_prediction"
            delta = np.asarray(right[key], np.float64) - np.asarray(left[key], np.float64)
            if not np.all(np.isfinite(delta)):
                raise ValueError(f"{key} contains non-finite runtime drift")
            metrics[context] = {"maxAbsDrift": float(np.max(np.abs(delta))),
                                "meanAbsDrift": float(np.mean(np.abs(delta))),
                                "rmseDrift": float(np.sqrt(np.mean(np.square(delta)))),
                                "emptyObservedMaxAbsDrift": empty_drift,
                                "values": int(delta.size)}
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--checkpoint", default="step-020000.safetensors")
    p.add_argument("--python-runtime", required=True,
                   help="Candidate Python executable, or wsl.exe with --runtime-arg /path/to/python")
    p.add_argument("--runtime-arg", action="append", default=[])
    p.add_argument("--runtime-path-style", choices=("native", "wsl"), default="native")
    p.add_argument("--runtime-site-packages",
                   help="Linux site-packages admitted explicitly by the -S worker launcher")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker-output", type=Path)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    sys.dont_write_bytecode = True
    if a.worker:
        _run(a.model, a.checkpoint, a.worker_output)
        return
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = _verify_manifest(a.model)
    reference = a.output / "windows-reference.npz"
    replay = a.output / "runtime-replay.npz"
    _run(a.model, a.checkpoint, reference)
    cv = (lambda x: _wsl_path(x)) if a.runtime_path_style == "wsl" else (lambda x: str(x.resolve()))
    if not a.runtime_site_packages:
        raise ValueError("--runtime-site-packages is required for isolated replay")
    launcher = Path(__file__).with_name("run_isolated_joint_world_worker.py")
    isolation = a.output / "runtime-isolation.json"
    command = [a.python_runtime, *a.runtime_arg, "-S", cv(launcher),
               "--site-packages", a.runtime_site_packages,
               "--verifier", cv(Path(__file__)), "--model", cv(a.model),
               "--checkpoint", a.checkpoint, "--python-runtime", a.python_runtime,
               "--worker-output", cv(replay), "--isolation-output", cv(isolation)]
    subprocess.run(command, check=True)
    metrics = _compare(reference, replay)
    maximum_drift = max(v["maxAbsDrift"] for v in metrics.values())
    if maximum_drift > 1e-5:
        raise ValueError(f"forecast runtime drift exceeds 1e-5: {maximum_drift}")
    report = {"schema": "slp.joint-world-portability/v1", "cpuOnly": True,
              "contexts": metrics, "maxAbsDrift": maximum_drift,
              "maxAbsDriftTolerance": 1e-5,
              "torchThreads": 4, "runtimeIsolation": json.loads(isolation.read_text()),
              "exportManifest": manifest,
              "referenceSha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
              "replaySha256": hashlib.sha256(replay.read_bytes()).hexdigest()}
    (a.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
