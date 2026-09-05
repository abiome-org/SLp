#!/usr/bin/env python3
"""Append-only finalization of completed shared-context checkpoints.

The frozen training runner completed both arms and saved their models before a
four-row target-free probe exposed a bounded slicing bug.  This continuation
does not train or alter weights.  It corrects only the probe slice, performs
portable replay and fitting diagnostics, and freezes metadata-only development
forecasts before any development count member is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results/slp11-transition/human-essential-count-shared-context-seed731-v1"
RUNNER = ROOT / "scripts/run_slp11_count_world_shared_context.py"
K562_BASE = ROOT / "scripts/run_slp11_k562_count_latent_state.py"
EXPECTED = {
    "protocol": "2b21d0aa4f609b5bc9d68632225b596783420efd0bd590aad4d3229d505a1954",
    "manifest": "",  # populated by prepare after validating frozen members
    "runner": "8fdf92ecf837eb72c4d107cfc1ed053eff071671172079fd2db001dcdbb10091",
    "k562-only": "49f782166291f8ae658f6895f92e1d1268b599a74fda50f206cacb039bf7afbb",
    "joint-alternating": "cff1f691130bdf78bc574dfdcc449d5aeb8a3f81914f37ecf3f8c49686b87dff",
}


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


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def bounded_slices(total: int, limit: int | None, chunk: int):
    """Yield slices that never pass a requested row limit."""
    if total < 0 or chunk <= 0 or (limit is not None and limit < 0):
        raise ValueError("invalid row bound")
    stop = total if limit is None else min(total, limit)
    return tuple(slice(left, min(left + chunk, stop)) for left in range(0, stop, chunk))


@torch.no_grad()
def direct_prediction(model, panel, device, *, limit: int, chunk: int = 16):
    query = torch.as_tensor(np.array(panel.query_features, copy=True), device=device)
    basal = torch.as_tensor(np.array(panel.basal_rate, copy=True), device=device)
    contexts = model.encode_context(query, basal, torch.ones_like(basal, dtype=torch.bool))
    pieces = []
    groups = len(panel.context_ids)
    for rows in bounded_slices(len(panel.gene_ids), limit, chunk):
        action = torch.as_tensor(
            np.array(panel.gene_action_features[rows, None], copy=True), device=device
        )
        count = len(action)
        expanded = action[:, None].expand(-1, groups, -1, -1).reshape(
            count * groups, 1, -1
        )
        prior = model.prior_from_context(
            expanded,
            torch.ones((count * groups, 1), dtype=torch.bool, device=device),
            contexts.repeat(count, 1),
        )
        rate = model.population_mean(
            prior,
            query,
            basal[None].expand(count, -1, -1).reshape(count * groups, -1),
        ).reshape(count, groups, -1)
        weights = torch.as_tensor(
            np.array(panel.population_context_weights[rows], copy=True), device=device
        )
        pieces.append(torch.log1p((rate * weights[..., None]).sum(1)).cpu().numpy())
    return np.concatenate(pieces) if pieces else np.empty((0, len(panel.query_ids)))


def protocol(artifact: Path):
    manifest = json.loads((artifact / "artifact-manifest.json").read_text(encoding="utf-8"))
    return {
        "schema": "slp.human-essential-count-shared-context-finalization-protocol/v2",
        "reason": "Both frozen arms completed. The original four-row GPU replay failed because limit=4 returned one full 16-row chunk, producing a 4-versus-16 shape mismatch.",
        "scope": "No retraining or weight changes. Correct bounded probe slicing; write the previously absent target-free-gpu-probe.npz expected by the frozen verifier; verify GPU and isolated CPU artifact replay; compute fitting-only diagnostics; freeze metadata-only K562/RPE1 development forecasts; then stop before development outcome access.",
        "supersedes": {
            "protocol": "FINALIZATION-PROTOCOL.json",
            "sha256": sha256(artifact / "FINALIZATION-PROTOCOL.json"),
            "failure": "Its corrected probe used a versioned filename while the frozen isolated verifier requires target-free-gpu-probe.npz.",
        },
        "sliceRule": "For total rows N, requested limit L and chunk C, emit [left,min(left+C,min(N,L))) while left<min(N,L).",
        "modelSha256": {
            arm: manifest["sha256"][f"arms/{arm}.safetensors"]
            for arm in ("k562-only", "joint-alternating")
        },
        "originalProtocolSha256": sha256(artifact / "protocol.json"),
        "originalArtifactManifestSha256": sha256(artifact / "artifact-manifest.json"),
        "failingRunnerSha256": sha256(artifact / "source/runner.py"),
        "finalizerSha256": sha256(Path(__file__).resolve()),
        "developmentCountMembersOpened": False,
        "reconstructionHeldOpened": False,
        "testOpened": False,
    }


def prepare(artifact: Path = ARTIFACT):
    if sha256(artifact / "protocol.json") != EXPECTED["protocol"]:
        raise ValueError("original protocol changed")
    if sha256(artifact / "source/runner.py") != EXPECTED["runner"]:
        raise ValueError("failing runner changed")
    manifest = json.loads((artifact / "artifact-manifest.json").read_text(encoding="utf-8"))
    for arm in ("k562-only", "joint-alternating"):
        path = artifact / f"arms/{arm}.safetensors"
        if sha256(path) != EXPECTED[arm] or sha256(path) != manifest["sha256"][f"arms/{arm}.safetensors"]:
            raise ValueError(f"frozen model changed: {arm}")
    value = protocol(artifact)
    path = artifact / "FINALIZATION-PROTOCOL-V2.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("finalization protocol changed")
    else:
        write_json(path, value)
    return value


def load_models(artifact, runner, panels):
    core = load_module(artifact / "source/count_latent_state.py", "shared_final_core")
    device = torch.device("cuda")
    models = {}
    for arm in runner.ARMS:
        model = core.CountLatentState(core.Config(**runner.MODEL_CONFIG)).to(device)
        model.load_state_dict(load_file(str(artifact / f"arms/{arm}.safetensors")))
        models[arm] = model.eval()
    return models, core


def corrected_gpu_probe(artifact, runner, models, panels):
    inference = load_module(artifact / "source/inference.py", "shared_final_inference")
    with np.load(artifact / "target-free-probe-inputs.npz", allow_pickle=False) as archive:
        inputs = {name: np.asarray(archive[name]) for name in archive.files}
    arrays, metrics = {}, {}
    for source, panel in panels.items():
        raw = inputs[f"{source}_raw_action_features"]
        weights = inputs[f"{source}_context_weights"]
        for arm, model in models.items():
            predictor = inference.Predictor(artifact, arm, source, device="cuda")
            actual = predictor.predict(raw, weights)
            direct = direct_prediction(model, panel, torch.device("cuda"), limit=4)
            difference = np.abs(actual["mean_log1p_cp10k"].astype(np.float64) - direct)
            empty = predictor.predict(
                raw, weights, action_mask=np.zeros((4, 1), np.bool_)
            )
            expected_empty = weights.astype(np.float64) @ panel.basal_rate.astype(np.float64)
            empty_difference = np.abs(empty["mean_cp10k"] - expected_empty)
            key = f"{source}_{arm}"
            arrays[f"{key}_mean_log1p_cp10k"] = actual["mean_log1p_cp10k"]
            arrays[f"{key}_empty_cp10k"] = empty["mean_cp10k"]
            metrics[key] = {
                "requestedRows": 4,
                "directRows": len(direct),
                "maximumAbsoluteLog1pDifference": float(difference.max()),
                "emptyMaximumAbsoluteCp10kDifference": float(empty_difference.max()),
            }
            if len(direct) != 4 or difference.max() > 1e-6 or empty_difference.max() > 1e-3:
                raise RuntimeError(f"corrected GPU replay failed: {key}")
    path = artifact / "target-free-gpu-probe.npz"
    np.savez_compressed(path, **arrays)
    return metrics, {"path": path.name, "sha256": sha256(path)}


def k562_reconstruction(artifact, models, panels, core):
    """Authorized append-only four-draw K562 reconstruction-held diagnostic."""
    base = load_module(K562_BASE, "shared_context_k562_reconstruction")
    panel = panels["k562"]
    registry = json.loads(runner_module().REGISTRY.read_text(encoding="utf-8"))
    with np.load(
        runner_module().REGISTRY.parent / registry["static"]["path"],
        allow_pickle=False,
    ) as static:
        static_values = {name: np.asarray(static[name]) for name in static.files}
    resources = {
        "registered": {
            "query_features": panel.query_features,
            "basal_rate": panel.basal_rate,
            "basal_observed": np.ones(panel.basal_rate.shape, np.bool_),
            "gem_group_ids": np.asarray(
                np.load(runner_module().BASELINES["k562"], allow_pickle=False)["gem_group"]
            ),
        },
        "static": static_values,
    }
    return {
        arm: base.evaluate_fitting_reconstruction(core, model, resources)
        for arm, model in models.items()
    }


_RUNNER_CACHE = None


def runner_module():
    global _RUNNER_CACHE
    if _RUNNER_CACHE is None:
        _RUNNER_CACHE = load_module(RUNNER, "shared_final_runner")
    return _RUNNER_CACHE


def finalize(artifact: Path = ARTIFACT):
    prepare(artifact)
    if (artifact / "FROZEN-FITTING-ONLY.json").exists():
        raise FileExistsError("append-only finalization already complete")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for exact retained-model replay")
    runner = runner_module()
    panels, _ = runner.load_panels()
    models, _ = load_models(artifact, runner, panels)
    gpu_metrics, probe = corrected_gpu_probe(artifact, runner, models, panels)
    process = subprocess.run(
        [sys.executable, str(artifact / "source/runner.py"), "verify", "--output", str(artifact)],
        capture_output=True,
        check=False,
        text=True,
        timeout=300,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
    )
    if process.returncode:
        raise RuntimeError(f"isolated CPU replay failed: {process.stderr[-2000:]}")
    cpu = json.loads(process.stdout.strip().splitlines()[-1])
    if not cpu["passes"]:
        raise RuntimeError("isolated CPU replay exceeded tolerance")
    write_json(artifact / "isolated-cpu-verification.json", cpu)
    metrics, per_gene = runner.fitting_diagnostics(models, panels, torch.device("cuda"), artifact)
    write_json(artifact / "fitting-diagnostics.json", metrics)
    forecast = runner.freeze_development_forecasts(artifact, panels)
    freeze = {
        "schema": "slp.human-essential-count-shared-context-fitting-freeze/v1",
        "originalProtocolSha256": sha256(artifact / "protocol.json"),
        "originalArtifactManifestSha256": sha256(artifact / "artifact-manifest.json"),
        "finalizationProtocolSha256": sha256(artifact / "FINALIZATION-PROTOCOL-V2.json"),
        "modelSha256": {arm: EXPECTED[arm] for arm in runner.ARMS},
        "targetFreeGpuProbe": probe,
        "isolatedCpuVerificationSha256": sha256(artifact / "isolated-cpu-verification.json"),
        "fittingDiagnosticsSha256": sha256(artifact / "fitting-diagnostics.json"),
        "fittingPerGeneDiagnostics": per_gene,
        "developmentForecastFreezeSha256": sha256(
            artifact / "FORECASTS-FROZEN-BEFORE-DEVELOPMENT.json"
        ),
        "developmentCountMembersOpened": False,
        "reconstructionHeldOpened": False,
        "testOpened": False,
    }
    write_json(artifact / "FROZEN-FITTING-ONLY.json", freeze)
    report = {
        "schema": "slp.human-essential-count-shared-context-fitting-finalization/v1",
        "freezeSha256": sha256(artifact / "FROZEN-FITTING-ONLY.json"),
        "gpuReplay": gpu_metrics,
        "cpuReplay": cpu,
        "fittingDiagnostics": metrics,
        "developmentForecastFreeze": forecast,
        "originalFailurePreserved": "four-row expected output compared with unbounded 16-row direct chunk after both models were saved",
        "modelsRetrainedOrChanged": False,
        "developmentOpened": False,
        "reconstructionHeldOpened": False,
        "testOpened": False,
    }
    write_json(artifact / "fitting-finalization-report.json", report)
    print(json.dumps({"freeze": freeze, "fittingDiagnostics": metrics}))
    return report


def reconstruction(artifact: Path = ARTIFACT):
    if not (artifact / "FROZEN-FITTING-ONLY.json").is_file():
        raise ValueError("forecast/fitting freeze must precede held reconstruction access")
    runner = runner_module()
    panels, _ = runner.load_panels()
    models, core = load_models(artifact, runner, panels)
    values = k562_reconstruction(artifact, models, panels, core)
    path = artifact / "k562-reconstruction-held-diagnostic.json"
    write_json(path, values)
    receipt = {
        "schema": "slp.human-essential-count-shared-context-k562-reconstruction-held/v1",
        "authorizedAfterFittingFreeze": True,
        "fittingFreezeSha256": sha256(artifact / "FROZEN-FITTING-ONLY.json"),
        "modelSha256": {arm: EXPECTED[arm] for arm in runner.ARMS},
        "diagnosticSha256": sha256(path),
        "developmentCountMembersOpened": False,
        "testOpened": False,
    }
    write_json(artifact / "K562-RECONSTRUCTION-HELD-RECEIPT.json", receipt)
    print(json.dumps(receipt))
    return receipt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "finalize", "reconstruction"))
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(2):
        {"prepare": prepare, "finalize": finalize, "reconstruction": reconstruction}[
            args.mode
        ](args.artifact)
