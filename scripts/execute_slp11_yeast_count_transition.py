#!/usr/bin/env python3
"""GPU profile and fixed execution for the prepared yeast count transition."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from safetensors.torch import save_file

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "MODEL_CARD.md").exists():
            return parent
    raise RuntimeError("cannot locate project root")


ROOT = _project_root()
OUTPUT = ROOT / "results/slp11-transition/yeast-rna-world-transition-seed731-v1"
FITTING = (
    ROOT / "data/derived/slp11-yeast-atlas-counts/nadal-ribelles-rna-neural-fitting-v1"
)
MOMENTS = (
    ROOT
    / "data/derived/slp11-yeast-atlas-counts/nadal-ribelles-raw-rna-development-v1/moments-manifest.json"
)
STATIC = (
    ROOT
    / "data/derived/slp11-yeast-shared-static/current-sgd-strict-query-full-raw-actions-esm8m-complete-shared-go-v2/yeast-static-esm8m-shared-go-mf-cc-features.npz"
)
BASELINE = ROOT / "results/slp11-transition/yeast-raw-count-batch-ridge-v1"
BASELINE_SCORING = (
    ROOT
    / "results/slp11-transition/yeast-raw-count-batch-ridge-roundoff-scoring-v1/report.json"
)
PINS = {
    "protocol": "5198af9e04828a67081b2fdda5bae1b3e03747f9a677990acd5d9c96b754b89a",
    "fittingManifest": "9962374c84e29a39074eaf493e7dab87af750c5d76314f5636e0db3d23555d15",
    "targets": "020d980d1384edbbe63fbe72789e104b3807955bb2d164f21472a0fadfb3a93d",
    "metadata": "c27f3e1671fbe6d1490ff3ff9e7582e601f8dfc96dd56f75236c8fc4411b8cfc",
    "reference": "09704a2cc40dd065cf2941e92e7a8eacad9d1cc7501ec99ee8f9201aebf1f194",
    "moments": "70a49ecaeb271fc72ecc93ede207c59a816e74d1ae3133bbf3a2803cce5d8eba",
    "static": "81cda9469380c9efa000a40b2cd5e816a1d397ce777288fa53b0bcf26a55dc25",
    "baseline": "291b8b34d9b03b0bedb6a40723cbab07b6fd2c094dbffdb5a5a644141454c128",
}
CONTEXTS = ("Control", "NaCl")
STEPS = 12_000
BATCH = 64
SEED = 731


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_python(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: object) -> None:
    def clean(item: object) -> object:
        if isinstance(item, dict):
            return {str(key): clean(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(entry) for entry in item]
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    path.write_text(
        json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def verify_prepared() -> dict[str, object]:
    checks = {
        OUTPUT / "protocol.json": PINS["protocol"],
        FITTING / "manifest.json": PINS["fittingManifest"],
        FITTING / "train-targets.npy": PINS["targets"],
        FITTING / "train-metadata.npz": PINS["metadata"],
        FITTING / "reference.npz": PINS["reference"],
        MOMENTS: PINS["moments"],
        STATIC: PINS["static"],
        BASELINE_SCORING: PINS["baseline"],
    }
    for path, expected in checks.items():
        if sha256(path) != expected:
            raise ValueError(f"prepared input drift: {path}")
    ready = json.loads((OUTPUT / "GPU-READY.json").read_text())
    if sha256(Path(__file__)) != ready["executorSha256"]:
        raise ValueError("executing source differs from reviewed executor")
    return ready


def batches(rows: int):
    generator = np.random.default_rng(SEED)
    emitted = 0
    while emitted < STEPS:
        order = generator.permutation(rows)
        complete = rows - rows % BATCH
        for start in range(0, complete, BATCH):
            yield order[start : start + BATCH]
            emitted += 1
            if emitted == STEPS:
                return


def model_runtime(device: torch.device):
    core = load_python(
        OUTPUT / "source/control_transition_model.py", "yeast_execute_core"
    )
    with np.load(FITTING / "reference.npz", allow_pickle=False) as archive:
        ref = {name: archive[name] for name in archive.files}
    torch.manual_seed(SEED)
    model = core.MinimalControlTransition(
        core.Config(577, 577, hidden_dim=128, state_dim=128, dropout=0.2)
    ).to(device)
    runtime = {
        "action": torch.as_tensor(ref["action_features_normalized"], device=device),
        "query": torch.as_tensor(ref["query_features_normalized"], device=device),
        "control": torch.as_tensor(ref["control_mean"], device=device),
        "amplitude": torch.as_tensor(ref["delta_amplitude"], device=device),
        "scale": torch.as_tensor(ref["objective_query_scale"], device=device),
        "basal": torch.as_tensor(ref["basal_values_normalized"], device=device),
        "basal_mask": torch.as_tensor(
            ref["basal_mask"], dtype=torch.bool, device=device
        ),
        "basal_index": torch.as_tensor(
            ref["basal_query_indices"], dtype=torch.int64, device=device
        ),
        "batch_context": torch.as_tensor(
            ref["batch_context_index"], dtype=torch.int64, device=device
        ),
    }
    return core, model, runtime, ref


def forward(model, action_index, batch_index, runtime):
    context = runtime["batch_context"][batch_index]
    return model(
        runtime["action"][action_index],
        runtime["query"],
        runtime["control"][batch_index],
        runtime["amplitude"],
        runtime["scale"][context],
        runtime["query"][runtime["basal_index"]],
        runtime["basal"][batch_index],
        runtime["basal_mask"][batch_index],
    )


def profile(steps: int) -> dict[str, object]:
    verify_prepared()
    if (OUTPUT / "cuda-profile.json").exists():
        raise FileExistsError("refusing to overwrite CUDA profile")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no fallback")
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    _core, model, runtime, _ref = model_runtime(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    synthetic_targets = torch.zeros((38978, 6683), device=device)
    action = torch.arange(BATCH, device=device) % runtime["action"].shape[0]
    batch = torch.arange(BATCH, device=device) % runtime["control"].shape[0]
    context = runtime["batch_context"][batch]
    weight = torch.ones(BATCH, device=device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = forward(model, action, batch, runtime)["mean"]
        loss = (
            ((prediction - synthetic_targets[:BATCH]) / runtime["scale"][context]) ** 2
        ).mean(1)
        (loss * weight).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    result = {
        "schema": "slp.yeast-count-transition-cuda-profile/v1",
        "targetFreeSynthetic": True,
        "batch": BATCH,
        "queries": 6683,
        "steps": steps,
        "seconds": elapsed,
        "secondsPerStep": elapsed / steps,
        "projectedSecondsFor12000Updates": elapsed / steps * STEPS,
        "peakAllocatedBytes": torch.cuda.max_memory_allocated(),
        "peakReservedBytes": torch.cuda.max_memory_reserved(),
        "peakHostRssBytes": psutil.Process().memory_info().rss,
        "fitsTenGiB": torch.cuda.max_memory_reserved() < 10 * (1 << 30),
    }
    write_json(OUTPUT / "cuda-profile.json", result)
    write_json(
        OUTPUT / "PROFILE-FROZEN.json",
        {
            "profileSha256": sha256(OUTPUT / "cuda-profile.json"),
            "biologicalTargetsUsed": False,
        },
    )
    return result


def _row_corr(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    original_left, original_right = left, right
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    eps = np.finfo(np.float64).eps
    left_tol = (
        8
        * eps
        * np.sqrt(left.shape[1])
        * np.maximum(1.0, np.max(np.abs(original_left), axis=1))
    )
    right_tol = (
        8
        * eps
        * np.sqrt(right.shape[1])
        * np.maximum(1.0, np.max(np.abs(original_right), axis=1))
    )
    valid = (left_norm > left_tol) & (right_norm > right_tol)
    return np.divide(
        np.sum(left * right, axis=1),
        left_norm * right_norm,
        out=np.full(len(left), np.nan),
        where=valid,
    )


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    anchored_truth = truth - truth[:1]
    anchored_prediction = prediction - prediction[:1]
    centered_truth = anchored_truth - anchored_truth.mean(axis=0)
    centered_prediction = anchored_prediction - anchored_prediction.mean(axis=0)
    correlation = _row_corr(centered_truth, centered_prediction)
    ordinary = _row_corr(truth, prediction)
    return {
        "geneProfileMse": float(np.mean((truth - prediction) ** 2)),
        "independentlyQueryCenteredPearson": (
            float(np.nanmean(correlation)) if np.isfinite(correlation).any() else None
        ),
        "independentlyQueryCenteredUndefinedGenes": int(
            np.count_nonzero(~np.isfinite(correlation))
        ),
        "ordinaryGeneProfilePearson": float(np.nanmean(ordinary))
        if np.isfinite(ordinary).any()
        else None,
        "ordinaryUndefinedGenes": int(np.count_nonzero(~np.isfinite(ordinary))),
    }


def _load_batch_reference(context: str) -> tuple[np.ndarray, np.ndarray]:
    path = BASELINE / f"{context.lower()}-batchReference.npz"
    with np.load(path, allow_pickle=False) as archive:
        return archive["batch_ids"].astype(str), archive["batch_only_means"].astype(
            np.float64
        )


def evaluate(
    model, runtime, ref, device: torch.device
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    # This function is called only after FROZEN-BEFORE-VALIDATION exists.
    manifest = json.loads(MOMENTS.read_text())
    prepared_runner = load_python(
        OUTPUT / "source/trainer.py", "yeast_prepared_allowlist_reader"
    )
    with np.load(STATIC, allow_pickle=False) as archive:
        static_ids = archive["entity_id"].astype(str)
        static_values = archive["feature_values"].astype(np.float32)
    static_lookup = dict(zip(static_ids, static_values, strict=True))
    baseline = json.loads(BASELINE_SCORING.read_text())
    report: dict[str, object] = {}
    arrays: dict[str, np.ndarray] = {}
    model.eval()
    for context_index, context in enumerate(CONTEXTS):
        entries = [entry for entry in manifest["shards"] if entry["context"] == context]
        gene_ids = sorted(
            {
                str(gene)
                for entry in entries
                for gene, role in zip(
                    np.load(entry["path"])["group_action_id"],
                    np.load(entry["path"])["development_role"],
                    strict=True,
                )
                if role == "validation"
            }
        )
        gene_lookup = {gene: index for index, gene in enumerate(gene_ids)}
        truth_sum = np.zeros((len(gene_ids), 6683), dtype=np.float64)
        prediction_sum = np.zeros_like(truth_sum)
        reference_sum = np.zeros_like(truth_sum)
        total_cells = np.zeros(len(gene_ids), dtype=np.int64)
        reference_batches, reference_values = _load_batch_reference(context)
        reference_lookup = {
            batch: index for index, batch in enumerate(reference_batches)
        }
        for entry in entries:
            path = Path(entry["path"])
            with np.load(path, allow_pickle=False) as archive:
                actions = archive["group_action_id"].astype(str)
                roles = archive["development_role"].astype(str)
                cells = archive["num_cells"].astype(np.int64)
                batch_id = str(archive["batch_id"].item())
            selected = np.flatnonzero(roles == "validation")
            # Loading begins only here, after model/reference freeze.
            target_sum = np.asarray(
                prepared_runner.stored_npz_memmap(path, "sum")[selected],
                dtype=np.float64,
            )
            raw_actions = np.stack([static_lookup[gene] for gene in actions[selected]])
            normalized = (raw_actions - ref["feature_mean"]) / ref["feature_std"]
            batch_index = np.flatnonzero(
                (ref["batch_ids"] == batch_id)
                & (ref["batch_context_index"] == context_index)
            )
            if len(batch_index) != 1:
                raise ValueError("validation batch has no exact WT reference")
            predictions = []
            with torch.no_grad():
                for start in range(0, len(selected), 256):
                    stop = min(start + 256, len(selected))
                    local_action = torch.as_tensor(
                        normalized[start:stop], device=device
                    )
                    local_batch = torch.full(
                        (stop - start,),
                        int(batch_index[0]),
                        dtype=torch.int64,
                        device=device,
                    )
                    context_tensor = runtime["batch_context"][local_batch]
                    predictions.append(
                        model(
                            local_action,
                            runtime["query"],
                            runtime["control"][local_batch],
                            runtime["amplitude"],
                            runtime["scale"][context_tensor],
                            runtime["query"][runtime["basal_index"]],
                            runtime["basal"][local_batch],
                            runtime["basal_mask"][local_batch],
                        )["mean"]
                        .cpu()
                        .numpy()
                    )
            prediction = np.concatenate(predictions)
            batch_reference = reference_values[reference_lookup[batch_id]]
            for local, gene in enumerate(actions[selected]):
                index = gene_lookup[str(gene)]
                count = int(cells[selected[local]])
                truth_sum[index] += target_sum[local]
                prediction_sum[index] += count * prediction[local]
                reference_sum[index] += count * batch_reference
                total_cells[index] += count
        truth = truth_sum / total_cells[:, None]
        prediction = prediction_sum / total_cells[:, None]
        reference_profile = reference_sum / total_cells[:, None]
        truth_residual = truth - reference_profile
        prediction_residual = prediction - reference_profile
        raw_metrics = metrics(truth, prediction)
        residual_metrics = metrics(truth_residual, prediction_residual)
        comparators = baseline["validation"][context]["metrics"]
        mse_checks = {
            name: raw_metrics["geneProfileMse"]
            <= 0.98 * comparators[name]["raw"]["geneProfileMse"]
            for name in ("pooled", "batch", "pooledMean", "batchMean")
        }
        comparator_r = [
            comparators[name]["batchMeanSubtracted"][
                "independentlyQueryCenteredPearson"
            ]
            for name in ("pooled", "batch", "pooledMean", "batchMean")
            if comparators[name]["batchMeanSubtracted"][
                "independentlyQueryCenteredPearson"
            ]
            is not None
        ]
        world_r = residual_metrics["independentlyQueryCenteredPearson"]
        r_checks = {
            "finite": world_r is not None and np.isfinite(world_r),
            "atLeastPoint10": world_r is not None and world_r >= 0.1,
            "nonregressionEveryDefinedBaseline": world_r is not None
            and all(world_r >= value for value in comparator_r),
        }
        report[context] = {
            "genes": len(gene_ids),
            "raw": raw_metrics,
            "sameBatchReferenceResidual": residual_metrics,
            "negativePredictionFraction": float(np.mean(prediction < 0)),
            "minimumPrediction": float(prediction.min()),
            "mseChecks": mse_checks,
            "correlationChecks": r_checks,
            "passes": all(mse_checks.values()) and all(r_checks.values()),
        }
        prefix = context.lower()
        arrays[f"{prefix}_gene_ids"] = np.asarray(gene_ids)
        arrays[f"{prefix}_truth"] = truth.astype(np.float32)
        arrays[f"{prefix}_prediction"] = prediction.astype(np.float32)
        arrays[f"{prefix}_batch_reference"] = reference_profile.astype(np.float32)
    return report, arrays


def execute() -> dict[str, object]:
    verify_prepared()
    if (OUTPUT / "model.safetensors").exists() or (OUTPUT / "report.json").exists():
        raise FileExistsError("refusing to overwrite trained evidence")
    profile_report = json.loads((OUTPUT / "cuda-profile.json").read_text())
    profile_frozen = json.loads((OUTPUT / "PROFILE-FROZEN.json").read_text())
    if sha256(OUTPUT / "cuda-profile.json") != profile_frozen["profileSha256"]:
        raise ValueError("CUDA profile drift")
    if (
        not profile_report["fitsTenGiB"]
        or profile_report["projectedSecondsFor12000Updates"] >= 900
        or profile_report["peakHostRssBytes"] >= 6 * (1 << 30)
    ):
        raise RuntimeError("frozen CUDA resource rule failed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no fallback")
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    _core, model, runtime, ref = model_runtime(device)
    with np.load(FITTING / "train-metadata.npz", allow_pickle=False) as archive:
        action_index = archive["action_index"].astype(np.int64)
        batch_index = archive["batch_index"].astype(np.int64)
        row_weight = archive["row_weight"].astype(np.float32)
    target_array = np.load(FITTING / "train-targets.npy", mmap_mode="r")
    targets = torch.as_tensor(np.asarray(target_array), device=device)
    action_tensor = torch.as_tensor(action_index, dtype=torch.int64, device=device)
    batch_tensor = torch.as_tensor(batch_index, dtype=torch.int64, device=device)
    weights = torch.as_tensor(row_weight, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    losses: list[float] = []
    started = time.monotonic()
    model.train()
    for step, rows in enumerate(batches(len(action_index)), 1):
        row = torch.as_tensor(rows, dtype=torch.int64, device=device)
        local_batch = batch_tensor[row]
        context = runtime["batch_context"][local_batch]
        optimizer.zero_grad(set_to_none=True)
        prediction = forward(model, action_tensor[row], local_batch, runtime)["mean"]
        per_row = (((prediction - targets[row]) / runtime["scale"][context]) ** 2).mean(
            1
        )
        loss = (per_row * weights[row]).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite training objective at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 1000 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "recentLoss": float(np.mean(losses[-100:])),
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
        if step % 1000 == 0 and psutil.Process().memory_info().rss >= 6 * (1 << 30):
            raise MemoryError("training exceeded frozen 6 GiB host RSS cap")
        if time.monotonic() - started > 900:
            raise TimeoutError("training exceeded frozen 900 second cap")
    fit_seconds = time.monotonic() - started
    model_path = OUTPUT / "model.safetensors"
    save_file(
        {name: value.detach().cpu() for name, value in model.state_dict().items()},
        str(model_path),
    )
    shutil.copy2(FITTING / "reference.npz", OUTPUT / "reference.npz")
    freeze = {
        "schema": "slp.yeast-count-transition-freeze/v1",
        "modelSha256": sha256(model_path),
        "referenceSha256": sha256(OUTPUT / "reference.npz"),
        "updates": STEPS,
        "validationEvaluations": 0,
        "fitSeconds": fit_seconds,
    }
    write_json(OUTPUT / "FROZEN-BEFORE-VALIDATION.json", freeze)

    probe_rows = np.asarray([0, 1, len(action_index) - 2, len(action_index) - 1])
    raw_actions = (
        ref["action_features_normalized"][action_index[probe_rows]] * ref["feature_std"]
        + ref["feature_mean"]
    ).astype(np.float32)
    model.eval()
    with torch.no_grad():
        expected = (
            forward(
                model,
                action_tensor[torch.as_tensor(probe_rows, device=device)],
                batch_tensor[torch.as_tensor(probe_rows, device=device)],
                runtime,
            )["mean"]
            .cpu()
            .numpy()
        )
    np.savez(
        OUTPUT / "target-free-probe.npz",
        raw_action_features=raw_actions,
        batch_index=batch_index[probe_rows],
        expected_mean=expected,
    )
    artifact_hashes = {
        "model.safetensors": sha256(model_path),
        "reference.npz": sha256(OUTPUT / "reference.npz"),
        "target-free-probe.npz": sha256(OUTPUT / "target-free-probe.npz"),
        "source/control_transition_model.py": sha256(
            OUTPUT / "source/control_transition_model.py"
        ),
        "source/inference.py": sha256(OUTPUT / "source/inference.py"),
        "source/verify_artifact.py": sha256(OUTPUT / "source/verify_artifact.py"),
    }
    write_json(
        OUTPUT / "artifact-manifest.json",
        {"schema": "slp.yeast-count-transition-artifact/v1", "sha256": artifact_hashes},
    )
    verification = subprocess.run(
        [sys.executable, str(OUTPUT / "source/verify_artifact.py"), str(OUTPUT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    portable = json.loads(verification.stdout)
    validation, arrays = evaluate(model, runtime, ref, device)
    np.savez(OUTPUT / "validation-predictions.npz", **arrays)
    result = {
        "schema": "slp.yeast-count-transition-report/v1",
        "hypothesis": "Shared static-action and measured-WT-state transition beats every fixed static/mean baseline in both environments.",
        "protocolSha256": PINS["protocol"],
        "gpuReadySha256": sha256(OUTPUT / "GPU-READY.json"),
        "modelSha256": sha256(model_path),
        "referenceSha256": sha256(OUTPUT / "reference.npz"),
        "profile": profile_report,
        "fitSeconds": fit_seconds,
        "trainingFinalRecentLoss": float(np.mean(losses[-100:])),
        "portableVerification": portable,
        "validation": validation,
        "passesAllContexts": all(value["passes"] for value in validation.values()),
        "validationEvaluations": 1,
        "limitations": [
            "Development diagnostic only; no final/test/benchmark outcomes are accessed.",
            "Gaussian-mean approximation to aggregate RNA, not a single-cell generator.",
            "The shared nonlinear transition is not an identified dynamical or causal mechanism.",
            "Negative means are reported without after-hoc projection.",
        ],
    }
    write_json(OUTPUT / "report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=20)
    args = parser.parse_args()
    if args.profile == args.execute:
        parser.error("choose exactly one of --profile or --execute")
    result = profile(args.profile_steps) if args.profile else execute()
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            default=lambda value: value.item()
            if isinstance(value, np.generic)
            else value,
        )
    )


if __name__ == "__main__":
    main()
