#!/usr/bin/env python3
"""Matched joint count-world test of fitting-derived native query descriptors."""
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
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/slp11-transition/human-essential-count-response-query33-seed731-v2"
OLD_RUNNER = ROOT / "scripts/run_slp11_count_world_shared_context.py"
REGISTRY = ROOT / "data/derived/slp11-human-essential-joint-training-registry-v1/registry.json"
PANEL_DATA = ROOT / "modules/slp-1-1-count-panel-data-v1/panel_data.py"
FEATURE_ADAPTER = ROOT / "modules/slp-1-1-count-response-query-features-v1/response_query_features.py"
FEATURE_DIR = ROOT / "data/derived/slp11-human-count-response-query33/rank32-alpha1000-full-fitting-v3"
FEATURE_MANIFEST = FEATURE_DIR / "manifest.json"
FEATURE_PACK = {source: FEATURE_DIR / f"response-query33-{source}.npz" for source in ("k562", "rpe1")}
TRAINING_STEP = ROOT / "modules/slp-1-1-count-world-training-v1/training_step.py"
CORE = ROOT / "modules/slp-1-1-count-world-training-v1/count_latent_state.py"
OBJECTIVE = ROOT / "modules/slp-1-1-count-world-training-v1/molecular_mean_objective.py"
INFERENCE = ROOT / "modules/slp-1-1-count-world-response-query-inference-v1/inference.py"
K562_BASE = ROOT / "scripts/run_slp11_k562_count_latent_state.py"
OLD_FORECAST = ROOT / "results/slp11-transition/human-essential-count-shared-context-seed731-v1"
RANK_DIR = ROOT / "results/slp11-transition/human-essential-count-response-rank32-seed731-v1"
TRUTH_DIR = ROOT / "results/slp11-transition/human-essential-count-shared-context-development-evaluation-v2"

SEED = 731
AUX_SEED = 1731
POPULATION_SEED = 1732
ARMS = ("static-zero33", "response33")
MODEL_CONFIG = {"feature_dim": 610, "hidden_dim": 128, "state_dim": 32, "key_dim": 64, "dropout": 0.1}
TRAINING = {
    "countUpdates": 12_000,
    "meanAuxUpdates": 4_000,
    "batchSize": 128,
    "controlRows": 64,
    "targetRows": 64,
    "populationGenes": 16,
    "learningRate": 0.0005,
    "weightDecay": 0.01,
    "gradientClip": 1.0,
    "meanWeight": 0.1,
    "maxSecondsPerArm": 1500,
    "logEvery": 100,
}
EXPECTED = {
    REGISTRY: "4de798e53a4d8149c200088e054caa4c9b71ecea91e6c00c68ecd3a6c938127c",
    PANEL_DATA: "a8f1ee3537041d20e1dda330c20ec0f73b3265ac63024eb3114ec1161d072c66",
    FEATURE_ADAPTER: "6a47563e97fd4cd710788917b35e40e5895abb5da340d66b8e53f5189f141807",
    FEATURE_MANIFEST: "8af044cedd683364ca789dd083f6c815740b8111fd59aec0aa326fdb734a27ba",
    FEATURE_PACK["k562"]: "db5c56edf6f81921f4b13ce4dc3c3fed1058508deb593fac9a2b8f92241e24ba",
    FEATURE_PACK["rpe1"]: "06e0179fa0f937b3e86cb7bb0ea53a20d0fc66194945093f95d9b35f92f38f00",
    TRAINING_STEP: "da544a7b969ddda4f6f4b44c77a8327c8be394746e6ea81ab012e56cc03a4062",
    CORE: "75df347a82151074c0ce6f4c732106e70ed17126aff07d017294894421d30bac",
    OBJECTIVE: "f9dc1fc1d7c6f1071f5bdb98e45a5140116cb583975bf3a76892814883989cd9",
    OLD_RUNNER: "8fdf92ecf837eb72c4d107cfc1ed053eff071671172079fd2db001dcdbb10091",
    K562_BASE: "9d6668ceb61a3bb0b9dc540a42430b523632b86ddcf547ec2175bfb2fe155920",
    RANK_DIR / "development-forecast-k562.npz": "53d147035d3f04569ea7ca7a9956ebf9704f3cefe6ed30caeda0a6fa513b04b1",
    RANK_DIR / "development-forecast-rpe1.npz": "8f7819fc5e45744fe73ea971e2745404ddba1a86b015fa2354ea6fe692afe58c",
}
EVALUATION_EXPECTED = {
    TRUTH_DIR / "development-truth-k562.npz": "abe9fafc8df755e9a90f8e544ef1737ed799db7332cef864afd08ae4e1c99588",
    TRUTH_DIR / "development-truth-rpe1.npz": "a8b6df1dd24863a76ba8e7bac740110c81008ab65de91d7671ba59b708f08d93",
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


def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def protocol():
    pins = dict(EXPECTED)
    pins[INFERENCE] = sha256(INFERENCE)
    return {
        "schema": "slp.human-essential-count-response-query33-protocol/v1",
        "hypothesis": "Source-native fitting-derived rank32 query loadings plus residual intercept improve held-gene molecular forecasts relative to the exact-zero33 width control and the fixed rank32 feature-linear comparator.",
        "advancementRule": "In both K562 and RPE1, response33 must reduce gene-profile MSE at least 1% versus static-zero33 and fixed rank32, attain independently query-centered control-residual Pearson >=.10 and not regress that Pearson versus fixed rank32; K562 reconstruction-held four-antithetic-draw ELBO must be no more than 1% worse than static-zero33.",
        "accessibleModalities": "Native fitting/control counts, static577 ESM8M/GO, source-native fitting-outcome-derived rank32 query loading and intercept. The response coordinates are not static priors; action response coordinates are exact zeros.",
        "modelConfig": MODEL_CONFIG,
        "training": {**TRAINING, "initializationSeed": SEED, "countSamplingSeed": SEED, "meanAuxCountSamplingSeed": AUX_SEED, "meanAuxPopulationSeed": POPULATION_SEED, "optimizerResetAtPhaseBoundary": True, "sourceSchedule": "odd K562/even RPE1 in both arms and phases", "finalCheckpointOnly": True, "earlyStopping": False},
        "featureArms": {"static-zero33": "static577 query/action plus exact zero33", "response33": "static577 query plus native normalized response33; static577 action plus exact zero33"},
        "scope": "Measured native panels only; no unmeasured-query, new-context, count-generation, dynamics, test, or benchmark claim.",
        "pins": {str(path.relative_to(ROOT)): value for path, value in pins.items()},
        "postForecastEvaluationPins": {
            str(path.relative_to(ROOT)): value
            for path, value in EVALUATION_EXPECTED.items()
        },
        "preFreezeDevelopmentValueAccess": "No development NPZ member is decoded. Externally supplied truth-artifact checksums are recorded now and verified only after both arm forecasts freeze.",
        "runnerSha256": sha256(Path(__file__).resolve()),
        "developmentOpenedAtProtocolFreeze": False,
        "testOpened": False,
    }


def prepare(output: Path = OUTPUT):
    for path, expected in EXPECTED.items():
        if sha256(path) != expected:
            raise ValueError(f"frozen input changed: {path}")
    output.mkdir(parents=True, exist_ok=True)
    value = protocol()
    path = output / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("frozen response-query protocol changed")
    else:
        write_json(path, value)
    return value


def load_panels_by_arm():
    panel_module = load_module(PANEL_DATA, "response_query_panel_data")
    adapter = load_module(FEATURE_ADAPTER, "response_query_feature_adapter")
    original = panel_module.load_panels(REGISTRY, ROOT)
    packs = {source: load_npz(FEATURE_PACK[source]) for source in original}
    result = {
        arm: {source: adapter.augment_panel(panel, packs[source], arm) for source, panel in original.items()}
        for arm in ARMS
    }
    for source in original:
        if not np.all(result["static-zero33"][source].query_features[:, 577:] == 0):
            raise AssertionError("control query padding is nonzero")
        if not np.all(result["response33"][source].gene_action_features[:, 577:] == 0):
            raise AssertionError("response-arm action padding is nonzero")
        for arm in ARMS:
            panel = result[arm][source]
            if panel.counts is not original[source].counts or panel.population_targets is not original[source].population_targets:
                raise AssertionError("feature adapter changed quantitative panel identity")
    return result, original


def initialize(core, device):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    return core.CountLatentState(core.Config(**MODEL_CONFIG)).to(device)


def panel_tensors(panels, device):
    return {
        source: {
            "query": torch.as_tensor(panel.query_features, device=device),
            "basal": torch.as_tensor(panel.basal_rate, device=device),
            "mask": torch.ones(panel.basal_rate.shape, dtype=torch.bool, device=device),
            "observed": torch.ones((TRAINING["batchSize"], len(panel.query_ids)), dtype=torch.bool, device=device),
        }
        for source, panel in panels.items()
    }


def train_arm(core, step, base, panels, arm, device):
    model = initialize(core, device).train()
    initial = base.parameter_norms(model)
    tensors = panel_tensors(panels, device)
    history, source_counts, traces = [], defaultdict(int), {}
    started = time.perf_counter()
    phases = (("count", TRAINING["countUpdates"], SEED, False), ("meanAux", TRAINING["meanAuxUpdates"], AUX_SEED, True))
    for phase, updates, cell_seed, use_mean in phases:
        optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING["learningRate"], weight_decay=TRAINING["weightDecay"])
        cell_rng = {source: np.random.default_rng(cell_seed) for source in panels}
        pop_rng = {source: np.random.default_rng(POPULATION_SEED) for source in panels}
        row_trace, pop_trace, window = hashlib.sha256(), hashlib.sha256(), []
        for update in range(1, updates + 1):
            if time.perf_counter() - started > TRAINING["maxSecondsPerArm"]:
                raise TimeoutError(f"{arm} exceeded frozen wall cap")
            source = "k562" if update % 2 else "rpe1"
            panel, local = panels[source], tensors[source]
            cells, rows = base.as_cell_batch(step, panel, local, cell_rng[source], device)
            row_trace.update(source.encode() + np.asarray(rows, dtype="<i8").tobytes())
            populations = None
            if use_mean:
                populations, selected = base.as_population_batch(step, panel, pop_rng[source], device)
                pop_trace.update(source.encode() + np.asarray(selected, dtype="<i8").tobytes())
            optimizer.zero_grad(set_to_none=True)
            result = step.training_losses(model, local["query"], local["basal"], local["mask"], cells, populations, mean_weight=TRAINING["meanWeight"] if use_mean else 0.0, fitting_mean_scale=panel.fitting_mean_scale if use_mean else None)
            result["loss"].backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING["gradientClip"])
            if not torch.isfinite(gradient):
                raise FloatingPointError(f"nonfinite gradient {arm}/{phase}/{update}")
            optimizer.step()
            source_counts[f"{phase}:{source}"] += 1
            count = result["count_result"]
            window.append({"loss": float(result["loss"].detach()), "countElbo": float(result["count_elbo"].detach()), "reconstructionPerQuery": float(count["reconstruction_per_query"].mean().detach()), "klPerCell": float(count["kl_per_cell"].mean().detach()), "meanMse": float(result["mean_mse"].detach()), "gradientNormBeforeClip": float(gradient.detach())})
            if update % TRAINING["logEvery"] == 0:
                item = {"phase": phase, "update": update, "source": source, "elapsedSeconds": time.perf_counter() - started, **{key: float(np.mean([entry[key] for entry in window])) for key in window[0]}}
                history.append(item)
                if update % 1000 == 0:
                    print(json.dumps({"event": "training", "arm": arm, **item}), flush=True)
                window.clear()
        traces[f"{phase}CountRowsSha256"] = row_trace.hexdigest()
        traces[f"{phase}PopulationRowsSha256"] = pop_trace.hexdigest()
    final = base.parameter_norms(model)
    return model.eval(), {"seconds": time.perf_counter() - started, "sourceUpdates": dict(source_counts), "initialParameterNorms": initial, "finalParameterNorms": final, "parameterNormChanges": {name: final[name] - initial[name] for name in initial}, "traces": traces, "history": history, "finite": True, "completedUpdates": 16_000}


@torch.no_grad()
def population_prediction(model, panel, device, *, limit=None, chunk=16):
    query = torch.as_tensor(panel.query_features, device=device)
    basal = torch.as_tensor(panel.basal_rate, device=device)
    context = model.encode_context(query, basal, torch.ones_like(basal, dtype=torch.bool))
    stop = len(panel.gene_ids) if limit is None else min(int(limit), len(panel.gene_ids))
    groups, result = len(panel.context_ids), []
    for left in range(0, stop, chunk):
        right = min(left + chunk, stop)
        action = torch.as_tensor(panel.gene_action_features[left:right, None], device=device)
        count = len(action)
        expanded = action[:, None].expand(-1, groups, -1, -1).reshape(count * groups, 1, -1)
        prior = model.prior_from_context(expanded, torch.ones((count * groups, 1), dtype=torch.bool, device=device), context.repeat(count, 1))
        rate = model.population_mean(prior, query, basal[None].expand(count, -1, -1).reshape(count * groups, -1)).reshape(count, groups, -1)
        weights = torch.as_tensor(panel.population_context_weights[left:right], device=device)
        result.append(torch.log1p((rate * weights[..., None]).sum(1)).cpu().numpy())
    return np.concatenate(result) if result else np.empty((0, len(panel.query_ids)))


def action_normalizer():
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    static = load_npz(REGISTRY.parent / registry["static"]["path"])
    mean = np.concatenate((np.asarray(static["feature_mean"], np.float64), np.zeros(33)))
    scale = np.concatenate((np.asarray(static["feature_scale"], np.float64), np.ones(33)))
    return static, mean, scale, np.asarray(static["feature_clip"], np.float32)


def save_artifact(output, models, histories, panels_by_arm):
    (output / "arms").mkdir(exist_ok=True)
    (output / "source").mkdir(exist_ok=True)
    static, mean, scale, clip = action_normalizer()
    probe = {}
    lookup = {value: row for row, value in enumerate(static["entity_id"].astype(str))}
    for arm, model in models.items():
        save_file(model.state_dict(), str(output / "arms" / f"{arm}.safetensors"))
        write_json(output / "arms" / f"{arm}-training.json", histories[arm])
        for source, panel in panels_by_arm[arm].items():
            np.savez_compressed(output / f"reference-{arm}-{source}.npz", schema=np.asarray("slp.count-world-response-query-reference/v1"), source_id=np.asarray(source), feature_mode=np.asarray(arm), query_ids=panel.query_ids, context_ids=panel.context_ids, query_features=panel.query_features, basal_rate=panel.basal_rate, feature_mean=mean, feature_scale=scale, feature_clip=clip)
            if arm == ARMS[0]:
                raw = np.asarray(static["feature_values"][[lookup[value] for value in panel.gene_ids[:4]]], np.float32)
                probe[f"{source}_raw_static577"] = raw
                probe[f"{source}_context_weights"] = panel.population_context_weights[:4]
    np.savez_compressed(output / "target-free-probe-inputs.npz", **probe)
    copies = ((CORE, "count_latent_state.py"), (INFERENCE, "inference.py"), (TRAINING_STEP, "training_step.py"), (OBJECTIVE, "molecular_mean_objective.py"), (FEATURE_ADAPTER, "response_query_features.py"), (PANEL_DATA, "panel_data.py"), (Path(__file__).resolve(), "runner.py"))
    for source, name in copies:
        shutil.copyfile(source, output / "source" / name)
    files = {str(path.relative_to(output)).replace("\\", "/"): sha256(path) for path in output.rglob("*") if path.is_file() and path.name not in {"protocol.json", "artifact-manifest.json"}}
    manifest = {"schema": "slp.human-essential-count-response-query33-artifact/v1", "protocolSha256": sha256(output / "protocol.json"), "arms": {arm: {"modelPath": f"arms/{arm}.safetensors", "panels": {source: {"referencePath": f"reference-{arm}-{source}.npz"} for source in ("k562", "rpe1")}} for arm in ARMS}, "sha256": files, "developmentOpened": False, "reconstructionHeldOpened": False, "testOpened": False}
    write_json(output / "artifact-manifest.json", manifest)
    return manifest


def portable_probe(output, models, panels_by_arm, device):
    inference = load_module(output / "source/inference.py", "response_query_artifact_inference")
    inputs = load_npz(output / "target-free-probe-inputs.npz")
    arrays, metrics = {}, {}
    for arm in ARMS:
        for source, panel in panels_by_arm[arm].items():
            raw = inputs[f"{source}_raw_static577"]
            weights = inputs[f"{source}_context_weights"]
            actual = inference.Predictor(output, arm, source, device=str(device)).predict(raw, weights)
            direct = population_prediction(models[arm], panel, device, limit=4)
            difference = float(np.max(np.abs(actual["mean_log1p_cp10k"].astype(np.float64) - direct)))
            empty = inference.Predictor(output, arm, source, device=str(device)).predict(raw, weights, action_mask=np.zeros((4, 1), np.bool_))
            expected_empty = weights.astype(np.float64) @ panel.basal_rate.astype(np.float64)
            empty_difference = float(np.max(np.abs(empty["mean_cp10k"] - expected_empty)))
            key = f"{arm}_{source}"
            arrays[f"{key}_mean"] = actual["mean_log1p_cp10k"]
            metrics[key] = {"maximumAbsoluteLog1pDifference": difference, "emptyMaximumAbsoluteCp10kDifference": empty_difference}
            if difference > 1e-6 or empty_difference > 1e-3:
                raise RuntimeError(f"portable replay failed: {key}")
    np.savez_compressed(output / "target-free-gpu-probe.npz", **arrays)
    return metrics


def isolated_cpu_probe(output):
    code = """import importlib.util,json,numpy as np,sys
from pathlib import Path
r=Path(sys.argv[1]); s=importlib.util.spec_from_file_location('isolated',r/'source/inference.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
z=np.load(r/'target-free-probe-inputs.npz'); e=np.load(r/'target-free-gpu-probe.npz'); out={}
for arm in ('static-zero33','response33'):
 for source in ('k562','rpe1'):
  p=m.Predictor(r,arm,source,device='cpu'); a=p.predict(z[f'{source}_raw_static577'],z[f'{source}_context_weights'])['mean_log1p_cp10k']; out[f'{arm}_{source}']=float(np.max(np.abs(a-e[f'{arm}_{source}_mean'])))
print(json.dumps(out))"""
    process = subprocess.run([sys.executable, "-c", code, str(output.resolve())], cwd=os.environ.get("TEMP"), env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONPATH": ""}, capture_output=True, text=True, timeout=300, check=False)
    if process.returncode:
        raise RuntimeError(process.stderr[-2000:])
    result = json.loads(process.stdout)
    if max(result.values()) > 1e-6:
        raise RuntimeError("isolated CPU replay differs")
    return result


def fitting_diagnostics(output, models, panels_by_arm, base, device):
    arrays, metrics = {}, {source: {} for source in ("k562", "rpe1")}
    for source, source_metrics in metrics.items():
        panel0 = panels_by_arm[ARMS[0]][source]
        anchor = np.log1p(panel0.population_context_weights.astype(np.float64) @ panel0.basal_rate.astype(np.float64))
        arrays[f"{source}_gene_ids"] = panel0.gene_ids
        for arm in ARMS:
            panel = panels_by_arm[arm][source]
            prediction = population_prediction(models[arm], panel, device)
            summary, mse, corr = base.stable_profile_metrics(prediction, panel.population_targets, anchor)
            source_metrics[arm] = summary
            arrays[f"{source}_{arm}_mse"] = mse
            arrays[f"{source}_{arm}_pearson"] = corr
    path = output / "fitting-per-gene-diagnostics.npz"
    np.savez_compressed(path, **arrays)
    write_json(output / "fitting-diagnostics.json", metrics)
    return metrics, sha256(path)


def validation_metadata(base, source, panel):
    return base.validation_metadata(source, panel)


def freeze_forecasts(output, models, panels_by_arm, original, base, device):
    inference = load_module(output / "source/inference.py", "response_query_forecast_inference")
    receipts = {}
    for source, original_panel in original.items():
        metadata = validation_metadata(base, source, original_panel)
        old = load_npz(OLD_FORECAST / f"development-forecasts-{source}.npz")
        rank = load_npz(RANK_DIR / f"development-forecast-{source}.npz")
        identity = {**metadata, "query_ids": original_panel.query_ids}
        for key in ("gene_ids", "query_ids", "gem_group_ids", "cell_count", "gem_cell_count"):
            if not np.array_equal(old[key], rank[key]) or not np.array_equal(old[key], identity[key]):
                raise ValueError(f"frozen forecast identity differs: {source}/{key}")
        predictions = {arm: inference.Predictor(output, arm, source, device=str(device)).predict(metadata["raw_action_features"], metadata["gem_weights"])["mean_log1p_cp10k"] for arm in ARMS}
        path = output / f"development-forecast-{source}.npz"
        np.savez_compressed(path, schema=np.asarray("slp.human-essential-count-response-query33-development-forecast/v1"), source_id=np.asarray(source), context_id=np.asarray(metadata["context_id"]), gene_ids=metadata["gene_ids"], query_ids=original_panel.query_ids, gem_group_ids=metadata["gem_group_ids"], cell_count=metadata["cell_count"], gem_cell_count=metadata["gem_cell_count"], control_prediction=old["control_prediction"], anchored_mean_prediction=old["anchored_mean_prediction"], static_ridge_prediction=old["static_ridge_prediction"], rank32_prediction=rank["rank32_prediction"], static_zero33_prediction=predictions["static-zero33"], response33_prediction=predictions["response33"])
        receipts[source] = {"path": path.name, "sha256": sha256(path), "genes": len(metadata["gene_ids"]), "queries": len(original_panel.query_ids)}
    manifest = json.loads((output / "artifact-manifest.json").read_text(encoding="utf-8"))
    freeze = {"schema": "slp.human-essential-count-response-query33-forecast-freeze/v1", "forecastsFrozenBeforeDevelopmentCountAccess": True, "forecasts": receipts, "models": {arm: {"path": manifest["arms"][arm]["modelPath"], "sha256": manifest["sha256"][manifest["arms"][arm]["modelPath"]]} for arm in ARMS}, "armSpecificReferences": {arm: {source: {"path": manifest["arms"][arm]["panels"][source]["referencePath"], "sha256": manifest["sha256"][manifest["arms"][arm]["panels"][source]["referencePath"]]} for source in original} for arm in ARMS}, "developmentCountMembersOpened": False, "testOpened": False}
    write_json(output / "FORECASTS-FROZEN-BEFORE-DEVELOPMENT.json", freeze)
    return freeze


def profile(output: Path = OUTPUT):
    prepare(output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    panels_by_arm, _ = load_panels_by_arm()
    core = load_module(CORE, "response_query_profile_core")
    step = load_module(TRAINING_STEP, "response_query_profile_step")
    base = load_module(OLD_RUNNER, "response_query_profile_base")
    device = torch.device("cuda")
    times = {}
    torch.cuda.reset_peak_memory_stats()
    for arm in ARMS:
        model = initialize(core, device).train()
        tensors = panel_tensors(panels_by_arm[arm], device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING["learningRate"], weight_decay=TRAINING["weightDecay"])
        samples = []
        for source in ("k562", "rpe1"):
            cell_rng, pop_rng = np.random.default_rng(SEED), np.random.default_rng(POPULATION_SEED)
            for repeat in range(4):
                cells, _ = base.as_cell_batch(step, panels_by_arm[arm][source], tensors[source], cell_rng, device)
                populations, _ = base.as_population_batch(step, panels_by_arm[arm][source], pop_rng, device)
                optimizer.zero_grad(set_to_none=True)
                started = time.perf_counter()
                result = step.training_losses(model, tensors[source]["query"], tensors[source]["basal"], tensors[source]["mask"], cells, populations, mean_weight=.1, fitting_mean_scale=panels_by_arm[arm][source].fitting_mean_scale)
                result["loss"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step(); torch.cuda.synchronize()
                if repeat:
                    samples.append(time.perf_counter() - started)
        times[arm] = float(np.mean(samples))
    projected = {arm: times[arm] * 16_000 for arm in ARMS}
    report = {"schema": "slp.human-essential-count-response-query33-cuda-profile/v1", "secondsPerStep": times, "projectedSecondsPerArm": projected, "peakAllocatedBytes": int(torch.cuda.max_memory_allocated()), "peakReservedBytes": int(torch.cuda.max_memory_reserved()), "fitsCaps": bool(max(projected.values()) < 1500 and torch.cuda.max_memory_reserved() < 10 * 2**30), "developmentOpened": False, "testOpened": False}
    write_json(output / "cuda-profile.json", report); print(json.dumps(report)); return report


def preflight(output: Path = OUTPUT):
    """Exercise the complete artifact/reference/reload path with untrained models."""
    prepare(output)
    destination = output / "preflight-untrained"
    if (destination / "COMPLETE.json").exists():
        raise FileExistsError("immutable untrained artifact preflight already complete")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output / "protocol.json", destination / "protocol.json")
    panels_by_arm, _ = load_panels_by_arm()
    core = load_module(CORE, "response_query_preflight_core")
    device = torch.device("cuda")
    models = {arm: initialize(core, device).eval() for arm in ARMS}
    histories = {
        arm: {
            "preflightUntrained": True,
            "completedUpdates": 0,
            "sourceUpdates": {},
        }
        for arm in ARMS
    }
    save_artifact(destination, models, histories, panels_by_arm)
    gpu = portable_probe(destination, models, panels_by_arm, device)
    cpu = isolated_cpu_probe(destination)
    receipt = {
        "schema": "slp.human-essential-count-response-query33-untrained-preflight/v1",
        "preflightUntrained": True,
        "trainingUse": False,
        "models": {
            arm: sha256(destination / "arms" / f"{arm}.safetensors")
            for arm in ARMS
        },
        "artifactManifestSha256": sha256(destination / "artifact-manifest.json"),
        "gpuReplay": gpu,
        "isolatedCpuReplay": cpu,
        "developmentOpened": False,
        "reconstructionHeldOpened": False,
        "testOpened": False,
    }
    write_json(destination / "COMPLETE.json", receipt)
    print(json.dumps(receipt))
    return receipt


def run(output: Path = OUTPUT):
    prepare(output)
    if (output / "FROZEN-FITTING-ONLY.json").exists():
        raise FileExistsError("immutable fitting run already complete")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    profile_report = json.loads((output / "cuda-profile.json").read_text(encoding="utf-8"))
    if not profile_report["fitsCaps"]:
        raise RuntimeError("profile exceeds frozen caps")
    panels_by_arm, original = load_panels_by_arm()
    core = load_module(CORE, "response_query_train_core")
    step = load_module(TRAINING_STEP, "response_query_train_step")
    base = load_module(OLD_RUNNER, "response_query_train_base")
    device = torch.device("cuda")
    models, histories = {}, {}
    for arm in ARMS:
        models[arm], histories[arm] = train_arm(core, step, base, panels_by_arm[arm], arm, device)
    save_artifact(output, models, histories, panels_by_arm)
    gpu = portable_probe(output, models, panels_by_arm, device)
    cpu = isolated_cpu_probe(output)
    fitting, per_gene_sha = fitting_diagnostics(output, models, panels_by_arm, base, device)
    freeze_forecasts(output, models, panels_by_arm, original, base, device)
    result = {"schema": "slp.human-essential-count-response-query33-fitting-freeze/v1", "protocolSha256": sha256(output / "protocol.json"), "artifactManifestSha256": sha256(output / "artifact-manifest.json"), "models": {arm: sha256(output / "arms" / f"{arm}.safetensors") for arm in ARMS}, "gpuReplay": gpu, "isolatedCpuReplay": cpu, "fittingMetrics": fitting, "fittingPerGeneSha256": per_gene_sha, "forecastFreezeSha256": sha256(output / "FORECASTS-FROZEN-BEFORE-DEVELOPMENT.json"), "developmentOpened": False, "reconstructionHeldOpened": False, "testOpened": False}
    write_json(output / "FROZEN-FITTING-ONLY.json", result); print(json.dumps(result)); return result


@torch.no_grad()
def reconstruction_k562(output, models, panels_by_arm):
    base = load_module(K562_BASE, "response_query_reconstruction_base")
    static, _, _, _ = action_normalizer()
    normalized = np.asarray(static["normalized_feature_values"], np.float32)
    lookup = {value: row for row, value in enumerate(static["entity_id"].astype(str))}
    device = torch.device("cuda")
    totals = {
        arm: {
            name: {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0, "cells": 0}
            for name in ("all", "control", "target")
        }
        for arm in ARMS
    }
    manifest = json.loads((base.RAW_CELL_DIR / "manifest.json").read_text(encoding="utf-8"))
    for arm in ARMS:
        rng = np.random.default_rng(SEED)
        model, panel = models[arm], panels_by_arm[arm]["k562"]
        query = torch.as_tensor(panel.query_features, device=device); basal = torch.as_tensor(panel.basal_rate, device=device)
        gem_lookup = {int(value): row for row, value in enumerate(load_npz(OLD_FORECAST / "development-forecasts-k562.npz")["gem_group_ids"])}
        for shard_entry in (item for item in manifest["shards"] if item["role"] == "reconstruction-held"):
            path = base.RAW_CELL_DIR / shard_entry["path"]
            if base.sha256(path) != shard_entry["sha256"]:
                raise ValueError("held reconstruction shard changed")
            shard = base._load_raw_shard(path)
            csr = base._csr_from_raw_shard(shard)
            for left in range(0, len(shard["source_row_index"]), 128):
                stop = min(left + 128, len(shard["source_row_index"])); n = stop - left
                eps_half = rng.standard_normal((2, n, 32)).astype(np.float32)
                eps = np.concatenate((eps_half, -eps_half))
                counts = csr[left:stop].toarray().astype(np.float32); library = np.asarray(shard["library_size"][left:stop], np.float32); actions = shard["action_ids"][left:stop].astype(str)
                indices = np.asarray([lookup[action] if action else -1 for action in actions]); active = indices >= 0
                action_values = np.zeros((len(actions), 1, 610), np.float32); action_values[active, 0, :577] = normalized[indices[active]]
                gem = np.asarray([gem_lookup[int(value)] for value in shard["gem_group"][left:stop]], np.int64); unique, inverse = np.unique(gem, return_inverse=True)
                context = model.encode_context(query, basal[torch.as_tensor(unique, device=device)], torch.ones((len(unique), len(panel.query_ids)), dtype=torch.bool, device=device))
                prior = model.prior_from_context(torch.as_tensor(action_values, device=device), torch.as_tensor(active[:, None], device=device), context[torch.as_tensor(inverse, device=device)])
                losses, reconstructions, final = [], [], None
                for epsilon in eps:
                    item = model.elbo(torch.as_tensor(counts, device=device), torch.ones_like(torch.as_tensor(counts, device=device), dtype=torch.bool), torch.as_tensor(library, device=device), query, basal[torch.as_tensor(gem, device=device)], prior, epsilon=torch.as_tensor(epsilon, device=device))
                    losses.append(item["loss_per_cell"].cpu().numpy())
                    reconstructions.append(item["reconstruction_per_query"].cpu().numpy())
                    final = item
                loss = np.mean(losses, axis=0)
                reconstruction = np.mean(reconstructions, axis=0)
                kl = final["kl_per_cell"].cpu().numpy()
                for name, selected in (("all", np.ones(len(actions), bool)), ("control", ~active), ("target", active)):
                    totals[arm][name]["loss"] += float(loss[selected].sum())
                    totals[arm][name]["reconstruction"] += float(reconstruction[selected].sum())
                    totals[arm][name]["kl"] += float(kl[selected].sum())
                    totals[arm][name]["cells"] += int(selected.sum())
        for value in totals[arm].values():
            cells = value["cells"]
            value["lossMean"] = value.pop("loss") / cells
            value["reconstructionMean"] = value.pop("reconstruction") / cells
            value["klMean"] = value.pop("kl") / cells
            value["klPerQueryMean"] = value["klMean"] / 8563
    return totals


def validate_internal_freeze(output: Path) -> None:
    """Validate every frozen model, reference and forecast before truth access."""
    frozen_path = output / "FROZEN-FITTING-ONLY.json"
    forecast_path = output / "FORECASTS-FROZEN-BEFORE-DEVELOPMENT.json"
    manifest_path = output / "artifact-manifest.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        frozen["protocolSha256"] != sha256(output / "protocol.json")
        or frozen["artifactManifestSha256"] != sha256(manifest_path)
        or frozen["forecastFreezeSha256"] != sha256(forecast_path)
        or forecast.get("forecastsFrozenBeforeDevelopmentCountAccess") is not True
        or forecast.get("developmentCountMembersOpened") is not False
    ):
        raise ValueError("fitting or forecast freeze receipt changed")
    for arm in ARMS:
        model_name = manifest["arms"][arm]["modelPath"]
        actual = sha256(output / model_name)
        if actual != manifest["sha256"][model_name] or actual != frozen["models"][arm]:
            raise ValueError(f"frozen model changed: {arm}")
        for source in ("k562", "rpe1"):
            reference_name = manifest["arms"][arm]["panels"][source]["referencePath"]
            if (
                sha256(output / reference_name) != manifest["sha256"][reference_name]
                or sha256(output / reference_name)
                != forecast["armSpecificReferences"][arm][source]["sha256"]
            ):
                raise ValueError(f"frozen reference changed: {arm}/{source}")
    for source in ("k562", "rpe1"):
        name = forecast["forecasts"][source]["path"]
        if sha256(output / name) != forecast["forecasts"][source]["sha256"]:
            raise ValueError(f"frozen development forecast changed: {source}")


def evaluate(output: Path = OUTPUT):
    if not (output / "FROZEN-FITTING-ONLY.json").exists():
        raise ValueError("fitting and forecasts must be frozen first")
    for path, expected in {**EXPECTED, **EVALUATION_EXPECTED}.items():
        if sha256(path) != expected:
            raise ValueError(f"frozen evaluation input changed: {path}")
    validate_internal_freeze(output)
    base = load_module(OLD_RUNNER, "response_query_eval_base")
    per_gene, metrics = {}, {}
    for source in ("k562", "rpe1"):
        forecast = load_npz(output / f"development-forecast-{source}.npz")
        truth = load_npz(TRUTH_DIR / f"development-truth-{source}.npz")
        for key in ("gene_ids", "query_ids", "gem_group_ids", "cell_count", "gem_cell_count"):
            if not np.array_equal(forecast[key], truth[key]):
                raise ValueError(f"truth identity mismatch: {source}/{key}")
        anchor = forecast["control_prediction"]
        metrics[source] = {}
        for name in ("anchored_mean_prediction", "static_ridge_prediction", "rank32_prediction", "static_zero33_prediction", "response33_prediction"):
            summary, mse, corr = base.stable_profile_metrics(forecast[name], truth["truth_log1p_mean_cp10k"], anchor)
            metrics[source][name] = summary; per_gene[f"{source}_{name}_mse"] = mse; per_gene[f"{source}_{name}_pearson"] = corr
        per_gene[f"{source}_gene_ids"] = forecast["gene_ids"]
    np.savez_compressed(output / "development-per-gene-metrics.npz", **per_gene)
    panels_by_arm, _ = load_panels_by_arm(); core = load_module(CORE, "response_query_eval_core"); device = torch.device("cuda")
    manifest = json.loads((output / "artifact-manifest.json").read_text(encoding="utf-8")); models = {}
    for arm in ARMS:
        model = core.CountLatentState(core.Config(**MODEL_CONFIG)).to(device); model.load_state_dict(load_file(str(output / manifest["arms"][arm]["modelPath"]))); models[arm] = model.eval()
    reconstruction = reconstruction_k562(output, models, panels_by_arm)
    gate = all(metrics[source]["response33_prediction"]["geneProfileMse"] <= .99 * metrics[source][baseline]["geneProfileMse"] and metrics[source]["response33_prediction"]["independentlyQueryCenteredPearson"] is not None and metrics[source]["response33_prediction"]["independentlyQueryCenteredPearson"] >= .1 and metrics[source]["response33_prediction"]["independentlyQueryCenteredPearson"] >= metrics[source]["rank32_prediction"]["independentlyQueryCenteredPearson"] for source in metrics for baseline in ("static_zero33_prediction", "rank32_prediction")) and reconstruction["response33"]["all"]["lossMean"] <= 1.01 * reconstruction["static-zero33"]["all"]["lossMean"]
    report = {"schema": "slp.human-essential-count-response-query33-report/v1", "metrics": metrics, "k562ReconstructionHeld": reconstruction, "advancementPassed": bool(gate), "forecastFreezeSha256": sha256(output / "FORECASTS-FROZEN-BEFORE-DEVELOPMENT.json"), "perGeneSha256": sha256(output / "development-per-gene-metrics.npz"), "developmentTruthOpenedAfterForecastFreeze": True, "testOpened": False, "interpretation": "Measured-panel fitting-outcome-derived query conditioning; no unmeasured-query, new-context, count-generation, or dynamics claim."}
    write_json(output / "report.json", report); write_json(output / "COMPLETE.json", {"reportSha256": sha256(output / "report.json"), "advancementPassed": bool(gate)}); print(json.dumps(report)); return report


def parse_args():
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("prepare", "profile", "preflight", "run", "evaluate")); parser.add_argument("--output", type=Path, default=OUTPUT); return parser.parse_args()


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    args = parse_args()
    with threadpool_limits(limits=2):
        {
            "prepare": prepare,
            "profile": profile,
            "preflight": preflight,
            "run": run,
            "evaluate": evaluate,
        }[args.mode](args.output)
