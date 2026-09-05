#!/usr/bin/env python3
"""Fit two matched native-panel count-world arms without opening development.

The K562-only arm and alternating K562/RPE1 arm have identical initialization,
optimizer settings, update budgets, and shared static normalizer.  Native query
axes, controls, library denominators, and fitting mean scales remain separate.
"""
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
from safetensors.torch import save_file
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/derived/slp11-human-essential-joint-training-registry-v1/registry.json"
PANEL_DATA = ROOT / "modules/slp-1-1-count-panel-data-v1/panel_data.py"
TRAINING_STEP = ROOT / "modules/slp-1-1-count-world-training-v1/training_step.py"
CORE = ROOT / "modules/slp-1-1-count-world-training-v1/count_latent_state.py"
OBJECTIVE = ROOT / "modules/slp-1-1-count-world-training-v1/molecular_mean_objective.py"
INFERENCE = ROOT / "modules/slp-1-1-count-world-inference-v1/inference.py"
OUTPUT = ROOT / "results/slp11-transition/human-essential-count-shared-context-seed731-v1"
ROUTING = {
    "k562": ROOT / "data/derived/slp11-human-k562-essential-singlecell-metadata-v1/cell-routing-metadata.npz",
    "rpe1": ROOT / "data/derived/slp11-human-rpe1-essential-singlecell-metadata-v1/cell-routing-metadata.npz",
}
BASELINES = {
    "k562": ROOT / "results/slp11-transition/k562-essential-count-anchored-static-ridge-seed731-v1/model.npz",
    "rpe1": ROOT / "results/slp11-transition/rpe1-essential-count-anchored-static-ridge-seed731-v1/model.npz",
}
RIDGE_CORE = ROOT / "results/slp11-transition/k562-essential-count-anchored-static-ridge-seed731-v1/source/count_static_ridge.py"

SEED = 731
AUX_SEED = 1731
POPULATION_SEED = 1732
ARMS = ("k562-only", "joint-alternating")
MODEL_CONFIG = {
    "feature_dim": 577,
    "hidden_dim": 128,
    "state_dim": 32,
    "key_dim": 64,
    "dropout": 0.1,
}
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
    TRAINING_STEP: "da544a7b969ddda4f6f4b44c77a8327c8be394746e6ea81ab012e56cc03a4062",
    CORE: "75df347a82151074c0ce6f4c732106e70ed17126aff07d017294894421d30bac",
    OBJECTIVE: "f9dc1fc1d7c6f1071f5bdb98e45a5140116cb583975bf3a76892814883989cd9",
    PANEL_DATA: "a8f1ee3537041d20e1dda330c20ec0f73b3265ac63024eb3114ec1161d072c66",
    ROUTING["k562"]: "47c89c5082c0a9d4008c6b567407c530933a36fb7603621c37cbe913143f15ad",
    ROUTING["rpe1"]: "10f3d313a5671122bde10a9bd586e3a2808d6f9b554f737ddcbbc28becc5e2f2",
    BASELINES["k562"]: "dbb669d2eb8d844ec9be7c88a2ed21f5592de434d1b2e916412bda4a52fe1cf3",
    BASELINES["rpe1"]: "bd144e36b5618c6225828501492edfa5449cef07442041c1d1cc20645b1473bc",
    RIDGE_CORE: "1032eeff59382fae3874da9a389033192e113e0f5ac2c8d01f09f8441d969e62",
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
    path.write_text(
        json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def source_for_update(arm: str, update: int) -> str:
    """Return the predeclared source for a one-based optimizer update."""
    if update <= 0:
        raise ValueError("updates are one-based")
    if arm == "k562-only":
        return "k562"
    if arm == "joint-alternating":
        return "k562" if update % 2 else "rpe1"
    raise ValueError(f"unknown arm: {arm}")


def stable_profile_metrics(prediction, truth, anchor):
    """Equal-gene MSE and independently query-centered residual profile r."""
    pred = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    base = np.asarray(anchor, dtype=np.float64)
    if (
        pred.ndim != 2
        or pred.shape != target.shape
        or pred.shape != base.shape
        or not np.isfinite(pred).all()
        or not np.isfinite(target).all()
        or not np.isfinite(base).all()
    ):
        raise ValueError("finite aligned profile matrices required")
    p = pred - base
    t = target - base
    p = p - p[:1]
    t = t - t[:1]
    p = p - p.mean(0, keepdims=True)
    t = t - t.mean(0, keepdims=True)
    p -= p.mean(1, keepdims=True)
    t -= t.mean(1, keepdims=True)
    denominator = np.sqrt(np.sum(p * p, 1) * np.sum(t * t, 1))
    defined = denominator > 1e-12
    correlation = np.full(len(p), np.nan)
    correlation[defined] = np.sum(p[defined] * t[defined], 1) / denominator[defined]
    per_gene_mse = np.mean(np.square(pred - target), axis=1)
    return {
        "geneProfileMse": float(per_gene_mse.mean()),
        "independentlyQueryCenteredPearson": (
            float(correlation[defined].mean()) if defined.any() else None
        ),
        "definedGenes": int(defined.sum()),
        "undefinedGenes": int((~defined).sum()),
    }, per_gene_mse, correlation


def protocol() -> dict[str, object]:
    expected = dict(EXPECTED)
    expected[PANEL_DATA] = sha256(PANEL_DATA)
    expected[INFERENCE] = sha256(INFERENCE)
    return {
        "schema": "slp.human-essential-count-shared-context-protocol/v1",
        "hypothesis": "At an equal update budget, alternating a second source-native cell context improves unseen-gene molecular forecasts by learning a shared static-feature transition rather than a K562-specific transition.",
        "advancementRuleForLaterFrozenDevelopment": "The joint arm must reduce K562 MSE at least 1% versus the K562-only arm; in K562 and RPE1 it must reduce MSE at least 1% versus each frozen anchored mean and static ridge, have independently query-centered residual profile Pearson at least .10 and no lower than ridge; K562 reconstruction-held ELBO must be no more than 1% worse than K562-only.",
        "currentBoundary": "Fit, freeze portable artifacts, and compute fitting-only diagnostics. Do not open development, reconstruction-held, test, or benchmark counts until a separate root review.",
        "accessibleModalities": "Raw native-panel fitting/control counts, reconstruction-training NT controls, source-native query rosters, GEM context identity, and shared static577 ESM8M/GO features. Rejected guide and control-coexpression features are excluded.",
        "modelConfig": MODEL_CONFIG,
        "training": {
            **TRAINING,
            "initializationSeed": SEED,
            "countPhaseSamplingSeedPerSource": SEED,
            "meanAuxPhaseCountSamplingSeedPerSource": AUX_SEED,
            "meanAuxPopulationSamplingSeedPerSource": POPULATION_SEED,
            "optimizerResetAtPhaseBoundary": True,
            "jointSchedule": "odd updates K562, even updates RPE1 within each phase; exactly half the source exposure and equal total updates versus K562-only",
            "meanScale": "source-specific fitting anchored-mean MSE from panel registry adapter",
            "finalCheckpointOnly": True,
            "earlyStopping": False,
        },
        "nativePanels": "Separate K562 8563-query/48-GEM and RPE1 8749-query/56-GEM likelihoods and controls; no joined target matrix or shared library denominator.",
        "pins": {str(path.relative_to(ROOT)): value for path, value in expected.items()},
        "runnerSha256": sha256(Path(__file__).resolve()),
        "developmentOpened": False,
        "reconstructionHeldOpened": False,
        "testOpened": False,
        "benchmarkAccessed": False,
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
            raise ValueError("frozen shared-context protocol changed")
    else:
        write_json(path, value)
    return value


def load_panels():
    panel_module = load_module(PANEL_DATA, "slp_count_panel_data_runtime")
    return panel_module.load_panels(REGISTRY, ROOT), panel_module


def initialize_model(core, device):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    return core.CountLatentState(core.Config(**MODEL_CONFIG)).to(device)


def panel_tensors(panels, device):
    result = {}
    for name, panel in panels.items():
        result[name] = {
            "query": torch.as_tensor(panel.query_features, device=device),
            "basal": torch.as_tensor(panel.basal_rate, device=device),
            "mask": torch.ones(panel.basal_rate.shape, dtype=torch.bool, device=device),
            "observed": torch.ones(
                (TRAINING["batchSize"], len(panel.query_ids)),
                dtype=torch.bool,
                device=device,
            ),
        }
    return result


def as_cell_batch(step, panel, tensors, rng, device):
    sample = panel.sample_cells(
        rng, TRAINING["controlRows"], TRAINING["targetRows"]
    )
    return step.CellBatch(
        actions=torch.as_tensor(sample["actions"], device=device),
        action_mask=torch.as_tensor(sample["action_mask"], device=device),
        context_index=torch.as_tensor(sample["context_index"], device=device),
        counts=torch.as_tensor(sample["counts"], device=device),
        observed=tensors["observed"],
        library=torch.as_tensor(sample["library"], device=device),
    ), sample["row_index"]


def as_population_batch(step, panel, rng, device):
    sample = panel.sample_populations(rng, TRAINING["populationGenes"])
    return step.PopulationBatch(
        actions=torch.as_tensor(sample["actions"], device=device),
        action_mask=torch.as_tensor(sample["action_mask"], device=device),
        context_weights=torch.as_tensor(sample["context_weights"], device=device),
        target_log1p_mean=torch.as_tensor(sample["target_log1p_mean"], device=device),
    ), sample["gene_index"]


def parameter_norms(model):
    values = defaultdict(float)
    for name, parameter in model.named_parameters():
        value = float(parameter.detach().double().square().sum())
        if not np.isfinite(value):
            raise FloatingPointError(name)
        values[name.split(".", 1)[0]] += value
    return {name: float(total**0.5) for name, total in sorted(values.items())}


def train_arm(core, step, panels, tensors, arm, device):
    model = initialize_model(core, device).train()
    initial = parameter_norms(model)
    source_counts = defaultdict(int)
    histories = []
    traces = {}
    started_arm = time.perf_counter()
    phases = (
        ("count", TRAINING["countUpdates"], SEED, False),
        ("meanAux", TRAINING["meanAuxUpdates"], AUX_SEED, True),
    )
    for phase, updates, count_seed, use_mean in phases:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TRAINING["learningRate"],
            weight_decay=TRAINING["weightDecay"],
        )
        count_rng = {
            name: np.random.default_rng(count_seed) for name in panels
        }
        population_rng = {
            name: np.random.default_rng(POPULATION_SEED) for name in panels
        }
        trace = hashlib.sha256()
        population_trace = hashlib.sha256()
        window = []
        for update in range(1, updates + 1):
            if time.perf_counter() - started_arm > TRAINING["maxSecondsPerArm"]:
                raise TimeoutError(f"{arm} exceeded frozen wall cap")
            source = source_for_update(arm, update)
            panel = panels[source]
            local = tensors[source]
            cells, rows = as_cell_batch(
                step, panel, local, count_rng[source], device
            )
            trace.update(source.encode() + np.asarray(rows, dtype="<i8").tobytes())
            populations = None
            selected = np.empty(0, np.int64)
            if use_mean:
                populations, selected = as_population_batch(
                    step, panel, population_rng[source], device
                )
                population_trace.update(
                    source.encode() + np.asarray(selected, dtype="<i8").tobytes()
                )
            optimizer.zero_grad(set_to_none=True)
            result = step.training_losses(
                model,
                local["query"],
                local["basal"],
                local["mask"],
                cells,
                populations,
                mean_weight=TRAINING["meanWeight"] if use_mean else 0.0,
                fitting_mean_scale=panel.fitting_mean_scale if use_mean else None,
            )
            result["loss"].backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), TRAINING["gradientClip"]
            )
            if not torch.isfinite(gradient):
                raise FloatingPointError(f"nonfinite gradient: {arm}/{phase}/{update}")
            optimizer.step()
            source_counts[f"{phase}:{source}"] += 1
            count_result = result["count_result"]
            window.append(
                {
                    "loss": float(result["loss"].detach()),
                    "countElbo": float(result["count_elbo"].detach()),
                    "reconstructionPerQuery": float(
                        count_result["reconstruction_per_query"].mean().detach()
                    ),
                    "klPerCell": float(count_result["kl_per_cell"].mean().detach()),
                    "meanMse": float(result["mean_mse"].detach()),
                    "gradientNormBeforeClip": float(gradient.detach()),
                }
            )
            if update % TRAINING["logEvery"] == 0:
                item = {
                    "phase": phase,
                    "update": update,
                    "source": source,
                    "elapsedSeconds": time.perf_counter() - started_arm,
                    **{
                        key: float(np.mean([entry[key] for entry in window]))
                        for key in window[0]
                    },
                }
                histories.append(item)
                if update % 1000 == 0:
                    print(json.dumps({"event": "training", "arm": arm, **item}), flush=True)
                window.clear()
        traces[f"{phase}CountRowsSha256"] = trace.hexdigest()
        traces[f"{phase}PopulationRowsSha256"] = population_trace.hexdigest()
    final = parameter_norms(model)
    return model.eval(), {
        "seconds": time.perf_counter() - started_arm,
        "sourceUpdates": dict(source_counts),
        "initialParameterNorms": initial,
        "finalParameterNorms": final,
        "parameterNormChanges": {name: final[name] - initial[name] for name in initial},
        "traces": traces,
        "history": histories,
        "finite": True,
        "completedUpdates": TRAINING["countUpdates"] + TRAINING["meanAuxUpdates"],
    }


@torch.no_grad()
def population_prediction(model, panel, device, chunk=16, limit=None):
    query = torch.as_tensor(panel.query_features, device=device)
    basal = torch.as_tensor(panel.basal_rate, device=device)
    mask = torch.ones_like(basal, dtype=torch.bool)
    context = model.encode_context(query, basal, mask)
    result = []
    groups = len(panel.context_ids)
    gene_count = len(panel.gene_ids) if limit is None else min(int(limit), len(panel.gene_ids))
    for left in range(0, gene_count, chunk):
        actions = torch.as_tensor(panel.gene_action_features[left : left + chunk, None], device=device)
        count = len(actions)
        expanded = actions[:, None].expand(-1, groups, -1, -1).reshape(count * groups, 1, -1)
        prior = model.prior_from_context(
            expanded,
            torch.ones((count * groups, 1), dtype=torch.bool, device=device),
            context.repeat(count, 1),
        )
        rate = model.population_mean(
            prior,
            query,
            basal[None].expand(count, -1, -1).reshape(count * groups, -1),
        ).reshape(count, groups, -1)
        weights = torch.as_tensor(
            panel.population_context_weights[left : left + count], device=device
        )
        result.append(torch.log1p((rate * weights[..., None]).sum(1)).cpu().numpy())
    return np.concatenate(result, axis=0)


def fitting_diagnostics(models, panels, device, output):
    arrays = {}
    metrics = {}
    for source, panel in panels.items():
        anchor = np.log1p(
            panel.population_context_weights.astype(np.float64)
            @ panel.basal_rate.astype(np.float64)
        )
        metrics[source] = {}
        arrays[f"{source}_gene_ids"] = panel.gene_ids
        for arm, model in models.items():
            prediction = population_prediction(model, panel, device)
            summary, mse, correlation = stable_profile_metrics(
                prediction, panel.population_targets, anchor
            )
            metrics[source][arm] = summary
            arrays[f"{source}_{arm}_mse"] = mse
            arrays[f"{source}_{arm}_centered_pearson"] = correlation
        mean_prediction = anchor + (
            panel.population_targets.astype(np.float64) - anchor
        ).mean(0, keepdims=True)
        summary, mse, correlation = stable_profile_metrics(
            mean_prediction, panel.population_targets, anchor
        )
        metrics[source]["fittingAnchoredMean"] = summary
        arrays[f"{source}_fittingAnchoredMean_mse"] = mse
        arrays[f"{source}_fittingAnchoredMean_centered_pearson"] = correlation
    path = output / "fitting-per-gene-diagnostics.npz"
    np.savez_compressed(path, **arrays)
    return metrics, {"path": path.name, "sha256": sha256(path)}


def references(output, panels):
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    shared_path = REGISTRY.parent / registry["static"]["path"]
    with np.load(shared_path, allow_pickle=False) as archive:
        mean = np.asarray(archive["feature_mean"], np.float64)
        scale = np.asarray(archive["feature_scale"], np.float64)
        clip = np.asarray(archive["feature_clip"], np.float32)
        raw_ids = archive["entity_id"].astype(str)
        raw_features = np.asarray(archive["feature_values"], np.float32)
    lookup = {value: row for row, value in enumerate(raw_ids)}
    probe = {}
    for source, panel in panels.items():
        path = output / f"reference-{source}.npz"
        np.savez_compressed(
            path,
            schema=np.asarray("slp.count-world-native-panel-reference/v1"),
            source_id=np.asarray(source),
            query_ids=panel.query_ids,
            context_ids=panel.context_ids,
            query_features=panel.query_features,
            basal_rate=panel.basal_rate,
            feature_mean=mean,
            feature_scale=scale,
            feature_clip=clip,
        )
        raw = raw_features[[lookup[value] for value in panel.gene_ids[:4]]]
        probe[f"{source}_raw_action_features"] = raw
        probe[f"{source}_context_weights"] = panel.population_context_weights[:4]
    return probe


def save_artifact(output, models, training, panels):
    (output / "arms").mkdir(exist_ok=True)
    (output / "source").mkdir(exist_ok=True)
    for arm, model in models.items():
        save_file(model.state_dict(), str(output / "arms" / f"{arm}.safetensors"))
        write_json(output / "arms" / f"{arm}-training.json", training[arm])
    probe = references(output, panels)
    np.savez_compressed(output / "target-free-probe-inputs.npz", **probe)
    for source, destination in (
        (CORE, output / "source/count_latent_state.py"),
        (INFERENCE, output / "source/inference.py"),
        (TRAINING_STEP, output / "source/training_step.py"),
        (OBJECTIVE, output / "source/molecular_mean_objective.py"),
        (PANEL_DATA, output / "source/panel_data.py"),
        (Path(__file__).resolve(), output / "source/runner.py"),
    ):
        shutil.copyfile(source, destination)
    hashes = {
        str(path.relative_to(output)).replace("\\", "/"): sha256(path)
        for path in output.rglob("*")
        if path.is_file() and path.name not in {"artifact-manifest.json", "protocol.json"}
    }
    manifest = {
        "schema": "slp.human-essential-count-shared-context-artifact/v1",
        "protocolSha256": sha256(output / "protocol.json"),
        "arms": {arm: {"modelPath": f"arms/{arm}.safetensors"} for arm in ARMS},
        "panels": {
            source: {"referencePath": f"reference-{source}.npz"} for source in panels
        },
        "sha256": hashes,
        "developmentOpened": False,
        "reconstructionHeldOpened": False,
        "testOpened": False,
    }
    write_json(output / "artifact-manifest.json", manifest)
    return manifest


def target_free_gpu_probe(output, models, panels):
    inference = load_module(output / "source/inference.py", "count_world_artifact_inference")
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    with np.load(REGISTRY.parent / registry["static"]["path"], allow_pickle=False) as static:
        ids = static["entity_id"].astype(str)
        raw = np.asarray(static["feature_values"], np.float32)
    lookup = {value: row for row, value in enumerate(ids)}
    arrays = {}
    metrics = {}
    for source, panel in panels.items():
        raw_action = raw[[lookup[value] for value in panel.gene_ids[:4]]]
        weights = panel.population_context_weights[:4]
        for arm, model in models.items():
            predictor = inference.Predictor(output, arm, source, device="cuda")
            actual = predictor.predict(raw_action, weights)
            direct = population_prediction(
                model, panel, next(model.parameters()).device, limit=4
            )
            difference = np.abs(actual["mean_log1p_cp10k"].astype(np.float64) - direct)
            empty = predictor.predict(
                raw_action, weights, action_mask=np.zeros((4, 1), np.bool_)
            )
            expected_empty = weights.astype(np.float64) @ panel.basal_rate.astype(np.float64)
            key = f"{source}_{arm}"
            arrays[f"{key}_mean_log1p_cp10k"] = actual["mean_log1p_cp10k"]
            arrays[f"{key}_empty_cp10k"] = empty["mean_cp10k"]
            metrics[key] = {
                "maximumAbsoluteLog1pDifference": float(difference.max()),
                "emptyMaximumAbsoluteCp10kDifference": float(
                    np.max(np.abs(empty["mean_cp10k"] - expected_empty))
                ),
                "finite": bool(np.isfinite(actual["mean_cp10k"]).all()),
            }
            if difference.max() > 1e-6 or metrics[key]["emptyMaximumAbsoluteCp10kDifference"] > 1e-3:
                raise RuntimeError(f"portable CUDA replay failed: {key}")
    path = output / "target-free-gpu-probe.npz"
    np.savez_compressed(path, **arrays)
    return metrics, {"path": path.name, "sha256": sha256(path)}


def validation_metadata(source, panel):
    """Load identities and GEM composition only; never open expression arrays."""
    with np.load(ROUTING[source], allow_pickle=False) as archive:
        role = archive["intervention_role"].astype(str)
        selected = role == "validation"
        action = archive["action_ids"][selected].astype(str)
        gem = np.asarray(archive["gem_group"][selected])
        is_control = np.asarray(archive["is_control"][selected], bool)
        context_id = str(archive["context_id"])
    expected = {"k562": (47_914, 305), "rpe1": (39_014, 360)}[source]
    genes = np.asarray(sorted(set(action.tolist())))
    if selected.sum() != expected[0] or len(genes) != expected[1] or is_control.any():
        raise ValueError(f"{source} validation identity boundary drift")
    with np.load(BASELINES[source], allow_pickle=False) as baseline:
        gem_group = np.asarray(baseline["gem_group"])
        baseline_queries = baseline["query_ids"].astype(str)
    lookup = {int(value): row for row, value in enumerate(gem_group)}
    if len(lookup) != len(gem_group) or any(int(value) not in lookup for value in gem):
        raise ValueError(f"{source} validation GEM identity drift")
    gem_count = np.zeros((len(genes), len(gem_group)), np.int64)
    gene_lookup = {gene: row for row, gene in enumerate(genes)}
    for gene, value in zip(action, gem):
        gem_count[gene_lookup[gene], lookup[int(value)]] += 1
    cell_count = gem_count.sum(1)
    expected_cells = np.bincount(
        np.asarray([gene_lookup[gene] for gene in action]), minlength=len(genes)
    )
    if not np.array_equal(cell_count, expected_cells):
        raise AssertionError("validation cell-count reconstruction drift")
    weights = gem_count / cell_count[:, None]
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    with np.load(REGISTRY.parent / registry["static"]["path"], allow_pickle=False) as static:
        ids = static["entity_id"].astype(str)
        raw = np.asarray(static["feature_values"], np.float32)
    static_lookup = {value: row for row, value in enumerate(ids)}
    try:
        raw_actions = raw[[static_lookup[gene] for gene in genes]]
    except KeyError as error:
        raise ValueError(f"validation gene lacks shared static row: {error}") from error
    if not np.array_equal(panel.query_ids, baseline_queries):
        raise ValueError(f"{source} baseline query order drift")
    return {
        "source_id": source,
        "context_id": context_id,
        "gene_ids": genes,
        "gem_group_ids": gem_group,
        "cell_count": cell_count,
        "gem_cell_count": gem_count,
        "gem_weights": weights,
        "raw_action_features": raw_actions,
    }


def freeze_development_forecasts(output, panels):
    """Freeze both neural arms and matched baselines before count-value access."""
    inference = load_module(output / "source/inference.py", "count_world_forecast_inference")
    ridge_core = load_module(RIDGE_CORE, "count_world_forecast_ridge")
    receipts = {}
    for source, panel in panels.items():
        metadata = validation_metadata(source, panel)
        with np.load(BASELINES[source], allow_pickle=False) as archive:
            baseline = {name: np.asarray(archive[name]) for name in archive.files}
        anchor = ridge_core.control_anchor(
            baseline["basal_rate"], metadata["gem_cell_count"]
        )
        ridge = ridge_core.absolute_prediction(
            anchor,
            ridge_core.predict_residual(
                baseline,
                metadata["raw_action_features"],
                str(baseline["selected_alpha"]),
            ),
        )
        mean = ridge_core.absolute_prediction(
            anchor, np.broadcast_to(baseline["target_mean"], anchor.shape)
        )
        predictions = {}
        for arm in ARMS:
            predictor = inference.Predictor(output, arm, source, device="cuda")
            predictions[arm] = predictor.predict(
                metadata["raw_action_features"], metadata["gem_weights"]
            )["mean_log1p_cp10k"]
        path = output / f"development-forecasts-{source}.npz"
        np.savez_compressed(
            path,
            schema=np.asarray("slp.human-essential-count-shared-context-development-forecasts/v1"),
            source_id=np.asarray(source),
            context_id=np.asarray(metadata["context_id"]),
            gene_ids=metadata["gene_ids"],
            query_ids=panel.query_ids,
            gem_group_ids=metadata["gem_group_ids"],
            cell_count=metadata["cell_count"],
            gem_cell_count=metadata["gem_cell_count"],
            control_prediction=anchor,
            anchored_mean_prediction=mean,
            static_ridge_prediction=ridge,
            k562_only_prediction=predictions["k562-only"],
            joint_prediction=predictions["joint-alternating"],
        )
        receipts[source] = {
            "path": path.name,
            "sha256": sha256(path),
            "sourceId": source,
            "contextId": metadata["context_id"],
            "genes": len(metadata["gene_ids"]),
            "queries": len(panel.query_ids),
            "cellsRepresentedByMetadata": int(metadata["cell_count"].sum()),
        }
    manifest = json.loads((output / "artifact-manifest.json").read_text(encoding="utf-8"))
    receipt = {
        "schema": "slp.human-essential-count-shared-context-development-forecast-freeze/v1",
        "forecastsFrozenBeforeDevelopmentCountAccess": True,
        "forecasts": receipts,
        "models": {
            arm: {
                "path": manifest["arms"][arm]["modelPath"],
                "sha256": manifest["sha256"][manifest["arms"][arm]["modelPath"]],
            }
            for arm in ARMS
        },
        "references": {
            source: {
                "path": manifest["panels"][source]["referencePath"],
                "sha256": manifest["sha256"][manifest["panels"][source]["referencePath"]],
            }
            for source in panels
        },
        "baselines": {
            source: {
                "path": str(BASELINES[source].relative_to(ROOT)).replace("\\", "/"),
                "sha256": EXPECTED[BASELINES[source]],
            }
            for source in panels
        },
        "routingMetadata": {
            source: {
                "path": str(ROUTING[source].relative_to(ROOT)).replace("\\", "/"),
                "sha256": EXPECTED[ROUTING[source]],
            }
            for source in panels
        },
        "developmentCountMembersOpened": False,
        "testOpened": False,
    }
    write_json(output / "FORECASTS-FROZEN-BEFORE-DEVELOPMENT.json", receipt)
    return receipt


def profile(output: Path = OUTPUT):
    prepare(output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    panels, _ = load_panels()
    core = load_module(CORE, "count_world_profile_core")
    step = load_module(TRAINING_STEP, "count_world_profile_step")
    device = torch.device("cuda")
    model = initialize_model(core, device).train()
    tensors = panel_tensors(panels, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    count_rng = {name: np.random.default_rng(SEED) for name in panels}
    pop_rng = {name: np.random.default_rng(POPULATION_SEED) for name in panels}
    torch.cuda.reset_peak_memory_stats()
    times = {}
    for source, panel in panels.items():
        samples = []
        for repeat in range(5):
            cells, _ = as_cell_batch(step, panel, tensors[source], count_rng[source], device)
            populations, _ = as_population_batch(step, panel, pop_rng[source], device)
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            result = step.training_losses(
                model,
                tensors[source]["query"],
                tensors[source]["basal"],
                tensors[source]["mask"],
                cells,
                populations,
                mean_weight=0.1,
                fitting_mean_scale=panel.fitting_mean_scale,
            )
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            if repeat:
                samples.append(time.perf_counter() - started)
        times[source] = float(np.mean(samples))
    projected = {
        "k562-only": times["k562"] * 16_000,
        "joint-alternating": (times["k562"] + times["rpe1"]) * 8_000,
    }
    report = {
        "schema": "slp.human-essential-count-shared-context-cuda-profile/v1",
        "secondsPerFullMeanAuxStep": times,
        "conservativeProjectedSecondsPerArm": projected,
        "peakAllocatedBytes": int(torch.cuda.max_memory_allocated()),
        "peakReservedBytes": int(torch.cuda.max_memory_reserved()),
        "fitsCaps": bool(
            max(projected.values()) < TRAINING["maxSecondsPerArm"]
            and torch.cuda.max_memory_reserved() < 10 * 2**30
        ),
        "fittingCountsRead": True,
        "developmentOpened": False,
        "testOpened": False,
    }
    write_json(output / "cuda-profile.json", report)
    print(json.dumps(report))
    return report


def verify_cpu(output: Path):
    inference = load_module(output / "source/inference.py", "count_world_cpu_verify")
    with np.load(output / "target-free-probe-inputs.npz", allow_pickle=False) as probe:
        values = {name: np.asarray(probe[name]) for name in probe.files}
    with np.load(output / "target-free-gpu-probe.npz", allow_pickle=False) as expected:
        expected_values = {name: np.asarray(expected[name]) for name in expected.files}
    metrics = {}
    passes = True
    for source in ("k562", "rpe1"):
        raw = values[f"{source}_raw_action_features"]
        weights = values[f"{source}_context_weights"]
        for arm in ARMS:
            predictor = inference.Predictor(output, arm, source, device="cpu")
            actual = predictor.predict(raw, weights)
            target = expected_values[f"{source}_{arm}_mean_log1p_cp10k"]
            difference = float(np.max(np.abs(actual["mean_log1p_cp10k"] - target)))
            empty1 = predictor.predict(raw, weights, action_mask=np.zeros((4, 1), np.bool_))
            empty2 = predictor.predict(raw, weights, action_mask=np.zeros((4, 1), np.bool_))
            repeat = float(np.max(np.abs(empty1["mean_cp10k"] - empty2["mean_cp10k"])))
            metrics[f"{source}_{arm}"] = {
                "cpuGpuMaximumAbsoluteLog1pDifference": difference,
                "repeatedEmptyMaximumAbsoluteDifference": repeat,
            }
            passes &= difference <= 1e-6 and repeat == 0
    result = {"passes": bool(passes), "metrics": metrics}
    print(json.dumps(result))
    return result


def run(output: Path = OUTPUT):
    prepare(output)
    if (output / "FROZEN-FITTING-ONLY.json").exists():
        raise FileExistsError("immutable fitting-only run already complete")
    profile_report = json.loads((output / "cuda-profile.json").read_text(encoding="utf-8"))
    if not profile_report["fitsCaps"]:
        raise RuntimeError("actual-shape CUDA profile exceeds frozen resource caps")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    panels, _ = load_panels()
    core = load_module(CORE, "count_world_training_core")
    step = load_module(TRAINING_STEP, "count_world_training_step")
    device = torch.device("cuda")
    tensors = panel_tensors(panels, device)
    models, training = {}, {}
    for arm in ARMS:
        models[arm], training[arm] = train_arm(
            core, step, panels, tensors, arm, device
        )
    save_artifact(output, models, training, panels)
    gpu_probe, probe_artifact = target_free_gpu_probe(output, models, panels)
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "verify", "--output", str(output)],
        capture_output=True,
        check=False,
        text=True,
        timeout=300,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
    )
    if process.returncode:
        raise RuntimeError(f"isolated CPU replay failed: {process.stderr[-2000:]}")
    cpu_replay = json.loads(process.stdout.strip().splitlines()[-1])
    if not cpu_replay["passes"]:
        raise RuntimeError("isolated CPU replay exceeded frozen tolerance")
    write_json(output / "isolated-cpu-verification.json", cpu_replay)
    metrics, per_gene = fitting_diagnostics(models, panels, device, output)
    write_json(output / "fitting-diagnostics.json", metrics)
    forecast_freeze = freeze_development_forecasts(output, panels)
    freeze = {
        "schema": "slp.human-essential-count-shared-context-fitting-freeze/v1",
        "protocolSha256": sha256(output / "protocol.json"),
        "artifactManifestSha256": sha256(output / "artifact-manifest.json"),
        "modelSha256": {
            arm: sha256(output / "arms" / f"{arm}.safetensors") for arm in ARMS
        },
        "targetFreeGpuProbe": probe_artifact,
        "isolatedCpuVerificationSha256": sha256(output / "isolated-cpu-verification.json"),
        "fittingDiagnosticsSha256": sha256(output / "fitting-diagnostics.json"),
        "fittingPerGeneDiagnostics": per_gene,
        "developmentForecastFreezeSha256": sha256(
            output / "FORECASTS-FROZEN-BEFORE-DEVELOPMENT.json"
        ),
        "developmentOpened": False,
        "reconstructionHeldOpened": False,
        "testOpened": False,
    }
    write_json(output / "FROZEN-FITTING-ONLY.json", freeze)
    report = {
        "schema": "slp.human-essential-count-shared-context-fitting-report/v1",
        "protocolSha256": freeze["protocolSha256"],
        "artifactManifestSha256": freeze["artifactManifestSha256"],
        "freezeSha256": sha256(output / "FROZEN-FITTING-ONLY.json"),
        "profile": profile_report,
        "training": training,
        "portableGpuReplay": gpu_probe,
        "portableCpuReplay": cpu_replay,
        "fittingDiagnostics": metrics,
        "developmentForecastFreeze": forecast_freeze,
        "interpretation": "Fitting-only evidence for two matched count-world arms. The shared model retains native measurement panels and has not been evaluated on unseen intervention genes. It is not a validated cell generator or identified causal state model.",
        "developmentOpened": False,
        "reconstructionHeldOpened": False,
        "testOpened": False,
        "benchmarkAccessed": False,
    }
    write_json(output / "fitting-report.json", report)
    print(json.dumps({"freeze": freeze, "fittingDiagnostics": metrics}))
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "profile", "run", "verify"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(2):
        {"prepare": prepare, "profile": profile, "run": run, "verify": verify_cpu}[
            args.mode
        ](args.output)
