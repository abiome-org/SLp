#!/usr/bin/env python3
"""Fit the frozen K562 single-cell conditional count-latent pilot.

The development outcome is intentionally loaded only after the final model,
portable reference and target-free CPU replay have been frozen.  This module
also contains the small deterministic sampling and scoring contracts used by
focused tests; real-data orchestration is added only against checksum-pinned
count/static/baseline manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "modules/slp-1-1-count-latent-state-v1/count_latent_state.py"
INFERENCE = ROOT / "modules/slp-1-1-count-latent-inference-v1/inference.py"
SEED = 731
MODEL_CONFIG = {
    "feature_dim": 577,
    "hidden_dim": 128,
    "state_dim": 32,
    "key_dim": 64,
    "dropout": 0.1,
}
TRAINING = {
    "updates": 12_000,
    "batch_size": 128,
    "control_rows_per_batch": 64,
    "target_rows_per_batch": 64,
    "learning_rate": 0.0005,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "beta": 1.0,
    "log_every_updates": 100,
    "seed": SEED,
    "max_training_seconds": 900,
}
RAW_CELL_DIR = ROOT / "data/derived/slp11-human-k562-essential-raw-cells-v2"
TRAINING_MMAP_DIR = ROOT / "data/derived/slp11-human-k562-essential-count-latent-training-mmap-v1"
STATIC_DIR = ROOT / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1"
STATIC_PATH = STATIC_DIR / "k562-essential-count-static577.npz"
ROSTER_PATH = STATIC_DIR / "roster-index.npz"
CONTROL_PATH = ROOT / "data/derived/slp11-human-k562-essential-count-control/reconstruction-train-nt-gem-v1/gem-control-reference.npz"
OUTPUT = ROOT / "results/slp11-transition/k562-essential-count-latent-state-seed731-v1"
ROUTING_PATH = ROOT / "data/derived/slp11-human-k562-essential-singlecell-metadata-v1/cell-routing-metadata.npz"
BASELINE_DIR = ROOT / "results/slp11-transition/k562-essential-count-anchored-static-ridge-seed731-v1"
HASHES = {
    "core": "75df347a82151074c0ce6f4c732106e70ed17126aff07d017294894421d30bac",
    "rawManifest": "859b3fb0b0aeb830e25dce17e86edfc2d8ec3fcdbcec57beeeebf6d1a8faf685",
    "trainingMmapManifest": "4be7a48848cb5c96d32e4da0097af23a27ca4bb3405c31d0e5a6569fe5c1c49d",
    "trainingCounts": "e9bbfe69bd59cedf7131bd176632bb9fbd8dce59a0789ed7e18896ac34e4b511",
    "trainingRows": "5d8631e50b3dcabc9448eaa112eb94bc1335967e5b9098b6e278b6340a9a226b",
    "static": "6706f8867adedef8822897bc275ea90680584f84afd24771e4beb3c8ecf07659",
    "roster": "f2ee702a0714ca7f11f4fd2aa96f4c1825617c0e4f2bcdac42135cd0ba938d7b",
    "control": "c72d28e9eb6633fa237b11e0c16258d875eadaacf31e5b8b3def862150b36d13",
    "routing": "47c89c5082c0a9d4008c6b567407c530933a36fb7603621c37cbe913143f15ad",
    "baselineModel": "dbb669d2eb8d844ec9be7c88a2ed21f5592de434d1b2e916412bda4a52fe1cf3",
    "baselineFreeze": "a57e4d406be62f1ad3c41736f119cde780c20d30c4e7b02e465f265c36fb296f",
    "baselineProtocol": "9ec8520d7c47ecb37f40b4f06f8a54f13f05a34fdced6b2ecf359ac88fa30f0",
    "baselineCore": "1032eeff59382fae3874da9a389033192e113e0f5ac2c8d01f09f8441d969e62",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(entry) for entry in item]
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    path.write_text(
        json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class BalancedSamplingIndex:
    """Metadata-only row pools for the fixed 64-control/64-target sampler."""

    control_gems: tuple[int, ...]
    control_rows: dict[int, np.ndarray]
    target_genes: tuple[str, ...]
    target_populations: dict[str, tuple[str, ...]]
    target_rows: dict[tuple[str, str], np.ndarray]


def build_balanced_sampling_index(
    action_ids: np.ndarray,
    population_ids: np.ndarray,
    gem_group: np.ndarray,
    intervention_role: np.ndarray,
    reconstruction_role: np.ndarray,
    is_control: np.ndarray,
) -> BalancedSamplingIndex:
    """Freeze target-gene→population→cell and control-GEM→cell pools."""
    action = np.asarray(action_ids, dtype=str)
    population = np.asarray(population_ids, dtype=str)
    gem = np.asarray(gem_group, dtype=np.int64)
    intervention = np.asarray(intervention_role, dtype=str)
    reconstruction = np.asarray(reconstruction_role, dtype=str)
    control = np.asarray(is_control)
    n = len(action)
    if (
        any(len(values) != n for values in (population, gem, intervention, reconstruction, control))
        or control.dtype != np.bool_
        or np.any(control & (intervention != "control"))
        or np.any((~control) & (action == ""))
    ):
        raise ValueError("cell routing arrays disagree")
    eligible = reconstruction == "train"
    control_rows = {
        int(group): np.flatnonzero(eligible & control & (gem == group))
        for group in sorted(set(gem[eligible & control].tolist()))
    }
    if not control_rows or any(len(rows) == 0 for rows in control_rows.values()):
        raise ValueError("every registered control GEM requires reconstruction-train cells")
    target_selected = eligible & (~control) & (intervention == "train")
    target_genes = tuple(sorted(set(action[target_selected].tolist())))
    target_populations: dict[str, tuple[str, ...]] = {}
    target_rows: dict[tuple[str, str], np.ndarray] = {}
    for gene in target_genes:
        populations = tuple(sorted(set(population[target_selected & (action == gene)].tolist())))
        if not populations:
            raise AssertionError("target gene lacks a population")
        target_populations[gene] = populations
        for item in populations:
            rows = np.flatnonzero(target_selected & (action == gene) & (population == item))
            if not len(rows):
                raise AssertionError("target population lacks cells")
            target_rows[(gene, item)] = rows
    if not target_genes:
        raise ValueError("no reconstruction-train target genes")
    return BalancedSamplingIndex(
        tuple(control_rows), control_rows, target_genes, target_populations, target_rows
    )


def draw_balanced_rows(
    index: BalancedSamplingIndex,
    generator: np.random.Generator,
    controls: int = 64,
    targets: int = 64,
) -> np.ndarray:
    """Draw GEM-uniform controls and gene/population-uniform targets."""
    if controls <= 0 or targets <= 0:
        raise ValueError("both sampler halves must be positive")
    control_result = np.empty(controls, dtype=np.int64)
    for row in range(controls):
        group = index.control_gems[generator.integers(len(index.control_gems))]
        choices = index.control_rows[group]
        control_result[row] = choices[generator.integers(len(choices))]
    target_result = np.empty(targets, dtype=np.int64)
    for row in range(targets):
        gene = index.target_genes[generator.integers(len(index.target_genes))]
        populations = index.target_populations[gene]
        population = populations[generator.integers(len(populations))]
        choices = index.target_rows[(gene, population)]
        target_result[row] = choices[generator.integers(len(choices))]
    return np.concatenate((control_result, target_result))


def normalize_static_features(
    feature_values: np.ndarray,
    fitting_action_rows: np.ndarray,
    clip: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one scaler on unique fitting-action rows and apply it to all genes."""
    values = np.asarray(feature_values, dtype=np.float64)
    rows = np.asarray(fitting_action_rows, dtype=np.int64)
    if (
        values.ndim != 2
        or rows.ndim != 1
        or not len(rows)
        or len(set(rows.tolist())) != len(rows)
        or np.any(rows < 0)
        or np.any(rows >= len(values))
        or not np.isfinite(values).all()
        or not np.isfinite(clip)
        or clip <= 0
    ):
        raise ValueError("invalid static feature/scaler inputs")
    mean = values[rows].mean(0)
    scale = values[rows].std(0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    normalized = np.clip((values - mean) / scale, -clip, clip).astype(np.float32)
    return normalized, mean.astype(np.float32), scale.astype(np.float32)


def basal_rates_from_control_sums(
    count_sums: np.ndarray,
    library_sums: np.ndarray,
    queries: int,
    pseudocount: float = 0.5,
) -> np.ndarray:
    """Compute the frozen aggregate control CP10k anchor for each GEM."""
    counts = np.asarray(count_sums, dtype=np.float64)
    libraries = np.asarray(library_sums, dtype=np.float64)
    if (
        counts.ndim != 2
        or counts.shape[1] != queries
        or libraries.shape != (len(counts),)
        or not np.isfinite(counts).all()
        or not np.isfinite(libraries).all()
        or np.any(counts < 0)
        or np.any(libraries <= 0)
        or not np.allclose(counts.sum(1), libraries, rtol=0, atol=0)
        or not np.isfinite(pseudocount)
        or pseudocount <= 0
    ):
        raise ValueError("invalid control count sufficient statistics")
    return (10_000.0 * (counts + pseudocount) / (
        libraries[:, None] + pseudocount * queries
    )).astype(np.float32)


def registered_model_inputs(
    static: dict[str, np.ndarray],
    roster: dict[str, np.ndarray],
    control: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Join the frozen static and control resources by exact stable axes."""
    query_ids = np.asarray(roster["query_ids"], dtype=str)
    query_rows = np.asarray(roster["query_entity_index"], dtype=np.int64)
    entities = np.asarray(static["entity_id"], dtype=str)
    normalized = np.asarray(static["normalized_feature_values"], dtype=np.float32)
    raw = np.asarray(static["feature_values"], dtype=np.float32)
    control_queries = np.asarray(control["query_ids"], dtype=str)
    if (
        normalized.shape != raw.shape
        or normalized.shape != (len(entities), 577)
        or query_rows.shape != (8563,)
        or np.any(query_rows < 0)
        or np.any(query_rows >= len(entities))
        or not np.array_equal(entities[query_rows], query_ids)
        or not np.array_equal(control_queries, query_ids)
        or not np.isfinite(normalized).all()
        or not np.isfinite(raw).all()
    ):
        raise ValueError("static/query/control identity axes disagree")
    basal = basal_rates_from_control_sums(
        control["raw_count_sum"], control["library_count_sum"], len(query_ids)
    )
    gems = np.asarray(control["gem_group"])
    if basal.shape != (48, 8563) or gems.shape != (48,) or len(set(gems.tolist())) != 48:
        raise ValueError("expected 48 unique control GEM groups")
    return {
        "query_ids": query_ids,
        "query_features": normalized[query_rows],
        "query_raw_features": raw[query_rows],
        "feature_mean": np.asarray(static["feature_mean"], dtype=np.float32),
        "feature_scale": np.asarray(static["feature_scale"], dtype=np.float32),
        "feature_clip": np.asarray(np.finfo(np.float32).max, dtype=np.float32),
        "basal_rate": basal,
        "basal_observed": np.ones_like(basal, dtype=np.bool_),
        "gem_group_ids": gems,
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def load_training_resources() -> dict[str, object]:
    """Load only fitting/control resources; no held or development counts."""
    paths = {
        "core": CORE,
        "rawManifest": RAW_CELL_DIR / "manifest.json",
        "trainingMmapManifest": TRAINING_MMAP_DIR / "manifest.json",
        "trainingCounts": TRAINING_MMAP_DIR / "counts.uint16",
        "trainingRows": TRAINING_MMAP_DIR / "rows.npz",
        "static": STATIC_PATH,
        "roster": ROSTER_PATH,
        "control": CONTROL_PATH,
    }
    for name, path in paths.items():
        if sha256(path) != HASHES[name]:
            raise ValueError(f"frozen fitting input hash mismatch: {name}")
    rows = load_npz(paths["trainingRows"])
    static = load_npz(paths["static"])
    roster = load_npz(paths["roster"])
    control = load_npz(paths["control"])
    registered = registered_model_inputs(
        static,
        roster,
        {
            "query_ids": control["query_ids"],
            "raw_count_sum": control["control_raw_count_sum"],
            "library_count_sum": control["control_library_count_sum"],
            "gem_group": control["gem_group"],
        },
    )
    if not np.array_equal(registered["basal_rate"], control["basal_rate"]):
        raise ValueError("registered basal reconstruction is not bit-exact")
    if not np.array_equal(rows["query_ids"].astype(str), registered["query_ids"]):
        raise ValueError("training mmap query roster mismatch")
    counts = np.memmap(
        paths["trainingCounts"], mode="r", dtype="<u2", shape=(197_804, 8563), order="C"
    )
    sampling = build_balanced_sampling_index(
        rows["action_ids"], rows["population_ids"], rows["gem_group"],
        rows["intervention_role"], rows["reconstruction_role"], rows["is_control"],
    )
    if len(sampling.target_genes) != 1443 or len(sampling.control_gems) != 48:
        raise ValueError("training sampler gene/GEM coverage drift")
    entity_ids = static["entity_id"].astype(str)
    entity_lookup = {value: row for row, value in enumerate(entity_ids)}
    action_index = np.full(len(rows["action_ids"]), -1, dtype=np.int64)
    for row, action in enumerate(rows["action_ids"].astype(str)):
        if action:
            action_index[row] = entity_lookup[action]
    fitting = roster["fitting_action_ids"].astype(str)
    if set(rows["action_ids"].astype(str)) - {""} != set(fitting.tolist()):
        raise ValueError("mmap action IDs differ from fitting static roster")
    gem_lookup = {int(value): row for row, value in enumerate(registered["gem_group_ids"])}
    gem_index = np.asarray([gem_lookup[int(value)] for value in rows["gem_group"]], np.int64)
    return {
        "paths": paths,
        "counts": counts,
        "rows": rows,
        "static": static,
        "roster": roster,
        "control": control,
        "registered": registered,
        "sampling": sampling,
        "actionEntityIndex": action_index,
        "gemIndex": gem_index,
    }


def gene_gem_weights(
    action_ids: np.ndarray,
    gem_group: np.ndarray,
    gene_ids: np.ndarray,
    gem_ids: np.ndarray,
) -> np.ndarray:
    """Metadata-only validation cell proportions by held gene and GEM."""
    action = np.asarray(action_ids, dtype=str)
    gem = np.asarray(gem_group)
    genes = np.asarray(gene_ids, dtype=str)
    groups = np.asarray(gem_ids)
    if action.shape != gem.shape or len(set(genes.tolist())) != len(genes):
        raise ValueError("invalid gene/GEM metadata")
    lookup = {value: row for row, value in enumerate(groups.tolist())}
    if len(lookup) != len(groups) or any(value not in lookup for value in gem.tolist()):
        raise ValueError("unregistered GEM group")
    weights = np.zeros((len(genes), len(groups)), dtype=np.float64)
    for row, gene in enumerate(genes):
        selected = action == gene
        if not selected.any():
            raise ValueError("held gene has no cells")
        for value in gem[selected]:
            weights[row, lookup[value]] += 1
    return weights / weights.sum(1, keepdims=True)


def aggregate_gene_cp10k(
    counts: np.ndarray,
    library: np.ndarray,
    action_ids: np.ndarray,
    gene_ids: np.ndarray,
) -> np.ndarray:
    """Equal-cell gene means in CP10k, with ln1p applied only after averaging."""
    values = np.asarray(counts)
    exposure = np.asarray(library, dtype=np.float64)
    actions = np.asarray(action_ids, dtype=str)
    genes = np.asarray(gene_ids, dtype=str)
    if (
        values.ndim != 2
        or exposure.shape != (len(values),)
        or actions.shape != (len(values),)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or not np.isfinite(exposure).all()
        or np.any(exposure <= 0)
        or not np.allclose(values.sum(1), exposure, rtol=0, atol=0)
    ):
        raise ValueError("invalid raw-count gene aggregation inputs")
    output = np.empty((len(genes), values.shape[1]), dtype=np.float64)
    for row, gene in enumerate(genes):
        selected = actions == gene
        if not selected.any():
            raise ValueError("held gene has no cells")
        output[row] = np.mean(values[selected] * (10_000.0 / exposure[selected, None]), 0)
    return np.log1p(output)


def _row_pearson(left: np.ndarray, right: np.ndarray, tolerance: float = 1e-12):
    a = left - left.mean(1, keepdims=True)
    b = right - right.mean(1, keepdims=True)
    denominator = np.sqrt(np.sum(a * a, 1) * np.sum(b * b, 1))
    defined = denominator > tolerance
    values = np.full(len(left), np.nan)
    values[defined] = np.sum(a[defined] * b[defined], 1) / denominator[defined]
    return values, defined


def profile_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    anchor: np.ndarray,
) -> dict[str, float | int | None]:
    """Score equal-gene anchored profiles with stable independent centering."""
    pred = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    base = np.asarray(anchor, dtype=np.float64)
    if pred.shape != target.shape or base.shape != target.shape or pred.ndim != 2:
        raise ValueError("profile arrays must share [G,Q]")
    if not np.isfinite(pred).all() or not np.isfinite(target).all() or not np.isfinite(base).all():
        raise ValueError("profile arrays must be finite")
    pred_residual = pred - base
    truth_residual = target - base
    # Anchor before averaging to avoid fabricating residuals from summation roundoff.
    pred_stable = pred_residual - pred_residual[:1]
    truth_stable = truth_residual - truth_residual[:1]
    pred_centered = pred_stable - pred_stable.mean(0, keepdims=True)
    truth_centered = truth_stable - truth_stable.mean(0, keepdims=True)
    correlation, defined = _row_pearson(pred_centered, truth_centered)
    return {
        "geneProfileMse": float(np.mean((pred - target) ** 2)),
        "independentlyQueryCenteredPearson": (
            float(np.mean(correlation[defined])) if defined.any() else None
        ),
        "independentlyQueryCenteredDefinedGenes": int(defined.sum()),
        "independentlyQueryCenteredUndefinedGenes": int((~defined).sum()),
        "genes": len(pred),
        "queries": pred.shape[1],
    }


def parameter_group_norms(model) -> dict[str, float]:
    """Compact finite parameter-norm audit grouped by model component."""
    groups: dict[str, float] = defaultdict(float)
    for name, parameter in model.named_parameters():
        group = name.split(".", 1)[0]
        value = float(parameter.detach().double().square().sum())
        if not np.isfinite(value):
            raise FloatingPointError(f"nonfinite parameter: {name}")
        groups[group] += value
    return {name: float(np.sqrt(value)) for name, value in sorted(groups.items())}


def profile_target_free_cuda(core, query_features: np.ndarray, repeats: int = 20):
    """Measure an actual-shape synthetic B128 optimizer step on CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU training fallback")
    query_array = np.asarray(query_features, dtype=np.float32)
    if query_array.shape != (8563, 577) or not np.isfinite(query_array).all():
        raise ValueError("profile requires the frozen 8563x577 normalized query pack")
    if repeats <= 0:
        raise ValueError("profile repeats must be positive")
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    model = core.CountLatentState(core.Config(**MODEL_CONFIG)).to(device).train()
    query = torch.as_tensor(query_array, device=device)
    basal = torch.rand(48, 8563, device=device) * 30 + 0.1
    basal_mask = torch.ones_like(basal, dtype=torch.bool)
    actions = torch.randn(128, 1, 577, device=device)
    action_mask = torch.ones(128, 1, dtype=torch.bool, device=device)
    action_mask[:64] = False
    counts = torch.randint(0, 4, (128, 8563), device=device, dtype=torch.int64).float()
    observed = torch.ones_like(counts, dtype=torch.bool)
    library = counts.sum(1)
    group_index = torch.arange(128, device=device) % 48
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"]
    )
    torch.cuda.reset_peak_memory_stats()
    times = []
    for step in range(repeats + 3):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        context = model.encode_context(query, basal, basal_mask)
        prior = model.prior_from_context(actions, action_mask, context[group_index])
        output = model.elbo(
            counts, observed, library, query, basal[group_index], prior
        )
        loss = output["loss_per_cell"].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite target-free profile loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING["gradient_clip"])
        optimizer.step()
        torch.cuda.synchronize()
        if step >= 3:
            times.append(time.perf_counter() - started)
    seconds = float(np.mean(times))
    projected = seconds * TRAINING["updates"]
    return {
        "schema": "slp.k562-count-latent-target-free-cuda-profile/v1",
        "batch": 128,
        "queries": 8563,
        "features": 577,
        "controlGemGroups": 48,
        "repeats": repeats,
        "meanOptimizerStepSeconds": seconds,
        "projectedTrainingSeconds": projected,
        "peakAllocatedBytes": int(torch.cuda.max_memory_allocated()),
        "peakReservedBytes": int(torch.cuda.max_memory_reserved()),
        "fitsFrozenCaps": bool(projected < TRAINING["max_training_seconds"] and torch.cuda.max_memory_reserved() < 10 * 2**30),
        "syntheticCountsOnly": True,
        "outcomeValuesAccessed": False,
    }


def deterministic_antithetic_noise(
    cells: int,
    state_dim: int,
    draws: int = 4,
    seed: int = SEED,
) -> np.ndarray:
    """Freeze paired epsilon/-epsilon posterior draws for fit-only evaluation."""
    if cells <= 0 or state_dim <= 0 or draws <= 0 or draws % 2:
        raise ValueError("positive cells/state and an even draw count are required")
    generator = np.random.default_rng(seed)
    half = generator.standard_normal((draws // 2, cells, state_dim)).astype(np.float32)
    return np.concatenate((half, -half), axis=0)


def latent_diagnostics(core, posterior: dict[str, torch.Tensor], prior: dict[str, torch.Tensor]):
    """Return auditable KL activity and variance-clamp fractions."""
    q_mean, q_logvar = posterior["mean"], posterior["logvar"]
    p_mean, p_logvar = prior["mean"], prior["logvar"]
    per_dimension = 0.5 * (
        p_logvar - q_logvar + (q_logvar - p_logvar).exp()
        + (q_mean - p_mean).square() * (-p_logvar).exp() - 1
    )
    if not torch.isfinite(per_dimension).all():
        raise FloatingPointError("nonfinite latent diagnostics")
    return {
        "totalKlPerCell": float(per_dimension.sum(1).mean().detach()),
        "klPerQuery": float(per_dimension.sum(1).mean().detach() / 8563),
        "activeLatentUnitsKlAbovePoint01": int((per_dimension.mean(0) > 0.01).sum()),
        "posteriorLogvarLowerClampFraction": float((q_logvar <= -8.0).float().mean()),
        "posteriorLogvarUpperClampFraction": float((q_logvar >= 4.0).float().mean()),
        "priorLogvarLowerClampFraction": float((p_logvar <= -8.0).float().mean()),
        "priorLogvarUpperClampFraction": float((p_logvar >= 4.0).float().mean()),
    }


def action_batch(
    normalized_features: np.ndarray,
    action_entity_index: np.ndarray,
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(action_entity_index, dtype=np.int64)[rows]
    active = indices >= 0
    values = np.zeros((len(rows), 1, normalized_features.shape[1]), dtype=np.float32)
    values[active, 0] = normalized_features[indices[active]]
    return values, active[:, None]


@torch.no_grad()
def direct_population_prediction(
    model,
    normalized_actions: np.ndarray,
    action_mask: np.ndarray,
    gem_weights: np.ndarray,
    query_features: np.ndarray,
    basal_rate: np.ndarray,
    device: torch.device,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute target-free GEM-mixture prior means from the in-memory model."""
    actions = np.asarray(normalized_actions, dtype=np.float32)
    mask = np.asarray(action_mask)
    weights = np.asarray(gem_weights, dtype=np.float64)
    batch, groups = weights.shape
    if (
        actions.shape[:2] != mask.shape
        or actions.shape[0] != batch
        or mask.dtype != np.bool_
        or basal_rate.shape != (groups, len(query_features))
        or np.any(weights < 0)
        or not np.isfinite(weights).all()
        or np.any(weights.sum(1) <= 0)
    ):
        raise ValueError("invalid direct prior forecast arrays")
    weights = weights / weights.sum(1, keepdims=True)
    query = torch.as_tensor(query_features, device=device)
    basal = torch.as_tensor(basal_rate, device=device)
    basal_mask = torch.ones_like(basal, dtype=torch.bool)
    context = model.encode_context(query, basal, basal_mask)
    prior = model.prior_from_context(
        torch.as_tensor(np.repeat(actions, groups, axis=0), device=device),
        torch.as_tensor(np.repeat(mask, groups, axis=0), device=device),
        context.repeat(batch, 1),
    )
    pieces = []
    for left in range(0, len(query), chunk_size):
        local = basal[:, left : left + chunk_size]
        local = local.unsqueeze(0).expand(batch, -1, -1).reshape(batch * groups, -1)
        mean = model.population_mean(
            prior, query[left : left + chunk_size], local
        ).reshape(batch, groups, -1)
        pieces.append(
            (mean * torch.as_tensor(weights[..., None], device=device)).sum(1).cpu().numpy()
        )
    cp10k = np.concatenate(pieces, axis=1)
    return cp10k, np.log1p(cp10k)


def _load_raw_shard(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _csr_from_raw_shard(values: dict[str, np.ndarray]) -> sparse.csr_matrix:
    shape = tuple(np.asarray(values["raw_shape"], dtype=np.int64).tolist())
    matrix = sparse.csr_matrix(
        (values["raw_data"], values["raw_indices"], values["raw_indptr"]),
        shape=shape,
    )
    if shape[1] != 8563 or matrix.has_canonical_format is False:
        raise ValueError("raw count shard is not canonical 8563-query CSR")
    return matrix


def build_training_mmap(
    raw_dir: Path = RAW_CELL_DIR,
    output: Path = TRAINING_MMAP_DIR,
) -> dict[str, object]:
    """Materialize the sole random-access uint16 reconstruction-training pack."""
    if output.exists():
        raise FileExistsError(f"immutable mmap output exists: {output}")
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = [item for item in manifest["shards"] if item["role"] in {"fit", "control"}]
    expected_rows = 188_195 + 9_609
    if (
        manifest["counts"]["roleRows"]["fit"] != 188_195
        or manifest["counts"]["roleRows"]["control"] != 9_609
        or sum(int(item["rows"]) for item in selected) != expected_rows
    ):
        raise ValueError("canonical reconstruction-training row counts drifted")
    started = time.perf_counter()
    metadata_names = (
        "source_row_index", "cell_ids", "action_ids", "guide_pair_ids",
        "population_ids", "gem_group", "intervention_role", "reconstruction_role",
        "is_control", "library_size",
    )
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in metadata_names}
    for item in selected:
        path = raw_dir / item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"raw shard checksum mismatch: {item['path']}")
        shard = _load_raw_shard(path)
        for name in metadata_names:
            pieces[name].append(shard[name])
    combined = {name: np.concatenate(parts) for name, parts in pieces.items()}
    order = np.argsort(combined["source_row_index"], kind="stable")
    combined = {name: values[order] for name, values in combined.items()}
    if (
        len(combined["source_row_index"]) != expected_rows
        or len(np.unique(combined["source_row_index"])) != expected_rows
        or not np.all(np.diff(combined["source_row_index"]) > 0)
        or not np.all(combined["reconstruction_role"].astype(str) == "train")
        or np.any(combined["library_size"] <= 0)
    ):
        raise ValueError("combined training metadata is not unique positive reconstruction-train")
    source_to_local = {
        int(source): row for row, source in enumerate(combined["source_row_index"])
    }
    output.mkdir(parents=True)
    partial = output / "counts.uint16.partial"
    matrix = np.memmap(partial, mode="w+", dtype="<u2", shape=(expected_rows, 8563))
    maximum_count = 0
    written = np.zeros(expected_rows, dtype=np.bool_)
    for item in selected:
        shard = _load_raw_shard(raw_dir / item["path"])
        csr = _csr_from_raw_shard(shard)
        local = np.asarray(
            [source_to_local[int(source)] for source in shard["source_row_index"]],
            dtype=np.int64,
        )
        dense = csr.toarray()
        maximum_count = max(maximum_count, int(dense.max(initial=0)))
        if maximum_count > np.iinfo(np.uint16).max or np.any(dense < 0):
            raise OverflowError("raw counts do not fit the frozen uint16 contract")
        if not np.array_equal(dense.astype(np.int64).sum(1), shard["library_size"]):
            raise ValueError("shard count/library mismatch during mmap materialization")
        matrix[local] = dense.astype(np.uint16)
        written[local] = True
    if not written.all():
        raise AssertionError("not every reconstruction-training mmap row was written")
    matrix.flush()
    del matrix
    final_counts = output / "counts.uint16"
    partial.replace(final_counts)
    row_path = output / "rows.npz"
    with np.load(raw_dir / "control-gem-moments.npz", allow_pickle=False) as controls:
        query_ids = np.asarray(controls["query_ids"])
    np.savez_compressed(
        row_path,
        schema=np.asarray("slp.k562-count-latent-training-mmap/v1"),
        raw_manifest_sha256=np.asarray(sha256(manifest_path)),
        query_ids=query_ids,
        **combined,
    )
    source_dir = output / "source"
    source_dir.mkdir()
    source_path = source_dir / "prepare.py"
    source_path.write_bytes(Path(__file__).resolve().read_bytes())
    receipt = {
        "schema": "slp.k562-count-latent-training-mmap-manifest/v1",
        "inputs": {
            "rawCellManifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
            "rawCellManifestSha256": sha256(manifest_path),
        },
        "counts": {
            "path": "counts.uint16", "dtype": "uint16 little-endian C-order",
            "shape": [expected_rows, 8563], "bytes": final_counts.stat().st_size,
            "sha256": sha256(final_counts), "maximumCount": maximum_count,
        },
        "rows": {
            "path": "rows.npz", "sha256": sha256(row_path), "rows": expected_rows,
            "ordering": "strict ascending canonical source_row_index",
        },
        "roles": {"fit": 188_195, "control": 9_609},
        "testExcludedRows": 0,
        "developmentValidationRows": 0,
        "reconstructionHeldRows": 0,
        "runtime": {"seconds": time.perf_counter() - started, "peakRssBytes": _rss()},
        "implementation": {
            "path": "source/prepare.py",
            "sha256": sha256(source_path),
        },
    }
    write_json(output / "manifest.json", receipt)
    return receipt


def _rss() -> int:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return 0


def train_model(core, resources: dict[str, object], output: Path):
    """Run the fixed final-only 12,000-update fit on reconstruction-train cells."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU training fallback")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    model = core.CountLatentState(core.Config(**MODEL_CONFIG)).to(device).train()
    initializer_path = output / "initializer.safetensors"
    save_file(model.state_dict(), str(initializer_path))
    initial_norms = parameter_group_norms(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING["learning_rate"],
        weight_decay=TRAINING["weight_decay"],
    )
    registered = resources["registered"]
    static = resources["static"]
    query = torch.as_tensor(registered["query_features"], device=device)
    basal = torch.as_tensor(registered["basal_rate"], device=device)
    basal_mask = torch.as_tensor(registered["basal_observed"], device=device)
    normalized = np.asarray(static["normalized_feature_values"], dtype=np.float32)
    observed = torch.ones((128, 8563), dtype=torch.bool, device=device)
    generator = np.random.default_rng(SEED)
    started = time.perf_counter()
    history = []
    window: list[dict[str, float]] = []
    for update in range(1, TRAINING["updates"] + 1):
        if time.perf_counter() - started > TRAINING["max_training_seconds"]:
            raise TimeoutError("training exceeded the frozen 900 second cap before 12,000 updates")
        rows = draw_balanced_rows(resources["sampling"], generator)
        count_array = np.asarray(resources["counts"][rows], dtype=np.float32)
        libraries = np.asarray(resources["rows"]["library_size"][rows], dtype=np.float32)
        if not np.array_equal(count_array.astype(np.int64).sum(1), libraries.astype(np.int64)):
            raise ValueError("random-access count/library mismatch")
        actions, action_mask = action_batch(
            normalized, resources["actionEntityIndex"], rows
        )
        gem_index = resources["gemIndex"][rows]
        unique_gem, inverse = np.unique(gem_index, return_inverse=True)
        optimizer.zero_grad(set_to_none=True)
        contexts = model.encode_context(
            query, basal[torch.as_tensor(unique_gem, device=device)],
            basal_mask[torch.as_tensor(unique_gem, device=device)],
        )
        prior = model.prior_from_context(
            torch.as_tensor(actions, device=device),
            torch.as_tensor(action_mask, device=device),
            contexts[torch.as_tensor(inverse, device=device)],
        )
        result = model.elbo(
            torch.as_tensor(count_array, device=device), observed,
            torch.as_tensor(libraries, device=device), query,
            basal[torch.as_tensor(gem_index, device=device)], prior,
        )
        loss = result["loss_per_cell"].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at update {update}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), TRAINING["gradient_clip"]
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"nonfinite gradient at update {update}")
        optimizer.step()
        diagnostic = latent_diagnostics(core, result["posterior"], prior)
        window.append({
            "loss": float(loss.detach()),
            "reconstructionPerQuery": float(result["reconstruction_per_query"].mean().detach()),
            "gradientNormBeforeClip": float(gradient_norm.detach()),
            **diagnostic,
        })
        if update % TRAINING["log_every_updates"] == 0:
            item = {"update": update, "elapsedSeconds": time.perf_counter() - started}
            for name in window[0]:
                item[name] = float(np.mean([entry[name] for entry in window]))
            history.append(item)
            print(json.dumps({"event": "training", **item}), flush=True)
            window.clear()
    elapsed = time.perf_counter() - started
    model.eval()
    final_norms = parameter_group_norms(model)
    return model, {
        "updates": TRAINING["updates"],
        "seconds": elapsed,
        "history": history,
        "initialParameterNorms": initial_norms,
        "finalParameterNorms": final_norms,
        "initializerSha256": sha256(initializer_path),
    }


@torch.no_grad()
def evaluate_fitting_reconstruction(core, model, resources: dict[str, object]):
    """Four-draw antithetic ELBO diagnostic on fitting-gene held cells."""
    manifest = json.loads((RAW_CELL_DIR / "manifest.json").read_text(encoding="utf-8"))
    shards = [item for item in manifest["shards"] if item["role"] == "reconstruction-held"]
    device = next(model.parameters()).device
    registered = resources["registered"]
    static = resources["static"]
    query = torch.as_tensor(registered["query_features"], device=device)
    basal = torch.as_tensor(registered["basal_rate"], device=device)
    basal_mask = torch.as_tensor(registered["basal_observed"], device=device)
    normalized = np.asarray(static["normalized_feature_values"], dtype=np.float32)
    entity_lookup = {
        value: row for row, value in enumerate(static["entity_id"].astype(str))
    }
    gem_lookup = {
        int(value): row for row, value in enumerate(registered["gem_group_ids"])
    }
    rng = np.random.default_rng(SEED)
    totals = {
        name: {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0, "cells": 0}
        for name in ("all", "control", "target")
    }
    for item in shards:
        path = RAW_CELL_DIR / item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError("reconstruction-held shard checksum mismatch")
        shard = _load_raw_shard(path)
        csr = _csr_from_raw_shard(shard)
        for left in range(0, len(shard["source_row_index"]), 128):
            stop = min(left + 128, len(shard["source_row_index"]))
            counts = csr[left:stop].toarray().astype(np.float32)
            library = np.asarray(shard["library_size"][left:stop], dtype=np.float32)
            actions = shard["action_ids"][left:stop].astype(str)
            indices = np.asarray(
                [entity_lookup[action] if action else -1 for action in actions], np.int64
            )
            active = indices >= 0
            action_values = np.zeros((len(actions), 1, 577), dtype=np.float32)
            action_values[active, 0] = normalized[indices[active]]
            gem = np.asarray(
                [gem_lookup[int(value)] for value in shard["gem_group"][left:stop]],
                np.int64,
            )
            unique, inverse = np.unique(gem, return_inverse=True)
            context = model.encode_context(
                query, basal[torch.as_tensor(unique, device=device)],
                basal_mask[torch.as_tensor(unique, device=device)],
            )
            prior = model.prior_from_context(
                torch.as_tensor(action_values, device=device),
                torch.as_tensor(active[:, None], device=device),
                context[torch.as_tensor(inverse, device=device)],
            )
            epsilon_half = rng.standard_normal((2, len(actions), 32)).astype(np.float32)
            epsilons = np.concatenate((epsilon_half, -epsilon_half), axis=0)
            draw_loss, draw_reconstruction = [], []
            final = None
            for epsilon in epsilons:
                final = model.elbo(
                    torch.as_tensor(counts, device=device),
                    torch.ones_like(torch.as_tensor(counts, device=device), dtype=torch.bool),
                    torch.as_tensor(library, device=device), query,
                    basal[torch.as_tensor(gem, device=device)], prior,
                    epsilon=torch.as_tensor(epsilon, device=device),
                )
                draw_loss.append(final["loss_per_cell"].cpu().numpy())
                draw_reconstruction.append(final["reconstruction_per_query"].cpu().numpy())
            loss = np.mean(draw_loss, axis=0)
            reconstruction = np.mean(draw_reconstruction, axis=0)
            kl = final["kl_per_cell"].cpu().numpy()
            for name, selected in (
                ("all", np.ones(len(actions), dtype=np.bool_)),
                ("control", ~active), ("target", active),
            ):
                totals[name]["loss"] += float(loss[selected].sum())
                totals[name]["reconstruction"] += float(reconstruction[selected].sum())
                totals[name]["kl"] += float(kl[selected].sum())
                totals[name]["cells"] += int(selected.sum())
    for values in totals.values():
        cells = values["cells"]
        if cells <= 0:
            raise ValueError("fitting reconstruction stratum lacks cells")
        for name in ("loss", "reconstruction", "kl"):
            values[name + "Mean"] = values.pop(name) / cells
        values["klPerQueryMean"] = values["klMean"] / 8563
    if totals["all"]["cells"] != 21_900:
        raise ValueError("reconstruction-held cell count drift")
    return {
        "schema": "slp.k562-count-latent-fitting-reconstruction-diagnostic/v1",
        "draws": 4,
        "noise": "NumPy PCG64 seed731, two draws then their exact negatives per ordered batch",
        "dropout": "disabled final checkpoint",
        "selectionUse": False,
        "strata": totals,
        "developmentOutcomesAccessed": False,
        "testAccessed": False,
    }


def validation_metadata(resources: dict[str, object]) -> dict[str, np.ndarray]:
    """Read identity and GEM composition only; no validation matrix values."""
    if sha256(ROUTING_PATH) != HASHES["routing"]:
        raise ValueError("routing metadata checksum mismatch")
    routing = load_npz(ROUTING_PATH)
    selected = routing["intervention_role"].astype(str) == "validation"
    if int(selected.sum()) != 47_914 or np.any(routing["is_control"][selected]):
        raise ValueError("development-validation metadata boundary drift")
    action = routing["action_ids"][selected].astype(str)
    gem = np.asarray(routing["gem_group"][selected])
    genes = np.asarray(sorted(set(action.tolist())))
    if len(genes) != 305:
        raise ValueError("held intervention gene count drift")
    weights = gene_gem_weights(
        action, gem, genes, resources["registered"]["gem_group_ids"]
    )
    cell_count = np.asarray([np.sum(action == gene) for gene in genes], np.int64)
    gem_count = np.rint(weights * cell_count[:, None]).astype(np.int64)
    if not np.array_equal(gem_count.sum(1), cell_count):
        raise AssertionError("validation GEM count reconstruction mismatch")
    static = resources["static"]
    lookup = {value: row for row, value in enumerate(static["entity_id"].astype(str))}
    entity = np.asarray([lookup[gene] for gene in genes], np.int64)
    return {
        "gene_ids": genes,
        "cell_count": cell_count,
        "gem_cell_count": gem_count,
        "gem_weights": weights,
        "raw_action_features": np.asarray(static["feature_values"][entity], np.float32),
        "normalized_action_features": np.asarray(
            static["normalized_feature_values"][entity], np.float32
        ),
    }


def freeze_target_free_forecasts(
    output: Path,
    resources: dict[str, object],
    metadata: dict[str, np.ndarray],
) -> dict[str, object]:
    """Save neural and baseline forecasts before any validation count member opens."""
    baseline_paths = {
        "model": BASELINE_DIR / "model.npz",
        "freeze": BASELINE_DIR / "FROZEN-BEFORE-DEVELOPMENT.json",
        "protocol": BASELINE_DIR / "protocol.json",
        "core": BASELINE_DIR / "source/count_static_ridge.py",
    }
    for name, path in baseline_paths.items():
        if sha256(path) != HASHES["baseline" + name.capitalize()]:
            raise ValueError(f"frozen baseline checksum mismatch: {name}")
    baseline_core = load_source(baseline_paths["core"], "count_static_ridge_forecast")
    baseline = load_npz(baseline_paths["model"])
    if not np.array_equal(baseline["query_ids"].astype(str), resources["registered"]["query_ids"]):
        raise ValueError("baseline query roster mismatch")
    anchor = baseline_core.control_anchor(
        baseline["basal_rate"], metadata["gem_cell_count"]
    )
    residual = baseline_core.predict_residual(
        baseline, metadata["raw_action_features"], str(baseline["selected_alpha"])
    )
    ridge = baseline_core.absolute_prediction(anchor, residual)
    mean = baseline_core.absolute_prediction(
        anchor, np.broadcast_to(baseline["target_mean"], anchor.shape)
    )
    inference = load_source(output / "source/inference.py", "count_latent_saved_inference")
    predictor = inference.Predictor(output, device="cuda")
    neural = predictor.predict(
        metadata["raw_action_features"], metadata["gem_weights"], chunk_size=1024
    )["mean_log1p_cp10k"]
    arrays = {
        "schema": np.asarray("slp.k562-count-latent-frozen-development-forecasts/v1"),
        "gene_ids": metadata["gene_ids"],
        "query_ids": resources["registered"]["query_ids"],
        "gem_group_ids": resources["registered"]["gem_group_ids"],
        "gem_cell_count": metadata["gem_cell_count"],
        "cell_count": metadata["cell_count"],
        "anchor": anchor,
        "control_prediction": anchor,
        "anchored_mean_prediction": mean,
        "static_ridge_prediction": ridge,
        "neural_prediction": neural,
    }
    path = output / "development-forecasts-before-outcomes.npz"
    np.savez_compressed(path, **arrays)
    receipt = {
        "schema": "slp.k562-count-latent-development-forecast-freeze/v1",
        "forecastSha256": sha256(path),
        "genes": 305,
        "queries": 8563,
        "cellsRepresentedByMetadata": int(metadata["cell_count"].sum()),
        "baselineModelSha256": HASHES["baselineModel"],
        "modelSha256": sha256(output / "model.safetensors"),
        "validationCountMembersOpened": False,
        "testOpened": False,
    }
    write_json(output / "FORECASTS-FROZEN-BEFORE-DEVELOPMENT.json", receipt)
    return receipt


def aggregate_validation_truth(
    genes: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Open the isolated development role once and aggregate equal-cell CP10k."""
    raw_manifest = json.loads((RAW_CELL_DIR / "manifest.json").read_text(encoding="utf-8"))
    shards = [
        item for item in raw_manifest["shards"]
        if item["role"] == "development-validation"
    ]
    lookup = {gene: row for row, gene in enumerate(np.asarray(genes, dtype=str))}
    total = np.zeros((len(genes), 8563), dtype=np.float64)
    cells = np.zeros(len(genes), dtype=np.int64)
    libraries = []
    zeros = 0
    values = 0
    for item in shards:
        path = RAW_CELL_DIR / item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError("development shard checksum mismatch")
        shard = _load_raw_shard(path)
        matrix = _csr_from_raw_shard(shard)
        action = shard["action_ids"].astype(str)
        library = np.asarray(shard["library_size"], dtype=np.float64)
        libraries.extend(library.tolist())
        zeros += matrix.shape[0] * matrix.shape[1] - matrix.nnz
        values += matrix.shape[0] * matrix.shape[1]
        for gene in sorted(set(action.tolist())):
            selected = action == gene
            rate = matrix[selected].astype(np.float64).multiply(
                (10_000.0 / library[selected])[:, None]
            )
            row = lookup[gene]
            total[row] += np.asarray(rate.sum(0)).ravel()
            cells[row] += int(selected.sum())
    if int(cells.sum()) != 47_914 or np.any(cells <= 0):
        raise ValueError("development aggregation cell support drift")
    truth = np.log1p(total / cells[:, None])
    library_array = np.asarray(libraries)
    return truth, {
        "cells": int(cells.sum()),
        "genes": len(genes),
        "queries": 8563,
        "libraryMinimum": float(library_array.min()),
        "libraryMedian": float(np.median(library_array)),
        "libraryMaximum": float(library_array.max()),
        "rawCountZeroFraction": zeros / values,
        "testOpened": False,
    }


def frozen_protocol() -> dict[str, object]:
    return {
        "schema": "slp.k562-essential-count-latent-state-protocol/v1",
        "hypothesis": "A control-conditioned count-latent prior learned from fitting intervention cells improves held-gene aggregate molecular means over matched anchored mean and static-feature ridge baselines.",
        "advancementRule": "On all 305 development held genes and 8,563 queries, raw ln1p(mean-cell-CP10k) gene-profile MSE must be at least 1% below both the frozen anchored mean and static577 ridge, and independently query-centered anchor-residual profile Pearson must be >=0.10 and not below ridge.",
        "accessibleModalities": [
            "raw UMI counts for reconstruction-training fitting interventions and verified non-targeting controls",
            "reconstruction-training non-targeting control rates by GEM group",
            "static ESM8M plus shared MF/CC GO features for interventions and queries",
            "identity and GEM metadata for development-validation cells before forecast freeze",
        ],
        "forbidden": [
            "test-excluded cell count rows", "HepG2", "Jurkat",
            "synthetic-lethality outcomes", "cell count as a mean or state feature",
        ],
        "endpoint": "For each held gene, ln1p of the equal-cell average raw CP10k across its development cells. Neural expectation averages prior CP10k over that gene's metadata-only GEM cell proportions before ln1p.",
        "modelMeaning": "A conditional aggregate-mean approximation with an NB single-cell fitting likelihood and shared Gaussian latent state; not an identified biological state or validated single-cell generator.",
        "modelConfig": MODEL_CONFIG,
        "training": {
            **TRAINING,
            "sampling": "each update draws 64 controls by GEM-uniform then cell-uniform sampling and 64 targets by fitting-gene-uniform, exact sgID_AB population-uniform, then cell-uniform sampling",
            "objective": "mean across cells of (sum observed-query NB NLL + beta1 diagonal Gaussian KL)/8563",
            "selection": "exact final update only; no early stopping or development feedback",
            "contextEfficiency": "encode only distinct GEM controls in the update, then index cells",
        },
        "controlAnchor": "10000*(reconstruction-training NT pooled raw count +0.5)/(pooled full-8563 library +0.5*8563), separately for 48 GEM groups",
        "staticNormalization": "population mean/SD on 1,443 unique fitting action genes only, shared by actions and queries; scale1 when SD<=1e-5; no clipping",
        "fittingDiagnostic": "after final-checkpoint freeze only, four deterministic antithetic posterior draws on 21,900 reconstruction-held cells; never selects or changes the checkpoint",
        "inputs": {
            "rawCellManifest": HASHES["rawManifest"],
            "trainingMmapManifest": HASHES["trainingMmapManifest"],
            "trainingCounts": HASHES["trainingCounts"],
            "trainingRows": HASHES["trainingRows"],
            "static": HASHES["static"],
            "staticRoster": HASHES["roster"],
            "controlReference": HASHES["control"],
            "routingMetadata": HASHES["routing"],
            "baselineModel": HASHES["baselineModel"],
            "baselineFreeze": HASHES["baselineFreeze"],
            "baselineProtocol": HASHES["baselineProtocol"],
            "baselineCore": HASHES["baselineCore"],
        },
        "profile": {
            "sha256": "a13a474cb133a4c189ebbd26b088c072a0d084bc703776eaa3ddee476051124c",
            "sourceSha256": "905abfe1a357c4d2bb40eb64f0a50e46cf384c49d36736e77299f34e3070985f",
            "coreSha256": HASHES["core"],
            "projectedTrainingSeconds": 353.1265798956156,
            "peakReservedBytes": 1233125376,
            "outcomeValuesAccessed": False,
        },
        "sources": {
            "runnerSha256": sha256(Path(__file__).resolve()),
            "coreSha256": sha256(CORE),
            "inferenceSha256": sha256(INFERENCE),
        },
        "developmentEvaluations": 1,
        "testAccessed": False,
        "benchmarkAccessed": False,
    }


def parameter_delta_norms(model, initializer_path: Path) -> dict[str, float]:
    initial = load_file(str(initializer_path))
    groups: dict[str, float] = defaultdict(float)
    for name, parameter in model.state_dict().items():
        group = name.split(".", 1)[0]
        delta = parameter.detach().cpu().double() - initial[name].double()
        groups[group] += float(delta.square().sum())
    return {name: float(np.sqrt(value)) for name, value in sorted(groups.items())}


def prepare_pilot(output: Path = OUTPUT) -> dict[str, object]:
    """Freeze the exact executable protocol without reading count outcomes."""
    if (output / "protocol.json").exists() or (output / "source").exists():
        raise FileExistsError("pilot protocol/source already frozen")
    output.mkdir(parents=True, exist_ok=True)
    source = output / "source"
    source.mkdir()
    shutil.copy2(Path(__file__).resolve(), source / "runner.py")
    shutil.copy2(CORE, source / "count_latent_state.py")
    shutil.copy2(INFERENCE, source / "inference.py")
    protocol = frozen_protocol()
    protocol_path = output / "protocol.json"
    write_json(protocol_path, protocol)
    receipt = {
        "schema": "slp.k562-count-latent-prepared/v1",
        "protocolSha256": sha256(protocol_path),
        "source": {
            "runner.py": sha256(source / "runner.py"),
            "count_latent_state.py": sha256(source / "count_latent_state.py"),
            "inference.py": sha256(source / "inference.py"),
        },
        "profileSha256": sha256(output / "cuda-profile.json"),
        "developmentCountMembersOpened": False,
        "testOpened": False,
    }
    write_json(output / "PREPARED.json", receipt)
    return receipt


def execute_pilot(output: Path = OUTPUT) -> dict[str, object]:
    """Train, freeze/reload, freeze forecasts, then score development once."""
    if (output / "model.safetensors").exists():
        raise FileExistsError("immutable pilot execution already exists")
    if not (output / "protocol.json").exists():
        prepare_pilot(output)
    source = output / "source"
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_sources = {
        "runner.py": protocol["sources"]["runnerSha256"],
        "count_latent_state.py": protocol["sources"]["coreSha256"],
        "inference.py": protocol["sources"]["inferenceSha256"],
    }
    for name, expected in expected_sources.items():
        if sha256(source / name) != expected:
            raise ValueError(f"frozen execution source drift: {name}")
    if sha256(Path(__file__).resolve()) != expected_sources["runner.py"]:
        raise ValueError("invoke the exact frozen runner revision")
    resources = load_training_resources()
    core = load_source(CORE, "count_latent_training_core")
    model, training = train_model(core, resources, output)
    training["parameterDeltaNorms"] = parameter_delta_norms(
        model, output / "initializer.safetensors"
    )
    write_json(output / "loss-history.json", training)
    save_file(model.state_dict(), str(output / "model.safetensors"))
    reference = resources["registered"]
    np.savez_compressed(output / "reference.npz", **reference)

    fitting_ids = resources["roster"]["fitting_action_ids"].astype(str)
    entity_lookup = {
        value: row for row, value in enumerate(resources["static"]["entity_id"].astype(str))
    }
    entity = np.asarray([entity_lookup[value] for value in fitting_ids[:2]], np.int64)
    raw_probe = np.zeros((3, 577), dtype=np.float32)
    raw_probe[:2] = resources["static"]["feature_values"][entity]
    normalized_probe = np.zeros((3, 1, 577), dtype=np.float32)
    normalized_probe[:2, 0] = resources["static"]["normalized_feature_values"][entity]
    probe_mask = np.asarray([[True], [True], [False]])
    probe_weights = np.zeros((3, 48), dtype=np.float64)
    probe_weights[0, 0] = 1
    probe_weights[1] = 1 / 48
    probe_weights[2, 1] = 1
    model.eval()
    expected_cp10k, expected_log = direct_population_prediction(
        model, normalized_probe, probe_mask, probe_weights,
        reference["query_features"], reference["basal_rate"], torch.device("cuda"),
    )
    np.savez_compressed(
        output / "target-free-probe.npz",
        raw_action_features=raw_probe, action_mask=probe_mask,
        gem_group_weights=probe_weights, expected_cp10k=expected_cp10k,
        expected_log1p_cp10k=expected_log,
    )
    artifact_hashes = {
        name: sha256(output / name)
        for name in (
            "model.safetensors", "initializer.safetensors", "reference.npz",
            "loss-history.json", "target-free-probe.npz",
            "source/runner.py", "source/count_latent_state.py", "source/inference.py",
        )
    }
    write_json(
        output / "artifact-manifest.json",
        {"protocolSha256": sha256(protocol_path), "sha256": artifact_hashes},
    )
    saved_inference = load_source(source / "inference.py", "count_latent_cpu_verify")
    predictor = saved_inference.Predictor(output, device="cpu")
    cpu = predictor.predict(
        raw_probe, probe_weights, action_mask=probe_mask, chunk_size=1024
    )
    repeated = predictor.predict(
        raw_probe, probe_weights, action_mask=probe_mask, chunk_size=1024
    )
    maximum = float(np.max(np.abs(cpu["mean_cp10k"] - expected_cp10k)))
    empty_expected = reference["basal_rate"][1]
    verification = {
        "freshCpuMaximumAbsoluteCp10kDifference": maximum,
        "freshCpuWithin1e5": maximum <= 1e-5,
        "repeatedMeanBitExact": bool(np.array_equal(
            cpu["mean_cp10k"], repeated["mean_cp10k"]
        )),
        "emptyMeanMaximumAbsoluteDifference": float(np.max(np.abs(
            cpu["mean_cp10k"][2] - empty_expected
        ))),
        "emptyMeanExact": bool(np.array_equal(cpu["mean_cp10k"][2], empty_expected)),
        "countOrLibraryMeanInputs": False,
    }
    if (
        not verification["freshCpuWithin1e5"]
        or not verification["repeatedMeanBitExact"]
        or not verification["emptyMeanExact"]
    ):
        raise RuntimeError("portable target-free CPU verification failed")
    write_json(output / "inference-verification.json", verification)
    freeze = {
        "schema": "slp.k562-count-latent-final-checkpoint-freeze/v1",
        "protocolSha256": sha256(protocol_path),
        "artifactManifestSha256": sha256(output / "artifact-manifest.json"),
        "modelSha256": artifact_hashes["model.safetensors"],
        "referenceSha256": artifact_hashes["reference.npz"],
        "portableVerification": verification,
        "updates": training["updates"],
        "developmentCountMembersOpened": False,
        "testOpened": False,
    }
    write_json(output / "FROZEN-BEFORE-DEVELOPMENT.json", freeze)
    del predictor

    fitting_diagnostic = evaluate_fitting_reconstruction(core, model, resources)
    write_json(output / "fitting-reconstruction-diagnostic.json", fitting_diagnostic)
    metadata = validation_metadata(resources)
    forecast_freeze = freeze_target_free_forecasts(output, resources, metadata)
    truth, count_diagnostic = aggregate_validation_truth(metadata["gene_ids"])
    forecasts = load_npz(output / "development-forecasts-before-outcomes.npz")
    metrics = {
        name: profile_metrics(forecasts[key], truth, forecasts["anchor"])
        for name, key in (
            ("neural", "neural_prediction"),
            ("staticRidge", "static_ridge_prediction"),
            ("anchoredMean", "anchored_mean_prediction"),
            ("pureControl", "control_prediction"),
        )
    }
    candidate = metrics["neural"]
    ridge = metrics["staticRidge"]
    mean = metrics["anchoredMean"]
    candidate_r = candidate["independentlyQueryCenteredPearson"]
    ridge_r = ridge["independentlyQueryCenteredPearson"]
    gate = {
        "mseAtLeastOnePercentBelowStaticRidge": (
            candidate["geneProfileMse"] <= 0.99 * ridge["geneProfileMse"]
        ),
        "mseAtLeastOnePercentBelowAnchoredMean": (
            candidate["geneProfileMse"] <= 0.99 * mean["geneProfileMse"]
        ),
        "centeredRAtLeastPoint10": candidate_r is not None and candidate_r >= 0.10,
        "centeredRNonregressionVsStaticRidge": (
            candidate_r is not None and ridge_r is not None and candidate_r >= ridge_r
        ),
    }
    gate["passed"] = all(gate.values())
    report = {
        "schema": "slp.k562-essential-count-latent-state-result/v1",
        "protocolSha256": sha256(protocol_path),
        "training": training,
        "fittingReconstruction": fitting_diagnostic,
        "portableVerification": verification,
        "forecastFreeze": forecast_freeze,
        "development": {
            "metrics": metrics,
            "countDiagnostic": count_diagnostic,
            "negativePredictionFraction": float(np.mean(
                forecasts["neural_prediction"] < 0
            )),
            "gate": gate,
        },
        "interpretation": "Adaptive held-gene development evidence for a conditional aggregate-mean count model. It is not a validated single-cell generator, identified latent biology, test result, or benchmark claim.",
        "developmentEvaluations": 1,
        "testAccessed": False,
        "benchmarkAccessed": False,
        "artifacts": {
            **artifact_hashes,
            "development-forecasts-before-outcomes.npz": sha256(
                output / "development-forecasts-before-outcomes.npz"
            ),
            "fitting-reconstruction-diagnostic.json": sha256(
                output / "fitting-reconstruction-diagnostic.json"
            ),
        },
    }
    write_json(output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build-training-mmap", "prepare", "run"))
    parser.add_argument("--raw-dir", type=Path, default=RAW_CELL_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.raw_dir = args.raw_dir.resolve()
    if args.output is None:
        args.output = TRAINING_MMAP_DIR if args.mode == "build-training-mmap" else OUTPUT
    args.output = args.output.resolve()
    return args


if __name__ == "__main__":
    arguments = parse_args()
    result = (
        build_training_mmap(arguments.raw_dir, arguments.output)
        if arguments.mode == "build-training-mmap"
        else prepare_pilot(arguments.output)
        if arguments.mode == "prepare"
        else execute_pilot(arguments.output)
    )
    print(json.dumps(result, indent=2))
