"""Streaming batch-aware ridge diagnostic for Nadal-Ribelles yeast RNA counts.

The runner intentionally never concatenates population-by-query matrices.  It
keeps population metadata in memory, reads one compressed moments shard at a
time, and retains only fitting sufficient statistics or gene-collapsed target
profiles.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/derived/slp11-yeast-atlas-counts/nadal-ribelles-raw-rna-development-v1/moments-manifest.json"
DEFAULT_STATIC = ROOT / "data/derived/slp11-yeast-shared-static/current-sgd-strict-query-full-raw-actions-esm8m-complete-shared-go-v2/yeast-static-esm8m-shared-go-mf-cc-features.npz"
DEFAULT_OUTPUT = ROOT / "results/slp11-transition/yeast-raw-count-batch-ridge-v1"
EXPECTED_MANIFEST_SHA256 = "70a49ecaeb271fc72ecc93ede207c59a816e74d1ae3133bbf3a2803cce5d8eba"
EXPECTED_STATIC_SHA256 = "81cda9469380c9efa000a40b2cd5e816a1d397ce777288fa53b0bcf26a55dc25"
EXPECTED_CORE_SHA256 = "5d897f45ca1318ffe1d447cbafbb1732d0e428efa5f6a7b3dcfe4c32841c18c8"
EXPECTED_FOLDS_SHA256 = "88e51be7dfbb175844f6d2f6c884d482129f38b24af15b3d4528bff82088e57f"
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0)
MAX_RSS_BYTES = 6 * 1024**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _guard(started: float, peak: int) -> None:
    if time.perf_counter() - started > 900.0:
        raise RuntimeError("900 second runtime cap exceeded")
    if peak >= MAX_RSS_BYTES:
        raise MemoryError("6 GiB RSS cap exceeded")


def _load_python(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE_PATH = ROOT / "modules/slp-1-1-batch-ridge-v1/batch_ridge.py"
FOLDS_PATH = ROOT / "modules/slp-1-1-yeast-static-baseline-v1/static_baseline.py"


@dataclass(frozen=True)
class ShardMeta:
    path: Path
    sha256: str
    context: str
    batch: str
    actions: np.ndarray
    roles: np.ndarray
    is_control: np.ndarray
    num_cells: np.ndarray
    total_cells: np.ndarray
    zero_library_cells: np.ndarray


@dataclass(frozen=True)
class Dataset:
    shards: tuple[ShardMeta, ...]
    query_ids: np.ndarray
    contexts: tuple[str, ...]
    manifest_path: Path
    manifest_sha256: str


@dataclass(frozen=True)
class StaticFeatures:
    ids: np.ndarray
    values: np.ndarray
    row_by_id: dict[str, int]
    esm_present: np.ndarray
    go_present: np.ndarray


def _scalar(z: np.lib.npyio.NpzFile, key: str) -> str:
    value = z[key]
    if value.shape != ():
        raise ValueError(f"{key} must be scalar")
    return str(value.item())


def load_metadata(manifest_path: Path, *, verify_hashes: bool = True) -> Dataset:
    manifest_path = Path(manifest_path).resolve()
    manifest_hash = sha256(manifest_path)
    if manifest_path == DEFAULT_MANIFEST.resolve() and manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError("canonical moments manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "slp.yeast-batch-count-moments-manifest/v1":
        raise ValueError("unexpected moments manifest schema")
    required = {
        "schema", "context", "batch_id", "query_ids", "group_action_id",
        "development_role", "is_control", "sum", "sum_squares", "num_cells",
        "total_cells", "zero_library_cells", "mean_observed", "variance_observed",
    }
    query_ids = None
    shards: list[ShardMeta] = []
    for entry in manifest["shards"]:
        path = Path(entry["path"]).resolve()
        if verify_hashes and sha256(path) != entry["sha256"]:
            raise ValueError(f"moments shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != required:
                raise ValueError(f"unexpected members in {path}")
            if _scalar(z, "schema") != "slp.yeast-batch-count-moments/v1":
                raise ValueError("unexpected shard schema")
            context, batch = _scalar(z, "context"), _scalar(z, "batch_id")
            if context != entry["context"] or batch != entry["batchId"]:
                raise ValueError("manifest/shard context or batch mismatch")
            q = np.asarray(z["query_ids"]).astype(str)
            actions = np.asarray(z["group_action_id"]).astype(str)
            roles = np.asarray(z["development_role"]).astype(str)
            controls = np.asarray(z["is_control"], dtype=bool)
            num = np.asarray(z["num_cells"], dtype=np.int64)
            total = np.asarray(z["total_cells"], dtype=np.int64)
            zero = np.asarray(z["zero_library_cells"], dtype=np.int64)
            mean_obs = np.asarray(z["mean_observed"], dtype=bool)
            var_obs = np.asarray(z["variance_observed"], dtype=bool)
        if query_ids is None:
            query_ids = q
        elif not np.array_equal(q, query_ids):
            raise ValueError("query axes differ across shards")
        n = len(actions)
        if any(len(x) != n for x in (roles, controls, num, total, zero, mean_obs, var_obs)):
            raise ValueError("population metadata dimensions differ")
        if not set(roles).issubset({"train", "validation", "control"}):
            raise ValueError("unknown development role")
        if not np.array_equal(controls, roles == "control"):
            raise ValueError("control flag and role differ")
        if np.any(controls & (actions != "CONTROL:WT")) or np.any((~controls) & (actions == "CONTROL:WT")):
            raise ValueError("WT identity and control flag differ")
        if np.any(num <= 0) or np.any(total <= 0) or np.any(zero < 0) or not np.array_equal(num + zero, total):
            raise ValueError("invalid cell support")
        if not mean_obs.all() or not np.array_equal(var_obs, num > 1):
            raise ValueError("observability flags disagree with cell support")
        shards.append(ShardMeta(path, entry["sha256"], context, batch, actions, roles, controls, num, total, zero))
    assert query_ids is not None
    contexts = tuple(sorted({s.context for s in shards}))
    for context in contexts:
        fit_batches = {s.batch for s in shards if s.context == context and np.any(s.roles == "train")}
        val_batches = {s.batch for s in shards if s.context == context and np.any(s.roles == "validation")}
        if not val_batches.issubset(fit_batches):
            raise ValueError(f"validation batch lacks fitting support in {context}")
    return Dataset(tuple(shards), query_ids, contexts, manifest_path, manifest_hash)


def load_static(path: Path) -> StaticFeatures:
    path = Path(path).resolve()
    if path == DEFAULT_STATIC.resolve() and sha256(path) != EXPECTED_STATIC_SHA256:
        raise ValueError("canonical static feature hash mismatch")
    with np.load(path, allow_pickle=False) as z:
        ids = np.asarray(z["entity_id"]).astype(str)
        taxa = np.asarray(z["entity_taxon"], dtype=np.int64)
        values = np.asarray(z["feature_values"], dtype=np.float64)
        esm = np.asarray(z["esm_present"], dtype=bool)
        go = np.asarray(z["go_direct_annotation_present"], dtype=bool)
    if values.shape != (len(ids), 577) or taxa.shape != ids.shape or np.any(taxa != 4932):
        raise ValueError("invalid static feature identities or dimensions")
    if len(set(ids)) != len(ids) or not np.isfinite(values).all():
        raise ValueError("static identifiers must be unique and values finite")
    return StaticFeatures(ids, values, {v: i for i, v in enumerate(ids)}, esm, go)


def population_weights(actions: np.ndarray, num_cells: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions).astype(str)
    num_cells = np.asarray(num_cells, dtype=np.float64)
    if actions.ndim != 1 or num_cells.shape != actions.shape or np.any(num_cells <= 0):
        raise ValueError("invalid actions or cell counts")
    totals = {gene: float(num_cells[actions == gene].sum()) for gene in set(actions)}
    return np.asarray([n / totals[g] for g, n in zip(actions, num_cells, strict=True)])


def pool_genes(actions: np.ndarray, num_cells: np.ndarray, *values: np.ndarray):
    actions = np.asarray(actions).astype(str)
    num_cells = np.asarray(num_cells, dtype=np.float64)
    genes = np.asarray(sorted(set(actions)))
    pooled = []
    for value in values:
        value = np.asarray(value, dtype=np.float64)
        if value.ndim != 2 or value.shape[0] != len(actions):
            raise ValueError("value rows do not match actions")
        pooled.append(np.stack([np.average(value[actions == g], axis=0, weights=num_cells[actions == g]) for g in genes]))
    return (genes, *pooled)


def fit_feature_normalizer(features: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    actions = np.asarray(actions).astype(str)
    if features.ndim != 2 or features.shape[0] != len(actions):
        raise ValueError("features/actions mismatch")
    first = np.asarray([np.flatnonzero(actions == g)[0] for g in sorted(set(actions))])
    unique = features[first]
    mean, scale = unique.mean(axis=0), unique.std(axis=0)
    scale[scale == 0] = 1.0
    return mean, scale


def _gene_features(static: StaticFeatures, genes: Iterable[str]) -> np.ndarray:
    genes = list(genes)
    missing = [g for g in genes if g not in static.row_by_id]
    if missing:
        raise ValueError(f"static features missing {len(missing)} actions")
    return np.stack([static.values[static.row_by_id[g]] for g in genes])


def feature_normalizer_for_genes(static: StaticFeatures, genes: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    genes = sorted(set(genes))
    raw = _gene_features(static, genes)
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    scale[scale == 0] = 1.0
    return mean, scale


def _rss() -> int:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return 0


def _role_genes(dataset: Dataset, context: str, role: str) -> np.ndarray:
    return np.asarray(sorted({g for s in dataset.shards if s.context == context for g, r in zip(s.actions, s.roles, strict=True) if r == role}))


def _gene_totals(dataset: Dataset, context: str, role: str, allowed: set[str]) -> dict[str, float]:
    totals = {g: 0.0 for g in allowed}
    for shard in dataset.shards:
        if shard.context != context:
            continue
        mask = (shard.roles == role) & np.isin(shard.actions, list(allowed))
        for gene, cells in zip(shard.actions[mask], shard.num_cells[mask], strict=True):
            totals[gene] += float(cells)
    if any(v <= 0 for v in totals.values()):
        raise ValueError("allowed gene has no positive-cell population")
    return totals


def build_statistics(dataset: Dataset, static: StaticFeatures, context: str, allowed_genes: Iterable[str], mean: np.ndarray, scale: np.ndarray):
    core = _load_python(CORE_PATH, "slp11_batch_ridge_core_runtime")
    allowed = set(allowed_genes)
    totals = _gene_totals(dataset, context, "train", allowed)
    batch_stats = core.BatchRidgeStatistics(577, len(dataset.query_ids))
    pooled_stats = core.BatchRidgeStatistics(577, len(dataset.query_ids))
    peak = _rss()
    for shard in dataset.shards:
        if shard.context != context:
            continue
        mask = (shard.roles == "train") & np.isin(shard.actions, list(allowed))
        if not mask.any():
            continue
        with np.load(shard.path, allow_pickle=False) as z:
            sums = np.asarray(z["sum"])[mask]
        actions = shard.actions[mask]
        cells = shard.num_cells[mask].astype(np.float64)
        y = sums / cells[:, None]
        x = (_gene_features(static, actions) - mean) / scale
        weight = np.asarray([n / totals[g] for g, n in zip(actions, cells, strict=True)])
        batch_stats.update(shard.batch, x, y, weight)
        pooled_stats.update("pooled", x, y, weight)
        peak = max(peak, _rss())
        del sums, y, x
        gc.collect()
    return batch_stats, pooled_stats, peak


def batch_only_model(stats) -> dict[str, np.ndarray]:
    labels = sorted(stats.batches)
    means = np.stack([stats.batches[b]["y_sum"] / stats.batches[b]["weight"] for b in labels])
    return {
        "coefficients": np.zeros((stats.xx.shape[0], stats.xy.shape[1])),
        "batch_ids": np.asarray(labels),
        "batch_offsets": means.copy(),
        "batch_only_means": means.copy(),
        "alpha": np.asarray(np.inf),
    }


def stream_pooled_truth(dataset: Dataset, context: str, role: str, genes: Iterable[str]) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]], int]:
    genes = np.asarray(sorted(set(genes)))
    index = {g: i for i, g in enumerate(genes)}
    totals = _gene_totals(dataset, context, role, set(genes))
    pooled = np.zeros((len(genes), len(dataset.query_ids)), dtype=np.float64)
    batch_weights: dict[str, dict[str, float]] = {g: {} for g in genes}
    peak = _rss()
    for shard in dataset.shards:
        if shard.context != context:
            continue
        mask = (shard.roles == role) & np.isin(shard.actions, genes)
        if not mask.any():
            continue
        with np.load(shard.path, allow_pickle=False) as z:
            sums = np.asarray(z["sum"])[mask]
        actions = shard.actions[mask]
        cells = shard.num_cells[mask].astype(np.float64)
        means = sums / cells[:, None]
        for gene, n, value in zip(actions, cells, means, strict=True):
            w = float(n / totals[gene])
            pooled[index[gene]] += w * value
            batch_weights[gene][shard.batch] = batch_weights[gene].get(shard.batch, 0.0) + w
        peak = max(peak, _rss())
        del sums, means
        gc.collect()
    return genes, pooled, batch_weights, peak


def _batch_row(model: dict[str, np.ndarray], batch: str, key: str) -> np.ndarray:
    where = np.flatnonzero(model["batch_ids"] == batch)
    if len(where) != 1:
        raise ValueError(f"batch {batch} has no unique fitting reference")
    return model[key][where[0]]


def predict_pooled_genes(model: dict[str, np.ndarray], batch_reference: dict[str, np.ndarray], static: StaticFeatures, genes: np.ndarray, batch_weights: dict[str, dict[str, float]], mean: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = (_gene_features(static, genes) - mean) / scale
    linear = x @ model["coefficients"]
    raw, residual = np.empty_like(linear), np.empty_like(linear)
    for i, gene in enumerate(genes):
        raw[i] = linear[i]
        residual[i] = linear[i]
        for batch, weight in batch_weights[gene].items():
            model_batch = "pooled" if model["batch_ids"].tolist() == ["pooled"] else batch
            raw[i] += weight * _batch_row(model, model_batch, "batch_offsets")
            residual[i] += weight * (_batch_row(model, model_batch, "batch_offsets") - _batch_row(batch_reference, batch, "batch_only_means"))
    return raw, residual


def residualize_truth(truth: np.ndarray, genes: np.ndarray, batch_weights: dict[str, dict[str, float]], batch_reference: dict[str, np.ndarray]) -> np.ndarray:
    out = truth.copy()
    for i, gene in enumerate(genes):
        for batch, weight in batch_weights[gene].items():
            out[i] -= weight * _batch_row(batch_reference, batch, "batch_only_means")
    return out


def _row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    original_a, original_b = a, b
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    eps = np.finfo(np.float64).eps
    # Subtraction of a large constant profile can leave roundoff-scale residue.
    # Treat it as undefined relative to the original row magnitude and width.
    tol_a = 8 * eps * np.sqrt(a.shape[1]) * np.maximum(1.0, np.max(np.abs(original_a), axis=1))
    tol_b = 8 * eps * np.sqrt(b.shape[1]) * np.maximum(1.0, np.max(np.abs(original_b), axis=1))
    valid = (norm_a > tol_a) & (norm_b > tol_b)
    denom = norm_a * norm_b
    return np.divide(np.sum(a * b, axis=1), denom, out=np.full(len(a), np.nan), where=valid)


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    if truth.shape != prediction.shape or truth.ndim != 2 or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("metrics need aligned finite matrices")
    ordinary = _row_corr(truth, prediction)
    # Subtract a reference row first: summing a large common expression profile
    # can otherwise leave rounding residue that looks like gene variation.
    truth_delta = truth - truth[:1]
    prediction_delta = prediction - prediction[:1]
    query_centered = _row_corr(
        truth_delta - truth_delta.mean(axis=0),
        prediction_delta - prediction_delta.mean(axis=0),
    )
    ordinary_value = float(np.nanmean(ordinary)) if np.isfinite(ordinary).any() else None
    centered_value = float(np.nanmean(query_centered)) if np.isfinite(query_centered).any() else None
    return {
        "geneProfileMse": float(np.mean((truth - prediction) ** 2, axis=1).mean()),
        "ordinaryGeneProfilePearson": ordinary_value,
        "ordinaryUndefinedGenes": int((~np.isfinite(ordinary)).sum()),
        "independentlyQueryCenteredPearson": centered_value,
        "independentlyQueryCenteredUndefinedGenes": int((~np.isfinite(query_centered)).sum()),
    }


def _folds(genes: np.ndarray) -> np.ndarray:
    helper = _load_python(FOLDS_PATH, "slp11_yeast_static_folds_runtime")
    return helper.grouped_folds(genes, folds=3, seed=731)


def _candidate(stats, candidate: str):
    return batch_only_model(stats) if candidate == "mean" else stats.solve(float(candidate))


def cross_validate_context(dataset: Dataset, static: StaticFeatures, context: str, started: float) -> dict[str, object]:
    genes = _role_genes(dataset, context, "train")
    assignments = _folds(genes)
    candidates = [str(alpha) for alpha in ALPHAS] + ["mean"]
    squared_error = {arm: {candidate: 0.0 for candidate in candidates} for arm in ("pooled", "batch")}
    elements = 0
    fold_counts = []
    peak = _rss()
    for fold in range(3):
        held = genes[assignments == fold]
        fitting = genes[assignments != fold]
        mean, scale = feature_normalizer_for_genes(static, fitting)
        batch_stats, pooled_stats, stats_peak = build_statistics(dataset, static, context, fitting, mean, scale)
        peak = max(peak, stats_peak)
        held_ids, truth, batch_weights, truth_peak = stream_pooled_truth(dataset, context, "train", held)
        peak = max(peak, truth_peak)
        elements += truth.size
        fold_counts.append({"fold": fold, "fittingGenes": len(fitting), "heldGenes": len(held)})
        for arm, stats in (("pooled", pooled_stats), ("batch", batch_stats)):
            for candidate in candidates:
                model = _candidate(stats, candidate)
                prediction, _ = predict_pooled_genes(model, batch_only_model(batch_stats), static, held_ids, batch_weights, mean, scale)
                squared_error[arm][candidate] += float(np.square(truth - prediction).sum())
                peak = max(peak, _rss())
                del model, prediction
        del batch_stats, pooled_stats, truth
        gc.collect()
        _guard(started, peak)
    scores = {arm: {candidate: squared_error[arm][candidate] / elements for candidate in candidates} for arm in squared_error}
    selected = {arm: min(candidates, key=lambda candidate: (scores[arm][candidate], candidates.index(candidate))) for arm in scores}
    return {
        "genes": genes,
        "foldAssignments": assignments,
        "foldCounts": fold_counts,
        "candidateMse": scores,
        "selected": selected,
        "peakRssBytes": peak,
    }


def fit_final_context(dataset: Dataset, static: StaticFeatures, context: str, selected: dict[str, str]) -> dict[str, object]:
    genes = _role_genes(dataset, context, "train")
    mean, scale = feature_normalizer_for_genes(static, genes)
    batch_stats, pooled_stats, peak = build_statistics(dataset, static, context, genes, mean, scale)
    return {
        "featureMean": mean,
        "featureScale": scale,
        "pooled": _candidate(pooled_stats, selected["pooled"]),
        "batch": _candidate(batch_stats, selected["batch"]),
        "pooledMean": batch_only_model(pooled_stats),
        "batchMean": batch_only_model(batch_stats),
        "batchReference": batch_only_model(batch_stats),
        "fittingGenes": genes,
        "peakRssBytes": peak,
    }


def _save_model(path: Path, model: dict[str, np.ndarray]) -> None:
    np.savez_compressed(
        path,
        coefficients=np.asarray(model["coefficients"], dtype=np.float64),
        batch_ids=np.asarray(model["batch_ids"]).astype(str),
        batch_offsets=np.asarray(model["batch_offsets"], dtype=np.float64),
        batch_only_means=np.asarray(model["batch_only_means"], dtype=np.float64),
        alpha=np.asarray(model["alpha"], dtype=np.float64),
    )


def save_final_context(output: Path, context: str, fitted: dict[str, object], query_ids: np.ndarray) -> dict[str, str]:
    slug = context.lower().replace(" ", "-")
    hashes = {}
    for name in ("pooled", "batch", "pooledMean", "batchMean", "batchReference"):
        path = output / f"{slug}-{name}.npz"
        _save_model(path, fitted[name])
        hashes[path.name] = sha256(path)
    reference = output / f"{slug}-reference.npz"
    np.savez(
        reference,
        query_ids=np.asarray(query_ids).astype(str),
        fitting_action_ids=np.asarray(fitted["fittingGenes"]).astype(str),
        feature_mean=np.asarray(fitted["featureMean"], dtype=np.float64),
        feature_scale=np.asarray(fitted["featureScale"], dtype=np.float64),
    )
    hashes[reference.name] = sha256(reference)
    return hashes


def evaluate_context(dataset: Dataset, static: StaticFeatures, context: str, fitted: dict[str, object], output: Path) -> dict[str, object]:
    validation = _role_genes(dataset, context, "validation")
    genes, truth, batch_weights, peak = stream_pooled_truth(dataset, context, "validation", validation)
    batch_reference = fitted["batchReference"]
    truth_residual = residualize_truth(truth, genes, batch_weights, batch_reference)
    predictions: dict[str, np.ndarray] = {}
    residual_predictions: dict[str, np.ndarray] = {}
    results = {}
    for name in ("pooled", "batch", "pooledMean", "batchMean"):
        raw, residual = predict_pooled_genes(
            fitted[name], batch_reference, static, genes, batch_weights,
            fitted["featureMean"], fitted["featureScale"],
        )
        predictions[name], residual_predictions[name] = raw, residual
        results[name] = {"raw": metrics(truth, raw), "batchMeanSubtracted": metrics(truth_residual, residual)}
        peak = max(peak, _rss())
    batch_mse = results["batch"]["raw"]["geneProfileMse"]
    comparison_mses = [results[name]["raw"]["geneProfileMse"] for name in ("pooled", "pooledMean", "batchMean")]
    residual_r = results["batch"]["batchMeanSubtracted"]["independentlyQueryCenteredPearson"]
    pooled_r = results["pooled"]["batchMeanSubtracted"]["independentlyQueryCenteredPearson"]
    finite_mse = bool(np.isfinite(batch_mse) and np.isfinite(comparison_mses).all())
    finite_r = bool(np.isfinite(residual_r) and np.isfinite(pooled_r))
    gate = {
        "allRequiredMseFinite": finite_mse,
        "allRequiredResidualCenteredCorrelationsFinite": finite_r,
        "batchRidgeAtLeast2PctBetterThanPooledRidge": bool(finite_mse and batch_mse <= 0.98 * results["pooled"]["raw"]["geneProfileMse"]),
        "batchRidgeAtLeast2PctBetterThanPooledMean": bool(finite_mse and batch_mse <= 0.98 * results["pooledMean"]["raw"]["geneProfileMse"]),
        "batchRidgeAtLeast2PctBetterThanBatchMean": bool(finite_mse and batch_mse <= 0.98 * results["batchMean"]["raw"]["geneProfileMse"]),
        "batchResidualCenteredRAtLeastPoint1": bool(finite_r and residual_r >= 0.1),
        "batchResidualCenteredRNonregressionVsPooled": bool(finite_r and residual_r >= pooled_r),
    }
    gate["passes"] = all(gate.values())
    slug = context.lower().replace(" ", "-")
    prediction_path = output / f"{slug}-validation-gene-predictions.npz"
    np.savez(
        prediction_path,
        action_ids=genes,
        query_ids=dataset.query_ids,
        truth=truth,
        truth_batch_mean_subtracted=truth_residual,
        **{f"prediction_{key}": value for key, value in predictions.items()},
        **{f"prediction_{key}_batch_mean_subtracted": value for key, value in residual_predictions.items()},
    )
    return {
        "validationGenes": len(genes),
        "metrics": results,
        "gate": gate,
        "predictionPath": str(prediction_path),
        "predictionSha256": sha256(prediction_path),
        "peakRssBytes": peak,
    }


def control_diagnostics(dataset: Dataset) -> dict[str, object]:
    result = {}
    for context in dataset.contexts:
        rows = [(s.batch, int(s.num_cells[s.is_control][0])) for s in dataset.shards if s.context == context]
        result[context] = {
            "controlPopulations": len(rows),
            "controlCells": sum(v for _, v in rows),
            "minimumCellsPerBatch": min(v for _, v in rows),
            "maximumCellsPerBatch": max(v for _, v in rows),
            "batches": [{"batch": b, "cells": n} for b, n in rows],
            "sumSquaresUse": "retained in source moments; not treated as independent replicate variance and not used by this point baseline",
        }
    return result


def static_coverage(dataset: Dataset, static: StaticFeatures) -> dict[str, object]:
    genes = sorted({g for s in dataset.shards for g, r in zip(s.actions, s.roles, strict=True) if r in {"train", "validation"}})
    missing = [g for g in genes if g not in static.row_by_id]
    if missing:
        raise ValueError(f"static pack misses {len(missing)} development genes")
    rows = np.asarray([static.row_by_id[g] for g in genes])
    return {
        "developmentGenes": len(genes),
        "staticCoveredGenes": len(genes),
        "esmPresentGenes": int(static.esm_present[rows].sum()),
        "directSharedGoPresentGenes": int(static.go_present[rows].sum()),
    }


def _persist_source(output: Path) -> dict[str, str]:
    source = output / "source"
    source.mkdir(parents=True, exist_ok=True)
    copies = {
        "runner.py": Path(__file__),
        "batch_ridge.py": CORE_PATH,
        "static_baseline.py": FOLDS_PATH,
    }
    hashes = {}
    for name, original in copies.items():
        destination = source / name
        shutil.copy2(original, destination)
        hashes[f"source/{name}"] = sha256(destination)
    return hashes


def _write_target_free_probe(output: Path, dataset: Dataset, static: StaticFeatures, fitted: dict[str, object]) -> tuple[Path, dict[str, object]]:
    arrays: dict[str, np.ndarray] = {"context_names": np.asarray(dataset.contexts)}
    metadata = {"models": ["pooled", "batch", "pooledMean", "batchMean"], "contexts": {}}
    for context_index, context in enumerate(dataset.contexts):
        slug = context.lower().replace(" ", "-")
        genes = np.asarray(fitted[context]["fittingGenes"][:2])
        batch = str(fitted[context]["batchReference"]["batch_ids"][0])
        raw = _gene_features(static, genes)
        normalized = (raw - fitted[context]["featureMean"]) / fitted[context]["featureScale"]
        arrays[f"raw_features_{context_index}"] = raw
        arrays[f"action_ids_{context_index}"] = genes
        arrays[f"batch_id_{context_index}"] = np.asarray(batch)
        metadata["contexts"][context] = {"slug": slug, "batch": batch}
        for name in metadata["models"]:
            model = fitted[context][name]
            label = "pooled" if model["batch_ids"].tolist() == ["pooled"] else batch
            arrays[f"expected_{context_index}_{name}"] = normalized @ model["coefficients"] + _batch_row(model, label, "batch_offsets")
    path = output / "target-free-fitting-static-probe.npz"
    np.savez_compressed(path, **arrays)
    return path, metadata


def verify_artifact(output: Path) -> dict[str, object]:
    output = Path(output).resolve()
    manifest_path = output / "artifact-manifest-before-validation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in manifest["fileHashes"].items():
        if sha256(output / relative) != expected:
            raise ValueError(f"artifact hash mismatch: {relative}")
    with np.load(output / "target-free-fitting-static-probe.npz", allow_pickle=False) as probe:
        contexts = np.asarray(probe["context_names"]).astype(str)
        maximum = 0.0
        for context_index, context in enumerate(contexts):
            slug = context.lower().replace(" ", "-")
            with np.load(output / f"{slug}-reference.npz", allow_pickle=False) as reference:
                raw = np.asarray(probe[f"raw_features_{context_index}"], dtype=np.float64)
                x = (raw - reference["feature_mean"]) / reference["feature_scale"]
            batch = str(probe[f"batch_id_{context_index}"].item())
            for name in ("pooled", "batch", "pooledMean", "batchMean"):
                with np.load(output / f"{slug}-{name}.npz", allow_pickle=False) as z:
                    model = {key: np.asarray(z[key]) for key in z.files}
                label = "pooled" if model["batch_ids"].astype(str).tolist() == ["pooled"] else batch
                actual = x @ model["coefficients"] + _batch_row(model, label, "batch_offsets")
                expected = np.asarray(probe[f"expected_{context_index}_{name}"], dtype=np.float64)
                maximum = max(maximum, float(np.max(np.abs(actual - expected))))
    if maximum > 1e-12:
        raise ValueError(f"fresh artifact replay differs by {maximum}")
    result = {"schema": "slp.yeast-batch-ridge-target-free-replay/v1", "maximumAbsoluteDifference": maximum, "tolerance": 1e-12, "passes": True}
    _write_json(output / "target-free-replay.json", result)
    return result


def prepare_protocol(dataset: Dataset, static: StaticFeatures, output: Path) -> dict[str, object]:
    profile_path = output / "resource-profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if not profile["passesRssGuard"] or not profile["passesProjectedRuntimeGuard"]:
        raise ValueError("resource profile does not pass fixed guards")
    protocol = {
        "schema": "slp.yeast-raw-count-batch-ridge-protocol/v2",
        "supersedesUnfittedProtocolSha256": sha256(output / "protocol.json") if (output / "protocol.json").exists() else None,
        "hypothesis": "Explicit fitting-derived source-batch intercepts improve unseen-gene absolute RNA profiles beyond pooled feature ridge and both corresponding mean limits.",
        "fixedRule": "In each Control and NaCl environment, batch ridge must reduce raw gene-profile MSE by at least 2% versus pooled ridge, pooled mean, and batch mean; independently query-centered Pearson after subtraction of the same fitting-derived batch-only mean from truth and prediction must be finite, at least 0.10, and no lower than pooled ridge.",
        "target": "population mean of per-cell ln(1 + 10000 * count / sum_all_6951_source_RNA_rows); absolute, no WT subtraction",
        "fit": "context-separate; controls excluded; each gene total weight one and populations weighted by positive-library cells within gene/context",
        "features": "577 raw ESM2-t6 plus shared GO MF/CC coordinates; unique fitting-gene mean/population-SD normalization, constant SD replaced by one, refit inside folds",
        "innerSelection": {"folds": 3, "seed": 731, "helperDomain": "slp11-yeast-static-baseline-inner-v1", "criterion": "raw equal-gene profile MSE", "candidates": [*ALPHAS, "mean"]},
        "evaluation": "cell-weighted population collapse within held gene across source batches; validation is accessed only after final model files and freeze receipt are written",
        "limitations": ["batch intercepts are nuisance effects, not WT states", "cell variance is not independent-replicate uncertainty", "point baseline only; no Gaussian likelihood or world-model claim"],
        "manifest": str(dataset.manifest_path),
        "manifestSha256": dataset.manifest_sha256,
        "static": str(DEFAULT_STATIC.resolve()),
        "staticSha256": EXPECTED_STATIC_SHA256,
        "coreSha256": sha256(CORE_PATH),
        "foldHelperSha256": sha256(FOLDS_PATH),
        "runnerSha256": sha256(Path(__file__)),
        "resourceProfileSha256": sha256(profile_path),
        "resourceProfile": profile,
        "staticCoverage": static_coverage(dataset, static),
        "controlDiagnostics": control_diagnostics(dataset),
        "biologicalFitExecuted": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "protocol-v2.json"
    _write_json(path, protocol)
    return protocol


def run_experiment(dataset: Dataset, static: StaticFeatures, output: Path) -> dict[str, object]:
    if (output / "FROZEN-BEFORE-VALIDATION.json").exists() or (output / "report.json").exists():
        raise FileExistsError("immutable experiment output already contains frozen/results state")
    protocol_path = output / "protocol-v2.json"
    protocol_hash = sha256(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["runnerSha256"] != sha256(Path(__file__)) or protocol["biologicalFitExecuted"]:
        raise ValueError("protocol does not pin this unfitted runner")
    started = time.perf_counter()
    cv, fitted, artifact_hashes = {}, {}, {}
    peak = _rss()
    for context in dataset.contexts:
        cv[context] = cross_validate_context(dataset, static, context, started)
        peak = max(peak, cv[context]["peakRssBytes"])
        _guard(started, peak)
        fitted[context] = fit_final_context(dataset, static, context, cv[context]["selected"])
        peak = max(peak, fitted[context]["peakRssBytes"])
        artifact_hashes.update(save_final_context(output, context, fitted[context], dataset.query_ids))
        _guard(started, peak)
    source_hashes = _persist_source(output)
    probe_path, probe_metadata = _write_target_free_probe(output, dataset, static, fitted)
    artifact_hashes[probe_path.name] = sha256(probe_path)
    artifact_hashes.update(source_hashes)
    artifact_manifest = {
        "schema": "slp.yeast-batch-ridge-artifact-manifest/v1",
        "protocolSha256": protocol_hash,
        "fileHashes": artifact_hashes,
        "probe": probe_metadata,
    }
    artifact_manifest_path = output / "artifact-manifest-before-validation.json"
    _write_json(artifact_manifest_path, artifact_manifest)
    verifier = output / "source/runner.py"
    completed = subprocess.run(
        [sys.executable, str(verifier), "--verify-artifact", str(output)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    replay = json.loads(completed.stdout)
    _guard(started, max(peak, _rss()))
    freeze = {
        "schema": "slp.yeast-batch-ridge-freeze-before-validation/v1",
        "protocolSha256": protocol_hash,
        "artifactHashes": artifact_hashes,
        "artifactManifestSha256": sha256(artifact_manifest_path),
        "targetFreeFreshProcessReplay": replay,
        "selected": {c: cv[c]["selected"] for c in dataset.contexts},
        "elapsedBeforeValidationSeconds": time.perf_counter() - started,
    }
    freeze_path = output / "FROZEN-BEFORE-VALIDATION.json"
    _write_json(freeze_path, freeze)
    validation = {context: evaluate_context(dataset, static, context, fitted[context], output) for context in dataset.contexts}
    peak = max(peak, *(validation[c]["peakRssBytes"] for c in dataset.contexts))
    elapsed = time.perf_counter() - started
    report = {
        "schema": "slp.yeast-raw-count-batch-ridge-development-diagnostic/v1",
        "label": "development diagnostic",
        "hypothesis": protocol["hypothesis"],
        "fixedRule": protocol["fixedRule"],
        "protocolSha256": protocol_hash,
        "freezeReceiptSha256": sha256(freeze_path),
        "crossValidation": {c: {k: v for k, v in cv[c].items() if k not in {"genes", "foldAssignments"}} for c in dataset.contexts},
        "validation": validation,
        "passesAllContexts": all(validation[c]["gate"]["passes"] for c in dataset.contexts),
        "controlDiagnostics": protocol["controlDiagnostics"],
        "staticCoverage": protocol["staticCoverage"],
        "runtimeSeconds": elapsed,
        "peakRssBytes": peak,
        "runtimeCapSeconds": 900.0,
        "rssCapBytes": MAX_RSS_BYTES,
        "completedWithinCaps": bool(elapsed <= 900 and peak < MAX_RSS_BYTES),
        "limitations": protocol["limitations"],
    }
    report_path = output / "report.json"
    _guard(started, peak)
    _write_json(report_path, report)
    return report


def run_profile(dataset: Dataset, static: StaticFeatures, output: Path, context: str | None = None) -> dict[str, object]:
    context = context or dataset.contexts[0]
    fitting = _role_genes(dataset, context, "train")
    mean, scale = feature_normalizer_for_genes(static, fitting)
    start, baseline_rss = time.perf_counter(), _rss()
    stats_batch, stats_pooled, peak = build_statistics(dataset, static, context, fitting, mean, scale)
    build_seconds = time.perf_counter() - start
    solve_start = time.perf_counter()
    batch_model = stats_batch.solve(10_000.0)
    peak = max(peak, _rss())
    pooled_model = stats_pooled.solve(10_000.0)
    peak = max(peak, _rss())
    solve_seconds = time.perf_counter() - solve_start
    largest_rows = max(len(s.actions) for s in dataset.shards)
    largest_sum_bytes = largest_rows * len(dataset.query_ids) * 8
    # 8 statistics builds: 3 folds + final in each of two contexts.  Each CV
    # fold solves 16 models; final fitting solves two.  This deliberately uses
    # the measured slowest solve rather than claiming a benchmark runtime.
    projected = 8 * build_seconds + 100 * (solve_seconds / 2) + 120.0
    report = {
        "schema": "slp.yeast-batch-ridge-resource-profile/v1",
        "createdBeforeBiologicalProtocolFreeze": True,
        "contextProfiled": context,
        "fittingGenes": len(fitting),
        "queries": len(dataset.query_ids),
        "featureDimensions": 577,
        "shards": len(dataset.shards),
        "streamingStatisticsBuildSeconds": build_seconds,
        "twoModelSolveSeconds": solve_seconds,
        "conservativeProjectedFullSeconds": projected,
        "baselineRssBytes": baseline_rss,
        "measuredPeakRssBytes": peak,
        "rssIncrementBytes": max(0, peak - baseline_rss),
        "largestSingleSumMemberBytes": largest_sum_bytes,
        "rssLimitBytes": MAX_RSS_BYTES,
        "runtimeLimitSeconds": 900.0,
        "passesRssGuard": bool(peak < MAX_RSS_BYTES),
        "passesProjectedRuntimeGuard": bool(projected < 900.0),
        "retainedMatrices": ["two sufficient-statistic states", "one current shard sum", "no population target stack"],
        "manifestSha256": dataset.manifest_sha256,
        "staticSha256": EXPECTED_STATIC_SHA256,
        "coreSha256": sha256(CORE_PATH),
        "foldHelperSha256": sha256(FOLDS_PATH),
        "profileModelsFinite": bool(np.isfinite(batch_model["coefficients"]).all() and np.isfinite(pooled_model["coefficients"]).all()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--static", type=Path, default=DEFAULT_STATIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--verify-artifact", type=Path)
    parser.add_argument("--skip-shard-hashes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_artifact is not None:
        result = verify_artifact(args.verify_artifact)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return
    if sha256(CORE_PATH) != EXPECTED_CORE_SHA256 or sha256(FOLDS_PATH) != EXPECTED_FOLDS_SHA256:
        raise SystemExit("pinned numerical source hash mismatch")
    dataset = load_metadata(args.manifest, verify_hashes=not args.skip_shard_hashes)
    static = load_static(args.static)
    if args.profile_only:
        report = run_profile(dataset, static, args.output / "resource-profile.json")
        print(json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False))
        return
    if args.prepare:
        protocol = prepare_protocol(dataset, static, args.output)
        print(json.dumps(_json_safe(protocol), indent=2, sort_keys=True, allow_nan=False))
        return
    if args.run:
        report = run_experiment(dataset, static, args.output)
        print(json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False))
        return
    raise SystemExit("choose --profile-only, --prepare, or --run")


if __name__ == "__main__":
    main()
