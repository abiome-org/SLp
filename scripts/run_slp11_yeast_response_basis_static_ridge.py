#!/usr/bin/env python3
"""Fitting-only yeast response-basis static-ridge development diagnostic.

Rank-32 response bases are estimated from independent A/B fitting-cell halves.
Static ridge is then fitted to batch-reference residuals in those coordinates.
Development validation is evaluated in the original 6,683-query space.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOMENTS = ROOT / "data/derived/slp11-yeast-atlas-counts/nadal-ribelles-raw-rna-development-v1/moments-manifest.json"
STATIC = ROOT / "data/derived/slp11-yeast-shared-static/current-sgd-strict-query-full-raw-actions-esm8m-complete-shared-go-v2/yeast-static-esm8m-shared-go-mf-cc-features.npz"
SPLIT_HALF = ROOT / "results/slp11-transition/yeast-rna-fitting-split-half-v1/split-half-sufficient-statistics.npz"
PRIOR_BASIS = ROOT / "results/slp11-transition/yeast-rna-fitting-crosscov-basis-rank32-v1"
BASELINE = ROOT / "results/slp11-transition/yeast-raw-count-batch-ridge-v1"
CORRECTED = ROOT / "results/slp11-transition/yeast-raw-count-batch-ridge-roundoff-scoring-v1/report.json"
RUNNER_SOURCE = ROOT / "scripts/run_slp11_yeast_batch_ridge.py"
BASIS_SOURCE = ROOT / "scripts/run_slp11_yeast_crosscov_response_basis.py"
CORE_SOURCE = ROOT / "modules/slp-1-1-batch-ridge-v1/batch_ridge.py"
FOLD_SOURCE = ROOT / "modules/slp-1-1-yeast-static-baseline-v1/static_baseline.py"
OUTPUT = ROOT / "results/slp11-transition/yeast-response-basis-static-ridge-rank32-v1"

PINS = {
    MOMENTS: "70a49ecaeb271fc72ecc93ede207c59a816e74d1ae3133bbf3a2803cce5d8eba",
    STATIC: "81cda9469380c9efa000a40b2cd5e816a1d397ce777288fa53b0bcf26a55dc25",
    SPLIT_HALF: "dab8b4bbf21bd0a584e77f5fd69d82df41e366ad6d034e9ba7be62896b588689",
    CORRECTED: "291b8b34d9b03b0bedb6a40723cbab07b6fd2c094dbffdb5a5a644141454c128",
    RUNNER_SOURCE: "762761c5da905833a78d336fba75cd0c40d65ad28c04c5f2f054c320f5eaef33",
    BASIS_SOURCE: "05267977e47dc2af661ef3937456b982fb2ca5384780e6a632ea2132d56a113d",
    CORE_SOURCE: "5d897f45ca1318ffe1d447cbafbb1732d0e428efa5f6a7b3dcfe4c32841c18c8",
    FOLD_SOURCE: "88e51be7dfbb175844f6d2f6c884d482129f38b24af15b3d4528bff82088e57f",
}
BASELINE_REPORT_SHA256 = "e15c9b14dc37b4eae01ef1e5bc847860a2d39273c76c930cb12030e622488824"
SPLIT_HALF_SCHEMA = "slp.yeast-fitting-split-half-sufficient-statistics/v1"
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0)
ARMS = ("pca32", "positiveCrossCovariance32")
RANK = 32
SEED = 731
MAX_SECONDS = 900.0
MAX_RSS = 4 * 1024**3


class DiagnosticError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def write_json(path: Path, value: object) -> None:
    Path(path).write_bytes(canonical_json(value))


def deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, member.getvalue(), compresslevel=9)
    path.write_bytes(output.getvalue())


def load_python(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DiagnosticError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rss() -> int:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return 0


def guard(started: float, peak: int) -> None:
    if time.perf_counter() - started > MAX_SECONDS:
        raise RuntimeError("900 second runtime cap exceeded")
    if peak >= MAX_RSS:
        raise MemoryError("4 GiB RSS cap exceeded")


def validate_pins() -> dict[str, str]:
    found = {}
    for path, expected in PINS.items():
        actual = sha256_file(path)
        if actual != expected:
            raise DiagnosticError(f"pinned input mismatch: {path}")
        found[str(path.relative_to(ROOT)).replace("\\", "/")] = actual
    baseline_report = BASELINE / "report.json"
    if sha256_file(baseline_report) != BASELINE_REPORT_SHA256:
        raise DiagnosticError("baseline report hash mismatch")
    found[str(baseline_report.relative_to(ROOT)).replace("\\", "/")] = BASELINE_REPORT_SHA256
    return found


def load_split_half() -> dict[str, np.ndarray]:
    with np.load(SPLIT_HALF, allow_pickle=False) as z:
        values = {key: np.asarray(z[key]) for key in z.files}
    if str(values["schema"].item()) != SPLIT_HALF_SCHEMA:
        raise DiagnosticError("split-half schema mismatch")
    return values


def split_profiles(values: dict[str, np.ndarray], context_index: int, allowed: set[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = values["gene_ids"].astype(str)
    counts = np.asarray(values["half_num_cells"][context_index], dtype=np.int64)
    observed = (counts[0] > 0) & (counts[1] > 0) & np.isin(ids, sorted(allowed))
    sums = np.asarray(values["half_sum"][context_index][:, observed, :], dtype=np.float64)
    selected = counts[:, observed]
    a = sums[0] / selected[0, :, None]
    b = sums[1] / selected[1, :, None]
    return ids[observed], a, b


def fit_basis(values: dict[str, np.ndarray], context_index: int, allowed: set[str], basis_module, *, rank: int = RANK) -> dict[str, np.ndarray]:
    ids, a, b = split_profiles(values, context_index, allowed)
    if len(ids) <= rank:
        raise DiagnosticError("too few fitting-only split-half genes for basis")
    fitted = basis_module.fit_bases(a, b, rank=rank, seed=SEED)
    fitted["basis_gene_ids"] = ids
    return fitted


def projection_reconstruction(residual: np.ndarray, basis: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residual = np.asarray(residual, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    scores = residual @ basis
    return scores, reference + scores @ basis.T


def batch_reference(dataset, context: str, allowed_genes: set[str], runner) -> dict[str, np.ndarray]:
    totals = runner._gene_totals(dataset, context, "train", allowed_genes)
    sums: dict[str, np.ndarray] = {}
    weights: dict[str, float] = {}
    for shard in dataset.shards:
        if shard.context != context:
            continue
        mask = (shard.roles == "train") & np.isin(shard.actions, sorted(allowed_genes))
        if not mask.any():
            continue
        with np.load(shard.path, allow_pickle=False) as z:
            block = np.asarray(z["sum"], dtype=np.float64)[mask]
        actions = shard.actions[mask]
        cells = shard.num_cells[mask].astype(np.float64)
        y = block / cells[:, None]
        w = np.asarray([n / totals[g] for g, n in zip(actions, cells, strict=True)])
        if shard.batch not in sums:
            sums[shard.batch] = np.zeros(y.shape[1], dtype=np.float64)
            weights[shard.batch] = 0.0
        sums[shard.batch] += w @ y
        weights[shard.batch] += float(w.sum())
        del block, y
    labels = sorted(sums)
    means = np.stack([sums[label] / weights[label] for label in labels])
    return {"batch_ids": np.asarray(labels), "batch_only_means": means}


def batch_row(reference: dict[str, np.ndarray], batch: str) -> np.ndarray:
    where = np.flatnonzero(reference["batch_ids"].astype(str) == batch)
    if len(where) != 1:
        raise DiagnosticError(f"no unique reference for {batch}")
    return reference["batch_only_means"][where[0]]


def build_latent_statistics(dataset, static, context: str, allowed_genes: set[str], feature_mean: np.ndarray, feature_scale: np.ndarray, basis: np.ndarray, reference: dict[str, np.ndarray], runner, core):
    totals = runner._gene_totals(dataset, context, "train", allowed_genes)
    stats = core.BatchRidgeStatistics(577, basis.shape[1])
    peak = rss()
    for shard in dataset.shards:
        if shard.context != context:
            continue
        mask = (shard.roles == "train") & np.isin(shard.actions, sorted(allowed_genes))
        if not mask.any():
            continue
        with np.load(shard.path, allow_pickle=False) as z:
            block = np.asarray(z["sum"], dtype=np.float64)[mask]
        actions = shard.actions[mask]
        cells = shard.num_cells[mask].astype(np.float64)
        y = block / cells[:, None]
        latent = (y - batch_row(reference, shard.batch)) @ basis
        x = (runner._gene_features(static, actions) - feature_mean) / feature_scale
        w = np.asarray([n / totals[g] for g, n in zip(actions, cells, strict=True)])
        stats.update(shard.batch, x, latent, w)
        peak = max(peak, rss())
        del block, y, latent, x
        gc.collect()
    return stats, peak


def mean_model(stats, runner) -> dict[str, np.ndarray]:
    return runner.batch_only_model(stats)


def candidate_model(stats, candidate: str, runner) -> dict[str, np.ndarray]:
    return mean_model(stats, runner) if candidate == "mean" else stats.solve(float(candidate))


def predict_genes(model: dict[str, np.ndarray], reference: dict[str, np.ndarray], static, genes: np.ndarray, batch_weights: dict[str, dict[str, float]], feature_mean: np.ndarray, feature_scale: np.ndarray, basis: np.ndarray, runner) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = (runner._gene_features(static, genes) - feature_mean) / feature_scale
    latent = x @ model["coefficients"]
    raw_reference = np.zeros((len(genes), basis.shape[0]), dtype=np.float64)
    for i, gene in enumerate(genes):
        for batch, weight in batch_weights[gene].items():
            where = np.flatnonzero(model["batch_ids"].astype(str) == batch)
            if len(where) != 1:
                raise DiagnosticError(f"latent model lacks batch {batch}")
            latent[i] += weight * model["batch_offsets"][where[0]]
            raw_reference[i] += weight * batch_row(reference, batch)
    residual = latent @ basis.T
    return raw_reference + residual, residual, latent


def response_basis_for_arm(fitted: dict[str, np.ndarray], arm: str) -> np.ndarray:
    return fitted["pca_basis"] if arm == "pca32" else fitted["cross_basis"]


def cv_context(dataset, static, split, context: str, context_index: int, runner, basis_module, core, started: float) -> dict[str, object]:
    genes = runner._role_genes(dataset, context, "train")
    assignments = runner._folds(genes)
    candidates = [str(value) for value in ALPHAS] + ["mean"]
    errors = {arm: {candidate: 0.0 for candidate in candidates} for arm in ARMS}
    elements = 0
    folds = []
    peak = rss()
    for fold in range(3):
        fit_genes = genes[assignments != fold]
        held_genes = genes[assignments == fold]
        fit_set = set(fit_genes)
        fitted_basis = fit_basis(split, context_index, fit_set, basis_module)
        feature_mean, feature_scale = runner.feature_normalizer_for_genes(static, fit_genes)
        reference = batch_reference(dataset, context, fit_set, runner)
        held_ids, truth, batch_weights, truth_peak = runner.stream_pooled_truth(dataset, context, "train", held_genes)
        peak = max(peak, truth_peak)
        fold_report = {"fold": fold, "fittingMomentGenes": len(fit_genes), "heldMomentGenes": len(held_ids), "basisFittingGenes": len(fitted_basis["basis_gene_ids"]), "crossPositiveRank": int(fitted_basis["cross_basis"].shape[1])}
        elements += truth.size
        for arm in ARMS:
            basis = response_basis_for_arm(fitted_basis, arm)
            stats, stats_peak = build_latent_statistics(dataset, static, context, fit_set, feature_mean, feature_scale, basis, reference, runner, core)
            peak = max(peak, stats_peak)
            for candidate in candidates:
                model = candidate_model(stats, candidate, runner)
                prediction, _, _ = predict_genes(model, reference, static, held_ids, batch_weights, feature_mean, feature_scale, basis, runner)
                errors[arm][candidate] += float(np.square(truth - prediction).sum())
            del stats
        folds.append(fold_report)
        del truth, fitted_basis, reference
        gc.collect()
        guard(started, peak)
    scores = {arm: {candidate: errors[arm][candidate] / elements for candidate in candidates} for arm in ARMS}
    selected = {arm: min(candidates, key=lambda c: (scores[arm][c], candidates.index(c))) for arm in ARMS}
    return {"genes": genes, "foldAssignments": assignments, "folds": folds, "candidateRawMse": scores, "selectedAlpha": selected, "peakRssBytes": peak}


def final_context(dataset, static, split, context: str, context_index: int, selected: dict[str, str], runner, basis_module, core, started: float):
    genes = runner._role_genes(dataset, context, "train")
    fit_set = set(genes)
    fitted_basis = fit_basis(split, context_index, fit_set, basis_module)
    feature_mean, feature_scale = runner.feature_normalizer_for_genes(static, genes)
    reference = batch_reference(dataset, context, fit_set, runner)
    arms = {}
    peak = rss()
    for arm in ARMS:
        basis = response_basis_for_arm(fitted_basis, arm)
        stats, stats_peak = build_latent_statistics(dataset, static, context, fit_set, feature_mean, feature_scale, basis, reference, runner, core)
        peak = max(peak, stats_peak)
        arms[arm] = {"basis": basis, "model": candidate_model(stats, selected[arm], runner)}
        del stats
    guard(started, peak)
    return {"fittingGenes": genes, "featureMean": feature_mean, "featureScale": feature_scale, "reference": reference, "basisFit": fitted_basis, "arms": arms, "peakRssBytes": peak}


def save_final(output: Path, context: str, fitted: dict[str, object], query_ids: np.ndarray) -> dict[str, str]:
    slug = context.lower()
    hashes = {}
    reference_path = output / f"{slug}-reference.npz"
    deterministic_npz(reference_path, {"batch_ids": fitted["reference"]["batch_ids"], "batch_only_means": fitted["reference"]["batch_only_means"], "feature_mean": fitted["featureMean"], "feature_scale": fitted["featureScale"], "fitting_action_ids": fitted["fittingGenes"], "query_ids": query_ids})
    hashes[reference_path.name] = sha256_file(reference_path)
    for arm in ARMS:
        item = fitted["arms"][arm]
        model = item["model"]
        path = output / f"{slug}-{arm}-model.npz"
        deterministic_npz(path, {"alpha": model["alpha"], "basis": item["basis"], "batch_ids": model["batch_ids"], "batch_offsets": model["batch_offsets"], "coefficients": model["coefficients"], "query_ids": query_ids})
        hashes[path.name] = sha256_file(path)
    basis_path = output / f"{slug}-all-fitting-bases.npz"
    deterministic_npz(basis_path, {key: value for key, value in fitted["basisFit"].items() if isinstance(value, np.ndarray)})
    hashes[basis_path.name] = sha256_file(basis_path)
    return hashes


def baseline_validation(context: str) -> dict[str, np.ndarray]:
    path = BASELINE / f"{context.lower()}-validation-gene-predictions.npz"
    with np.load(path, allow_pickle=False) as z:
        return {key: np.asarray(z[key]) for key in z.files}


def evaluate_context(dataset, static, context: str, fitted: dict[str, object], output: Path, runner) -> dict[str, object]:
    validation = runner._role_genes(dataset, context, "validation")
    genes, truth, batch_weights, peak = runner.stream_pooled_truth(dataset, context, "validation", validation)
    if len(genes) != 346:
        raise DiagnosticError(f"{context} must retain every 346 validation genes")
    truth_residual = runner.residualize_truth(truth, genes, batch_weights, fitted["reference"])
    prior = baseline_validation(context)
    if not np.array_equal(genes, prior["action_ids"].astype(str)) or not np.array_equal(dataset.query_ids, prior["query_ids"].astype(str)) or not np.array_equal(truth, prior["truth"]):
        raise DiagnosticError("validation identities or raw truth differ from frozen batch-ridge comparator")
    baseline_mse = float(np.mean(np.square(truth - prior["prediction_batch"])))
    results = {}
    arrays = {"action_ids": genes, "query_ids": dataset.query_ids, "truth": truth, "truth_batch_reference_subtracted": truth_residual, "baseline_batch_prediction": prior["prediction_batch"]}
    for arm in ARMS:
        item = fitted["arms"][arm]
        raw, residual, latent = predict_genes(item["model"], fitted["reference"], static, genes, batch_weights, fitted["featureMean"], fitted["featureScale"], item["basis"], runner)
        raw_metrics = runner.metrics(truth, raw)
        residual_metrics = runner.metrics(truth_residual, residual)
        projected_truth_residual = (truth_residual @ item["basis"]) @ item["basis"].T
        projected_mse = float(np.mean(np.square(projected_truth_residual - residual)))
        results[arm] = {"raw": raw_metrics, "batchReferenceSubtracted": residual_metrics, "projectedResidualMse": projected_mse, "rawMseImprovementFractionVsBatchRidge": float(1.0 - raw_metrics["geneProfileMse"] / baseline_mse)}
        arrays[f"prediction_{arm}"] = raw
        arrays[f"prediction_{arm}_batch_reference_subtracted"] = residual
        arrays[f"prediction_{arm}_latent"] = latent
    prediction_path = output / f"{context.lower()}-validation-gene-predictions.npz"
    deterministic_npz(prediction_path, arrays)
    return {"validationGenes": len(genes), "baselineBatchRawMse": baseline_mse, "metrics": results, "predictionPath": str(prediction_path), "predictionSha256": sha256_file(prediction_path), "peakRssBytes": peak}


def protocol(input_hashes: dict[str, str]) -> dict[str, object]:
    return {
        "schema": "slp.yeast-response-basis-static-ridge-protocol/v1",
        "status": "frozen-before-fitting-and-development-validation",
        "hypothesis": "Denoising the fitting response space with a prespecified rank-32 basis improves held-gene raw-profile prediction in both yeast environments.",
        "advancementRule": "In each of Control and NaCl, an arm must reduce full 6683-query raw validation gene-profile MSE by at least 1% versus the frozen batch ridge and achieve independently query-centered Pearson at least 0.10 after subtracting the identical fitting-only batch reference from truth and prediction. No aggregate rescues an environment failure.",
        "arms": {"pca32": "rank-32 PCA of separately centered fitting A/B half means", "positiveCrossCovariance32": "rank-32 largest positive eigenspace of symmetric fitting A/B cross-covariance"},
        "basisEstimation": "For each inner fold and environment, use only split-half genes assigned to the two fitting folds. Center A and B separately across those genes. Final bases use every fitting-only split-half gene in that environment. No validation profiles enter a basis.",
        "projection": "The learned basis supplies directions only. Each raw moment population first subtracts its fitting-fold, exact-batch mean profile. The resulting zero-origin residual is projected as z=(y-batch_mean)@P; no split-half profile mean is added to a response target.",
        "ridge": "For each arm, fit z = standardized_static577@W + unpenalized exact-batch latent offset. Static normalization uses unique moment fitting genes only. Population weights are cell fractions within gene, so each fitting gene totals weight one.",
        "reconstruction": "For a held gene, mix fitting batch-reference profiles and latent batch offsets using that gene's observed cell fractions; raw_prediction=mixed_batch_reference + latent_prediction@P.T.",
        "selection": {"folds": 3, "domain": "slp11-yeast-static-baseline-inner-v1", "seed": 731, "candidates": [*ALPHAS, "mean"], "criterion": "full raw query-space equal-gene MSE over held fitting genes"},
        "evaluation": "Every 346 development-validation genes in each environment; primary full raw query-space MSE and batch-reference-subtracted independently query-centered correlation; projected residual MSE secondary.",
        "accessBoundary": {"fittingMoments": True, "fittingSplitHalfStatistics": True, "developmentValidationMoments": True, "heldTestOrSL": False},
        "compute": {"cpuThreads": 2, "maximumSeconds": MAX_SECONDS, "maximumMemoryGiB": 4},
        "snapshots": input_hashes,
        "limitations": ["A split-half response basis can retain shared batch and clone effects.", "This is a point predictor diagnostic, not a biological dynamics or uncertainty model."],
    }


def profile(dataset, static, split, runner, basis_module, core) -> dict[str, object]:
    started = time.perf_counter()
    context = "Control"
    genes = runner._role_genes(dataset, context, "train")
    fold_ids = runner._folds(genes)
    fitting = genes[fold_ids != 0]
    fitted_basis = fit_basis(split, 0, set(fitting), basis_module)
    mean, scale = runner.feature_normalizer_for_genes(static, fitting)
    reference = batch_reference(dataset, context, set(fitting), runner)
    stats, peak = build_latent_statistics(dataset, static, context, set(fitting), mean, scale, fitted_basis["pca_basis"], reference, runner, core)
    elapsed = time.perf_counter() - started
    result = {"schema": "slp.yeast-response-basis-static-ridge-profile/v1", "oneFoldOneArmSeconds": elapsed, "projectedFullSeconds": elapsed * 14.0, "peakRssBytes": peak, "passesTime": elapsed * 14.0 < MAX_SECONDS, "passesMemory": peak < MAX_RSS, "fittingGenes": len(fitting), "basisGenes": len(fitted_basis["basis_gene_ids"]), "queries": len(dataset.query_ids)}
    del stats
    return result


def persist_sources(output: Path) -> dict[str, str]:
    source = output / "source"
    source.mkdir(parents=True, exist_ok=True)
    copies = {"runner.py": Path(__file__), "batch_ridge_runner.py": RUNNER_SOURCE, "batch_ridge.py": CORE_SOURCE, "response_basis.py": BASIS_SOURCE, "static_baseline.py": FOLD_SOURCE}
    hashes = {}
    for name, path in copies.items():
        dest = source / name
        shutil.copy2(path, dest)
        hashes[f"source/{name}"] = sha256_file(dest)
    return hashes


def run(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    input_hashes = validate_pins()
    runner = load_python(RUNNER_SOURCE, "_slp11_response_ridge_batch_runner")
    basis_module = load_python(BASIS_SOURCE, "_slp11_response_ridge_basis")
    core = load_python(CORE_SOURCE, "_slp11_response_ridge_core")
    dataset = runner.load_metadata(MOMENTS)
    static = runner.load_static(STATIC)
    split = load_split_half()
    profile_result = profile(dataset, static, split, runner, basis_module, core)
    write_json(output / "resource-profile.json", profile_result)
    if not profile_result["passesTime"] or not profile_result["passesMemory"]:
        raise RuntimeError("profile rejects full diagnostic")
    fixed_protocol = protocol(input_hashes)
    write_json(output / "FROZEN-BEFORE-FITTING.json", fixed_protocol)
    protocol_hash = sha256_file(output / "FROZEN-BEFORE-FITTING.json")
    started = time.perf_counter()
    cv = {}
    fitted = {}
    artifact_hashes = {}
    for context_index, context in enumerate(dataset.contexts):
        cv[context] = cv_context(dataset, static, split, context, context_index, runner, basis_module, core, started)
        fitted[context] = final_context(dataset, static, split, context, context_index, cv[context]["selectedAlpha"], runner, basis_module, core, started)
        artifact_hashes.update(save_final(output, context, fitted[context], dataset.query_ids))
    freeze = {"schema": "slp.yeast-response-basis-static-ridge-prevalidation/v1", "protocolSha256": protocol_hash, "selectedAlpha": {context: cv[context]["selectedAlpha"] for context in dataset.contexts}, "modelHashes": artifact_hashes}
    write_json(output / "FROZEN-BEFORE-VALIDATION.json", freeze)
    validation = {context: evaluate_context(dataset, static, context, fitted[context], output, runner) for context in dataset.contexts}
    baseline_report = json.loads(CORRECTED.read_text(encoding="utf-8"))
    gates = {}
    for context in dataset.contexts:
        gates[context] = {}
        for arm in ARMS:
            item = validation[context]["metrics"][arm]
            r = item["batchReferenceSubtracted"]["independentlyQueryCenteredPearson"]
            gates[context][arm] = {"rawMseAtLeast1PctBetterThanBatchRidge": bool(item["raw"]["geneProfileMse"] <= 0.99 * validation[context]["baselineBatchRawMse"]), "residualCenteredRAtLeastPoint1": bool(r is not None and r >= 0.10)}
            gates[context][arm]["passes"] = all(gates[context][arm].values())
    arm_passes = {arm: all(gates[c][arm]["passes"] for c in dataset.contexts) for arm in ARMS}
    runtime = time.perf_counter() - started
    source_hashes = persist_sources(output)
    report = {"schema": "slp.yeast-response-basis-static-ridge-development-diagnostic/v1", "hypothesis": fixed_protocol["hypothesis"], "fixedRule": fixed_protocol["advancementRule"], "protocolSha256": protocol_hash, "crossValidation": {c: {k: v for k, v in cv[c].items() if k not in {"genes", "foldAssignments"}} for c in dataset.contexts}, "validation": validation, "gates": gates, "armPassesBothEnvironments": arm_passes, "decision": "advance" if any(arm_passes.values()) else "fail", "baselineCorrectedScoring": {"reportSha256": PINS[CORRECTED], "metrics": {c: baseline_report["validation"][c]["metrics"]["batch"] for c in dataset.contexts}}, "runtimeSeconds": runtime, "peakRssBytes": max([rss(), profile_result["peakRssBytes"]] + [int(cv[c]["peakRssBytes"]) for c in dataset.contexts]), "completedWithinCaps": bool(runtime < MAX_SECONDS and rss() < MAX_RSS), "artifactHashes": {**artifact_hashes, **source_hashes}, "interpretation": "Fitting-only response-space denoising with development held-gene evaluation; not test, SL, or causal evidence."}
    write_json(output / "report.json", report)
    report_hash = sha256_file(output / "report.json")
    receipt = {"schema": "slp.yeast-response-basis-static-ridge-receipt/v1", "reportSha256": report_hash, "decision": report["decision"], "runtimeSeconds": runtime}
    write_json(output / "execution-receipt.json", receipt)
    return report


def verify(output: Path) -> dict[str, object]:
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    for name, expected in report["artifactHashes"].items():
        if sha256_file(output / name) != expected:
            raise DiagnosticError(f"artifact mismatch: {name}")
    for context in ("control", "nacl"):
        with np.load(output / f"{context}-reference.npz", allow_pickle=False) as ref:
            query_ids = ref["query_ids"].astype(str)
        for arm in ARMS:
            with np.load(output / f"{context}-{arm}-model.npz", allow_pickle=False) as model:
                if not np.array_equal(query_ids, model["query_ids"].astype(str)) or model["basis"].shape != (6683, 32):
                    raise DiagnosticError("model/query replay contract failed")
    return {"schema": "slp.yeast-response-basis-static-ridge-replay/v1", "passes": True, "reportSha256": sha256_file(output / "report.json")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = verify(args.output) if args.verify else run(args.output) if args.run else None
    if result is None:
        raise SystemExit("choose --run or --verify")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
