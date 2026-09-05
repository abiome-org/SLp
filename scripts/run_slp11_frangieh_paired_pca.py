#!/usr/bin/env python3
"""Fit the frozen paired-cell PCA128 and held-gene static latent ridge."""

from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "2"

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "modules/slp-1-1-paired-pca-v1/paired_pca.py"
SHARDS = ROOT / "data/derived/slp11-frangieh/paired-singlecell-train-control-v1"
REFERENCE = ROOT / "results/slp11-transition/frangieh-paired-state-physical1156-seed731-v2/reference.npz"
STATIC = ROOT / "data/derived/slp11-frangieh-static/ensembl116-goa2022-fixed-neighbor-v1/frangieh-extended-static-esm-go-fixed-physical-features.npz"
DEVELOPMENT = ROOT / "data/derived/slp11-frangieh/paired-development-v1/development.npz"
BASELINE = ROOT / "results/slp11-transition/frangieh-paired-state-vs-static-scoring-v1/report.json"
STATIC_PREDICTIONS = ROOT / "results/slp11-transition/frangieh-specieswide-physical-ridge-v1/predictions.npz"
PRIOR_PREDICTIONS = ROOT / "results/slp11-transition/frangieh-paired-state-physical1156-seed731-v2/predictions.npz"
METRICS = ROOT / "results/slp11-transition/frangieh-cell-state-ae-latent-ridge-seed731-v1/source/frangieh_basal_ridge.py"
OUTPUT = ROOT / "results/slp11-transition/frangieh-paired-pca128-latent-ridge-seed731-v1"
CONTEXTS = ("Co-culture", "Control", "IFNγ")
HASHES = {
    "manifest": "e791b5cf35da96fa71951a4a240ed58b53e278d3c57e44066680abd3f386a9c7",
    "reference": "8b82e4781b73a721f995dd218ef341ea8324b87d3c9189bfe40644d436800e73",
    "static": "347fd1bf87d8fc3d0b447676082b4bcb64f021c9f12c7df4d1754dc262b2bf72",
    "development": "4bbb1eec9ede66211f1316b2841bb0037032ef975cd6c92d34aba0adb5fed744",
    "baseline": "d0c577e093198e9060a582cc5852b0db61246daa5772ae0c1e8451addc584b90",
    "static_predictions": "1e342a75e4a1cc67d6d0a6e3c1e4acefb95d7a51fad7a1bf47fcbff978c7abfe",
    "prior_predictions": "36ebe74677f7bb75e467bf8f225cc313417590772de356f98470d32a5e26b50b",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    def clean(item: object) -> object:
        if isinstance(item, dict):
            return {str(key): clean(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(entry) for entry in item]
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    path.write_text(
        json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_shard(path: Path) -> dict[str, np.ndarray | sparse.csr_matrix]:
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name] for name in archive.files}
    result["rna"] = sparse.csr_matrix(
        (result["rna_data"], result["rna_indices"], result["rna_indptr"]),
        shape=tuple(result["rna_shape"]),
    )
    return result


def verify_inputs() -> tuple[
    dict[str, object], dict[str, np.ndarray], np.ndarray, np.ndarray,
]:
    paths = {
        "manifest": SHARDS / "manifest.json", "reference": REFERENCE,
        "static": STATIC, "development": DEVELOPMENT, "baseline": BASELINE,
        "static_predictions": STATIC_PREDICTIONS, "prior_predictions": PRIOR_PREDICTIONS,
    }
    for name, path in paths.items():
        if sha256(path) != HASHES[name]:
            raise ValueError(f"input hash drift: {name}")
    manifest = json.loads((SHARDS / "manifest.json").read_text(encoding="utf-8"))
    if manifest["counts"] != {
        "cells": 103862, "excluded_validation_cells": 19606,
        "protein_molecular_channels": 20, "reconstruction_train_cells": 93397,
        "reconstruction_validation_cells": 10465, "rna_nnz": 320301640,
        "rna_queries": 18063, "shards": 51, "source_train_cells": 64515,
        "verified_control_cells": 39347,
    }:
        raise ValueError("single-cell population drift")
    expected = 0
    training_actions = []
    for item in manifest["shards"]:
        if item["row_start"] != expected or sha256(SHARDS / item["path"]) != item["sha256"]:
            raise ValueError("single-cell shard roster/hash drift")
        expected = item["row_stop"]
        with np.load(SHARDS / item["path"], allow_pickle=False) as shard:
            if not set(shard["source_split"].astype(str)) <= {"train", "control"}:
                raise ValueError("held source split entered PCA shards")
            training_actions.append(shard["action_ids"].astype(str))
    with np.load(REFERENCE, allow_pickle=False) as archive:
        reference = {name: archive[name] for name in archive.files}
    if not np.array_equal(reference["context_names"].astype(str), CONTEXTS):
        raise ValueError("reference context order drift")
    with np.load(DEVELOPMENT, allow_pickle=False) as development:
        held_actions = development["action_ids"][development["split_validation"]].astype(str)
        if len(development["split_validation"]) != 403:
            raise ValueError("held validation roster drift")
    return manifest, reference, np.concatenate(training_actions), held_actions


def batches(manifest: dict[str, object], split: str):
    for item in manifest["shards"]:
        shard = load_shard(SHARDS / item["path"])
        rows = np.asarray(shard["reconstruction_split"] == split)
        if rows.any():
            yield shard["rna"][rows], np.asarray(shard["protein_values"])[rows]


def reconstruction_metrics(model, manifest: dict[str, object]) -> dict[str, object]:
    totals = defaultdict(float)
    cells = 0
    for rna, protein in batches(manifest, "validation"):
        for start in range(0, len(protein), 128):
            raw_rna = rna[start:start + 128]
            raw_protein = protein[start:start + 128].astype(np.float64)
            scores = model.encode(raw_rna, raw_protein)
            prediction_rna, prediction_protein = model.decode_raw(scores)
            target_rna = raw_rna.toarray().astype(np.float64)
            for head, prediction, target, mean, sd in (
                (
                    "rna", prediction_rna, target_rna,
                    model.stats.rna_mean, model.stats.rna_sd,
                ),
                (
                    "protein", prediction_protein, raw_protein,
                    model.stats.protein_mean, model.stats.protein_sd,
                ),
            ):
                totals[f"{head}_raw"] += float(np.square(prediction - target).sum())
                totals[f"{head}_mean_raw"] += float(np.square(target - mean).sum())
                totals[f"{head}_standard"] += float(
                    np.square((prediction - target) / sd).sum()
                )
                totals[f"{head}_mean_standard"] += float(
                    np.square((target - mean) / sd).sum()
                )
            cells += len(raw_protein)
    if cells != 10465:
        raise ValueError("reconstruction validation count drift")
    report = {"cells": cells, "heads": {}}
    for head, queries in (("rna", 18063), ("protein", 20)):
        values = {
            "rawMse": totals[f"{head}_raw"] / (cells * queries),
            "trainingMeanRawMse": totals[f"{head}_mean_raw"] / (cells * queries),
            "standardizedMse": totals[f"{head}_standard"] / (cells * queries),
            "trainingMeanStandardizedMse": (
                totals[f"{head}_mean_standard"] / (cells * queries)
            ),
        }
        values["rawFractionalImprovement"] = 1 - values["rawMse"] / values["trainingMeanRawMse"]
        values["standardizedFractionalImprovement"] = (
            1 - values["standardizedMse"] / values["trainingMeanStandardizedMse"]
        )
        report["heads"][head] = values
    report["balancedStandardizedMse"] = 0.5 * sum(
        report["heads"][head]["standardizedMse"] for head in ("rna", "protein")
    )
    return report


def aggregate_gene_scores(model, manifest: dict[str, object]):
    guide_sum: dict[tuple[str, str, str], np.ndarray] = {}
    guide_count = defaultdict(int)
    for item in manifest["shards"]:
        shard = load_shard(SHARDS / item["path"])
        scores = model.encode(shard["rna"], np.asarray(shard["protein_values"]))
        for gene, context, guide, score in zip(
            shard["action_ids"], shard["context_ids"], shard["target_guide_sets"],
            scores, strict=True,
        ):
            gene = str(gene)
            if not gene:
                continue
            key = (gene, str(context), str(guide))
            guide_sum[key] = guide_sum.get(key, np.zeros(scores.shape[1])) + score
            guide_count[key] += 1
    gene_guides = defaultdict(list)
    for (gene, context, _guide), total in guide_sum.items():
        gene_guides[(gene, context)].append(total / guide_count[(gene, context, _guide)])
    return {
        key: np.mean(values, axis=0) for key, values in gene_guides.items()
    }, len(guide_sum)


def fit_forecast_artifact(core, pca, manifest, reference):
    gene_scores, guides = aggregate_gene_scores(pca, manifest)
    control_scores = pca.encode(
        sparse.csr_matrix(reference["rna_controls"]), reference["protein_controls"],
    )
    with np.load(STATIC, allow_pickle=False) as item:
        static_ids = item["entity_id"].astype(str)
        static_values = item["feature_values"].astype(np.float64)
    static_lookup = dict(zip(static_ids, static_values, strict=True))

    def normalize(genes):
        raw = np.stack([static_lookup[gene] for gene in genes])
        return np.clip(
            (raw - reference["feature_mean"]) / reference["feature_scale"],
            -float(reference["feature_clip"]), float(reference["feature_clip"]),
        )

    coefficients, intercepts, counts = [], [], {}
    for context_index, context in enumerate(CONTEXTS):
        genes = sorted(gene for gene, condition in gene_scores if condition == context)
        if len(genes) != 151:
            raise ValueError(f"fitting gene count drift in {context}")
        target = np.stack([gene_scores[(gene, context)] for gene in genes]) - control_scores[context_index]
        coefficient, intercept = core.fit_latent_ridge(normalize(genes), target, alpha=10000.0)
        coefficients.append(coefficient)
        intercepts.append(intercept)
        counts[context] = len(genes)
    ridge = core.LatentRidge(
        reference["feature_mean"], reference["feature_scale"],
        float(reference["feature_clip"]), np.stack(coefficients),
        np.stack(intercepts), 10000.0,
    )
    artifact = core.PcaForecastArtifact(
        pca, ridge, reference["rna_controls"], reference["protein_controls"],
        np.asarray(CONTEXTS),
    )
    return artifact, static_lookup, counts, guides


def collapse(helper, data, indices, head):
    return helper.collapse_gene_profiles(
        data["action_ids"][indices], data["context_ids"][indices],
        data[f"{head}_targets"][indices],
    )


def forecast_and_score(core, artifact_path: Path, reference, static_lookup):
    artifact = core.PcaForecastArtifact.load(artifact_path)
    # Quantitative held-gene outcomes are opened only after PCA and ridge are frozen.
    with np.load(DEVELOPMENT, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    helper = load_source(METRICS, "slp11_paired_pca_metrics")
    indices = data["split_validation"]
    action, context, rna_truth, rna_guides = collapse(helper, data, indices, "rna")
    pa, pc, protein_truth, protein_guides = collapse(helper, data, indices, "protein")
    if not np.array_equal(action, pa) or not np.array_equal(context, pc):
        raise ValueError("held RNA/protein gene roster differs")
    if not np.array_equal(rna_guides, protein_guides):
        raise ValueError("held RNA/protein guide counts differ")
    raw_features = np.stack([static_lookup[gene] for gene in action])
    context_index = np.asarray([CONTEXTS.index(item) for item in context], dtype=np.int64)
    rna_prediction, protein_prediction, latent_delta = artifact.forecast(
        raw_features, context_index,
    )
    reload_rna, reload_protein, reload_delta = core.PcaForecastArtifact.load(
        artifact_path,
    ).forecast(raw_features.copy(), context_index.copy())
    reload_drift = max(
        float(np.max(np.abs(left - right)))
        for left, right in (
            (rna_prediction, reload_rna), (protein_prediction, reload_protein),
            (latent_delta, reload_delta),
        )
    )
    with np.load(STATIC_PREDICTIONS, allow_pickle=False) as static_prediction, np.load(
        PRIOR_PREDICTIONS, allow_pickle=False,
    ) as prior:
        prior_lookup = {
            (str(gene), CONTEXTS[int(condition)]): row
            for row, (gene, condition) in enumerate(
                zip(prior["action_ids"], prior["context_index"], strict=True),
            )
        }
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        contexts = {}
        all_pass = True
        for context_name in CONTEXTS:
            key = context_name.replace("-", "_").replace("γ", "gamma")
            rows = context == context_name
            genes = action[rows]
            contexts[context_name] = {"heads": {}}
            for head, truth, prediction, train_scale in (
                ("rna", rna_truth[rows], rna_prediction[rows], artifact.pca.stats.rna_sd),
                (
                    "protein", protein_truth[rows], protein_prediction[rows],
                    artifact.pca.stats.protein_sd,
                ),
            ):
                static_head = "adt" if head == "protein" else "rna"
                if not np.array_equal(genes, static_prediction[f"{key}_{static_head}_action_ids"]):
                    raise ValueError("static comparator action alignment drift")
                if not np.array_equal(truth, static_prediction[f"{key}_{static_head}_truth"]):
                    raise ValueError("static comparator truth alignment drift")
                prior_rows = [prior_lookup[(str(gene), context_name)] for gene in genes]
                if not np.array_equal(truth, prior[f"{head}_truth"][prior_rows]):
                    raise ValueError("prior paired comparator truth alignment drift")
                metrics = helper.metrics(prediction, truth, train_scale)
                old = baseline["contexts"][context_name]["heads"][head]
                baselines = {**old["baselines"], "priorPairedWorld": old["world"]}
                checks = {}
                for label, comparator in baselines.items():
                    comparator_r = comparator["query_centroid_adjusted_profile_pearson"]
                    checks[label] = {
                        "mseImprovement": 1 - metrics["raw_mse"] / comparator["raw_mse"],
                        "mseAtLeastOnePercent": (
                            1 - metrics["raw_mse"] / comparator["raw_mse"] >= 0.01
                        ),
                        "rNonregression": (
                            comparator_r is None
                            or metrics["query_centroid_adjusted_profile_pearson"] >= comparator_r
                        ),
                    }
                score = metrics["query_centroid_adjusted_profile_pearson"]
                passed = (
                    np.isfinite(score) and score >= 0.10
                    and all(item["mseAtLeastOnePercent"] and item["rNonregression"] for item in checks.values())
                )
                all_pass &= bool(passed)
                contexts[context_name]["heads"][head] = {
                    "pca": metrics, "baselines": baselines, "checks": checks,
                    "passed": bool(passed), "genes": len(genes),
                }
    predictions = {
        "action_ids": action, "context_ids": context, "context_index": context_index,
        "rna_truth": rna_truth, "protein_truth": protein_truth,
        "rna_prediction": rna_prediction.astype(np.float32),
        "protein_prediction": protein_prediction.astype(np.float32),
        "latent_delta": latent_delta.astype(np.float32),
        "guide_counts": rna_guides,
    }
    return predictions, contexts, bool(all_pass), reload_drift


def run(args: argparse.Namespace) -> None:
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    for path in (CORE, Path(__file__), METRICS):
        shutil.copyfile(path, source / path.name)
    core = load_source(CORE, "slp11_paired_pca_core")
    manifest, reference, training_actions, held_actions = verify_inputs()
    core.assert_held_genes_excluded(training_actions, held_actions)
    protocol = {
        "schema": "slp.frangieh-paired-pca128-latent-ridge-protocol/v1",
        "hypothesis": "A fixed rank-128 linear basis learned from paired fitting/control cells captures both assays and supports static-feature forecasts of held intervention genes.",
        "representation": {
            "population": "reconstruction_split=train only (93,397 cells); original held-gene cells are absent from every shard",
            "normalization": "same per-query reconstruction-train population mean and SD floor0.05 as frozen AE",
            "balancedInput": "concat(RNA standardized/sqrt(2*18063), protein standardized/sqrt(2*20))",
            "algorithm": "seed731 Gaussian[18083,160], QR; three covariance subspace passes; final Rayleigh eigendecomposition; top128",
            "arithmetic": "float64 sparse affine products; no dense full-cell matrix",
            "reconstruction": "full-input reconstruction on the fixed 10,465 reconstruction-validation cells; per-head raw and standardized MSE against training mean",
        },
        "forecast": {
            "timing": "PCA and context-local latent ridges frozen before held outcomes open",
            "aggregation": "linear cell scores averaged equally within exact guide, then guides equally within gene/context",
            "aggregationEquivalence": "linear PCA encoding makes this exactly equivalent up to floating point to encoding raw guide pseudobulk means",
            "controls": "frozen measured context control means from reference.npz",
            "actionFeatures": "raw physical1156 pack with frozen reference feature mean/scale/clip",
            "ridge": "context-local alpha10000, 151 fitting genes per context, sample-space exact solve",
            "gate": "all 3 contexts x 2 heads require at least1% raw-MSE improvement over mean, base577, physical1156 and prior paired world; centered profile r>=.10 and nonregression against every defined comparator",
            "selectionOnHeldGenes": False,
        },
        "meaning": "assay-specific PCA baseline with learned fixed query loadings and a feature-linear latent forecast; not a vocabulary-free world model, nonlinear dynamics, or identified biological state",
        "inputs": {
            name: {"path": str(path), "sha256": HASHES[name]}
            for name, path in (
                ("manifest", SHARDS / "manifest.json"), ("reference", REFERENCE),
                ("static", STATIC), ("development", DEVELOPMENT), ("baseline", BASELINE),
                ("static_predictions", STATIC_PREDICTIONS),
                ("prior_predictions", PRIOR_PREDICTIONS),
            )
        },
        "population": manifest["counts"],
        "compute": {"cpuThreads": 2, "maxSeconds": 600, "memoryGiB": 6, "gpuUsed": False},
        "sourceHashes": {path.name: sha256(path) for path in source.iterdir()},
        "testAccessed": False, "benchmarkAccessed": False, "jurkatAccessed": False,
        "omfCli": "unavailable on this Windows host",
    }
    write_json(output / "protocol.json", protocol)
    print(json.dumps({"event": "protocol-frozen", "path": str(output / "protocol.json")}), flush=True)

    stats = core.fit_stats(lambda: batches(manifest, "train"), floor=0.05)
    if stats.count != 93397:
        raise ValueError("PCA fitting count drift")
    print(json.dumps({"event": "statistics-fitted", "seconds": time.monotonic() - started}), flush=True)

    def progress(stage: str, iteration: int) -> None:
        print(json.dumps({"event": stage, "pass": iteration, "seconds": time.monotonic() - started}), flush=True)
        if time.monotonic() - started > 540:
            raise TimeoutError("PCA run cannot complete within frozen 600-second cap")

    pca = core.fit_streaming_pca(
        lambda: batches(manifest, "train"), stats, rank=128, oversample=32,
        passes=3, seed=731, progress=progress,
    )
    reconstruction = reconstruction_metrics(pca, manifest)
    print(json.dumps({"event": "reconstruction-scored", "metrics": reconstruction}), flush=True)
    artifact, static_lookup, fitting_counts, guide_count = fit_forecast_artifact(
        core, pca, manifest, reference,
    )
    artifact_path = output / "pca-forecast.npz"
    artifact.save(artifact_path)
    frozen_before_held = {
        "artifactSha256": sha256(artifact_path),
        "fittingGenesByContext": fitting_counts,
        "fittingGuides": guide_count,
        "seconds": time.monotonic() - started,
    }
    write_json(output / "FROZEN-BEFORE-HELD.json", frozen_before_held)
    predictions, contexts, decision, reload_drift = forecast_and_score(
        core, artifact_path, reference, static_lookup,
    )
    np.savez_compressed(output / "predictions.npz", **predictions)
    report = {
        "schema": "slp.frangieh-paired-pca128-latent-ridge-report/v1",
        "protocolSha256": sha256(output / "protocol.json"),
        "reconstruction": reconstruction,
        "forecast": {"contexts": contexts, "passed": decision},
        "portableTargetFreeReloadMaximumDrift": reload_drift,
        "artifacts": {
            "pcaForecastSha256": sha256(artifact_path),
            "predictionsSha256": sha256(output / "predictions.npz"),
            "frozenBeforeHeldSha256": sha256(output / "FROZEN-BEFORE-HELD.json"),
        },
        "elapsedSeconds": time.monotonic() - started,
        "testAccessed": False, "benchmarkAccessed": False,
        "jurkatAccessed": False, "gpuUsed": False,
    }
    write_json(output / "report.json", report)
    print(json.dumps({"event": "finished", "passed": decision, "report": str(output / "report.json")}), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=OUTPUT)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
