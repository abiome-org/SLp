#!/usr/bin/env python3
"""Run a frozen source-specific yeast mean/linear/Nyström comparison."""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "modules/slp-1-1-yeast-static-baseline-v1"
sys.path.insert(0, str(MODULE_DIR))

from static_baseline import (
    MeanModel,
    NystromRidgeModel,
    RidgeModel,
    evaluate_gene_profiles,
    fit_mean,
    fit_nystrom_map,
    fit_ridge,
    fit_target_scale,
    grouped_folds,
)

ALPHAS = (0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0)
EXPECTED_CORPUS_SHA = "42f754425637bdf0413dbac6c36206737b5e402e04ba9732aa329cf2f1e702d5"
EXPECTED_ESM_SHA = "96f5e1b81036e0d42238ed6ac797f9fd399006f4d5f8227e96d9ee11358318ca"
EXPECTED_GO_SHA = "535086e9bed410d41028fdab225c92e3ffbdefe523385bbbf3595b2cb8209ff5"
EXPECTED_COVERAGE_SHA = "029a48d5a1dd0aff10fec9da995054381b67122d7419ea94456a1c0b7c3724c2"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _finite_json(value: object) -> object:
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_pack(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as item:
        if set(item.files) != {"feature_values", "entity_taxon", "entity_id"}:
            raise ValueError("static feature pack contract mismatch")
        values = item["feature_values"].astype(np.float64)
        taxon = item["entity_taxon"]
        identities = item["entity_id"].astype(str)
    if values.shape[0] != len(identities) or not np.isfinite(values).all():
        raise ValueError("invalid static feature values")
    if not np.all(taxon == 4932) or len(set(identities)) != len(identities):
        raise ValueError("feature identities must be unique taxon-4932 entities")
    return values, taxon, identities


def aligned_static_features(
    action_ids: np.ndarray,
    esm_path: Path,
    go_path: Path,
) -> tuple[np.ndarray, dict[str, int]]:
    esm, esm_taxon, esm_ids = load_pack(esm_path)
    go, go_taxon, go_ids = load_pack(go_path)
    if not np.array_equal(esm_taxon, go_taxon) or not np.array_equal(esm_ids, go_ids):
        raise ValueError("ESM and GO feature entity axes differ")
    index = {identity: row for row, identity in enumerate(esm_ids)}
    result = np.zeros((len(action_ids), esm.shape[1] + go.shape[1] + 2), dtype=np.float64)
    sequence_present = np.zeros(len(action_ids), dtype=bool)
    annotation_present = np.zeros(len(action_ids), dtype=bool)
    for row, identity in enumerate(action_ids.astype(str)):
        source = index.get(identity)
        if source is None:
            continue
        result[row, : esm.shape[1]] = esm[source]
        result[row, esm.shape[1] : esm.shape[1] + go.shape[1]] = go[source]
        sequence_present[row] = True
        annotation_present[row] = bool(np.any(go[source] != 0))
    result[:, -2] = sequence_present
    result[:, -1] = annotation_present
    return result, {
        "records": len(action_ids),
        "sequence_present_records": int(sequence_present.sum()),
        "go_annotation_present_records": int(annotation_present.sum()),
        "missing_sequence_records": int((~sequence_present).sum()),
    }


def cv_score(
    prediction: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    scale: np.ndarray,
) -> tuple[float, int]:
    mask = observed & np.isfinite(prediction) & np.isfinite(scale)[None, :]
    scores = []
    for row in range(target.shape[0]):
        selected = mask[row]
        if selected.any():
            scores.append(float(np.mean(((prediction[row, selected] - target[row, selected]) / scale[selected]) ** 2)))
    if not scores:
        raise ValueError("inner validation fold has no scoreable records")
    return float(np.mean(scores)), len(scores)


def fit_context(
    features: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    action_ids: np.ndarray,
    inner_fold: np.ndarray,
    *,
    landmarks: int,
    seed: int,
    scale_floor: float,
) -> tuple[MeanModel, RidgeModel, NystromRidgeModel, dict[str, object]]:
    scores = {
        "mean": [],
        "linear": {str(alpha): [] for alpha in ALPHAS},
        "nystrom": {str(alpha): [] for alpha in ALPHAS},
    }
    fold_counts = []
    for fold in range(3):
        fitting = inner_fold != fold
        held = inner_fold == fold
        if not fitting.any() or not held.any():
            raise ValueError("empty inner fold")
        scale = fit_target_scale(target[fitting], observed[fitting], floor=scale_floor)
        mean = fit_mean(target[fitting], observed[fitting])
        score, genes = cv_score(mean.predict(features[held]), target[held], observed[held], scale)
        scores["mean"].append(score)
        mapping = fit_nystrom_map(
            features[fitting], action_ids[fitting], landmarks=landmarks, seed=seed,
        )
        mapped_fitting = mapping.transform(features[fitting])
        mapped_held = mapping.transform(features[held])
        for alpha in ALPHAS:
            linear = fit_ridge(features[fitting], target[fitting], observed[fitting], alpha)
            score, _ = cv_score(linear.predict(features[held]), target[held], observed[held], scale)
            scores["linear"][str(alpha)].append(score)
            nonlinear = fit_ridge(mapped_fitting, target[fitting], observed[fitting], alpha)
            score, _ = cv_score(nonlinear.predict(mapped_held), target[held], observed[held], scale)
            scores["nystrom"][str(alpha)].append(score)
        fold_counts.append({"fold": fold, "fittingGenes": int(fitting.sum()), "heldGenes": genes})
    linear_alpha = min(ALPHAS, key=lambda alpha: (np.mean(scores["linear"][str(alpha)]), alpha))
    nystrom_alpha = min(ALPHAS, key=lambda alpha: (np.mean(scores["nystrom"][str(alpha)]), alpha))
    mean = fit_mean(target, observed)
    linear = fit_ridge(features, target, observed, linear_alpha)
    mapping = fit_nystrom_map(features, action_ids, landmarks=landmarks, seed=seed)
    nonlinear = NystromRidgeModel(
        mapping,
        fit_ridge(mapping.transform(features), target, observed, nystrom_alpha),
    )
    return mean, linear, nonlinear, {
        "folds": fold_counts,
        "mean": {"foldScores": scores["mean"], "meanScore": float(np.mean(scores["mean"]))},
        "linear": {
            "foldScores": scores["linear"],
            "meanScores": {key: float(np.mean(value)) for key, value in scores["linear"].items()},
            "selectedAlpha": linear_alpha,
        },
        "nystrom": {
            "foldScores": scores["nystrom"],
            "meanScores": {key: float(np.mean(value)) for key, value in scores["nystrom"].items()},
            "selectedAlpha": nystrom_alpha,
            "effectiveRank": int(mapping.eigenvalues.size),
            "bandwidth": mapping.bandwidth,
        },
    }


def run(args: argparse.Namespace) -> None:
    started = time.monotonic()
    paths = {
        "corpus": args.corpus.resolve(strict=True),
        "esm": args.esm.resolve(strict=True),
        "go": args.go.resolve(strict=True),
        "coverage": args.coverage.resolve(strict=True),
    }
    expected = {
        "corpus": EXPECTED_CORPUS_SHA,
        "esm": EXPECTED_ESM_SHA,
        "go": EXPECTED_GO_SHA,
        "coverage": EXPECTED_COVERAGE_SHA,
    }
    for name, path in paths.items():
        if sha256(path) != expected[name]:
            raise ValueError(f"{name} hash mismatch")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    module_path = MODULE_DIR / "static_baseline.py"
    for path in (module_path, Path(__file__)):
        shutil.copyfile(path, source / path.name)
    with np.load(paths["corpus"], allow_pickle=False) as item:
        target = item["targets"].astype(np.float64)
        observed = item["observed"]
        action_ids = item["action_ids"].astype(str)
        context_index = item["context_index"]
        context_ids = item["context_ids"].astype(str)
        train = item["split_train"]
        validation = item["split_validation"]
        test = item["split_test"]
        value_space = str(item["target_value_space"])
        taxon = int(item["ncbi_taxon"])
    if len(test) or taxon != 4932 or value_space != "author-logfoldchanges-unknown-upstream-transform":
        raise ValueError("corpus access/value-space contract mismatch")
    if set(action_ids[train]) & set(action_ids[validation]):
        raise ValueError("outer train/validation intervention overlap")
    features, feature_coverage = aligned_static_features(action_ids, paths["esm"], paths["go"])
    folds = grouped_folds(action_ids[train], folds=3, seed=args.seed)
    protocol = {
        "schema": "slp.nadal-ribelles-yeast-static-baselines/v1",
        "status": "source-specific adaptive development comparison; not a world model, confirmation, or SOTA claim",
        "hypothesis": "A median-bandwidth RBF map of static ESM2 and archived direct GO features improves held-intervention response profiles beyond fitting means and feature-linear ridge in both source environments.",
        "advancementRule": "In each context, Nyström gene-macro MSE must be at least 2% lower than both mean and selected linear ridge; independently query-centered gene-profile Pearson must be at least 0.10 and no lower than every defined comparator.",
        "data": {
            "path": str(paths["corpus"]), "sha256": expected["corpus"],
            "organism": "Saccharomyces cerevisiae", "ncbiTaxon": 4932,
            "endpoint": value_space,
            "endpointInterpretation": "The paper describes Wilcoxon mutant-versus-WT differential expression and log2 fold change within each condition; the deposited summary-generating code does not establish the exact numerical upstream estimator, so the stored value-space label is retained.",
            "contexts": context_ids.tolist(), "records": len(action_ids),
            "trainRecords": len(train), "validationRecords": len(validation),
            "testRecords": 0, "protectedOutcomesAccessed": False,
            "missingness": "observed mask authoritative; masked payload never enters a fit or metric; queries without fitting support are excluded from scoring",
            "basalContextFabricated": False,
        },
        "features": {
            "definition": "ESM2-t6-8M 320 + archived 2022 direct GO MF/CC SVD256 + sequence-present and GO-annotation-present flags",
            "width": features.shape[1], "coverage": feature_coverage,
            "missingEntityEncoding": "zero modality values with explicit presence flags",
            "queryFeaturesUsed": False, "quantitativeOutcomesUsedAsFeatures": False,
            "yeastHumanRelabeling": False,
            "inputs": {name: {"path": str(paths[name]), "sha256": expected[name]} for name in ("esm", "go", "coverage")},
        },
        "selection": {
            "contextsFitSeparately": True,
            "innerFolds": 3, "innerFoldMethod": "stable-action hash, seed731, shared across contexts",
            "alphaGrid": list(ALPHAS),
            "criterion": "equal-gene mean of per-gene masked squared error standardized by inner-fitting-only per-query target SD",
            "targetSdFloor": args.scale_floor,
            "featureScaling": "inner-fitting-only for CV and outer-fitting-only for final models",
            "nystrom": {"landmarks": args.landmarks, "bandwidth": "median positive inner-fitting action-feature distance", "landmarksSource": "inner fitting genes only"},
        },
        "evaluation": {
            "primaryMse": "equal-gene mean of within-gene observed-query MSE in stored source units",
            "primaryCorrelation": "mean per-gene profile Pearson after prediction and truth are independently centered by their validation gene means for each query",
            "trainingCentroidAdjustedAndOrdinaryReported": True,
            "allMaskedQueryExcluded": True,
        },
        "compute": {"cpuThreads": 2, "memoryCapGiB": 6, "gpuUsed": False, "omfCli": "unavailable on this Windows host"},
        "sourceHashes": {path.name: sha256(path) for path in source.iterdir()},
    }
    write_json(output / "protocol.json", protocol)
    print(json.dumps({"event": "protocol-frozen", "output": str(output)}), flush=True)

    predictions = {
        "validation_indices": validation,
        "mean": np.full((len(validation), target.shape[1]), np.nan, dtype=np.float32),
        "linear": np.full((len(validation), target.shape[1]), np.nan, dtype=np.float32),
        "nystrom": np.full((len(validation), target.shape[1]), np.nan, dtype=np.float32),
    }
    context_reports = {}
    maximum_reload_drift = 0.0
    for context, context_name in enumerate(context_ids):
        context_train = train[context_index[train] == context]
        context_validation = validation[context_index[validation] == context]
        train_positions = np.searchsorted(train, context_train)
        mean, linear, nonlinear, selection = fit_context(
            features[context_train], target[context_train], observed[context_train],
            action_ids[context_train], folds[train_positions], landmarks=args.landmarks,
            seed=args.seed, scale_floor=args.scale_floor,
        )
        scale = fit_target_scale(target[context_train], observed[context_train], floor=args.scale_floor)
        methods = {
            "mean": mean.predict(features[context_validation]),
            "linear": linear.predict(features[context_validation]),
            "nystrom": nonlinear.predict(features[context_validation]),
        }
        model_paths = {
            "mean": output / f"{context_name}-mean.npz",
            "linear": output / f"{context_name}-linear.npz",
            "nystrom": output / f"{context_name}-nystrom.npz",
        }
        np.savez_compressed(model_paths["mean"], model_type=np.asarray("masked-mean-v1"), intercept=mean.intercept, training_scale=scale)
        linear.save(model_paths["linear"])
        nonlinear.save(model_paths["nystrom"])
        reload_predictions = {
            "mean": MeanModel(np.load(model_paths["mean"], allow_pickle=False)["intercept"]).predict(features[context_validation]),
            "linear": RidgeModel.load(model_paths["linear"]).predict(features[context_validation]),
            "nystrom": NystromRidgeModel.load(model_paths["nystrom"]).predict(features[context_validation]),
        }
        metrics = {}
        local_positions = np.searchsorted(validation, context_validation)
        for name, prediction in methods.items():
            finite = np.isfinite(prediction) & np.isfinite(reload_predictions[name])
            drift = float(np.max(np.abs(prediction[finite] - reload_predictions[name][finite]))) if finite.any() else 0.0
            maximum_reload_drift = max(maximum_reload_drift, drift)
            predictions[name][local_positions] = prediction.astype(np.float32)
            metrics[name] = evaluate_gene_profiles(
                prediction, target[context_validation], observed[context_validation],
                mean.intercept, scale,
            )
        rbf_mse = metrics["nystrom"]["gene_macro_mse"]
        mean_mse = metrics["mean"]["gene_macro_mse"]
        linear_mse = metrics["linear"]["gene_macro_mse"]
        rbf_r = metrics["nystrom"]["gene_macro_independent_query_centered_profile_pearson"]
        linear_r = metrics["linear"]["gene_macro_independent_query_centered_profile_pearson"]
        gate = {
            "mseImprovementVsMean": 1.0 - rbf_mse / mean_mse,
            "mseImprovementVsLinear": 1.0 - rbf_mse / linear_mse,
            "msePass": rbf_mse <= 0.98 * mean_mse and rbf_mse <= 0.98 * linear_mse,
            "correlationThresholdPass": rbf_r is not None and rbf_r >= 0.10,
            "correlationNonregressionPass": linear_r is None or (rbf_r is not None and rbf_r >= linear_r),
        }
        gate["passed"] = all(bool(gate[key]) for key in ("msePass", "correlationThresholdPass", "correlationNonregressionPass"))
        context_reports[str(context_name)] = {
            "trainGenes": len(context_train), "validationGenes": len(context_validation),
            "selection": selection, "metrics": metrics, "gate": gate,
            "models": {name: {"path": str(path), "sha256": sha256(path)} for name, path in model_paths.items()},
        }
        print(json.dumps({"event": "context-finished", "context": str(context_name), "gate": gate}), flush=True)
    np.savez_compressed(output / "validation-predictions.npz", **predictions)
    report = {
        "schema": "slp.nadal-ribelles-yeast-static-baseline-report/v1",
        "protocolSha256": sha256(output / "protocol.json"),
        "contexts": context_reports,
        "advancementPassed": all(item["gate"]["passed"] for item in context_reports.values()),
        "maximumTargetFreeReloadDrift": maximum_reload_drift,
        "predictionArtifact": {
            "path": str(output / "validation-predictions.npz"),
            "sha256": sha256(output / "validation-predictions.npz"),
        },
        "elapsedSeconds": time.monotonic() - started,
        "protectedOutcomesAccessed": False,
        "developmentTestOutcomesAccessed": False,
        "gpuUsed": False,
    }
    write_json(output / "report.json", _finite_json(report))
    print(json.dumps({"event": "finished", "report": str(output / "report.json")}), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--corpus", type=Path, required=True)
    result.add_argument("--esm", type=Path, required=True)
    result.add_argument("--go", type=Path, required=True)
    result.add_argument("--coverage", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--landmarks", type=int, default=256)
    result.add_argument("--seed", type=int, default=731)
    result.add_argument("--scale-floor", type=float, default=0.05)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
