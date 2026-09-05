#!/usr/bin/env python3
"""Run the human training-response-basis development comparison."""

from __future__ import annotations

import os

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "4"

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from human_baselines import fit_context_references
from response_basis import fit_grouped_oof_response_basis_grid
from transition_baselines import compare_paired_nll, evaluate
from transition_calibration import fit_grouped_oof_ridge

EXPECTED_DEVELOPMENT_SHA256 = (
    "82904b7b52ab34d71e94abb2311c93a420321697d53eab12dabae5b247376f75"
)
EXPECTED_FEATURE_SHA256 = (
    "9c0ade1b580f46f26938e5eab6e0222b9e543e44bc2c7d5113336c80459bfb52"
)
REQUIRED_DEVELOPMENT_FIELDS = {
    "targets",
    "observed",
    "action_ids",
    "query_ids",
    "context_index",
    "context_ids",
    "basal_control",
    "record_ids",
    "split_train",
    "split_validation",
    "split_test",
}
REQUIRED_FEATURE_FIELDS = {"feature_values", "entity_taxon", "entity_id"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_development(path: Path) -> dict[str, np.ndarray]:
    if sha256_file(path) != EXPECTED_DEVELOPMENT_SHA256:
        raise ValueError("development bundle SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != REQUIRED_DEVELOPMENT_FIELDS:
            raise ValueError("development bundle member contract drifted")
        bundle = {name: archive[name] for name in archive.files}
    if len(bundle["split_test"]):
        raise ValueError("development bundle unexpectedly contains test indices")
    if not np.all(bundle["observed"]):
        raise ValueError("response-basis comparison requires the complete shared panel")
    training_actions = set(bundle["action_ids"][bundle["split_train"]].tolist())
    validation_actions = set(bundle["action_ids"][bundle["split_validation"]].tolist())
    if training_actions & validation_actions:
        raise ValueError("outer training and development-validation actions overlap")
    return bundle


def load_aligned_features(
    path: Path, action_ids: np.ndarray
) -> tuple[np.ndarray, tuple[int, int], int]:
    if sha256_file(path) != EXPECTED_FEATURE_SHA256:
        raise ValueError("human ESM feature SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != REQUIRED_FEATURE_FIELDS:
            raise ValueError("feature NPZ member contract drifted")
        values = archive["feature_values"]
        taxa = archive["entity_taxon"]
        identifiers = archive["entity_id"]
    keys = tuple(
        (int(taxon), str(identifier))
        for taxon, identifier in zip(taxa, identifiers, strict=True)
    )
    if values.ndim != 2 or values.shape[0] != len(keys) or not np.isfinite(values).all():
        raise ValueError("feature NPZ contains an invalid matrix")
    if len(keys) != len(set(keys)):
        raise ValueError("feature NPZ contains duplicate composite entity keys")
    index = {key: row for row, key in enumerate(keys)}
    requested = tuple((9606, str(action)) for action in action_ids)
    missing = sorted(set(requested) - set(index))
    if missing:
        raise ValueError(f"features are missing action entity keys: {missing[:8]}")
    aligned = values[np.asarray([index[key] for key in requested], dtype=np.int64)]
    zero_rows = int(np.count_nonzero(np.all(aligned[:, :-1] == 0.0, axis=1)))
    return aligned.astype(np.float64, copy=False), values.shape, zero_rows


def gene_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    action_ids: np.ndarray,
    reference: np.ndarray,
    scale: np.ndarray,
) -> dict[str, object]:
    groups: dict[str, list[int]] = {}
    for row, action in enumerate(action_ids):
        groups.setdefault(str(action), []).append(row)
    scale_matrix = np.broadcast_to(scale, prediction.shape)
    reports = [
        evaluate(
            prediction[rows],
            target[rows],
            mask[rows],
            reference,
            scale_matrix[rows],
        )
        for rows in groups.values()
    ]
    result: dict[str, object] = dict(
        evaluate(prediction, target, mask, reference, scale_matrix)
    )
    for metric in (
        "nll",
        "mse",
        "profile_pearson_mean",
        "profile_centroid_adjusted_pearson_mean",
    ):
        values = [float(report[metric]) for report in reports if np.isfinite(report[metric])]
        result["gene_macro_" + metric] = float(np.mean(values)) if values else math.nan
    result["intervention_genes"] = len(groups)
    return result


def context_metrics(
    prediction: np.ndarray,
    bundle: dict[str, np.ndarray],
    validation: np.ndarray,
    reference: np.ndarray,
    scale: np.ndarray,
) -> dict[str, object]:
    report: dict[str, object] = {}
    for context, context_id_value in enumerate(bundle["context_ids"]):
        selected = bundle["context_index"][validation] == context
        rows = validation[selected]
        if len(rows) == 0:
            raise ValueError(f"context {context} has no validation records")
        report[str(context_id_value)] = gene_metrics(
            prediction[selected],
            bundle["targets"][rows],
            bundle["observed"][rows],
            bundle["action_ids"][rows],
            reference[context],
            scale[selected],
        )
    return report


def _selection_nll(by_context: dict[str, object]) -> float:
    values = [
        float(metrics["gene_macro_nll"])
        for metrics in by_context.values()
        if np.isfinite(metrics["gene_macro_nll"])
    ]
    if not values:
        raise ValueError("no finite per-context development NLL is available")
    return float(np.mean(values))


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    development_path = args.development.resolve(strict=True)
    feature_path = args.features.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    bundle = load_development(development_path)
    features, feature_artifact_shape, zero_feature_rows = load_aligned_features(
        feature_path, bundle["action_ids"]
    )
    training = bundle["split_train"]
    validation = bundle["split_validation"]
    grid = [
        {"rank": rank, "alpha": alpha}
        for rank in args.ranks
        for alpha in args.alphas
    ]
    source_files = [
        Path(__file__),
        MODULE / "response_basis.py",
        MODULE / "human_baselines.py",
        MODULE / "transition_baselines.py",
        MODULE / "transition_calibration.py",
    ]
    protocol = {
        "schema": "slp.human-training-response-basis-development/v1",
        "scope": "human development comparison; outer training fit and validation selection only",
        "hypothesis": (
            "A training-only low-rank molecular response basis mapped from static human "
            "ESM intervention features improves held-gene development NLL over matched "
            "context means and raw-feature linear ridge."
        ),
        "advancementRule": (
            "Select the grid member with minimum unweighted mean of K562 and RPE1 "
            "development-validation gene-macro Gaussian NLL; improvement over both "
            "controls is required to advance the decoder idea."
        ),
        "candidateGridFrozenBeforeFit": grid,
        "rankProtocol": (
            "one deterministic rank-max randomized SVD per fitting set; lower ranks are "
            "nested leading-component prefixes"
        ),
        "modelIdentity": (
            "per-context training mean plus shared training-response basis whose state "
            "coefficients are predicted by feature-linear ridge"
        ),
        "outerSplit": {
            "source": "development bundle frozen split_train/split_validation",
            "trainingRecords": len(training),
            "validationRecords": len(validation),
            "trainingActions": len(set(bundle["action_ids"][training].tolist())),
            "validationActions": len(set(bundle["action_ids"][validation].tolist())),
            "actionOverlap": 0,
            "testRowsPresentInLoadedBundle": False,
            "testArtifactAccessed": False,
            "originalProtectedHoldoutsAccessed": False,
        },
        "calibration": {
            "method": (
                "action-gene-grouped OOF over outer training rows; every fold refits "
                "context means, response SVD, and coefficient ridge"
            ),
            "folds": args.folds,
            "seed": args.seed,
            "scaleFloor": args.scale_floor,
        },
        "selection": {
            "data": "development validation outcomes only",
            "metric": "unweighted context mean of gene-macro Gaussian NLL",
            "exploratory": True,
            "validationOutcomesUsedForModelOrScaleFit": False,
        },
        "valueSpace": "log2(1 + 10000*x/sum(x_shared_7226))",
        "development": {
            "path": str(development_path),
            "sha256": EXPECTED_DEVELOPMENT_SHA256,
            "contexts": [str(item) for item in bundle["context_ids"]],
            "queries": int(bundle["targets"].shape[1]),
        },
        "features": {
            "name": "human ESM2-t6-8M static intervention features",
            "path": str(feature_path),
            "sha256": EXPECTED_FEATURE_SHA256,
            "artifactShape": list(feature_artifact_shape),
            "alignedShape": list(features.shape),
            "join": "exact (NCBI taxon 9606, stable Ensembl action ID)",
            "alignedRowsWithZeroEmbeddingCoordinates": zero_feature_rows,
        },
        "controls": {
            "mean": "per-context outer-training perturbation mean",
            "rawFeatureLinearRidgeAlpha": args.raw_ridge_alpha,
        },
        "sourceHashes": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in source_files
        },
        "blasThreads": 4,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    print(json.dumps({"event": "protocol-frozen", "grid": grid}), flush=True)

    references = fit_context_references(
        bundle["targets"],
        bundle["observed"],
        bundle["context_index"],
        bundle["action_ids"],
        training,
        bundle["basal_control"],
        bundle["context_ids"],
        folds=args.folds,
        seed=args.seed,
        scale_floor=args.scale_floor,
    )
    mean_prediction = references.predict(
        "training_perturbation_mean", bundle["context_index"][validation]
    )
    mean_scale = references.scales(
        "training_perturbation_mean", bundle["context_index"][validation]
    )
    mean_by_context = context_metrics(
        mean_prediction,
        bundle,
        validation,
        references.perturbation_mean,
        mean_scale,
    )

    raw_models = {}
    raw_prediction = np.empty_like(bundle["targets"][validation], dtype=np.float64)
    raw_scale = np.empty_like(raw_prediction)
    for context in range(len(bundle["context_ids"])):
        train_rows = training[bundle["context_index"][training] == context]
        validation_selected = bundle["context_index"][validation] == context
        keys = [(9606, str(action)) for action in bundle["action_ids"][train_rows]]
        model = fit_grouped_oof_ridge(
            features[train_rows],
            bundle["targets"][train_rows],
            bundle["observed"][train_rows],
            keys,
            args.raw_ridge_alpha,
            folds=args.folds,
            seed=args.seed,
            scale_floor=args.scale_floor,
        )
        raw_models[context] = model
        raw_prediction[validation_selected] = model.predict(features[validation[validation_selected]])
        raw_scale[validation_selected] = model.residual_scale_.values
    raw_by_context = context_metrics(
        raw_prediction,
        bundle,
        validation,
        references.perturbation_mean,
        raw_scale,
    )
    controls = {
        "context-training-mean": {
            "selectionNll": _selection_nll(mean_by_context),
            "byContext": mean_by_context,
            "scaleProvenance": references.scale_provenance,
        },
        "context-specific-raw-ESM-feature-linear-ridge": {
            "configuration": {"alpha": args.raw_ridge_alpha},
            "selectionNll": _selection_nll(raw_by_context),
            "byContext": raw_by_context,
            "scaleProvenance": next(iter(raw_models.values())).residual_scale_.provenance,
        },
    }
    print(
        json.dumps(
            {
                "event": "controls",
                "meanSelectionNll": controls["context-training-mean"]["selectionNll"],
                "rawRidgeSelectionNll": controls[
                    "context-specific-raw-ESM-feature-linear-ridge"
                ]["selectionNll"],
            }
        ),
        flush=True,
    )

    models = fit_grouped_oof_response_basis_grid(
        features[training],
        bundle["targets"][training],
        bundle["observed"][training],
        bundle["context_index"][training],
        bundle["action_ids"][training],
        ranks=args.ranks,
        alphas=args.alphas,
        folds=args.folds,
        seed=args.seed,
        scale_floor=args.scale_floor,
    )
    candidates: list[dict[str, object]] = []
    for configuration in grid:
        model = models[(configuration["rank"], float(configuration["alpha"]))]
        prediction = model.predict(
            features[validation], bundle["context_index"][validation]
        )
        scales = model.scales(bundle["context_index"][validation])
        by_context = context_metrics(
            prediction,
            bundle,
            validation,
            references.perturbation_mean,
            scales,
        )
        selection_nll = _selection_nll(by_context)
        candidate = {
            "configuration": configuration,
            "selectionNll": selection_nll,
            "byContext": by_context,
            "selectionNllDeltaVsMean": controls["context-training-mean"][
                "selectionNll"
            ]
            - selection_nll,
            "selectionNllDeltaVsRawRidge": controls[
                "context-specific-raw-ESM-feature-linear-ridge"
            ]["selectionNll"]
            - selection_nll,
            "pairedVsMeanByContext": {},
            "pairedVsRawRidgeByContext": {},
            "scaleProvenance": model.residual_scale_.provenance,
            "foldAudit": list(model.fold_audit_),
        }
        for context, context_id_value in enumerate(bundle["context_ids"]):
            selected = bundle["context_index"][validation] == context
            rows = validation[selected]
            context_id = str(context_id_value)
            candidate["pairedVsMeanByContext"][context_id] = compare_paired_nll(
                prediction[selected],
                mean_prediction[selected],
                bundle["targets"][rows],
                bundle["observed"][rows],
                scales[selected],
                mean_scale[selected],
            )
            candidate["pairedVsRawRidgeByContext"][context_id] = compare_paired_nll(
                prediction[selected],
                raw_prediction[selected],
                bundle["targets"][rows],
                bundle["observed"][rows],
                scales[selected],
                raw_scale[selected],
            )
        candidates.append(candidate)
        print(
            json.dumps(
                {
                    "event": "candidate",
                    "configuration": configuration,
                    "selectionNll": selection_nll,
                    "adjustedPearsonByContext": {
                        context: metrics[
                            "gene_macro_profile_centroid_adjusted_pearson_mean"
                        ]
                        for context, metrics in by_context.items()
                    },
                }
            ),
            flush=True,
        )
    champion_index = min(
        range(len(candidates)), key=lambda index: candidates[index]["selectionNll"]
    )
    champion = candidates[champion_index]
    advances = bool(
        champion["selectionNll"] < controls["context-training-mean"]["selectionNll"]
        and champion["selectionNll"]
        < controls["context-specific-raw-ESM-feature-linear-ridge"]["selectionNll"]
    )
    report = {
        "protocol": protocol,
        "controls": controls,
        "candidates": candidates,
        "championIndex": champion_index,
        "champion": champion,
        "advancementRulePassed": advances,
        "interpretation": (
            "The exploratory decoder idea passes its declared development rule."
            if advances
            else "The exploratory decoder idea does not pass its declared development rule."
        ),
        "testArtifactAccessed": False,
        "originalProtectedHoldoutsAccessed": False,
        "elapsedSeconds": time.monotonic() - started,
    }
    write_json(output / "report.json", report)
    print(
        json.dumps(
            {
                "event": "finished",
                "output": str(output),
                "champion": champion["configuration"],
                "advancementRulePassed": advances,
            }
        ),
        flush=True,
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--development",
        type=Path,
        default=ROOT
        / "data"
        / "derived"
        / "slp11-human"
        / "replogle-k562-rpe1-development-v1.npz",
    )
    result.add_argument(
        "--features",
        type=Path,
        default=ROOT
        / "data"
        / "derived"
        / "slp11-human-sequence"
        / "esm2-t6-8m-ensembl116-full-v1"
        / "human-sequence-esm2-features.npz",
    )
    result.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "slp11-transition" / "human-response-basis-v1",
    )
    result.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 64])
    result.add_argument(
        "--alphas", type=float, nargs="+", default=[100.0, 1000.0, 10000.0]
    )
    result.add_argument("--raw-ridge-alpha", type=float, default=10000.0)
    result.add_argument("--folds", type=int, default=3)
    result.add_argument("--seed", type=int, default=731)
    result.add_argument("--scale-floor", type=float, default=0.05)
    return result


def main() -> int:
    try:
        with threadpool_limits(limits=4):
            run(parser().parse_args())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
