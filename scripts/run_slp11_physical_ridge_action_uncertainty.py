#!/usr/bin/env python3
"""Test static-action variance for the frozen full physical-feature ridge."""

from __future__ import annotations

import os

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "2"

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from action_uncertainty import (
    RecordVarianceMoments,
    estimate_record_variance_moments,
    fit_action_variance_multiplier,
)
from exposure_uncertainty import fit_exposure_uncertainty
from transition_baselines import evaluate
from transition_calibration import fit_grouped_oof_ridge

EXPECTED_DATA_SHA256 = "006b4bb127a09073a7f409d81a7bccce96bb961879cb5e57dce56b48eb8e664b"
EXPECTED_FEATURE_SHA256 = "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7"
RIDGE_ALPHA = 10_000.0
VARIANCE_ALPHA = 10_000.0
MOMENT_FLOOR = 0.05
FACTOR_MIN = 0.25
FACTOR_MAX = 4.0
SCALE_FLOOR = 0.05
OOF_FOLDS = 3
SEED = 731
COUNT_STRATA = ((1.0, 50.0, "1-49"), (50.0, 150.0, "50-149"), (150.0, math.inf, "150+"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_data(path: Path) -> dict[str, np.ndarray]:
    if sha256_file(path) != EXPECTED_DATA_SHA256:
        raise ValueError("pinned three-context development data SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    required = {
        "action_ids", "basal_control", "context_ids", "context_index",
        "control_context_index", "control_core", "control_num_cells_filtered",
        "control_observed", "control_targets", "num_cells_filtered", "observed",
        "record_ids", "split_test", "split_train", "split_validation",
        "target_value_space", "targets",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"development data lacks required members: {sorted(missing)}")
    if len(data["split_test"]):
        raise ValueError("development bundle unexpectedly contains test rows")
    if not np.all(data["control_core"]):
        raise ValueError("sampling calibration requires verified core controls only")
    train, validation = data["split_train"], data["split_validation"]
    if set(data["action_ids"][train].tolist()) & set(data["action_ids"][validation].tolist()):
        raise ValueError("outer training and validation intervention genes overlap")
    return data


def load_features(path: Path, action_ids: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    if sha256_file(path) != EXPECTED_FEATURE_SHA256:
        raise ValueError("pinned full physical feature SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"feature_values", "entity_taxon", "entity_id"}:
            raise ValueError("physical feature member contract drifted")
        values = archive["feature_values"]
        taxa = archive["entity_taxon"]
        identifiers = archive["entity_id"]
    if values.ndim != 2 or values.shape[1] != 1156 or not np.isfinite(values).all():
        raise ValueError("full physical feature matrix is invalid")
    keys = [(int(taxon), str(identifier)) for taxon, identifier in zip(taxa, identifiers, strict=True)]
    if len(keys) != len(set(keys)):
        raise ValueError("physical feature composite identities are duplicated")
    lookup = {key: row for row, key in enumerate(keys)}
    requested = [(9606, str(action)) for action in action_ids]
    missing = sorted(set(requested) - set(lookup))
    if missing:
        raise ValueError(f"physical features lack action identities: {missing[:8]}")
    rows = np.asarray([lookup[key] for key in requested], dtype=np.int64)
    return values[rows].astype(np.float64, copy=False), values.shape


def gene_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    observed: np.ndarray,
    action_ids: np.ndarray,
    reference: np.ndarray,
    scale: np.ndarray,
    value_space: str,
) -> dict[str, object]:
    groups: dict[str, list[int]] = {}
    for row, action in enumerate(action_ids):
        groups.setdefault(str(action), []).append(row)
    reports = [
        evaluate(
            prediction[rows], truth[rows], observed[rows], reference, scale[rows],
            value_space=value_space,
        )
        for rows in groups.values()
    ]
    result = dict(evaluate(prediction, truth, observed, reference, scale, value_space=value_space))
    for metric in (
        "nll", "mse", "profile_pearson_mean",
        "profile_centroid_adjusted_pearson_mean",
    ):
        values = [float(item[metric]) for item in reports if np.isfinite(item[metric])]
        result["gene_macro_" + metric] = float(np.mean(values)) if values else math.nan
    result["intervention_genes"] = len(groups)
    return result


def calibration_summary(
    prediction: np.ndarray,
    truth: np.ndarray,
    observed: np.ndarray,
    scale: np.ndarray,
) -> dict[str, float | int]:
    residual = prediction - truth
    selected = observed
    standardized = residual[selected] / scale[selected]
    point_nll = 0.5 * (
        math.log(2.0 * math.pi) + 2.0 * np.log(scale[selected]) + standardized**2
    )
    return {
        "observedTargets": int(np.count_nonzero(selected)),
        "nllNatsPerTarget": float(np.mean(point_nll)),
        "residualMse": float(np.mean(np.square(residual[selected]))),
        "meanPredictedVariance": float(np.mean(np.square(scale[selected]))),
        "standardizedResidualSquaredMean": float(np.mean(np.square(standardized))),
        "coverage80": float(np.mean(np.abs(standardized) <= 1.2815515655446004)),
        "coverage95": float(np.mean(np.abs(standardized) <= 1.959963984540054)),
    }


def count_strata(
    prediction: np.ndarray,
    truth: np.ndarray,
    observed: np.ndarray,
    base_scale: np.ndarray,
    adjusted_scale: np.ndarray,
    num_cells: np.ndarray,
) -> dict[str, object]:
    report: dict[str, object] = {}
    for lower, upper, label in COUNT_STRATA:
        selected = (num_cells >= lower) & (num_cells < upper)
        if not np.any(selected):
            report[label] = {"records": 0}
            continue
        report[label] = {
            "records": int(np.count_nonzero(selected)),
            "cellCountRange": [int(np.min(num_cells[selected])), int(np.max(num_cells[selected]))],
            "base": calibration_summary(
                prediction[selected], truth[selected], observed[selected], base_scale[selected]
            ),
            "actionDependent": calibration_summary(
                prediction[selected], truth[selected], observed[selected], adjusted_scale[selected]
            ),
        }
    return report


def moment_summary(moments: RecordVarianceMoments) -> dict[str, object]:
    selected = moments.raw_values[moments.identifiable]
    return {
        "records": int(moments.raw_values.size),
        "identifiableRecords": moments.identifiable_count,
        "unidentifiableRecords": int(np.count_nonzero(~moments.identifiable)),
        "floorFraction": moments.floor_fraction,
        "pathologicalFloorMass": bool(moments.floor_fraction >= 0.25),
        "rawMomentQuantiles": {
            str(probability): float(value)
            for probability, value in zip(
                (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0),
                np.quantile(selected, (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)),
                strict=True,
            )
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    data_path = args.data.resolve(strict=True)
    feature_path = args.features.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")

    data = load_data(data_path)
    features, feature_artifact_shape = load_features(feature_path, data["action_ids"])
    train, validation = data["split_train"], data["split_validation"]
    source_files = (
        Path(__file__), MODULE / "action_uncertainty.py", MODULE / "exposure_uncertainty.py",
        MODULE / "transition_baselines.py", MODULE / "transition_calibration.py",
    )
    protocol = {
        "schema": "slp.physical-ridge-action-uncertainty-development/v1",
        "hypothesis": (
            "A static-action feature predictor of residual biological variance improves "
            "held-gene likelihood beyond context/query biological variance plus control-derived "
            "sampling variance divided by cell exposure."
        ),
        "fixedAdvancementRule": (
            "Base minus action-dependent development-validation gene-macro NLL must be at "
            "least 0.01 nats per target in every one of the three contexts."
        ),
        "frozenBeforeFit": True,
        "fit": {
            "meanModel": "full 1156-column physical-feature linear multioutput ridge per context",
            "meanRidgeAlpha": RIDGE_ALPHA,
            "meanCalibration": "three-fold stable-intervention-grouped OOF within each context",
            "oofFolds": OOF_FOLDS,
            "seed": SEED,
            "varianceMoment": (
                "sum_observed(residual_oof^2 - sampling_variance[context,query]/num_cells) "
                "/ sum_observed(biological_variance[context,query])"
            ),
            "momentFloor": MOMENT_FLOOR,
            "variancePredictor": "standardized static-feature-linear ridge on log(floored moment)",
            "varianceRidgeAlpha": VARIANCE_ALPHA,
            "predictedFactorBounds": [FACTOR_MIN, FACTOR_MAX],
            "scaleFloor": SCALE_FLOOR,
            "application": (
                "variance=factor(action)*biological[context,query] "
                "+ sampling[context,query]/num_cells"
            ),
            "pathologicalFloorMassThreshold": 0.25,
            "countStrata": [
                {"label": label, "lowerInclusive": lower, "upperExclusive": upper}
                for lower, upper, label in COUNT_STRATA
            ],
            "hyperparameterSearch": False,
        },
        "separation": {
            "cellCountRole": "uncertainty only",
            "actionFactorRole": "biological variance only",
            "meanForecastChangeAllowed": False,
            "targetValueSpace": str(data["target_value_space"].item()),
        },
        "outerSplit": {
            "trainingRecords": len(train),
            "validationRecords": len(validation),
            "trainingActions": len(set(data["action_ids"][train].tolist())),
            "validationActions": len(set(data["action_ids"][validation].tolist())),
            "actionOverlap": 0,
            "testRowsInBundle": 0,
            "protectedOrRetiredOutcomeAccess": False,
        },
        "data": {"path": str(data_path), "sha256": EXPECTED_DATA_SHA256},
        "features": {
            "path": str(feature_path), "sha256": EXPECTED_FEATURE_SHA256,
            "artifactShape": list(feature_artifact_shape), "alignedShape": list(features.shape),
        },
        "sourceHashes": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in source_files
        },
        "cpuThreads": 2,
        "timeLimitSeconds": 300,
        "scope": "statistical baseline uncertainty diagnostic; no world-model claim",
    }
    output.mkdir(parents=True, exist_ok=False)
    source_output = output / "source"
    source_output.mkdir()
    for path in source_files:
        shutil.copyfile(path, source_output / path.name)
    write_json(output / "protocol.json", protocol)
    print(json.dumps({"event": "protocol-frozen", "output": str(output)}), flush=True)

    targets = data["targets"]
    observed = data["observed"]
    contexts = data["context_index"]
    validation_prediction = np.empty((len(validation), targets.shape[1]), dtype=np.float64)
    biological = np.empty((len(data["context_ids"]), targets.shape[1]), dtype=np.float64)
    sampling = np.empty_like(biological)
    sampling_from_controls = np.empty_like(biological, dtype=bool)
    validation_position = {int(row): position for position, row in enumerate(validation)}
    moment_rows: list[np.ndarray] = []
    moment_parts: list[RecordVarianceMoments] = []
    exposure_provenance: dict[str, object] = {}

    for context, context_name in enumerate(data["context_ids"]):
        fitting = train[contexts[train] == context]
        scoring = validation[contexts[validation] == context]
        keys = [(9606, str(action)) for action in data["action_ids"][fitting]]
        ridge, oof_prediction = fit_grouped_oof_ridge(
            features[fitting], targets[fitting], observed[fitting], keys,
            alpha=RIDGE_ALPHA, folds=OOF_FOLDS, seed=SEED,
            scale_floor=SCALE_FLOOR, return_oof=True,
        )
        local_validation_prediction = ridge.predict(features[scoring])
        positions = np.asarray([validation_position[int(row)] for row in scoring], dtype=np.int64)
        validation_prediction[positions] = local_validation_prediction

        control_selected = data["control_context_index"] == context
        exposure = fit_exposure_uncertainty(
            targets[fitting].astype(np.float64) - oof_prediction,
            observed[fitting], data["num_cells_filtered"][fitting],
            np.zeros(len(fitting), dtype=np.int64),
            control_targets=data["control_targets"][control_selected],
            control_observed=data["control_observed"][control_selected],
            control_num_cells=data["control_num_cells_filtered"][control_selected],
            control_context_index=np.zeros(np.count_nonzero(control_selected), dtype=np.int64),
            scale_floor=SCALE_FLOOR,
        )
        biological[context] = exposure.biological_variance_[0]
        sampling[context] = exposure.sampling_variance_[0]
        sampling_from_controls[context] = exposure.sampling_from_controls_[0]
        residuals = targets[fitting].astype(np.float64) - oof_prediction
        moments = estimate_record_variance_moments(
            residuals, observed[fitting], data["num_cells_filtered"][fitting],
            np.zeros(len(fitting), dtype=np.int64), exposure.biological_variance_,
            exposure.sampling_variance_, moment_floor=MOMENT_FLOOR,
        )
        moment_rows.append(fitting)
        moment_parts.append(moments)
        exposure_provenance[str(context_name)] = {
            "componentProvenance": exposure.component_provenance,
            "identifiabilityWarning": exposure.identifiability_warning,
            "samplingFromCoreControlsFraction": float(exposure.sampling_from_controls_.mean()),
            "moment": moment_summary(moments),
        }
        del ridge, oof_prediction, residuals
        print(json.dumps({"event": "context-oof-complete", "context": str(context_name)}), flush=True)

    ordered_moment_rows = np.concatenate(moment_rows)
    combined_moments = RecordVarianceMoments(
        raw_values=np.concatenate([item.raw_values for item in moment_parts]),
        positive_values=np.concatenate([item.positive_values for item in moment_parts]),
        identifiable=np.concatenate([item.identifiable for item in moment_parts]),
        observed_counts=np.concatenate([item.observed_counts for item in moment_parts]),
        floor=MOMENT_FLOOR,
    )
    variance_model = fit_action_variance_multiplier(
        features[ordered_moment_rows], combined_moments, alpha=VARIANCE_ALPHA,
        factor_min=FACTOR_MIN, factor_max=FACTOR_MAX,
    )
    validation_factor = variance_model.multipliers(features[validation])

    value_space = str(data["target_value_space"].item())
    context_results: dict[str, object] = {}
    for context, context_name in enumerate(data["context_ids"]):
        selected = contexts[validation] == context
        rows = validation[selected]
        prediction = validation_prediction[selected]
        truth = targets[rows]
        mask = observed[rows]
        counts = data["num_cells_filtered"][rows].astype(np.float64)
        base_variance = biological[context][None, :] + sampling[context][None, :] / counts[:, None]
        adjusted_variance = (
            validation_factor[selected, None] * biological[context][None, :]
            + sampling[context][None, :] / counts[:, None]
        )
        base_scale = np.sqrt(np.maximum(base_variance, SCALE_FLOOR**2))
        adjusted_scale = np.sqrt(np.maximum(adjusted_variance, SCALE_FLOOR**2))
        base_metrics = gene_metrics(
            prediction, truth, mask, data["action_ids"][rows], data["basal_control"][context],
            base_scale, value_space,
        )
        action_metrics = gene_metrics(
            prediction, truth, mask, data["action_ids"][rows], data["basal_control"][context],
            adjusted_scale, value_space,
        )
        improvement = float(base_metrics["gene_macro_nll"] - action_metrics["gene_macro_nll"])
        context_results[str(context_name)] = {
            "records": len(rows),
            "base": base_metrics,
            "actionDependent": action_metrics,
            "baseMinusActionGeneMacroNllNatsPerTarget": improvement,
            "advancementThreshold": 0.01,
            "advancementPassed": bool(improvement >= 0.01),
            "factor": {
                "minimum": float(np.min(validation_factor[selected])),
                "median": float(np.median(validation_factor[selected])),
                "maximum": float(np.max(validation_factor[selected])),
                "atLowerClampFraction": float(np.mean(validation_factor[selected] == FACTOR_MIN)),
                "atUpperClampFraction": float(np.mean(validation_factor[selected] == FACTOR_MAX)),
            },
            "calibrationByCount": count_strata(
                prediction, truth, mask, base_scale, adjusted_scale, counts
            ),
        }
        print(
            json.dumps(
                {"event": "validation-scored", "context": str(context_name),
                 "baseGeneMacroNll": base_metrics["gene_macro_nll"],
                 "actionGeneMacroNll": action_metrics["gene_macro_nll"],
                 "improvement": improvement}
            ),
            flush=True,
        )

    prediction_hash = sha256_array(validation_prediction)
    artifact_path = output / "uncertainty.npz"
    np.savez_compressed(
        artifact_path,
        biological_variance=biological,
        sampling_variance=sampling,
        sampling_from_controls=sampling_from_controls,
        action_feature_mean=variance_model.feature_mean_,
        action_feature_scale=variance_model.feature_scale_,
        action_log_variance_coefficient=variance_model.coefficient_,
        action_log_variance_intercept=np.asarray(variance_model.intercept_),
        validation_record_ids=data["record_ids"][validation],
        validation_biological_factor=validation_factor,
    )
    floor_pathological = combined_moments.floor_fraction >= 0.25
    all_contexts_pass = all(
        bool(result["advancementPassed"]) for result in context_results.values()
    )
    report = {
        "schema": "slp.physical-ridge-action-uncertainty-development-report/v1",
        "results": context_results,
        "advancementPassed": bool(all_contexts_pass),
        "advancementDecision": (
            "advance-action-dependent-baseline-uncertainty"
            if all_contexts_pass
            else "reject-as-next-likelihood-lever"
        ),
        "fittingMoment": moment_summary(combined_moments),
        "pathologicalFloorMass": bool(floor_pathological),
        "exposureCalibration": exposure_provenance,
        "samplingFromCoreControlsFraction": float(sampling_from_controls.mean()),
        "meanForecastBitIdentical": True,
        "meanForecastSha256ForBothArms": prediction_hash,
        "artifact": {"path": str(artifact_path), "sha256": sha256_file(artifact_path)},
        "elapsedSeconds": float(time.monotonic() - started),
        "cpuTimeLimitSeconds": 300,
        "testOrProtectedOutcomesAccessed": False,
        "interpretationScope": "full physical-feature ridge baseline uncertainty only",
    }
    write_json(output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/derived/slp11-human-gwps/complete-panel-v1/development.npz",
    )
    parser.add_argument(
        "--features", type=Path,
        default=(ROOT / "data/derived/slp11-human-physical/direct-experiments700-v1/"
                 "human-esm-go-physical-features.npz"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=(ROOT / "results/slp11-transition/"
                 "human-gwps-physical-ridge-action-uncertainty-v1"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        result = run(parse_args())
    print(json.dumps({"event": "complete", "advancementPassed": result["advancementPassed"]}))
