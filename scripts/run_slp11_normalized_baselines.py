#!/usr/bin/env python3
"""Run corrected-human static-fusion probabilistic development baselines."""

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

from exposure_uncertainty import (
    ExposureGaussianUncertainty,
    fit_exposure_uncertainty,
)
from response_basis import fit_grouped_oof_response_basis_grid
from transition_baselines import evaluate
from transition_calibration import (
    fit_grouped_oof_mean,
    fit_grouped_oof_ridge,
)

EXPECTED_DATA_SHA256 = "88de5164fca4e2504ac5b459ab4226c161eb586dd04700d5784da4bb53048659"
EXPECTED_FEATURE_SHA256 = "b3de49e18d3c75676985b8790d1ce85de0d87d526bbd7c0c5b555828a1fb11a0"
REQUIRED_DATA_FIELDS = {
    "action_ids", "basal_control", "context_basal_expression", "context_ids",
    "context_index", "context_value_space", "control_context_index", "control_core",
    "control_num_cells_filtered", "control_observed", "control_record_ids",
    "control_targets", "num_cells_filtered", "num_cells_role", "observed",
    "query_ids", "record_ids", "split_test", "split_train", "split_validation",
    "target_value_space", "targets",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
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
        raise ValueError("corrected human development SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != REQUIRED_DATA_FIELDS:
            raise ValueError("corrected human development member contract drifted")
        data = {name: archive[name] for name in archive.files}
    if len(data["split_test"]):
        raise ValueError("development bundle unexpectedly contains test rows")
    if not np.all(data["observed"]) or not np.all(data["control_observed"]):
        raise ValueError("response-basis arm requires the expected complete shared panel")
    if not np.all(data["control_core"]):
        raise ValueError("control inputs must contain only verified core controls")
    training_actions = set(data["action_ids"][data["split_train"]].tolist())
    validation_actions = set(data["action_ids"][data["split_validation"]].tolist())
    if training_actions & validation_actions:
        raise ValueError("outer training and validation interventions overlap")
    return data


def load_features(path: Path, action_ids: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    if sha256_file(path) != EXPECTED_FEATURE_SHA256:
        raise ValueError("static-fusion feature SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"feature_values", "entity_taxon", "entity_id"}:
            raise ValueError("static-fusion feature member contract drifted")
        values = archive["feature_values"]
        taxa = archive["entity_taxon"]
        identifiers = archive["entity_id"]
    keys = tuple(
        (int(taxon), str(identifier))
        for taxon, identifier in zip(taxa, identifiers, strict=True)
    )
    if values.ndim != 2 or values.shape[0] != len(keys) or not np.isfinite(values).all():
        raise ValueError("static-fusion feature values are invalid")
    if len(keys) != len(set(keys)):
        raise ValueError("static-fusion composite keys are duplicated")
    lookup = {key: row for row, key in enumerate(keys)}
    requested = tuple((9606, str(action)) for action in action_ids)
    missing = sorted(set(requested) - set(lookup))
    if missing:
        raise ValueError(f"static-fusion features lack action keys: {missing[:8]}")
    aligned = values[np.asarray([lookup[key] for key in requested], dtype=np.int64)]
    return aligned.astype(np.float64, copy=False), values.shape


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
    result: dict[str, object] = dict(
        evaluate(
            prediction, truth, observed, reference, scale, value_space=value_space
        )
    )
    for metric in (
        "nll", "mse", "profile_pearson_mean",
        "profile_centroid_adjusted_pearson_mean",
    ):
        values = [float(report[metric]) for report in reports if np.isfinite(report[metric])]
        result["gene_macro_" + metric] = float(np.mean(values)) if values else math.nan
    result["intervention_genes"] = len(groups)
    return result


def metrics_by_context(
    prediction: np.ndarray,
    scale: np.ndarray,
    data: dict[str, np.ndarray],
    validation: np.ndarray,
    references: np.ndarray,
    value_space: str,
) -> dict[str, object]:
    report = {}
    for context, context_id_value in enumerate(data["context_ids"]):
        selected = data["context_index"][validation] == context
        rows = validation[selected]
        report[str(context_id_value)] = gene_metrics(
            prediction[selected], data["targets"][rows], data["observed"][rows],
            data["action_ids"][rows], references[context], scale[selected], value_space,
        )
    return report


def selection_nll(by_context: dict[str, object]) -> float:
    return float(np.mean([metrics["gene_macro_nll"] for metrics in by_context.values()]))


def exposure_summary(model: ExposureGaussianUncertainty) -> dict[str, object]:
    return {
        "componentProvenance": model.component_provenance,
        "identifiabilityWarning": model.identifiability_warning,
        "samplingFromCoreControlsFraction": float(model.sampling_from_controls_.mean()),
        "biologicalVarianceMean": float(model.biological_variance_.mean()),
        "samplingVarianceMean": float(model.sampling_variance_.mean()),
        "scaleFloor": model.scale_floor,
    }


def response_exposure_from_oof_sufficient_statistics(
    model,
    control_calibration: ExposureGaussianUncertainty,
    training_counts: np.ndarray,
    training_contexts: np.ndarray,
) -> ExposureGaussianUncertainty:
    """Apply the common core-control slope to response-basis OOF residual MSE."""

    if not np.all(control_calibration.sampling_from_controls_):
        raise ValueError("response calibration requires identifiable core-control slopes")
    if np.any(model.residual_scale_.values <= model.residual_scale_.values.dtype.type(0.05)):
        raise ValueError("response OOF RMSE hit its floor; exact sufficient variance unavailable")
    inverse_exposure_mean = np.asarray([
        np.mean(1.0 / training_counts[training_contexts == context])
        for context in range(model.context_means_.shape[0])
    ])
    oof_variance = np.square(model.residual_scale_.values)
    biological = np.maximum(
        oof_variance
        - control_calibration.sampling_variance_ * inverse_exposure_mean[:, None],
        0.0,
    )
    return ExposureGaussianUncertainty(
        biological_variance_=biological,
        sampling_variance_=control_calibration.sampling_variance_.copy(),
        residual_counts_=model.residual_scale_.counts.copy(),
        control_counts_=control_calibration.control_counts_.copy(),
        sampling_from_controls_=control_calibration.sampling_from_controls_.copy(),
        scale_floor=control_calibration.scale_floor,
        component_provenance=(
            "core-control-sampling-slope-plus-response-basis-refit-OOF-residual-MSE"
        ),
        identifiability_warning=None,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    data_path = args.data.resolve(strict=True)
    feature_path = args.features.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    data = load_data(data_path)
    features, feature_artifact_shape = load_features(feature_path, data["action_ids"])
    training, validation = data["split_train"], data["split_validation"]
    value_space = str(data["target_value_space"].item())
    grid = {
        "fullFeatureRidge": [{"alpha": alpha} for alpha in args.alphas],
        "responseBasis": [
            {"rank": rank, "alpha": alpha}
            for rank in args.ranks for alpha in args.alphas
        ],
    }
    source_files = [
        Path(__file__), MODULE / "transition_baselines.py",
        MODULE / "transition_calibration.py", MODULE / "exposure_uncertainty.py",
        MODULE / "response_basis.py", MODULE / "human_baselines.py",
    ]
    protocol = {
        "schema": "slp.human-normalized-static-baselines-development/v1",
        "scope": "corrected author-normalized human development comparison",
        "hypothesis": (
            "Static ESM2+GO intervention features predict held-gene author-normalized "
            "pseudobulk responses better than context means, and a training-response "
            "basis may improve on unrestricted per-query ridge."
        ),
        "advancementRule": (
            "Choose minimum equal-context development-validation gene-macro Gaussian "
            "NLL after all predictions and training-only exposure scales are fixed."
        ),
        "candidateGridFrozenBeforeFit": grid,
        "selectionIsExploratoryDevelopmentTuning": True,
        "pointMetricsScaleIndependent": True,
        "outerSplit": {
            "source": "frozen split_train and split_validation members",
            "trainingRecords": len(training), "validationRecords": len(validation),
            "trainingActions": len(set(data["action_ids"][training].tolist())),
            "validationActions": len(set(data["action_ids"][validation].tolist())),
            "actionOverlap": 0, "testRowsInLoadedBundle": 0,
            "testArtifactAccessed": False, "protectedHoldoutAccessed": False,
        },
        "calibration": {
            "folds": args.folds, "seed": args.seed, "scaleFloor": args.scale_floor,
            "grouping": "stable intervention gene within context for mean/ridge; across contexts for response basis",
            "exposureModel": "variance=context/query biological + sampling/num_cells",
            "samplingSource": "verified core non-targeting controls",
            "responseBasisOOF": "every fold refits context means, response SVD, and coefficient ridge",
            "cellCountRole": "likelihood scale only; never a point-prediction feature",
        },
        "data": {"path": str(data_path), "sha256": EXPECTED_DATA_SHA256,
                 "targetValueSpace": value_space,
                 "numCellsRole": str(data["num_cells_role"].item()),
                 "coreControlRecords": len(data["control_targets"])},
        "features": {"path": str(feature_path), "sha256": EXPECTED_FEATURE_SHA256,
                     "artifactShape": list(feature_artifact_shape),
                     "alignedShape": list(features.shape),
                     "modalities": ["ESM2-t6 static protein", "archived GO MF/CC SVD"]},
        "sourceHashes": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in source_files
        },
        "blasThreads": 4,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    print(json.dumps({"event": "protocol-frozen", "grid": grid}), flush=True)

    targets, observed, contexts = data["targets"], data["observed"], data["context_index"]
    control_args = {
        "control_targets": data["control_targets"],
        "control_observed": data["control_observed"],
        "control_num_cells": data["control_num_cells_filtered"],
        "control_context_index": data["control_context_index"],
    }
    mean_models = []
    mean_oof = np.empty_like(targets[training], dtype=np.float64)
    mean_prediction = np.empty_like(targets[validation], dtype=np.float64)
    for context in range(len(data["context_ids"])):
        train_selected = contexts[training] == context
        validation_selected = contexts[validation] == context
        train_rows = training[train_selected]
        keys = [(9606, str(action)) for action in data["action_ids"][train_rows]]
        model, oof = fit_grouped_oof_mean(
            targets[train_rows], observed[train_rows], keys, folds=args.folds,
            seed=args.seed, scale_floor=args.scale_floor, return_oof=True,
        )
        mean_models.append(model)
        mean_oof[train_selected] = oof
        mean_prediction[validation_selected] = model.predict(
            np.zeros((int(validation_selected.sum()), 1))
        )
    references = np.stack([model.intercept_ for model in mean_models])
    mean_exposure = fit_exposure_uncertainty(
        targets[training] - mean_oof, observed[training],
        data["num_cells_filtered"][training], contexts[training],
        **control_args, scale_floor=args.scale_floor,
    )
    mean_scale = mean_exposure.scales(
        data["num_cells_filtered"][validation], contexts[validation]
    )
    mean_context = metrics_by_context(
        mean_prediction, mean_scale, data, validation, references, value_space
    )
    mean_report = {"selectionNll": selection_nll(mean_context),
                   "byContext": mean_context,
                   "exposure": exposure_summary(mean_exposure)}
    print(json.dumps({"event": "mean", "selectionNll": mean_report["selectionNll"]}), flush=True)

    arrays: dict[str, np.ndarray] = {
        "mean_biological_variance": mean_exposure.biological_variance_,
        "mean_sampling_variance": mean_exposure.sampling_variance_,
    }
    candidates: list[dict[str, object]] = []
    for alpha in args.alphas:
        oof_prediction = np.empty_like(targets[training], dtype=np.float64)
        prediction = np.empty_like(targets[validation], dtype=np.float64)
        for context in range(len(data["context_ids"])):
            train_selected = contexts[training] == context
            validation_selected = contexts[validation] == context
            train_rows = training[train_selected]
            keys = [(9606, str(action)) for action in data["action_ids"][train_rows]]
            model, oof = fit_grouped_oof_ridge(
                features[train_rows], targets[train_rows], observed[train_rows], keys,
                alpha, folds=args.folds, seed=args.seed, scale_floor=args.scale_floor,
                return_oof=True,
            )
            oof_prediction[train_selected] = oof
            prediction[validation_selected] = model.predict(features[validation[validation_selected]])
        exposure = fit_exposure_uncertainty(
            targets[training] - oof_prediction, observed[training],
            data["num_cells_filtered"][training], contexts[training],
            **control_args, scale_floor=args.scale_floor,
        )
        scale = exposure.scales(data["num_cells_filtered"][validation], contexts[validation])
        by_context = metrics_by_context(
            prediction, scale, data, validation, references, value_space
        )
        score = selection_nll(by_context)
        label = str(int(alpha))
        arrays[f"ridge_alpha_{label}_biological_variance"] = exposure.biological_variance_
        arrays[f"ridge_alpha_{label}_sampling_variance"] = exposure.sampling_variance_
        candidate = {"family": "context-specific-full-feature-linear-ridge",
                     "configuration": {"alpha": alpha}, "selectionNll": score,
                     "selectionNllDeltaVsMean": mean_report["selectionNll"] - score,
                     "byContext": by_context, "exposure": exposure_summary(exposure)}
        candidates.append(candidate)
        print(json.dumps({"event": "ridge", "alpha": alpha, "selectionNll": score}), flush=True)

    response_models = fit_grouped_oof_response_basis_grid(
        features[training], targets[training], observed[training], contexts[training],
        data["action_ids"][training], ranks=args.ranks, alphas=args.alphas,
        folds=args.folds, seed=args.seed, scale_floor=args.scale_floor,
    )
    for rank in args.ranks:
        for alpha in args.alphas:
            model = response_models[(rank, float(alpha))]
            exposure = response_exposure_from_oof_sufficient_statistics(
                model, mean_exposure, data["num_cells_filtered"][training], contexts[training]
            )
            prediction = model.predict(features[validation], contexts[validation])
            scale = exposure.scales(data["num_cells_filtered"][validation], contexts[validation])
            by_context = metrics_by_context(
                prediction, scale, data, validation, references, value_space
            )
            score = selection_nll(by_context)
            label = f"rank_{rank}_alpha_{int(alpha)}"
            arrays[f"response_{label}_biological_variance"] = exposure.biological_variance_
            arrays[f"response_{label}_sampling_variance"] = exposure.sampling_variance_
            candidate = {"family": "training-response-basis-feature-linear-ridge",
                         "configuration": {"rank": rank, "alpha": alpha},
                         "selectionNll": score,
                         "selectionNllDeltaVsMean": mean_report["selectionNll"] - score,
                         "byContext": by_context, "exposure": exposure_summary(exposure),
                         "foldAudit": list(model.fold_audit_)}
            candidates.append(candidate)
            print(json.dumps({"event": "response-basis", "rank": rank,
                              "alpha": alpha, "selectionNll": score}), flush=True)

    champion_index = min(range(len(candidates)), key=lambda index: candidates[index]["selectionNll"])
    np.savez_compressed(output / "exposure-uncertainty.npz", **arrays)
    report = {"protocol": protocol, "mean": mean_report, "candidates": candidates,
              "championIndex": champion_index, "champion": candidates[champion_index],
              "testArtifactAccessed": False, "protectedHoldoutAccessed": False,
              "elapsedSeconds": time.monotonic() - started,
              "exposureArtifact": {"path": str(output / "exposure-uncertainty.npz"),
                                   "sha256": sha256_file(output / "exposure-uncertainty.npz")}}
    write_json(output / "report.json", report)
    print(json.dumps({"event": "finished", "champion": report["champion"]["configuration"],
                      "family": report["champion"]["family"],
                      "selectionNll": report["champion"]["selectionNll"],
                      "elapsedSeconds": report["elapsedSeconds"]}), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path, default=ROOT / "data" / "derived" / "slp11-human" /
                        "replogle-k562-rpe1-author-normalized-development-v2.npz")
    result.add_argument("--features", type=Path, default=ROOT / "data" / "derived" /
                        "slp11-human-static-fusion" / "esm2-t6-plus-go-svd-v1" /
                        "human-static-esm-go-features.npz")
    result.add_argument("--output", type=Path, default=ROOT / "results" /
                        "slp11-transition" / "human-normalized-baselines-v1")
    result.add_argument("--alphas", type=float, nargs="+",
                        default=[100.0, 1000.0, 10000.0, 100000.0])
    result.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 64])
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
