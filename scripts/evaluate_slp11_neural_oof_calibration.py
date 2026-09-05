"""Evaluate frozen neural-OOF uncertainty on normalized development validation.

Only the world Gaussian scale changes. The seed-731 checkpoint, molecular means,
query/context inputs, and matched training-only mean/ridge predictors remain
fixed. This is development evaluation and never opens the routed test artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent))
import audit_slp11_normalized_candidate as audit

PARENT_RUN = Path(
    "results/slp11-transition/human-normalized-fusion-response32-exposure-seed731-v1"
)
CALIBRATION_RUN = Path(
    "results/slp11-transition/human-normalized-neural-oof-calibration-v1"
)
OUTPUT = Path(
    "results/slp11-transition/"
    "human-normalized-fusion-response32-neural-oof-scale-seed731-v1"
)


class CalibratedEvaluationError(ValueError):
    """The immutable calibrated development-evaluation contract was violated."""


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _component_scales(
    biological: np.ndarray,
    sampling: np.ndarray,
    num_cells: np.ndarray,
    context_index: np.ndarray,
) -> np.ndarray:
    counts = np.asarray(num_cells, dtype=np.float64)
    contexts = np.asarray(context_index, dtype=np.int64)
    variance = biological[contexts] + sampling[contexts] / counts[:, None]
    return np.sqrt(np.maximum(variance, audit.SCALE_FLOOR**2)).astype(np.float32)


def _point(summary: dict[str, np.ndarray]) -> dict[str, float | None]:
    adjusted = summary["adjusted_pearson"]
    return {
        "geneMacroNll": float(summary["nll"].mean()),
        "geneMacroAdjustedPearson": (
            float(np.nanmean(adjusted)) if np.isfinite(adjusted).any() else None
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    parent = Path(args.parent_run)
    calibration_run = Path(args.calibration_run)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"immutable evaluation output already exists: {output}")
    data_path = Path(args.data)
    feature_path = Path(args.features)
    if audit.sha256(data_path) != audit.DATA_SHA256:
        raise CalibratedEvaluationError("development bundle SHA-256 drift")
    if audit.sha256(feature_path) != audit.FEATURE_SHA256:
        raise CalibratedEvaluationError("fusion feature SHA-256 drift")
    parent_report = json.loads((parent / "report.json").read_text(encoding="utf-8"))
    calibration_report = json.loads(
        (calibration_run / "report.json").read_text(encoding="utf-8")
    )
    if (
        parent_report["protocol"]["test_accessed"]
        or calibration_report["testAccessed"]
        or calibration_report["outerValidationUsedForFittingCalibrationOrEpochChoice"]
    ):
        raise CalibratedEvaluationError("parent or calibration provenance permits forbidden access")
    with np.load(data_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    if len(data["split_test"]):
        raise CalibratedEvaluationError("only the train/validation development bundle is allowed")
    train = data["split_train"]
    validation = data["split_validation"]
    if set(data["action_ids"][train]) & set(data["action_ids"][validation]):
        raise CalibratedEvaluationError("training and validation interventions overlap")

    with np.load(feature_path, allow_pickle=False) as archive:
        feature_keys = list(
            zip(archive["entity_taxon"].tolist(), archive["entity_id"].tolist())
        )
        feature_values = archive["feature_values"].astype(np.float32)
    lookup = dict(zip(feature_keys, feature_values))
    x = np.stack([lookup[(9606, str(gene))] for gene in data["action_ids"]])
    Predictor, baseline_calibration, uncertainty_module = audit._load_run_modules(parent)
    predictor = Predictor(parent, device="cpu")
    with np.load(parent / "reference.npz", allow_pickle=False) as archive:
        saved = {name: archive[name] for name in archive.files}
    with np.load(calibration_run / "neural-oof-exposure.npz", allow_pickle=False) as archive:
        neural_biological = archive["biological_variance"].astype(np.float64)
        neural_sampling = archive["sampling_variance"].astype(np.float64)
        neural_queries = archive["query_ids"]
        neural_contexts = archive["context_ids"]
        sampling_from_controls = archive["sampling_from_controls"]
    if (
        not np.array_equal(saved["query_ids"], data["query_ids"])
        or not np.array_equal(neural_queries, data["query_ids"])
        or not np.array_equal(neural_contexts, data["context_ids"])
        or not sampling_from_controls.all()
    ):
        raise CalibratedEvaluationError("calibration identity or control provenance drift")

    y = data["targets"]
    observed = data["observed"]
    context = data["context_index"]
    context_ids = data["context_ids"].astype(str)
    action_ids = data["action_ids"].astype(str)
    action_keys = [(9606, gene) for gene in action_ids]
    means, ridges = [], []
    oof_mean = np.empty_like(y[train], dtype=np.float64)
    oof_ridge = np.empty_like(y[train], dtype=np.float64)
    with threadpool_limits(limits=4):
        for context_index in range(len(context_ids)):
            rows = train[context[train] == context_index]
            positions = np.flatnonzero(context[train] == context_index)
            keys = [action_keys[row] for row in rows]
            mean, oof_mean[positions] = baseline_calibration.fit_grouped_oof_mean(
                y[rows], observed[rows], keys, scale_floor=audit.SCALE_FLOOR, return_oof=True
            )
            ridge, oof_ridge[positions] = baseline_calibration.fit_grouped_oof_ridge(
                x[rows],
                y[rows],
                observed[rows],
                keys,
                audit.RIDGE_ALPHA,
                scale_floor=audit.SCALE_FLOOR,
                return_oof=True,
            )
            means.append(mean)
            ridges.append(ridge)
    control_args = {
        "control_targets": data["control_targets"],
        "control_observed": data["control_observed"],
        "control_num_cells": data["control_num_cells_filtered"],
        "control_context_index": data["control_context_index"],
    }
    mean_exposure = uncertainty_module.fit_exposure_uncertainty(
        y[train] - oof_mean,
        observed[train],
        data["num_cells_filtered"][train],
        context[train],
        **control_args,
        scale_floor=audit.SCALE_FLOOR,
    )
    ridge_exposure = uncertainty_module.fit_exposure_uncertainty(
        y[train] - oof_ridge,
        observed[train],
        data["num_cells_filtered"][train],
        context[train],
        **control_args,
        scale_floor=audit.SCALE_FLOOR,
    )

    world_prediction = np.empty((len(validation), y.shape[1]), dtype=np.float32)
    inherited_scale = np.empty_like(world_prediction)
    for start in range(0, len(validation), args.batch_size):
        positions = np.arange(start, min(start + args.batch_size, len(validation)))
        rows = validation[positions]
        local_context = context[rows]
        scale = predictor.measurement_scales(
            data["num_cells_filtered"][rows], local_context, np.arange(y.shape[1])
        )
        prediction = predictor.predict(
            x[rows],
            saved["query_features"],
            saved["reference"][local_context],
            saved["reference_scale"][local_context],
            measurement_scale=scale,
            context_features=np.broadcast_to(
                saved["context_features"],
                (len(rows), *saved["context_features"].shape),
            ),
            context_values=saved["context_values"][local_context],
            context_mask=np.ones(
                (len(rows), saved["context_values"].shape[1]), dtype=np.bool_
            ),
        )
        world_prediction[positions] = prediction["mean"]
        inherited_scale[positions] = prediction["marginal_scale"]
    neural_scale = _component_scales(
        neural_biological,
        neural_sampling,
        data["num_cells_filtered"][validation],
        context[validation],
    )

    rng = np.random.default_rng(audit.SEED)
    context_results: dict[str, object] = {}
    for context_index, context_id in enumerate(context_ids):
        positions = np.flatnonzero(context[validation] == context_index)
        rows = validation[positions]
        target = y[rows]
        mask = observed[rows]
        genes = action_ids[rows]
        reference = np.asarray(means[context_index].intercept_)
        mean_prediction = means[context_index].predict(x[rows])
        ridge_prediction = ridges[context_index].predict(x[rows])
        mean_scale = mean_exposure.scales(data["num_cells_filtered"][rows], context[rows])
        ridge_scale = ridge_exposure.scales(data["num_cells_filtered"][rows], context[rows])
        summaries = {
            "inheritedWorld": audit.gene_summaries(
                world_prediction[positions],
                target,
                mask,
                genes,
                reference,
                inherited_scale[positions],
            ),
            "neuralOofWorld": audit.gene_summaries(
                world_prediction[positions],
                target,
                mask,
                genes,
                reference,
                neural_scale[positions],
            ),
            "mean": audit.gene_summaries(
                mean_prediction, target, mask, genes, reference, mean_scale
            ),
            "ridge": audit.gene_summaries(
                ridge_prediction, target, mask, genes, reference, ridge_scale
            ),
        }
        parent_context = parent_report["results"][context_id]
        reproduced = {
            "inheritedWorldNllAbsDrift": abs(
                _point(summaries["inheritedWorld"])["geneMacroNll"]
                - parent_context["world"]["gene_macro_nll"]
            ),
            "worldAdjustedPearsonAbsDrift": abs(
                _point(summaries["inheritedWorld"])["geneMacroAdjustedPearson"]
                - parent_context["world"][
                    "gene_macro_profile_centroid_adjusted_pearson_mean"
                ]
            ),
            "meanNllAbsDrift": abs(
                _point(summaries["mean"])["geneMacroNll"]
                - parent_context["mean"]["gene_macro_nll"]
            ),
            "ridgeNllAbsDrift": abs(
                _point(summaries["ridge"])["geneMacroNll"]
                - parent_context["ridge"]["gene_macro_nll"]
            ),
        }
        if max(reproduced.values()) > 1e-6:
            raise CalibratedEvaluationError("parent point metrics did not reproduce")
        comparisons = {
            name: audit.paired_bootstrap(
                summaries["neuralOofWorld"], summaries[name], rng, audit.BOOTSTRAPS
            )
            for name in ("mean", "ridge", "inheritedWorld")
        }
        point = {name: _point(summary) for name, summary in summaries.items()}
        point["neuralOofWorld"]["deltaNllVsMean"] = (
            point["mean"]["geneMacroNll"] - point["neuralOofWorld"]["geneMacroNll"]
        )
        point["neuralOofWorld"]["deltaNllVsRidge"] = (
            point["ridge"]["geneMacroNll"] - point["neuralOofWorld"]["geneMacroNll"]
        )
        rule_passed = (
            min(
                point["neuralOofWorld"]["deltaNllVsMean"],
                point["neuralOofWorld"]["deltaNllVsRidge"],
            )
            >= 0.02
            and point["neuralOofWorld"]["geneMacroAdjustedPearson"] >= 0.10
        )
        context_results[context_id] = {
            "records": len(rows),
            "interventionGenes": len(np.unique(genes)),
            "pointMetrics": point,
            "parentMetricReproduction": reproduced,
            "developmentRulePassed": rule_passed,
            "bootstrap": comparisons,
            "calibration": {
                "inheritedWorld": audit.calibration_moments(
                    world_prediction[positions],
                    target,
                    mask,
                    inherited_scale[positions],
                    data["num_cells_filtered"][rows],
                    genes,
                ),
                "neuralOofWorld": audit.calibration_moments(
                    world_prediction[positions],
                    target,
                    mask,
                    neural_scale[positions],
                    data["num_cells_filtered"][rows],
                    genes,
                ),
            },
        }

    parent_hashes = {
        name: audit.sha256(parent / name)
        for name in (
            "model.safetensors",
            "model-config.json",
            "reference.npz",
            "exposure-uncertainty.npz",
            "protocol.json",
            "report.json",
        )
    }
    calibration_hashes = {
        name: audit.sha256(calibration_run / name)
        for name in ("neural-oof-exposure.npz", "protocol.json", "report.json")
    }
    output.mkdir(parents=True)
    for name in ("model.safetensors", "model-config.json", "reference.npz"):
        shutil.copy2(parent / name, output / name)
    shutil.copytree(parent / "source", output / "source")
    shutil.copy2(parent / "exposure-uncertainty.npz", output / "parent-exposure-uncertainty.npz")
    world_exposure_path = output / "world-exposure-uncertainty.npz"
    np.savez_compressed(
        world_exposure_path,
        world_biological_variance=neural_biological,
        world_sampling_variance=neural_sampling,
        world_query_ids=data["query_ids"],
        world_context_ids=data["context_ids"],
        world_scale_floor=np.asarray(audit.SCALE_FLOOR),
        world_sampling_from_controls=sampling_from_controls,
    )
    protocol = {
        "schema": "slp.normalized-human-neural-oof-scale-evaluation/v1",
        "label": "development evaluation",
        "scope": "frozen seed-731 world means; development validation only",
        "scaleChange": (
            "world likelihood only: inherited mean-OOF biological variance replaced by "
            "training-only neural held-gene OOF biological variance; core-control "
            "sampling variance retained"
        ),
        "fixedRule": {
            "deltaNatsAgainstMeanAndRidge": 0.02,
            "adjustedPearson": 0.10,
            "bothContextsRequired": True,
        },
        "bootstrap": {
            "seed": audit.SEED,
            "resamples": audit.BOOTSTRAPS,
            "unit": "intervention gene with all records retained",
        },
        "parentRun": {"path": str(parent), "hashes": parent_hashes},
        "calibrationRun": {"path": str(calibration_run), "hashes": calibration_hashes},
        "runner": {"path": str(Path(__file__)), "sha256": audit.sha256(Path(__file__))},
        "dataSha256": audit.DATA_SHA256,
        "featuresSha256": audit.FEATURE_SHA256,
        "worldMeansChanged": False,
        "validationScaleFittingOrAdjustment": False,
        "testAccessed": False,
        "benchmarkAccessed": False,
    }
    _write_json(output / "protocol.json", protocol)
    report = {
        "schema": "slp.normalized-human-neural-oof-scale-evaluation-result/v1",
        "label": "development evaluation",
        "contexts": context_results,
        "developmentRulePassed": all(
            result["developmentRulePassed"] for result in context_results.values()
        ),
        "worldMeanPredictionSha256": _array_sha256(world_prediction.astype("<f4")),
        "worldExposureArtifact": {
            "path": world_exposure_path.name,
            "sha256": audit.sha256(world_exposure_path),
        },
        "copiedParentFilesVerified": {
            name: audit.sha256(output / name) == parent_hashes[name]
            for name in ("model.safetensors", "model-config.json", "reference.npz")
        },
        "elapsedSeconds": time.monotonic() - started,
        "validationScaleFittingOrAdjustment": False,
        "testAccessed": False,
        "benchmarkAccessed": False,
    }
    _write_json(output / "report.json", report)
    _write_json(
        output / "parent-provenance.json",
        {"parentRun": parent_hashes, "calibrationRun": calibration_hashes},
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/derived/slp11-human/replogle-k562-rpe1-author-normalized-development-v2.npz",
    )
    parser.add_argument(
        "--features",
        default=(
            "data/derived/slp11-human-static-fusion/esm2-t6-plus-go-svd-v1/"
            "human-static-esm-go-features.npz"
        ),
    )
    parser.add_argument("--parent-run", default=str(PARENT_RUN))
    parser.add_argument("--calibration-run", default=str(CALIBRATION_RUN))
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--batch-size", type=int, default=32)
    result = run(parser.parse_args(argv))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
