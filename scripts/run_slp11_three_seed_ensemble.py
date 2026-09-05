"""Calibrate and evaluate a frozen three-seed human world-model mean ensemble.

Calibration first constructs held-gene OOF molecular means for seeds 731, 732,
and 733 using their already selected epoch counts. Only after the ensemble OOF
scale artifact is frozen are outer development-validation outcomes evaluated.
The Gaussian scale is observation uncertainty around the ensemble mean, not a
claim of full Bayesian epistemic uncertainty.
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
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent))
import audit_slp11_normalized_candidate as audit
import run_slp11_neural_oof_calibration as neural_oof

MEMBERS = ((731, 20), (732, 93), (733, 45))
RUN_PATTERN = "human-normalized-fusion-response32-exposure-seed{seed}-v1"
OUTPUT = Path(
    "results/slp11-transition/human-normalized-fusion-response32-ensemble731-733-v1"
)
SEED731_CALIBRATION_SHA256 = (
    "7f7c5cffd6a607108731232a68a202d03c242a7371d1430ac2972535a5e9170d"
)


class EnsembleExperimentError(ValueError):
    """The frozen ensemble calibration or evaluation contract was violated."""


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def average_member_oof(
    predictions: Sequence[np.ndarray], train: np.ndarray, outer_validation: np.ndarray
) -> np.ndarray:
    """Average aligned member OOF means while retaining validation as NaN."""

    if len(predictions) < 2:
        raise EnsembleExperimentError("an ensemble needs at least two OOF members")
    arrays = [np.asarray(item, dtype=np.float32) for item in predictions]
    if any(item.shape != arrays[0].shape for item in arrays):
        raise EnsembleExperimentError("member OOF prediction shapes differ")
    train = np.asarray(train, dtype=np.int64)
    outer_validation = np.asarray(outer_validation, dtype=np.int64)
    if any(not np.isfinite(item[train]).all() for item in arrays):
        raise EnsembleExperimentError("a member lacks fitting-row OOF predictions")
    if any(np.isfinite(item[outer_validation]).any() for item in arrays):
        raise EnsembleExperimentError("a member predicted outer validation before scale freeze")
    result = np.full_like(arrays[0], np.nan)
    result[train] = np.mean(np.stack([item[train] for item in arrays]), axis=0)
    return result


def _source_run(root: Path, seed: int) -> Path:
    return root / RUN_PATTERN.format(seed=seed)


def _member_hashes(run: Path) -> dict[str, str]:
    names = (
        "model.safetensors",
        "model-config.json",
        "reference.npz",
        "exposure-uncertainty.npz",
        "protocol.json",
        "report.json",
    )
    return {name: audit.sha256(run / name) for name in names}


def _load_ensemble_predictor(source_path: Path):
    spec = importlib.util.spec_from_file_location("slp11_ensemble_inference", source_path)
    if spec is None or spec.loader is None:
        raise EnsembleExperimentError("could not load ensemble inference wrapper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EnsemblePredictor


def _gene_mse(
    prediction: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    action_ids: np.ndarray,
) -> float:
    residual = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    row_mse = np.sum(np.where(observed, residual**2, 0.0), axis=1) / observed.sum(axis=1)
    values = [row_mse[action_ids == gene].mean() for gene in np.unique(action_ids)]
    return float(np.mean(values))


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    root = Path(args.results_root)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"immutable ensemble output already exists: {output}")
    runs = {seed: _source_run(root, seed) for seed, _ in MEMBERS}
    reports = {
        seed: json.loads((run / "report.json").read_text(encoding="utf-8"))
        for seed, run in runs.items()
    }
    for seed, epochs in MEMBERS:
        protocol = reports[seed]["protocol"]
        if (
            reports[seed]["best_epoch"] != epochs
            or protocol["data_sha256"] != audit.DATA_SHA256
            or protocol["features_sha256"] != audit.FEATURE_SHA256
            or protocol["args"]["seed"] != seed
            or protocol["test_accessed"]
        ):
            raise EnsembleExperimentError(f"seed-{seed} child protocol drift")
    relevant = (
        "transition_model.py",
        "transition_calibration.py",
        "transition_baselines.py",
        "response_queries.py",
        "exposure_uncertainty.py",
        "inference.py",
    )
    for name in relevant:
        hashes = {audit.sha256(run / "source" / name) for run in runs.values()}
        if len(hashes) != 1:
            raise EnsembleExperimentError(f"member source differs for {name}")

    source_modules = neural_oof._load_source_modules(runs[731])
    data, actions, static_queries = neural_oof._load_inputs(
        Path(args.data), Path(args.features)
    )
    train = data["split_train"].astype(np.int64)
    validation = data["split_validation"].astype(np.int64)
    action_ids = data["action_ids"].astype(str)
    fold_ids = source_modules["calibration"].grouped_fold_ids(
        [(9606, action_ids[row]) for row in train],
        folds=neural_oof.FOLDS,
        seed=neural_oof.SEED,
    )
    plan = neural_oof.make_fold_plan(action_ids, train, validation, fold_ids)
    if args.plan_only:
        return {
            "members": [{"seed": seed, "epochs": epochs} for seed, epochs in MEMBERS],
            "planning": neural_oof._serializable_plan(plan),
        }
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA unavailable; no executor fallback")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)

    member_parent_hashes = {seed: _member_hashes(run) for seed, run in runs.items()}
    output.mkdir(parents=True)
    protocol = {
        "schema": "slp.three-seed-ensemble-development-experiment/v1",
        "hypothesis": (
            "Averaging reproducible seed variation reduces held-gene model error enough "
            "to pass the fixed rule in both human contexts."
        ),
        "fixedRule": {
            "deltaNatsAgainstMeanAndRidge": 0.02,
            "adjustedPearson": 0.10,
            "bothContextsRequired": True,
        },
        "calibrationBeforeValidation": True,
        "oofEpochsFrozenFromChildBestEpochs": {
            str(seed): epochs for seed, epochs in MEMBERS
        },
        "foldAssignment": "three global intervention-gene folds, seed 731",
        "folds": neural_oof._serializable_plan(plan),
        "earlyStoppingWithinOof": False,
        "dataSha256": audit.DATA_SHA256,
        "featuresSha256": audit.FEATURE_SHA256,
        "children": {
            str(seed): {
                "path": str(runs[seed]),
                "hashes": member_parent_hashes[seed],
            }
            for seed, _epochs in MEMBERS
        },
        "runner": {"path": str(Path(__file__)), "sha256": audit.sha256(Path(__file__))},
        "ensembleInference": {
            "path": str(Path(args.ensemble_inference)),
            "sha256": audit.sha256(Path(args.ensemble_inference)),
        },
        "validationUsedForFittingCalibrationOrEpochChoice": False,
        "testAccessed": False,
        "benchmarkAccessed": False,
    }
    _write_json(output / "protocol.json", protocol)

    member_oof: list[np.ndarray] = []
    member_fold_metadata: dict[str, object] = {}
    individual_exposures: dict[int, object] = {}
    seed731_rematerialization_drift: float | None = None
    control_args = {
        "control_targets": data["control_targets"],
        "control_observed": data["control_observed"],
        "control_num_cells": data["control_num_cells_filtered"],
        "control_context_index": data["control_context_index"],
    }
    for seed, epochs in MEMBERS:
        fold_metadata: list[dict[str, object]] = []

        def predict_fold(
            fitting: np.ndarray,
            held: np.ndarray,
            fold: int,
            _seed: int = seed,
            _epochs: int = epochs,
            _fold_metadata: list[dict[str, object]] = fold_metadata,
        ) -> np.ndarray:
            prediction, metadata = neural_oof._fit_predict_fold(
                fitting,
                held,
                fold,
                data=data,
                actions=actions,
                static_queries=static_queries,
                modules=source_modules,
                device=device,
                max_seconds=args.max_seconds,
                started=started,
                model_seed=_seed,
                epochs=_epochs,
            )
            _fold_metadata.append(metadata)
            print(
                json.dumps(
                    {
                        "event": "ensemble-oof-fold-complete",
                        "seed": _seed,
                        "epochs": _epochs,
                        "fold": fold,
                        "seconds": round(time.monotonic() - started, 2),
                    }
                ),
                flush=True,
            )
            return prediction

        predictions = neural_oof.collect_oof_predictions(
            len(data["targets"]), data["targets"].shape[1], plan, predict_fold
        )
        member_oof.append(predictions)
        member_fold_metadata[str(seed)] = sorted(
            fold_metadata, key=lambda item: item["fold"]
        )
        exposure = source_modules["uncertainty"].fit_exposure_uncertainty(
            data["targets"][train] - predictions[train],
            data["observed"][train],
            data["num_cells_filtered"][train],
            data["context_index"][train],
            **control_args,
            scale_floor=neural_oof.SCALE_FLOOR,
        )
        individual_exposures[seed] = exposure
        if seed == 731:
            prior_path = Path(args.seed731_calibration) / "neural-oof-exposure.npz"
            if audit.sha256(prior_path) != SEED731_CALIBRATION_SHA256:
                raise EnsembleExperimentError("prior seed-731 calibration artifact drift")
            with np.load(prior_path, allow_pickle=False) as archive:
                drift = max(
                    float(
                        np.max(
                            np.abs(
                                archive["biological_variance"]
                                - exposure.biological_variance_
                            )
                        )
                    ),
                    float(
                        np.max(
                            np.abs(
                                archive["sampling_variance"]
                                - exposure.sampling_variance_
                            )
                        )
                    ),
                )
            seed731_rematerialization_drift = drift
            if drift > 1e-6:
                raise EnsembleExperimentError(
                    f"rematerialized seed-731 OOF calibration differs: max abs {drift:.9g}"
                )

    ensemble_oof = average_member_oof(member_oof, train, validation)
    ensemble_exposure = source_modules["uncertainty"].fit_exposure_uncertainty(
        data["targets"][train] - ensemble_oof[train],
        data["observed"][train],
        data["num_cells_filtered"][train],
        data["context_index"][train],
        **control_args,
        scale_floor=neural_oof.SCALE_FLOOR,
    )

    # Freeze the ensemble and its observation scale before indexing validation outcomes.
    members_dir = output / "members"
    members_dir.mkdir()
    member_manifest = []
    for seed, epochs in MEMBERS:
        member_dir = members_dir / f"seed{seed}"
        member_dir.mkdir()
        for name in ("model.safetensors", "model-config.json", "reference.npz"):
            shutil.copy2(runs[seed] / name, member_dir / name)
        member_manifest.append(
            {
                "seed": seed,
                "frozenBestEpoch": epochs,
                "artifactPath": f"members/seed{seed}",
                "parentPath": str(runs[seed]),
                "parentHashes": member_parent_hashes[seed],
                "copiedHashes": {
                    name: audit.sha256(member_dir / name)
                    for name in ("model.safetensors", "model-config.json", "reference.npz")
                },
            }
        )
    shutil.copytree(runs[731] / "source", output / "source")
    shutil.copy2(
        Path(args.ensemble_inference), output / "source" / "ensemble_inference.py"
    )
    exposure_path = output / "ensemble-exposure-uncertainty.npz"
    np.savez_compressed(
        exposure_path,
        ensemble_biological_variance=ensemble_exposure.biological_variance_,
        ensemble_sampling_variance=ensemble_exposure.sampling_variance_,
        ensemble_residual_counts=ensemble_exposure.residual_counts_,
        ensemble_control_counts=ensemble_exposure.control_counts_,
        ensemble_sampling_from_controls=ensemble_exposure.sampling_from_controls_,
        ensemble_query_ids=data["query_ids"],
        ensemble_context_ids=data["context_ids"],
        ensemble_scale_floor=np.asarray(neural_oof.SCALE_FLOOR),
    )
    manifest = {
        "schema": "slp.transition-world-mean-ensemble/v1",
        "label": "development ensemble; not release or Bayesian posterior",
        "members": member_manifest,
        "meanCombination": "arithmetic mean of three independently trained molecular means",
        "latentCombination": "none; member states retained on a separate member axis",
        "uncertainty": (
            "Gaussian observation uncertainty fitted from ensemble held-gene OOF residuals "
            "plus source core controls; no full Bayesian epistemic claim"
        ),
        "exposureArtifact": {
            "path": exposure_path.name,
            "sha256": audit.sha256(exposure_path),
        },
        "queryCount": len(data["query_ids"]),
        "contexts": data["context_ids"].tolist(),
    }
    _write_json(output / "ensemble-manifest.json", manifest)
    calibration_report = {
        "schema": "slp.three-seed-ensemble-oof-calibration/v1",
        "label": "fitting-only development calibration",
        "members": [{"seed": seed, "epochs": epochs} for seed, epochs in MEMBERS],
        "folds": neural_oof._serializable_plan(plan),
        "memberFoldFits": member_fold_metadata,
        "ensembleOofPredictionSha256": _array_sha256(
            ensemble_oof[train].astype("<f4")
        ),
        "seed731RematerializedCalibrationMaxAbsDrift": seed731_rematerialization_drift,
        "seed731RematerializationTolerance": 1e-6,
        "componentProvenance": ensemble_exposure.component_provenance,
        "samplingFromControlsFraction": float(
            ensemble_exposure.sampling_from_controls_.mean()
        ),
        "identifiabilityWarning": ensemble_exposure.identifiability_warning,
        "validationOutcomesUsed": False,
        "testAccessed": False,
        "benchmarkAccessed": False,
    }
    _write_json(output / "calibration-report.json", calibration_report)
    # Calibration is now immutable. Load the portable wrapper for validation means.
    EnsemblePredictor = _load_ensemble_predictor(
        output / "source" / "ensemble_inference.py"
    )
    predictor = EnsemblePredictor(output, device="cpu")
    ensemble_prediction = np.empty(
        (len(validation), data["targets"].shape[1]), dtype=np.float32
    )
    member_predictions = np.empty(
        (len(MEMBERS), len(validation), data["targets"].shape[1]), dtype=np.float32
    )
    ensemble_scale = np.empty_like(ensemble_prediction)
    for start in range(0, len(validation), args.batch_size):
        positions = np.arange(start, min(start + args.batch_size, len(validation)))
        rows = validation[positions]
        result = predictor.predict(
            actions[rows], data["num_cells_filtered"][rows], data["context_index"][rows]
        )
        ensemble_prediction[positions] = result["mean"]
        member_predictions[:, positions] = result["member_means"]
        ensemble_scale[positions] = result["marginal_scale"]

    # Reproduce strongest matched references from training only.
    contexts = data["context_index"]
    observed = data["observed"]
    targets = data["targets"]
    keys = [(9606, gene) for gene in action_ids]
    means, ridges = [], []
    oof_mean = np.empty_like(targets[train], dtype=np.float64)
    oof_ridge = np.empty_like(targets[train], dtype=np.float64)
    with threadpool_limits(limits=4):
        for context_index in range(len(data["context_ids"])):
            rows = train[contexts[train] == context_index]
            positions = np.flatnonzero(contexts[train] == context_index)
            local_keys = [keys[row] for row in rows]
            mean, oof_mean[positions] = source_modules[
                "calibration"
            ].fit_grouped_oof_mean(
                targets[rows],
                observed[rows],
                local_keys,
                scale_floor=neural_oof.SCALE_FLOOR,
                return_oof=True,
            )
            ridge, oof_ridge[positions] = source_modules[
                "calibration"
            ].fit_grouped_oof_ridge(
                actions[rows],
                targets[rows],
                observed[rows],
                local_keys,
                audit.RIDGE_ALPHA,
                scale_floor=neural_oof.SCALE_FLOOR,
                return_oof=True,
            )
            means.append(mean)
            ridges.append(ridge)
    mean_exposure = source_modules["uncertainty"].fit_exposure_uncertainty(
        targets[train] - oof_mean,
        observed[train],
        data["num_cells_filtered"][train],
        contexts[train],
        **control_args,
        scale_floor=neural_oof.SCALE_FLOOR,
    )
    ridge_exposure = source_modules["uncertainty"].fit_exposure_uncertainty(
        targets[train] - oof_ridge,
        observed[train],
        data["num_cells_filtered"][train],
        contexts[train],
        **control_args,
        scale_floor=neural_oof.SCALE_FLOOR,
    )

    rng = np.random.default_rng(audit.SEED)
    results: dict[str, object] = {}
    for context_index, context_id in enumerate(data["context_ids"].astype(str)):
        positions = np.flatnonzero(contexts[validation] == context_index)
        rows = validation[positions]
        genes = action_ids[rows]
        reference = means[context_index].intercept_
        mean_prediction = means[context_index].predict(actions[rows])
        ridge_prediction = ridges[context_index].predict(actions[rows])
        mean_scale = mean_exposure.scales(
            data["num_cells_filtered"][rows], contexts[rows]
        )
        ridge_scale = ridge_exposure.scales(
            data["num_cells_filtered"][rows], contexts[rows]
        )
        predictions = {
            "ensemble": ensemble_prediction[positions],
            "mean": mean_prediction,
            "ridge": ridge_prediction,
        }
        scales = {
            "ensemble": ensemble_scale[positions],
            "mean": mean_scale,
            "ridge": ridge_scale,
        }
        summaries = {
            name: audit.gene_summaries(
                prediction,
                targets[rows],
                observed[rows],
                genes,
                reference,
                scales[name],
            )
            for name, prediction in predictions.items()
        }
        points = {
            name: {
                "geneMacroNll": float(summary["nll"].mean()),
                "geneMacroAdjustedPearson": (
                    float(np.nanmean(summary["adjusted_pearson"]))
                    if np.isfinite(summary["adjusted_pearson"]).any()
                    else None
                ),
                "geneMacroMse": _gene_mse(
                    predictions[name], targets[rows], observed[rows], genes
                ),
            }
            for name, summary in summaries.items()
        }
        points["ensemble"]["deltaNllVsMean"] = (
            points["mean"]["geneMacroNll"] - points["ensemble"]["geneMacroNll"]
        )
        points["ensemble"]["deltaNllVsRidge"] = (
            points["ridge"]["geneMacroNll"] - points["ensemble"]["geneMacroNll"]
        )
        rule = (
            min(
                points["ensemble"]["deltaNllVsMean"],
                points["ensemble"]["deltaNllVsRidge"],
            )
            >= 0.02
            and points["ensemble"]["geneMacroAdjustedPearson"] >= 0.10
        )
        member_disagreement = np.var(
            member_predictions[:, positions].astype(np.float64), axis=0
        )
        ensemble_residual_mse = np.mean(
            (
                ensemble_prediction[positions].astype(np.float64)
                - targets[rows].astype(np.float64)
            )
            ** 2
        )
        results[context_id] = {
            "records": len(rows),
            "interventionGenes": len(np.unique(genes)),
            "pointMetrics": points,
            "developmentRulePassed": rule,
            "bootstrap": {
                baseline: audit.paired_bootstrap(
                    summaries["ensemble"], summaries[baseline], rng, audit.BOOTSTRAPS
                )
                for baseline in ("mean", "ridge")
            },
            "calibration": audit.calibration_moments(
                ensemble_prediction[positions],
                targets[rows],
                observed[rows],
                ensemble_scale[positions],
                data["num_cells_filtered"][rows],
                genes,
            ),
            "seedDisagreementDiagnostic": {
                "meanMemberVariance": float(member_disagreement.mean()),
                "ensembleResidualMse": float(ensemble_residual_mse),
                "memberVarianceFractionOfResidualMse": float(
                    member_disagreement.mean() / ensemble_residual_mse
                ),
                "interpretation": (
                    "descriptive variation among three fitted means; not Bayesian "
                    "epistemic variance and not added to observation uncertainty"
                ),
            },
        }
    report = {
        "schema": "slp.three-seed-ensemble-development-result/v1",
        "label": "development evaluation",
        "results": results,
        "developmentRulePassed": all(
            item["developmentRulePassed"] for item in results.values()
        ),
        "ensembleMeanPredictionSha256": _array_sha256(
            ensemble_prediction.astype("<f4")
        ),
        "memberMeanPredictionSha256": {
            str(seed): _array_sha256(member_predictions[index].astype("<f4"))
            for index, (seed, _epochs) in enumerate(MEMBERS)
        },
        "elapsedSeconds": time.monotonic() - started,
        "validationUsedForFittingCalibrationOrEpochChoice": False,
        "testAccessed": False,
        "benchmarkAccessed": False,
    }
    _write_json(output / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/slp11-transition")
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
    parser.add_argument(
        "--seed731-calibration",
        default="results/slp11-transition/human-normalized-neural-oof-calibration-v1",
    )
    parser.add_argument(
        "--ensemble-inference",
        default="modules/slp-1-1-world-transition-v1/ensemble_inference.py",
    )
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-seconds", type=int, default=300)
    parser.add_argument("--plan-only", action="store_true")
    result = run(parser.parse_args(argv))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
