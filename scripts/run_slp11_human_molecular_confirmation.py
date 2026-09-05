"""Run the first frozen human molecular confirmation for the three-seed ensemble.

The protocol and synthetic contract report are written before the test NPZ is
opened. Mean and ridge baselines, including grouped OOF exposure uncertainty,
are fit exclusively from the original development training indices. Test
outcomes are evaluated once and never used to alter a model, scale, or rule.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
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

ENSEMBLE_MANIFEST_SHA256 = (
    "a972d994f80c124f948b9b4a313d9e76bdd5c1a3477ebc4082c143ae96c50a70"
)
DEVELOPMENT_SHA256 = "88de5164fca4e2504ac5b459ab4226c161eb586dd04700d5784da4bb53048659"
FEATURE_SHA256 = "b3de49e18d3c75676985b8790d1ce85de0d87d526bbd7c0c5b555828a1fb11a0"
TEST_SHA256 = "7bf755248513f41c552e4a4bde2d5958f0f5ea4243eeeb5ec77128642b0697d1"
RIDGE_ALPHA = 10_000.0
FOLDS = 3
SEED = 731
BOOTSTRAPS = 1_000
SCALE_FLOOR = 0.05
OUTPUT = Path(
    "results/slp11-transition/"
    "human-normalized-fusion-response32-ensemble731-733-molecular-confirmation-v1"
)


class MolecularConfirmationError(ValueError):
    """The frozen human molecular confirmation contract was violated."""


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): audit.sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _load_ensemble_class(source: Path):
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("confirmation_ensemble_inference", source)
    if spec is None or spec.loader is None:
        raise MolecularConfirmationError("could not load pinned ensemble inference")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EnsemblePredictor, module.combine_member_outputs


def synthetic_contract_checks(
    combine_member_outputs: object, grouped_fold_ids: object
) -> dict[str, object]:
    """Exercise mean-only combination, separate states, and gene-grouped folds."""

    outputs = [
        {
            "mean": np.full((2, 3), value, dtype=np.float32),
            "state": np.full((2, 4), value * 10, dtype=np.float32),
        }
        for value in (1.0, 2.0, 3.0)
    ]
    combined = combine_member_outputs(outputs, np.full((2, 3), 0.5, dtype=np.float32))
    if not np.allclose(combined["mean"], 2.0) or combined["member_states"].shape != (3, 2, 4):
        raise MolecularConfirmationError("synthetic ensemble mean/state contract failed")
    keys = [(9606, "A"), (9606, "A"), (9606, "B"), (9606, "C")]
    assignments = grouped_fold_ids(keys, folds=3, seed=SEED)
    if assignments[0] != assignments[1] or len(np.unique(assignments)) != 3:
        raise MolecularConfirmationError("synthetic grouped-fold contract failed")
    return {
        "passed": True,
        "ensembleMean": combined["mean"].tolist(),
        "memberStateShape": list(combined["member_states"].shape),
        "groupedFoldAssignments": assignments.tolist(),
        "testArtifactOpened": False,
    }


def load_test_after_protocol(
    test_path: Path, protocol_path: Path, synthetic_path: Path
) -> dict[str, np.ndarray]:
    """Open the test arrays only after both frozen pre-access records exist."""

    if not protocol_path.is_file() or not synthetic_path.is_file():
        raise MolecularConfirmationError("protocol and synthetic checks must precede test access")
    synthetic = json.loads(synthetic_path.read_text(encoding="utf-8"))
    if synthetic.get("passed") is not True or synthetic.get("testArtifactOpened") is not False:
        raise MolecularConfirmationError("synthetic pre-access checks are incomplete")
    if audit.sha256(test_path) != TEST_SHA256:
        raise MolecularConfirmationError("reserved molecular test SHA-256 drift")
    with np.load(test_path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def fixed_rule(
    ensemble_nll: float,
    mean_nll: float,
    ridge_nll: float,
    adjusted_pearson: float,
) -> dict[str, object]:
    delta_mean = mean_nll - ensemble_nll
    delta_ridge = ridge_nll - ensemble_nll
    return {
        "deltaNllVsMean": delta_mean,
        "deltaNllVsRidge": delta_ridge,
        "adjustedPearson": adjusted_pearson,
        "passed": min(delta_mean, delta_ridge) >= 0.02 and adjusted_pearson >= 0.10,
    }


def _gene_mse(
    prediction: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    action_ids: np.ndarray,
) -> float:
    residual = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    row_mse = np.sum(np.where(observed, residual**2, 0.0), axis=1) / observed.sum(axis=1)
    return float(
        np.mean([row_mse[action_ids == gene].mean() for gene in np.unique(action_ids)])
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    ensemble_path = Path(args.ensemble)
    development_path = Path(args.development)
    feature_path = Path(args.features)
    test_path = Path(args.test)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"immutable confirmation output already exists: {output}")
    if audit.sha256(ensemble_path / "ensemble-manifest.json") != ENSEMBLE_MANIFEST_SHA256:
        raise MolecularConfirmationError("ensemble manifest SHA-256 drift")
    if audit.sha256(development_path) != DEVELOPMENT_SHA256:
        raise MolecularConfirmationError("development snapshot SHA-256 drift")
    if audit.sha256(feature_path) != FEATURE_SHA256:
        raise MolecularConfirmationError("fusion feature SHA-256 drift")
    if audit.sha256(test_path) != TEST_SHA256:
        raise MolecularConfirmationError("reserved molecular test SHA-256 drift")

    ensemble_hashes = _tree_hashes(ensemble_path)
    output.mkdir(parents=True)
    source_output = output / "source"
    source_output.mkdir()
    runner_copy = source_output / Path(__file__).name
    audit_copy = source_output / Path(audit.__file__).name
    shutil.copy2(Path(__file__), runner_copy)
    shutil.copy2(Path(audit.__file__), audit_copy)
    protocol = {
        "schema": "slp.human-molecular-confirmation-protocol/v1",
        "label": "first reserved human molecular confirmation",
        "hypothesis": (
            "The frozen three-seed molecular-mean ensemble transfers to reserved "
            "intervention genes in both K562 and RPE1."
        ),
        "fixedRule": {
            "eachContext": {
                "geneMacroNllGainAgainstMean": 0.02,
                "geneMacroNllGainAgainstFeatureRidge": 0.02,
                "geneMacroCentroidAdjustedPearson": 0.10,
            },
            "allContextsRequired": True,
            "decisionUsesPointMetricsOnly": True,
        },
        "bootstrap": {
            "resamples": BOOTSTRAPS,
            "seed": SEED,
            "unit": "intervention gene; retain every record for each sampled gene",
            "role": "uncertainty interval only; not an additional decision threshold",
        },
        "baseline": {
            "fitRows": "original development split_train only",
            "trainingRecords": 3281,
            "ridgeAlpha": RIDGE_ALPHA,
            "scale": (
                "three-fold gene-grouped OOF residuals seed 731 plus unchanged "
                "development core controls"
            ),
            "validationRefitOrTuning": False,
        },
        "inputs": {
            "ensemble": {
                "path": str(ensemble_path),
                "manifestSha256": ENSEMBLE_MANIFEST_SHA256,
                "fileHashes": ensemble_hashes,
            },
            "development": {
                "path": str(development_path),
                "sha256": DEVELOPMENT_SHA256,
            },
            "features": {"path": str(feature_path), "sha256": FEATURE_SHA256},
            "reservedMolecularTest": {
                "path": str(test_path),
                "sha256": TEST_SHA256,
                "targetsLoadedBeforeProtocol": False,
            },
        },
        "runner": {
            "path": str(Path(__file__)),
            "sha256": audit.sha256(Path(__file__)),
            "sourceCopies": {
                runner_copy.relative_to(output).as_posix(): audit.sha256(runner_copy),
                audit_copy.relative_to(output).as_posix(): audit.sha256(audit_copy),
            },
        },
        "ensembleOrRuleMayChangeAfterTest": False,
        "syntheticChecksRequiredBeforeTest": True,
        "slBenchmarkAccessed": False,
    }
    protocol_path = output / "protocol.json"
    _write_json(protocol_path, protocol)

    Predictor, combine = _load_ensemble_class(
        ensemble_path / "source" / "ensemble_inference.py"
    )
    # The ensemble wrapper pins the artifact's self-contained source directory.
    baseline_calibration = importlib.import_module("transition_calibration")
    uncertainty_module = importlib.import_module("exposure_uncertainty")
    synthetic = synthetic_contract_checks(combine, baseline_calibration.grouped_fold_ids)
    synthetic_path = output / "synthetic-checks.json"
    _write_json(synthetic_path, synthetic)

    # Fit every comparator and its uncertainty before opening the test arrays.
    with np.load(development_path, allow_pickle=False) as archive:
        development = {name: archive[name] for name in archive.files}
    train = development["split_train"]
    if len(train) != 3281 or len(development["split_test"]):
        raise MolecularConfirmationError("development training split count drift")
    with np.load(feature_path, allow_pickle=False) as archive:
        keys = list(zip(archive["entity_taxon"].tolist(), archive["entity_id"].tolist()))
        feature_values = archive["feature_values"].astype(np.float32)
    lookup = dict(zip(keys, feature_values))
    development_actions = np.stack(
        [lookup[(9606, str(gene))] for gene in development["action_ids"]]
    )
    y_train = development["targets"][train]
    observed_train = development["observed"][train]
    context_train = development["context_index"][train]
    action_ids_train = development["action_ids"][train].astype(str)
    means, ridges = [], []
    oof_mean = np.empty_like(y_train, dtype=np.float64)
    oof_ridge = np.empty_like(y_train, dtype=np.float64)
    with threadpool_limits(limits=4):
        for context_index in range(len(development["context_ids"])):
            local = np.flatnonzero(context_train == context_index)
            rows = train[local]
            local_keys = [(9606, action_ids_train[position]) for position in local]
            mean, oof_mean[local] = baseline_calibration.fit_grouped_oof_mean(
                development["targets"][rows],
                development["observed"][rows],
                local_keys,
                folds=FOLDS,
                seed=SEED,
                scale_floor=SCALE_FLOOR,
                return_oof=True,
            )
            ridge, oof_ridge[local] = baseline_calibration.fit_grouped_oof_ridge(
                development_actions[rows],
                development["targets"][rows],
                development["observed"][rows],
                local_keys,
                RIDGE_ALPHA,
                folds=FOLDS,
                seed=SEED,
                scale_floor=SCALE_FLOOR,
                return_oof=True,
            )
            means.append(mean)
            ridges.append(ridge)
    controls = {
        "control_targets": development["control_targets"],
        "control_observed": development["control_observed"],
        "control_num_cells": development["control_num_cells_filtered"],
        "control_context_index": development["control_context_index"],
    }
    mean_exposure = uncertainty_module.fit_exposure_uncertainty(
        y_train - oof_mean,
        observed_train,
        development["num_cells_filtered"][train],
        context_train,
        **controls,
        scale_floor=SCALE_FLOOR,
    )
    ridge_exposure = uncertainty_module.fit_exposure_uncertainty(
        y_train - oof_ridge,
        observed_train,
        development["num_cells_filtered"][train],
        context_train,
        **controls,
        scale_floor=SCALE_FLOOR,
    )
    baseline_path = output / "baseline-exposure-uncertainty.npz"
    np.savez_compressed(
        baseline_path,
        mean_biological_variance=mean_exposure.biological_variance_,
        mean_sampling_variance=mean_exposure.sampling_variance_,
        ridge_biological_variance=ridge_exposure.biological_variance_,
        ridge_sampling_variance=ridge_exposure.sampling_variance_,
        query_ids=development["query_ids"],
        context_ids=development["context_ids"],
        scale_floor=np.asarray(SCALE_FLOOR),
    )
    shutil.copy2(
        ensemble_path / "ensemble-exposure-uncertainty.npz",
        output / "frozen-ensemble-exposure-uncertainty.npz",
    )
    predictor = Predictor(ensemble_path, device="cpu")

    # First and only interpreted load of the reserved molecular test.
    test = load_test_after_protocol(test_path, protocol_path, synthetic_path)
    required = {
        "targets",
        "observed",
        "action_ids",
        "query_ids",
        "context_index",
        "context_ids",
        "record_ids",
        "num_cells_filtered",
        "split_train",
        "split_validation",
        "split_test",
        "target_value_space",
    }
    if not required.issubset(test):
        raise MolecularConfirmationError("reserved molecular test schema is incomplete")
    records = len(test["action_ids"])
    if (
        len(test["split_train"])
        or len(test["split_validation"])
        or not np.array_equal(test["split_test"], np.arange(records))
    ):
        raise MolecularConfirmationError("reserved molecular test routing drift")
    if (
        not np.array_equal(test["query_ids"], development["query_ids"])
        or not np.array_equal(test["context_ids"], development["context_ids"])
        or not np.array_equal(
            test["target_value_space"], development["target_value_space"]
        )
        or set(test["action_ids"].tolist()) & set(development["action_ids"].tolist())
    ):
        raise MolecularConfirmationError("reserved molecular identity isolation failed")
    if not np.isfinite(test["targets"][test["observed"]]).all():
        raise MolecularConfirmationError("reserved observed outcomes are nonfinite")

    test_actions = np.stack([lookup[(9606, str(gene))] for gene in test["action_ids"]])
    ensemble_prediction = np.empty_like(test["targets"], dtype=np.float32)
    ensemble_scale = np.empty_like(test["targets"], dtype=np.float32)
    for start in range(0, records, args.batch_size):
        rows = np.arange(start, min(start + args.batch_size, records))
        prediction = predictor.predict(
            test_actions[rows],
            test["num_cells_filtered"][rows],
            test["context_index"][rows],
        )
        ensemble_prediction[rows] = prediction["mean"]
        ensemble_scale[rows] = prediction["marginal_scale"]
    mean_prediction = np.empty_like(test["targets"], dtype=np.float64)
    ridge_prediction = np.empty_like(test["targets"], dtype=np.float64)
    mean_scale = np.empty_like(test["targets"], dtype=np.float64)
    ridge_scale = np.empty_like(test["targets"], dtype=np.float64)
    for context_index in range(len(test["context_ids"])):
        rows = np.flatnonzero(test["context_index"] == context_index)
        mean_prediction[rows] = means[context_index].predict(test_actions[rows])
        ridge_prediction[rows] = ridges[context_index].predict(test_actions[rows])
        mean_scale[rows] = mean_exposure.scales(
            test["num_cells_filtered"][rows], test["context_index"][rows]
        )
        ridge_scale[rows] = ridge_exposure.scales(
            test["num_cells_filtered"][rows], test["context_index"][rows]
        )

    prediction_path = output / "predictions.npz"
    np.savez_compressed(
        prediction_path,
        ensemble_mean=ensemble_prediction,
        ensemble_scale=ensemble_scale,
        mean_baseline=mean_prediction.astype(np.float32),
        mean_baseline_scale=mean_scale.astype(np.float32),
        ridge_baseline=ridge_prediction.astype(np.float32),
        ridge_baseline_scale=ridge_scale.astype(np.float32),
        action_ids=test["action_ids"],
        query_ids=test["query_ids"],
        context_index=test["context_index"],
        context_ids=test["context_ids"],
        record_ids=test["record_ids"],
        num_cells_filtered=test["num_cells_filtered"],
        observed=test["observed"],
    )

    rng = np.random.default_rng(SEED)
    results: dict[str, object] = {}
    for context_index, context_id in enumerate(test["context_ids"].astype(str)):
        rows = np.flatnonzero(test["context_index"] == context_index)
        genes = test["action_ids"][rows].astype(str)
        reference = means[context_index].intercept_
        predictions = {
            "ensemble": ensemble_prediction[rows],
            "mean": mean_prediction[rows],
            "ridge": ridge_prediction[rows],
        }
        scales = {
            "ensemble": ensemble_scale[rows],
            "mean": mean_scale[rows],
            "ridge": ridge_scale[rows],
        }
        summaries = {
            name: audit.gene_summaries(
                prediction,
                test["targets"][rows],
                test["observed"][rows],
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
                    predictions[name],
                    test["targets"][rows],
                    test["observed"][rows],
                    genes,
                ),
            }
            for name, summary in summaries.items()
        }
        decision = fixed_rule(
            points["ensemble"]["geneMacroNll"],
            points["mean"]["geneMacroNll"],
            points["ridge"]["geneMacroNll"],
            points["ensemble"]["geneMacroAdjustedPearson"],
        )
        results[context_id] = {
            "records": len(rows),
            "interventionGenes": len(np.unique(genes)),
            "pointMetrics": points,
            "fixedRule": decision,
            "bootstrap": {
                baseline: audit.paired_bootstrap(
                    summaries["ensemble"], summaries[baseline], rng, BOOTSTRAPS
                )
                for baseline in ("mean", "ridge")
            },
            "calibrationByCellCount": {
                name: audit.calibration_moments(
                    predictions[name],
                    test["targets"][rows],
                    test["observed"][rows],
                    scales[name],
                    test["num_cells_filtered"][rows],
                    genes,
                )
                for name in predictions
            },
        }
    report = {
        "schema": "slp.human-molecular-confirmation-result/v1",
        "label": "first reserved human molecular confirmation",
        "results": results,
        "fixedRulePassed": all(item["fixedRule"]["passed"] for item in results.values()),
        "decisionUsesPointMetricsOnly": True,
        "bootstrapUsedAsThreshold": False,
        "artifacts": {
            "predictions": {
                "path": prediction_path.name,
                "sha256": audit.sha256(prediction_path),
            },
            "baselineExposure": {
                "path": baseline_path.name,
                "sha256": audit.sha256(baseline_path),
            },
            "frozenEnsembleExposureSha256": audit.sha256(
                output / "frozen-ensemble-exposure-uncertainty.npz"
            ),
        },
        "testEvaluations": 1,
        "modelOrRuleChangedAfterOutcome": False,
        "slBenchmarkAccessed": False,
        "elapsedSeconds": time.monotonic() - started,
    }
    _write_json(output / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ensemble",
        default=(
            "results/slp11-transition/"
            "human-normalized-fusion-response32-ensemble731-733-v1"
        ),
    )
    parser.add_argument(
        "--development",
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
        "--test",
        default="data/derived/slp11-human/replogle-k562-rpe1-author-normalized-test-only-v2.npz",
    )
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--batch-size", type=int, default=24)
    result = run(parser.parse_args(argv))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
