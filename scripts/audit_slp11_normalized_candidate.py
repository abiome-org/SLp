"""Audit normalized human development-validation predictions with gene bootstrap.

This script accepts only the train/validation development bundle. It reloads the
completed world model through the inference code retained with that run,
recomputes the alpha-10000 ridge and grouped fitting-only OOF exposure scales,
and never refits or recalibrates on validation outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

SEED = 731
BOOTSTRAPS = 1_000
RIDGE_ALPHA = 10_000.0
SCALE_FLOOR = 0.05
DATA_SHA256 = "88de5164fca4e2504ac5b459ab4226c161eb586dd04700d5784da4bb53048659"
FEATURE_SHA256 = "b3de49e18d3c75676985b8790d1ce85de0d87d526bbd7c0c5b555828a1fb11a0"
VALUE_SPACE = "author-per-gemgroup-core-control-z-score-pseudobulk-mean-v1"
COUNT_BINS = (("lt30", 0.0, 30.0), ("30to99", 30.0, 100.0), ("ge100", 100.0, math.inf))


class CandidateAuditError(ValueError):
    """An input violates the frozen development-audit contract."""


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pearson_rows(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return row Pearson correlations, NaN when either vector is constant."""

    result = np.full(left.shape[0], np.nan, dtype=np.float64)
    for row in range(left.shape[0]):
        selected = mask[row]
        x = left[row, selected].astype(np.float64, copy=False)
        y = right[row, selected].astype(np.float64, copy=False)
        x = x - x.mean()
        y = y - y.mean()
        denominator = math.sqrt(float(x @ x) * float(y @ y))
        if denominator > np.finfo(np.float64).eps:
            result[row] = float((x @ y) / denominator)
    return result


def gene_summaries(
    prediction: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    action_ids: np.ndarray,
    reference: np.ndarray,
    scale: np.ndarray,
) -> dict[str, np.ndarray]:
    """Collapse row diagnostics to intervention genes exactly as gene-macro scoring."""

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.bool_)
    scale = np.broadcast_to(np.asarray(scale, dtype=np.float64), target.shape)
    reference = np.broadcast_to(np.asarray(reference, dtype=np.float64), target.shape)
    if prediction.shape != target.shape or observed.shape != target.shape:
        raise CandidateAuditError("prediction, target, and observed shapes must match")
    if np.any(~np.isfinite(prediction[observed])) or np.any(scale[observed] <= 0):
        raise CandidateAuditError("observed predictions and scales must be finite and positive")

    residual = prediction - target
    component = 0.5 * (
        math.log(2.0 * math.pi) + 2.0 * np.log(scale) + np.square(residual / scale)
    )
    counts = observed.sum(axis=1)
    if np.any(counts == 0):
        raise CandidateAuditError("every validation record must have an observed target")
    row_nll = np.where(observed, component, 0.0).sum(axis=1) / counts
    adjusted = _pearson_rows(prediction - reference, target - reference, observed)

    genes = np.asarray(sorted(set(map(str, action_ids.tolist()))))
    gene_nll = np.empty(len(genes), dtype=np.float64)
    gene_r = np.empty(len(genes), dtype=np.float64)
    record_counts = np.empty(len(genes), dtype=np.int64)
    string_ids = action_ids.astype(str)
    for index, gene in enumerate(genes):
        rows = string_ids == gene
        record_counts[index] = int(rows.sum())
        gene_nll[index] = row_nll[rows].mean()
        supported = adjusted[rows & np.isfinite(adjusted)]
        gene_r[index] = supported.mean() if supported.size else np.nan
    return {
        "gene_ids": genes,
        "nll": gene_nll,
        "adjusted_pearson": gene_r,
        "record_counts": record_counts,
    }


def bootstrap_mean(values: np.ndarray, draws: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or draws.ndim != 2:
        raise CandidateAuditError("bootstrap values and draws are invalid")
    samples = values[draws].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95Low": float(np.quantile(samples, 0.025)),
        "ci95High": float(np.quantile(samples, 0.975)),
        "bootstrapStandardError": float(samples.std(ddof=1)),
        "probabilityPositive": float(np.mean(samples > 0.0)),
        "genes": int(values.size),
    }


def paired_bootstrap(
    world: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    rng: np.random.Generator,
    bootstraps: int,
) -> dict[str, object]:
    if not np.array_equal(world["gene_ids"], baseline["gene_ids"]):
        raise CandidateAuditError("paired bootstrap gene order differs")
    genes = len(world["gene_ids"])
    draws = rng.integers(0, genes, size=(bootstraps, genes))
    result: dict[str, object] = {
        "deltaNllBaselineMinusWorld": bootstrap_mean(
            baseline["nll"] - world["nll"], draws
        )
    }
    world_r = world["adjusted_pearson"]
    baseline_r = baseline["adjusted_pearson"]
    supported = np.isfinite(world_r) & np.isfinite(baseline_r)
    if supported.all():
        result["deltaAdjustedPearsonWorldMinusBaseline"] = bootstrap_mean(
            world_r - baseline_r, draws
        )
    else:
        result["deltaAdjustedPearsonWorldMinusBaseline"] = {
            "estimate": None,
            "reason": "baseline adjusted profile is constant and Pearson is undefined",
            "supportedGenes": int(supported.sum()),
            "genes": genes,
        }
    return result


def calibration_moments(
    prediction: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    scale: np.ndarray,
    num_cells: np.ndarray,
    action_ids: np.ndarray,
) -> dict[str, object]:
    """Summarize fixed-scale standardized residuals in predefined exposure bins."""

    residual = np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    scale = np.broadcast_to(np.asarray(scale, dtype=np.float64), residual.shape)
    observed = np.asarray(observed, dtype=np.bool_)
    num_cells = np.asarray(num_cells, dtype=np.float64)
    row_mse = np.sum(np.where(observed, residual**2, 0.0), axis=1) / observed.sum(axis=1)
    correlation = spearmanr(row_mse, 1.0 / num_cells).statistic
    bins: dict[str, object] = {}
    for name, lower, upper in COUNT_BINS:
        rows = (num_cells >= lower) & (num_cells < upper)
        if not rows.any():
            bins[name] = {"records": 0, "genes": 0, "observedValues": 0}
            continue
        mask = observed[rows]
        z = (residual[rows] / scale[rows])[mask]
        row_z2 = np.sum(np.where(mask, (residual[rows] / scale[rows]) ** 2, 0.0), axis=1) / mask.sum(axis=1)
        bins[name] = {
            "records": int(rows.sum()),
            "genes": int(np.unique(action_ids[rows]).size),
            "observedValues": int(mask.sum()),
            "cellCountMin": float(num_cells[rows].min()),
            "cellCountMedian": float(np.median(num_cells[rows])),
            "cellCountMax": float(num_cells[rows].max()),
            "zMean": float(z.mean()),
            "zSecondMoment": float(np.mean(z**2)),
            "zVariance": float(z.var()),
            "zFourthMoment": float(np.mean(z**4)),
            "meanRecordZSecondMoment": float(row_z2.mean()),
            "medianRecordZSecondMoment": float(np.median(row_z2)),
            "withinOneScale": float(np.mean(np.abs(z) <= 1.0)),
            "within1Point96Scale": float(np.mean(np.abs(z) <= 1.96)),
            "residualRms": float(np.sqrt(np.mean(residual[rows][mask] ** 2))),
            "meanScale": float(scale[rows][mask].mean()),
        }
    return {
        "bins": bins,
        "spearmanRecordMseVsInverseCellCount": float(correlation),
    }


def _load_run_modules(run_dir: Path):
    source = str((run_dir / "source").resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    return (
        importlib.import_module("inference").Predictor,
        importlib.import_module("transition_calibration"),
        importlib.import_module("exposure_uncertainty"),
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    data_path = Path(args.data)
    feature_path = Path(args.features)
    run_dir = Path(args.run)
    output_path = Path(args.output) / "report.json"
    if sha256(data_path) != DATA_SHA256:
        raise CandidateAuditError("development bundle SHA-256 drift")
    if sha256(feature_path) != FEATURE_SHA256:
        raise CandidateAuditError("static fusion feature SHA-256 drift")
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    if protocol["data_sha256"] != DATA_SHA256 or protocol["test_accessed"]:
        raise CandidateAuditError("candidate protocol does not match the permitted development run")
    if protocol["args"]["ridge_alpha"] != RIDGE_ALPHA:
        raise CandidateAuditError("candidate ridge alpha differs from frozen audit alpha")

    with np.load(data_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    if len(data["split_test"]) or str(data["target_value_space"]) != VALUE_SPACE:
        raise CandidateAuditError("only the author-normalized development bundle is allowed")
    train = data["split_train"]
    validation = data["split_validation"]
    if set(data["action_ids"][train]) & set(data["action_ids"][validation]):
        raise CandidateAuditError("training and validation intervention genes overlap")

    with np.load(feature_path, allow_pickle=False) as archive:
        keys = list(zip(archive["entity_taxon"].tolist(), archive["entity_id"].tolist()))
        values = archive["feature_values"].astype(np.float32)
    lookup = dict(zip(keys, values))
    x = np.stack([lookup[(9606, str(gene))] for gene in data["action_ids"]])

    Predictor, calibration, uncertainty = _load_run_modules(run_dir)
    predictor = Predictor(run_dir, device="cpu")
    with np.load(run_dir / "reference.npz", allow_pickle=False) as archive:
        saved = {name: archive[name] for name in archive.files}
    if not np.array_equal(saved["query_ids"], data["query_ids"]):
        raise CandidateAuditError("candidate and development query identities differ")

    y = data["targets"]
    observed = data["observed"]
    context = data["context_index"]
    context_ids = data["context_ids"].astype(str)
    action_ids = data["action_ids"].astype(str)
    action_keys = [(9606, gene) for gene in action_ids]
    means: list[object] = []
    ridges: list[object] = []
    oof_mean = np.empty_like(y[train], dtype=np.float64)
    oof_ridge = np.empty_like(y[train], dtype=np.float64)
    with threadpool_limits(limits=4):
        for context_index in range(len(context_ids)):
            rows = train[context[train] == context_index]
            positions = np.flatnonzero(context[train] == context_index)
            local_keys = [action_keys[row] for row in rows]
            mean, oof_mean[positions] = calibration.fit_grouped_oof_mean(
                y[rows], observed[rows], local_keys, scale_floor=SCALE_FLOOR, return_oof=True
            )
            ridge, oof_ridge[positions] = calibration.fit_grouped_oof_ridge(
                x[rows],
                y[rows],
                observed[rows],
                local_keys,
                RIDGE_ALPHA,
                scale_floor=SCALE_FLOOR,
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
    mean_exposure = uncertainty.fit_exposure_uncertainty(
        y[train] - oof_mean,
        observed[train],
        data["num_cells_filtered"][train],
        context[train],
        **control_args,
        scale_floor=SCALE_FLOOR,
    )
    ridge_exposure = uncertainty.fit_exposure_uncertainty(
        y[train] - oof_ridge,
        observed[train],
        data["num_cells_filtered"][train],
        context[train],
        **control_args,
        scale_floor=SCALE_FLOOR,
    )

    world_prediction = np.empty((len(validation), y.shape[1]), dtype=np.float32)
    world_scale = np.empty_like(world_prediction)
    for start in range(0, len(validation), args.batch_size):
        positions = np.arange(start, min(start + args.batch_size, len(validation)))
        rows = validation[positions]
        local_context = context[rows]
        measurement_scale = predictor.measurement_scales(
            data["num_cells_filtered"][rows], local_context, np.arange(y.shape[1])
        )
        result = predictor.predict(
            x[rows],
            saved["query_features"],
            saved["reference"][local_context],
            saved["reference_scale"][local_context],
            measurement_scale=measurement_scale,
            context_features=np.broadcast_to(
                saved["context_features"],
                (len(rows), *saved["context_features"].shape),
            ),
            context_values=saved["context_values"][local_context],
            context_mask=np.ones((len(rows), saved["context_values"].shape[1]), dtype=bool),
        )
        world_prediction[positions] = result["mean"]
        world_scale[positions] = result["marginal_scale"]

    rng = np.random.default_rng(args.seed)
    contexts: dict[str, object] = {}
    component_drift: dict[str, float] = {}
    with np.load(run_dir / "exposure-uncertainty.npz", allow_pickle=False) as archive:
        component_drift = {
            "meanBiologicalVarianceMaxAbs": float(
                np.max(np.abs(archive["mean_biological_variance"] - mean_exposure.biological_variance_))
            ),
            "meanSamplingVarianceMaxAbs": float(
                np.max(np.abs(archive["mean_sampling_variance"] - mean_exposure.sampling_variance_))
            ),
            "ridgeBiologicalVarianceMaxAbs": float(
                np.max(np.abs(archive["ridge_biological_variance"] - ridge_exposure.biological_variance_))
            ),
            "ridgeSamplingVarianceMaxAbs": float(
                np.max(np.abs(archive["ridge_sampling_variance"] - ridge_exposure.sampling_variance_))
            ),
        }

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
        predictions = {
            "world": world_prediction[positions],
            "mean": mean_prediction,
            "ridge": ridge_prediction,
        }
        scales = {"world": world_scale[positions], "mean": mean_scale, "ridge": ridge_scale}
        summaries = {
            name: gene_summaries(prediction, target, mask, genes, reference, scales[name])
            for name, prediction in predictions.items()
        }
        draws = rng.integers(0, len(summaries["world"]["gene_ids"]), size=(args.bootstraps, len(summaries["world"]["gene_ids"])))
        world_r = bootstrap_mean(summaries["world"]["adjusted_pearson"], draws)
        ridge_r = bootstrap_mean(summaries["ridge"]["adjusted_pearson"], draws)
        comparisons = {}
        for name in ("mean", "ridge"):
            # Use a fresh deterministic stream per comparator while preserving seed 731.
            comparisons[name] = paired_bootstrap(summaries["world"], summaries[name], rng, args.bootstraps)
        contexts[context_id] = {
            "records": len(rows),
            "interventionGenes": len(summaries["world"]["gene_ids"]),
            "pointEstimates": {
                name: {
                    "geneMacroNll": float(summary["nll"].mean()),
                    "geneMacroAdjustedPearson": (
                        float(np.nanmean(summary["adjusted_pearson"]))
                        if np.isfinite(summary["adjusted_pearson"]).any()
                        else None
                    ),
                }
                for name, summary in summaries.items()
            },
            "bootstrap": {
                "worldAdjustedPearson": world_r,
                "ridgeAdjustedPearson": ridge_r,
                "comparisons": comparisons,
            },
            "calibration": {
                name: calibration_moments(
                    predictions[name],
                    target,
                    mask,
                    scales[name],
                    data["num_cells_filtered"][rows],
                    genes,
                )
                for name in predictions
            },
        }

    ridge_intervals = [
        contexts[context_id]["bootstrap"]["comparisons"]["ridge"]
        for context_id in context_ids
    ]
    stable_nll_benefit = all(
        result["deltaNllBaselineMinusWorld"]["ci95Low"] > 0.0
        for result in ridge_intervals
    )
    passes_nll_magnitude = all(
        result["deltaNllBaselineMinusWorld"]["estimate"] >= 0.02
        for result in ridge_intervals
    )
    stable_adjusted_r_advantage = all(
        result["deltaAdjustedPearsonWorldMinusBaseline"]["ci95Low"] > 0.0
        for result in ridge_intervals
    )
    uniformly_overconservative = all(
        details["zSecondMoment"] < 0.8
        for context_id in context_ids
        for details in contexts[context_id]["calibration"]["world"]["bins"].values()
        if details["records"]
    )
    report = {
        "schema": "slp.normalized-human-candidate-development-audit/v1",
        "label": "development diagnostics",
        "hypothesis": (
            "The normalized response-query world model has a stable held-gene advantage "
            "over matched feature ridge in both contexts, and its inherited mean-OOF "
            "exposure scale is not materially overconservative."
        ),
        "fixedDecision": (
            "Treat benefit as stable only when the paired 95% gene-bootstrap NLL-delta "
            "lower bound is above zero in both contexts. Recommend fitting-only model-OOF "
            "scale calibration when world z second moments are below 0.8 across supported "
            "cell-count bins without an increasing low-count excess."
        ),
        "scope": "v2 development validation only; no model refit, validation recalibration, or test access",
        "seed": args.seed,
        "bootstraps": args.bootstraps,
        "bootstrapUnit": "intervention gene; every record for a sampled gene retained",
        "countBins": [
            {"id": name, "lowerInclusive": lower, "upperExclusive": upper if np.isfinite(upper) else None}
            for name, lower, upper in COUNT_BINS
        ],
        "ridge": {
            "alpha": RIDGE_ALPHA,
            "fit": "context-specific full training split",
            "scale": "three-fold gene-grouped training OOF residuals plus core-control exposure",
        },
        "worldScale": "candidate inference measurement_scales using stored mean-OOF exposure components",
        "inputs": {
            "development": {"path": str(data_path), "sha256": DATA_SHA256},
            "features": {"path": str(feature_path), "sha256": FEATURE_SHA256},
            "candidate": {
                "path": str(run_dir),
                "checkpointSha256": sha256(run_dir / "model.safetensors"),
                "reportSha256": sha256(run_dir / "report.json"),
            },
        },
        "exposureComponentReproduction": component_drift,
        "contexts": contexts,
        "decision": {
            "positiveNllBenefitStableInBothContexts": stable_nll_benefit,
            "passesPoint02NatMagnitudeInBothContexts": passes_nll_magnitude,
            "adjustedPearsonAdvantageOverRidgeStableInBothContexts": stable_adjusted_r_advantage,
            "worldScaleUniformlyOverconservativeByFixedRule": uniformly_overconservative,
            "recommendModelOofScaleCalibrationByFixedRule": uniformly_overconservative,
            "interpretation": (
                "The NLL benefit is positive under intervention-gene bootstrap, but its "
                "magnitude misses the advancement rule and the adjusted-Pearson advantage "
                "over ridge is uncertain. Inherited exposure scales are overconservative "
                "at high cell counts and underconservative at low counts, so these "
                "validation diagnostics do not support a simple model-specific scale "
                "recalibration. Any model-OOF scale experiment must remain fitting-only."
            ),
        },
        "elapsedSeconds": time.monotonic() - started,
        "testAccessed": False,
        "benchmarkAccessed": False,
    }
    _write_json(output_path, report)
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
    parser.add_argument(
        "--run",
        default="results/slp11-transition/human-normalized-fusion-response32-exposure-seed731-v1",
    )
    parser.add_argument(
        "--output", default="results/slp11-transition/human-normalized-candidate-audit-v1"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    if args.seed != SEED or args.bootstraps != BOOTSTRAPS:
        raise CandidateAuditError("audit requires frozen seed 731 and 1000 bootstraps")
    report = run(args)
    print(json.dumps({"output": str(Path(args.output) / "report.json"), "contexts": report["contexts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
