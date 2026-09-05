#!/usr/bin/env python3
"""Run the fitting-only Nyström RBF feature-ridge development comparison."""

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

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from kernel_baseline import fit_nystrom_rbf
from transition_baselines import compare_paired_nll, evaluate
from transition_calibration import (
    fit_grouped_oof_mean,
    fit_grouped_oof_ridge,
)
from transition_data import load_corpus, split_by_gene


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


def load_aligned_features(path: Path, required_keys) -> tuple[np.ndarray, tuple[int, int]]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"feature_values", "entity_taxon", "entity_id"}:
            raise ValueError("feature NPZ member contract mismatch")
        values = archive["feature_values"]
        taxa = archive["entity_taxon"]
        identifiers = archive["entity_id"]
    keys = tuple((int(taxon), str(entity)) for taxon, entity in zip(taxa, identifiers, strict=True))
    if values.ndim != 2 or values.shape[0] != len(keys) or not np.isfinite(values).all():
        raise ValueError("feature NPZ values are invalid")
    if len(keys) != len(set(keys)):
        raise ValueError("feature NPZ composite entity keys are duplicated")
    index = {key: row for row, key in enumerate(keys)}
    missing = sorted(set(required_keys) - set(index))
    if missing:
        raise ValueError(f"feature NPZ is missing required action entity keys: {missing[:8]}")
    selected = values[np.asarray([index[key] for key in required_keys], dtype=np.int64)]
    return selected.astype(np.float64, copy=False), values.shape


def gene_metrics(prediction, target, mask, keys, reference, scale):
    groups: dict[tuple[int, str], list[int]] = {}
    for row, key in enumerate(keys):
        groups.setdefault(key, []).append(row)
    scale_matrix = np.broadcast_to(scale, prediction.shape)
    reports = []
    for rows in groups.values():
        reports.append(
            evaluate(
                prediction[rows],
                target[rows],
                mask[rows],
                reference,
                scale_matrix[rows],
            )
        )
    result = evaluate(prediction, target, mask, reference, scale_matrix)
    for metric in (
        "nll",
        "mse",
        "profile_pearson_mean",
        "profile_centroid_adjusted_pearson_mean",
    ):
        values = [report[metric] for report in reports if np.isfinite(report[metric])]
        result["gene_macro_" + metric] = float(np.mean(values)) if values else math.nan
    result["intervention_genes"] = len(groups)
    return result


def run(args) -> dict[str, object]:
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    corpus_path = args.corpus.resolve(strict=True)
    feature_path = args.features.resolve(strict=True)
    corpus = load_corpus(corpus_path)
    split = split_by_gene(corpus["action_keys"], seed=args.seed)
    training, validation = split["train"], split["validation"]
    if len(training) == 0 or len(validation) == 0 or len(split["test"]) == 0:
        raise ValueError("outer grouped development split must have non-empty partitions")
    action_entity_keys = tuple(
        corpus["entity_keys"][int(index)] for index in corpus["action_index"]
    )
    action_features, feature_artifact_shape = load_aligned_features(
        feature_path, action_entity_keys
    )
    targets = corpus["targets"]
    observed = corpus["observed"]
    action_keys = corpus["action_keys"]

    grid = [
        {"landmarks": landmarks, "bandwidthFactor": factor, "alpha": args.alpha}
        for landmarks in args.landmarks
        for factor in args.bandwidth_factors
    ]
    source_files = [
        Path(__file__),
        MODULE / "kernel_baseline.py",
        MODULE / "transition_baselines.py",
        MODULE / "transition_calibration.py",
        MODULE / "transition_data.py",
    ]
    protocol = {
        "schema": "slp.kernel-feature-ridge-development/v1",
        "scope": "fitting-corpus-only exploratory development comparison",
        "hypothesis": (
            "training-only RBF similarity over static features improves unseen-"
            "development-gene molecular prediction over matched raw-feature linear ridge"
        ),
        "selectionRule": "minimum development-validation gene-macro Gaussian NLL",
        "candidateGridFrozenBeforeFit": grid,
        "baselineIdentity": "Nystrom RBF features plus feature-linear multioutput ridge; not pure kernel ridge regression",
        "split": {
            "method": "split_by_gene",
            "seed": args.seed,
            "counts": {name: len(rows) for name, rows in split.items()},
            "internalTestScored": False,
            "originalProtectedHoldoutsAccessed": False,
        },
        "calibration": {
            "method": "gene-grouped OOF over outer training records only",
            "folds": args.folds,
            "scaleFloor": args.scale_floor,
        },
        "features": {
            "name": args.feature_name,
            "path": str(feature_path),
            "sha256": sha256_file(feature_path),
            "shape": list(feature_artifact_shape),
            "join": "exact composite key for every action record",
        },
        "corpus": {
            "path": str(corpus_path),
            "sha256": sha256_file(corpus_path),
            "productionArchiveMatch": corpus["metadata"]["production_archive_match"],
        },
        "sourceHashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files},
        "blasThreads": 4,
        "outcomesUsedForFitting": "outer training only",
        "outcomesUsedForSelection": "development validation only",
    }
    write_json(output / "protocol.json", protocol)
    print(json.dumps({"event": "protocol-frozen", "grid": grid}), flush=True)

    train_keys = [action_keys[index] for index in training]
    validation_keys = [action_keys[index] for index in validation]
    mean = fit_grouped_oof_mean(
        targets[training],
        observed[training],
        train_keys,
        folds=args.folds,
        seed=args.seed,
        scale_floor=args.scale_floor,
    )
    linear = fit_grouped_oof_ridge(
        action_features[training],
        targets[training],
        observed[training],
        train_keys,
        args.alpha,
        folds=args.folds,
        seed=args.seed,
        scale_floor=args.scale_floor,
    )
    mean_prediction = mean.predict(action_features[validation])
    linear_prediction = linear.predict(action_features[validation])
    controls = {
        "mean": gene_metrics(
            mean_prediction,
            targets[validation],
            observed[validation],
            validation_keys,
            mean.intercept_,
            mean.residual_scale_.values,
        ),
        "raw-feature-linear-ridge": gene_metrics(
            linear_prediction,
            targets[validation],
            observed[validation],
            validation_keys,
            mean.intercept_,
            linear.residual_scale_.values,
        ),
    }
    candidates: list[dict[str, object]] = []
    for configuration in grid:
        transform = fit_nystrom_rbf(
            action_features[training],
            train_keys,
            n_landmarks=configuration["landmarks"],
            bandwidth_factor=configuration["bandwidthFactor"],
            seed=args.seed,
        )
        training_kernel_features = transform.transform(action_features[training])
        validation_kernel_features = transform.transform(action_features[validation])
        model = fit_grouped_oof_ridge(
            training_kernel_features,
            targets[training],
            observed[training],
            train_keys,
            args.alpha,
            folds=args.folds,
            seed=args.seed,
            scale_floor=args.scale_floor,
        )
        prediction = model.predict(validation_kernel_features)
        metrics = gene_metrics(
            prediction,
            targets[validation],
            observed[validation],
            validation_keys,
            mean.intercept_,
            model.residual_scale_.values,
        )
        candidate = {
            "configuration": configuration,
            "effectiveKernelRank": int(transform.eigenvalues_.size),
            "medianTrainingGenePairDistance": transform.median_pair_distance_,
            "bandwidth": transform.bandwidth_,
            "landmarkKeysSha256": hashlib.sha256(
                "".join(f"{key[0]}|{key[1]}\n" for key in transform.landmark_keys_).encode()
            ).hexdigest(),
            "scaleProvenance": model.residual_scale_.provenance,
            "metrics": metrics,
            "pairedVsMean": compare_paired_nll(
                prediction,
                mean_prediction,
                targets[validation],
                observed[validation],
                model.residual_scale_.values,
                mean.residual_scale_.values,
            ),
            "pairedVsRawLinear": compare_paired_nll(
                prediction,
                linear_prediction,
                targets[validation],
                observed[validation],
                model.residual_scale_.values,
                linear.residual_scale_.values,
            ),
        }
        candidates.append(candidate)
        print(
            json.dumps(
                {
                    "event": "candidate",
                    "configuration": configuration,
                    "geneMacroNll": metrics["gene_macro_nll"],
                    "geneMacroAdjustedPearson": metrics[
                        "gene_macro_profile_centroid_adjusted_pearson_mean"
                    ],
                }
            ),
            flush=True,
        )
    champion_index = min(
        range(len(candidates)),
        key=lambda index: candidates[index]["metrics"]["gene_macro_nll"],
    )
    report = {
        "protocol": protocol,
        "controls": controls,
        "candidates": candidates,
        "championIndex": champion_index,
        "champion": candidates[champion_index],
        "internalTestScored": False,
        "originalProtectedHoldoutsAccessed": False,
        "elapsedSeconds": time.monotonic() - started,
    }
    write_json(output / "report.json", report)
    print(
        json.dumps(
            {
                "event": "finished",
                "output": str(output),
                "champion": candidates[champion_index]["configuration"],
            }
        ),
        flush=True,
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--corpus", type=Path, required=True)
    result.add_argument("--features", type=Path, required=True)
    result.add_argument("--feature-name", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--landmarks", type=int, nargs="+", default=[128])
    result.add_argument("--bandwidth-factors", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    result.add_argument("--alpha", type=float, default=1000.0)
    result.add_argument("--folds", type=int, default=3)
    result.add_argument("--seed", type=int, default=731)
    result.add_argument("--scale-floor", type=float, default=0.05)
    return result


def main() -> int:
    try:
        run(parser().parse_args())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
