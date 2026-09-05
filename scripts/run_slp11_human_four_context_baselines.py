"""Run frozen mean/physical-ridge point baselines on four-context development."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
UTILITY_PATH = ROOT / "modules/slp-1-1-world-transition-v1/four_context_baselines.py"
FROZEN_BASELINE_PATH = (
    ROOT
    / "results/slp11-transition/physical-features-ridge-screen-v1/source/transition_baselines.py"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


METRICS = _load("four_context_baselines", UTILITY_PATH)
FROZEN = _load("frozen_transition_baselines", FROZEN_BASELINE_PATH)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _features(action_ids: np.ndarray, archive: object) -> np.ndarray:
    entity_ids = archive["entity_id"].astype(str)
    entity_taxon = archive["entity_taxon"]
    values = archive["feature_values"]
    if (
        values.shape[1] != 1156
        or np.any(entity_taxon != 9606)
        or len(set(entity_ids)) != len(entity_ids)
    ):
        raise ValueError("physical feature contract drift")
    lookup = {gene: index for index, gene in enumerate(entity_ids)}
    missing = sorted(set(action_ids.tolist()) - set(lookup))
    if missing:
        raise ValueError(f"physical feature rows missing for actions: {missing[:3]}")
    return values[np.asarray([lookup[gene] for gene in action_ids], dtype=np.int64)]


def run(
    data_path: Path,
    feature_path: Path,
    original_predictions_path: Path,
    output_dir: Path,
) -> dict:
    if (output_dir / "predictions.npz").exists() or (
        output_dir / "report.json"
    ).exists():
        raise FileExistsError("immutable baseline output already exists")
    with (
        np.load(data_path, allow_pickle=False) as data,
        np.load(feature_path, allow_pickle=False) as feature_archive,
        np.load(original_predictions_path, allow_pickle=False) as original,
    ):
        action_ids = data["action_ids"].astype(str)
        targets = data["targets"]
        observed = data["observed"]
        context_index = data["context_index"]
        contexts = data["context_ids"].astype(str)
        split_train = data["split_train"]
        split_validation = data["split_validation"]
        if len(data["split_test"]) or len(contexts) != 4:
            raise ValueError("four-context development split contract drift")
        features = _features(action_ids, feature_archive)
        target_spaces = data["target_value_space_by_context"].astype(str)
        report = {
            "schema": "slp.human-four-context-point-baselines/v1",
            "ridge_alpha": 10000.0,
            "ridge_fit": "context-local original records; exact frozen source-three implementation",
            "evaluation_estimand": "equal construct/population record mean per intervention gene",
            "query_scale": "SD across fitting gene profiles per query, ddof0, floor0.05",
            "contexts": {},
            "feature_coverage": {
                "records": len(action_ids),
                "unique_action_genes": len(set(action_ids)),
                "missing_records": 0,
                "missing_unique_action_genes": 0,
            },
        }
        saved = {}
        decisions = []
        for context, context_id in enumerate(contexts):
            fitting = split_train[context_index[split_train] == context]
            validation = split_validation[context_index[split_validation] == context]
            ridge = FROZEN.fit_ridge(
                features[fitting], targets[fitting], observed[fitting], 10000.0
            )
            mean = FROZEN.fit_mean(targets[fitting], observed[fitting])
            ridge_rows = ridge.predict(features[validation])
            mean_rows = mean.predict(features[validation])

            original_equality = None
            if context < 3:
                expected = original[f"context{context}_physical"]
                cast = ridge_rows.astype(np.float32)
                if not np.array_equal(cast, expected):
                    raise ValueError(
                        f"context {context} ridge predictions differ from frozen source-three"
                    )
                original_equality = {
                    "exact_float32": True,
                    "rows": len(expected),
                    "maximum_absolute_error": 0.0,
                }

            fit_genes, fit_profiles, fit_mask, fit_record_counts = (
                METRICS.collapse_equal_records(
                    action_ids[fitting], targets[fitting], observed[fitting]
                )
            )
            genes, truth, mask, validation_record_counts = (
                METRICS.collapse_equal_records(
                    action_ids[validation], targets[validation], observed[validation]
                )
            )
            ridge_genes, ridge_profiles, ridge_mask, _ = METRICS.collapse_equal_records(
                action_ids[validation], ridge_rows, observed[validation]
            )
            mean_genes, mean_profiles, mean_mask, _ = METRICS.collapse_equal_records(
                action_ids[validation], mean_rows, observed[validation]
            )
            if not (
                np.array_equal(genes, ridge_genes)
                and np.array_equal(genes, mean_genes)
                and np.array_equal(mask, ridge_mask)
                and np.array_equal(mask, mean_mask)
            ):
                raise ValueError("collapsed forecast axes differ")
            scale = METRICS.fitting_query_scale(fit_profiles, fit_mask, floor=0.05)
            mean_metrics = METRICS.point_metrics(
                mean_profiles, truth, mask, scale, mean.intercept_
            )
            ridge_metrics = METRICS.point_metrics(
                ridge_profiles, truth, mask, scale, mean.intercept_
            )
            improvement = (
                mean_metrics["gene_profile_raw_mse"]
                - ridge_metrics["gene_profile_raw_mse"]
            ) / mean_metrics["gene_profile_raw_mse"]
            ridge_r = ridge_metrics["independently_query_centered_profile_pearson"]
            passed = bool(
                improvement >= 0.01 and ridge_r is not None and ridge_r >= 0.1
            )
            decisions.append(passed)
            report["contexts"][context_id] = {
                "target_value_space": target_spaces[context],
                "fitting_records": len(fitting),
                "fitting_genes": len(fit_genes),
                "validation_records": len(validation),
                "validation_genes": len(genes),
                "fitting_constructs_per_gene": {
                    "minimum": int(fit_record_counts.min()),
                    "median": float(np.median(fit_record_counts)),
                    "maximum": int(fit_record_counts.max()),
                },
                "validation_constructs_per_gene": {
                    "minimum": int(validation_record_counts.min()),
                    "median": float(np.median(validation_record_counts)),
                    "maximum": int(validation_record_counts.max()),
                },
                "mean": mean_metrics,
                "ridge": ridge_metrics,
                "ridge_fractional_raw_mse_improvement": float(improvement),
                "rule_pass": passed,
                "original_source3_physical_prediction_equality": original_equality,
            }
            saved[f"context{context}_action_ids"] = genes
            saved[f"context{context}_truth"] = truth
            saved[f"context{context}_observed"] = mask
            saved[f"context{context}_mean"] = mean_profiles
            saved[f"context{context}_ridge"] = ridge_profiles
            saved[f"context{context}_fitting_query_scale"] = scale.astype(np.float32)
            saved[f"context{context}_fitting_target_centroid"] = mean.intercept_.astype(
                np.float32
            )
        report["decision"] = {
            "advance": bool(all(decisions)),
            "passed_contexts": int(sum(decisions)),
            "total_contexts": len(decisions),
            "rule": "ridge >=1% raw gene-profile MSE improvement over fitting mean and independent query-centered profile r >=0.10 in every context; undefined r fails",
        }
        report["uncertainty"] = {
            "likelihood_scored": False,
            "hepg2_independent_control_target_pseudobulks_available": False,
            "reason": "point-only metrics avoid inventing a compatible HepG2 Gaussian sampling scale",
        }
        report["claim_limits"] = [
            "Target units differ by context and are never pooled into one numeric score.",
            "HepG2 is retired adaptive development, not unseen-context confirmation.",
            "Jurkat outcomes are excluded and remain untouched.",
            "Ridge fitting retains original row weighting for exact source-three reproduction; evaluation alone collapses constructs equally per gene.",
        ]
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.npz"
    np.savez_compressed(predictions_path, **saved)
    report["predictions"] = {
        "path": str(predictions_path),
        "sha256": sha256(predictions_path),
        "bytes": predictions_path.stat().st_size,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--original-predictions", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    for name, path in (
        ("data", args.data),
        ("features", args.features),
        ("original_predictions", args.original_predictions),
    ):
        if sha256(path) != protocol["inputs"][name]["sha256"]:
            raise ValueError(f"{name} SHA-256 mismatch")
    with threadpool_limits(limits=2):
        result = run(
            args.data, args.features, args.original_predictions, args.output_dir
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
