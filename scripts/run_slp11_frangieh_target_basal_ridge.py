"""Frozen paired-head ridge comparison for Frangieh development profiles."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "modules/slp-1-1-world-transition-v1/frangieh_basal_ridge.py"
SPEC = importlib.util.spec_from_file_location("frangieh_basal_ridge", MODULE_PATH)
RIDGE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RIDGE)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _choose_alpha(x: np.ndarray, y: np.ndarray, action_ids: np.ndarray) -> tuple[float, dict]:
    folds = np.asarray([RIDGE.cv_fold(action) for action in action_ids], dtype=np.int64)
    if set(folds) != {0, 1, 2}:
        raise ValueError("all three fitting-gene folds are required")
    report = {}
    scores = []
    for alpha in ALPHAS:
        fold_reports = []
        gene_scores = []
        for fold in range(3):
            fit = folds != fold
            held = folds == fold
            prediction, stats = RIDGE.dual_ridge_predict(x[fit], y[fit], x[held], alpha)
            per_gene = np.mean(((prediction - y[held]) / stats["target_scale"]) ** 2, axis=1)
            gene_scores.extend(per_gene.tolist())
            fold_reports.append(
                {
                    "fold": fold,
                    "fitting_genes": int(np.sum(fit)),
                    "held_genes": int(np.sum(held)),
                    "gene_macro_scaled_mse": float(np.mean(per_gene)),
                }
            )
        score = float(np.mean(gene_scores))
        report[str(alpha)] = {"gene_macro_scaled_mse": score, "folds": fold_reports}
        scores.append((score, alpha))
    return min(scores)[1], report


def _feature_matrix(
    action_ids: np.ndarray,
    feature_lookup: dict[str, int],
    feature_values: np.ndarray,
    basal_lookup: dict[str, float],
    include_basal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    static = np.zeros((len(action_ids), feature_values.shape[1]), dtype=np.float32)
    available = np.zeros((len(action_ids), 1), dtype=np.float32)
    basal = np.empty((len(action_ids), 1), dtype=np.float32)
    for index, action in enumerate(action_ids):
        row = feature_lookup.get(action)
        if row is not None:
            static[index] = feature_values[row]
            available[index] = 1.0
        if action not in basal_lookup:
            raise ValueError(f"action is absent from the stable RNA query axis: {action}")
        basal[index] = basal_lookup[action]
    values = np.concatenate([static, available], axis=1)
    if include_basal:
        values = np.concatenate([values, basal], axis=1)
    return values, available[:, 0].astype(bool)


def run(data_path: Path, feature_path: Path, output_dir: Path) -> dict:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(data_path, allow_pickle=False) as data, np.load(feature_path, allow_pickle=False) as static:
        action, context, rna, guide_counts = RIDGE.collapse_gene_profiles(
            data["action_ids"], data["context_ids"], data["rna_targets"]
        )
        p_action, p_context, protein, p_counts = RIDGE.collapse_gene_profiles(
            data["action_ids"], data["context_ids"], data["protein_targets"]
        )
        if not (
            np.array_equal(action, p_action)
            and np.array_equal(context, p_context)
            and np.array_equal(guide_counts, p_counts)
        ):
            raise ValueError("RNA and ADT collapsed axes differ")
        query_ids = data["rna_query_ids"].astype(str)
        query_lookup = {value: index for index, value in enumerate(query_ids)}
        control_lookup = {
            str(ctx): np.asarray(values, dtype=np.float32)
            for ctx, values in zip(
                data["control_context_ids"].astype(str), data["control_rna_targets"], strict=True
            )
        }
        feature_ids = static["entity_id"].astype(str)
        feature_values = np.asarray(static["feature_values"], dtype=np.float32)
        if feature_values.shape[1] != 1156 or np.any(static["entity_taxon"] != 9606):
            raise ValueError("unexpected physical static feature contract")
        feature_lookup = {value: index for index, value in enumerate(feature_ids)}

        report = {
            "schema": "slp.frangieh-target-basal-ridge/v1",
            "hypothesis": "Matching-condition control RNA abundance of the target gene improves unseen-gene point prediction beyond static features alone.",
            "estimand": "One profile per intervention gene and condition, formed by equal weighting of its guide-resolved pseudobulk records.",
            "alpha_grid": list(ALPHAS),
            "heads_are_separate": True,
            "contexts": {},
            "fixed_rule": {
                "mse": "Arm B raw MSE is at least 1 percent lower than Arm A",
                "correlation": "Arm B independently-centered profile Pearson does not regress versus Arm A",
                "scope": "both heads in each of all three contexts; no averaging across failures",
            },
            "feature_coverage": {},
            "artifacts": {},
        }
        predictions = {}
        decisions = []
        for ctx in sorted(set(context)):
            rows = context == ctx
            ids = action[rows]
            # Use the repository split contract directly; CV has a separate namespace.
            split = np.asarray([RIDGE.development_split(x) for x in ids])
            fit, held = split == "train", split == "validation"
            if not np.any(fit) or not np.any(held):
                raise ValueError(f"missing outer split in {ctx}")
            basal_lookup = {gene: float(control_lookup[ctx][query_lookup[gene]]) for gene in ids}
            x_a, available = _feature_matrix(ids, feature_lookup, feature_values, basal_lookup, False)
            x_b, available_b = _feature_matrix(ids, feature_lookup, feature_values, basal_lookup, True)
            if not np.array_equal(available, available_b):
                raise AssertionError("arm coverage differs")
            report["feature_coverage"][ctx] = {
                "all_genes": len(ids),
                "available": int(np.sum(available)),
                "unavailable_zero_filled": int(np.sum(~available)),
                "train_available": int(np.sum(available[fit])),
                "validation_available": int(np.sum(available[held])),
            }
            context_report = {"heads": {}, "train_genes": int(np.sum(fit)), "validation_genes": int(np.sum(held))}
            for head, y in (("rna", rna[rows]), ("adt", protein[rows])):
                head_report = {"arms": {}}
                arm_predictions = {}
                outer_stats = None
                for arm, x in (("static", x_a), ("static_plus_target_basal", x_b)):
                    alpha, cv = _choose_alpha(x[fit], y[fit], ids[fit])
                    prediction, stats = RIDGE.dual_ridge_predict(x[fit], y[fit], x[held], alpha)
                    arm_predictions[arm] = prediction
                    head_report["arms"][arm] = {
                        "selected_alpha": alpha,
                        "cv": cv,
                        "validation": RIDGE.metrics(prediction, y[held], stats["target_scale"]),
                    }
                    outer_stats = stats
                mean_prediction = np.broadcast_to(outer_stats["target_mean"], y[held].shape).astype(np.float32)
                head_report["context_mean_baseline"] = RIDGE.metrics(
                    mean_prediction, y[held], outer_stats["target_scale"]
                )
                a_metrics = head_report["arms"]["static"]["validation"]
                b_metrics = head_report["arms"]["static_plus_target_basal"]["validation"]
                improvement = (a_metrics["raw_mse"] - b_metrics["raw_mse"]) / a_metrics["raw_mse"]
                mse_pass = improvement >= 0.01
                r_pass = (
                    b_metrics["query_centroid_adjusted_profile_pearson"]
                    >= a_metrics["query_centroid_adjusted_profile_pearson"] - 1e-12
                )
                head_report["arm_b_vs_a"] = {
                    "fractional_raw_mse_improvement": float(improvement),
                    "mse_pass": bool(mse_pass),
                    "centered_profile_pearson_no_regression_pass": bool(r_pass),
                    "pass": bool(mse_pass and r_pass),
                }
                decisions.append(bool(mse_pass and r_pass))
                context_report["heads"][head] = head_report
                key = ctx.replace("γ", "gamma").replace("-", "_")
                predictions[f"{key}_{head}_action_ids"] = ids[held]
                predictions[f"{key}_{head}_truth"] = y[held]
                predictions[f"{key}_{head}_mean"] = mean_prediction
                predictions[f"{key}_{head}_static"] = arm_predictions["static"]
                predictions[f"{key}_{head}_static_plus_target_basal"] = arm_predictions[
                    "static_plus_target_basal"
                ]
            report["contexts"][ctx] = context_report
        report["decision"] = {
            "advance": bool(all(decisions)),
            "passed_checks": int(sum(decisions)),
            "total_checks": len(decisions),
        }
        report["runtime_seconds"] = float(time.perf_counter() - started)
    prediction_path = output_dir / "predictions.npz"
    np.savez_compressed(prediction_path, **predictions)
    report["artifacts"]["predictions.npz"] = {
        "sha256": digest(prediction_path),
        "bytes": prediction_path.stat().st_size,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--features-sha256", required=True)
    args = parser.parse_args()
    if digest(args.data) != args.data_sha256 or digest(args.features) != args.features_sha256:
        raise ValueError("input hash mismatch")
    print(json.dumps(run(args.data, args.features, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
