"""Compare expanded base and species-wide physical features on Frangieh development."""

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
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _predict_candidate(
    label: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
) -> tuple[np.ndarray, dict]:
    if label == "mean_limit":
        return RIDGE.mean_limit_predict(y_train, len(x_eval))
    return RIDGE.dual_ridge_predict(x_train, y_train, x_eval, float(label))


def choose_candidate(x: np.ndarray, y: np.ndarray, action_ids: np.ndarray) -> tuple[str, dict]:
    folds = np.asarray([RIDGE.cv_fold(action) for action in action_ids])
    labels = [str(alpha) for alpha in ALPHAS] + ["mean_limit"]
    report = {}
    ranking = []
    for order, label in enumerate(labels):
        fold_reports = []
        all_gene_mse = []
        for fold in range(3):
            fit, held = folds != fold, folds == fold
            prediction, stats = _predict_candidate(label, x[fit], y[fit], x[held])
            per_gene = np.mean(((prediction - y[held]) / stats["target_scale"]) ** 2, axis=1)
            all_gene_mse.extend(per_gene.tolist())
            fold_reports.append(
                {
                    "fold": fold,
                    "fitting_genes": int(np.sum(fit)),
                    "held_genes": int(np.sum(held)),
                    "gene_macro_scaled_mse": float(np.mean(per_gene)),
                }
            )
        score = float(np.mean(all_gene_mse))
        report[label] = {"gene_macro_scaled_mse": score, "folds": fold_reports}
        ranking.append((score, order, label))
    return min(ranking)[2], report


def _features_for_actions(
    action_ids: np.ndarray, feature_ids: np.ndarray, values: np.ndarray, columns: slice
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {value: index for index, value in enumerate(feature_ids)}
    width = len(range(*columns.indices(values.shape[1])))
    output = np.zeros((len(action_ids), width), dtype=np.float32)
    available = np.zeros(len(action_ids), dtype=np.float32)
    for index, action in enumerate(action_ids):
        source = lookup.get(action)
        if source is not None:
            output[index] = values[source, columns]
            available[index] = 1.0
    return np.concatenate([output, available[:, None]], axis=1), available.astype(bool)


def _correlation_no_regression(base: float, physical: float) -> bool:
    if np.isfinite(base):
        return bool(np.isfinite(physical) and physical >= base - 1e-12)
    return bool(not np.isfinite(physical) or np.isfinite(physical))


def run(data_path: Path, feature_path: Path, output_dir: Path) -> dict:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(data_path, allow_pickle=False) as data, np.load(feature_path, allow_pickle=False) as features:
        action, context, rna, guide_counts = RIDGE.collapse_gene_profiles(
            data["action_ids"], data["context_ids"], data["rna_targets"]
        )
        p_action, p_context, adt, p_counts = RIDGE.collapse_gene_profiles(
            data["action_ids"], data["context_ids"], data["protein_targets"]
        )
        if not (
            np.array_equal(action, p_action)
            and np.array_equal(context, p_context)
            and np.array_equal(guide_counts, p_counts)
        ):
            raise ValueError("paired heads have different collapsed axes")
        feature_ids = features["entity_id"].astype(str)
        feature_values = np.asarray(features["feature_values"], dtype=np.float32)
        if feature_values.shape[1] != 1156 or np.any(features["entity_taxon"] != 9606):
            raise ValueError("unexpected feature contract")
        if len(set(feature_ids)) != len(feature_ids) or not np.isfinite(feature_values).all():
            raise ValueError("invalid feature identities or values")

        report = {
            "schema": "slp.frangieh-specieswide-physical-ridge/v1",
            "hypothesis": "Species-wide fixed-universe physical-neighbor features reduce unseen-gene MSE by at least one percent without adjusted-profile-correlation regression relative to the matched expanded base features.",
            "estimand": "Equal mean of guide-resolved pseudobulks for one target-gene profile per condition.",
            "candidate_grid": [str(alpha) for alpha in ALPHAS] + ["mean_limit"],
            "contexts": {},
            "feature_coverage": {},
            "fixed_rule": "Physical1156 must improve raw MSE by at least 1 percent and not regress query-centroid-adjusted profile Pearson versus base577 in every context and both separate heads.",
            "claim_limits": [
                "Same-study development validation, not independent confirmation.",
                "RNA and ADT are separate multi-output ridge heads, not a joint multimodal model.",
                "Physical associations are static evidence and do not identify causal direction.",
                "Candidate selection uses fitting-gene CV MSE only; outer validation does not select alpha.",
            ],
        }
        saved = {}
        decisions = []
        for ctx in sorted(set(context)):
            rows = context == ctx
            ids = action[rows]
            split = np.asarray([RIDGE.development_split(value) for value in ids])
            fit, held = split == "train", split == "validation"
            x_base, available_base = _features_for_actions(
                ids, feature_ids, feature_values, slice(0, 577)
            )
            x_physical, available_physical = _features_for_actions(
                ids, feature_ids, feature_values, slice(0, 1156)
            )
            if not np.array_equal(available_base, available_physical):
                raise ValueError("feature-arm availability differs")
            report["feature_coverage"][ctx] = {
                "genes": len(ids),
                "available": int(np.sum(available_base)),
                "zero_filled": int(np.sum(~available_base)),
                "train_available": int(np.sum(available_base[fit])),
                "validation_available": int(np.sum(available_base[held])),
            }
            context_report = {"train_genes": int(np.sum(fit)), "validation_genes": int(np.sum(held)), "heads": {}}
            for head, targets in (("rna", rna[rows]), ("adt", adt[rows])):
                head_report = {"arms": {}}
                forecasts = {}
                for arm, design in (("base577", x_base), ("physical1156", x_physical)):
                    selected, cv = choose_candidate(design[fit], targets[fit], ids[fit])
                    prediction, stats = _predict_candidate(selected, design[fit], targets[fit], design[held])
                    forecasts[arm] = prediction
                    head_report["arms"][arm] = {
                        "selected_candidate": selected,
                        "cv": cv,
                        "validation": RIDGE.metrics(prediction, targets[held], stats["target_scale"]),
                    }
                base_metrics = head_report["arms"]["base577"]["validation"]
                physical_metrics = head_report["arms"]["physical1156"]["validation"]
                improvement = (base_metrics["raw_mse"] - physical_metrics["raw_mse"]) / base_metrics[
                    "raw_mse"
                ]
                r_pass = _correlation_no_regression(
                    base_metrics["query_centroid_adjusted_profile_pearson"],
                    physical_metrics["query_centroid_adjusted_profile_pearson"],
                )
                passed = bool(improvement >= 0.01 and r_pass)
                decisions.append(passed)
                head_report["physical_vs_base"] = {
                    "fractional_raw_mse_improvement": float(improvement),
                    "mse_pass": bool(improvement >= 0.01),
                    "adjusted_profile_pearson_no_regression_pass": r_pass,
                    "pass": passed,
                }
                context_report["heads"][head] = head_report
                key = ctx.replace("γ", "gamma").replace("-", "_")
                saved[f"{key}_{head}_action_ids"] = ids[held]
                saved[f"{key}_{head}_truth"] = targets[held]
                saved[f"{key}_{head}_base577"] = forecasts["base577"]
                saved[f"{key}_{head}_physical1156"] = forecasts["physical1156"]
            report["contexts"][ctx] = context_report
        report["decision"] = {
            "advance": bool(all(decisions)),
            "passed_checks": int(sum(decisions)),
            "total_checks": len(decisions),
        }
        report["runtime_seconds"] = float(time.perf_counter() - started)
    predictions_path = output_dir / "predictions.npz"
    np.savez_compressed(predictions_path, **saved)
    report["predictions"] = {"sha256": digest(predictions_path), "bytes": predictions_path.stat().st_size}
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
