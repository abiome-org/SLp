"""Correct the query-centroid-adjusted profile metric without refitting."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "modules/slp-1-1-world-transition-v1/frangieh_basal_ridge.py"
SPEC = importlib.util.spec_from_file_location("frangieh_basal_ridge", MODULE_PATH)
METRICS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(METRICS)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def rescore(parent_report_path: Path, predictions_path: Path, output_path: Path) -> dict:
    parent = json.loads(parent_report_path.read_text(encoding="utf-8"))
    report = {
        "schema": "slp.frangieh-target-basal-ridge-corrected-scoring/v1",
        "parent": {
            "report_sha256": digest(parent_report_path),
            "predictions_sha256": digest(predictions_path),
        },
        "correction": "Subtract each forecast's and truth's per-query centroid across validation genes, then center each residual gene profile across queries before Pearson correlation.",
        "refit": False,
        "contexts": {},
        "fixed_rule": parent["fixed_rule"],
    }
    decisions = []
    with np.load(predictions_path, allow_pickle=False) as predictions:
        for context, context_parent in parent["contexts"].items():
            key = context.replace("γ", "gamma").replace("-", "_")
            context_report = {}
            for head, head_parent in context_parent["heads"].items():
                truth = predictions[f"{key}_{head}_truth"]
                methods = {}
                for method in ("mean", "static", "static_plus_target_basal"):
                    forecast = predictions[f"{key}_{head}_{method}"]
                    score, per_gene = METRICS.query_centroid_adjusted_profile_pearson(forecast, truth)
                    methods[method] = {
                        "query_centroid_adjusted_profile_pearson": score,
                        "undefined_genes": int(np.sum(~np.isfinite(per_gene))),
                        "genes": len(per_gene),
                    }
                a = methods["static"]["query_centroid_adjusted_profile_pearson"]
                b = methods["static_plus_target_basal"]["query_centroid_adjusted_profile_pearson"]
                r_pass = bool(np.isfinite(a) and np.isfinite(b) and b >= a - 1e-12)
                mse_pass = bool(head_parent["arm_b_vs_a"]["mse_pass"])
                decision = mse_pass and r_pass
                decisions.append(decision)
                context_report[head] = {
                    "methods": methods,
                    "raw_mse_unchanged": {
                        "static": head_parent["arms"]["static"]["validation"]["raw_mse"],
                        "static_plus_target_basal": head_parent["arms"]["static_plus_target_basal"][
                            "validation"
                        ]["raw_mse"],
                        "fractional_improvement": head_parent["arm_b_vs_a"][
                            "fractional_raw_mse_improvement"
                        ],
                    },
                    "mse_pass": mse_pass,
                    "corrected_correlation_no_regression_pass": r_pass,
                    "pass": decision,
                }
            report["contexts"][context] = context_report
    report["decision"] = {
        "advance": bool(all(decisions)),
        "passed_checks": int(sum(decisions)),
        "total_checks": len(decisions),
        "mse_failure_cannot_change": True,
    }
    report["claim_limits"] = [
        "Development validation only; no independent confirmation.",
        "Corrected scoring reuses saved forecasts and performs no fitting or alpha selection.",
        "RNA and ADT remain separate point-prediction heads.",
        "Mean forecasts are constant across genes after query-centroid removal and therefore correctly have undefined profile correlation.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-report-sha256", required=True)
    parser.add_argument("--predictions-sha256", required=True)
    args = parser.parse_args()
    if digest(args.parent_report) != args.parent_report_sha256:
        raise ValueError("parent report hash mismatch")
    if digest(args.predictions) != args.predictions_sha256:
        raise ValueError("predictions hash mismatch")
    print(json.dumps(rescore(args.parent_report, args.predictions, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
