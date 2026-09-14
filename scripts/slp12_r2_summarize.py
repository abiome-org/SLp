"""Summarize matched outer folds without selecting models from test scores."""

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

METRICS = ("average_precision", "trapezoidal_pr_auc", "auroc")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(plan, journal, plan_sha, *, allow_partial=False):
    if plan["schema"] != "slp.r2-training-packet/v1" or journal["plan_sha256"] != plan_sha:
        raise ValueError("Plan/journal identity mismatch")
    protocols = {p["name"]: p for p in plan["protocols"]}
    if len(protocols) != 30 or len(plan["protocols"]) != 30:
        raise ValueError("Expected 30 unique declared folds")
    families = tuple(plan["variants"])
    if set(families) != {"feature_mlp", "no_pretraining", "mixed_pretraining"}:
        raise ValueError("Expected all three matched families")
    folds = journal["folds"]
    if not set(folds) <= set(protocols):
        raise ValueError("Undeclared completed fold")
    complete = set(folds) == set(protocols)
    if not complete and not allow_partial:
        raise ValueError("Incomplete suite; use --allow-partial for progress only")
    grouped = {p["benchmark"]: [] for p in protocols.values()}
    for name, fold in folds.items():
        spec = protocols[name]
        if set(fold["families"]) != set(families):
            raise ValueError("Incomplete matched comparison")
        cohort = None
        selected = []
        for family, result in fold["families"].items():
            metric = result["metrics"]
            if (metric["partition"] != "outer" or metric["fold"] != name
                    or metric["benchmark"] != spec["benchmark"]
                    or metric["protocol_sha256"] != spec["metadata_sha256"]
                    or metric["forbidden_human_exposures"] != 0):
                raise ValueError("Outer protocol/exposure contract mismatch")
            n, positives = metric["n"], metric["positives"]
            if type(n) is not int or type(positives) is not int or not 0 < positives < n:
                raise ValueError("Invalid binary evaluation counts")
            if not math.isclose(metric["prevalence"], positives / n, abs_tol=1e-12):
                raise ValueError("Prevalence/count mismatch")
            if any(not isinstance(metric[k], (int, float)) or not math.isfinite(metric[k])
                   or not 0 <= metric[k] <= 1 for k in METRICS):
                raise ValueError("Invalid evaluation metric")
            current = (n, positives, metric["prevalence"], tuple(metric["forbidden_genes"]))
            if cohort is not None and current != cohort:
                raise ValueError("Families evaluated different cohorts or masks")
            cohort = current
            if fold["selection"]["selected"] == result["selection"]["selected"]:
                selected.append(family)
        if len(selected) != 1 or fold["metrics"] != fold["families"][selected[0]]["metrics"]:
            raise ValueError("Reported winner differs from the sealed inner selection")
        grouped[spec["benchmark"]].append((name, fold, selected[0]))
    expected = Counter(p["benchmark"] for p in protocols.values())
    benchmarks = {}
    for benchmark, rows in grouped.items():
        models = {}
        for family in (*families, "inner_selected"):
            values = [fold["metrics"] if family == "inner_selected"
                      else fold["families"][family]["metrics"] for _, fold, _ in rows]
            models[family] = {key: statistics.mean(v[key] for v in values) if values else None
                              for key in METRICS}
        pairs = sum(fold["metrics"]["n"] for _, fold, _ in rows)
        positives = sum(fold["metrics"]["positives"] for _, fold, _ in rows)
        delta = {}
        for key in METRICS:
            values = [fold["families"]["mixed_pretraining"]["metrics"][key]
                      - fold["families"]["no_pretraining"]["metrics"][key] for _, fold, _ in rows]
            delta[key] = {"mean": statistics.mean(values) if values else None,
                          "positive_folds": sum(v > 0 for v in values),
                          "tied_folds": sum(v == 0 for v in values)}
        benchmarks[benchmark] = {
            "completed_folds": len(rows), "expected_folds": expected[benchmark],
            "folds": [name for name, _, _ in rows], "total_pair_rows": pairs, "positive_rows": positives,
            "pooled_prevalence": positives / pairs if pairs else None,
            "mean_fold_prevalence": statistics.mean(f["metrics"]["prevalence"] for _, f, _ in rows) if rows else None,
            "mean_metrics": models, "inner_selection_counts": dict(Counter(s for _, _, s in rows)),
            "paired_pretraining_minus_sl_only": delta,
        }
    macro = None
    if complete:
        macro = {family: {key: statistics.mean(b["mean_metrics"][family][key] for b in benchmarks.values())
                          for key in METRICS} for family in (*families, "inner_selected")}
    return {
        "schema": "slp.r2-matched-benchmark-summary/v1", "plan_sha256": plan_sha,
        "complete": complete, "completed_folds": len(folds), "expected_folds": len(protocols),
        "benchmarks": benchmarks, "equal_benchmark_means": macro,
        "weighting": "Equal fold weight within each benchmark; equal benchmark weight in the complete-suite macro score. No pooling of predictions.",
        "selection": "inner_selected preserves each fold's pre-test inner choice; family test scores do not select a new model.",
        "interpretation": plan["development_reuse"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "journal", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    plan_bytes, journal_bytes = args.plan.read_bytes(), args.journal.read_bytes()
    report = summarize(json.loads(plan_bytes), json.loads(journal_bytes),
                       hashlib.sha256(plan_bytes).hexdigest(), allow_partial=args.allow_partial)
    report.update(journal_sha256=hashlib.sha256(journal_bytes).hexdigest(), summarizer_sha256=digest(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "complete": report["complete"],
                      "completed_folds": report["completed_folds"]}))
