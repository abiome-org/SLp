"""Audited scoring amendment for official single-class SL validation folds.

This adjunct leaves captured training code, recipes, row sets and checkpoints
unchanged. Ordinary two-class metrics and selection retain their original path.
"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np


def amendment():
    return {
        "id": "single-class-scoring-v1",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "policy": "Undefined discrimination metrics are null; inner-only single-class selection minimizes binary log loss.",
    }


def metrics(labels, scores, original):
    y, s = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape or not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError("Metrics require finite scores for every binary label")
    n, positive = len(y), int(y.sum())
    loss = float(np.mean(np.logaddexp(0.0, s) - y * s)) if n else None
    if 0 < positive < n:
        result = original(labels, scores)
    else:
        result = {
            "n": n, "positives": positive, "prevalence": positive / n if n else None,
            "average_precision": None, "trapezoidal_pr_auc": None, "auroc": None,
        }
    result.update(binary_log_loss=loss,
                  metric_status="defined" if 0 < positive < n else "undefined_single_class" if n else "undefined_empty",
                  evaluation_amendment=amendment())
    return result


def select(reports, benchmarks, inner_folds, original):
    if all(r["average_precision"] is not None for r in reports):
        return original(reports, benchmarks, inner_folds)
    # The campaign selects independently within one official outer fold.
    # Never mix fallback evidence with AP or borrow another fold's validation.
    if len(benchmarks) != 1 or not isinstance(inner_folds, dict) or set(inner_folds) != set(benchmarks):
        raise ValueError("Single-class fallback requires one matched inner fold")
    benchmark = benchmarks[0]
    if len(inner_folds[benchmark]) != 1 or not reports:
        raise ValueError("Single-class fallback requires one matched inner fold")
    fold = inner_folds[benchmark][0]
    scores, cohort = {}, None
    for r in reports:
        loss, n, positives = r.get("binary_log_loss"), r["n"], r["positives"]
        if (r["partition"] != "inner" or r["forbidden_human_exposures"] != 0
                or r["benchmark"] != benchmark or r["fold"] != fold
                or r.get("metric_status") != "undefined_single_class"
                or r["average_precision"] is not None or n <= 0 or positives not in (0, n)
                or not isinstance(loss, (float, int)) or not np.isfinite(loss) or loss < 0
                or r["checkpoint"] in scores):
            raise ValueError("Invalid single-class inner selection evidence")
        current = (n, positives, tuple(r.get("forbidden_genes", [])), r.get("protocol_sha256"))
        if cohort is not None and current != cohort:
            raise ValueError("Single-class selection cohorts differ")
        cohort = current
        scores[r["checkpoint"]] = {"mean_binary_log_loss": loss, "benchmarks": {benchmark: loss}}
    winner = min(sorted(scores), key=lambda c: scores[c]["mean_binary_log_loss"])
    return {"selected": winner, "scores": scores,
            "selector": "minimum inner binary log loss; AP undefined on official single-class validation",
            "evaluation_amendment": amendment()}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        root = Path(sys.argv[2]).resolve()
        sys.path.insert(0, str(root))
        import evaluate
        original = evaluate.binary_metrics
        evaluate.binary_metrics = lambda y, s: metrics(y, s, original)
        sys.argv = [str(root / "score.py"), *sys.argv[3:]]
        runpy.run_path(str(root / "score.py"), run_name="__main__")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "plan", "run-id", "transport"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--execute-research", action="store_true")
    parser.add_argument("--allow-outer-test", action="store_true")
    args = parser.parse_args()
    transport = Path(args.transport)
    if hashlib.sha256(transport.read_bytes()).hexdigest() != "a8bc42dde93ecf27c510bd4b563726153a7c7e8ee7b4c52e87f7b3bc8c076067":
        raise ValueError("Transport adjunct changed")
    sys.path.insert(0, args.root)
    import campaign
    transport_api = runpy.run_path(str(transport))
    campaign.upload_directory = transport_api["upload_directory"]
    original_get = campaign.get_json
    campaign.get_json = lambda *a, **kw: transport_api["retry_io"](lambda: original_get(*a, **kw))
    original_select = campaign.select_checkpoint
    campaign.select_checkpoint = lambda r, b, f: select(r, b, f, original_select)

    class Runner(campaign.Runner):
        def execute(self, argv, log):
            command = self.command(argv)
            if len(command) > 1 and command[1] == "score.py":
                command = [command[0], str(Path(__file__).resolve()), "score", str(self.root), *command[2:]]
            return super().execute(command, log)

    runner = Runner(args)
    runner.validate()
    record = amendment()
    saved = runner.saved.setdefault("evaluation_amendments", {})
    if record["id"] not in saved:
        directory = runner.state / record["id"]
        directory.mkdir(exist_ok=True)
        (directory / "slp12_r2_degenerate.py").write_bytes(Path(__file__).read_bytes())
        (directory / "amendment.json").write_text(json.dumps(record, sort_keys=True))
        saved[record["id"]] = {**record, "artifact": runner.publish(directory, record["id"])}
        campaign.write_json(runner.journal, runner.saved)
    elif saved[record["id"]]["source_sha256"] != record["source_sha256"]:
        raise ValueError("Published evaluation amendment changed")
    print(json.dumps({"event": "evaluation_amendment", **record}), flush=True)
    print(json.dumps(runner.run()), flush=True)


if __name__ == "__main__":
    main()
