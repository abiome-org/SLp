"""slbench command line.

    slbench build [--stage ...]
    slbench eval PREDS --split dev|dev_semi|test|test_semi [--boot 200] [--out result.json]
    slbench compare A_PREDS B_PREDS --split dev [--boot 200]
    slbench check-leakage TRAIN_PAIRS [--allow-dev] [--allow-unknown]
    slbench baseline NAME --split dev [--out preds.parquet]
    slbench battery [--split dev] [--boot N]     # score the SL model battery -> results/reports/models_<split>.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="slbench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--stage", default="all")

    e = sub.add_parser("eval")
    e.add_argument("preds")
    e.add_argument("--split", default="dev", choices=["dev", "dev_semi", "test", "test_semi"])
    e.add_argument("--boot", type=int, default=0, help="family-cluster bootstrap reps for a 95%% CI")
    e.add_argument("--allow-missing", action="store_true")
    e.add_argument("--out", type=Path)

    ae = sub.add_parser("ancestry-eval", help="matched human cell-line ancestry diagnostic")
    ae.add_argument("preds", help="complete SLB test predictions")
    ae.add_argument("--out", type=Path, help="write the full line-level JSON scorecard")

    c = sub.add_parser("compare", help="paired bootstrap: is B better than A?")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--split", default="dev", choices=["dev", "dev_semi", "test", "test_semi"])
    c.add_argument("--boot", type=int, default=200)
    c.add_argument("--out", type=Path, help="write the paired comparison as JSON")

    lk = sub.add_parser("check-leakage")
    lk.add_argument("records")
    lk.add_argument("--allow-dev", action="store_true", help="only test families are forbidden")
    lk.add_argument("--allow-unknown", action="store_true",
                    help="do not fail on genes that cannot be resolved or are absent from the homology graph")

    sub.add_parser("card", help="write results/reports/data_card.md from the built benchmark")
    lb = sub.add_parser("leaderboard", help="verify leaderboard.yaml and update the README leaderboard")
    lb.add_argument("--verify-ci", action="store_true",
                    help="also re-run each entry's bootstrap CI (~15 s per entry; or set SLB_VERIFY_CI=1)")
    sub.add_parser("audit", help="re-run label reproducibility checks, write results/reports/replication.md")
    vf = sub.add_parser("verify", help="check artifact hashes and row counts against manifest.json")
    vf.add_argument("--raw", action="store_true", help="also hash all pinned raw source files")
    ep = sub.add_parser("export-public", help="copy model inputs without private test labels or propensities")
    ep.add_argument("out", type=Path)
    bt = sub.add_parser("battery", help="score every model in battery.yaml, write results/reports/models_<split>.md")
    bt.add_argument("--split", default="dev", choices=["dev", "dev_semi", "test", "test_semi"])
    bt.add_argument("--boot", type=int, default=0)

    bl = sub.add_parser("baseline")
    bl.add_argument("name")
    bl.add_argument("--split", default="dev")
    bl.add_argument("--out", type=Path)

    a = ap.parse_args(argv)
    if a.cmd != "build":
        from slbench.evaluate import BENCH

        if not (BENCH / "manifest.json").exists():
            raise SystemExit(f"no benchmark at {BENCH}: set SLB_BENCH=data/release/slb to use the public bundle "
                             "(README, Quickstart), or build the private one with `slbench build`")
    if a.cmd == "build":
        from slbench import build
        sys.argv = ["build", "--stage", a.stage]
        build.main()
    elif a.cmd == "eval":
        from slbench import evaluate
        res = evaluate.evaluate(evaluate.read_predictions(a.preds), a.split, a.boot, a.allow_missing)
        res["predictions_sha256"] = evaluate.file_sha256(a.preds)
        print(evaluate.format_report(res))
        if a.out:
            evaluate.save(res, a.out)
    elif a.cmd == "ancestry-eval":
        import json

        import polars as pl

        from slbench import ancestry, evaluate

        gold = evaluate.load_gold("test")
        preds = evaluate.read_predictions(a.preds)
        evaluate._log_test_eval("test", preds)
        scored, missing = evaluate.validated_join(gold, preds, inputs=evaluate.input_ids("test"))
        if missing:
            raise ValueError("ancestry evaluation requires complete test predictions")
        contexts = pl.read_parquet(evaluate.BENCH / "contexts.parquet")
        ancestry.verify_dutil_annotations(contexts)
        bench = ancestry.AncestryBenchmark(gold, contexts)
        res = {"protocol": bench.cfg, "benchmark": evaluate.bench_id(), "split": "test",
               "manifest_sha256": evaluate.file_sha256(evaluate.BENCH / "manifest.json"),
               "predictions_sha256": evaluate.file_sha256(a.preds),
               "panel_support": bench.support,
               "complete_case_panel_support": bench.complete_case_support,
               "comparisons": bench.evaluate(scored),
               "complete_case_comparisons": bench.evaluate(scored, complete_case=True)}
        for group, row in res["comparisons"].items():
            print(f"{group} vs EUR: AUROC {row['auroc'][group]:.3f} vs {row['auroc']['EUR']:.3f}; "
                  f"gap {row['gap_auroc']:+.3f}; {row['verdict']}")
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(res, indent=2, sort_keys=True, allow_nan=False) + "\n")
    elif a.cmd == "compare":
        from slbench import evaluate
        r = evaluate.compare(evaluate.read_predictions(a.a), evaluate.read_predictions(a.b), a.split, a.boot)
        r.update({"benchmark": evaluate.bench_id(), "manifest_sha256": evaluate.file_sha256(evaluate.BENCH / "manifest.json"),
                  "scorer_version": evaluate.SCORER_VERSION,
                  "a_predictions_sha256": evaluate.file_sha256(a.a), "b_predictions_sha256": evaluate.file_sha256(a.b)})
        print(f"A={r['a']:.4f}  B={r['b']:.4f}  B-A={r['delta']:+.4f}  95% CI [{r['delta_ci95'][0]:+.4f}, "
              f"{r['delta_ci95'][1]:+.4f}]  P(B<=A)={r['p_b_not_better']:.3f}")
        print("  per species: " + "  ".join(f"{k}={v:+.4f}" for k, v in r["species_delta"].items()))
        if a.out:
            evaluate.save(r, a.out)
    elif a.cmd == "check-leakage":
        from slbench import leakage
        sys.exit(leakage.main(a.records, a.allow_dev, a.allow_unknown))
    elif a.cmd == "audit":
        from slbench import audit
        audit.main()
    elif a.cmd in {"verify", "export-public"}:
        import json

        from slbench import evaluate, release
        result = (release.verify_benchmark(evaluate.BENCH, a.raw) if a.cmd == "verify" else
                  release.export_public(evaluate.BENCH, a.out))
        print(json.dumps(result, indent=2))
    elif a.cmd == "leaderboard":
        from slbench import leaderboard
        leaderboard.main(verify_ci=a.verify_ci or None)
    elif a.cmd == "card":
        from slbench import card
        card.main()
    elif a.cmd == "battery":
        from slbench import battery
        battery.run(a.split, a.boot)
        print(battery.report_path(a.split).read_text())
    elif a.cmd == "baseline":
        from slbench import baselines
        out = a.out or Path(f"results/{a.name}_{a.split}.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        baselines.run(a.name, a.split).write_parquet(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
