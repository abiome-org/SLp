"""slpbench command line.

    slpbench build [--stage ...]
    slpbench eval PREDS --split dev|dev_semi|test|test_semi [--boot 200] [--out result.json]
    slpbench compare A_PREDS B_PREDS --split dev [--boot 200]
    slpbench check-leakage TRAIN_PAIRS [--allow-dev]
    slpbench baseline NAME --split dev [--out preds.parquet]
    slpbench battery [--split dev] [--boot N]     # score the SL model battery -> MODELS.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="slpbench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--stage", default="all")

    e = sub.add_parser("eval")
    e.add_argument("preds")
    e.add_argument("--split", default="dev", choices=["dev", "dev_semi", "test", "test_semi"])
    e.add_argument("--boot", type=int, default=0, help="family-cluster bootstrap reps for a 95%% CI")
    e.add_argument("--allow-missing", action="store_true")
    e.add_argument("--out", type=Path)

    c = sub.add_parser("compare", help="paired bootstrap: is B better than A?")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--split", default="dev", choices=["dev", "dev_semi", "test", "test_semi"])
    c.add_argument("--boot", type=int, default=200)
    c.add_argument("--out", type=Path, help="write the paired comparison as JSON")

    lk = sub.add_parser("check-leakage")
    lk.add_argument("records")
    lk.add_argument("--allow-dev", action="store_true", help="only test families are forbidden")

    sub.add_parser("card", help="regenerate DATA_CARD.md from the built benchmark")
    sub.add_parser("leaderboard", help="regenerate LEADERBOARD.md from leaderboard.yaml")
    sub.add_parser("audit", help="re-run label reproducibility checks, write REPLICATION.md")
    vf = sub.add_parser("verify", help="check artifact hashes and row counts against manifest.json")
    vf.add_argument("--raw", action="store_true", help="also hash all pinned raw source files")
    ep = sub.add_parser("export-public", help="copy model inputs without private test labels or propensities")
    ep.add_argument("out", type=Path)
    bt = sub.add_parser("battery", help="score every model in models/battery.yaml, write MODELS.md")
    bt.add_argument("--split", default="dev", choices=["dev", "dev_semi", "test", "test_semi"])
    bt.add_argument("--boot", type=int, default=0)

    bl = sub.add_parser("baseline")
    bl.add_argument("name")
    bl.add_argument("--split", default="dev")
    bl.add_argument("--out", type=Path)

    a = ap.parse_args(argv)
    if a.cmd == "build":
        from slpbench import build
        sys.argv = ["build", "--stage", a.stage]
        build.main()
    elif a.cmd == "eval":
        from slpbench import evaluate
        res = evaluate.evaluate(evaluate.read_predictions(a.preds), a.split, a.boot, a.allow_missing)
        res["predictions_sha256"] = evaluate.file_sha256(a.preds)
        print(evaluate.format_report(res))
        if a.out:
            evaluate.save(res, a.out)
    elif a.cmd == "compare":
        from slpbench import evaluate
        r = evaluate.compare(evaluate.read_predictions(a.a), evaluate.read_predictions(a.b), a.split, a.boot)
        r.update({"benchmark": evaluate.BENCH.name, "manifest_sha256": evaluate.file_sha256(evaluate.BENCH / "manifest.json"),
                  "scorer_version": evaluate.SCORER_VERSION,
                  "a_predictions_sha256": evaluate.file_sha256(a.a), "b_predictions_sha256": evaluate.file_sha256(a.b)})
        print(f"A={r['a']:.4f}  B={r['b']:.4f}  B-A={r['delta']:+.4f}  95% CI [{r['delta_ci95'][0]:+.4f}, "
              f"{r['delta_ci95'][1]:+.4f}]  P(B<=A)={r['p_b_not_better']:.3f}")
        print("  per species: " + "  ".join(f"{k}={v:+.4f}" for k, v in r["species_delta"].items()))
        if a.out:
            evaluate.save(r, a.out)
    elif a.cmd == "check-leakage":
        from slpbench import leakage
        sys.exit(leakage.main(a.records, a.allow_dev))
    elif a.cmd == "audit":
        from slpbench import audit
        audit.main()
    elif a.cmd in {"verify", "export-public"}:
        import json

        from slpbench import evaluate, release
        result = (release.verify_benchmark(evaluate.BENCH, a.raw) if a.cmd == "verify" else
                  release.export_public(evaluate.BENCH, a.out))
        print(json.dumps(result, indent=2))
    elif a.cmd == "leaderboard":
        from slpbench import leaderboard
        leaderboard.main()
    elif a.cmd == "card":
        from slpbench import card
        card.main()
    elif a.cmd == "battery":
        from slpbench import battery
        battery.run(a.split, a.boot)
        print(battery.report_path(a.split).read_text())
    elif a.cmd == "baseline":
        from slpbench import baselines
        out = a.out or Path(f"results/{a.name}_{a.split}.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        baselines.run(a.name, a.split).write_parquet(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
