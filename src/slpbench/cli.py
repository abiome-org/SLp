"""slpbench command line.

    slpbench build [--stage ...]
    slpbench eval PREDS --split dev|dev_semi|test|test_semi [--boot 200] [--out result.json]
    slpbench compare A_PREDS B_PREDS --split dev [--boot 200]
    slpbench check-leakage TRAIN_PAIRS [--allow-dev]
    slpbench baseline NAME --split dev [--out preds.parquet]
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

    lk = sub.add_parser("check-leakage")
    lk.add_argument("records")
    lk.add_argument("--allow-dev", action="store_true", help="only test families are forbidden")

    sub.add_parser("card", help="regenerate DATA_CARD.md from the built benchmark")
    sub.add_parser("leaderboard", help="regenerate LEADERBOARD.md from leaderboard.yaml")
    sub.add_parser("audit", help="re-run label reproducibility checks, write REPLICATION.md")

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
        print(evaluate.format_report(res))
        if a.out:
            evaluate.save(res, a.out)
    elif a.cmd == "compare":
        from slpbench import evaluate
        r = evaluate.compare(evaluate.read_predictions(a.a), evaluate.read_predictions(a.b), a.split, a.boot)
        print(f"A={r['a']:.4f}  B={r['b']:.4f}  B-A={r['delta']:+.4f}  95% CI [{r['delta_ci95'][0]:+.4f}, "
              f"{r['delta_ci95'][1]:+.4f}]  P(B<=A)={r['p_b_not_better']:.3f}")
        print("  per species: " + "  ".join(f"{k}={v:+.4f}" for k, v in r["species_delta"].items()))
    elif a.cmd == "check-leakage":
        from slpbench import leakage
        sys.exit(leakage.main(a.records, a.allow_dev))
    elif a.cmd == "audit":
        from slpbench import audit
        audit.main()
    elif a.cmd == "leaderboard":
        from slpbench import leaderboard
        leaderboard.main()
    elif a.cmd == "card":
        from slpbench import card
        card.main()
    elif a.cmd == "baseline":
        from slpbench import baselines
        out = a.out or Path(f"results/{a.name}_{a.split}.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        baselines.run(a.name, a.split).write_parquet(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
