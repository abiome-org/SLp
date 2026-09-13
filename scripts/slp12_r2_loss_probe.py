"""Bounded MSE pretraining comparison, restricted to the first inner fold.

Use a separately captured source directory and fresh weights. Original frozen
campaign checkpoints, recipes and test partitions are not modified.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main(args):
    root, data_root = args.root, args.data_root
    output = root / "diagnosis"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "slp.r2-numeric-loss-probe/v1",
        "scope": "musl-s42-f0 inner only; development selection",
        "change": "fresh MSE quantitative pretraining, SL-only adaptation",
        "started_at": time.time(),
        "deadline": args.deadline,
        "source": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.glob("*.py"))
        },
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "results": {},
    }

    def command(argv, logfile, timeout):
        if (data_root / "protocols/musl-s42-f0-test-manifest.json").exists():
            raise ValueError("Outer test opened during inner-only intervention")
        remaining = args.deadline - time.time()
        if remaining < 60:
            raise TimeoutError("Numerical repair deadline reached")
        with logfile.open("w") as log:
            subprocess.run(
                [sys.executable, *map(str, argv)],
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=min(timeout, remaining),
            )

    for variant in ("human_pretraining", "mixed_pretraining"):
        recipe_file = root / "campaign" / (variant + ".json")
        recipe = json.loads(recipe_file.read_text())
        assert recipe["pretrain"]["numeric_loss"] == "mse"
        assert recipe["adapt"]["sl_fraction"] == 1.0
        base = root / "runs/musl-s42-f0" / variant
        common = [
            root / "fit.py", "--features", data_root / "features",
            "--data", data_root / "mixed-corpus.json", "--fold",
            data_root / "protocols/musl-s42-f0-fold.json", "--recipe", recipe_file,
            "--scope", "inner", "--deadline", args.allocation_deadline,
            "--cache", data_root / "packed-cache",
        ]
        pretrain = base / "pretrain"
        pretrain.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"event": "pretrain", "variant": variant, "at": time.time()}), flush=True)
        command([*common, "--stage", "pretrain", "--output", pretrain], pretrain / "fit.log", 1600)
        completed = json.loads((pretrain / "complete.json").read_text())
        if completed["result"]["completed_update"] != 8000:
            raise ValueError("Incomplete pretraining comparison")
        for update in (2000, 8000):
            run = base / f"adapt-u{update:06d}"
            run.mkdir(parents=True, exist_ok=True)
            command(
                [*common, "--stage", "adapt", "--output", run,
                 "--initialize", pretrain / f"checkpoint-u{update:06d}.pt"],
                run / "fit.log", 400,
            )
            complete = json.loads((run / "complete.json").read_text())
            if complete["result"]["completed_update"] != 1000:
                raise ValueError("Incomplete SL adaptation comparison")
            name = f"{variant}-mse-u{update:06d}"
            command(
                [root / "score.py", "--checkpoint", run / "checkpoint.pt",
                 "--features", data_root / "features", "--fold",
                 data_root / "protocols/musl-s42-f0-fold.json", "--basal",
                 run / "basal.json", "--scope", "inner", "--candidate", name,
                 "--output", run / "scores"], run / "score.log", 120,
            )
            metrics = json.loads((run / "scores/metrics.json").read_text())
            report["results"][name] = metrics
            (output / "report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps({"event": "scored", "variant": name,
                              "average_precision": metrics["average_precision"],
                              "at": time.time()}), flush=True)
    command(
        [args.specificity, "--root", root, "--deadline", args.deadline,
         "--output", output / "specificity.json"], output / "specificity.log", 240,
    )
    report["completed_at"] = time.time()
    (output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/slp-r2"))
    parser.add_argument("--specificity", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--allocation-deadline", type=float, required=True)
    main(parser.parse_args())
