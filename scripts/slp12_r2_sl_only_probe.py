"""Bounded inner-fold adaptation ablation using the frozen fitting/scoring CLIs.

Compare SL-only adaptation to the original quantitative-replay recipe before
opening this fold's official test. Checkpoints and corpus stay on the GPU disk.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main(args):
    root = args.root
    if (root / "protocols/musl-s42-f0-test-manifest.json").exists():
        raise ValueError("This diagnostic must precede official outer-test access")
    output = root / "diagnosis/sl-only"
    output.mkdir(parents=True, exist_ok=True)
    recipe = json.loads((root / "campaign/mixed_pretraining.json").read_text())
    recipe["adapt"]["sl_fraction"] = 1.0
    recipe_file = output / "recipe.json"
    recipe_file.write_text(json.dumps(recipe, sort_keys=True))
    report = {
        "schema": "slp.r2-adaptation-probe/v1",
        "scope": "musl-s42-f0 inner only",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "change": "SL fraction 0.75 to 1.0; other adaptation settings unchanged",
        "deadline": args.deadline,
        "started_at": time.time(),
        "results": {},
    }
    candidates = [
        ("no-pretraining", None),
        ("human-u002000", "human_pretraining/pretrain/checkpoint-u002000.pt"),
        ("human-u008000", "human_pretraining/pretrain/checkpoint-u008000.pt"),
        ("mixed-u008000", "mixed_pretraining/pretrain/checkpoint-u008000.pt"),
    ]
    for name, initializer in candidates:
        if args.deadline - time.time() < 300:
            raise TimeoutError("Insufficient time for another bounded probe")
        run = output / name
        run.mkdir(exist_ok=True)
        command = [
            sys.executable,
            str(root / "fit.py"),
            "--features",
            str(root / "features"),
            "--data",
            str(root / "mixed-corpus.json"),
            "--fold",
            str(root / "protocols/musl-s42-f0-fold.json"),
            "--recipe",
            str(recipe_file),
            "--scope",
            "inner",
            "--stage",
            "adapt",
            "--deadline",
            str(args.allocation_deadline),
            "--output",
            str(run),
            "--cache",
            str(root / "packed-cache"),
        ]
        if initializer:
            command += ["--initialize", str(root / "runs/musl-s42-f0" / initializer)]
        print(
            json.dumps({"event": "probe_start", "candidate": name, "at": time.time()}),
            flush=True,
        )
        with (run / "fit.log").open("w") as log:
            subprocess.run(
                command,
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=min(300, args.deadline - time.time() - 30),
            )
        complete = json.loads((run / "complete.json").read_text())
        if complete["result"]["completed_update"] != 1000:
            raise ValueError("Incomplete adaptation probe")
        score = [
            sys.executable,
            str(root / "score.py"),
            "--checkpoint",
            str(run / "checkpoint.pt"),
            "--features",
            str(root / "features"),
            "--fold",
            str(root / "protocols/musl-s42-f0-fold.json"),
            "--basal",
            str(run / "basal.json"),
            "--scope",
            "inner",
            "--candidate",
            name + "-sl-only",
            "--output",
            str(run / "scores"),
        ]
        with (run / "score.log").open("w") as log:
            subprocess.run(
                score,
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=120,
            )
        report["results"][name] = json.loads((run / "scores/metrics.json").read_text())
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(
            json.dumps(
                {
                    "event": "probe_scored",
                    "candidate": name,
                    "average_precision": report["results"][name]["average_precision"],
                    "at": time.time(),
                }
            ),
            flush=True,
        )
    report["completed_at"] = time.time()
    (output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/workspace/slp-r2"))
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--allocation-deadline", type=float, required=True)
    main(p.parse_args())
