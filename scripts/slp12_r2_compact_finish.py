"""Finish the bounded compact comparison after its primary GPU run completes."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    root, data = args.root, args.data_root
    output = root / "compact-followup"
    output.mkdir(exist_ok=False)
    report = {"started_at": time.time(), "deadline": args.deadline,
              "scope": "retained checkpoints of the final first-inner-fold comparison",
              "driver_sha256": digest(__file__),
              "source": {p.name: digest(p) for p in sorted(root.glob("*.py"))},
              "metrics": {}}

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def guard():
        if time.time() > args.deadline - 30:
            raise TimeoutError("Compact checkpoint selection deadline")
        if list((data / "protocols").glob("*-test-manifest.json")):
            raise ValueError("Outer labels opened during inner-only development")

    def command(argv, log, seconds):
        guard()
        with log.open("w") as file:
            subprocess.run([sys.executable, *map(str, argv)], cwd=root,
                           stdout=file, stderr=subprocess.STDOUT, check=True,
                           timeout=min(seconds, args.deadline - time.time() - 20))

    def score(checkpoint, basal, candidate):
        destination = output / candidate
        command([root / "score.py", "--checkpoint", checkpoint,
                 "--features", root / "features", "--fold", data / "protocols/musl-s42-f0-fold.json",
                 "--basal", basal, "--scope", "inner", "--candidate", candidate,
                 "--output", destination], output / (candidate + ".log"), 120)
        report["metrics"][candidate] = json.loads((destination / "metrics.json").read_text())
        save()

    save()
    try:
        primary = root / "world-diagnosis/report.json"
        while True:
            guard()
            if primary.exists() and json.loads(primary.read_text()).get("completed_at"):
                break
            proc = Path("/proc") / str(args.primary_pid)
            if not proc.exists() or "slp-r2-compact-probe.py" not in (proc / "cmdline").read_text():
                raise RuntimeError("Primary comparison exited without completion")
            time.sleep(15)
        report["primary_report_sha256"] = digest(primary)
        mixed = root / "world-diagnosis/mixed_pretraining"
        score(mixed / "adapt/checkpoint-u001000.pt", mixed / "adapt/basal.json", "mixed32k-sl1k")
        # Selection of a retained pretraining point preserves the 32k LR schedule.
        # Only its fresh SL adapter is fitted here; the base is never retrained.
        recipe = json.loads((root / "world-campaign/mixed_pretraining.json").read_text())
        recipe["adapt"]["stop_at_update"] = 1000
        recipe["adapt"]["retain_updates"] = [1000]
        recipe_path = output / "mixed8k-sl1k-recipe.json"
        recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")
        report["early_recipe_sha256"] = digest(recipe_path)
        adapter = output / "mixed8k-adapt"
        adapter.mkdir()
        command([root / "fit.py", "--features", root / "features", "--data", data / "mixed-corpus.json",
                 "--fold", data / "protocols/musl-s42-f0-fold.json", "--recipe", recipe_path,
                 "--scope", "inner", "--stage", "adapt", "--output", adapter,
                 "--initialize", mixed / "pretrain/checkpoint-u008000.pt",
                 "--deadline", args.deadline, "--cache", data / "packed-cache"], adapter / "fit.log", 450)
        if json.loads((adapter / "complete.json").read_text())["result"]["completed_update"] != 1000:
            raise ValueError("Incomplete early-checkpoint adaptation")
        score(adapter / "checkpoint.pt", adapter / "basal.json", "mixed8k-sl1k")
        command([args.specificity, "--root", root, "--data-root", data,
                 "--run-root", root / "world-diagnosis", "--recipe", root / "world-campaign/mixed_pretraining.json",
                 "--variants", "mixed_pretraining", "--updates", 8000, 32000,
                 "--output", output / "specificity.json", "--deadline", args.deadline],
                output / "specificity.log", 360)
        report["specificity_sha256"] = digest(output / "specificity.json")
        report["completed_at"] = time.time()
    except Exception as error:
        report["failed_at"] = time.time()
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/slp-r2"))
    parser.add_argument("--primary-pid", type=int, required=True)
    parser.add_argument("--specificity", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    main(parser.parse_args())
