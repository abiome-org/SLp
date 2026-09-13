"""Matched static-GO world-model transfer, using only the first inner fold."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main(args):
    root, data = args.root, args.data_root
    sys.path.insert(0, str(root))
    from data import digest

    report = {
        "schema": "slp.r2-go-world-probe/v1",
        "scope": "first MuSL inner development split; repeated feedback",
        "started_at": time.time(),
        "deadline": args.deadline,
        "source": {p.name: digest(p) for p in sorted(root.glob("*.py"))},
        "driver_sha256": digest(__file__),
        "features_sha256": digest(root / "features/manifest.json"),
        "results": {},
    }
    output = root / "world-diagnosis"
    output.mkdir(exist_ok=False)

    def command(argv, log, seconds):
        if (data / "protocols/musl-s42-f0-test-manifest.json").exists():
            raise ValueError("Outer test opened during inner-only development")
        remaining = args.deadline - time.time()
        if remaining < 60:
            raise TimeoutError("GO world-model comparison deadline")
        with log.open("w") as file:
            subprocess.run(
                [sys.executable, *map(str, argv)],
                cwd=root,
                stdout=file,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=min(seconds, remaining),
            )

    for variant in ("no_pretraining", "mixed_pretraining"):
        recipe_path = root / "world-campaign" / (variant + ".json")
        recipe = json.loads(recipe_path.read_text())
        assert recipe["adapt"]["sl_fraction"] == 1
        base = output / variant
        base.mkdir()
        common = [
            root / "fit.py",
            "--features",
            root / "features",
            "--data",
            data / "mixed-corpus.json",
            "--fold",
            data / "protocols/musl-s42-f0-fold.json",
            "--recipe",
            recipe_path,
            "--scope",
            "inner",
            "--deadline",
            args.allocation_deadline,
            "--cache",
            data / "packed-cache",
        ]
        initialize = []
        if variant == "mixed_pretraining":
            assert recipe["pretrain"]["numeric_loss"] == "mse"
            assert not recipe["pretrain"].get("center_rna_queries", False)
            run = base / "pretrain"
            run.mkdir()
            print(
                json.dumps(
                    {"event": "pretrain", "variant": variant, "at": time.time()}
                ),
                flush=True,
            )
            command(
                [*common, "--stage", "pretrain", "--output", run], run / "fit.log", 1800
            )
            if (
                json.loads((run / "complete.json").read_text())["result"][
                    "completed_update"
                ]
                != 8000
            ):
                raise ValueError("Incomplete GO pretraining")
            initialize = ["--initialize", run / "checkpoint.pt"]
        run = base / "adapt"
        run.mkdir()
        command(
            [*common, "--stage", "adapt", "--output", run, *initialize],
            run / "fit.log",
            450,
        )
        if (
            json.loads((run / "complete.json").read_text())["result"][
                "completed_update"
            ]
            != 1000
        ):
            raise ValueError("Incomplete GO SL adaptation")
        name = variant + "-static-go"
        command(
            [
                root / "score.py",
                "--checkpoint",
                run / "checkpoint.pt",
                "--features",
                root / "features",
                "--fold",
                data / "protocols/musl-s42-f0-fold.json",
                "--basal",
                run / "basal.json",
                "--scope",
                "inner",
                "--candidate",
                name,
                "--output",
                run / "scores",
            ],
            run / "score.log",
            120,
        )
        report["results"][name] = json.loads((run / "scores/metrics.json").read_text())
        report["results"][name]["recipe_sha256"] = digest(recipe_path)
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(
            json.dumps(
                {
                    "event": "scored",
                    "variant": name,
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
    p.add_argument("--root", type=Path, default=Path("/workspace/slp-r2-go"))
    p.add_argument("--data-root", type=Path, default=Path("/workspace/slp-r2"))
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--allocation-deadline", type=float, required=True)
    main(p.parse_args())
