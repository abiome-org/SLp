"""Finite first-inner-fold comparison with newly prepared static GO inputs."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main(args):
    root, data = args.root, args.data_root
    sys.path.insert(0, str(root))
    import cloud_io

    from data import digest

    if (data / "protocols/musl-s42-f0-test-manifest.json").exists():
        raise ValueError("Outer test was opened")
    output = root / "diagnosis"
    output.mkdir(exist_ok=False)
    report = {
        "schema": "slp.r2-go-probe/v1",
        "scope": "musl-s42-f0 inner development only",
        "started_at": time.time(),
        "deadline": args.deadline,
        "source": {p.name: digest(p) for p in sorted(root.glob("*.py"))},
        "driver_sha256": digest(__file__),
        "materializer_sha256": digest(args.materializer),
    }
    paths = []
    try:
        for species in ("human", "yeast"):
            expected = json.loads(
                (args.metadata / ("go-" + species + "-manifest.json")).read_text()
            )
            path = cloud_io.fetch_shards(
                expected["job"], "annotations-manifest.json", root / ("go-" + species)
            )
            if json.loads(path.read_text()) != expected:
                raise ValueError("Prepared GO provenance changed")
            paths.append(path)
    finally:
        Path(os.environ["SLP_TRANSFER_TICKETS"]).unlink(missing_ok=True)
    spec = importlib.util.spec_from_file_location("go_features", args.materializer)
    materializer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(materializer)
    features = materializer.prepare(
        data / "features",
        args.metadata / "identity-manifest-raw.json",
        paths,
        root / "features",
        args.preparer_sha,
    )
    dimension = features["functional_annotations"]["annotation_dim"]
    report["features_sha256"] = digest(root / "features/manifest.json")
    report["annotation_dim"] = dimension
    recipe = json.loads((data / "campaign/feature_mlp.json").read_text())
    assert recipe["baseline"] == "feature_mlp" and recipe["adapt"]["sl_fraction"] == 1
    recipe["model"]["annotation_dim"] = dimension
    recipe_path = output / "feature_mlp.json"
    recipe_path.write_text(json.dumps(recipe, indent=2))
    report["recipe_sha256"] = digest(recipe_path)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    run = output / "feature_mlp"
    run.mkdir()

    def command(argv, logfile, timeout):
        if (data / "protocols/musl-s42-f0-test-manifest.json").exists():
            raise ValueError("Outer test opened during development")
        remaining = args.deadline - time.time()
        if remaining < 60:
            raise TimeoutError("GO comparison deadline")
        with logfile.open("w") as log:
            subprocess.run(
                [sys.executable, *map(str, argv)],
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=min(remaining, timeout),
            )

    command(
        [
            root / "fit.py",
            "--features",
            root / "features",
            "--data",
            data / "mixed-corpus.json",
            "--fold",
            data / "protocols/musl-s42-f0-fold.json",
            "--recipe",
            recipe_path,
            "--stage",
            "adapt",
            "--scope",
            "inner",
            "--deadline",
            args.allocation_deadline,
            "--cache",
            data / "packed-cache",
            "--output",
            run,
        ],
        run / "fit.log",
        600,
    )
    complete = json.loads((run / "complete.json").read_text())
    if complete["result"]["completed_update"] != recipe["adapt"]["updates"]:
        raise ValueError("Incomplete matched baseline training")
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
            "feature_mlp-static-go",
            "--output",
            run / "scores",
        ],
        run / "score.log",
        120,
    )
    report["metrics"] = json.loads((run / "scores/metrics.json").read_text())
    report["completed_at"] = time.time()
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "event": "scored",
                "average_precision": report["metrics"]["average_precision"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("/workspace/slp-r2"))
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--materializer", type=Path, required=True)
    p.add_argument("--preparer-sha", required=True)
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--allocation-deadline", type=float, required=True)
    main(p.parse_args())
