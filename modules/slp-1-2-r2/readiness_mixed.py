"""Disposable real-corpus check of the ordinary two-phase fitting CLI."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from cloud_io import fetch_shards, get_json, request, put_json
from data import digest
from population import fetch as fetch_population

JOB = "readiness-r2-20260912-v2"
POPULATIONS = [
    "population-k562-r2-20260912-v1",
    "population-rpe1-r2-20260912-v1",
    "population-norman-r2-20260912-v1",
    "population-gwps-r2-20260912-v1",
    "population-hepg2-r2-20260912-v1",
    "population-mcf10a-full-d0-r2-20260912-v2",
    "population-mcf10a-full-d6-r2-20260912-v1",
    "population-mcf10a-tgfb1-d6-r2-20260912-v1",
    "population-yeast-r2-20260912-v4",
]


def prepare(root):
    """Materialize the already admitted corpus on cloud disk, without fitting."""
    root = Path(root)
    packed = [
        fetch_shards(
            "pack-human-r2-20260912-v1",
            "human-combinations-manifest.json",
            root / "human-pack",
        ),
        fetch_shards(
            "pack-depmap-r2-20260912-v1", "depmap-manifest.json", root / "depmap-pack"
        ),
    ]

    def fetch(job):
        path = fetch_population(job, root / job)
        print(json.dumps({"event": "population_verified", "job": job}), flush=True)
        return path

    with ThreadPoolExecutor(max_workers=3) as pool:
        populations = list(pool.map(fetch, POPULATIONS))
    basal_job = "basal-depmap-r2-20260912-v2"
    receipt = get_json(basal_job, "complete.json")
    path = root / "depmap-basal.json"
    with request(basal_job, "basal.json") as response:
        body = response.read(32 * 1024 * 1024 + 1)
    path.write_bytes(body)
    if len(body) != receipt["basal_bytes"] or digest(path) != receipt["basal_sha256"]:
        raise ValueError("Observational DepMap checksum mismatch")

    def spec(path):
        return {"path": str(path.relative_to(root)), "sha256": digest(path)}

    data = {
        "schema": "slp.fitting-corpus/v1",
        "packed": [spec(p) for p in packed],
        "populations": [spec(p) for p in populations],
        "basal": [spec(path)],
    }
    data_path = root / "mixed-corpus.json"
    data_path.write_text(json.dumps(data, sort_keys=True))
    return data_path, data, receipt


def main(args):
    start = time.monotonic()
    root = Path(args.root)
    output = root / "mixed-readiness"
    output.mkdir(exist_ok=True)
    if args.deadline - time.time() < 1200:
        raise ValueError("Insufficient guarded mixed-corpus readiness window")
    data_path, data, receipt = prepare(root)
    phase = {
        "quantitative_weights": {"rna": 0.5, "fitness": 0.25, "interaction": 0.25},
        "temperature": 0.7,
        "updates": 40,
        "batch_size": 16,
        "max_seconds": 600,
        "checkpoint_every": 40,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "warmup": 4,
        "max_queries": 128,
    }
    recipe = {
        "purpose": "disposable-readiness",
        "seed": 731,
        "cpu_threads": 4,
        "model": {},
        "pretrain": phase,
        "adapt": {**phase, "updates": 12, "checkpoint_every": 12, "sl_fraction": 0.75},
    }
    recipe_path = root / "mixed-readiness-recipe.json"
    recipe_path.write_text(json.dumps(recipe, sort_keys=True))
    reports = {}
    for stage in ("pretrain", "adapt"):
        command = [
            sys.executable,
            str(Path(__file__).parent / "fit.py"),
            "--features",
            str(root / "features"),
            "--recipe",
            str(recipe_path),
            "--data",
            str(data_path),
            "--fold",
            str(root / "benchmark/musl-s42-f0-fold.json"),
            "--scope",
            "inner",
            "--stage",
            stage,
            "--deadline",
            str(args.deadline),
            "--output",
            str(output / stage),
        ]
        if stage == "adapt":
            command += ["--initialize", str(output / "pretrain/checkpoint.pt")]
        subprocess.run(
            command, check=True, timeout=max(1, args.deadline - time.time() - 300)
        )
        result = json.loads((output / stage / "complete.json").read_text())
        identity = json.loads((output / stage / "identity.json").read_text())
        exposure = json.loads((output / stage / "exposure.json").read_text())
        if result["result"]["completed_update"] != recipe[stage]["updates"]:
            raise ValueError("Incomplete disposable optimizer budget")
        if set(identity["human_fitted_intervention_genes"]) & set(
            identity["forbidden_genes"]
        ):
            raise ValueError("Independent exposure union check failed")
        if stage == "adapt" and not any(
            x["reason"] == "human-only adaptation" for x in exposure["excluded_sources"]
        ):
            raise ValueError("Nonhuman RNA was not explicitly excluded from adaptation")
        reports[stage] = {"result": result, "identity": identity, "exposure": exposure}
    complete = {
        "state": "complete",
        "purpose": "disposable ordinary-CLI mixed-corpus readiness",
        "research_training_launched": False,
        "training_ready": False,
        "stages": reports,
        "corpus": data,
        "observational_context_receipt": receipt,
        "elapsed_seconds": time.monotonic() - start,
        "source": {
            p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    put_json(JOB, "complete.json", complete)
    (output / "complete.json").write_text(json.dumps(complete, sort_keys=True))
    print(
        json.dumps(
            {
                "event": "mixed_readiness_complete",
                "elapsed_seconds": complete["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/workspace/slp-r2")
    parser.add_argument("--deadline", type=float, required=True)
    try:
        main(parser.parse_args())
    except Exception as exc:
        report = {
            "state": "failed",
            "type": type(exc).__name__,
            "detail": str(exc)[:500]
            if isinstance(
                exc,
                (ValueError, KeyError, AssertionError, subprocess.CalledProcessError),
            )
            else None,
            "frames": [
                {"file": Path(f.filename).name, "line": f.lineno}
                for f in traceback.extract_tb(exc.__traceback__)
            ],
        }
        put_json(JOB, "failed.json", report)
        print(json.dumps(report), flush=True)
        raise SystemExit(1) from None
