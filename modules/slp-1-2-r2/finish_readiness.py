"""Close readiness with the real corpus, retained checkpoint and R2 continuation."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from cloud_io import download_directory, fetch_shards, get_json, put_json
from data import Features, digest
from readiness_mixed import prepare

JOB = "optimizer-readiness-r2-20260912-v1"


def exact(a, b, name="state"):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or not torch.equal(a, b):
            raise ValueError("Resume mismatch: " + name)
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise ValueError("Resume keys mismatch: " + name)
        for k in a:
            exact(a[k], b[k], name + "/" + str(k))
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError("Resume sequence mismatch: " + name)
        for i, (x, y) in enumerate(zip(a, b)):
            exact(x, y, name + "/" + str(i))
    elif a != b:
        raise ValueError("Resume value mismatch: " + name)


def main(args):
    root = Path(args.root)
    out = root / "finish-readiness"
    out.mkdir(exist_ok=True)
    source = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}
    started = time.monotonic()
    if args.action == "prepare":
        download_directory("readiness-r2-20260912-v3", root / "previous-bundle")
        features = root / "features"
        features.mkdir(exist_ok=True)
        for name in ("sequence.npy", "annotation.npy", "known.npy", "genes.json"):
            os.link(root / "previous-bundle" / name, features / name)
        shutil.copyfile(
            root / "previous-bundle/features-manifest.json", features / "manifest.json"
        )
        Features(features)
        prepare(root)
        directory = root / "benchmark"
        directory.mkdir(exist_ok=True)
        job, name = "protocol-r2-20260912-v4", "musl-s42-f0-fold.json"
        (directory / name).write_text(json.dumps(get_json(job, name), sort_keys=True))
        fetch_shards(job, "musl-s42-f0-train-manifest.json", directory)
        print(
            json.dumps(
                {"event": "corpus_ready", "elapsed_seconds": time.monotonic() - started}
            ),
            flush=True,
        )
        return
    if args.deadline - time.time() < 1500:
        raise ValueError("Require time for optimizer persistence and cleanup")
    phase = dict(
        quantitative_weights={"rna": 0.5, "fitness": 0.25, "interaction": 0.25},
        updates=8,
        batch_size=16,
        max_seconds=600,
        checkpoint_every=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup=2,
        max_queries=128,
        context_dropout=0.1,
        retain_updates=[4, 8],
    )
    recipe = {
        "purpose": "disposable-readiness",
        "seed": 731,
        "cpu_threads": 4,
        "model": {},
        "pretrain": phase,
        "adapt": {**phase, "sl_fraction": 0.75},
    }
    recipe_path = out / "recipe.json"
    recipe_path.write_text(json.dumps(recipe, sort_keys=True))

    def fit(stage, destination, *extra):
        command = [
            sys.executable,
            str(Path(__file__).with_name("fit.py")),
            "--features",
            str(root / "features"),
            "--data",
            str(root / "mixed-corpus.json"),
            "--fold",
            str(root / "benchmark/musl-s42-f0-fold.json"),
            "--recipe",
            str(recipe_path),
            "--scope",
            "inner",
            "--stage",
            stage,
            "--deadline",
            str(args.deadline),
            "--output",
            str(destination),
            "--cache",
            str(root / "packed-cache"),
            *extra,
        ]
        with (out / (destination.name + ".log")).open("w") as log:
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=max(1, args.deadline - time.time() - 600),
            )
        report = json.loads((destination / "complete.json").read_text())
        if report["result"]["completed_update"] != 8:
            raise ValueError("Disposable fitting did not complete")
        print(
            json.dumps(
                {"event": "fit_complete", "stage": stage, "run": destination.name}
            ),
            flush=True,
        )
        return report

    saved_pretrain = out / "pretrain/complete.json"
    reports = {
        "pretrain": json.loads(saved_pretrain.read_text())
        if saved_pretrain.exists()
        else fit("pretrain", out / "pretrain")
    }
    if args.action == "checkpoint":
        print(
            json.dumps(
                {
                    "event": "checkpoint_ready_for_backup",
                    "checkpoint": str(out / "pretrain/checkpoint-u000004.pt"),
                    "remote_upload_performed": False,
                }
            ),
            flush=True,
        )
        return
    checkpoint = out / "pretrain/checkpoint-u000004.pt"
    store = Path(args.checkpoint_store)

    def transfer(action, path):
        command = [
            sys.executable,
            str(store),
            action,
            "--path",
            str(path),
            "--job",
            JOB,
            "--module-directory",
            str(Path(__file__).parent),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=max(1, args.deadline - time.time() - 600),
        )
        receipt = json.loads(result.stdout)
        print(json.dumps({"event": "checkpoint_" + action, **receipt}), flush=True)
        return receipt

    published = transfer("publish", checkpoint)
    restored = transfer("restore", out / "restored")
    if published["sha256"] != restored["sha256"]:
        raise ValueError("Remote checkpoint hash changed")
    reports["resumed"] = fit(
        "pretrain", out / "resumed", "--resume", str(out / "restored/checkpoint.pt")
    )
    before = torch.load(
        out / "pretrain/checkpoint.pt", map_location="cpu", weights_only=True, mmap=True
    )
    after = torch.load(
        out / "resumed/checkpoint.pt", map_location="cpu", weights_only=True, mmap=True
    )
    exact(before, after)
    del before, after
    reports["adapt"] = fit(
        "adapt", out / "adapt", "--initialize", str(out / "resumed/checkpoint.pt")
    )
    if source != {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}:
        raise ValueError("Source changed during the final check")
    report = {
        "state": "complete",
        "purpose": "disposable engineering readiness",
        "research_training_launched": False,
        "official_test_scoring": False,
        "optimizer_r2_roundtrip_exact": True,
        "continued_full_state_bitwise_exact": True,
        "checkpoint": published,
        "stages": reports,
        "source": source,
        "corpus_sha256": digest(root / "mixed-corpus.json"),
        "elapsed_seconds": time.monotonic() - started,
    }
    (out / "complete.json").write_text(json.dumps(report, sort_keys=True))
    put_json(JOB, "complete.json", report)
    print(
        json.dumps(
            {"event": "readiness_complete", "seconds": report["elapsed_seconds"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "checkpoint", "verify"))
    p.add_argument("--root", default="/workspace/slp-r2")
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument(
        "--checkpoint-store", default="/workspace/slp12_r2_checkpoint_store.py"
    )
    main(p.parse_args())
