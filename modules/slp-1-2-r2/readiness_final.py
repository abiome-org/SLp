"""Final bounded CUDA, all-fold exposure and portable real-context checks."""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from artifact import export, load
from audit_folds import audit, load_protocols
from benchmarks import Benchmark
from cloud_io import put_json, upload_directory, download_directory
from data import Features, Scales, BatchBuilder, digest
from fit import views
from model import Config, WorldModel

JOB = "readiness-r2-20260912-v3"


def main(args):
    start = time.monotonic()
    root = Path(args.root)
    out = root / "final-readiness"
    out.mkdir(exist_ok=True)
    if args.deadline - time.time() < 900:
        raise ValueError("Insufficient guarded final readiness time")
    torch.manual_seed(731)
    torch.set_num_threads(4)
    source = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    features = Features(root / "features")
    genes = list(features.index)
    corpus = json.loads((root / "mixed-corpus.json").read_text())
    protocols = load_protocols(os.environ["SLP_TRANSFER_TICKETS"])
    arrays = []
    cache = root / "mixed-readiness/packed-cache/cache.json"
    c = json.loads(cache.read_text())
    arrays.append(
        (
            cache.parent / "measurements.npy",
            c["templates"],
            "packed-human-combinations-and-DepMap",
        )
    )
    for spec in corpus["populations"]:
        path = root / spec["path"]
        m = json.loads(path.read_text())
        arrays.append((path.parent / "units.npy", m["templates"], m["cohort"]))
    exposure = audit(protocols, genes, arrays)
    (out / "all-folds.json").write_text(json.dumps(exposure, sort_keys=True))
    print(
        json.dumps({"event": "all_fold_masks_checked", "views": len(protocols)}),
        flush=True,
    )
    base = json.loads((root / "mixed-readiness-recipe.json").read_text())
    base["pretrain"].update(updates=24, checkpoint_every=24, context_dropout=0.1)
    base["adapt"].update(updates=8, checkpoint_every=8, context_dropout=0.1)
    recipe = out / "recipe.json"
    recipe.write_text(json.dumps(base, sort_keys=True))
    results = {}

    def run(stage, recipe_path, directory, initialize=None):
        command = [
            sys.executable,
            str(Path(__file__).parent / "fit.py"),
            "--features",
            str(root / "features"),
            "--recipe",
            str(recipe_path),
            "--data",
            str(root / "mixed-corpus.json"),
            "--fold",
            str(root / "benchmark/musl-s42-f0-fold.json"),
            "--scope",
            "inner",
            "--stage",
            stage,
            "--deadline",
            str(args.deadline),
            "--output",
            str(directory),
        ]
        if initialize:
            command += ["--initialize", str(initialize)]
        subprocess.run(
            command, check=True, timeout=max(1, args.deadline - time.time() - 300)
        )
        result = json.loads((directory / "complete.json").read_text())
        if (
            result["result"]["completed_update"]
            != json.loads(recipe_path.read_text())[stage]["updates"]
        ):
            raise ValueError("Final disposable budget was incomplete")
        return result

    results["pretrain"] = run("pretrain", recipe, out / "pretrain")
    results["adapt"] = run(
        "adapt", recipe, out / "adapt", out / "pretrain/checkpoint.pt"
    )
    for kind in ("feature_mlp", "sequence_similarity"):
        r = copy.deepcopy(base)
        r["baseline"] = kind
        r["adapt"].update(
            updates=4, checkpoint_every=4, sl_fraction=1.0, context_dropout=0.0
        )
        path = out / (kind + ".json")
        path.write_text(json.dumps(r, sort_keys=True))
        results[kind] = run("adapt", path, out / kind)
    benchmark = Benchmark(root / "benchmark/musl-s42-f0-fold.json", features)
    # Exercise the human-pretraining source mask on the full data, independent of adaptation.
    human_views, excluded, _ = views(
        root / "mixed-corpus.json",
        features,
        benchmark.fold("inner"),
        "pretrain",
        out / "packed-cache",
        taxa=[9606],
    )
    for v in human_views:
        for f in v.families:
            if any(
                v.templates[int(i)]["taxon"] != 9606
                for i in np.unique(v.rows["template"][f["indices"]])
            ):
                raise ValueError("Human-only comparison includes nonhuman data")
    del human_views
    state = torch.load(
        out / "adapt/checkpoint.pt", map_location="cpu", weights_only=True, mmap=True
    )
    model = WorldModel(Config(**state["config"])).cuda().eval()
    model.load_state_dict(state["model"])
    identity = state["identity"]
    basal = json.loads((out / "adapt/basal.json").read_text())
    builder = BatchBuilder(
        features, Scales([]), identity["vocabulary"], model.config, basal
    )
    rows = benchmark.records("inner", "fit")[:16]
    batch = {k: v.cuda() for k, v in builder([[r] for r in rows]).items()}
    with torch.no_grad():
        expected = model(batch)["sl_logit"].float().cpu()
    export(
        out / "export",
        model,
        root / "features",
        vocabulary=identity["vocabulary"],
        basal=basal,
        provenance={
            "disposable": True,
            "research_checkpoint": False,
            "identity": identity,
        },
    )
    artifact = upload_directory(out / "export", JOB)
    del model, state
    torch.cuda.empty_cache()
    download_directory(JOB, out / "replay")
    restored, *_ = load(out / "replay", "cuda")
    with torch.no_grad():
        torch.testing.assert_close(
            restored(batch)["sl_logit"].float().cpu(), expected, rtol=0, atol=0
        )
    if source != {
        p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))
    }:
        raise ValueError("Source changed during readiness")
    report = {
        "state": "complete",
        "purpose": "final engineering readiness",
        "research_training_launched": False,
        "training_ready": False,
        "all_fold_exposure": exposure,
        "phase_checks": results,
        "human_only_pretrain_sources_excluded": excluded,
        "r2_real_context_export_replay_exact": True,
        "bundle_bytes": sum(f["bytes"] for f in artifact["files"].values()),
        "elapsed_seconds": time.monotonic() - start,
        "source": source,
        "corpus_sha256": digest(root / "mixed-corpus.json"),
        "remaining_launch_decisions": [
            "Costanzo source permission remains unresolved",
            "research campaign budget and launch remain unapproved",
        ],
    }
    put_json(JOB, "complete.json", report)
    (out / "complete.json").write_text(json.dumps(report, sort_keys=True))
    print(
        json.dumps(
            {
                "event": "final_readiness_complete",
                "elapsed_seconds": report["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/workspace/slp-r2")
    p.add_argument("--deadline", type=float, required=True)
    try:
        main(p.parse_args())
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
