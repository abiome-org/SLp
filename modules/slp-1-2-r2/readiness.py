"""Bounded disposable checks on real Cloudflare data; no research selection."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from artifact import export, load
from benchmarks import Benchmark, verify_training_labels
from cloud_io import (
    fetch_shards,
    get_json,
    put_json,
    upload_directory,
    download_directory,
)
from data import Features, Scales, BatchBuilder, ConditionSampler, digest
from model import Config, WorldModel
from packed import materialize, View, Sampler, Builder, vocabulary
from train import objective, optimize, save_checkpoint, resume

JOB = "readiness-r2-20260912-v1"


def announce(event, **values):
    print(json.dumps({"event": event, **values}, allow_nan=False), flush=True)


def weights_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def step(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    loss = objective(model(batch), batch)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite real-data loss")
    loss.backward()
    if model.blocks[0].qkv.weight.grad.abs().sum() == 0:
        raise ValueError("No backbone SL gradient")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def main(args):
    started = time.monotonic()
    root = Path(args.root)
    device = "cuda"
    if args.deadline - time.time() < 900:
        raise ValueError(
            "Insufficient guarded time for readiness and artifact verification"
        )
    if not torch.cuda.is_available():
        raise ValueError("CUDA readiness requires CUDA")
    torch.manual_seed(731)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    features = Features(root / "features")
    genes = list(features.index)
    manifest = fetch_shards(
        "pack-human-r2-20260912-v1",
        "human-combinations-manifest.json",
        root / "human-pack",
    )
    cache = materialize([manifest], root / "corpus-cache", genes)
    protocol_job = "protocol-r2-20260912-v4"
    fold_name = "musl-s42-f0-fold.json"
    fold = get_json(protocol_job, fold_name)
    fold_dir = root / "benchmark"
    fold_dir.mkdir(exist_ok=True)
    (fold_dir / fold_name).write_text(json.dumps(fold, sort_keys=True))
    # Only outer-training labels are fetched. Test labels are not needed here.
    fetch_shards(protocol_job, fold["partitions"]["train"], fold_dir)
    benchmark = Benchmark(fold_dir / fold_name, features)
    mask = benchmark.fold("inner")
    view = View(cache, genes, mask, "pretrain")
    held = np.array([g in mask.forbidden for g in genes] + [False])
    for family in view.families:
        for lo in range(0, len(family["indices"]), 262144):
            rows = view.rows[family["indices"][lo : lo + 262144]]
            if held[rows["targets"]].any():
                raise ValueError("Independent real-corpus exposure audit failed")
    vocab = vocabulary(
        [*view.templates, {"assay": "human-SL", "taxon": 9606, "scope": "pan-cancer"}]
    )
    packed_manifest = json.loads(manifest.read_text())
    vocab.update({k: packed_manifest[k + "s"] for k in ("mechanism", "method")})
    config = Config()
    model = WorldModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    sampler = Sampler(
        view, {kind: 1.0 for kind in {f["kind"] for f in view.families}}, seed=731
    )
    builder = Builder(features, view.templates, view.scales, vocab, config)
    identity = {
        "purpose": "disposable numerical readiness",
        "corpus": view.signature,
        "features": digest(root / "features/manifest.json"),
        "protocol": benchmark.identity,
        "updates": 40,
        "source": {
            p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    announce(
        "real_corpus_loaded",
        input_rows=view.receipt["input_rows"],
        fitting_rows=view.receipt["fitting_rows"],
        parameters=sum(p.numel() for p in model.parameters()),
        elapsed_seconds=time.monotonic() - started,
    )
    torch.cuda.reset_peak_memory_stats()
    result = optimize(
        model,
        sampler,
        builder,
        optimizer,
        updates=40,
        batch_size=32,
        max_seconds=min(600, args.deadline - time.time() - 600),
        device=device,
        identity=identity,
        checkpoint_path=root / "disposable.pt",
        checkpoint_every=40,
        warmup=4,
    )
    if result["completed_update"] != 40:
        raise ValueError("Readiness numerical budget did not finish")
    peak = int(torch.cuda.max_memory_allocated())
    labels = benchmark.records("inner", "fit")
    verify_training_labels(labels, mask)
    label_sampler = ConditionSampler(labels, {"sl": 1.0}, seed=918)
    label_builder = BatchBuilder(features, Scales([]), vocab, config)

    def label_batch():
        return {
            k: v.to(device)
            for k, v in label_builder([label_sampler.draw() for _ in range(32)]).items()
        }

    model.train()
    sl_losses = [step(model, optimizer, label_batch()) for _ in range(4)]
    checkpoint = root / "resume-check.pt"
    save_checkpoint(
        checkpoint, model, optimizer, label_sampler, update=44, identity=identity
    )
    batch = label_batch()
    step(model, optimizer, batch)
    expected_weights = weights_digest(model)
    assert (
        resume(
            checkpoint,
            model,
            optimizer,
            label_sampler,
            identity=identity,
            device=device,
        )
        == 44
    )
    resumed_batch = label_batch()
    for key in batch:
        torch.testing.assert_close(batch[key], resumed_batch[key], rtol=0, atol=0)
    step(model, optimizer, resumed_batch)
    if weights_digest(model) != expected_weights:
        raise ValueError("CUDA optimizer/RNG/sampler continuation was not exact")
    model.eval()
    with torch.no_grad():
        expected = model(batch)["sl_logit"].float().cpu()
        context_cache = model.prepare_context(batch)
        cached = model(batch, context_cache)["sl_logit"].float().cpu()
        torch.testing.assert_close(expected, cached, rtol=1e-5, atol=2e-6)
        swapped = {
            k: v[:, [1, 0]] if k.startswith("action_") else v for k, v in batch.items()
        }
        torch.testing.assert_close(
            model(swapped)["sl_logit"].float().cpu(), expected, rtol=1e-5, atol=2e-6
        )
    announce(
        "cuda_checks_passed",
        peak_allocated_bytes=peak,
        seconds_per_update=result["elapsed_seconds"] / 40,
        sl_gradient_updates=4,
        resume_exact=True,
    )
    provenance = {
        **identity,
        "disposable": True,
        "research_checkpoint": False,
        "exposure": view.receipt,
    }
    export(
        root / "export",
        model,
        root / "features",
        vocabulary=vocab,
        basal={},
        provenance=provenance,
    )
    before_transfer = time.monotonic()
    artifact = upload_directory(root / "export", JOB)
    upload_seconds = time.monotonic() - before_transfer
    del optimizer, model
    torch.cuda.empty_cache()
    download_directory(JOB, root / "replay")
    restored, _, _, _, receipt = load(root / "replay", device)
    with torch.no_grad():
        torch.testing.assert_close(
            restored(batch)["sl_logit"].float().cpu(), expected, rtol=0, atol=0
        )
    report = {
        "state": "complete",
        "purpose": "disposable real-data readiness",
        "research_training_launched": False,
        "training_ready": False,
        "coverage_scope": "human combination pack and MuSL inner-fold labels; molecular and nonhuman packs not exercised",
        "model_parameters": sum(p.numel() for p in restored.parameters()),
        "peak_allocated_bytes": peak,
        "quantitative_check": result,
        "sl_gradient_losses": sl_losses,
        "cuda_resume_exact": True,
        "r2_export_replay_exact": True,
        "context_cache_matches": True,
        "action_permutation_matches": True,
        "source_identity": identity,
        "exposure": view.receipt,
        "bundle_bytes": sum(v["bytes"] for v in artifact["files"].values()),
        "upload_seconds": upload_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    put_json(JOB, "complete.json", report)
    announce(
        "readiness_complete",
        elapsed_seconds=report["elapsed_seconds"],
        research_training_launched=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/workspace/slp-r2")
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        report = {
            "state": "failed",
            "type": type(exc).__name__,
            "detail": str(exc)[:500]
            if isinstance(exc, (ValueError, AssertionError))
            else None,
            "frames": [
                {"file": Path(f.filename).name, "line": f.lineno}
                for f in traceback.extract_tb(exc.__traceback__)
            ],
        }
        put_json(JOB, "failed.json", report)
        print(json.dumps(report), flush=True)
        raise SystemExit(1) from None
