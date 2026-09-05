#!/usr/bin/env python3
"""Run the frozen transcript-query feature repair for the Frangieh cell-state AE."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts/run_slp11_frangieh_cell_state.py"
CORE = ROOT / "modules/slp-1-1-cell-state-v1/cell_state.py"
INFERENCE = ROOT / "modules/slp-1-1-cell-state-v1/inference.py"
VERIFIER = ROOT / "scripts/verify_slp11_frangieh_cell_state_artifact.py"
TRANSCRIPT = ROOT / "data/derived/slp11-human-transcript-sequence/ensembl116-kmer4-v1/human-transcript-kmer4-features.npz"
TRANSCRIPT_SHA = "af165a97a0169dd7419e86ebdbc5fc3855dc7b868c7f774b817720d8cf3631d3"
OUTPUT = ROOT / "results/slp11-transition/frangieh-cell-state-transcript-query-ae-latent-ridge-seed731-v3"
PROFILE = ROOT / "results/slp11-transition/frangieh-cell-state-transcript-query-profile-v3.json"
V1_ARTIFACT = ROOT / "results/slp11-transition/frangieh-cell-state-ae-latent-ridge-seed731-v1"
V1_REPORT_SHA = "cada9a66568dda2340a95dd0bbd6b96bcc7af2ac76bf98f9b8f4e1d681bc182f"
V1_PREDICTIONS_SHA = "a5cc6724ad55c5d3f2ad709be36a5fcbcb77e7255d7f79af29145556e2a24b96"
SEED = 731
BATCH = 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def extend_reference(base, reference: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if sha256(TRANSCRIPT) != TRANSCRIPT_SHA:
        raise ValueError("transcript feature pack hash mismatch")
    with np.load(TRANSCRIPT, allow_pickle=False) as archive:
        ids = archive["entity_id"].astype(str)
        taxa = archive["entity_taxon"]
        values = archive["feature_values"].astype(np.float32)
    if values.shape != (85_410, 259) or not np.all(taxa == 9606) or len(np.unique(ids)) != len(ids):
        raise ValueError("transcript feature schema drift")
    lookup = dict(zip(ids, values, strict=True))
    with np.load(base.DEVELOPMENT, allow_pickle=False) as development:
        fitting_genes = np.unique(development["action_ids"][development["split_train"]].astype(str))
    if len(fitting_genes) != 151 or any(gene not in lookup for gene in fitting_genes):
        raise ValueError("fitting action roster drift")
    fitting = np.stack([lookup[gene] for gene in fitting_genes]).astype(np.float64)
    mean = fitting.mean(axis=0)
    scale = fitting.std(axis=0)
    scale[scale < 1e-8] = 1.0
    query_raw = np.stack([lookup[str(gene)] for gene in reference["rna_query_ids"]])
    transcript = np.clip((query_raw - mean) / scale, -10.0, 10.0).astype(np.float32)
    result = dict(reference)
    result["rna_query_features"] = np.concatenate(
        [reference["rna_query_features"], transcript], axis=1
    ).astype(np.float32)
    result["rna_transcript_feature_mean"] = mean.astype(np.float32)
    result["rna_transcript_feature_scale"] = scale.astype(np.float32)
    result["rna_transcript_raw_presence"] = query_raw[:, 258].astype(bool)
    result["rna_original_feature_count"] = np.asarray(1156, dtype=np.int64)
    metadata = {
        "fittingGenes": len(fitting_genes),
        "fittingGeneRosterSha256": hashlib.sha256("\n".join(fitting_genes).encode()).hexdigest(),
        "queryRows": len(query_raw),
        "queryRowsPresent": int(np.count_nonzero(query_raw[:, 258])),
        "queryRowsMissing": int(np.count_nonzero(query_raw[:, 258] == 0)),
        "featureMeanFloat32Sha256": hashlib.sha256(result["rna_transcript_feature_mean"].tobytes()).hexdigest(),
        "featureScaleFloat32Sha256": hashlib.sha256(result["rna_transcript_feature_scale"].tobytes()).hexdigest(),
        "normalization": "population mean/SD on 151 unique fitting action genes; SD<1e-8 replaced1; clip[-10,10]",
    }
    return result, metadata


def initialize_extended(core, device: torch.device):
    """Recreate v1 seed initialization, preserving every old-compatible tensor."""
    torch.manual_seed(SEED)
    old = core.CellState(core.Config(1156, 20, key_dim=64, state_dim=128, hidden_dim=256, dropout=0.1))
    old_state = old.state_dict()
    torch.manual_seed(SEED)
    model = core.CellState(core.Config(1415, 20, key_dim=64, state_dim=128, hidden_dim=256, dropout=0.1))
    with torch.no_grad():
        for name, target in model.state_dict().items():
            source = old_state[name]
            if target.shape == source.shape:
                target.copy_(source)
            elif name in ("keys.rna.0.weight", "observation.rna.0.weight"):
                if target.shape[1] != 1415 or source.shape[1] != 1156:
                    raise ValueError(f"unexpected expanded tensor: {name}")
                target.zero_()
                target[:, :1156].copy_(source)
            else:
                raise ValueError(f"unrecognized extended tensor: {name}")
    return model.to(device), old.to(device)


def initial_parity(model, old, reference: dict[str, np.ndarray], device: torch.device) -> float:
    rng = np.random.default_rng(731)
    rna = torch.as_tensor(
        rng.normal(size=(2, len(reference["rna_query_features"]))).astype(np.float32),
        device=device,
    )
    protein = torch.as_tensor(
        rng.normal(size=(2, len(reference["protein_query_features"]))).astype(np.float32),
        device=device,
    )
    rna_mask = torch.ones_like(rna, dtype=torch.bool)
    protein_mask = torch.ones_like(protein, dtype=torch.bool)
    old_query = torch.as_tensor(reference["rna_query_features"][:, :1156], device=device)
    new_query = torch.as_tensor(reference["rna_query_features"], device=device)
    protein_query = torch.as_tensor(reference["protein_query_features"], device=device)
    model.eval(); old.eval()
    with torch.no_grad():
        old_state = old.encode(old_query, rna, rna_mask, protein_query, protein, protein_mask)
        new_state = model.encode(new_query, rna, rna_mask, protein_query, protein, protein_mask)
        differences = [torch.max(torch.abs(old_state - new_state))]
        for head, old_features, new_features in (
            ("rna", old_query, new_query), ("protein", protein_query, protein_query)
        ):
            differences.append(torch.max(torch.abs(
                old.observe(old_state, old_features, head)
                - model.observe(new_state, new_features, head)
            )))
    maximum = float(torch.max(torch.stack(differences)).cpu())
    if maximum > 1e-6:
        raise ValueError(f"extended initialization parity exceeds tolerance: {maximum}")
    return maximum


def profile(base, device_name: str, steps: int) -> dict[str, object]:
    manifest, original = base.verify_inputs()
    reference, metadata = extend_reference(base, original)
    core = load_source(CORE, "transcript_profile_core")
    device = torch.device(device_name)
    torch.use_deterministic_algorithms(True)
    model, old = initialize_extended(core, device)
    parity = initial_parity(model, old, reference, device)
    del old
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    rf = torch.as_tensor(reference["rna_query_features"], device=device)
    pf = torch.as_tensor(reference["protein_query_features"], device=device)
    rv = torch.zeros((BATCH, 18_063), device=device)
    pv = torch.zeros((BATCH, 20), device=device)
    torch.cuda.synchronize() if device.type == "cuda" else None
    started = time.monotonic()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        state = model.encode(
            rf, rv, base.denoising_mask(tuple(rv.shape), step, device),
            pf, pv, base.denoising_mask(tuple(pv.shape), step + 10_000, device),
        )
        loss = core.balanced_reconstruction_loss(
            model.observe(state, rf, "rna"), rv, torch.ones_like(rv, dtype=torch.bool),
            model.observe(state, pf, "protein"), pv, torch.ones_like(pv, dtype=torch.bool),
        )
        loss.backward(); optimizer.step()
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.monotonic() - started
    steps_per_epoch = 0
    for item in manifest["shards"]:
        shard = base.load_shard(base.SHARDS / item["path"])
        steps_per_epoch += math.ceil(int(np.sum(shard["reconstruction_split"] == "train")) / BATCH)
    return {
        "steps": steps, "seconds": elapsed, "secondsPerStep": elapsed / steps,
        "stepsPerEpoch": steps_per_epoch,
        "projected20EpochComputeSeconds": elapsed / steps * steps_per_epoch * 20,
        "batch": BATCH, "rnaQueries": 18_063, "rnaFeatures": 1415,
        "initialV1ParityMaxAbsoluteDifference": parity,
        "transcript": metadata, "targetFreeSynthetic": True,
    }


def prepare(base, output: Path, profile_path: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    manifest, original = base.verify_inputs()
    _reference, transcript = extend_reference(base, original)
    profile_report = json.loads(profile_path.read_text())
    if profile_report["projected20EpochComputeSeconds"] >= 900:
        raise ValueError("extended profile exceeds prospective compute allowance")
    output.mkdir(parents=True)
    source = output / "source"; source.mkdir()
    sources = {
        "cell_state.py": CORE, "inference.py": INFERENCE, "verify.py": VERIFIER,
        "trainer.py": Path(__file__), "base_trainer.py": BASE,
        "frangieh_basal_ridge.py": ROOT / "modules/slp-1-1-world-transition-v1/frangieh_basal_ridge.py",
    }
    source_hashes = {}
    for name, path in sources.items():
        shutil.copy2(path, source / name); source_hashes[name] = sha256(source / name)
    protocol = {
        "schema": "slp.frangieh-cell-state-transcript-query-protocol/v3",
        "hypothesis": "Adding static transcript-sequence descriptors to RNA query keys and observations repairs RNA reconstruction and improves held-gene forecasts.",
        "changeFromV1": "RNA query features append normalized transcript259 to existing frozen1156 for encoder keys and affine observation; action features remain physical1156.",
        "initialization": "recreate seed731 v1 initialization; copy every old-compatible tensor bit-exact; expanded RNA first-layer old1156 columns copied and new259 columns zero-initialized but trainable; forward parity tolerance1e-6 allows GEMM shape rounding",
        "transcript": {**transcript, "path": str(TRANSCRIPT), "sha256": TRANSCRIPT_SHA},
        "training": "identical v1 seed731,B256,AdamWlr.0005/decay.01,20% deterministic denoising,equal-modality standardized MSE,max20epochs,validation every2,patience4",
        "reconstructionGate": "RNA and protein each >=5% standardized MSE gain versus reconstruction-training mean",
        "forecastGate": "all 3 contexts x2 heads: >=1% raw MSE gain versus mean,base577,physical1156,prior paired world; r>=.10 and nonregression; v1 descriptive",
        "profile": profile_report, "population": manifest["counts"],
        "inputs": {
            "singleCellManifestSha256": base.HASHES["manifest"],
            "referenceSha256": base.HASHES["reference"],
            "staticSha256": base.HASHES["static"],
            "developmentSha256": base.HASHES["development"],
            "baselineSha256": base.HASHES["baseline"],
            "cellStateV1ReportSha256": V1_REPORT_SHA,
            "cellStateV1PredictionsSha256": V1_PREDICTIONS_SHA,
        },
        "sourceHashes": source_hashes,
        "executionSourceContract": "repository base helper must equal frozen source/base_trainer.py SHA; repository root semantics supply pinned input paths",
        "supersedesPreparedOnly": {"path": "results/slp11-transition/frangieh-cell-state-transcript-query-ae-latent-ridge-seed731-v2", "reason": "copied base helper resolved repository input paths relative to artifact; stopped before shard matrix access or fitting"},
        "limitations": ["single seed", "one query-feature ablation", "held-gene forecast is adaptive development", "no learned gene identity", "the latent ridge plus affine observation remains a low-rank linear action forecast"],
        "testAccessed": False, "benchmarkAccessed": False, "jurkatAccessed": False,
    }
    base.write_json(output / "protocol.json", protocol)
    prepared = {"protocolSha256": sha256(output / "protocol.json"), "sourceHashes": source_hashes}
    base.write_json(output / "PREPARED.json", prepared)
    return prepared


def add_v1_comparison(base, output: Path, predictions: dict[str, np.ndarray], results: dict) -> None:
    if sha256(V1_ARTIFACT / "report.json") != V1_REPORT_SHA:
        raise ValueError("cell-state v1 report drift")
    if sha256(V1_ARTIFACT / "predictions.npz") != V1_PREDICTIONS_SHA:
        raise ValueError("cell-state v1 predictions drift")
    with np.load(V1_ARTIFACT / "predictions.npz", allow_pickle=False) as archive:
        old = {name: archive[name] for name in archive.files}
    for key in ("action_ids", "context_ids", "rna_truth", "protein_truth"):
        if not np.array_equal(predictions[key], old[key]):
            raise ValueError(f"cell-state v1 comparison alignment drift: {key}")
    helper = load_source(output / "source/frangieh_basal_ridge.py", "transcript_v1_metrics")
    for context in base.CONTEXTS:
        rows = predictions["context_ids"] == context
        for head in ("rna", "protein"):
            results[context]["heads"][head]["cellStateV1Descriptive"] = helper.metrics(
                old[f"{head}_prediction"][rows], old[f"{head}_truth"][rows],
                np.ones(old[f"{head}_truth"].shape[1], dtype=np.float32),
            )


def train(base, manifest, reference, stats, device, output):
    core = load_source(output / "source/cell_state.py", "transcript_training_core")
    model, old = initialize_extended(core, device)
    parity = initial_parity(model, old, reference, device); del old
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    rf = torch.as_tensor(reference["rna_query_features"], device=device)
    pf = torch.as_tensor(reference["protein_query_features"], device=device)
    started = time.monotonic(); best = None; best_score = float("inf"); stale = 0; history = []
    for epoch in range(1, 21):
        model.train(); losses = []
        shard_order = np.random.default_rng(SEED + epoch).permutation(len(manifest["shards"]))
        for shard_index in shard_order:
            item = manifest["shards"][int(shard_index)]; shard = base.load_shard(base.SHARDS / item["path"])
            rows = np.flatnonzero(shard["reconstruction_split"] == "train")
            rows = np.random.default_rng(SEED * 1000 + epoch * 53 + int(shard_index)).permutation(rows)
            for offset in range(0, len(rows), BATCH):
                selection = rows[offset : offset + BATCH]
                _, _, rs, ps = base.batch_arrays(shard, selection, stats, device)
                token = epoch * 1_000_000 + item["row_start"] + offset
                optimizer.zero_grad(set_to_none=True)
                state = model.encode(
                    rf, rs, base.denoising_mask(tuple(rs.shape), token, device),
                    pf, ps, base.denoising_mask(tuple(ps.shape), token + 500_000_000, device),
                )
                loss = core.balanced_reconstruction_loss(
                    model.observe(state, rf, "rna"), rs, torch.ones_like(rs, dtype=torch.bool),
                    model.observe(state, pf, "protein"), ps, torch.ones_like(ps, dtype=torch.bool),
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite reconstruction loss")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                losses.append(float(loss.detach()))
        event = {"epoch": epoch, "trainLoss": float(np.mean(losses))}
        if epoch % 2 == 0:
            validation = base.evaluate_reconstruction(model, core, manifest, reference, stats, device)
            event["validation"] = validation; score = validation["balancedStandardizedMse"]
            if score < best_score:
                best_score = score; stale = 0
                best = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                best_epoch = epoch
            else:
                stale += 1
            print(json.dumps({"event": "transcript_reconstruction", **event}), flush=True)
        history.append(event)
        if stale >= 4 or time.monotonic() - started > 1100:
            break
    if best is None:
        raise RuntimeError("no complete checkpoint")
    model.load_state_dict(best)
    final = base.evaluate_reconstruction(model, core, manifest, reference, stats, device)
    from safetensors.torch import save_file
    save_file(best, str(output / "model.safetensors"))
    return model, {
        "bestEpoch": best_epoch, "history": history, "validation": final,
        "seconds": time.monotonic() - started, "initialV1ParityMaxAbsoluteDifference": parity,
    }


def execute(output: Path, device_name: str) -> dict[str, object]:
    started = time.monotonic()
    prepared = json.loads((output / "PREPARED.json").read_text())
    if sha256(output / "protocol.json") != prepared["protocolSha256"]:
        raise ValueError("protocol drift")
    for name, digest in prepared["sourceHashes"].items():
        if sha256(output / "source" / name) != digest:
            raise ValueError(f"source drift: {name}")
    if sha256(Path(__file__)) != prepared["sourceHashes"]["trainer.py"]:
        raise ValueError("executing trainer differs from frozen source")
    if sha256(BASE) != prepared["sourceHashes"]["base_trainer.py"]:
        raise ValueError("repository base helper differs from frozen copy")
    base = load_source(BASE, "checksum_matched_transcript_base")
    manifest, original = base.verify_inputs(); reference, _ = extend_reference(base, original)
    stats = base.streaming_stats(manifest); device = torch.device(device_name)
    torch.use_deterministic_algorithms(True)
    model, train_report = train(base, manifest, reference, stats, device, output)
    genes, controls = base.encode_populations(model, manifest, reference, stats, device)
    coef, intercept, _, normalized, counts = base.fit_latent_ridges(genes, controls, reference)
    train_report["latentFittingGenesByContext"] = counts
    predictions, results, decision = base.forecast_and_score(
        model, reference, stats, coef, intercept, normalized, device, output
    )
    add_v1_comparison(base, output, predictions, results)
    report = base.save_artifact(
        output, model, reference, stats, coef, intercept, predictions,
        train_report, results, decision, started,
    )
    report["variant"] = {
        "rnaQueryFeatures": 1415, "oldFeatures": 1156, "newTranscriptFeatures": 259,
        "initialV1ParityMaxAbsoluteDifference": train_report["initialV1ParityMaxAbsoluteDifference"],
    }
    base.write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--profile", action="store_true")
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--profile-path", type=Path, default=PROFILE)
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    base = load_source(BASE, "transcript_base")
    if args.profile:
        result = profile(base, args.device, args.profile_steps); base.write_json(args.profile_path, result)
    elif args.prepare_only:
        result = prepare(base, args.output, args.profile_path)
    else:
        result = execute(args.output, args.device)
    print(json.dumps(result, sort_keys=True, default=lambda value: value.item()))


if __name__ == "__main__":
    main()
