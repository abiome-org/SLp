#!/usr/bin/env python3
"""Fit a paired-cell denoising state and a frozen held-gene latent ridge forecast."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from scipy import sparse
from sklearn.linear_model import Ridge

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "modules/slp-1-1-cell-state-v1/cell_state.py"
INFERENCE = ROOT / "modules/slp-1-1-cell-state-v1/inference.py"
VERIFIER = ROOT / "scripts/verify_slp11_frangieh_cell_state_artifact.py"
SHARDS = ROOT / "data/derived/slp11-frangieh/paired-singlecell-train-control-v1"
REFERENCE = ROOT / "results/slp11-transition/frangieh-paired-state-physical1156-seed731-v2/reference.npz"
STATIC = ROOT / "data/derived/slp11-frangieh-static/ensembl116-goa2022-fixed-neighbor-v1/frangieh-extended-static-esm-go-fixed-physical-features.npz"
DEVELOPMENT = ROOT / "data/derived/slp11-frangieh/paired-development-v1/development.npz"
BASELINE = ROOT / "results/slp11-transition/frangieh-paired-state-vs-static-scoring-v1/report.json"
STATIC_PREDICTIONS = ROOT / "results/slp11-transition/frangieh-specieswide-physical-ridge-v1/predictions.npz"
PRIOR_PREDICTIONS = ROOT / "results/slp11-transition/frangieh-paired-state-physical1156-seed731-v2/predictions.npz"
OUTPUT = ROOT / "results/slp11-transition/frangieh-cell-state-ae-latent-ridge-seed731-v1"
HASHES = {
    "manifest": "e791b5cf35da96fa71951a4a240ed58b53e278d3c57e44066680abd3f386a9c7",
    "reference": "8b82e4781b73a721f995dd218ef341ea8324b87d3c9189bfe40644d436800e73",
    "static": "347fd1bf87d8fc3d0b447676082b4bcb64f021c9f12c7df4d1754dc262b2bf72",
    "development": "4bbb1eec9ede66211f1316b2841bb0037032ef975cd6c92d34aba0adb5fed744",
    "baseline": "d0c577e093198e9060a582cc5852b0db61246daa5772ae0c1e8451addc584b90",
}
CONTEXTS = ("Co-culture", "Control", "IFNγ")
SEED = 731
BATCH = 256


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    def clean(item: object) -> object:
        if isinstance(item, dict):
            return {str(key): clean(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(entry) for entry in item]
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_shard(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    matrix = sparse.csr_matrix(
        (arrays["rna_data"], arrays["rna_indices"], arrays["rna_indptr"]),
        shape=tuple(arrays["rna_shape"]),
    )
    arrays["rna"] = matrix
    return arrays


def verify_inputs() -> tuple[dict, dict[str, np.ndarray]]:
    if sha256(SHARDS / "manifest.json") != HASHES["manifest"]:
        raise ValueError("single-cell manifest hash mismatch")
    for name, path in (("reference", REFERENCE), ("static", STATIC), ("development", DEVELOPMENT), ("baseline", BASELINE)):
        if sha256(path) != HASHES[name]:
            raise ValueError(f"{name} hash mismatch")
    manifest = json.loads((SHARDS / "manifest.json").read_text())
    if manifest["counts"]["cells"] != 103_862 or manifest["counts"]["excluded_validation_cells"] != 19_606:
        raise ValueError("single-cell access population drift")
    expected_start = 0
    for item in manifest["shards"]:
        path = SHARDS / item["path"]
        if item["row_start"] != expected_start or sha256(path) != item["sha256"]:
            raise ValueError(f"shard order/hash drift: {path.name}")
        expected_start = item["row_stop"]
    with np.load(REFERENCE, allow_pickle=False) as archive:
        reference = {name: archive[name] for name in archive.files}
    if reference["rna_query_features"].shape != (18_063, 1_156) or reference["protein_query_features"].shape != (20, 20):
        raise ValueError("frozen query feature shape drift")
    return manifest, reference


def streaming_stats(manifest: dict) -> dict[str, np.ndarray | int]:
    rna_sum = np.zeros(18_063, dtype=np.float64)
    rna_square = np.zeros(18_063, dtype=np.float64)
    protein_sum = np.zeros(20, dtype=np.float64)
    protein_square = np.zeros(20, dtype=np.float64)
    count = 0
    for item in manifest["shards"]:
        shard = load_shard(SHARDS / item["path"])
        rows = shard["reconstruction_split"] == "train"
        matrix = shard["rna"][rows].astype(np.float64)
        rna_sum += np.asarray(matrix.sum(axis=0)).ravel()
        rna_square += np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel()
        protein = shard["protein_values"][rows].astype(np.float64)
        protein_sum += protein.sum(axis=0)
        protein_square += np.square(protein).sum(axis=0)
        count += int(rows.sum())
    if count != 93_397:
        raise ValueError("reconstruction fitting count drift")
    result = {"count": count}
    for head, total, square in (("rna", rna_sum, rna_square), ("protein", protein_sum, protein_square)):
        mean = total / count
        variance = np.maximum(square / count - np.square(mean), 0.0)
        result[f"{head}_mean"] = mean.astype(np.float32)
        result[f"{head}_sd"] = np.maximum(np.sqrt(variance), 0.05).astype(np.float32)
    return result


def aggregate_guide_states(
    action: np.ndarray, context: np.ndarray, guide: np.ndarray, state: np.ndarray
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray]]:
    """Equal cells within guide, then equal guides within gene/context; controls by cell."""
    guide_sum, guide_count = {}, defaultdict(int)
    control_sum, control_count = {}, defaultdict(int)
    for gene, condition, construct, vector in zip(action, context, guide, state, strict=True):
        if gene == "":
            control_sum[condition] = control_sum.get(condition, np.zeros(state.shape[1])) + vector
            control_count[condition] += 1
        else:
            key = (gene, condition, construct)
            guide_sum[key] = guide_sum.get(key, np.zeros(state.shape[1])) + vector
            guide_count[key] += 1
    guide_means = {key: value / guide_count[key] for key, value in guide_sum.items()}
    gene_guides = defaultdict(list)
    for (gene, condition, _), value in guide_means.items():
        gene_guides[(gene, condition)].append(value)
    genes = {key: np.mean(values, axis=0).astype(np.float32) for key, values in gene_guides.items()}
    controls = {condition: (control_sum[condition] / control_count[condition]).astype(np.float32) for condition in control_sum}
    return genes, controls


def denoising_mask(shape: tuple[int, int], token: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(SEED * 1_000_003 + token)
    return torch.rand(shape, generator=generator, device=device) >= 0.20


def profile(device_name: str, steps: int) -> dict[str, object]:
    manifest, reference = verify_inputs()
    core = load_source(CORE, "cell_state_profile")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    model = core.CellState(core.Config(1156, 20, key_dim=64, state_dim=128, hidden_dim=256, dropout=0.1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    rf = torch.as_tensor(reference["rna_query_features"], device=device)
    pf = torch.as_tensor(reference["protein_query_features"], device=device)
    rv = torch.zeros((BATCH, 18_063), device=device)
    pv = torch.zeros((BATCH, 20), device=device)
    ro = torch.ones_like(rv, dtype=torch.bool)
    po = torch.ones_like(pv, dtype=torch.bool)
    torch.cuda.synchronize() if device.type == "cuda" else None
    started = time.monotonic()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        state = model.encode(rf, rv, denoising_mask(rv.shape, step, device), pf, pv, denoising_mask(pv.shape, step + 10_000, device))
        loss = core.balanced_reconstruction_loss(model.observe(state, rf, "rna"), rv, ro, model.observe(state, pf, "protein"), pv, po)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.monotonic() - started
    steps_per_epoch = 0
    for item in manifest["shards"]:
        shard = load_shard(SHARDS / item["path"])
        fitting_rows = int(np.sum(shard["reconstruction_split"] == "train"))
        steps_per_epoch += math.ceil(fitting_rows / BATCH)
    return {"steps": steps, "seconds": elapsed, "secondsPerStep": elapsed / steps, "stepsPerEpoch": steps_per_epoch, "projected20EpochComputeSeconds": elapsed / steps * steps_per_epoch * 20, "batch": BATCH, "rnaQueries": 18_063, "targetFreeSynthetic": True}


def prepare(output: Path, profile_path: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    manifest, reference = verify_inputs()
    profile_report = json.loads(profile_path.read_text())
    if profile_report["projected20EpochComputeSeconds"] >= 900:
        raise ValueError("profile exceeds prospective full-epoch compute allowance")
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    sources = {
        "cell_state.py": CORE,
        "inference.py": INFERENCE,
        "verify.py": VERIFIER,
        "trainer.py": Path(__file__),
        "frangieh_basal_ridge.py": ROOT / "modules/slp-1-1-world-transition-v1/frangieh_basal_ridge.py",
    }
    source_hashes = {}
    for name, path in sources.items():
        shutil.copy2(path, source / name)
        source_hashes[name] = sha256(source / name)
    inputs = {"manifest": {"path": str(SHARDS / "manifest.json"), "sha256": HASHES["manifest"]}, "reference": {"path": str(REFERENCE), "sha256": HASHES["reference"]}, "static": {"path": str(STATIC), "sha256": HASHES["static"]}, "development": {"path": str(DEVELOPMENT), "sha256": HASHES["development"]}, "baseline": {"path": str(BASELINE), "sha256": HASHES["baseline"]}, "staticPredictions": {"path": str(STATIC_PREDICTIONS), "sha256": sha256(STATIC_PREDICTIONS)}, "priorPredictions": {"path": str(PRIOR_PREDICTIONS), "sha256": sha256(PRIOR_PREDICTIONS)}}
    with np.load(STATIC, allow_pickle=False) as static:
        raw_static = dict(zip(static["entity_id"].astype(str), static["feature_values"], strict=True))
    query_rows = np.stack([raw_static[str(query)] for query in reference["rna_query_ids"]])
    query_coverage = {
        "rnaQueries": len(query_rows),
        "proteinPresentColumn320": int(np.count_nonzero(query_rows[:, 320])),
        "allZeroStaticRows": int(np.count_nonzero(np.all(query_rows == 0, axis=1))),
        "exactDistinctNormalizedRows": int(np.unique(reference["rna_query_features"], axis=0).shape[0]),
        "selectionApplied": False,
    }
    protocol = {
        "schema": "slp.frangieh-cell-state-ae-latent-ridge-protocol/v1",
        "hypothesis": "A denoising paired-cell state supports held-gene latent-delta forecasts beyond endpoint baselines.",
        "reconstruction": {"config": {"key": 64, "state": 128, "hidden": 256, "dropout": 0.1}, "normalization": "global per-query mean and population SD floor0.05 on reconstruction-train cells only", "denoising": "independent deterministic 20% channel input masks; complete observed targets", "objective": "equal RNA/protein standardized cell MSE", "seed": 731, "batch": 256, "optimizer": "AdamW lr.0005 weight_decay.01", "maxEpochs": 20, "validationEveryEpochs": 2, "patienceEvaluations": 4, "selection": "minimum reconstruction-validation balanced standardized MSE", "gate": "each RNA/protein standardized validation MSE at least5% below the reconstruction-training mean predictor; raw-unit MSE descriptive", "maxTrainingSeconds": 1200},
        "forecast": {"timing": "only after encoder checkpoint frozen", "stateAggregation": "equal cells within exact guide group, then equal guides per gene/context; controls equal all verified NT cells", "transition": "context-local Ridge alpha10000 from frozen normalized physical1156 action features to mean latent delta", "decode": "frozen affine observation basis; supplied measured context control means; AE train-cell SD", "selection": "none on held genes", "gate": "each of 3contexts x2heads: >=1% raw MSE gain versus mean,base577,physical1156,prior paired world; r>=.10 and nonregression against each defined baseline r"},
        "population": manifest["counts"], "queryFeatureCoverage": query_coverage,
        "profile": profile_report, "inputs": inputs, "sourceHashes": source_hashes,
        "limitations": ["reconstruction validation holds out cells, not intervention genes", "latent state is not biologically identified", "the affine decoder plus context-local latent ridge is a low-rank linear action forecast with a learned output prior", "held-gene forecast is one adaptive development evaluation", "cells are not before/after pairs", "all 18063 RNA queries remain fixed; 1849 raw all-zero static rows collapse to one normalized row and no learned gene identity is added"],
        "jurkatAccessed": False, "testAccessed": False, "benchmarkAccessed": False,
    }
    write_json(output / "protocol.json", protocol)
    prepared = {"protocolSha256": sha256(output / "protocol.json"), "sourceHashes": source_hashes, "inputs": inputs, "profile": profile_report}
    write_json(output / "PREPARED.json", prepared)
    return prepared


def batch_arrays(shard: dict, rows: np.ndarray, stats: dict, device: torch.device):
    rna = shard["rna"][rows].toarray().astype(np.float32)
    protein = shard["protein_values"][rows].astype(np.float32)
    rs = (rna - stats["rna_mean"]) / stats["rna_sd"]
    ps = (protein - stats["protein_mean"]) / stats["protein_sd"]
    return rna, protein, torch.as_tensor(rs, device=device), torch.as_tensor(ps, device=device)


def evaluate_reconstruction(model, core, manifest, reference, stats, device):
    model.eval()
    totals = {name: 0.0 for name in ("rnaStd", "proteinStd", "rnaRaw", "proteinRaw", "rnaMeanStd", "proteinMeanStd", "rnaMeanRaw", "proteinMeanRaw")}
    cells = 0
    rf = torch.as_tensor(reference["rna_query_features"], device=device)
    pf = torch.as_tensor(reference["protein_query_features"], device=device)
    with torch.no_grad():
        for shard_index, item in enumerate(manifest["shards"]):
            shard = load_shard(SHARDS / item["path"])
            selected = np.flatnonzero(shard["reconstruction_split"] == "validation")
            for offset in range(0, len(selected), BATCH):
                rows = selected[offset:offset + BATCH]
                rna, protein, rs, ps = batch_arrays(shard, rows, stats, device)
                rm = denoising_mask(tuple(rs.shape), item["row_start"] + offset, device)
                pm = denoising_mask(tuple(ps.shape), 10_000_000 + item["row_start"] + offset, device)
                state = model.encode(rf, rs, rm, pf, ps, pm)
                rp, pp = model.observe(state, rf, "rna"), model.observe(state, pf, "protein")
                totals["rnaStd"] += float(torch.square(rp - rs).sum().cpu())
                totals["proteinStd"] += float(torch.square(pp - ps).sum().cpu())
                totals["rnaMeanStd"] += float(torch.square(rs).sum().cpu())
                totals["proteinMeanStd"] += float(torch.square(ps).sum().cpu())
                rp_raw = rp.cpu().numpy() * stats["rna_sd"] + stats["rna_mean"]
                pp_raw = pp.cpu().numpy() * stats["protein_sd"] + stats["protein_mean"]
                totals["rnaRaw"] += float(np.square(rp_raw - rna).sum())
                totals["proteinRaw"] += float(np.square(pp_raw - protein).sum())
                totals["rnaMeanRaw"] += float(np.square(rna - stats["rna_mean"]).sum())
                totals["proteinMeanRaw"] += float(np.square(protein - stats["protein_mean"]).sum())
                cells += len(rows)
    report = {}
    for head, q in (("rna", 18_063), ("protein", 20)):
        model_raw = totals[f"{head}Raw"] / (cells * q)
        mean_raw = totals[f"{head}MeanRaw"] / (cells * q)
        model_standard = totals[f"{head}Std"] / (cells * q)
        mean_standard = totals[f"{head}MeanStd"] / (cells * q)
        report[head] = {"standardizedMse": model_standard, "trainingMeanStandardizedMse": mean_standard, "standardizedFractionalImprovement": 1 - model_standard / mean_standard, "rawMse": model_raw, "trainingMeanRawMse": mean_raw, "rawFractionalImprovement": 1 - model_raw / mean_raw, "gatePassed": 1 - model_standard / mean_standard >= 0.05}
    report["balancedStandardizedMse"] = 0.5 * (report["rna"]["standardizedMse"] + report["protein"]["standardizedMse"])
    report["cells"] = cells
    return report


def train(manifest, reference, stats, device, output):
    core = load_source(output / "source/cell_state.py", "cell_state_training")
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    model = core.CellState(core.Config(1156, 20, key_dim=64, state_dim=128, hidden_dim=256, dropout=0.1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    rf = torch.as_tensor(reference["rna_query_features"], device=device)
    pf = torch.as_tensor(reference["protein_query_features"], device=device)
    started = time.monotonic(); best = None; best_score = float("inf"); stale = 0; history = []
    for epoch in range(1, 21):
        model.train(); losses = []
        shard_order = np.random.default_rng(SEED + epoch).permutation(len(manifest["shards"]))
        for shard_index in shard_order:
            item = manifest["shards"][int(shard_index)]; shard = load_shard(SHARDS / item["path"])
            rows = np.flatnonzero(shard["reconstruction_split"] == "train")
            rows = np.random.default_rng(SEED * 1000 + epoch * 53 + int(shard_index)).permutation(rows)
            for offset in range(0, len(rows), BATCH):
                selection = rows[offset:offset + BATCH]
                _, _, rs, ps = batch_arrays(shard, selection, stats, device)
                token = epoch * 1_000_000 + item["row_start"] + offset
                optimizer.zero_grad(set_to_none=True)
                state = model.encode(rf, rs, denoising_mask(tuple(rs.shape), token, device), pf, ps, denoising_mask(tuple(ps.shape), token + 500_000_000, device))
                loss = core.balanced_reconstruction_loss(model.observe(state, rf, "rna"), rs, torch.ones_like(rs, dtype=torch.bool), model.observe(state, pf, "protein"), ps, torch.ones_like(ps, dtype=torch.bool))
                if not torch.isfinite(loss): raise FloatingPointError("nonfinite reconstruction loss")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(float(loss.detach()))
        event = {"epoch": epoch, "trainLoss": float(np.mean(losses))}
        if epoch % 2 == 0:
            validation = evaluate_reconstruction(model, core, manifest, reference, stats, device); event["validation"] = validation
            score = validation["balancedStandardizedMse"]
            if score < best_score:
                best_score = score; stale = 0; best = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}; best_epoch = epoch
            else: stale += 1
            print(json.dumps({"event": "reconstruction", **event}), flush=True)
        history.append(event)
        if stale >= 4 or time.monotonic() - started > 1100: break
    if best is None: raise RuntimeError("no complete reconstruction checkpoint")
    model.load_state_dict(best)
    final = evaluate_reconstruction(model, core, manifest, reference, stats, device)
    save_file(best, str(output / "model.safetensors"))
    return model, core, {"bestEpoch": best_epoch, "history": history, "validation": final, "seconds": time.monotonic() - started}


def encode_populations(model, manifest, reference, stats, device):
    rf = torch.as_tensor(reference["rna_query_features"], device=device); pf = torch.as_tensor(reference["protein_query_features"], device=device)
    actions=[]; contexts=[]; guides=[]; states=[]
    model.eval()
    with torch.no_grad():
        for item in manifest["shards"]:
            shard=load_shard(SHARDS/item["path"])
            for offset in range(0,len(shard["action_ids"]),BATCH):
                rows=np.arange(offset,min(offset+BATCH,len(shard["action_ids"])))
                _,_,rs,ps=batch_arrays(shard,rows,stats,device)
                state=model.encode(rf,rs,torch.ones_like(rs,dtype=torch.bool),pf,ps,torch.ones_like(ps,dtype=torch.bool)).cpu().numpy()
                actions.extend(shard["action_ids"][rows]); contexts.extend(shard["context_ids"][rows]); guides.extend(shard["target_guide_sets"][rows]); states.append(state)
    return aggregate_guide_states(np.asarray(actions),np.asarray(contexts),np.asarray(guides),np.concatenate(states))


def fit_latent_ridges(gene_states, controls, reference):
    with np.load(STATIC,allow_pickle=False) as z:
        ids=z["entity_id"].astype(str); values=z["feature_values"].astype(np.float32)
    lookup=dict(zip(ids,values)); coef=[]; intercept=[]; counts={}
    normalized=lambda genes: np.clip((np.stack([lookup[g] for g in genes])-reference["feature_mean"])/reference["feature_scale"],-float(reference["feature_clip"]),float(reference["feature_clip"])).astype(np.float32)
    for context in CONTEXTS:
        genes=sorted(g for g,c in gene_states if c==context); x=normalized(genes); y=np.stack([gene_states[(g,context)]-controls[context] for g in genes])
        ridge=Ridge(alpha=10000.0,fit_intercept=True,solver="cholesky").fit(x,y); coef.append(ridge.coef_.astype(np.float32)); intercept.append(ridge.intercept_.astype(np.float32)); counts[context]=len(genes)
    return np.stack(coef),np.stack(intercept),lookup,normalized,counts


def forecast_and_score(model, reference, stats, coef, intercept, normalized, device, output):
    # Held-gene quantitative targets are opened only after encoder and latent ridges freeze.
    with np.load(DEVELOPMENT,allow_pickle=False) as z: data={k:z[k] for k in z.files}
    helper=load_source(output/"source/frangieh_basal_ridge.py","cell_state_metrics")
    action,context,rna,_=helper.collapse_gene_profiles(data["action_ids"][data["split_validation"]],data["context_ids"][data["split_validation"]],data["rna_targets"][data["split_validation"]])
    pa,pc,protein,_=helper.collapse_gene_profiles(data["action_ids"][data["split_validation"]],data["context_ids"][data["split_validation"]],data["protein_targets"][data["split_validation"]])
    if not np.array_equal(action,pa) or not np.array_equal(context,pc): raise ValueError("held head identity drift")
    context_index=np.asarray([CONTEXTS.index(c) for c in context]); x=normalized(action); delta=np.empty((len(action),128),np.float32)
    for i in range(3):
        rows=context_index==i; delta[rows]=x[rows]@coef[i].T+intercept[i]
    model.eval(); forecasts={}
    with torch.no_grad():
        td=torch.as_tensor(delta,device=device)
        for head,truth in (("rna",rna),("protein",protein)):
            q=torch.as_tensor(reference[f"{head}_query_features"],device=device); controls=torch.as_tensor(reference[f"{head}_controls"][context_index],device=device); scale=torch.as_tensor(stats[f"{head}_sd"],device=device)
            forecasts[head]=model.observe_delta(td,q,head,controls,scale).cpu().numpy()
    with np.load(STATIC_PREDICTIONS,allow_pickle=False) as static_pred, np.load(PRIOR_PREDICTIONS,allow_pickle=False) as prior:
        prior_lookup={(str(g),CONTEXTS[int(c)]):i for i,(g,c) in enumerate(zip(prior["action_ids"],prior["context_index"],strict=True))}
        baseline_report=json.loads(BASELINE.read_text()); results={}; all_pass=True
        for ci,name in enumerate(CONTEXTS):
            key=name.replace("-","_").replace("γ","gamma"); rows=context==name; genes=action[rows]; results[name]={"heads":{}}
            for head,truth in (("rna",rna),("protein",protein)):
                static_head="adt" if head=="protein" else "rna"; truth_local=truth[rows]
                if not np.array_equal(genes,static_pred[f"{key}_{static_head}_action_ids"]) or not np.array_equal(truth_local,static_pred[f"{key}_{static_head}_truth"]): raise ValueError("static baseline alignment drift")
                prior_rows=[prior_lookup[(str(g),name)] for g in genes]
                if not np.array_equal(truth_local,prior[f"{head}_truth"][prior_rows]): raise ValueError("prior baseline truth drift")
                world=helper.metrics(forecasts[head][rows],truth_local,stats[f"{head}_sd"])
                old=baseline_report["contexts"][name]["heads"][head]
                baselines={**old["baselines"],"priorPairedWorld":old["world"]}; checks={}
                for label,b in baselines.items():
                    br=b["query_centroid_adjusted_profile_pearson"]
                    checks[label]={"mseImprovement":1-world["raw_mse"]/b["raw_mse"],"mseAtLeastOnePercent":1-world["raw_mse"]/b["raw_mse"]>=.01,"rNonregression":br is None or world["query_centroid_adjusted_profile_pearson"]>=br}
                passed=np.isfinite(world["query_centroid_adjusted_profile_pearson"]) and world["query_centroid_adjusted_profile_pearson"]>=.10 and all(v["mseAtLeastOnePercent"] and v["rNonregression"] for v in checks.values()); all_pass &= bool(passed)
                results[name]["heads"][head]={"world":world,"baselines":baselines,"checks":checks,"passed":bool(passed)}
    return {"action_ids":action,"context_ids":context,"rna_truth":rna,"protein_truth":protein,"rna_prediction":forecasts["rna"],"protein_prediction":forecasts["protein"]},results,bool(all_pass)


def save_artifact(output,model,reference,stats,coef,intercept,predictions,train_report,results,decision,started):
    arrays={"rna_query_features":reference["rna_query_features"],"protein_query_features":reference["protein_query_features"],"rna_query_ids":reference["rna_query_ids"],"protein_query_ids":reference["protein_query_ids"],"rna_controls":reference["rna_controls"],"protein_controls":reference["protein_controls"],"rna_mean":stats["rna_mean"],"rna_sd":stats["rna_sd"],"protein_mean":stats["protein_mean"],"protein_sd":stats["protein_sd"],"ridge_coef":coef,"ridge_intercept":intercept,"feature_mean":reference["feature_mean"],"feature_scale":reference["feature_scale"],"feature_clip":reference["feature_clip"],"context_names":np.asarray(CONTEXTS),"key_dim":np.asarray(64),"state_dim":np.asarray(128),"hidden_dim":np.asarray(256),"dropout":np.asarray(.1,np.float32)}
    for name in (
        "rna_transcript_feature_mean", "rna_transcript_feature_scale",
        "rna_transcript_raw_presence", "rna_original_feature_count",
    ):
        if name in reference:
            arrays[name] = reference[name]
    np.savez_compressed(output/"reference.npz",**arrays); np.savez_compressed(output/"predictions.npz",**predictions)
    # A quantitative-free forecast probe plus two already-admitted fitting cells.
    shard=load_shard(SHARDS/"paired-cells-00000.npz"); rows=np.arange(2)
    device=next(model.parameters()).device
    rna,protein,rs,ps=batch_arrays(shard,rows,stats,device)
    rf=torch.as_tensor(reference["rna_query_features"],device=device)
    pf=torch.as_tensor(reference["protein_query_features"],device=device)
    model.eval()
    with torch.no_grad():
        expected_state=model.encode(rf,rs,torch.ones_like(rs,dtype=torch.bool),pf,ps,torch.ones_like(ps,dtype=torch.bool)).cpu().numpy()
    source_hashes={
        f"source/{path.name}":sha256(path)
        for path in sorted((output/"source").glob("*.py"))
    }
    manifest={"schema":"slp.frangieh-cell-state-artifact/v1","sha256":{"model.safetensors":sha256(output/"model.safetensors"),"reference.npz":sha256(output/"reference.npz"),**source_hashes}}
    with np.load(STATIC, allow_pickle=False) as static:
        static_lookup = dict(zip(static["entity_id"].astype(str), static["feature_values"], strict=True))
    probe_genes = predictions["action_ids"][:3].astype(str)
    probe_features = np.stack([static_lookup[gene] for gene in probe_genes]).astype(np.float32)
    normalized_probe = np.clip(
        (probe_features - reference["feature_mean"])
        / reference["feature_scale"],
        -float(reference["feature_clip"]),
        float(reference["feature_clip"]),
    ).astype(np.float32)
    probe_context=np.arange(3,dtype=np.int64)
    probe_delta=np.empty((3,128),np.float32)
    for index in range(3):
        probe_delta[index]=normalized_probe[index]@coef[index].T+intercept[index]
    with torch.no_grad():
        delta=torch.as_tensor(probe_delta,device=device)
        expected_forecast={}
        for head in ("rna","protein"):
            expected_forecast[head]=model.observe_delta(
                delta,torch.as_tensor(reference[f"{head}_query_features"],device=device),head,
                torch.as_tensor(reference[f"{head}_controls"][probe_context],device=device),
                torch.as_tensor(stats[f"{head}_sd"],device=device),
            ).cpu().numpy()
    np.savez_compressed(
        output/"probe-input.npz", rna=rna, protein=protein,
        action_features=probe_features, context_index=probe_context,
    )
    manifest["sha256"]["probe-input.npz"] = sha256(output/"probe-input.npz")
    write_json(output/"artifact-manifest.json",manifest)
    np.savez_compressed(
        output/"probe-expected.npz", state=expected_state,
        empty_rna=reference["rna_controls"][probe_context], empty_protein=reference["protein_controls"][probe_context],
        forecast_rna=expected_forecast["rna"], forecast_protein=expected_forecast["protein"],
        forecast_state_delta=probe_delta,
    )
    manifest["sha256"]["probe-expected.npz"]=sha256(output/"probe-expected.npz"); write_json(output/"artifact-manifest.json",manifest)
    verify=subprocess.run([sys.executable,str(output/"source/verify.py"),str(output)],text=True,capture_output=True,check=True,cwd=ROOT)
    report={"schema":"slp.frangieh-cell-state-ae-latent-ridge-result/v1","reconstruction":train_report,"forecast":{"contexts":results,"passed":decision},"portableVerification":json.loads(verify.stdout),"elapsedSeconds":time.monotonic()-started,"artifacts":{"modelSha256":sha256(output/"model.safetensors"),"referenceSha256":sha256(output/"reference.npz"),"predictionsSha256":sha256(output/"predictions.npz"),"manifestSha256":sha256(output/"artifact-manifest.json")},"testAccessed":False,"benchmarkAccessed":False,"jurkatAccessed":False}
    write_json(output/"report.json",report); return report


def execute(output: Path,device_name: str):
    started=time.monotonic(); prepared=json.loads((output/"PREPARED.json").read_text());
    if sha256(output/"protocol.json")!=prepared["protocolSha256"]: raise ValueError("protocol drift")
    for name,digest in prepared["sourceHashes"].items():
        if sha256(output/"source"/name)!=digest: raise ValueError(f"source drift {name}")
    manifest,reference=verify_inputs(); stats=streaming_stats(manifest); device=torch.device(device_name)
    model,_core,train_report=train(manifest,reference,stats,device,output); genes,controls=encode_populations(model,manifest,reference,stats,device); coef,intercept,_,normalized,counts=fit_latent_ridges(genes,controls,reference); train_report["latentFittingGenesByContext"]=counts
    predictions,results,decision=forecast_and_score(model,reference,stats,coef,intercept,normalized,device,output)
    return save_artifact(output,model,reference,stats,coef,intercept,predictions,train_report,results,decision,started)


def main():
    parser=argparse.ArgumentParser(); modes=parser.add_mutually_exclusive_group(required=True); modes.add_argument("--profile",action="store_true"); modes.add_argument("--prepare-only",action="store_true"); modes.add_argument("--run",action="store_true"); parser.add_argument("--output",type=Path,default=OUTPUT); parser.add_argument("--profile-path",type=Path,default=ROOT/"results/slp11-transition/frangieh-cell-state-profile-v1.json"); parser.add_argument("--profile-steps",type=int,default=10); parser.add_argument("--device",choices=("cpu","cuda"),default="cuda"); args=parser.parse_args()
    if args.profile:
        result=profile(args.device,args.profile_steps); write_json(args.profile_path,result)
    elif args.prepare_only: result=prepare(args.output,args.profile_path)
    else: result=execute(args.output,args.device)
    print(json.dumps(result,sort_keys=True,default=lambda x:x.item()))


if __name__=="__main__": main()
