#!/usr/bin/env python3
"""Fit one frozen nonlinear residual transition on paired PCA128 states."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
PCA_ROOT = ROOT / "results/slp11-transition/frangieh-paired-pca128-latent-ridge-seed731-v1"
PCA_PAYLOAD = PCA_ROOT / "pca-forecast.npz"
PCA_SOURCE = PCA_ROOT / "source/paired_pca.py"
PCA_REPORT = PCA_ROOT / "report.json"
PCA_PREDICTIONS = PCA_ROOT / "predictions.npz"
REFERENCE = ROOT / "results/slp11-transition/frangieh-paired-state-physical1156-seed731-v2/reference.npz"
SHARDS = ROOT / "data/derived/slp11-frangieh/paired-singlecell-train-control-v1"
STATIC = ROOT / "data/derived/slp11-frangieh-static/ensembl116-goa2022-fixed-neighbor-v1/frangieh-extended-static-esm-go-fixed-physical-features.npz"
DEVELOPMENT = ROOT / "data/derived/slp11-frangieh/paired-development-v1/development.npz"
BASELINE_REPORT = ROOT / "results/slp11-transition/frangieh-paired-state-vs-static-scoring-v1/report.json"
CELL_AE = ROOT / "results/slp11-transition/frangieh-cell-state-ae-latent-ridge-seed731-v1"
TRANSITION = ROOT / "modules/slp-1-1-action-state-transition-v1/transition.py"
INFERENCE = ROOT / "modules/slp-1-1-action-state-transition-v1/inference.py"
VERIFIER = ROOT / "scripts/verify_slp11_frangieh_pca_residual_transition.py"
METRICS = PCA_ROOT / "source/frangieh_basal_ridge.py"
OUTPUT = ROOT / "results/slp11-transition/frangieh-pca128-residual-transition-seed731-v1"
PROFILE = ROOT / "results/slp11-transition/frangieh-pca128-residual-transition-profile-v1.json"
HASHES = {
    "pcaReport": "e9afd49a315946b68a0862903eb67615d9291080a86c8613f82378110c8cff4f",
    "pcaPayload": "5070bdb09f9949132d4d610f6ba379d1e96537cd162554897916a8d83c2b2e26",
    "pcaPredictions": "bd81085f55b33050fe8670bf2f2cb062d32de9134c4b0100eabd6f1f452638c0",
    "reference": "8b82e4781b73a721f995dd218ef341ea8324b87d3c9189bfe40644d436800e73",
    "manifest": "e791b5cf35da96fa71951a4a240ed58b53e278d3c57e44066680abd3f386a9c7",
    "static": "347fd1bf87d8fc3d0b447676082b4bcb64f021c9f12c7df4d1754dc262b2bf72",
    "development": "4bbb1eec9ede66211f1316b2841bb0037032ef975cd6c92d34aba0adb5fed744",
    "baseline": "d0c577e093198e9060a582cc5852b0db61246daa5772ae0c1e8451addc584b90",
    "cellAeReport": "cada9a66568dda2340a95dd0bbd6b96bcc7af2ac76bf98f9b8f4e1d681bc182f",
    "cellAePredictions": "a5cc6724ad55c5d3f2ad709be36a5fcbcb77e7255d7f79af29145556e2a24b96",
}
CONTEXTS = ("Co-culture", "Control", "IFNγ")
SEED = 731
BATCH = 32
UPDATES = 1000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module); return module


def write_json(path: Path, value: object) -> None:
    def clean(item: object) -> object:
        if isinstance(item, dict): return {str(key): clean(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)): return [clean(entry) for entry in item]
        if isinstance(item, np.generic): return item.item()
        if isinstance(item, float) and not np.isfinite(item): return None
        return item
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_inputs() -> dict[str, Path]:
    paths = {
        "pcaReport": PCA_REPORT, "pcaPayload": PCA_PAYLOAD,
        "pcaPredictions": PCA_PREDICTIONS, "reference": REFERENCE,
        "manifest": SHARDS / "manifest.json", "static": STATIC,
        "development": DEVELOPMENT, "baseline": BASELINE_REPORT,
        "cellAeReport": CELL_AE / "report.json",
        "cellAePredictions": CELL_AE / "predictions.npz",
    }
    for name, path in paths.items():
        if sha256(path) != HASHES[name]: raise ValueError(f"input hash mismatch: {name}")
    return paths


def load_pca(core_path: Path = PCA_SOURCE):
    core = load(core_path, "residual_frozen_pca")
    return core, core.PcaForecastArtifact.load(PCA_PAYLOAD)


def load_shard(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["rna"] = sparse.csr_matrix(
        (arrays["rna_data"], arrays["rna_indices"], arrays["rna_indptr"]),
        shape=tuple(arrays["rna_shape"]),
    )
    return arrays


def aggregate_fitting_profiles(manifest: dict) -> dict[str, np.ndarray]:
    guide_rna: dict[tuple[str, str, str], np.ndarray] = {}
    guide_protein: dict[tuple[str, str, str], np.ndarray] = {}
    guide_count = defaultdict(int)
    for item in manifest["shards"]:
        path = SHARDS / item["path"]
        if sha256(path) != item["sha256"]: raise ValueError(f"shard hash drift: {path.name}")
        shard = load_shard(path)
        selected = np.flatnonzero(shard["source_split"] == "train")
        keys = np.asarray([
            f"{shard['action_ids'][row]}\0{shard['context_ids'][row]}\0{shard['target_guide_sets'][row]}"
            for row in selected
        ])
        for encoded in np.unique(keys):
            gene, context, guide = encoded.split("\0"); key = (gene, context, guide)
            rows = selected[keys == encoded]
            rna_sum = np.asarray(shard["rna"][rows].astype(np.float64).sum(axis=0)).ravel()
            protein_sum = np.asarray(shard["protein_values"])[rows].astype(np.float64).sum(axis=0)
            guide_rna[key] = guide_rna.get(key, np.zeros(18_063)) + rna_sum
            guide_protein[key] = guide_protein.get(key, np.zeros(20)) + protein_sum
            guide_count[key] += len(rows)
    gene_rna: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    gene_protein: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for gene, context, guide in sorted(guide_rna):
        key = (gene, context, guide); gene_key = (gene, context)
        gene_rna[gene_key].append(guide_rna[key] / guide_count[key])
        gene_protein[gene_key].append(guide_protein[key] / guide_count[key])
    keys = sorted(gene_rna, key=lambda item: (CONTEXTS.index(item[1]), item[0]))
    result = {
        "action_ids": np.asarray([key[0] for key in keys]),
        "context_ids": np.asarray([key[1] for key in keys]),
        "context_index": np.asarray([CONTEXTS.index(key[1]) for key in keys], dtype=np.int64),
        "rna": np.stack([np.mean(gene_rna[key], axis=0) for key in keys]).astype(np.float32),
        "protein": np.stack([np.mean(gene_protein[key], axis=0) for key in keys]).astype(np.float32),
        "guide_counts": np.asarray([len(gene_rna[key]) for key in keys], dtype=np.int64),
    }
    if len(keys) != 453 or len(guide_rna) != 1399:
        raise ValueError("fitting gene/guide population drift")
    return result


def prepare_training(pca, profiles: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    with np.load(STATIC, allow_pickle=False) as archive:
        lookup = dict(zip(archive["entity_id"].astype(str), archive["feature_values"], strict=True))
    raw = np.stack([lookup[str(gene)] for gene in profiles["action_ids"]]).astype(np.float32)
    context = profiles["context_index"]
    normalized = pca.ridge.normalize(raw).astype(np.float32)
    base_delta = pca.ridge.predict(raw, context).astype(np.float32)
    control_state = pca.pca.encode(
        sparse.csr_matrix(pca.rna_controls), pca.protein_controls
    ).astype(np.float32)
    result = dict(profiles)
    result.update({
        "raw_action_features": raw, "action_features": normalized,
        "base_delta": base_delta, "control_state": control_state,
        "rna_scale": np.stack([
            np.maximum(profiles["rna"][context == index].std(0, dtype=np.float64), 0.05)
            for index in range(3)
        ]).astype(np.float32),
        "protein_scale": np.stack([
            np.maximum(profiles["protein"][context == index].std(0, dtype=np.float64), 0.05)
            for index in range(3)
        ]).astype(np.float32),
    })
    return result


def decoder_bases(pca) -> tuple[np.ndarray, np.ndarray]:
    components = pca.pca.components
    rna = components[:18_063] * pca.pca.stats.rna_sd[:, None] / pca.pca.stats.rna_weight
    protein = components[18_063:] * pca.pca.stats.protein_sd[:, None] / pca.pca.stats.protein_weight
    return rna.astype(np.float32), protein.astype(np.float32)


def initialize(device: torch.device, transition_path: Path = TRANSITION):
    core = load(transition_path, "residual_transition_training")
    torch.manual_seed(SEED); torch.use_deterministic_algorithms(True)
    model = core.ResidualStateTransition(core.Config(1156, 128, 128, 0.2)).to(device)
    return core, model


def loss_value(model, batch: np.ndarray, data: dict, pca, bases, device: torch.device):
    context = data["context_index"][batch]
    delta = model(
        torch.as_tensor(data["action_features"][batch], device=device),
        torch.as_tensor(data["control_state"][context], device=device),
        torch.as_tensor(data["base_delta"][batch], device=device),
        torch.ones(len(batch), dtype=torch.bool, device=device),
    )
    losses = []
    for head, basis in zip(("rna", "protein"), bases, strict=True):
        control = getattr(pca, f"{head}_controls")[context]
        prediction = torch.as_tensor(control, device=device) + delta @ torch.as_tensor(basis.T, device=device)
        target = torch.as_tensor(data[head][batch], device=device)
        scale = torch.as_tensor(data[f"{head}_scale"][context], device=device)
        losses.append(torch.square((prediction - target) / scale).mean())
    return 0.5 * (losses[0] + losses[1]), delta


def profile(device_name: str, steps: int) -> dict[str, object]:
    verify_inputs(); _, pca = load_pca(); device = torch.device(device_name)
    _, model = initialize(device); optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    bases = decoder_bases(pca); rng = np.random.default_rng(SEED)
    data = {
        "action_features": rng.normal(size=(96, 1156)).astype(np.float32),
        "control_state": rng.normal(size=(3, 128)).astype(np.float32),
        "base_delta": rng.normal(size=(96, 128)).astype(np.float32),
        "context_index": np.repeat(np.arange(3), 32),
        "rna": rng.normal(size=(96, 18_063)).astype(np.float32),
        "protein": rng.normal(size=(96, 20)).astype(np.float32),
        "rna_scale": np.ones((3, 18_063), np.float32),
        "protein_scale": np.ones((3, 20), np.float32),
    }
    torch.cuda.synchronize() if device.type == "cuda" else None; started = time.monotonic()
    for step in range(steps):
        batch = np.arange(step * BATCH, (step + 1) * BATCH) % 96
        optimizer.zero_grad(set_to_none=True); loss, _ = loss_value(model, batch, data, pca, bases, device)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    torch.cuda.synchronize() if device.type == "cuda" else None; elapsed = time.monotonic() - started
    return {"steps": steps, "seconds": elapsed, "secondsPerStep": elapsed / steps, "projected1000UpdateSeconds": elapsed / steps * 1000, "batch": BATCH, "parameters": sum(value.numel() for value in model.parameters()), "targetFreeSynthetic": True}


def prepare(output: Path, profile_path: Path) -> dict[str, object]:
    if output.exists(): raise FileExistsError(output)
    inputs = verify_inputs(); profile_report = json.loads(profile_path.read_text())
    if profile_report["projected1000UpdateSeconds"] >= 300: raise ValueError("profile exceeds GPU cap")
    output.mkdir(parents=True); source = output / "source"; source.mkdir()
    sources = {"transition.py": TRANSITION, "inference.py": INFERENCE, "paired_pca.py": PCA_SOURCE, "frangieh_basal_ridge.py": METRICS, "trainer.py": Path(__file__), "verify.py": VERIFIER}
    source_hashes = {}
    for name, path in sources.items(): shutil.copy2(path, source / name); source_hashes[name] = sha256(source / name)
    protocol = {
        "schema": "slp.frangieh-pca128-residual-transition-protocol/v1",
        "hypothesis": "A nonlinear action-and-control-state residual improves held-gene molecular forecasts beyond the frozen linear PCA-state ridge.",
        "model": "action1156 plus supplied raw control PCA128 -> Linear128/LayerNorm/GELU/dropout.2 -> zero-initialized Linear128 residual; output frozen ridge delta plus residual; empty action exact zero; combinations unsupported",
        "state": "frozen rank128 paired-assay PCA coordinates; context control states are raw supplied PCA scores with no learned normalizer",
        "trainingPopulation": "151 fitting genes x3 contexts; cells averaged within exact guide, guides equally within gene/context; original held-gene cells absent from shards",
        "objective": "equal RNA/protein decoded mean MSE standardized by per-query/context fitting-gene target SD floor0.05",
        "optimization": "1000 final-checkpoint-only updates; B32 globally equal-context deterministic sampling; AdamWlr.0005/decay.1,clip1,seed731; no early stopping or validation",
        "gate": "all 3contexts x2heads require >=1% raw MSE improvement versus mean,base577,physical1156,prior paired,PCA,cellAE; centered r>=.10 and nonregression versus every defined comparator",
        "profile": profile_report,
        "inputs": {name: {"path": str(path), "sha256": HASHES[name]} for name, path in inputs.items()},
        "sourceHashes": source_hashes,
        "limitations": ["assay-specific PCA output state and query loadings", "Gaussian point-mean objective only", "no generated-cell or uncertainty claim", "single actions only", "single development seed"],
        "testAccessed": False, "jurkatAccessed": False, "benchmarkAccessed": False,
    }
    write_json(output / "protocol.json", protocol)
    prepared = {"protocolSha256": sha256(output / "protocol.json"), "sourceHashes": source_hashes}
    write_json(output / "PREPARED.json", prepared); return prepared


def schedule(data: dict[str, np.ndarray]) -> np.ndarray:
    rng = np.random.default_rng(SEED); context_rows = [np.flatnonzero(data["context_index"] == index) for index in range(3)]
    contexts = np.arange(UPDATES * BATCH, dtype=np.int64) % 3; contexts = contexts.reshape(UPDATES, BATCH)
    result = np.empty_like(contexts)
    for step in range(UPDATES):
        shuffled = rng.permutation(contexts[step])
        result[step] = [rng.choice(context_rows[int(context)]) for context in shuffled]
    return result


def train(data, pca, output: Path, device: torch.device):
    _, model = initialize(device, output / "source/transition.py"); optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    batches = schedule(data); schedule_hash = hashlib.sha256(batches.tobytes()).hexdigest(); bases = decoder_bases(pca)
    started = time.monotonic(); history = []
    for step, batch in enumerate(batches, 1):
        optimizer.zero_grad(set_to_none=True); loss, _ = loss_value(model, batch, data, pca, bases, device)
        if not torch.isfinite(loss): raise FloatingPointError("nonfinite transition loss")
        loss.backward(); gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)); optimizer.step()
        if step == 1 or step % 100 == 0:
            event = {"update": step, "loss": float(loss.detach()), "gradientNormBeforeClip": gradient}; history.append(event); print(json.dumps(event), flush=True)
        if time.monotonic() - started > 300: raise TimeoutError("transition fitting exceeded 300 seconds")
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}; save_file(state, str(output / "transition.safetensors"))
    return model, {"seconds": time.monotonic() - started, "updates": UPDATES, "scheduleSha256": schedule_hash, "history": history}


def predict_in_memory(model, pca, data, device):
    model.eval(); outputs = []
    with torch.no_grad():
        for start in range(0, len(data["action_features"]), 256):
            rows = np.arange(start, min(start + 256, len(data["action_features"])))
            context = data["context_index"][rows]
            outputs.append(model(torch.as_tensor(data["action_features"][rows], device=device), torch.as_tensor(data["control_state"][context], device=device), torch.as_tensor(data["base_delta"][rows], device=device), torch.ones(len(rows), dtype=torch.bool, device=device)).cpu().numpy())
    delta = np.concatenate(outputs); rna_delta, protein_delta = pca.pca.decode_delta(delta)
    context = data["context_index"]
    return delta.astype(np.float32), (pca.rna_controls[context] + rna_delta).astype(np.float32), (pca.protein_controls[context] + protein_delta).astype(np.float32)


def load_held(pca) -> dict[str, np.ndarray]:
    helper = load(METRICS, "residual_held_collapse")
    with np.load(DEVELOPMENT, allow_pickle=False) as archive:
        indices = archive["split_validation"]
        action, context, rna, guides = helper.collapse_gene_profiles(archive["action_ids"][indices], archive["context_ids"][indices], archive["rna_targets"][indices])
        pa, pc, protein, pg = helper.collapse_gene_profiles(archive["action_ids"][indices], archive["context_ids"][indices], archive["protein_targets"][indices])
    if not np.array_equal(action, pa) or not np.array_equal(context, pc) or not np.array_equal(guides, pg): raise ValueError("held head alignment drift")
    with np.load(STATIC, allow_pickle=False) as archive:
        lookup = dict(zip(archive["entity_id"].astype(str), archive["feature_values"], strict=True))
    raw = np.stack([lookup[str(gene)] for gene in action]).astype(np.float32); ci = np.asarray([CONTEXTS.index(str(item)) for item in context])
    control_state = pca.pca.encode(sparse.csr_matrix(pca.rna_controls), pca.protein_controls).astype(np.float32)
    return {"action_ids": action, "context_ids": context, "context_index": ci, "rna": rna, "protein": protein, "guide_counts": guides, "raw_action_features": raw, "action_features": pca.ridge.normalize(raw).astype(np.float32), "base_delta": pca.ridge.predict(raw, ci).astype(np.float32), "control_state": control_state}


def score(output: Path, held: dict, delta, rna_prediction, protein_prediction):
    helper = load(output / "source/frangieh_basal_ridge.py", "residual_metrics")
    baseline = json.loads(BASELINE_REPORT.read_text()); pca_report = json.loads(PCA_REPORT.read_text()); cell_report = json.loads((CELL_AE / "report.json").read_text())
    with np.load(PCA_PREDICTIONS, allow_pickle=False) as pp, np.load(CELL_AE / "predictions.npz", allow_pickle=False) as cp:
        for key in ("action_ids", "context_ids", "rna_truth", "protein_truth"):
            held_key = key.replace("_truth", "") if key.endswith("_truth") else key
            if not np.array_equal(held[held_key], pp[key]) or not np.array_equal(pp[key], cp[key]): raise ValueError(f"comparator alignment drift: {key}")
    contexts = {}; all_pass = True
    for context in CONTEXTS:
        rows = held["context_ids"] == context; contexts[context] = {"heads": {}}
        for head, prediction in (("rna", rna_prediction), ("protein", protein_prediction)):
            truth = held[head][rows]; metrics = helper.metrics(prediction[rows], truth, np.ones(truth.shape[1]))
            old = baseline["contexts"][context]["heads"][head]
            comparators = {**old["baselines"], "priorPairedWorld": old["world"], "pca": pca_report["forecast"]["contexts"][context]["heads"][head]["pca"], "cellAE": cell_report["forecast"]["contexts"][context]["heads"][head]["world"]}
            checks = {}
            for label, comparator in comparators.items():
                comparator_r = comparator["query_centroid_adjusted_profile_pearson"]
                checks[label] = {"mseImprovement": 1 - metrics["raw_mse"] / comparator["raw_mse"], "mseAtLeastOnePercent": 1 - metrics["raw_mse"] / comparator["raw_mse"] >= .01, "rNonregression": comparator_r is None or metrics["query_centroid_adjusted_profile_pearson"] >= comparator_r}
            r = metrics["query_centroid_adjusted_profile_pearson"]
            passed = bool(np.isfinite(r) and r >= .10 and all(item["mseAtLeastOnePercent"] and item["rNonregression"] for item in checks.values())); all_pass &= passed
            contexts[context]["heads"][head] = {"world": metrics, "comparators": comparators, "checks": checks, "passed": passed}
    predictions = {"action_ids": held["action_ids"], "context_ids": held["context_ids"], "context_index": held["context_index"], "rna_truth": held["rna"], "protein_truth": held["protein"], "rna_prediction": rna_prediction, "protein_prediction": protein_prediction, "latent_delta": delta, "guide_counts": held["guide_counts"]}
    return predictions, contexts, bool(all_pass)


def package(output, model, pca, training, held, predictions, contexts, decision, train_report, started):
    shutil.copy2(PCA_PAYLOAD, output / "pca-forecast.npz")
    control_state = pca.pca.encode(sparse.csr_matrix(pca.rna_controls), pca.protein_controls).astype(np.float32)
    np.savez_compressed(output / "transition-reference.npz", control_state=control_state, rna_fitting_target_scale=training["rna_scale"], protein_fitting_target_scale=training["protein_scale"], context_names=np.asarray(CONTEXTS))
    np.savez_compressed(output / "predictions.npz", **predictions)
    probe_rows = np.asarray([0, 43, 86]); probe_input = {"action_features": held["raw_action_features"][probe_rows], "context_index": held["context_index"][probe_rows]}
    probe_data = {key: value[probe_rows] if key in ("action_features", "base_delta", "context_index") else value for key, value in held.items() if key in ("action_features", "base_delta", "context_index", "control_state")}
    probe_delta, probe_rna, probe_protein = predict_in_memory(model, pca, probe_data, next(model.parameters()).device)
    np.savez_compressed(output / "probe-input.npz", **probe_input); np.savez_compressed(output / "probe-expected.npz", rna=probe_rna, protein=probe_protein, state_delta=probe_delta)
    source_hashes = {f"source/{path.name}": sha256(path) for path in sorted((output / "source").glob("*.py"))}
    files = {"transition.safetensors": sha256(output / "transition.safetensors"), "pca-forecast.npz": sha256(output / "pca-forecast.npz"), "transition-reference.npz": sha256(output / "transition-reference.npz"), "probe-input.npz": sha256(output / "probe-input.npz"), "probe-expected.npz": sha256(output / "probe-expected.npz"), **source_hashes}
    write_json(output / "artifact-manifest.json", {"schema": "slp.residual-pca-state-transition-artifact/v1", "sha256": files})
    verification = subprocess.run([sys.executable, str(output / "source/verify.py"), str(output)], capture_output=True, text=True, check=True).stdout
    report = {"schema": "slp.frangieh-pca128-residual-transition-result/v1", "training": train_report, "forecast": {"contexts": contexts, "passed": decision}, "portableVerification": json.loads(verification), "elapsedSeconds": time.monotonic() - started, "artifacts": {"modelSha256": sha256(output / "transition.safetensors"), "referenceSha256": sha256(output / "transition-reference.npz"), "predictionsSha256": sha256(output / "predictions.npz"), "manifestSha256": sha256(output / "artifact-manifest.json")}, "testAccessed": False, "jurkatAccessed": False, "benchmarkAccessed": False}
    write_json(output / "report.json", report); return report


def execute(output: Path, device_name: str):
    started = time.monotonic(); prepared = json.loads((output / "PREPARED.json").read_text())
    if sha256(output / "protocol.json") != prepared["protocolSha256"]: raise ValueError("protocol drift")
    for name, digest in prepared["sourceHashes"].items():
        if sha256(output / "source" / name) != digest: raise ValueError(f"source drift: {name}")
    verify_inputs(); pca_core = load(output / "source/paired_pca.py", "residual_packaged_pca"); pca = pca_core.PcaForecastArtifact.load(PCA_PAYLOAD)
    manifest = json.loads((SHARDS / "manifest.json").read_text()); profiles = aggregate_fitting_profiles(manifest); training = prepare_training(pca, profiles)
    device = torch.device(device_name); model, train_report = train(training, pca, output, device)
    freeze = {"protocolSha256": prepared["protocolSha256"], "modelSha256": sha256(output / "transition.safetensors"), "trainingRecords": len(profiles["action_ids"]), "heldTargetsOpened": False}; write_json(output / "FROZEN-BEFORE-HELD.json", freeze)
    held = load_held(pca); delta, rna, protein = predict_in_memory(model, pca, held, device); predictions, contexts, decision = score(output, held, delta, rna, protein)
    return package(output, model, pca, training, held, predictions, contexts, decision, train_report, started)


def main() -> None:
    parser = argparse.ArgumentParser(); modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--profile", action="store_true"); modes.add_argument("--prepare-only", action="store_true"); modes.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT); parser.add_argument("--profile-path", type=Path, default=PROFILE); parser.add_argument("--profile-steps", type=int, default=20); parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda"); args = parser.parse_args()
    if args.profile: result = profile(args.device, args.profile_steps); write_json(args.profile_path, result)
    elif args.prepare_only: result = prepare(args.output, args.profile_path)
    else: result = execute(args.output, args.device)
    print(json.dumps(result, sort_keys=True, default=lambda item: item.item()))


if __name__ == "__main__": main()
