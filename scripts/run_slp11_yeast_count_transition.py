#!/usr/bin/env python3
"""Prepare, profile and run the first corrected-count yeast transition pilot."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import psutil
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
MOMENTS = (
    ROOT
    / "data/derived/slp11-yeast-atlas-counts/nadal-ribelles-raw-rna-development-v1/moments-manifest.json"
)
STATIC = (
    ROOT
    / "data/derived/slp11-yeast-shared-static/current-sgd-strict-query-full-raw-actions-esm8m-complete-shared-go-v2/yeast-static-esm8m-shared-go-mf-cc-features.npz"
)
WT = (
    ROOT
    / "results/slp11-transition/yeast-wildtype-batch-diagnostic-v1/wildtype-reference.npz"
)
BASELINE = ROOT / "results/slp11-transition/yeast-raw-count-batch-ridge-v1"
BASELINE_SCORING = (
    ROOT
    / "results/slp11-transition/yeast-raw-count-batch-ridge-roundoff-scoring-v1/report.json"
)
MODULE = ROOT / "modules/slp-1-1-yeast-count-transition-v1"
OUTPUT = ROOT / "results/slp11-transition/yeast-rna-world-transition-seed731-v1"
FITTING = (
    ROOT / "data/derived/slp11-yeast-atlas-counts/nadal-ribelles-rna-neural-fitting-v1"
)
HASHES = {
    "moments": "70a49ecaeb271fc72ecc93ede207c59a816e74d1ae3133bbf3a2803cce5d8eba",
    "static": "81cda9469380c9efa000a40b2cd5e816a1d397ce777288fa53b0bcf26a55dc25",
    "wt": "190dc64dd9ee8809f56f82b690265827376c72b36286e46e04b8aebee64fa1b5",
    "core": "fdb4555bd0f7c0a0786539da67048f6985f4ec2f36ef7aa45bd22c7c6bfbb2ef",
    "baselineReport": "e15c9b14dc37b4eae01ef1e5bc847860a2d39273c76c930cb12030e622488824",
    "baselineScoring": "291b8b34d9b03b0bedb6a40723cbab07b6fd2c094dbffdb5a5a644141454c128",
    "baselineProtocol": "2669369cca209616bb891f206ff3670854c9fabb5d041c00d4e1fcc4654fc53e",
    "baselineFreeze": "f64454d201aa812d7a24af59764f97e13375839229c3c38c133c91417cb86a3f",
    "controlBatchReference": "9ed278470e71ab4f491a4c4433b95803feaf095583a7267d3d5b65c29f09b99a",
    "naclBatchReference": "479f8e35ce1688b8b2773a8340c04cf61734e863bf9e516e9a760766f40b9ebc",
}
CONTEXTS = ("Control", "NaCl")
SEED = 731
BATCH_SIZE = 64
STEPS = 12_000
FEATURES = 577
QUERIES = 6683
CONTEXT_TOKENS = 64
RSS_CAP = 6 * (1 << 30)
PREPARE_CAP = 900.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_python(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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

    path.write_text(
        json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def stored_npz_memmap(path: Path, key: str) -> np.memmap:
    """Memory-map one uncompressed NPY member without decoding other rows/members."""
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(f"{key}.npy")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{key} is compressed and cannot be row-allowlisted")
        with path.open("rb") as stream:
            stream.seek(info.header_offset)
            header = stream.read(30)
            fields = struct.unpack("<IHHHHHIIIHH", header)
            if fields[0] != 0x04034B50:
                raise ValueError("invalid local ZIP header")
            stream.seek(fields[-2] + fields[-1], 1)
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"unsupported NPY member version {version}")
            offset = stream.tell()
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=offset,
        shape=shape,
        order="F" if fortran else "C",
    )


def row_weights(
    contexts: np.ndarray, actions: np.ndarray, num_cells: np.ndarray
) -> np.ndarray:
    """Equal environment/gene mass; within gene, population mass follows cells."""
    contexts = np.asarray(contexts)
    actions = np.asarray(actions).astype(str)
    cells = np.asarray(num_cells, dtype=np.float64)
    if (
        contexts.shape != actions.shape
        or cells.shape != actions.shape
        or np.any(cells <= 0)
        or np.any(contexts < 0)
    ):
        raise ValueError("invalid row-weight metadata")
    weights = np.empty(len(actions), dtype=np.float64)
    unique_contexts = np.unique(contexts)
    for context in unique_contexts:
        local = np.flatnonzero(contexts == context)
        genes = np.unique(actions[local])
        for gene in genes:
            rows = local[actions[local] == gene]
            weights[rows] = (
                len(actions)
                / (len(unique_contexts) * len(genes))
                * cells[rows]
                / cells[rows].sum()
            )
    if not np.isclose(weights.sum(), len(weights), rtol=1e-12, atol=1e-9):
        raise RuntimeError("row weights do not have global mean one")
    return weights


def _load_static() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    with np.load(STATIC, allow_pickle=False) as archive:
        ids = archive["entity_id"].astype(str)
        taxa = archive["entity_taxon"]
        values = archive["feature_values"].astype(np.float32)
    if (
        values.shape != (6744, FEATURES)
        or np.any(taxa != 4932)
        or len(set(ids)) != len(ids)
        or not np.isfinite(values).all()
    ):
        raise ValueError("static feature contract drift")
    return ids, values, {value: index for index, value in enumerate(ids)}


def _wt_reference() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(WT, allow_pickle=False) as archive:
        query_ids = archive["query_ids"].astype(str)
        control = archive["control_mean"].astype(np.float64)
        nacl = archive["nacl_mean"].astype(np.float64)
        control_batches = archive["control_batch_ids"].astype(str)
        nacl_batches = archive["nacl_batch_ids"].astype(str)
    means = np.concatenate((control, nacl))
    batch_ids = np.concatenate((control_batches, nacl_batches))
    contexts = np.concatenate(
        (np.zeros(len(control), dtype=np.int64), np.ones(len(nacl), dtype=np.int64))
    )
    return query_ids, means, batch_ids, contexts


def _metadata() -> tuple[list[dict[str, object]], np.ndarray]:
    manifest = json.loads(MOMENTS.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    query_ids = None
    for entry in manifest["shards"]:
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError("moments shard hash drift")
        with np.load(path, allow_pickle=False) as archive:
            q = archive["query_ids"].astype(str)
            actions = archive["group_action_id"].astype(str)
            roles = archive["development_role"].astype(str)
            cells = archive["num_cells"].astype(np.int64)
            context = str(archive["context"].item())
            batch = str(archive["batch_id"].item())
        if query_ids is None:
            query_ids = q
        elif not np.array_equal(q, query_ids):
            raise ValueError("query axes differ")
        if context != entry["context"] or batch != entry["batchId"]:
            raise ValueError("manifest shard identity drift")
        records.append(
            {
                "path": path,
                "context": context,
                "batch": batch,
                "actions": actions,
                "roles": roles,
                "cells": cells,
            }
        )
    assert query_ids is not None
    return records, query_ids


def _guard(started: float) -> None:
    if time.monotonic() - started > PREPARE_CAP:
        raise TimeoutError("preparation exceeded 900 seconds")
    if psutil.Process().memory_info().rss > RSS_CAP:
        raise MemoryError("preparation exceeded 6 GiB RSS")


def prepare() -> dict[str, object]:
    started = time.monotonic()
    if OUTPUT.exists() or FITTING.exists():
        raise FileExistsError("immutable output or fitting-data directory exists")
    for path, expected in (
        (MOMENTS, HASHES["moments"]),
        (STATIC, HASHES["static"]),
        (WT, HASHES["wt"]),
        (MODULE / "control_transition_model.py", HASHES["core"]),
        (BASELINE / "report.json", HASHES["baselineReport"]),
        (BASELINE_SCORING, HASHES["baselineScoring"]),
        (BASELINE / "protocol-v2.json", HASHES["baselineProtocol"]),
        (BASELINE / "FROZEN-BEFORE-VALIDATION.json", HASHES["baselineFreeze"]),
        (BASELINE / "control-batchReference.npz", HASHES["controlBatchReference"]),
        (BASELINE / "nacl-batchReference.npz", HASHES["naclBatchReference"]),
    ):
        if sha256(path) != expected:
            raise ValueError(f"input hash mismatch: {path}")
    OUTPUT.mkdir(parents=True)
    FITTING.mkdir(parents=True)
    source = OUTPUT / "source"
    source.mkdir()
    for name, path in {
        "control_transition_model.py": MODULE / "control_transition_model.py",
        "inference.py": MODULE / "inference.py",
        "trainer.py": Path(__file__),
    }.items():
        shutil.copy2(path, source / name)
    preparation_protocol = {
        "schema": "slp.yeast-count-transition-preparation-protocol/v1",
        "status": "frozen-before-fitting-moment-read",
        "hypothesis": "A shared static-action and measured-WT-state transition improves unseen-gene RNA profiles over pooled and batch feature-linear baselines in both environments.",
        "advancement": "Each environment: raw MSE at least 2% below pooled ridge, batch ridge, pooled mean and batch mean; same-batch-reference residual independently query-centered profile r finite >=.10 and no lower than every defined baseline r.",
        "model": "unchanged MinimalControlTransition fdb455; action/query577, hidden128, state128, dropout.2; no learned IDs, batch, genotype or clone embeddings",
        "training": "38978 fitting populations only; 12000 deterministic updates, batch64, seed731, AdamW lr.0005 decay.1, clip1, final checkpoint only",
        "objective": "per-environment fitting-gene query-SD standardized MSE; equal environment/gene weight and within-gene population weight proportional to positive cell count; global row-weight mean1",
        "validation": "8491 validation population moment rows remain unindexed until checkpoint/reference/freeze receipt exist; exactly one final evaluation",
        "inputs": HASHES,
        "sourceHashes": {path.name: sha256(path) for path in source.glob("*.py")},
        "testOrBenchmarkAccess": False,
    }
    write_json(OUTPUT / "preparation-protocol.json", preparation_protocol)

    records, query_ids = _metadata()
    _static_ids, static_values, static_lookup = _load_static()
    wt_query_ids, controls, batch_ids, batch_context = _wt_reference()
    if not np.array_equal(query_ids, wt_query_ids) or len(query_ids) != QUERIES:
        raise ValueError("WT/moments query axis mismatch")
    batch_lookup = {
        (CONTEXTS[int(context)], str(batch)): index
        for index, (batch, context) in enumerate(
            zip(batch_ids, batch_context, strict=True)
        )
    }
    train_rows = sum(
        int(np.count_nonzero(item["roles"] == "train")) for item in records
    )
    if train_rows != 38978:
        raise ValueError(f"fitting population count drift: {train_rows}")
    fitting_genes = sorted(
        {
            str(gene)
            for item in records
            for gene, role in zip(item["actions"], item["roles"], strict=True)
            if role == "train"
        }
    )
    missing = [
        gene for gene in [*fitting_genes, *query_ids] if gene not in static_lookup
    ]
    if missing:
        raise ValueError(f"static pack misses {len(missing)} required stable IDs")
    fitting_features = np.stack(
        [static_values[static_lookup[g]] for g in fitting_genes]
    )
    feature_mean = fitting_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = fitting_features.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = np.where(feature_std > 1e-5, feature_std, 1.0).astype(np.float32)
    query_features = np.stack([static_values[static_lookup[g]] for g in query_ids])
    query_normalized = ((query_features - feature_mean) / feature_std).astype(
        np.float32
    )
    action_features = np.stack([static_values[static_lookup[g]] for g in fitting_genes])
    action_normalized = ((action_features - feature_mean) / feature_std).astype(
        np.float32
    )
    fitting_lookup = {gene: index for index, gene in enumerate(fitting_genes)}

    target_path = FITTING / "train-targets.npy"
    targets = np.lib.format.open_memmap(
        target_path, mode="w+", dtype=np.float32, shape=(train_rows, QUERIES)
    )
    row_actions = np.empty(train_rows, dtype="<U14")
    row_action_index = np.empty(train_rows, dtype=np.int64)
    row_context = np.empty(train_rows, dtype=np.int64)
    row_batch = np.empty(train_rows, dtype=np.int64)
    row_cells = np.empty(train_rows, dtype=np.int64)
    residual_sum = np.zeros((2, len(fitting_genes), QUERIES), dtype=np.float64)
    residual_cells = np.zeros((2, len(fitting_genes)), dtype=np.int64)
    cursor = 0
    for item in records:
        train = np.flatnonzero(item["roles"] == "train")
        if not len(train):
            continue
        context_index = CONTEXTS.index(str(item["context"]))
        batch_index = batch_lookup[(str(item["context"]), str(item["batch"]))]
        member = stored_npz_memmap(Path(item["path"]), "sum")
        # Only these fitting rows are indexed; validation quantitative bytes remain unread.
        selected_sum = np.asarray(member[train], dtype=np.float64)
        cells = np.asarray(item["cells"])[train]
        actions = np.asarray(item["actions"])[train].astype(str)
        stop = cursor + len(train)
        targets[cursor:stop] = selected_sum / cells[:, None]
        row_actions[cursor:stop] = actions
        row_action_index[cursor:stop] = [fitting_lookup[g] for g in actions]
        row_context[cursor:stop] = context_index
        row_batch[cursor:stop] = batch_index
        row_cells[cursor:stop] = cells
        control = controls[batch_index]
        for local, (gene, cell_count) in enumerate(zip(actions, cells, strict=True)):
            gene_index = fitting_lookup[gene]
            residual_sum[context_index, gene_index] += (
                selected_sum[local] - cell_count * control
            )
            residual_cells[context_index, gene_index] += cell_count
        cursor = stop
        del member, selected_sum
        gc.collect()
        _guard(started)
    if cursor != train_rows:
        raise RuntimeError("fitting row materialization count mismatch")
    targets.flush()
    del targets

    observed_genes = residual_cells > 0
    profiles: list[np.ndarray] = []
    scales = np.empty((2, QUERIES), dtype=np.float32)
    centroids = np.empty((2, QUERIES), dtype=np.float32)
    for context in range(2):
        values = (
            residual_sum[context, observed_genes[context]]
            / residual_cells[context, observed_genes[context], None]
        )
        profiles.append(values)
        scales[context] = np.maximum(values.std(axis=0), 0.05).astype(np.float32)
        centroids[context] = values.mean(axis=0).astype(np.float32)
    pooled_centroid = sum(value.mean(axis=0) for value in profiles) / 2
    pooled_variance = (
        sum(np.mean((value - pooled_centroid) ** 2, axis=0) for value in profiles) / 2
    )
    amplitude = np.maximum(np.sqrt(pooled_variance), 0.05).astype(np.float32)
    weights = row_weights(row_context, row_actions, row_cells).astype(np.float32)

    control_variance = controls.var(axis=0, dtype=np.float64)
    selected_basal = np.argsort(-control_variance, kind="stable")[:CONTEXT_TOKENS]
    basal_mean = controls[:, selected_basal].mean(axis=0, dtype=np.float64)
    basal_std = controls[:, selected_basal].std(axis=0, dtype=np.float64)
    basal_std = np.where(basal_std > 1e-5, basal_std, 1.0)
    basal_values = ((controls[:, selected_basal] - basal_mean) / basal_std).astype(
        np.float32
    )
    basal_mask = np.ones_like(basal_values, dtype=np.bool_)

    metadata_path = FITTING / "train-metadata.npz"
    np.savez(
        metadata_path,
        schema=np.asarray("slp.yeast-count-transition-fitting/v1"),
        action_ids=row_actions,
        action_index=row_action_index,
        context_index=row_context,
        batch_index=row_batch,
        num_cells=row_cells,
        row_weight=weights,
    )
    reference_path = FITTING / "reference.npz"
    np.savez(
        reference_path,
        schema=np.asarray("slp.yeast-count-transition-reference/v1"),
        query_ids=query_ids,
        context_ids=np.asarray(CONTEXTS),
        batch_ids=batch_ids,
        batch_context_index=batch_context,
        control_mean=controls.astype(np.float32),
        fitting_action_ids=np.asarray(fitting_genes),
        action_features_normalized=action_normalized,
        query_features_normalized=query_normalized,
        feature_mean=feature_mean,
        feature_std=feature_std,
        basal_query_indices=selected_basal,
        basal_value_mean=basal_mean.astype(np.float32),
        basal_value_std=basal_std.astype(np.float32),
        basal_values_normalized=basal_values,
        basal_mask=basal_mask,
        delta_amplitude=amplitude,
        objective_query_scale=scales,
        fitting_residual_centroid=centroids,
    )
    fitting_manifest = {
        "schema": "slp.yeast-count-transition-fitting-manifest/v1",
        "sourceMomentsSha256": HASHES["moments"],
        "quantitativeRowsMaterialized": "development fitting only",
        "validationQuantitativeRowsRead": False,
        "targets": {
            "path": str(target_path.resolve()),
            "sha256": sha256(target_path),
            "shape": [train_rows, QUERIES],
            "dtype": "float32",
        },
        "metadata": {
            "path": str(metadata_path.resolve()),
            "sha256": sha256(metadata_path),
        },
        "reference": {
            "path": str(reference_path.resolve()),
            "sha256": sha256(reference_path),
        },
        "fittingGenes": len(fitting_genes),
        "fittingGenesByContext": [int(value.sum()) for value in observed_genes],
        "rowWeight": {
            "mean": float(weights.mean()),
            "minimum": float(weights.min()),
            "maximum": float(weights.max()),
        },
        "basalTokens": CONTEXT_TOKENS,
        "runtimeSeconds": time.monotonic() - started,
        "peakRssBoundBytes": RSS_CAP,
    }
    write_json(FITTING / "manifest.json", fitting_manifest)
    final_protocol = {
        **preparation_protocol,
        "schema": "slp.yeast-count-transition-pilot-protocol/v1",
        "status": "prepared-before-GPU-profile-or-training",
        "preparationProtocolSha256": sha256(OUTPUT / "preparation-protocol.json"),
        "fittingManifest": {
            "path": str((FITTING / "manifest.json").resolve()),
            "sha256": sha256(FITTING / "manifest.json"),
        },
        "referenceConstruction": {
            "forecastAnchor": "exact matching environment/batch author-WT mean from control-only raw cells",
            "basalTokens": "64 highest population-variance queries across all29 WT batch profiles; controls only",
            "basalValueNormalization": "per selected query mean/population-SD across29 WT batch profiles; SD<=1e-5 replaced1",
            "featureNormalization": "one mean/population-SD over unique fitting genes pooled across environments, applied to actions and queries; SD<=1e-5 replaced1",
            "deltaAmplitude": "per-query equal-environment/equal-gene population SD of cell-weighted gene residual profiles around exact WT batch means; floor.05",
            "objectiveScale": "per-environment per-query population SD across cell-weighted fitting-gene WT-residual profiles; floor.05",
        },
        "evaluation": "cell-weight validation populations to gene profiles; subtract same exact fitting-derived batch-only mean from truth and prediction; independently query-center residual matrices; no after-hoc output projection",
        "baselineReportSha256": HASHES["baselineReport"],
        "correctedBaselineScoringSha256": HASHES["baselineScoring"],
    }
    write_json(OUTPUT / "protocol.json", final_protocol)
    prepared = {
        "protocolSha256": sha256(OUTPUT / "protocol.json"),
        "fittingManifestSha256": sha256(FITTING / "manifest.json"),
        "sourceHashes": {path.name: sha256(path) for path in source.glob("*.py")},
        "validationEvaluations": 0,
    }
    write_json(OUTPUT / "PREPARED.json", prepared)
    return prepared


def cpu_profile() -> dict[str, object]:
    prepared = json.loads((OUTPUT / "PREPARED.json").read_text())
    if sha256(OUTPUT / "protocol.json") != prepared["protocolSha256"]:
        raise ValueError("prepared protocol drift")
    module = load_python(
        OUTPUT / "source/control_transition_model.py", "yeast_profile_core"
    )
    with np.load(FITTING / "reference.npz", allow_pickle=False) as ref:
        model = module.MinimalControlTransition(
            module.Config(
                FEATURES, FEATURES, hidden_dim=128, state_dim=128, dropout=0.2
            )
        )
        actions = torch.zeros((2, FEATURES))
        query = torch.as_tensor(ref["query_features_normalized"])
        batch = np.asarray([0, 14])
        context = torch.as_tensor(ref["batch_context_index"][batch])
        selected = torch.as_tensor(ref["basal_query_indices"], dtype=torch.int64)
        started = time.monotonic()
        output = model(
            actions,
            query,
            torch.as_tensor(ref["control_mean"][batch]),
            torch.as_tensor(ref["delta_amplitude"]),
            torch.as_tensor(ref["objective_query_scale"])[context],
            query[selected],
            torch.as_tensor(ref["basal_values_normalized"][batch]),
            torch.as_tensor(ref["basal_mask"][batch]),
        )
        output["mean"].square().mean().backward()
        elapsed = time.monotonic() - started
    result = {
        "schema": "slp.yeast-count-transition-cpu-shape-profile/v1",
        "device": "cpu",
        "targetFreeSynthetic": True,
        "batch": 2,
        "queries": QUERIES,
        "actionFeatures": FEATURES,
        "queryFeatures": FEATURES,
        "parameters": sum(value.numel() for value in model.parameters()),
        "forwardBackwardSeconds": elapsed,
        "trainingTargetBytes": (FITTING / "train-targets.npy").stat().st_size,
        "estimatedCudaResidentInputsBytes": (38978 * QUERIES + 6683 * 577 + 1516 * 577)
        * 4,
        "actualCudaProfileRun": False,
    }
    write_json(OUTPUT / "cpu-profile.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--profile-cpu", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.profile_cpu:
        parser.error("choose exactly one of --prepare or --profile-cpu")
    value = prepare() if args.prepare else cpu_profile()
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
