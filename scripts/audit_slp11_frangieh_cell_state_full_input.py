#!/usr/bin/env python3
"""No-refit full-input reconstruction diagnostic for the frozen cell-state AE."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results/slp11-transition/frangieh-cell-state-ae-latent-ridge-seed731-v1"
SHARDS = ROOT / "data/derived/slp11-frangieh/paired-singlecell-train-control-v1"
OUTPUT = ROOT / "results/slp11-transition/frangieh-cell-state-full-input-reconstruction-diagnostic-v1"
PARENT_REPORT_SHA = "cada9a66568dda2340a95dd0bbd6b96bcc7af2ac76bf98f9b8f4e1d681bc182f"
SHARD_MANIFEST_SHA = "e791b5cf35da96fa71951a4a240ed58b53e278d3c57e44066680abd3f386a9c7"
BATCH = 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_inference():
    path = ARTIFACT / "source/inference.py"
    spec = importlib.util.spec_from_file_location("full_input_frozen_inference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load artifact inference")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare() -> dict[str, object]:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha256(ARTIFACT / "report.json") != PARENT_REPORT_SHA:
        raise ValueError("parent report drift")
    if sha256(SHARDS / "manifest.json") != SHARD_MANIFEST_SHA:
        raise ValueError("shard manifest drift")
    OUTPUT.mkdir(parents=True)
    protocol = {
        "schema": "slp.frangieh-cell-state-full-input-reconstruction-diagnostic-protocol/v1",
        "question": "How does frozen reconstruction change when all observed inputs are supplied, matching forecast encoding?",
        "modelSelection": "none; frozen parent checkpoint",
        "population": "reconstruction-validation cells from fitting genes and verified controls only",
        "inputs": "all RNA18063 and protein20 channels observed; no denoising masks",
        "metrics": "standardized and raw per-cell-per-query MSE against the same reconstruction-training mean predictor",
        "parentReport": {"path": str(ARTIFACT / "report.json"), "sha256": PARENT_REPORT_SHA},
        "artifactManifestSha256": sha256(ARTIFACT / "artifact-manifest.json"),
        "shardManifestSha256": SHARD_MANIFEST_SHA,
        "sourceSha256": sha256(Path(__file__)),
        "interpretation": "descriptive only; cannot change the selected checkpoint or fixed gates",
        "heldGeneCellsAccessed": False,
    }
    write_json(OUTPUT / "protocol.json", protocol)
    return protocol


def run() -> dict[str, object]:
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    if protocol["sourceSha256"] != sha256(Path(__file__)):
        raise ValueError("diagnostic source drift")
    if sha256(ARTIFACT / "report.json") != PARENT_REPORT_SHA:
        raise ValueError("parent report drift")
    inference = load_inference()
    frozen = inference.FrozenCellState(ARTIFACT, device="cpu")
    reference = frozen.reference
    rna_features = frozen._tensor(reference["rna_query_features"])
    protein_features = frozen._tensor(reference["protein_query_features"])
    totals = {
        key: 0.0
        for key in (
            "rna_std", "protein_std", "rna_raw", "protein_raw",
            "rna_mean_std", "protein_mean_std", "rna_mean_raw", "protein_mean_raw",
        )
    }
    cells = 0
    manifest = json.loads((SHARDS / "manifest.json").read_text())
    for item in manifest["shards"]:
        shard_path = SHARDS / item["path"]
        if sha256(shard_path) != item["sha256"]:
            raise ValueError(f"shard drift: {item['path']}")
        with np.load(shard_path, allow_pickle=False) as archive:
            split = archive["reconstruction_split"]
            matrix = sparse.csr_matrix(
                (archive["rna_data"], archive["rna_indices"], archive["rna_indptr"]),
                shape=tuple(archive["rna_shape"]),
            )
            protein_values = archive["protein_values"]
        selected = np.flatnonzero(split == "validation")
        for offset in range(0, len(selected), BATCH):
            rows = selected[offset : offset + BATCH]
            rna = matrix[rows].toarray().astype(np.float32)
            protein = protein_values[rows].astype(np.float32)
            state = frozen._tensor(frozen.encode(rna, protein))
            with torch.no_grad():
                rna_std_prediction = frozen.model.observe(state, rna_features, "rna").numpy()
                protein_std_prediction = frozen.model.observe(state, protein_features, "protein").numpy()
            rna_std = (rna - reference["rna_mean"]) / reference["rna_sd"]
            protein_std = (protein - reference["protein_mean"]) / reference["protein_sd"]
            rna_prediction = rna_std_prediction * reference["rna_sd"] + reference["rna_mean"]
            protein_prediction = protein_std_prediction * reference["protein_sd"] + reference["protein_mean"]
            for head, raw, standard, raw_prediction, standard_prediction in (
                ("rna", rna, rna_std, rna_prediction, rna_std_prediction),
                ("protein", protein, protein_std, protein_prediction, protein_std_prediction),
            ):
                totals[f"{head}_std"] += float(np.square(standard_prediction - standard).sum())
                totals[f"{head}_raw"] += float(np.square(raw_prediction - raw).sum())
                totals[f"{head}_mean_std"] += float(np.square(standard).sum())
                totals[f"{head}_mean_raw"] += float(np.square(raw - reference[f"{head}_mean"]).sum())
            cells += len(rows)
    result: dict[str, object] = {
        "schema": "slp.frangieh-cell-state-full-input-reconstruction-diagnostic/v1",
        "cells": cells,
        "heads": {},
        "checkpointChanged": False,
        "heldGeneCellsAccessed": False,
    }
    for head, queries in (("rna", 18_063), ("protein", 20)):
        denominator = cells * queries
        model_std = totals[f"{head}_std"] / denominator
        mean_std = totals[f"{head}_mean_std"] / denominator
        model_raw = totals[f"{head}_raw"] / denominator
        mean_raw = totals[f"{head}_mean_raw"] / denominator
        result["heads"][head] = {
            "standardizedMse": model_std,
            "trainingMeanStandardizedMse": mean_std,
            "standardizedFractionalImprovement": 1 - model_std / mean_std,
            "rawMse": model_raw,
            "trainingMeanRawMse": mean_raw,
            "rawFractionalImprovement": 1 - model_raw / mean_raw,
        }
    write_json(OUTPUT / "report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.run:
        raise ValueError("choose exactly one of --prepare or --run")
    result = prepare() if args.prepare else run()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
