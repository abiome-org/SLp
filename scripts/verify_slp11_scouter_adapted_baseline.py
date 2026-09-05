#!/usr/bin/env python3
"""Verify frozen adapted-Scouter checkpoints reproduce their saved forecasts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

EXPECTED_CONTEXTS = (
    "replogle-2022-k562-essential-day-6",
    "replogle-2022-rpe1-essential-day-7",
    "replogle-2022-k562-gwps-day-8",
)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_model(path: Path):
    digest = sha256_file(path)
    spec = importlib.util.spec_from_file_location(
        f"slp11_scouter_verify_{digest[:12]}", path.resolve(strict=True)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen adapted Scouter source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(args: argparse.Namespace) -> dict[str, object]:
    artifact = args.artifact
    protocol = json.loads((artifact / "protocol.json").read_text(encoding="utf-8"))
    report = json.loads((artifact / "report.json").read_text(encoding="utf-8"))
    source = artifact / "source/scouter_model.py"
    expected_model_sha = protocol["source"]["model"]["sha256"]
    if sha256_file(source) != expected_model_sha:
        raise ValueError("frozen model source drift")
    model_module = load_model(source)
    with np.load(args.data, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    with np.load(args.features, allow_pickle=False) as archive:
        feature_ids = tuple(str(item) for item in archive["entity_id"])
        feature_values = archive["feature_values"]
    with np.load(artifact / "reference.npz", allow_pickle=False) as archive:
        reference = {name: archive[name] for name in archive.files}
    with np.load(artifact / "development-predictions.npz", allow_pickle=False) as archive:
        saved = {name: archive[name] for name in archive.files}
    validation = data["split_validation"].astype(np.int64)
    context = data["context_index"].astype(np.int64)
    feature_row = {gene: row for row, gene in enumerate(feature_ids)}
    action_values = np.stack(
        [feature_values[feature_row[str(gene)]] for gene in data["action_ids"]]
    )
    actions = torch.as_tensor(
        (action_values - reference["feature_mean"]) / reference["feature_std"],
        dtype=torch.float32,
    )
    controls = torch.as_tensor(
        reference["normalized_context_basal_expression"], dtype=torch.float32
    )
    reloaded = np.empty_like(saved["mean"])
    maximum_error = {}
    for context_index, context_name in enumerate(EXPECTED_CONTEXTS):
        model = model_module.ScouterAdaptedBaseline(
            model_module.Config(query_dim=7036, action_feature_dim=1156)
        ).eval()
        checkpoint = artifact / f"model-context-{context_index}.safetensors"
        if sha256_file(checkpoint) != report["contexts"][context_name]["checkpoint"]["sha256"]:
            raise ValueError("checkpoint SHA-256 drift")
        model.load_state_dict(load_file(str(checkpoint), device="cpu"))
        positions = np.flatnonzero(context[validation] == context_index)
        rows = validation[positions]
        predictions = []
        with torch.no_grad():
            for start in range(0, len(rows), 256):
                chunk = rows[start : start + 256]
                predictions.append(
                    model(
                        actions[chunk, None, :],
                        controls[context_index].expand(len(chunk), -1),
                    ).numpy()
                )
        values = np.concatenate(predictions)
        reloaded[positions] = values
        maximum_error[context_name] = float(
            np.max(np.abs(values - saved["mean"][positions]))
        )
    if (
        not np.array_equal(saved["record_ids"], data["record_ids"][validation])
        or not np.array_equal(saved["action_ids"], data["action_ids"][validation])
        or not np.array_equal(saved["context_index"], context[validation])
        or not np.array_equal(saved["query_ids"], data["query_ids"])
        or max(maximum_error.values()) > 1e-5
    ):
        raise RuntimeError("reloaded forecasts or stable identities disagree")
    result = {
        "schema": "slp.scouter-adapted-verification/v1",
        "artifact": str(artifact),
        "reportSha256": sha256_file(artifact / "report.json"),
        "protocolSha256": sha256_file(artifact / "protocol.json"),
        "predictionSha256": sha256_file(artifact / "development-predictions.npz"),
        "identityArraysExact": True,
        "maximumSourceReloadAbsoluteError": maximum_error,
        "sourceReloadTolerance": 1e-5,
        "modelsContainLearnedGeneIdEmbedding": any(
            isinstance(item, torch.nn.Embedding) for item in model.modules()
        ),
        "accessBoundary": {
            "testRowsInSnapshot": int(data["split_test"].size),
            "externalOutcomesRead": False,
        },
    }
    if result["modelsContainLearnedGeneIdEmbedding"]:
        raise RuntimeError("adapted baseline unexpectedly contains an ID embedding")
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
