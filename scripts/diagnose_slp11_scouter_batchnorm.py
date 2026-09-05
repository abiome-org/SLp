#!/usr/bin/env python3
"""Compare train/eval normalization behavior on frozen Scouter fitting batches."""

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

CONTEXTS = (
    "replogle-2022-k562-essential-day-6",
    "replogle-2022-rpe1-essential-day-7",
    "replogle-2022-k562-gwps-day-8",
)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_module(path: Path):
    digest = sha256_file(path)
    spec = importlib.util.spec_from_file_location(
        f"slp11_scouter_bn_diagnostic_{digest[:12]}", path.resolve(strict=True)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen Scouter source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summary(values: torch.Tensor) -> dict[str, float]:
    array = values.detach().double()
    return {
        "mean": float(array.mean()),
        "standardDeviation": float(array.std()),
        "rms": float(array.square().mean().sqrt()),
        "maximumAbsolute": float(array.abs().max()),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(2)
    source = args.artifact / "source/scouter_model.py"
    module = load_module(source)
    with np.load(args.data, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    with np.load(args.features, allow_pickle=False) as archive:
        feature_ids = tuple(str(item) for item in archive["entity_id"])
        feature_values = archive["feature_values"]
    with np.load(args.artifact / "reference.npz", allow_pickle=False) as archive:
        reference = {name: archive[name] for name in archive.files}
    with np.load(args.exposure, allow_pickle=False) as archive:
        biological = archive["mean_biological_variance"]
        sampling = archive["mean_sampling_variance"]
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
    context = data["context_index"].astype(np.int64)
    train = data["split_train"].astype(np.int64)
    result_contexts = {}
    for context_index, context_name in enumerate(CONTEXTS):
        rows = train[context[train] == context_index][:256]
        target = torch.as_tensor(data["targets"][rows], dtype=torch.float32)
        observed = torch.as_tensor(data["observed"][rows], dtype=torch.bool)
        variance = (
            biological[context_index]
            + sampling[context_index] / data["num_cells_filtered"][rows, None]
        )
        scale = torch.as_tensor(
            np.sqrt(np.maximum(variance, 0.05**2)), dtype=torch.float32
        )
        local_actions = actions[rows, None, :]
        local_controls = controls[context_index].expand(len(rows), -1)
        checkpoint = args.artifact / f"model-context-{context_index}.safetensors"
        checkpoint_hash_before = sha256_file(checkpoint)
        state = load_file(str(checkpoint), device="cpu")
        model = module.ScouterAdaptedBaseline(
            module.Config(query_dim=7036, action_feature_dim=1156)
        )
        model.load_state_dict(state)
        batch_norm = {
            name: {
                "runningMeanRms": float(value.running_mean.double().square().mean().sqrt()),
                "runningVarianceMinimum": float(value.running_var.min()),
                "runningVarianceMaximum": float(value.running_var.max()),
                "batchesTracked": int(value.num_batches_tracked),
            }
            for name, value in model.named_modules()
            if isinstance(value, torch.nn.BatchNorm1d)
        }
        model.eval()
        with torch.no_grad():
            eval_state = model.control_encoder(local_controls)
            eval_prediction = model(local_actions, local_controls)
            eval_loss = module.gaussian_loss(eval_prediction, target, observed, scale)
        model.load_state_dict(state)
        model.train()
        with torch.no_grad():
            train_state = model.control_encoder(local_controls)
            train_prediction = model(local_actions, local_controls)
            train_loss = module.gaussian_loss(train_prediction, target, observed, scale)
        checkpoint_hash_after = sha256_file(checkpoint)
        result_contexts[context_name] = {
            "fittingRows": len(rows),
            "controlInputVarianceAcrossBatch": float(local_controls.var(dim=0).max()),
            "batchNormBuffersAtSelectedEpoch": batch_norm,
            "trainMode": {
                "fixedExposureNll": float(train_loss),
                "prediction": summary(train_prediction),
                "controlState": summary(train_state),
            },
            "evalMode": {
                "fixedExposureNll": float(eval_loss),
                "prediction": summary(eval_prediction),
                "controlState": summary(eval_state),
            },
            "trainEval": {
                "predictionRmsDifference": float(
                    (train_prediction - eval_prediction).double().square().mean().sqrt()
                ),
                "controlStateRmsDifference": float(
                    (train_state - eval_state).double().square().mean().sqrt()
                ),
            },
            "checkpointUnchanged": checkpoint_hash_before == checkpoint_hash_after,
        }
    report = {
        "schema": "slp.scouter-adapted-normalization-diagnostic/v1",
        "purpose": (
            "Inspect train/eval normalization behavior under the adapted constant pooled-control "
            "input; no model fitting or checkpoint selection"
        ),
        "contexts": result_contexts,
        "interpretationBoundary": (
            "A discrepancy diagnoses normalization mode sensitivity in this adaptation; equality "
            "is an engineering contract and does not establish molecular performance."
        ),
        "accessBoundary": {
            "rowsUsed": "first 256 split_train records per context",
            "validationRowsUsed": False,
            "testRowsInSnapshot": int(data["split_test"].size),
            "externalOutcomesRead": False,
        },
        "source": {
            "modelSha256": sha256_file(source),
            "diagnosticSha256": sha256_file(Path(__file__)),
        },
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--exposure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
