#!/usr/bin/env python3
"""Evaluate a portable joint-world checkpoint on existing development data.

This is an adaptive development evaluation.  It reads no protected test data,
fits no parameters, and writes every forecast used in the report.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RESPONSE_DATA = ROOT / "data/derived/slp11-omf2-response-v1/development"
NORMAN_DATA = ROOT / "data/derived/slp11-joint-world-populations-v1/norman.npz"
RANK32 = ROOT / "results/slp11-transition/human-essential-count-response-rank32-seed731-v1"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def independently_query_center(value: np.ndarray) -> np.ndarray:
    """Match response-omf2: subtract row zero, then each query's row mean."""
    array = np.asarray(value, dtype=np.float64)
    shifted = array - array[:1]
    return shifted - shifted.mean(axis=0, keepdims=True)


def row_pearson(left: np.ndarray, right: np.ndarray) -> tuple[float | None, int]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    finite = denominator > 1e-12
    if not finite.any():
        return None, 0
    values = np.sum(left[finite] * right[finite], axis=1) / denominator[finite]
    return float(values.mean()), int(finite.sum())


def response_metrics(truth: np.ndarray, prediction: np.ndarray, anchor: np.ndarray) -> dict[str, object]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    if truth.shape != prediction.shape or truth.shape != anchor.shape:
        raise ValueError("response metric arrays must align")
    correlation, count = row_pearson(
        independently_query_center(truth - anchor),
        independently_query_center(prediction - anchor),
    )
    return {
        "geneProfileMse": float(np.square(truth - prediction).mean()),
        "independentlyQueryCenteredResidualPearson": correlation,
        "finiteCorrelationGenes": count,
    }


def composition_metrics(
    truth: np.ndarray, prediction: np.ndarray, observed_additive: np.ndarray
) -> dict[str, object]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    additive = np.asarray(observed_additive, dtype=np.float64)
    if truth.shape != prediction.shape or truth.shape != additive.shape:
        raise ValueError("composition metric arrays must align")
    correlation, count = row_pearson(prediction - additive, truth - additive)
    centered_correlation, centered_count = row_pearson(
        independently_query_center(prediction - additive),
        independently_query_center(truth - additive),
    )
    return {
        "mse": float(np.square(prediction - truth).mean()),
        "nonadditivePearson": correlation,
        "finiteNonadditiveRows": count,
        "independentlyQueryCenteredNonadditivePearson": centered_correlation,
        "finiteIndependentlyQueryCenteredNonadditiveRows": centered_count,
    }


def load_bundle(modeldir: Path, checkpoint: str, device: str):
    inference = modeldir / "inference.py"
    if not inference.is_file():
        raise FileNotFoundError(f"portable inference module missing: {inference}")
    sys.path.insert(0, str(modeldir))
    try:
        spec = importlib.util.spec_from_file_location("slp11_joint_world_bundle", inference)
        if spec is None or spec.loader is None:
            raise ImportError(inference)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.JointWorldBundle(modeldir, checkpoint, device)
    finally:
        sys.path.pop(0)


def _single_action(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    actions = np.zeros((len(features), 2, features.shape[1]), dtype=np.float32)
    mask = np.zeros((len(features), 2), dtype=np.bool_)
    actions[:, 0] = features
    mask[:, 0] = True
    return actions, mask


def evaluate_response(bundle, response_data: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    reports: dict[str, object] = {}
    forecasts: dict[str, np.ndarray] = {}
    for context in ("k562", "rpe1"):
        data = load_npz(response_data / f"{context}.npz")
        rank = load_npz(RANK32 / f"development-forecast-{context}.npz")
        query_ids = data["query_ids"].astype(str)
        if not np.array_equal(bundle.query_ids(context), query_ids):
            raise ValueError(f"{context} bundle query axis mismatch")
        if not np.array_equal(rank["query_ids"].astype(str), query_ids):
            raise ValueError(f"{context} rank32 query axis mismatch")
        if not np.array_equal(rank["gene_ids"].astype(str), data["gene_ids"].astype(str)):
            raise ValueError(f"{context} rank32 intervention axis mismatch")
        actions, mask = _single_action(data["features"])
        anchor = np.asarray(data["control_prediction"], dtype=np.float64)
        prediction = bundle.predict(context, actions, mask, anchor)
        truth = np.asarray(data["truth"], dtype=np.float64)
        ridge = np.asarray(data["static_ridge_prediction"], dtype=np.float64)
        rank32 = np.asarray(rank["rank32_prediction"], dtype=np.float64)
        reports[context] = {
            "jointWorld": response_metrics(truth, prediction, anchor),
            "staticRidge": response_metrics(truth, ridge, anchor),
            "retainedRank32": response_metrics(truth, rank32, anchor),
            "populations": len(truth),
            "queries": truth.shape[1],
        }
        forecasts.update({
            f"{context}_query_ids": query_ids,
            f"{context}_joint_world": prediction,
            f"{context}_static_ridge": ridge,
            f"{context}_retained_rank32": rank32,
            f"{context}_truth": truth,
            f"{context}_control": anchor,
        })
    return reports, forecasts


def _prior_only(bundle, context: str, actions: np.ndarray, mask: np.ndarray, observed: np.ndarray) -> np.ndarray:
    prior = bundle.priors[context]
    flat = prior.predict(actions.reshape(-1, actions.shape[-1])).reshape(
        len(actions), actions.shape[1], observed.shape[1]
    )
    return np.asarray(observed, dtype=np.float64) + (flat * mask[..., None]).sum(axis=1)


def _predicted_additive(left: np.ndarray, right: np.ndarray, basal: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    basal = np.asarray(basal, dtype=np.float64)
    if left.shape != right.shape or left.shape != basal.shape:
        raise ValueError("predicted singles and basal must align")
    return left + right - basal


def evaluate_norman(bundle, fold: int, norman_data: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    data = load_npz(norman_data)
    if not np.array_equal(bundle.query_ids("norman"), data["query_ids"].astype(str)):
        raise ValueError("Norman bundle query axis mismatch")
    selected = np.flatnonzero(data["combination_fold"] == fold)
    if not len(selected):
        raise ValueError(f"Norman fold {fold} is empty")
    rows = data["combination_rows"][selected]
    parents = data["combination_single_rows"][selected]
    left_rows, right_rows = parents[:, 0], parents[:, 1]
    common = np.asarray(data["combination_common_query_mask"][selected], dtype=np.bool_).all(axis=0)
    if int(common.sum()) != 7182:
        raise ValueError(f"Norman common query support changed: {int(common.sum())}")
    basal = np.asarray(data["basal"][rows], dtype=np.float32)
    truth = np.asarray(data["targets"][rows], dtype=np.float64)
    left = np.asarray(data["targets"][left_rows], dtype=np.float32)
    right = np.asarray(data["targets"][right_rows], dtype=np.float32)
    pair_actions = np.asarray(data["action_features"][rows], dtype=np.float32)
    pair_mask = np.asarray(data["action_mask"][rows], dtype=np.bool_)
    left_actions, one_mask = _single_action(pair_actions[:, 0])
    right_actions, _ = _single_action(pair_actions[:, 1])

    observed_left_parent = bundle.predict("norman", right_actions, one_mask, basal, observed=left)
    observed_right_parent = bundle.predict("norman", left_actions, one_mask, basal, observed=right)
    observed_parent_average = 0.5 * (observed_left_parent + observed_right_parent)
    direct_two_actions = bundle.predict("norman", pair_actions, pair_mask, basal)
    predicted_left = bundle.predict("norman", left_actions, one_mask, basal)
    predicted_right = bundle.predict("norman", right_actions, one_mask, basal)
    autonomous_left_then_right = bundle.predict(
        "norman", right_actions, one_mask, basal, observed=predicted_left
    )
    autonomous_right_then_left = bundle.predict(
        "norman", left_actions, one_mask, basal, observed=predicted_right
    )
    autonomous_average = 0.5 * (autonomous_left_then_right + autonomous_right_then_left)
    observed_additive = left.astype(np.float64) + right.astype(np.float64) - basal.astype(np.float64)
    prior_only = _prior_only(bundle, "norman", pair_actions, pair_mask, basal)
    observed_parent_prior = 0.5 * (
        _prior_only(bundle, "norman", right_actions, one_mask, left)
        + _prior_only(bundle, "norman", left_actions, one_mask, right)
    )
    predicted_additive = _predicted_additive(predicted_left, predicted_right, basal)
    values = {
        "observedParentAThenB": observed_left_parent,
        "observedParentBThenA": observed_right_parent,
        "observedParentAverage": observed_parent_average,
        "directTwoActions": direct_two_actions,
        "autonomousAThenB": autonomous_left_then_right,
        "autonomousBThenA": autonomous_right_then_left,
        "autonomousAverage": autonomous_average,
        "observedAdditive": observed_additive,
        "priorOnly": prior_only,
        "observedParentPrior": observed_parent_prior,
        "predictedAdditive": predicted_additive,
    }
    metrics = {
        name: composition_metrics(truth[:, common], prediction[:, common], observed_additive[:, common])
        for name, prediction in values.items()
    }
    report = {
        "fold": fold,
        "heldCombinations": len(selected),
        "commonQueries": int(common.sum()),
        "metrics": metrics,
        "relativeMse": {
            "autonomousAverageVsObservedAdditive": 1.0 - metrics["autonomousAverage"]["mse"] / metrics["observedAdditive"]["mse"],
            "autonomousAverageVsPriorOnly": 1.0 - metrics["autonomousAverage"]["mse"] / metrics["priorOnly"]["mse"],
            "observedParentAverageVsDirectTwoActions": 1.0 - metrics["observedParentAverage"]["mse"] / metrics["directTwoActions"]["mse"],
            "observedParentAverageVsObservedParentPrior": 1.0 - metrics["observedParentAverage"]["mse"] / metrics["observedParentPrior"]["mse"],
            "autonomousAverageVsPredictedAdditive": 1.0 - metrics["autonomousAverage"]["mse"] / metrics["predictedAdditive"]["mse"],
        },
    }
    arrays: dict[str, np.ndarray] = {
        "norman_query_ids": data["query_ids"].astype(str),
        "norman_common_query_mask": common,
        "norman_combination_rows": rows,
        "norman_combination_single_rows": parents,
        "norman_truth": truth,
        "norman_basal": basal,
        "norman_observed_single_a": left,
        "norman_observed_single_b": right,
    }
    arrays.update({f"norman_{name}": value for name, value in values.items()})
    return report, arrays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--response-data", type=Path, default=RESPONSE_DATA)
    parser.add_argument("--norman-data", type=Path, default=NORMAN_DATA)
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {args.output}")
    checkpoint_path = args.model / "checkpoints" / args.checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    bundle = load_bundle(args.model, args.checkpoint, args.device)
    fold = int(bundle.settings.get("training", {}).get("norman_fold", 0))
    response, response_arrays = evaluate_response(bundle, args.response_data)
    norman, norman_arrays = evaluate_norman(bundle, fold, args.norman_data)
    args.output.mkdir(parents=True)
    prediction_path = args.output / "predictions.npz"
    np.savez_compressed(prediction_path, **response_arrays, **norman_arrays)
    report = {
        "schema": "slp.joint-world-development-evaluation/v1",
        "interpretation": "Adaptive evaluation on existing K562, RPE1 and Norman development outcomes; not independent confirmation.",
        "trainingOrFitting": False,
        "protectedTestAccessed": False,
        "checkpoint": {
            "name": args.checkpoint,
            "sha256": sha256(checkpoint_path),
            "device": args.device,
        },
        "sources": response,
        "norman": norman,
        "predictions": {"path": prediction_path.name, "sha256": sha256(prediction_path)},
    }
    write_json(args.output / "report.json", report)
    print(json.dumps({"checkpointSha256": report["checkpoint"]["sha256"], "sources": response, "norman": norman}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
