"""Bounded Norman 2019 CRISPRa single/double molecular transfer pilot.

This entry point accepts only the mechanically routed development bundle. It
compares a source mean, an additive feature ridge, a random set transition,
and a human-initialized set transition on identical validation records.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from threadpoolctl import threadpool_limits
from transition_model import Config, TransitionWorld, gaussian_loss

TAXON = 9606
SEED = 731
DATA_SHA256 = "ab81e7ed07d7f111b3dfc964cece28a2db7de0dcf5975f6ff1a3bc2db0be683e"
FEATURE_SHA256 = "7b3d78af66f013e2d1df3a3f98924707ed111bc795757753e82a5e8f495408b5"
HUMAN_CHECKPOINT_SHA256 = "40f69aefea1e895fcbfccd89677c3b8df05ef5bfc5ed4b8b1a2c7c8aedfe39f6"
HUMAN_REFERENCE_SHA256 = "73558d5b651b2d51d5ae5534f0467d0ffb27d55cecabd1af2e5dea41db9bbfbc"
RIDGE_ALPHAS = (100.0, 1000.0, 10000.0)
QUERY_COUNT = 7_226
FEATURE_DIM = 577
RESPONSE_DIM = 32


class NormanTrainingError(ValueError):
    """The transfer pilot input or fixed protocol is invalid."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def constituent_split(action_id: str, seed: int = SEED) -> str:
    payload = f"slp11-development-v1|{seed}|{TAXON}|{action_id}".encode()
    bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def unpack_actions(
    action_ids: np.ndarray, offsets: np.ndarray
) -> tuple[tuple[str, ...], ...]:
    if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0:
        raise NormanTrainingError("invalid flattened action offsets")
    if np.any(np.diff(offsets) < 1) or offsets[-1] != len(action_ids):
        raise NormanTrainingError("each intervention record needs one or two actions")
    records = tuple(
        tuple(str(item) for item in action_ids[offsets[i] : offsets[i + 1]])
        for i in range(len(offsets) - 1)
    )
    if any(len(item) not in (1, 2) or list(item) != sorted(set(item)) for item in records):
        raise NormanTrainingError("action sets must contain one or two sorted unique ENSGs")
    return records


def action_feature_tensor(
    records: Sequence[Sequence[str]], lookup: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(records), 2, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((len(records), 2), dtype=np.bool_)
    for row, actions in enumerate(records):
        for column, action in enumerate(actions):
            if action not in lookup:
                raise NormanTrainingError(f"static feature missing for {action}")
            values[row, column] = lookup[action]
            mask[row, column] = True
    return values, mask


def oof_folds(record_ids: Sequence[str], folds: int = 5, seed: int = SEED) -> np.ndarray:
    values = np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(f"norman-oof|{seed}|{item}".encode()).digest()[:8],
                "big",
            )
            % folds
            for item in record_ids
        ],
        dtype=np.int64,
    )
    if set(values.tolist()) != set(range(folds)):
        raise NormanTrainingError("deterministic OOF assignment has an empty fold")
    return values


def ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    mean = x_train.mean(0)
    scale = x_train.std(0)
    scale = np.where(scale > 1e-5, scale, 1.0)
    train = (x_train - mean) / scale
    test = (x_test - mean) / scale
    target_mean = y_train.mean(0)
    centered = y_train - target_mean
    kernel = train @ train.T
    dual = np.linalg.solve(
        kernel + np.eye(len(train), dtype=np.float64) * alpha,
        centered,
    )
    return (target_mean + (test @ train.T) @ dual).astype(np.float32)


def oof_predictions(
    x: np.ndarray,
    y: np.ndarray,
    record_ids: Sequence[str],
    *,
    alpha: float | None,
) -> np.ndarray:
    fold_ids = oof_folds(record_ids)
    result = np.empty_like(y, dtype=np.float32)
    for fold in range(5):
        held = fold_ids == fold
        fit = ~held
        if alpha is None:
            result[held] = y[fit].mean(0)
        else:
            result[held] = ridge_predict(x[fit], y[fit], x[held], alpha)
    return result


def residual_scale(
    target: np.ndarray, prediction: np.ndarray, observed: np.ndarray
) -> np.ndarray:
    squared = np.where(observed, (target - prediction) ** 2, 0.0).sum(0)
    count = observed.sum(0)
    values = np.sqrt(np.divide(squared, count, out=np.zeros_like(squared), where=count > 0))
    return np.maximum(values, 0.05).astype(np.float32)


def validation_strata(records: Sequence[Sequence[str]]) -> dict[str, np.ndarray]:
    action_count = np.asarray([len(item) for item in records])
    held_count = np.asarray(
        [sum(constituent_split(action) == "validation" for action in item) for item in records]
    )
    if np.any(held_count < 1) or np.any(held_count > 2):
        raise NormanTrainingError("validation record lacks a validation constituent")
    return {
        "all": np.arange(len(records), dtype=np.int64),
        "single": np.flatnonzero(action_count == 1),
        "double": np.flatnonzero(action_count == 2),
        "oneHeldConstituent": np.flatnonzero(held_count == 1),
        "twoHeldConstituents": np.flatnonzero(held_count == 2),
    }


def metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    reference: np.ndarray,
    scale: np.ndarray,
    rows: np.ndarray,
) -> dict[str, object]:
    if not len(rows):
        return {"records": 0, "nll": None, "rmse": None, "adjustedPearson": None}
    pred = prediction[rows]
    truth = target[rows]
    mask = observed[rows]
    error = np.where(mask, truth - pred, 0.0)
    count = int(mask.sum())
    rmse = math.sqrt(float(np.square(error).sum()) / count)
    variance = np.maximum(np.square(scale), 1e-8)
    nll_values = 0.5 * (np.log(2 * math.pi * variance)[None, :] + error**2 / variance[None, :])
    nll = float(np.where(mask, nll_values, 0.0).sum() / count)
    correlations: list[float] = []
    for row in rows:
        keep = observed[row]
        actual = target[row, keep] - reference[keep]
        forecast = prediction[row, keep] - reference[keep]
        if actual.std() > 0 and forecast.std() > 0:
            correlations.append(float(np.corrcoef(actual, forecast)[0, 1]))
    return {
        "records": len(rows),
        "nll": nll,
        "rmse": rmse,
        "adjustedPearson": float(np.mean(correlations)) if correlations else None,
    }


def evaluate(
    prediction: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    reference: np.ndarray,
    scale: np.ndarray,
    records: Sequence[Sequence[str]],
) -> dict[str, object]:
    result = {
        name: metrics(prediction, target, observed, reference, scale, rows)
        for name, rows in validation_strata(records).items()
    }
    groups: dict[tuple[str, ...], list[int]] = {}
    for row, actions in enumerate(records):
        groups.setdefault(tuple(actions), []).append(row)
    grouped_prediction = np.zeros((len(groups), target.shape[1]), dtype=np.float32)
    grouped_target = np.zeros_like(grouped_prediction)
    grouped_observed = np.zeros_like(grouped_prediction, dtype=np.bool_)
    for output_row, rows in enumerate(groups.values()):
        index = np.asarray(rows, dtype=np.int64)
        counts = observed[index].sum(0)
        grouped_observed[output_row] = counts > 0
        grouped_prediction[output_row] = np.divide(
            np.where(observed[index], prediction[index], 0.0).sum(0),
            counts,
            out=np.zeros(target.shape[1], dtype=np.float32),
            where=counts > 0,
        )
        grouped_target[output_row] = np.divide(
            np.where(observed[index], target[index], 0.0).sum(0),
            counts,
            out=np.zeros(target.shape[1], dtype=np.float32),
            where=counts > 0,
        )
    result["canonicalActionSet"] = metrics(
        grouped_prediction,
        grouped_target,
        grouped_observed,
        reference,
        scale,
        np.arange(len(groups), dtype=np.int64),
    )
    return result


def train_network(
    *,
    name: str,
    initial_state: dict[str, torch.Tensor] | None,
    actions: np.ndarray,
    action_mask: np.ndarray,
    queries: np.ndarray,
    context_features: np.ndarray,
    context_values: np.ndarray,
    context_mask: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    reference: np.ndarray,
    reference_scale: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, torch.Tensor], dict[str, object]]:
    torch.manual_seed(args.seed)
    config = Config(
        feature_dim=FEATURE_DIM,
        hidden=128,
        state_dim=64,
        covariance_rank=0,
        dropout=0.2,
        learn_scale=False,
        query_feature_dim=FEATURE_DIM + RESPONSE_DIM,
    )
    device = torch.device(args.device)
    model = TransitionWorld(config)
    if initial_state is not None:
        model.load_state_dict(initial_state, strict=True)
    model.to(device)
    torch.manual_seed(args.seed)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    tensors = {
        "actions": torch.as_tensor(actions, dtype=torch.float32, device=device),
        "mask": torch.as_tensor(action_mask, dtype=torch.bool, device=device),
        "queries": torch.as_tensor(queries, dtype=torch.float32, device=device),
        "context_features": torch.as_tensor(
            context_features, dtype=torch.float32, device=device
        ),
        "context_values": torch.as_tensor(
            context_values, dtype=torch.float32, device=device
        ),
        "context_mask": torch.as_tensor(context_mask, dtype=torch.bool, device=device),
        "target": torch.as_tensor(target, dtype=torch.float32, device=device),
        "observed": torch.as_tensor(observed, dtype=torch.bool, device=device),
        "reference": torch.as_tensor(reference, dtype=torch.float32, device=device),
        "scale": torch.as_tensor(reference_scale, dtype=torch.float32, device=device),
    }

    def forward(rows: np.ndarray) -> dict[str, torch.Tensor]:
        index = torch.as_tensor(rows, dtype=torch.long, device=device)
        batch = len(rows)
        return model(
            tensors["actions"][index],
            tensors["queries"],
            tensors["reference"].expand(batch, -1),
            tensors["scale"],
            action_mask=tensors["mask"][index],
            context_features=tensors["context_features"].expand(batch, -1, -1),
            context_values=tensors["context_values"].expand(batch, -1),
            context_mask=tensors["context_mask"].expand(batch, -1),
        )

    def predict(rows: np.ndarray) -> np.ndarray:
        model.eval()
        values = []
        with torch.no_grad():
            for offset in range(0, len(rows), args.batch_size):
                values.append(
                    forward(rows[offset : offset + args.batch_size])["mean"].cpu().numpy()
                )
        return np.concatenate(values)

    generator = np.random.default_rng(args.seed)
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        if time.monotonic() - started > args.per_model_seconds:
            break
        model.train()
        order = generator.permutation(train_rows)
        losses = []
        for offset in range(0, len(order), args.batch_size):
            rows = order[offset : offset + args.batch_size]
            index = torch.as_tensor(rows, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = forward(rows)
            loss = gaussian_loss(
                prediction,
                tensors["target"][index],
                tensors["observed"][index],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name} generated a nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_prediction = predict(validation_rows)
        keep = observed[validation_rows]
        score = float(
            np.mean(
                np.square(validation_prediction - target[validation_rows]),
                where=keep,
            )
        )
        history.append(
            {"epoch": epoch, "trainNll": float(np.mean(losses)), "validationMse": score}
        )
        if score < best_score - 1e-7:
            best_score = score
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise NormanTrainingError(f"{name} produced no checkpoint")
    model.load_state_dict(best_state)
    prediction = predict(validation_rows)
    return prediction, best_state, {
        "config": asdict(config),
        "bestEpoch": best_epoch,
        "bestValidationMse": best_score,
        "epochsRun": len(history),
        "elapsedSeconds": time.monotonic() - started,
        "history": history,
        "initialization": "random" if initial_state is None else "human-CRISPRi-checkpoint",
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    for path, digest, label in (
        (args.data, DATA_SHA256, "Norman development data"),
        (args.features, FEATURE_SHA256, "Norman extended static features"),
        (args.human_checkpoint, HUMAN_CHECKPOINT_SHA256, "human checkpoint"),
        (args.human_reference, HUMAN_REFERENCE_SHA256, "human reference"),
    ):
        if sha256_file(path) != digest:
            raise NormanTrainingError(f"{label} SHA-256 mismatch")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with np.load(args.data, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    if len(data["split_test"]) or not len(data["split_train"]) or not len(
        data["split_validation"]
    ):
        raise NormanTrainingError("only a train/validation development bundle is accepted")
    if data["targets"].shape != (len(data["record_ids"]), QUERY_COUNT):
        raise NormanTrainingError("Norman target shape mismatch")
    records = unpack_actions(data["action_ids"], data["action_offsets"])
    if len(records) != len(data["targets"]):
        raise NormanTrainingError("action records and targets disagree")
    train = data["split_train"].astype(np.int64)
    validation = data["split_validation"].astype(np.int64)
    train_action_union = {item for row in train for item in records[row]}
    validation_held = {
        item
        for row in validation
        for item in records[row]
        if constituent_split(item) == "validation"
    }
    if train_action_union & validation_held:
        raise NormanTrainingError("held validation constituent leaks into fitting records")
    if any(
        any(constituent_split(item) == "test" for item in records[row])
        for row in np.concatenate((train, validation))
    ):
        raise NormanTrainingError("test constituent appears in development data")

    with np.load(args.features, allow_pickle=False) as archive:
        ids = archive["entity_id"].tolist()
        taxa = archive["entity_taxon"]
        feature_values = archive["feature_values"].astype(np.float32)
    if (
        len(ids) != len(set(ids))
        or feature_values.shape != (len(ids), FEATURE_DIM)
        or not np.array_equal(taxa, np.full(len(ids), TAXON, dtype=np.int64))
        or not np.isfinite(feature_values).all()
    ):
        raise NormanTrainingError("static feature pack contract mismatch")
    feature_lookup = dict(zip(ids, feature_values, strict=True))
    action_values, action_mask = action_feature_tensor(records, feature_lookup)
    additive_values = (action_values * action_mask[..., None]).sum(1)

    with np.load(args.human_reference, allow_pickle=False) as archive:
        human = {name: archive[name] for name in archive.files}
    if human["query_ids"].tolist() != data["query_ids"].tolist():
        raise NormanTrainingError("human query descriptors do not align to Norman queries")
    if human["query_features"].shape != (QUERY_COUNT, FEATURE_DIM + RESPONSE_DIM):
        raise NormanTrainingError("human response-query descriptor shape mismatch")
    feature_mean = human["feature_mean"].astype(np.float32)
    feature_std = human["feature_std"].astype(np.float32)
    if feature_mean.shape != (FEATURE_DIM,) or feature_std.shape != (FEATURE_DIM,):
        raise NormanTrainingError("human feature normalization shape mismatch")
    normalized_actions = (action_values - feature_mean) / feature_std
    normalized_actions = np.where(action_mask[..., None], normalized_actions, 0.0)
    queries = (
        human["query_features"] - human["query_feature_mean"]
    ) / human["query_feature_std"]
    selected = human["context_query_indices"].astype(np.int64)
    basal = data["context_basal_expression"][0].astype(np.float32)
    basal_observed = data["observed"].any(0)
    basal_mean = basal[basal_observed].mean()
    basal_std = max(float(basal[basal_observed].std()), 1e-5)
    normalized_basal = (basal - basal_mean) / basal_std
    context_values = normalized_basal[selected][None, :]
    context_mask = basal_observed[selected][None, :]
    context_features = queries[selected][None, :, :]

    y = data["targets"].astype(np.float32)
    observed = data["observed"].astype(np.bool_)
    if not np.isfinite(y[observed]).all():
        raise NormanTrainingError("Norman development targets contain nonfinite values")
    mean_reference = np.divide(
        np.where(observed[train], y[train], 0.0).sum(0),
        observed[train].sum(0),
        out=np.zeros(QUERY_COUNT, dtype=np.float64),
        where=observed[train].sum(0) > 0,
    ).astype(np.float32)
    mean_prediction = np.repeat(mean_reference[None, :], len(validation), axis=0)
    train_record_ids = ["+".join(records[index]) for index in train]
    mean_oof = oof_predictions(
        additive_values[train], y[train], train_record_ids, alpha=None
    )
    mean_scale = residual_scale(y[train], mean_oof, observed[train])

    ridge_candidates: dict[float, np.ndarray] = {}
    ridge_selection: dict[str, object] = {}
    for alpha in RIDGE_ALPHAS:
        prediction = ridge_predict(
            additive_values[train], y[train], additive_values[validation], alpha
        )
        keep = observed[validation]
        rmse = math.sqrt(
            float(np.square(prediction - y[validation])[keep].mean())
        )
        ridge_candidates[alpha] = prediction
        ridge_selection[str(int(alpha))] = {"validationRmse": rmse}
    selected_alpha = min(
        RIDGE_ALPHAS, key=lambda item: ridge_selection[str(int(item))]["validationRmse"]
    )
    ridge_prediction = ridge_candidates[selected_alpha]
    ridge_oof = oof_predictions(
        additive_values[train], y[train], train_record_ids, alpha=selected_alpha
    )
    ridge_scale = residual_scale(y[train], ridge_oof, observed[train])

    if args.device == "cuda" and not torch.cuda.is_available():
        raise NormanTrainingError("CUDA requested but unavailable; no fallback")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    human_state = load_file(str(args.human_checkpoint), device="cpu")
    network_predictions: dict[str, np.ndarray] = {}
    network_reports: dict[str, object] = {}
    network_states: dict[str, dict[str, torch.Tensor]] = {}
    for name, state in (("random", None), ("humanInitialized", human_state)):
        prediction, best_state, training = train_network(
            name=name,
            initial_state=state,
            actions=normalized_actions,
            action_mask=action_mask,
            queries=queries.astype(np.float32),
            context_features=context_features.astype(np.float32),
            context_values=context_values,
            context_mask=context_mask,
            target=y,
            observed=observed,
            train_rows=train,
            validation_rows=validation,
            reference=mean_reference,
            reference_scale=mean_scale,
            args=args,
        )
        network_predictions[name] = prediction
        network_reports[name] = training
        network_states[name] = best_state
        save_file(best_state, str(output / f"{name}-model.safetensors"))

    validation_records = tuple(records[index] for index in validation)
    results = {
        "mean": evaluate(
            mean_prediction,
            y[validation],
            observed[validation],
            mean_reference,
            mean_scale,
            validation_records,
        ),
        "additiveRidge": evaluate(
            ridge_prediction,
            y[validation],
            observed[validation],
            mean_reference,
            ridge_scale,
            validation_records,
        ),
        "setTransitionRandom": evaluate(
            network_predictions["random"],
            y[validation],
            observed[validation],
            mean_reference,
            mean_scale,
            validation_records,
        ),
        "setTransitionHumanInitialized": evaluate(
            network_predictions["humanInitialized"],
            y[validation],
            observed[validation],
            mean_reference,
            mean_scale,
            validation_records,
        ),
    }
    np.savez_compressed(
        output / "validation-predictions.npz",
        record_ids=data["record_ids"][validation],
        mean=mean_prediction,
        additive_ridge=ridge_prediction,
        set_transition_random=network_predictions["random"],
        set_transition_human_initialized=network_predictions["humanInitialized"],
    )
    np.savez_compressed(
        output / "norman-reference.npz",
        reference=mean_reference,
        reference_scale=mean_scale,
        context_query_indices=selected,
        context_features=context_features,
        context_values=context_values,
        context_mask=context_mask,
        basal_mean=np.asarray(basal_mean, dtype=np.float32),
        basal_std=np.asarray(basal_std, dtype=np.float32),
        context_ids=data["context_ids"],
        query_ids=data["query_ids"],
    )
    protocol = {
        "hypothesis": "human CRISPRi initialization improves Norman CRISPRa validation over the identical random set-transition architecture",
        "advancementRule": "human-initialized adjusted Pearson exceeds random initialization overall and does not regress double-intervention adjusted Pearson",
        "scope": "source/perturbation-mode transfer pilot; Norman-only fine-tuning",
        "data": {"path": str(args.data), "sha256": DATA_SHA256},
        "features": {"path": str(args.features), "sha256": FEATURE_SHA256},
        "humanInitialization": {
            "checkpointSha256": HUMAN_CHECKPOINT_SHA256,
            "referenceSha256": HUMAN_REFERENCE_SHA256,
            "quantitativeHumanRecordsMixedIntoFineTuning": False,
        },
        "code": {
            "trainNormanSha256": sha256_file(Path(__file__)),
            "transitionModelSha256": sha256_file(
                Path(__file__).with_name("transition_model.py")
            ),
        },
        "runtime": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "torch", "safetensors", "threadpoolctl")
        },
        "split": {
            "seed": SEED,
            "trainRecords": len(train),
            "validationRecords": len(validation),
            "testRecordsRead": 0,
            "heldConstituentPriority": "test > validation > train",
        },
        "normalization": {
            "targets": str(data["target_value_space"].item()),
            "actionAndQueryFeatures": "human-training-derived reference.npz",
            "ridgeActionFeatures": "sum then Norman-train-only fold standardization",
            "contextSource": str(data["context_value_space"].item()),
            "contextTransform": "within-Norman-context z-score over observed raw basal query values",
        },
        "uncertainty": {
            "mean": "five-fold canonical-action-set-grouped OOF training residual scale",
            "additiveRidge": "five-fold canonical-action-set-grouped OOF training residual scale at selected alpha",
            "networks": "fixed mean-arm OOF training residual scale",
            "minimumScale": 0.05,
            "calibrationScope": "Norman training records only",
        },
        "ridgeAlphaSelection": {
            "candidates": list(RIDGE_ALPHAS),
            "criterion": "development validation masked RMSE",
            "selected": selected_alpha,
            "results": ridge_selection,
        },
        "testDataAccessed": False,
        "benchmarkDataAccessed": False,
    }
    report = {
        "protocol": protocol,
        "results": results,
        "networkTraining": network_reports,
        "elapsedSeconds": time.monotonic() - started,
        "advancementRulePassed": (
            (results["setTransitionHumanInitialized"]["all"]["adjustedPearson"] or -1)
            > (results["setTransitionRandom"]["all"]["adjustedPearson"] or -1)
            and (
                results["setTransitionHumanInitialized"]["double"]["adjustedPearson"]
                or -1
            )
            >= (results["setTransitionRandom"]["double"]["adjustedPearson"] or -1)
        ),
        "checkpointSha256": {
            name: sha256_file(output / f"{name}-model.safetensors")
            for name in ("random", "humanInitialized")
        },
        "referenceSha256": sha256_file(output / "norman-reference.npz"),
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "report.json", report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--features", type=Path, required=True)
    result.add_argument("--human-checkpoint", type=Path, required=True)
    result.add_argument("--human-reference", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--epochs", type=int, default=120)
    result.add_argument("--patience", type=int, default=20)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--learning-rate", type=float, default=0.0005)
    result.add_argument("--weight-decay", type=float, default=0.1)
    result.add_argument("--per-model-seconds", type=float, default=100.0)
    result.add_argument("--seed", type=int, default=SEED)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.seed != SEED:
        raise NormanTrainingError("the pilot is fixed to seed 731")
    with threadpool_limits(limits=4):
        report = run(args)
    print(
        json.dumps(
            {
                "elapsedSeconds": report["elapsedSeconds"],
                "results": report["results"],
                "advancementRulePassed": report["advancementRulePassed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
