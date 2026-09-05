"""Fit neural model-specific uncertainty from fitting-only held-gene OOF residuals.

The frozen normalized human architecture is trained for exactly 20 epochs in
each of three global intervention-gene folds. Outer development-validation rows
are used only to prove exclusion and are never supplied to preprocessing,
training, epoch choice, prediction, or uncertainty fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

SEED = 731
FOLDS = 3
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.1
HIDDEN = 128
STATE_DIM = 64
DROPOUT = 0.2
QUERY_BASIS_RANK = 32
CONTEXT_TOKENS = 64
SCALE_FLOOR = 0.05
DATA_SHA256 = "88de5164fca4e2504ac5b459ab4226c161eb586dd04700d5784da4bb53048659"
FEATURE_SHA256 = "b3de49e18d3c75676985b8790d1ce85de0d87d526bbd7c0c5b555828a1fb11a0"
SOURCE_RUN = Path(
    "results/slp11-transition/human-normalized-fusion-response32-exposure-seed731-v1"
)


class NeuralOofCalibrationError(ValueError):
    """The fitting-only neural OOF calibration contract was violated."""


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def string_sha256(values: Sequence[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode()
    return hashlib.sha256(payload).hexdigest()


def make_fold_plan(
    action_ids: np.ndarray,
    train: np.ndarray,
    outer_validation: np.ndarray,
    fold_ids: np.ndarray,
    *,
    folds: int = FOLDS,
) -> list[dict[str, object]]:
    """Create global train-only folds and prove outer-validation exclusion."""

    action_ids = np.asarray(action_ids).astype(str)
    train = np.asarray(train)
    outer_validation = np.asarray(outer_validation)
    fold_ids = np.asarray(fold_ids)
    if train.ndim != 1 or outer_validation.ndim != 1 or fold_ids.shape != train.shape:
        raise NeuralOofCalibrationError("split and fold arrays must be one-dimensional")
    if any(array.dtype.kind not in "iu" for array in (train, outer_validation, fold_ids)):
        raise NeuralOofCalibrationError("split and fold arrays must contain integers")
    if len(train) == 0 or len(outer_validation) == 0:
        raise NeuralOofCalibrationError("both outer development partitions must be nonempty")
    if len(np.unique(train)) != len(train) or len(np.unique(outer_validation)) != len(
        outer_validation
    ):
        raise NeuralOofCalibrationError("outer split indices must be unique")
    if np.intersect1d(train, outer_validation).size:
        raise NeuralOofCalibrationError("outer validation overlaps fitting rows")
    if np.any(train < 0) or np.any(train >= len(action_ids)):
        raise NeuralOofCalibrationError("training index is out of range")
    if set(fold_ids.tolist()) != set(range(folds)):
        raise NeuralOofCalibrationError("every global OOF fold must be represented")
    local_genes = action_ids[train]
    for gene in np.unique(local_genes):
        if np.unique(fold_ids[local_genes == gene]).size != 1:
            raise NeuralOofCalibrationError(f"intervention gene crosses OOF folds: {gene}")

    plan: list[dict[str, object]] = []
    for fold in range(folds):
        held = train[fold_ids == fold]
        fitting = train[fold_ids != fold]
        if len(held) == 0 or len(fitting) == 0:
            raise NeuralOofCalibrationError("each fold needs fitting and held records")
        if (
            np.intersect1d(held, fitting).size
            or np.intersect1d(held, outer_validation).size
            or np.intersect1d(fitting, outer_validation).size
        ):
            raise NeuralOofCalibrationError("fold includes outer validation or overlaps itself")
        held_genes = sorted(set(action_ids[held].tolist()))
        fitting_genes = sorted(set(action_ids[fitting].tolist()))
        if set(held_genes) & set(fitting_genes):
            raise NeuralOofCalibrationError("held intervention appears in fold fitting rows")
        plan.append(
            {
                "fold": fold,
                "fittingRows": fitting,
                "heldRows": held,
                "fittingRowSha256": array_sha256(fitting.astype("<i8")),
                "heldRowSha256": array_sha256(held.astype("<i8")),
                "fittingGeneSha256": string_sha256(fitting_genes),
                "heldGeneSha256": string_sha256(held_genes),
                "fittingRecords": len(fitting),
                "heldRecords": len(held),
                "fittingInterventions": len(fitting_genes),
                "heldInterventions": len(held_genes),
            }
        )
    if not np.array_equal(np.sort(np.concatenate([item["heldRows"] for item in plan])), np.sort(train)):
        raise NeuralOofCalibrationError("OOF held rows do not partition the fitting split")
    return plan


def collect_oof_predictions(
    total_records: int,
    query_count: int,
    plan: Sequence[dict[str, object]],
    predict_fold: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
) -> np.ndarray:
    """Call one independently fitted predictor per fold and align its held rows."""

    result = np.full((total_records, query_count), np.nan, dtype=np.float32)
    for item in plan:
        fitting = np.asarray(item["fittingRows"], dtype=np.int64)
        held = np.asarray(item["heldRows"], dtype=np.int64)
        fold = int(item["fold"])
        prediction = np.asarray(predict_fold(fitting, held, fold), dtype=np.float32)
        if prediction.shape != (len(held), query_count) or not np.isfinite(prediction).all():
            raise NeuralOofCalibrationError("fold predictor returned invalid held-row means")
        if np.isfinite(result[held]).any():
            raise NeuralOofCalibrationError("a fitting row received multiple OOF predictions")
        result[held] = prediction
    return result


def _load_source_modules(source_run: Path):
    source_dir = (source_run / "source").resolve()
    required = (
        "transition_model.py",
        "transition_calibration.py",
        "transition_baselines.py",
        "response_queries.py",
        "exposure_uncertainty.py",
    )
    if any(not (source_dir / name).is_file() for name in required):
        raise NeuralOofCalibrationError("candidate source snapshot is incomplete")
    sys.path.insert(0, str(source_dir))
    all_sources = sorted(source_dir.glob("*.py"))
    return {
        "model": importlib.import_module("transition_model"),
        "calibration": importlib.import_module("transition_calibration"),
        "response": importlib.import_module("response_queries"),
        "uncertainty": importlib.import_module("exposure_uncertainty"),
        "source_dir": source_dir,
        "hashes": {path.name: sha256(path) for path in all_sources},
    }


def _load_inputs(data_path: Path, feature_path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if sha256(data_path) != DATA_SHA256 or sha256(feature_path) != FEATURE_SHA256:
        raise NeuralOofCalibrationError("development data or fusion feature checksum drift")
    with np.load(data_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    if len(data["split_test"]) or not len(data["split_train"]) or not len(
        data["split_validation"]
    ):
        raise NeuralOofCalibrationError("only the nonempty train/validation development bundle is allowed")
    if not data["observed"][data["split_train"]].all():
        raise NeuralOofCalibrationError("response-query fitting requires complete fitting measurements")
    with np.load(feature_path, allow_pickle=False) as archive:
        keys = list(zip(archive["entity_taxon"].tolist(), archive["entity_id"].tolist()))
        values = archive["feature_values"].astype(np.float32)
    if len(keys) != len(set(keys)) or not np.isfinite(values).all():
        raise NeuralOofCalibrationError("fusion feature identities or values are invalid")
    lookup = dict(zip(keys, values))
    actions = np.stack([lookup[(9606, str(gene))] for gene in data["action_ids"]])
    queries = np.stack([lookup[(9606, str(gene))] for gene in data["query_ids"]])
    return data, actions, queries


def _fit_predict_fold(
    fitting: np.ndarray,
    held: np.ndarray,
    fold: int,
    *,
    data: dict[str, np.ndarray],
    actions: np.ndarray,
    static_queries: np.ndarray,
    modules: dict[str, object],
    device: torch.device,
    max_seconds: int,
    started: float,
    model_seed: int,
    epochs: int,
) -> tuple[np.ndarray, dict[str, object]]:
    calibration = modules["calibration"]
    response = modules["response"]
    uncertainty = modules["uncertainty"]
    model_module = modules["model"]
    y = data["targets"]
    observed = data["observed"]
    context = data["context_index"]
    context_count = len(data["context_ids"])
    action_ids = data["action_ids"].astype(str)
    references, reference_scales = [], []
    local_oof_mean = np.empty_like(y[fitting], dtype=np.float64)
    for context_index in range(context_count):
        context_rows = fitting[context[fitting] == context_index]
        positions = np.flatnonzero(context[fitting] == context_index)
        keys = [(9606, action_ids[row]) for row in context_rows]
        mean, oof = calibration.fit_grouped_oof_mean(
            y[context_rows],
            observed[context_rows],
            keys,
            folds=FOLDS,
            seed=SEED,
            scale_floor=SCALE_FLOOR,
            return_oof=True,
        )
        local_oof_mean[positions] = oof
        references.append(mean.intercept_)
        reference_scales.append(mean.residual_scale_.values)
    references_array = np.stack(references)
    reference_scales_array = np.stack(reference_scales)
    control_args = {
        "control_targets": data["control_targets"],
        "control_observed": data["control_observed"],
        "control_num_cells": data["control_num_cells_filtered"],
        "control_context_index": data["control_context_index"],
    }
    training_exposure = uncertainty.fit_exposure_uncertainty(
        y[fitting] - local_oof_mean,
        observed[fitting],
        data["num_cells_filtered"][fitting],
        context[fitting],
        **control_args,
        scale_floor=SCALE_FLOOR,
    )
    feature_mean = actions[fitting].mean(axis=0, dtype=np.float64)
    feature_std = actions[fitting].std(axis=0, dtype=np.float64)
    feature_std = np.where(feature_std > 1e-5, feature_std, 1.0)
    descriptors, response_info = response.fit_query_response_descriptors(
        y[fitting],
        context[fitting],
        references_array,
        reference_scales_array,
        rank=QUERY_BASIS_RANK,
        seed=model_seed,
    )
    query_features = np.concatenate((static_queries, descriptors), axis=1)
    query_mean = np.concatenate((feature_mean, np.zeros(QUERY_BASIS_RANK)))
    query_std = np.concatenate((feature_std, np.ones(QUERY_BASIS_RANK)))

    basal = data["context_basal_expression"]
    selected = np.argsort(-basal.var(axis=0), kind="stable")[:CONTEXT_TOKENS]
    basal_normalized = (basal - basal.mean(axis=1, keepdims=True)) / np.maximum(
        basal.std(axis=1, keepdims=True), 1e-5
    )
    training_scale = training_exposure.scales(
        data["num_cells_filtered"][fitting], context[fitting]
    )

    torch.manual_seed(model_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(model_seed)
    generator = np.random.default_rng(model_seed)

    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    fitting_actions = tensor((actions[fitting] - feature_mean) / feature_std)
    held_actions = tensor((actions[held] - feature_mean) / feature_std)
    queries = tensor((query_features - query_mean) / query_std)
    target = tensor(y[fitting])
    mask = torch.as_tensor(observed[fitting], dtype=torch.bool, device=device)
    references_tensor = tensor(references_array)
    reference_scales_tensor = tensor(reference_scales_array)
    context_features = queries[selected]
    context_values = tensor(basal_normalized[:, selected])
    scale = tensor(training_scale)
    fitting_context = context[fitting]
    held_context = context[held]
    config = model_module.Config(
        feature_dim=actions.shape[1],
        hidden=HIDDEN,
        state_dim=STATE_DIM,
        covariance_rank=0,
        dropout=DROPOUT,
        learn_scale=False,
        query_feature_dim=query_features.shape[1],
    )
    model = model_module.TransitionWorld(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    def forward(local_rows: np.ndarray) -> dict[str, torch.Tensor]:
        local_context = fitting_context[local_rows]
        prediction = model(
            fitting_actions[local_rows],
            queries,
            references_tensor[local_context],
            reference_scales_tensor[local_context],
            context_features=context_features[None].expand(len(local_rows), -1, -1),
            context_values=context_values[local_context],
            context_mask=torch.ones(
                (len(local_rows), CONTEXT_TOKENS), dtype=torch.bool, device=device
            ),
        )
        prediction["scale"] = scale[local_rows]
        return prediction

    epoch_losses: list[float] = []
    for _epoch in range(epochs):
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError("neural OOF calibration exceeded its fixed wall-time cap")
        model.train()
        losses = []
        order = generator.permutation(len(fitting))
        for offset in range(0, len(order), BATCH_SIZE):
            local_rows = order[offset : offset + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            prediction = forward(local_rows)
            loss = model_module.gaussian_loss(prediction, target[local_rows], mask[local_rows])
            if not torch.isfinite(loss):
                raise FloatingPointError("neural OOF training produced nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_losses.append(float(np.mean(losses)))

    model.eval()
    predictions = []
    with torch.no_grad():
        for offset in range(0, len(held), BATCH_SIZE):
            local = slice(offset, offset + BATCH_SIZE)
            local_actions = held_actions[local]
            local_context = held_context[local]
            prediction = model(
                local_actions,
                queries,
                references_tensor[local_context],
                reference_scales_tensor[local_context],
                context_features=context_features[None].expand(len(local_actions), -1, -1),
                context_values=context_values[local_context],
                context_mask=torch.ones(
                    (len(local_actions), CONTEXT_TOKENS), dtype=torch.bool, device=device
                ),
            )
            predictions.append(prediction["mean"].cpu().numpy())
    prediction_array = np.concatenate(predictions)
    metadata = {
        "fold": fold,
        "epochs": epochs,
        "modelSeed": model_seed,
        "lastTrainingNll": epoch_losses[-1],
        "trainingNllSha256": array_sha256(np.asarray(epoch_losses, dtype="<f8")),
        "featureMeanSha256": array_sha256(feature_mean.astype("<f8")),
        "featureStdSha256": array_sha256(feature_std.astype("<f8")),
        "referenceSha256": array_sha256(references_array.astype("<f8")),
        "responseDescriptorSha256": array_sha256(descriptors.astype("<f4")),
        "responseBasis": response_info,
        "trainingExposureProvenance": training_exposure.component_provenance,
    }
    return prediction_array, metadata


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _serializable_plan(plan: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {key: value for key, value in item.items() if key not in {"fittingRows", "heldRows"}}
        for item in plan
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    source_run = Path(args.source_run)
    data_path = Path(args.data)
    feature_path = Path(args.features)
    modules = _load_source_modules(source_run)
    data, actions, static_queries = _load_inputs(data_path, feature_path)
    train = data["split_train"].astype(np.int64)
    validation = data["split_validation"].astype(np.int64)
    action_ids = data["action_ids"].astype(str)
    fold_ids = modules["calibration"].grouped_fold_ids(
        [(9606, action_ids[row]) for row in train], folds=FOLDS, seed=SEED
    )
    plan = make_fold_plan(action_ids, train, validation, fold_ids)
    planning = {
        "trainingRecords": len(train),
        "trainingInterventions": len(np.unique(action_ids[train])),
        "outerValidationRecordsExcluded": len(validation),
        "outerValidationInterventionsExcluded": len(np.unique(action_ids[validation])),
        "trainingRowSha256": array_sha256(train.astype("<i8")),
        "outerValidationRowSha256": array_sha256(validation.astype("<i8")),
        "folds": _serializable_plan(plan),
    }
    if args.plan_only:
        return {"planOnly": True, "planning": planning}
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA unavailable; no executor fallback")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": "slp.neural-oof-uncertainty-calibration/v1",
        "label": "fitting-only development uncertainty experiment",
        "hypothesis": (
            "Model-specific held-gene OOF residuals reduce inherited mean-model "
            "biological variance while retaining core-control sampling variance."
        ),
        "fixedArchitecture": {
            "featureDim": actions.shape[1],
            "hidden": HIDDEN,
            "stateDim": STATE_DIM,
            "dropout": DROPOUT,
            "queryBasisRank": QUERY_BASIS_RANK,
            "contextTokens": CONTEXT_TOKENS,
            "covarianceRank": 0,
            "learnScale": False,
        },
        "fixedTraining": {
            "epochs": EPOCHS,
            "earlyStopping": False,
            "batchSize": BATCH_SIZE,
            "learningRate": LEARNING_RATE,
            "weightDecay": WEIGHT_DECAY,
            "seedResetPerFold": SEED,
        },
        "outerValidationUse": "identity exclusion check only; no outcome access",
        "uncertainty": (
            "neural held-gene OOF residual biological component plus source core-control "
            "sampling component; no validation scale adjustment"
        ),
        "inputs": {
            "runner": {
                "path": str(Path(__file__)),
                "sha256": sha256(Path(__file__)),
            },
            "data": {"path": str(data_path), "sha256": DATA_SHA256},
            "features": {"path": str(feature_path), "sha256": FEATURE_SHA256},
            "sourceRun": {
                "path": str(source_run),
                "checkpointSha256": sha256(source_run / "model.safetensors"),
                "protocolSha256": sha256(source_run / "protocol.json"),
                "sourceHashes": modules["hashes"],
            },
        },
        "planning": planning,
        "testAccessed": False,
        "benchmarkAccessed": False,
        "runtime": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "scikit-learn", "threadpoolctl")
        },
    }
    _write_json(output / "protocol.json", protocol)
    fold_metadata: list[dict[str, object]] = []

    def predict_fold(fitting: np.ndarray, held: np.ndarray, fold: int) -> np.ndarray:
        prediction, metadata = _fit_predict_fold(
            fitting,
            held,
            fold,
            data=data,
            actions=actions,
            static_queries=static_queries,
            modules=modules,
            device=device,
            max_seconds=args.max_seconds,
            started=started,
            model_seed=SEED,
            epochs=EPOCHS,
        )
        fold_metadata.append(metadata)
        print(
            json.dumps(
                {
                    "event": "fold-complete",
                    "fold": fold,
                    "fittingRecords": len(fitting),
                    "heldRecords": len(held),
                    "seconds": round(time.monotonic() - started, 2),
                }
            ),
            flush=True,
        )
        return prediction

    with threadpool_limits(limits=4):
        all_predictions = collect_oof_predictions(
            len(data["targets"]), data["targets"].shape[1], plan, predict_fold
        )
    if not np.isfinite(all_predictions[train]).all() or np.isfinite(
        all_predictions[validation]
    ).any():
        raise NeuralOofCalibrationError("OOF coverage or outer-validation isolation failed")
    uncertainty = modules["uncertainty"].fit_exposure_uncertainty(
        data["targets"][train] - all_predictions[train],
        data["observed"][train],
        data["num_cells_filtered"][train],
        data["context_index"][train],
        control_targets=data["control_targets"],
        control_observed=data["control_observed"],
        control_num_cells=data["control_num_cells_filtered"],
        control_context_index=data["control_context_index"],
        scale_floor=SCALE_FLOOR,
    )
    inherited_path = source_run / "exposure-uncertainty.npz"
    with np.load(inherited_path, allow_pickle=False) as archive:
        inherited_biological = archive["mean_biological_variance"].astype(np.float64)
        inherited_sampling = archive["mean_sampling_variance"].astype(np.float64)
    if inherited_biological.shape != uncertainty.biological_variance_.shape:
        raise NeuralOofCalibrationError("inherited mean exposure component shape drift")
    variance_comparison: dict[str, object] = {}
    for context_index, context_id in enumerate(data["context_ids"].astype(str)):
        local_counts = data["num_cells_filtered"][
            train[data["context_index"][train] == context_index]
        ]
        exposures = sorted({20.0, 100.0, 200.0, float(np.median(local_counts))})
        scale_ratios = {}
        for exposure in exposures:
            neural_variance = (
                uncertainty.biological_variance_[context_index]
                + uncertainty.sampling_variance_[context_index] / exposure
            )
            inherited_variance = (
                inherited_biological[context_index]
                + inherited_sampling[context_index] / exposure
            )
            ratio = np.sqrt(
                np.maximum(neural_variance, SCALE_FLOOR**2)
                / np.maximum(inherited_variance, SCALE_FLOOR**2)
            )
            scale_ratios[f"{exposure:g}"] = {
                "q05": float(np.quantile(ratio, 0.05)),
                "median": float(np.median(ratio)),
                "q95": float(np.quantile(ratio, 0.95)),
            }
        variance_comparison[context_id] = {
            "trainingCellCountMedian": float(np.median(local_counts)),
            "neuralToInheritedBiologicalVarianceSumRatio": float(
                uncertainty.biological_variance_[context_index].sum()
                / inherited_biological[context_index].sum()
            ),
            "fractionQueriesWithLowerNeuralBiologicalVariance": float(
                np.mean(
                    uncertainty.biological_variance_[context_index]
                    < inherited_biological[context_index]
                )
            ),
            "samplingVarianceMaxAbsDrift": float(
                np.max(
                    np.abs(
                        uncertainty.sampling_variance_[context_index]
                        - inherited_sampling[context_index]
                    )
                )
            ),
            "neuralToInheritedScaleRatioByCellCount": scale_ratios,
        }
    artifact_path = output / "neural-oof-exposure.npz"
    np.savez_compressed(
        artifact_path,
        biological_variance=uncertainty.biological_variance_,
        sampling_variance=uncertainty.sampling_variance_,
        residual_counts=uncertainty.residual_counts_,
        control_counts=uncertainty.control_counts_,
        sampling_from_controls=uncertainty.sampling_from_controls_,
        query_ids=data["query_ids"],
        context_ids=data["context_ids"],
        scale_floor=np.asarray(SCALE_FLOOR),
    )
    report = {
        "schema": "slp.neural-oof-uncertainty-calibration-result/v1",
        "label": "fitting-only development uncertainty experiment",
        "artifact": {
            "path": artifact_path.name,
            "sha256": sha256(artifact_path),
            "bytes": artifact_path.stat().st_size,
        },
        "folds": sorted(fold_metadata, key=lambda item: item["fold"]),
        "componentProvenance": uncertainty.component_provenance,
        "identifiabilityWarning": uncertainty.identifiability_warning,
        "samplingFromControlsFraction": float(uncertainty.sampling_from_controls_.mean()),
        "inheritedMeanExposure": {
            "path": str(inherited_path),
            "sha256": sha256(inherited_path),
        },
        "trainingOnlyVarianceComparison": variance_comparison,
        "elapsedSeconds": time.monotonic() - started,
        "outerValidationUsedForFittingCalibrationOrEpochChoice": False,
        "outerValidationRowsLoadedFromDevelopmentBundleButNeverIndexedForOutcomes": True,
        "testAccessed": False,
        "benchmarkAccessed": False,
    }
    _write_json(output / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/derived/slp11-human/replogle-k562-rpe1-author-normalized-development-v2.npz",
    )
    parser.add_argument(
        "--features",
        default=(
            "data/derived/slp11-human-static-fusion/esm2-t6-plus-go-svd-v1/"
            "human-static-esm-go-features.npz"
        ),
    )
    parser.add_argument("--source-run", default=str(SOURCE_RUN))
    parser.add_argument(
        "--output",
        default="results/slp11-transition/human-normalized-neural-oof-calibration-v1",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-seconds", type=int, default=300)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
