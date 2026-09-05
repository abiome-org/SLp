"""Prepare or execute fitting-only OOF uncertainty for the frozen state-128 model."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "modules/slp-1-1-world-transition-v1"
sys.path.insert(0, str(HELPERS))

from minimal_control_oof import (
    advancement_checks,
    array_sha256,
    collect_oof_predictions,
    make_fold_plan,
    pooled_delta_amplitude,
    select_common_context_tokens,
)

DATA = (
    ROOT
    / "data/derived/slp11-human-gwps-fixed-panel-context-v1"
    / "replogle-k562-rpe1-gwps-complete-panel-development-v2-fixed-control-context.npz"
)
FEATURES = (
    ROOT
    / "data/derived/slp11-human-physical/direct-experiments700-v1"
    / "human-esm-go-physical-features.npz"
)
SOURCE = (
    ROOT
    / "results/slp11-transition/human-gwps-fixed-context-minimal-control-physical-state128-response32-seed731-v1/model"
)
OUTPUT = ROOT / "results/slp11-transition/human-gwps-physical-state128-neural-oof-calibration-v1"

SEED = 731
FOLDS = 3
EPOCHS = 61
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.1
HIDDEN_DIM = 128
STATE_DIM = 128
DROPOUT = 0.2
QUERY_BASIS_RANK = 32
CONTEXT_TOKENS = 64
COMMON_CONTEXT_QUERIES = 6789
SCALE_FLOOR = 0.05
MAX_SECONDS = 1800
EXPECTED_CONTEXTS = (
    "replogle-2022-k562-essential-day-6",
    "replogle-2022-rpe1-essential-day-7",
    "replogle-2022-k562-gwps-day-8",
)
EXPECTED = {
    DATA: "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c",
    FEATURES: "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7",
    SOURCE / "report.json": "49333ade99f04d96e9d4c4ccc2fc01c002170b38f02d10f88fdc8559d274203d",
    SOURCE / "model.safetensors": "b1e55f2bcc8a29b6b2467a92ebedfdc1cc80ff8c343a6ab36916d638b9c48cf3",
    SOURCE / "development-predictions.npz": "501384b600c5f90fbe6ea22918777288f048091e71377ce8963cda6bd105039e",
    SOURCE / "reference.npz": "a9f3fd2679b5a52e20dddddd427d8664b2c226f2db91bdae1e44a63e66568562",
    SOURCE / "exposure-uncertainty.npz": "9cf5f4a5352dccaa7cb3d6c84e2123b16b190220a1ef9e03c933a887be6c81dd",
}


class RunnerError(ValueError):
    """Raised when the frozen preparation or execution contract drifts."""


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _load_source_modules() -> dict[str, object]:
    source_dir = SOURCE / "source"
    required = (
        "control_transition_model.py",
        "exposure_uncertainty.py",
        "response_queries.py",
        "transition_baselines.py",
        "transition_calibration.py",
    )
    if any(not (source_dir / name).is_file() for name in required):
        raise RunnerError("frozen candidate source snapshot is incomplete")
    sys.path.insert(0, str(source_dir))
    specification = importlib.util.spec_from_file_location(
        "slp11_oof_control_transition", source_dir / "control_transition_model.py"
    )
    if specification is None or specification.loader is None:
        raise RunnerError("could not load frozen control transition model")
    model = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = model
    specification.loader.exec_module(model)
    return {
        "model": model,
        "uncertainty": importlib.import_module("exposure_uncertainty"),
        "response": importlib.import_module("response_queries"),
        "baselines": importlib.import_module("transition_baselines"),
        "calibration": importlib.import_module("transition_calibration"),
        "sourceHashes": {name: sha256(source_dir / name) for name in required},
    }


def _verify_inputs() -> None:
    for path, expected in EXPECTED.items():
        if sha256(path) != expected:
            raise RunnerError(f"pinned input drift: {path}")


def _load_identity() -> dict[str, np.ndarray]:
    with np.load(DATA, allow_pickle=False) as archive:
        names = (
            "action_ids", "context_ids", "context_basal_expression", "context_basal_observed",
            "context_index", "query_ids", "split_test", "split_train", "split_validation",
        )
        data = {name: archive[name] for name in names}
    if data["split_test"].size or tuple(data["context_ids"].tolist()) != EXPECTED_CONTEXTS:
        raise RunnerError("development split or context roster drift")
    if set(data["action_ids"][data["split_train"]]) & set(
        data["action_ids"][data["split_validation"]]
    ):
        raise RunnerError("outer validation intervention overlaps fitting interventions")
    return data


def _serializable_plan(plan: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {key: value for key, value in item.items() if key not in {"fittingRows", "heldRows"}}
        for item in plan
    ]


def prepare() -> dict[str, object]:
    """Freeze hashes, folds, architecture, training, and decision rule without GPU work."""

    _verify_inputs()
    modules = _load_source_modules()
    identity = _load_identity()
    train = identity["split_train"].astype(np.int64)
    validation = identity["split_validation"].astype(np.int64)
    fold_ids = modules["calibration"].grouped_fold_ids(
        [(9606, str(identity["action_ids"][row])) for row in train], folds=FOLDS, seed=SEED
    )
    plan = make_fold_plan(identity["action_ids"], train, validation, fold_ids, folds=FOLDS)
    selected, _ = select_common_context_tokens(
        identity["context_basal_expression"],
        identity["context_basal_observed"],
        tokens=CONTEXT_TOKENS,
        expected_common=COMMON_CONTEXT_QUERIES,
    )
    with np.load(FEATURES, allow_pickle=False) as archive:
        feature_shape = archive["feature_values"].shape
    if feature_shape[1] != 1156:
        raise RunnerError("physical feature dimension drift")
    if OUTPUT.exists():
        raise FileExistsError(f"immutable calibration preparation already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    source_output = OUTPUT / "source"
    source_output.mkdir()
    owned_sources = (Path(__file__), HELPERS / "minimal_control_oof.py")
    for path in owned_sources:
        shutil.copyfile(path, source_output / path.name)
    fold_plan_path = OUTPUT / "fold-plan.npz"
    np.savez_compressed(
        fold_plan_path,
        train=train,
        outer_validation=validation,
        fold_ids=fold_ids.astype(np.int64),
        context_query_indices=selected,
    )
    planning = {
        "trainingRecords": int(train.size),
        "trainingInterventions": int(np.unique(identity["action_ids"][train]).size),
        "outerValidationRecordsExcluded": int(validation.size),
        "outerValidationInterventionsExcluded": int(
            np.unique(identity["action_ids"][validation]).size
        ),
        "trainingRowSha256": array_sha256(train.astype("<i8")),
        "outerValidationRowSha256": array_sha256(validation.astype("<i8")),
        "contextQueryIndicesSha256": array_sha256(selected.astype("<i8")),
        "folds": _serializable_plan(plan),
    }
    protocol = {
        "schema": "slp.minimal-control-state128-neural-oof-calibration/v1",
        "label": "fitting-only model-specific uncertainty calibration preparation",
        "hypothesis": (
            "Frozen state-128 world means pass the fixed three-context likelihood rule when calibrated "
            "with residual variance from globally held intervention genes."
        ),
        "fixedRule": {
            "perContextMeanNllMinusWorldNllAtLeast": 0.02,
            "perContextRidgeNllMinusWorldNllAtLeast": 0.02,
            "perContextWorldAdjustedProfilePearsonAtLeast": 0.10,
            "requiredInEveryContext": True,
            "meanForecasts": "byte-identical saved frozen candidate means",
        },
        "inputs": {str(path.relative_to(ROOT)): digest for path, digest in EXPECTED.items()},
        "candidate": {
            "reportSha256": EXPECTED[SOURCE / "report.json"],
            "checkpointSha256": EXPECTED[SOURCE / "model.safetensors"],
            "frozenBestEpoch": EPOCHS,
        },
        "outerFolds": {
            "count": FOLDS,
            "seed": SEED,
            "group": "stable intervention gene globally across all three contexts",
            "exclusion": "each held gene is absent from every context in that fold fitting set",
        },
        "foldLocalPreprocessing": {
            "contextMeansAndScales": "per-context grouped inner-OOF mean, seed 731",
            "decoderAmplitude": "pooled per-query RMS from inner-OOF mean residuals; floor 0.05",
            "responseDescriptors": "rank-32 SVD refit on fold fitting responses only",
            "actionNormalization": "fold fitting rows only",
            "measurementScale": (
                "fold-fitting mean-OOF residual biological variance plus source core-control sampling/n"
            ),
            "contextTokens": (
                "64 control-only tokens selected from the exact 6789-query common observed mask; "
                "no perturbation outcome enters token selection"
            ),
        },
        "architecture": {
            "actionFeatureDim": 1156,
            "queryFeatureDim": 1188,
            "hiddenDim": HIDDEN_DIM,
            "stateDim": STATE_DIM,
            "dropout": DROPOUT,
        },
        "training": {
            "epochsPerFold": EPOCHS,
            "earlyStopping": False,
            "seedResetPerFold": SEED,
            "batchSize": BATCH_SIZE,
            "learningRate": LEARNING_RATE,
            "weightDecay": WEIGHT_DECAY,
            "loss": "uniform row mean of observed-query diagonal Gaussian NLL",
            "device": "CUDA only; no fallback",
            "wallTimeCapSeconds": MAX_SECONDS,
        },
        "finalCalibration": (
            "model-specific global held-gene OOF residual biological variance plus frozen core-control "
            "sampling slope; variance scale floor 0.05"
        ),
        "outerValidation": (
            "excluded from every fit, preprocessing step, epoch choice, and calibration; accessed only "
            "after OOF artifact freeze to score byte-identical candidate means"
        ),
        "planning": planning,
        "foldPlanArtifact": {
            "path": fold_plan_path.name,
            "sha256": sha256(fold_plan_path),
        },
        "sourceHashes": {
            **modules["sourceHashes"],
            **{path.name: sha256(source_output / path.name) for path in owned_sources},
        },
        "runtime": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "scikit-learn", "threadpoolctl")
        },
        "testAccessed": False,
        "benchmarkAccessed": False,
        "executionAuthorized": False,
    }
    write_new_json(OUTPUT / "protocol.json", protocol)
    return {
        "prepared": True,
        "output": str(OUTPUT),
        "protocolSha256": sha256(OUTPUT / "protocol.json"),
        "foldPlanSha256": sha256(fold_plan_path),
        "planning": planning,
    }


def _load_execution_inputs() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    with np.load(DATA, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    if data["split_test"].size or not np.all(data["observed"][data["split_train"]]):
        raise RunnerError("execution requires complete development fitting panels and no test rows")
    with np.load(FEATURES, allow_pickle=False) as archive:
        keys = list(zip(archive["entity_taxon"].tolist(), archive["entity_id"].tolist(), strict=True))
        values = archive["feature_values"].astype(np.float32)
    if len(keys) != len(set(keys)) or not np.isfinite(values).all():
        raise RunnerError("feature identity or values drift")
    lookup = dict(zip(keys, values, strict=True))
    actions = np.stack([lookup[(9606, str(item))] for item in data["action_ids"]])
    queries = np.stack([lookup[(9606, str(item))] for item in data["query_ids"]])
    return data, actions, queries


def _fit_predict_fold(
    fitting: np.ndarray,
    held: np.ndarray,
    fold: int,
    *,
    data: dict[str, np.ndarray],
    actions: np.ndarray,
    static_queries: np.ndarray,
    selected_tokens: np.ndarray,
    modules: dict[str, object],
    device: torch.device,
    started: float,
) -> tuple[np.ndarray, dict[str, object]]:
    calibration = modules["calibration"]
    uncertainty_module = modules["uncertainty"]
    response = modules["response"]
    model_module = modules["model"]
    targets = data["targets"]
    observed = data["observed"]
    context = data["context_index"]
    actions_ids = data["action_ids"].astype(str)
    context_count = len(EXPECTED_CONTEXTS)
    references: list[np.ndarray] = []
    reference_scales: list[np.ndarray] = []
    inner_oof_mean = np.empty_like(targets[fitting], dtype=np.float64)
    for context_number in range(context_count):
        context_rows = fitting[context[fitting] == context_number]
        positions = np.flatnonzero(context[fitting] == context_number)
        model, oof = calibration.fit_grouped_oof_mean(
            targets[context_rows],
            observed[context_rows],
            [(9606, actions_ids[row]) for row in context_rows],
            folds=FOLDS,
            seed=SEED,
            scale_floor=SCALE_FLOOR,
            return_oof=True,
        )
        references.append(model.intercept_)
        reference_scales.append(model.residual_scale_.values)
        inner_oof_mean[positions] = oof
    reference = np.stack(references)
    reference_scale = np.stack(reference_scales)
    training_uncertainty = uncertainty_module.fit_exposure_uncertainty(
        targets[fitting] - inner_oof_mean,
        observed[fitting],
        data["num_cells_filtered"][fitting],
        context[fitting],
        control_targets=data["control_targets"],
        control_observed=data["control_observed"],
        control_num_cells=data["control_num_cells_filtered"],
        control_context_index=data["control_context_index"],
        scale_floor=SCALE_FLOOR,
    )
    amplitude = pooled_delta_amplitude(
        targets[fitting], inner_oof_mean, observed[fitting], floor=SCALE_FLOOR
    )
    descriptors, response_info = response.fit_query_response_descriptors(
        targets[fitting],
        context[fitting],
        reference,
        reference_scale,
        rank=QUERY_BASIS_RANK,
        seed=SEED,
    )
    feature_mean = actions[fitting].mean(axis=0)
    feature_std = actions[fitting].std(axis=0)
    feature_std = np.where(feature_std > 1e-5, feature_std, 1.0)
    query_values = np.concatenate((static_queries, descriptors), axis=1)
    query_mean = np.concatenate((feature_mean, np.zeros(QUERY_BASIS_RANK, dtype=np.float32)))
    query_std = np.concatenate((feature_std, np.ones(QUERY_BASIS_RANK, dtype=np.float32)))
    selected_again, basal_normalized = select_common_context_tokens(
        data["context_basal_expression"],
        data["context_basal_observed"],
        tokens=CONTEXT_TOKENS,
        expected_common=COMMON_CONTEXT_QUERIES,
    )
    if not np.array_equal(selected_again, selected_tokens):
        raise RunnerError("fold context-token selection drift")
    scale = training_uncertainty.scales(
        data["num_cells_filtered"][fitting], context[fitting]
    )
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generator = np.random.default_rng(SEED)

    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    fitting_actions = tensor((actions[fitting] - feature_mean) / feature_std)
    held_actions = tensor((actions[held] - feature_mean) / feature_std)
    queries = tensor((query_values - query_mean) / query_std)
    fitting_targets = tensor(targets[fitting])
    fitting_observed = torch.as_tensor(observed[fitting], dtype=torch.bool, device=device)
    control_mean = tensor(data["basal_control"])
    amplitude_tensor = tensor(amplitude)
    scale_tensor = tensor(scale)
    basal_features = queries[selected_tokens]
    basal_values = tensor(basal_normalized[:, selected_tokens])
    basal_mask = torch.as_tensor(
        data["context_basal_observed"][:, selected_tokens], dtype=torch.bool, device=device
    )
    fitting_context = context[fitting]
    held_context = context[held]
    config = model_module.Config(
        action_feature_dim=actions.shape[1],
        query_feature_dim=query_values.shape[1],
        hidden_dim=HIDDEN_DIM,
        state_dim=STATE_DIM,
        dropout=DROPOUT,
    )
    model = model_module.MinimalControlTransition(config).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def forward(local_rows: np.ndarray) -> dict[str, torch.Tensor]:
        local_context = fitting_context[local_rows]
        return model(
            fitting_actions[local_rows],
            queries,
            control_mean[local_context],
            amplitude_tensor,
            scale_tensor[local_rows],
            basal_features,
            basal_values[local_context],
            basal_mask[local_context],
        )

    losses: list[float] = []
    for _epoch in range(EPOCHS):
        if time.monotonic() - started >= MAX_SECONDS:
            raise TimeoutError("neural OOF calibration exceeded 1800-second cap")
        model.train()
        epoch_losses = []
        order = generator.permutation(fitting.size)
        for offset in range(0, fitting.size, BATCH_SIZE):
            local_rows = order[offset : offset + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            prediction = forward(local_rows)
            loss = model_module.gaussian_loss(
                prediction, fitting_targets[local_rows], fitting_observed[local_rows]
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite OOF training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))
    model.eval()
    held_prediction: list[np.ndarray] = []
    held_scale = tensor(
        training_uncertainty.scales(data["num_cells_filtered"][held], held_context)
    )
    with torch.no_grad():
        for offset in range(0, held.size, BATCH_SIZE):
            local = slice(offset, offset + BATCH_SIZE)
            local_context = held_context[local]
            result = model(
                held_actions[local],
                queries,
                control_mean[local_context],
                amplitude_tensor,
                held_scale[local],
                basal_features,
                basal_values[local_context],
                basal_mask[local_context],
            )
            held_prediction.append(result["mean"].cpu().numpy())
    return np.concatenate(held_prediction), {
        "fold": fold,
        "epochs": EPOCHS,
        "lastTrainingNll": losses[-1],
        "trainingNllSha256": array_sha256(np.asarray(losses, dtype="<f8")),
        "featureMeanSha256": array_sha256(feature_mean.astype("<f8")),
        "featureStdSha256": array_sha256(feature_std.astype("<f8")),
        "referenceSha256": array_sha256(reference.astype("<f8")),
        "responseDescriptorSha256": array_sha256(descriptors.astype("<f4")),
        "responseBasis": response_info,
        "trainingExposureProvenance": training_uncertainty.component_provenance,
    }


def _gene_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    observed: np.ndarray,
    actions: np.ndarray,
    reference: np.ndarray,
    scale: np.ndarray,
    evaluate: object,
    value_space: str,
) -> dict[str, object]:
    groups: dict[str, list[int]] = {}
    for row, action in enumerate(actions.astype(str)):
        groups.setdefault(action, []).append(row)
    reports = [
        evaluate(prediction[rows], truth[rows], observed[rows], reference, scale[rows], value_space=value_space)
        for rows in groups.values()
    ]
    result = evaluate(prediction, truth, observed, reference, scale, value_space=value_space)
    for metric in ("nll", "profile_centroid_adjusted_pearson_mean"):
        values = [item[metric] for item in reports if np.isfinite(item[metric])]
        result["gene_macro_" + metric] = float(np.mean(values)) if values else None
    result["intervention_genes"] = len(groups)
    return result


def execute() -> dict[str, object]:
    """Execute the already prepared CUDA-only OOF calibration."""

    started = time.monotonic()
    _verify_inputs()
    protocol_path = OUTPUT / "protocol.json"
    plan_path = OUTPUT / "fold-plan.npz"
    if not protocol_path.is_file() or not plan_path.is_file():
        raise RunnerError("prepare must freeze protocol and fold plan before execution")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__)) != protocol["sourceHashes"][Path(__file__).name]:
        raise RunnerError("runner changed after protocol freeze")
    if (OUTPUT / "report.json").exists():
        raise FileExistsError("immutable OOF calibration result already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no executor fallback")
    modules = _load_source_modules()
    data, actions, static_queries = _load_execution_inputs()
    train = data["split_train"].astype(np.int64)
    validation = data["split_validation"].astype(np.int64)
    with np.load(plan_path, allow_pickle=False) as archive:
        if not np.array_equal(archive["train"], train) or not np.array_equal(
            archive["outer_validation"], validation
        ):
            raise RunnerError("prepared outer split drift")
        fold_ids = archive["fold_ids"]
        selected_tokens = archive["context_query_indices"]
    plan = make_fold_plan(data["action_ids"], train, validation, fold_ids, folds=FOLDS)
    device = torch.device("cuda")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    metadata: list[dict[str, object]] = []

    def predict_fold(fitting: np.ndarray, held: np.ndarray, fold: int) -> np.ndarray:
        prediction, details = _fit_predict_fold(
            fitting,
            held,
            fold,
            data=data,
            actions=actions,
            static_queries=static_queries,
            selected_tokens=selected_tokens,
            modules=modules,
            device=device,
            started=started,
        )
        metadata.append(details)
        print(json.dumps({"event": "fold-complete", "fold": fold}), flush=True)
        return prediction

    with threadpool_limits(limits=4):
        predictions = collect_oof_predictions(
            len(data["targets"]), data["targets"].shape[1], plan, predict_fold
        )
    if not np.isfinite(predictions[train]).all() or np.isfinite(predictions[validation]).any():
        raise RunnerError("OOF coverage or validation isolation failed")
    uncertainty = modules["uncertainty"].fit_exposure_uncertainty(
        data["targets"][train] - predictions[train],
        data["observed"][train],
        data["num_cells_filtered"][train],
        data["context_index"][train],
        control_targets=data["control_targets"],
        control_observed=data["control_observed"],
        control_num_cells=data["control_num_cells_filtered"],
        control_context_index=data["control_context_index"],
        scale_floor=SCALE_FLOOR,
    )
    artifact_path = OUTPUT / "neural-oof-exposure.npz"
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
    with np.load(SOURCE / "development-predictions.npz", allow_pickle=False) as archive:
        frozen_mean = archive["mean"]
        frozen_record_ids = archive["record_ids"]
        frozen_context = archive["context_index"]
    if not np.array_equal(frozen_record_ids, data["record_ids"][validation]) or not np.array_equal(
        frozen_context, data["context_index"][validation]
    ):
        raise RunnerError("frozen candidate validation prediction alignment drift")
    frozen_mean_before = array_sha256(frozen_mean)
    validation_scale = uncertainty.scales(
        data["num_cells_filtered"][validation], data["context_index"][validation]
    ).astype(np.float32)
    calibrated_path = OUTPUT / "calibrated-development-predictions.npz"
    np.savez_compressed(
        calibrated_path,
        mean=frozen_mean,
        scale=validation_scale,
        record_ids=frozen_record_ids,
        context_index=frozen_context,
    )
    with np.load(calibrated_path, allow_pickle=False) as archive:
        saved_mean = archive["mean"]
    if array_sha256(frozen_mean) != frozen_mean_before or not np.array_equal(
        saved_mean, frozen_mean
    ):
        raise RunnerError("frozen full-model means changed during calibration")
    candidate_report = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    results: dict[str, object] = {}
    passed: list[bool] = []
    value_space = str(data["target_value_space"].item())
    for context_number, context_name in enumerate(EXPECTED_CONTEXTS):
        local = data["context_index"][validation] == context_number
        rows = validation[local]
        reference = data["targets"][
            train[data["context_index"][train] == context_number]
        ].mean(axis=0, dtype=np.float64)
        metrics = _gene_metrics(
            frozen_mean[local],
            data["targets"][rows],
            data["observed"][rows],
            data["action_ids"][rows],
            reference,
            validation_scale[local],
            modules["baselines"].evaluate,
            value_space,
        )
        controls = candidate_report["results"][context_name]
        checks = advancement_checks(
            float(metrics["gene_macro_nll"]),
            float(controls["mean"]["gene_macro_nll"]),
            float(controls["ridge"]["gene_macro_nll"]),
            float(metrics["gene_macro_profile_centroid_adjusted_pearson_mean"]),
        )
        passed.append(all(checks.values()))
        results[context_name] = {"world": metrics, "checks": checks, "passed": all(checks.values())}
    report = {
        "schema": "slp.minimal-control-state128-neural-oof-calibration-result/v1",
        "results": results,
        "passedEveryContext": all(passed),
        "folds": sorted(metadata, key=lambda item: item["fold"]),
        "artifact": {"path": artifact_path.name, "sha256": sha256(artifact_path)},
        "calibratedPredictions": {
            "path": calibrated_path.name,
            "sha256": sha256(calibrated_path),
            "meanArraySha256": frozen_mean_before,
            "meansBitExact": True,
        },
        "componentProvenance": uncertainty.component_provenance,
        "samplingFromControlsFraction": float(uncertainty.sampling_from_controls_.mean()),
        "elapsedSeconds": time.monotonic() - started,
        "outerValidationUsedForFittingPreprocessingCalibrationOrEpochChoice": False,
        "testAccessed": False,
        "benchmarkAccessed": False,
    }
    write_new_json(OUTPUT / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    result = prepare() if arguments.prepare_only else execute()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
