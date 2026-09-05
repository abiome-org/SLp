#!/usr/bin/env python3
"""Prepare and run a fixed-step source3/source4 mean-objective experiment pair."""

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
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "modules/slp-1-1-world-transition-v1"
MODEL = ROOT / (
    "results/slp11-transition/"
    "human-gwps-fixed-context-minimal-control-physical-state128-response32-seed731-v1/"
    "model/source/control_transition_model.py"
)
REFERENCE = MODEL.parents[1] / "reference.npz"
OBJECTIVE = ROOT / "modules/slp-1-1-control-transition-v2/objective_weighting.py"
DATA = ROOT / "data/derived/slp11-human-four-context-v2/development.npz"
FEATURES = ROOT / (
    "data/derived/slp11-human-physical/direct-experiments700-v1/"
    "human-esm-go-physical-features.npz"
)
BASELINE_REPORT = ROOT / (
    "results/slp11-transition/human-four-context-physical-ridge-v2/report.json"
)
BASELINE_PREDICTIONS = BASELINE_REPORT.parent / "predictions.npz"
OUTPUT = ROOT / (
    "results/slp11-transition/"
    "human-source3-vs-four-context-mean-objective-seed731-v2"
)

HASHES = {
    "data": "ffe158aaed370e48d384c2970211bd266ef287630cb5382d56c3f7d6083007cf",
    "features": "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7",
    "baseline_report": "b88fc44c76a99318942d783041352d588388a0473e57211fb4d360f833158a72",
    "baseline_predictions": "0c40ed63c336d5fb1795466693c733711150ec6de84d9fc21585f1d38fe57bc0",
    "model": "fdb4555bd0f7c0a0786539da67048f6985f4ec2f36ef7aa45bd22c7c6bfbb2ef",
    "reference": "a9f3fd2679b5a52e20dddddd427d8664b2c226f2db91bdae1e44a63e66568562",
}
CONTEXTS = (
    "replogle-2022-k562-essential-day-6",
    "replogle-2022-rpe1-essential-day-7",
    "replogle-2022-k562-gwps-day-8",
    "nadig-2025-hepg2-day-7",
)
ARMS = {"source3": 3, "source4": 4}
STEPS = 12_000
BATCH_SIZE = 64
SEED = 731

sys.path.insert(0, str(WORLD))
sys.path.insert(0, str(OBJECTIVE.parent))
from four_context_baselines import collapse_equal_records, point_metrics
from mean_objective import (
    context_query_sd,
    deterministic_shuffled_batches,
    masked_standardized_mse,
)
from objective_weighting import EQUAL_CONTEXT_GENE_V1, training_row_weights


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
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    path.write_text(
        json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def json_default(value: object) -> object:
    """Convert NumPy scalars for a terminal-only JSON echo."""

    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def load_model(path: Path):
    if sha256(path) != HASHES["model"]:
        raise ValueError("frozen transition source SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location("slp11_mean_pair_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen transition source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def initialize_model(module: object, seed: int = SEED) -> torch.nn.Module:
    """Initialize the frozen architecture from one explicit training seed."""

    torch.manual_seed(seed)
    return module.MinimalControlTransition(
        module.Config(1156, 1188, hidden_dim=128, state_dim=128, dropout=0.2)
    )


def load_inputs() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    pinned = {
        "data": DATA,
        "features": FEATURES,
        "baseline_report": BASELINE_REPORT,
        "baseline_predictions": BASELINE_PREDICTIONS,
        "model": MODEL,
        "reference": REFERENCE,
    }
    for name, path in pinned.items():
        if sha256(path) != HASHES[name]:
            raise ValueError(f"{name} SHA-256 mismatch")
    with np.load(DATA, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    with np.load(REFERENCE, allow_pickle=False) as archive:
        reference = {name: archive[name] for name in archive.files}
    with np.load(FEATURES, allow_pickle=False) as archive:
        keys = list(zip(archive["entity_taxon"].tolist(), archive["entity_id"].tolist()))
        rows = archive["feature_values"].astype(np.float32)
    if len(keys) != len(set(keys)) or not np.isfinite(rows).all():
        raise ValueError("physical feature identity/value contract failed")
    lookup = dict(zip(keys, rows))
    try:
        actions = np.stack([lookup[(9606, str(gene))] for gene in data["action_ids"]])
    except KeyError as error:
        raise ValueError(f"physical features miss action {error}") from error
    return data, reference, actions.astype(np.float32, copy=False)


def frozen_reference_audit(
    data: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> dict[str, object]:
    if tuple(data["context_ids"].tolist()) != CONTEXTS or len(data["split_test"]):
        raise ValueError("four-context development identity/split drift")
    if not np.array_equal(reference["query_ids"], data["query_ids"]):
        raise ValueError("frozen query identity/order drift")
    if not np.array_equal(reference["control_mean"], data["basal_control"][:3]):
        raise ValueError("source3 control anchor is not bitwise preserved")
    if not np.array_equal(
        reference["context_features"],
        reference["query_features"][reference["context_query_indices"]],
    ):
        raise ValueError("frozen basal token/query feature contract drift")
    observed = data["context_basal_observed"]
    common = observed.all(axis=0)
    if int(common.sum()) != 6_789:
        raise ValueError("shared four-context control panel drift")
    basal = data["context_basal_expression"]
    means = np.asarray([basal[index, common].mean() for index in range(4)])[:, None]
    stds = np.maximum(
        np.asarray([basal[index, common].std() for index in range(4)])[:, None],
        1e-5,
    )
    normalized = np.where(observed, (basal - means) / stds, 0.0).astype(np.float32)
    selected = reference["context_query_indices"]
    if not np.array_equal(normalized[:3, selected], reference["context_values"]):
        raise ValueError("source3 basal token values are not bitwise preserved")
    if set(data["action_ids"][data["split_train"]]) & set(
        data["action_ids"][data["split_validation"]]
    ):
        raise ValueError("global intervention split leakage")
    return {
        "queryIdsBitwise": True,
        "source3ControlMeanBitwise": True,
        "source3BasalTokenValuesBitwise": True,
        "source3BasalTokenFeaturesBitwise": True,
        "sharedControlQueries": int(common.sum()),
        "contextValues": normalized[:, selected],
    }


def build_reference(
    data: dict[str, np.ndarray],
    frozen: dict[str, np.ndarray],
    objective_scale: np.ndarray,
    context_values: np.ndarray,
) -> dict[str, np.ndarray]:
    result = {
        "feature_mean": frozen["feature_mean"],
        "feature_std": frozen["feature_std"],
        "query_feature_mean": frozen["query_feature_mean"],
        "query_feature_std": frozen["query_feature_std"],
        "query_features": frozen["query_features"],
        "control_mean": data["basal_control"],
        "delta_amplitude": frozen["delta_amplitude"],
        "context_query_indices": frozen["context_query_indices"],
        "context_features": frozen["context_features"],
        "context_values": context_values,
        "context_mask": np.ones_like(context_values, dtype=bool),
        "context_ids": data["context_ids"],
        "query_ids": data["query_ids"],
        "objective_query_scale": objective_scale,
        "hidden_dim": np.asarray(128, dtype=np.int64),
        "state_dim": np.asarray(128, dtype=np.int64),
        "dropout": np.asarray(0.2, dtype=np.float32),
    }
    for name, values in result.items():
        if values.dtype.kind == "f" and not np.isfinite(values).all():
            raise ValueError(f"nonfinite portable reference field {name}")
    return result


def profile(device_name: str, steps: int) -> dict[str, object]:
    """Profile full 7,036-query fitting steps without loading molecular outcomes."""

    module = load_model(MODEL)
    with np.load(REFERENCE, allow_pickle=False) as archive:
        ref = {name: archive[name] for name in archive.files}
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for profile but unavailable")
    device = torch.device(device_name)
    model = initialize_model(module, SEED).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    query = torch.as_tensor(
        (ref["query_features"] - ref["query_feature_mean"])
        / ref["query_feature_std"],
        dtype=torch.float32,
        device=device,
    )
    selected = ref["context_query_indices"]
    action = torch.zeros(BATCH_SIZE, 1156, device=device)
    control = torch.zeros(BATCH_SIZE, len(query), device=device)
    amplitude = torch.as_tensor(ref["delta_amplitude"], device=device)
    scale = torch.ones_like(control)
    basal_values = torch.zeros(BATCH_SIZE, len(selected), device=device)
    basal_mask = torch.ones_like(basal_values, dtype=torch.bool)
    target = torch.zeros_like(control)
    observed = torch.ones_like(control, dtype=torch.bool)
    weight = torch.ones(BATCH_SIZE, device=device)
    torch.cuda.synchronize() if device.type == "cuda" else None
    started = time.monotonic()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            action,
            query,
            control,
            amplitude,
            scale,
            query[selected],
            basal_values,
            basal_mask,
        )
        loss = masked_standardized_mse(
            prediction["mean"], target, observed, scale, weight
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.monotonic() - started
    return {
        "schema": "slp.four-context-mean-profile/v1",
        "device": device_name,
        "syntheticTargetFree": True,
        "batchSize": BATCH_SIZE,
        "queriesPerStep": 7_036,
        "profileSteps": steps,
        "elapsedSeconds": elapsed,
        "secondsPerStep": elapsed / steps,
        "projectedSecondsPer12000StepArm": elapsed / steps * STEPS,
        "fullQueryChoiceAllowed": elapsed / steps * STEPS < 900,
    }


def prepare(output: Path, profile_path: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    profile_report = json.loads(profile_path.read_text(encoding="utf-8"))
    if not profile_report.get("fullQueryChoiceAllowed"):
        raise ValueError("full-query profile exceeds the frozen 900-second arm budget")
    data, reference, actions = load_inputs()
    audit = frozen_reference_audit(data, reference)
    train = data["split_train"]
    scales = context_query_sd(
        data["targets"], data["observed"], data["context_index"], train, 4, floor=0.05
    )
    context_values = audit.pop("contextValues")
    portable = build_reference(data, reference, scales, context_values)
    if not np.array_equal(actions.shape, np.asarray((15_212, 1_156))):
        raise ValueError("four-context action feature shape drift")

    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    source_inputs = {
        "run_pair.py": Path(__file__),
        "control_transition_model.py": MODEL,
        "objective_weighting.py": OBJECTIVE,
        "mean_objective.py": WORLD / "mean_objective.py",
        "four_context_baselines.py": WORLD / "four_context_baselines.py",
        "four_context_mean_inference.py": WORLD / "four_context_mean_inference.py",
        "verify_artifact.py": ROOT / "scripts/verify_slp11_four_context_mean_artifact.py",
    }
    source_hashes = {}
    for name, path in source_inputs.items():
        shutil.copy2(path, source / name)
        source_hashes[name] = sha256(source / name)
    np.savez_compressed(output / "frozen-reference.npz", **portable)
    common = {
        "architecture": {
            "model": "minimal-control-v2",
            "actionFeatures": 1156,
            "queryFeatures": 1188,
            "responseDescriptors": 32,
            "basalTokens": 64,
            "hidden": 128,
            "state": 128,
            "dropout": 0.2,
        },
        "training": {
            "optimizer": "AdamW",
            "learningRate": 0.0005,
            "weightDecay": 0.1,
            "gradientClip": 1.0,
            "batchSize": BATCH_SIZE,
            "optimizerSteps": STEPS,
            "seed": SEED,
            "earlyStopping": False,
            "checkpointSelection": "final step only",
            "querySampling": "all 7036 queries every step",
            "rowSampling": (
                "seeded uniform permutation of fitting rows; full batches only; "
                "cycle tail dropped; fixed global equal-context/equal-gene weights"
            ),
            "objective": (
                "masked row-mean squared error divided by context/query population SD "
                "from fitting rows only, floor 0.05; fixed row weights not minibatch-normalized"
            ),
            "hardSecondsPerArm": 1200,
        },
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": HASHES[name]}
            for name, path in {
                "data": DATA,
                "features": FEATURES,
                "baseline_report": BASELINE_REPORT,
                "baseline_predictions": BASELINE_PREDICTIONS,
                "model": MODEL,
                "reference": REFERENCE,
            }.items()
        },
        "frozenReference": {
            "path": "frozen-reference.npz",
            "sha256": sha256(output / "frozen-reference.npz"),
            "source3ResponseDescriptorsRefit": False,
            "amplitudeRefit": False,
            "identityVocabulary": False,
        },
        "sourceHashes": source_hashes,
        "inputAudit": audit,
        "profile": profile_report,
        "validationUse": "one evaluation after final optimizer step only",
        "uncertainty": "none; point means only",
        "runtimeContract": (
            "known four-context point-mean inference only; exposes mean, molecular delta, "
            "state, basal state and intervention delta; no scale/variance output and no "
            "implied unseen-context API"
        ),
        "hepg2Status": "retired adaptive development",
        "jurkatAccessed": False,
        "testAccessed": False,
        "benchmarkAccessed": False,
    }
    pair_protocol = {
        "schema": "slp.source3-source4-mean-objective-pair-protocol/v2",
        "hypothesis": (
            "adding retired HepG2 fitting molecular outcomes improves adaptive HepG2 "
            "forecasts without regressing any source3 development context"
        ),
        "fixedRule": {
            "adaptiveComparison": {
                "hepg2MseImprovementVsSource3Arm": 0.02,
                "hepg2MseImprovementVsRidge": 0.02,
                "hepg2IndependentPearsonAtLeast": 0.10,
                "hepg2PearsonNonregressionVsSource3AndRidge": True,
                "source3ContextsMseAndPearsonNonregressionVsSource3Arm": True,
            },
            "standaloneWorld": {
                "eachContextMseImprovementVsMeanAndRidge": 0.02,
                "eachContextIndependentPearsonAtLeast": 0.10,
                "eachContextPearsonNonregressionVsRidge": True,
            },
        },
        **common,
    }
    write_json(output / "protocol.json", pair_protocol)
    arm_hashes = {}
    for arm, count in ARMS.items():
        arm_path = output / f"arm-{arm}"
        arm_path.mkdir()
        arm_protocol = {
            "schema": "slp.four-context-mean-objective-arm-protocol/v2",
            "arm": arm,
            "fittingContexts": list(CONTEXTS[:count]),
            "fittingRows": int(np.sum(data["context_index"][train] < count)),
            "validationScoredAfterFinalStep": list(CONTEXTS),
            **common,
        }
        write_json(arm_path / "protocol.json", arm_protocol)
        arm_hashes[arm] = sha256(arm_path / "protocol.json")
    prepared = {
        "output": str(output),
        "pairProtocolSha256": sha256(output / "protocol.json"),
        "armProtocolSha256": arm_hashes,
        "frozenReferenceSha256": sha256(output / "frozen-reference.npz"),
        "sourceHashes": source_hashes,
        "profile": profile_report,
        "preparedOnly": True,
    }
    write_json(output / "PREPARED.json", prepared)
    return prepared


def predict_rows(
    model: torch.nn.Module,
    raw_actions: np.ndarray,
    context: np.ndarray,
    reference: dict[str, np.ndarray],
    device: torch.device,
) -> np.ndarray:
    query = torch.as_tensor(
        (reference["query_features"] - reference["query_feature_mean"])
        / reference["query_feature_std"],
        dtype=torch.float32,
        device=device,
    )
    selected = reference["context_query_indices"]
    basal_features = query[selected]
    outputs = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(raw_actions), 64):
            actions = raw_actions[offset : offset + 64]
            local_context = context[offset : offset + 64]
            normalized = (actions - reference["feature_mean"]) / reference["feature_std"]
            result = model(
                torch.as_tensor(normalized, dtype=torch.float32, device=device),
                query,
                torch.as_tensor(
                    reference["control_mean"][local_context],
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(reference["delta_amplitude"], device=device),
                torch.as_tensor(
                    reference["objective_query_scale"][local_context], device=device
                ),
                basal_features,
                torch.as_tensor(
                    reference["context_values"][local_context], device=device
                ),
                torch.as_tensor(
                    reference["context_mask"][local_context],
                    dtype=torch.bool,
                    device=device,
                ),
            )
            outputs.append(result["mean"].cpu().numpy())
    return np.concatenate(outputs)


def fit_arm(
    arm: str,
    context_count: int,
    data: dict[str, np.ndarray],
    actions: np.ndarray,
    reference: dict[str, np.ndarray],
    device: torch.device,
    output: Path,
    *,
    seed: int = SEED,
) -> dict[str, object]:
    arm_path = output / f"arm-{arm}"
    forbidden = [item for item in arm_path.iterdir() if item.name != "protocol.json"]
    if forbidden:
        raise FileExistsError(f"immutable arm contains non-protocol files: {forbidden}")
    started = time.monotonic()
    module = load_model(output / "source/control_transition_model.py")
    train = data["split_train"]
    train = train[data["context_index"][train] < context_count]
    local_context = data["context_index"][train]
    weights = training_row_weights(
        local_context,
        data["action_ids"][train],
        objective=EQUAL_CONTEXT_GENE_V1,
    ).astype(np.float32)
    train_actions = (
        (actions[train] - reference["feature_mean"]) / reference["feature_std"]
    ).astype(np.float32)
    # The only quantitative outcomes materialized for optimization are fitting rows.
    train_targets = data["targets"][train].astype(np.float32, copy=False)
    train_observed = data["observed"][train]

    torch.use_deterministic_algorithms(True)
    model = initialize_model(module, seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    query = torch.as_tensor(
        (reference["query_features"] - reference["query_feature_mean"])
        / reference["query_feature_std"],
        dtype=torch.float32,
        device=device,
    )
    selected = reference["context_query_indices"]
    basal_features = query[selected]
    amplitude = torch.as_tensor(reference["delta_amplitude"], device=device)
    train_action_tensor = torch.as_tensor(train_actions, device=device)
    target_tensor = torch.as_tensor(train_targets, device=device)
    observed_tensor = torch.as_tensor(train_observed, device=device)
    weight_tensor = torch.as_tensor(weights, device=device)
    scale_tensor = torch.as_tensor(reference["objective_query_scale"], device=device)
    control_tensor = torch.as_tensor(reference["control_mean"], device=device)
    context_value_tensor = torch.as_tensor(reference["context_values"], device=device)
    context_mask_tensor = torch.as_tensor(
        reference["context_mask"], dtype=torch.bool, device=device
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    losses = []
    completed = 0
    model.train()
    batches = deterministic_shuffled_batches(
        np.arange(len(train), dtype=np.int64),
        batch_size=BATCH_SIZE,
        steps=STEPS,
        seed=seed,
    )
    for step, rows in enumerate(batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        contexts = local_context[rows]
        prediction = model(
            train_action_tensor[rows],
            query,
            control_tensor[contexts],
            amplitude,
            scale_tensor[contexts],
            basal_features,
            context_value_tensor[contexts],
            context_mask_tensor[contexts],
        )
        loss = masked_standardized_mse(
            prediction["mean"],
            target_tensor[rows],
            observed_tensor[rows],
            scale_tensor[contexts],
            weight_tensor[rows],
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite standardized mean loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        completed = step
        if step % 1000 == 0:
            print(
                json.dumps(
                    {
                        "event": "step",
                        "arm": arm,
                        "step": step,
                        "recentLoss": float(np.mean(losses[-100:])),
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
        if time.monotonic() - started > 1200:
            write_json(
                arm_path / "INCOMPLETE.json",
                {"reason": "hard 1200-second arm cap", "completedSteps": completed},
            )
            return {"arm": arm, "complete": False, "completedSteps": completed}

    save_file(
        {name: value.detach().cpu() for name, value in model.state_dict().items()},
        str(arm_path / "model.safetensors"),
    )
    np.savez_compressed(arm_path / "reference.npz", **reference)
    probe_rows = []
    for index in range(4):
        candidates = data["split_train"][
            data["context_index"][data["split_train"]] == index
        ]
        if len(candidates) < 2:
            raise ValueError("each context requires two target-free probe actions")
        probe_rows.extend(candidates[:2].tolist())
    probe_rows = np.asarray(probe_rows, dtype=np.int64)
    probe_context = data["context_index"][probe_rows]
    probe_mean = predict_rows(
        model, actions[probe_rows], probe_context, reference, device
    )
    np.savez_compressed(
        arm_path / "target-free-probe.npz",
        raw_action_features=actions[probe_rows],
        context_index=probe_context,
        expected_mean=probe_mean,
    )
    manifest = {
        "schema": "slp.four-context-mean-objective-artifact-manifest/v1",
        "sha256": {
            "model.safetensors": sha256(arm_path / "model.safetensors"),
            "reference.npz": sha256(arm_path / "reference.npz"),
            "target-free-probe.npz": sha256(arm_path / "target-free-probe.npz"),
            "../source/control_transition_model.py": sha256(
                output / "source/control_transition_model.py"
            ),
            "../source/four_context_mean_inference.py": sha256(
                output / "source/four_context_mean_inference.py"
            ),
        },
        "probe": {
            "rows": 8,
            "contexts": [0, 1, 2, 3],
            "molecularOutcomesStored": False,
            "source3ArmHepg2Use": "static action features and control context only",
        },
    }
    write_json(arm_path / "artifact-manifest.json", manifest)
    verify = subprocess.run(
        [
            sys.executable,
            str(output / "source/verify_artifact.py"),
            str(arm_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    reload_audit = json.loads(verify.stdout)
    if time.monotonic() - started > 1200:
        write_json(
            arm_path / "INCOMPLETE.json",
            {"reason": "hard 1200-second arm cap including reload", "completedSteps": completed},
        )
        return {"arm": arm, "complete": False, "completedSteps": completed}

    validation = data["split_validation"]
    row_prediction = predict_rows(
        model,
        actions[validation],
        data["context_index"][validation],
        reference,
        device,
    )
    predictions = {}
    metrics = {}
    with np.load(BASELINE_PREDICTIONS, allow_pickle=False) as baseline:
        for index, name in enumerate(CONTEXTS):
            local = data["context_index"][validation] == index
            genes, collapsed, mask, _ = collapse_equal_records(
                data["action_ids"][validation][local], row_prediction[local], data["observed"][validation][local]
            )
            truth_genes, truth, truth_mask, _ = collapse_equal_records(
                data["action_ids"][validation][local], data["targets"][validation][local], data["observed"][validation][local]
            )
            if (
                not np.array_equal(genes, truth_genes)
                or not np.array_equal(genes, baseline[f"context{index}_action_ids"])
                or not np.array_equal(truth, baseline[f"context{index}_truth"])
                or not np.array_equal(truth_mask, baseline[f"context{index}_observed"])
                or not np.array_equal(mask, truth_mask)
            ):
                raise ValueError(f"collapsed validation contract drift in {name}")
            metrics[name] = point_metrics(
                collapsed,
                truth,
                truth_mask,
                baseline[f"context{index}_fitting_query_scale"],
                baseline[f"context{index}_fitting_target_centroid"],
            )
            predictions[f"context{index}_action_ids"] = genes
            predictions[f"context{index}_world"] = collapsed.astype(np.float32)
    np.savez_compressed(arm_path / "predictions.npz", **predictions)
    if time.monotonic() - started > 1200:
        write_json(
            arm_path / "INCOMPLETE.json",
            {
                "reason": "hard 1200-second arm cap including final scoring",
                "completedSteps": completed,
            },
        )
        return {"arm": arm, "complete": False, "completedSteps": completed}
    report = {
        "schema": "slp.four-context-mean-objective-arm-result/v2",
        "arm": arm,
        "seed": seed,
        "complete": True,
        "optimizerSteps": completed,
        "finalRecentTrainingLoss": float(np.mean(losses[-100:])),
        "fittingRows": len(train),
        "fittingContexts": list(CONTEXTS[:context_count]),
        "validationMetrics": metrics,
        "portableReload": reload_audit,
        "artifacts": {
            "model": {"path": "model.safetensors", "sha256": sha256(arm_path / "model.safetensors")},
            "reference": {"path": "reference.npz", "sha256": sha256(arm_path / "reference.npz")},
            "predictions": {"path": "predictions.npz", "sha256": sha256(arm_path / "predictions.npz")},
            "manifest": {"path": "artifact-manifest.json", "sha256": sha256(arm_path / "artifact-manifest.json")},
            "targetFreeProbe": {"path": "target-free-probe.npz", "sha256": sha256(arm_path / "target-free-probe.npz")},
        },
        "elapsedSeconds": time.monotonic() - started,
        "validationEvaluations": 1,
        "testAccessed": False,
        "benchmarkAccessed": False,
        "uncertaintyClaim": False,
    }
    write_json(arm_path / "report.json", report)
    final_elapsed = time.monotonic() - started
    if final_elapsed > 1200:
        report["complete"] = False
        report["capFailure"] = "hard 1200-second arm cap including report write"
        report["elapsedSeconds"] = final_elapsed
        write_json(arm_path / "report.json", report)
        write_json(
            arm_path / "INCOMPLETE.json",
            {"reason": report["capFailure"], "completedSteps": completed},
        )
    return report


def decide(
    reports: dict[str, dict[str, object]], baseline_report: dict[str, object]
) -> dict[str, object]:
    source3 = reports["source3"]["validationMetrics"]
    source4 = reports["source4"]["validationMetrics"]
    adaptive = {}
    standalone = {}
    for index, name in enumerate(CONTEXTS):
        base = baseline_report["contexts"][name]
        mean_mse = float(base["mean"]["gene_profile_raw_mse"])
        ridge_mse = float(base["ridge"]["gene_profile_raw_mse"])
        ridge_r = float(base["ridge"]["independently_query_centered_profile_pearson"])
        s3_mse = float(source3[name]["gene_profile_raw_mse"])
        s3_r = source3[name]["independently_query_centered_profile_pearson"]
        s4_mse = float(source4[name]["gene_profile_raw_mse"])
        s4_r = source4[name]["independently_query_centered_profile_pearson"]
        s3_r_finite = s3_r is not None and np.isfinite(float(s3_r))
        s4_r_finite = s4_r is not None and np.isfinite(float(s4_r))
        ridge_r_finite = np.isfinite(ridge_r)
        s3_r_value = float(s3_r) if s3_r_finite else float("-inf")
        s4_r_value = float(s4_r) if s4_r_finite else float("-inf")
        if index < 3:
            checks = {
                "mseNonregressionVsSource3Arm": s4_mse <= s3_mse,
                "bothPearsonsFinite": s3_r_finite and s4_r_finite,
                "pearsonNonregressionVsSource3Arm": (
                    s3_r_finite and s4_r_finite and s4_r_value >= s3_r_value
                ),
            }
        else:
            checks = {
                "mseImprovementVsSource3ArmAtLeast002": 1 - s4_mse / s3_mse >= 0.02,
                "mseImprovementVsRidgeAtLeast002": 1 - s4_mse / ridge_mse >= 0.02,
                "allRequiredPearsonsFinite": (
                    s3_r_finite and s4_r_finite and ridge_r_finite
                ),
                "independentPearsonAtLeast010": s4_r_finite and s4_r_value >= 0.10,
                "pearsonNonregressionVsSource3Arm": (
                    s3_r_finite and s4_r_finite and s4_r_value >= s3_r_value
                ),
                "pearsonNonregressionVsRidge": (
                    ridge_r_finite and s4_r_finite and s4_r_value >= ridge_r
                ),
            }
        adaptive[name] = {
            "checks": checks,
            "passed": all(checks.values()),
            "source3ArmMse": s3_mse,
            "source4ArmMse": s4_mse,
            "source3ArmIndependentPearson": s3_r,
            "source4ArmIndependentPearson": s4_r,
            "ridgeMse": ridge_mse,
            "ridgeIndependentPearson": ridge_r,
        }
        world_checks = {
            "mseImprovementVsMeanAtLeast002": 1 - s4_mse / mean_mse >= 0.02,
            "mseImprovementVsRidgeAtLeast002": 1 - s4_mse / ridge_mse >= 0.02,
            "requiredPearsonsFinite": s4_r_finite and ridge_r_finite,
            "independentPearsonAtLeast010": s4_r_finite and s4_r_value >= 0.10,
            "pearsonNonregressionVsRidge": (
                s4_r_finite and ridge_r_finite and s4_r_value >= ridge_r
            ),
        }
        standalone[name] = {
            "checks": world_checks,
            "passed": all(world_checks.values()),
            "meanMse": mean_mse,
            "ridgeMse": ridge_mse,
            "worldMse": s4_mse,
            "ridgeIndependentPearson": ridge_r,
            "worldIndependentPearson": s4_r,
        }
    return {
        "adaptiveComparison": {
            "contexts": adaptive,
            "passed": all(item["passed"] for item in adaptive.values()),
        },
        "standaloneWorld": {
            "contexts": standalone,
            "passed": all(item["passed"] for item in standalone.values()),
        },
    }


def execute(output: Path, device_name: str) -> dict[str, object]:
    prepared = json.loads((output / "PREPARED.json").read_text(encoding="utf-8"))
    if sha256(output / "protocol.json") != prepared["pairProtocolSha256"]:
        raise ValueError("pair protocol changed after freezing")
    for arm in ARMS:
        if sha256(output / f"arm-{arm}/protocol.json") != prepared["armProtocolSha256"][arm]:
            raise ValueError(f"{arm} protocol changed after freezing")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no fallback")
    data, frozen, actions = load_inputs()
    audit = frozen_reference_audit(data, frozen)
    train = data["split_train"]
    scale = context_query_sd(
        data["targets"], data["observed"], data["context_index"], train, 4, floor=0.05
    )
    reference = build_reference(data, frozen, scale, audit.pop("contextValues"))
    with np.load(output / "frozen-reference.npz", allow_pickle=False) as saved:
        if saved.files != list(reference) or any(
            not np.array_equal(saved[name], reference[name]) for name in saved.files
        ):
            raise ValueError("prepared frozen reference changed before fitting")
    device = torch.device(device_name)
    reports = {}
    for arm, count in ARMS.items():
        reports[arm] = fit_arm(arm, count, data, actions, reference, device, output)
        if not reports[arm]["complete"]:
            result = {"schema": "slp.source3-source4-mean-objective-pair-result/v2", "complete": False, "reports": reports}
            write_json(output / "report.json", result)
            return result
    baseline = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    decision = decide(reports, baseline)
    result = {
        "schema": "slp.source3-source4-mean-objective-pair-result/v2",
        "complete": True,
        "decision": decision,
        "arms": {
            arm: {
                "report": f"arm-{arm}/report.json",
                "reportSha256": sha256(output / f"arm-{arm}/report.json"),
                "metrics": reports[arm]["validationMetrics"],
            }
            for arm in ARMS
        },
        "protocol": {"path": "protocol.json", "sha256": sha256(output / "protocol.json")},
        "frozenReference": {"path": "frozen-reference.npz", "sha256": sha256(output / "frozen-reference.npz")},
        "hepg2Status": "retired adaptive development",
        "jurkatAccessed": False,
        "testAccessed": False,
        "benchmarkAccessed": False,
        "uncertaintyClaim": False,
    }
    write_json(output / "report.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--profile", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument("--profile-steps", type=int, default=30)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.profile:
        report = profile(args.device, args.profile_steps)
        destination = args.profile_output or (ROOT / "results/slp11-transition/four-context-mean-objective-profile-v2.json")
        if destination.exists():
            raise FileExistsError(f"immutable profile exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(destination, report)
    elif args.prepare_only:
        if args.profile_output is None:
            raise ValueError("--profile-output is required for protocol preparation")
        report = prepare(args.output.resolve(), args.profile_output.resolve(strict=True))
    else:
        report = execute(args.output.resolve(strict=True), args.device)
    print(json.dumps(report, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
