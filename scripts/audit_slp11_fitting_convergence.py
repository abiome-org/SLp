#!/usr/bin/env python3
"""Audit fitting convergence of frozen human and yeast transition pilots."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/slp11-transition/fitting-convergence-audit-v3"
HUMAN_DATA = ROOT / (
    "data/derived/slp11-human-gwps-fixed-panel-context-v1/"
    "replogle-k562-rpe1-gwps-complete-panel-development-v2-fixed-control-context.npz"
)
PHYSICAL = ROOT / (
    "data/derived/slp11-human-physical/direct-experiments700-v1/"
    "human-esm-go-physical-features.npz"
)
BP = ROOT / (
    "data/derived/slp11-human-go-bp/"
    "goa-2022-09-19-ensembl108-source3-fit-svd128-v1/"
    "human-go-bp-source3-fit-svd128-features.npz"
)
BP_ROOT = ROOT / (
    "results/slp11-transition/"
    "human-source3-bp-neural-mean-pair-seed731-v2-finalization-v1"
)
FIXED_ROOT = ROOT / (
    "results/slp11-transition/human-source3-bp-fixed-response-basis-seed731-v2"
)
RIDGE_ROOT = ROOT / ("results/slp11-transition/human-gwps-bp-ridge-source3-seed731-v2")
YEAST_ROOT = ROOT / "results/slp11-transition/yeast-rna-world-transition-seed731-v1"
YEAST_FIT = ROOT / (
    "data/derived/slp11-yeast-atlas-counts/nadal-ribelles-rna-neural-fitting-v1"
)
YEAST_RIDGE = ROOT / "results/slp11-transition/yeast-raw-count-batch-ridge-v1"
SEED = 731
SAMPLE_GENES = 256

PINS = {
    "humanData": "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c",
    "physical": "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7",
    "bp": "b29cbd70f08e227cddfc013e66cd1032212c8cb62e6e25162965a57101cd1fac",
    "bpModel": "f1e0acf79c5326d4553ee77f45ccaa0d02628042672413d7089a17991e5d99fc",
    "fixedModel": "d073e4d66bb498dbbc2048f656b90da069318ff8a736860c03436e37a58cc693",
    "yeastModel": "88fd54046458663035ca5b4f05c483d4a0d1a13f99b29e7aa6f10fe43714324d",
    "yeastTargets": "020d980d1384edbbe63fbe72789e104b3807955bb2d164f21472a0fadfb3a93d",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def deterministic_gene_sample(
    genes: np.ndarray, *, count: int, seed: int, label: str
) -> np.ndarray:
    """Select stable genes without using molecular values."""

    unique = np.unique(np.asarray(genes).astype(str))
    if not len(unique):
        raise ValueError("gene sample requires at least one identity")
    ranked = sorted(
        unique,
        key=lambda gene: hashlib.sha256(f"{seed}|{label}|{gene}".encode()).digest(),
    )
    return np.asarray(ranked[: min(count, len(ranked))])


def equal_context_gene_weights(contexts: np.ndarray, genes: np.ndarray) -> np.ndarray:
    contexts = np.asarray(contexts, dtype=np.int64)
    genes = np.asarray(genes).astype(str)
    weights = np.empty(len(genes), dtype=np.float64)
    for context in np.unique(contexts):
        rows = np.flatnonzero(contexts == context)
        _, inverse, counts = np.unique(
            genes[rows], return_inverse=True, return_counts=True
        )
        weights[rows] = len(genes) / (
            len(np.unique(contexts)) * len(counts) * counts[inverse]
        )
    return weights


def sampled_objective(
    prediction: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    scale: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    safe = np.where(observed, np.square((prediction - target) / scale), 0.0)
    per_row = safe.sum(1, dtype=np.float64) / observed.sum(1)
    exact = float(np.mean(per_row * weights, dtype=np.float64))
    normalized = float(np.sum(per_row * weights) / np.sum(weights))
    return {"exactFrozenWeightMean": exact, "sampleWeightNormalized": normalized}


def _optional_mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def _pearson(x: np.ndarray, y: np.ndarray, tolerance: float = 1e-12) -> float | None:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    left -= left.mean()
    right -= right.mean()
    denominator = np.sqrt(np.dot(left, left) * np.dot(right, right))
    if not np.isfinite(denominator) or denominator <= tolerance:
        return None
    return float(np.dot(left, right) / denominator)


def collapse_profiles(
    genes: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    row_mass: np.ndarray,
    reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    unique = np.unique(np.asarray(genes).astype(str))
    pred = np.zeros((len(unique), prediction.shape[1]), dtype=np.float64)
    truth = np.zeros_like(pred)
    mask = np.zeros_like(pred, dtype=np.bool_)
    ref = np.zeros_like(pred) if reference is not None else None
    for index, gene in enumerate(unique):
        rows = np.flatnonzero(genes == gene)
        mass = row_mass[rows, None] * observed[rows]
        total = mass.sum(0)
        mask[index] = total > 0
        pred[index] = np.divide(
            (prediction[rows] * mass).sum(0),
            total,
            out=np.zeros(pred.shape[1]),
            where=mask[index],
        )
        truth[index] = np.divide(
            (target[rows] * mass).sum(0),
            total,
            out=np.zeros(pred.shape[1]),
            where=mask[index],
        )
        if ref is not None:
            ref[index] = np.divide(
                (reference[rows] * mass).sum(0),
                total,
                out=np.zeros(pred.shape[1]),
                where=mask[index],
            )
    return pred, truth, mask, ref


def profile_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    reference: np.ndarray | None = None,
) -> dict[str, object]:
    pred = prediction if reference is None else prediction - reference
    truth = target if reference is None else target - reference
    valid = observed
    mse = float(np.square(pred - truth)[valid].mean(dtype=np.float64))
    pred_centered = pred - pred[0]
    truth_centered = truth - truth[0]
    counts = valid.sum(0)
    pred_mean = np.divide(
        np.where(valid, pred_centered, 0).sum(0),
        counts,
        out=np.zeros(pred.shape[1]),
        where=counts > 0,
    )
    truth_mean = np.divide(
        np.where(valid, truth_centered, 0).sum(0),
        counts,
        out=np.zeros(pred.shape[1]),
        where=counts > 0,
    )
    correlations = [
        _pearson(
            pred_centered[index, valid[index]] - pred_mean[valid[index]],
            truth_centered[index, valid[index]] - truth_mean[valid[index]],
        )
        for index in range(len(pred))
    ]
    return {
        "geneProfileMse": mse,
        "independentlyQueryCenteredPearson": _optional_mean(correlations),
        "undefinedGenes": sum(value is None for value in correlations),
        "genes": len(pred),
        "observedValues": int(valid.sum()),
    }


def parameter_delta(
    initial: dict[str, torch.Tensor], final: dict[str, torch.Tensor]
) -> dict[str, dict[str, float | int]]:
    groups = {
        "action": "action_encoder.",
        "context": "context_encoder.",
        "transition": "transition.",
        "query": "query_encoder.",
        "head": "mean_state.",
    }
    result = {}
    for label, prefix in groups.items():
        names = [name for name in final if name.startswith(prefix)]
        if not names:
            continue
        init_sq = delta_sq = 0.0
        maximum = 0.0
        changed = elements = 0
        for name in names:
            before = initial[name].detach().cpu().double()
            after = final[name].detach().cpu().double()
            delta = after - before
            init_sq += float(torch.sum(before * before))
            delta_sq += float(torch.sum(delta * delta))
            maximum = max(maximum, float(torch.max(torch.abs(delta))))
            changed += int(torch.count_nonzero(delta))
            elements += delta.numel()
        init_norm = np.sqrt(init_sq)
        delta_norm = np.sqrt(delta_sq)
        result[label] = {
            "initialL2": init_norm,
            "deltaL2": delta_norm,
            "deltaOverInitial": delta_norm / init_norm if init_norm else None,
            "maximumAbsoluteDelta": maximum,
            "changedElements": changed,
            "elements": elements,
        }
    return result


def _human_initializer(core) -> torch.nn.Module:
    torch.manual_seed(SEED)
    old = core.MinimalControlTransition(
        core.Config(1156, 1188, hidden_dim=128, state_dim=128, dropout=0.2)
    )
    old_state = old.state_dict()
    torch.manual_seed(SEED)
    new = core.MinimalControlTransition(
        core.Config(1285, 1188, hidden_dim=128, state_dim=128, dropout=0.2)
    )
    with torch.no_grad():
        for name, target in new.state_dict().items():
            source = old_state[name]
            if target.shape == source.shape:
                target.copy_(source)
            elif name == "action_encoder.0.weight":
                target.zero_()
                target[:, :1156].copy_(source)
            else:
                raise ValueError(f"unexpected expanded initializer tensor: {name}")
    return new


def _fixed_initializer(fixed_core, learned_core) -> torch.nn.Module:
    learned = _human_initializer(learned_core)
    torch.manual_seed(SEED)
    fixed = fixed_core.FixedQueryTransition(
        fixed_core.Config(1285, 1188, state_dim=128, hidden_dim=128, dropout=0.2)
    )
    with torch.no_grad():
        for name, target in fixed.state_dict().items():
            target.copy_(learned.state_dict()[name])
    return fixed


def _human_forward(
    model,
    actions: np.ndarray,
    contexts: np.ndarray,
    reference: dict[str, np.ndarray],
    device: torch.device,
    coordinates: np.ndarray | None,
) -> np.ndarray:
    model.eval()
    normalized = (
        (actions - reference["feature_mean"]) / reference["feature_std"]
    ).astype(np.float32)
    query = (
        (reference["query_features"] - reference["query_feature_mean"])
        / reference["query_feature_std"]
    ).astype(np.float32)
    selected = reference["context_query_indices"]
    query_tensor = torch.as_tensor(query, device=device)
    coordinate_tensor = (
        torch.as_tensor(coordinates, device=device)
        if coordinates is not None
        else query_tensor
    )
    chunks = []
    with torch.no_grad():
        for start in range(0, len(actions), 192):
            stop = min(start + 192, len(actions))
            local = contexts[start:stop]
            chunks.append(
                model(
                    torch.as_tensor(normalized[start:stop], device=device),
                    coordinate_tensor,
                    torch.as_tensor(reference["control_mean"][local], device=device),
                    torch.as_tensor(reference["delta_amplitude"], device=device),
                    torch.as_tensor(
                        reference["objective_query_scale"][local], device=device
                    ),
                    query_tensor[
                        torch.as_tensor(selected, dtype=torch.int64, device=device)
                    ],
                    torch.as_tensor(reference["context_values"][local], device=device),
                    torch.as_tensor(
                        reference["context_mask"][local],
                        dtype=torch.bool,
                        device=device,
                    ),
                )["mean"]
                .cpu()
                .numpy()
            )
    return np.concatenate(chunks)


def _human_ridge(
    raw_features: np.ndarray, context: int
) -> tuple[np.ndarray, np.ndarray]:
    path = RIDGE_ROOT / f"model-physical1156_bp128_present1-context-{context}.npz"
    with np.load(path, allow_pickle=False) as state:
        standardized = (raw_features - state["feature_mean"]) / state["feature_scale"]
        rotated = standardized @ state["eigenvectors"]
        alpha = float(state["selected_alpha"].item())
        ridge = (
            state["target_mean"]
            + (rotated / (state["eigenvalues"] + alpha)) @ state["rhs"]
        )
        mean = np.broadcast_to(state["target_mean"], ridge.shape).copy()
    return ridge.astype(np.float32), mean.astype(np.float32)


def audit_human(
    samples: dict[int, np.ndarray], device: torch.device
) -> dict[str, object]:
    with np.load(HUMAN_DATA, allow_pickle=False) as archive:
        action_ids = archive["action_ids"].astype(str)
        contexts = archive["context_index"].astype(np.int64)
        train = archive["split_train"].astype(np.int64)
        context_ids = archive["context_ids"].astype(str)
        # The compressed member contains all development rows; only pre-pinned
        # split_train indices participate in every computation below.
        all_targets = archive["targets"]
        all_observed = archive["observed"]
        rows = np.concatenate(
            [
                train[
                    (contexts[train] == context)
                    & np.isin(action_ids[train], samples[context])
                ]
                for context in range(3)
            ]
        )
        targets = np.asarray(all_targets[rows], dtype=np.float32)
        observed = np.asarray(all_observed[rows], dtype=np.bool_)
    with np.load(PHYSICAL, allow_pickle=False) as archive:
        physical = {
            gene: value
            for gene, value in zip(
                archive["entity_id"].astype(str), archive["feature_values"], strict=True
            )
        }
    with np.load(BP, allow_pickle=False) as archive:
        bp = {
            gene: (value, present)
            for gene, value, present in zip(
                archive["entity_id"].astype(str),
                archive["feature_values"],
                archive["annotation_present"],
                strict=True,
            )
        }
    raw = np.stack(
        [
            np.concatenate((physical[gene], bp[gene][0], [bp[gene][1]]))
            for gene in action_ids[rows]
        ]
    ).astype(np.float32)
    weights_all = equal_context_gene_weights(contexts[train], action_ids[train])
    weight_lookup = dict(zip(train.tolist(), weights_all.tolist(), strict=True))
    weights = np.asarray([weight_lookup[int(row)] for row in rows])

    learned_core = load_module(
        BP_ROOT / "source/control_transition_model.py", "fit_audit_bp_core"
    )
    fixed_core = load_module(
        FIXED_ROOT / "source/transition_model.py", "fit_audit_fixed_core"
    )
    learned_initial = _human_initializer(learned_core)
    learned_final = _human_initializer(learned_core)
    learned_state = load_file(str(BP_ROOT / "bp128-present/model.safetensors"))
    learned_final.load_state_dict(learned_state)
    fixed_initial = _fixed_initializer(fixed_core, learned_core)
    fixed_final = _fixed_initializer(fixed_core, learned_core)
    fixed_state = load_file(str(FIXED_ROOT / "model.safetensors"))
    fixed_final.load_state_dict(fixed_state)
    learned_reference = _load_npz(BP_ROOT / "bp128-present/reference.npz")
    fixed_reference = _load_npz(FIXED_ROOT / "reference.npz")
    coordinates = fixed_reference["fixed_query_coordinates"]

    predictions = {
        "learnedQueryInitializer": _human_forward(
            learned_initial.to(device),
            raw,
            contexts[rows],
            learned_reference,
            device,
            None,
        ),
        "learnedQueryFinal": _human_forward(
            learned_final.to(device),
            raw,
            contexts[rows],
            learned_reference,
            device,
            None,
        ),
        "fixedBasisInitializer": _human_forward(
            fixed_initial.to(device),
            raw,
            contexts[rows],
            fixed_reference,
            device,
            coordinates,
        ),
        "fixedBasisFinal": _human_forward(
            fixed_final.to(device),
            raw,
            contexts[rows],
            fixed_reference,
            device,
            coordinates,
        ),
    }
    result = {}
    for context, name in enumerate(context_ids):
        local = contexts[rows] == context
        ridge, mean = _human_ridge(raw[local], context)
        local_predictions = {
            key: value[local] for key, value in predictions.items()
        } | {"ridge": ridge, "mean": mean}
        values = {}
        for label, prediction in local_predictions.items():
            objective = sampled_objective(
                prediction,
                targets[local],
                observed[local],
                learned_reference["objective_query_scale"][context],
                weights[local],
            )
            collapsed = collapse_profiles(
                action_ids[rows][local],
                prediction,
                targets[local],
                observed[local],
                np.ones(local.sum()),
            )
            values[label] = {
                "sampledTrainingObjective": objective,
                "geneCollapsedRaw": profile_metrics(*collapsed[:3]),
            }
        result[name] = {
            "sampleGenes": samples[context].tolist(),
            "sampleRows": int(local.sum()),
            "models": values,
        }
    result["parameterMovement"] = {
        "learnedQuery": parameter_delta(learned_initial.state_dict(), learned_state),
        "fixedBasis": parameter_delta(fixed_initial.state_dict(), fixed_state),
    }
    result["savedTrainingHistory"] = {
        "learnedQuery": "not persisted; no first/last trajectory can be reconstructed",
        "fixedBasis": {
            "fullRunFinalRecentLoss": 0.8255207252502441,
            "historyPersisted": False,
        },
    }
    return result


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _yeast_forward(
    model,
    action_index: np.ndarray,
    batch_index: np.ndarray,
    reference: dict[str, np.ndarray],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    query = torch.as_tensor(reference["query_features_normalized"], device=device)
    action = torch.as_tensor(reference["action_features_normalized"], device=device)
    control = torch.as_tensor(reference["control_mean"], device=device)
    amplitude = torch.as_tensor(reference["delta_amplitude"], device=device)
    scale = torch.as_tensor(reference["objective_query_scale"], device=device)
    basal_index = torch.as_tensor(
        reference["basal_query_indices"], dtype=torch.int64, device=device
    )
    basal = torch.as_tensor(reference["basal_values_normalized"], device=device)
    basal_mask = torch.as_tensor(
        reference["basal_mask"], dtype=torch.bool, device=device
    )
    batch_context = torch.as_tensor(
        reference["batch_context_index"], dtype=torch.int64, device=device
    )
    chunks = []
    with torch.no_grad():
        for start in range(0, len(action_index), 192):
            stop = min(start + 192, len(action_index))
            local_action = torch.as_tensor(
                action_index[start:stop], dtype=torch.int64, device=device
            )
            local_batch = torch.as_tensor(
                batch_index[start:stop], dtype=torch.int64, device=device
            )
            context = batch_context[local_batch]
            chunks.append(
                model(
                    action[local_action],
                    query,
                    control[local_batch],
                    amplitude,
                    scale[context],
                    query[basal_index],
                    basal[local_batch],
                    basal_mask[local_batch],
                )["mean"]
                .cpu()
                .numpy()
            )
    return np.concatenate(chunks)


def _yeast_baselines(
    genes: np.ndarray,
    action_index: np.ndarray,
    batch_index: np.ndarray,
    context: int,
    reference: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = "control" if context == 0 else "nacl"
    ridge_reference = _load_npz(YEAST_RIDGE / f"{prefix}-reference.npz")
    model = _load_npz(YEAST_RIDGE / f"{prefix}-batch.npz")
    raw = (
        reference["action_features_normalized"][action_index] * reference["feature_std"]
        + reference["feature_mean"]
    )
    normalized = (raw - ridge_reference["feature_mean"]) / ridge_reference[
        "feature_scale"
    ]
    batch_labels = reference["batch_ids"][batch_index].astype(str)
    model_lookup = {
        label: index for index, label in enumerate(model["batch_ids"].astype(str))
    }
    offsets = np.stack(
        [model["batch_offsets"][model_lookup[label]] for label in batch_labels]
    )
    means = np.stack(
        [model["batch_only_means"][model_lookup[label]] for label in batch_labels]
    )
    ridge = normalized @ model["coefficients"] + offsets
    control = reference["control_mean"][batch_index]
    return (
        ridge.astype(np.float32),
        means.astype(np.float32),
        control.astype(np.float32),
    )


def audit_yeast(
    samples: dict[int, np.ndarray], device: torch.device
) -> dict[str, object]:
    metadata = _load_npz(YEAST_FIT / "train-metadata.npz")
    reference = _load_npz(YEAST_ROOT / "reference.npz")
    selected = np.concatenate(
        [
            np.flatnonzero(
                (metadata["context_index"] == context)
                & np.isin(metadata["action_ids"].astype(str), samples[context])
            )
            for context in range(2)
        ]
    )
    target_memmap = np.load(YEAST_FIT / "train-targets.npy", mmap_mode="r")
    targets = np.asarray(target_memmap[selected], dtype=np.float32)
    observed = np.ones_like(targets, dtype=np.bool_)
    core = load_module(
        YEAST_ROOT / "source/control_transition_model.py", "fit_audit_yeast_core"
    )
    torch.manual_seed(SEED)
    initial = core.MinimalControlTransition(
        core.Config(577, 577, hidden_dim=128, state_dim=128, dropout=0.2)
    )
    final = core.MinimalControlTransition(
        core.Config(577, 577, hidden_dim=128, state_dim=128, dropout=0.2)
    )
    final_state = load_file(str(YEAST_ROOT / "model.safetensors"))
    final.load_state_dict(final_state)
    predictions = {
        "initializer": _yeast_forward(
            initial.to(device),
            metadata["action_index"][selected],
            metadata["batch_index"][selected],
            reference,
            device,
        ),
        "final": _yeast_forward(
            final.to(device),
            metadata["action_index"][selected],
            metadata["batch_index"][selected],
            reference,
            device,
        ),
    }
    result = {}
    for context, name in enumerate(reference["context_ids"].astype(str)):
        local = metadata["context_index"][selected] == context
        local_genes = metadata["action_ids"][selected][local].astype(str)
        ridge, mean, control = _yeast_baselines(
            local_genes,
            metadata["action_index"][selected][local],
            metadata["batch_index"][selected][local],
            context,
            reference,
        )
        local_predictions = {
            key: value[local] for key, value in predictions.items()
        } | {"ridge": ridge, "mean": mean}
        values = {}
        for label, prediction in local_predictions.items():
            objective = sampled_objective(
                prediction,
                targets[local],
                observed[local],
                reference["objective_query_scale"][context],
                metadata["row_weight"][selected][local],
            )
            collapsed = collapse_profiles(
                local_genes,
                prediction,
                targets[local],
                observed[local],
                metadata["num_cells"][selected][local].astype(np.float64),
                control,
            )
            values[label] = {
                "sampledTrainingObjective": objective,
                "geneCollapsedRaw": profile_metrics(*collapsed[:3]),
                "geneCollapsedSameBatchReferenceResidual": profile_metrics(
                    collapsed[0], collapsed[1], collapsed[2], collapsed[3]
                ),
            }
        result[name] = {
            "sampleGenes": samples[context].tolist(),
            "sampleRows": int(local.sum()),
            "models": values,
        }
    result["parameterMovement"] = parameter_delta(initial.state_dict(), final_state)
    result["savedTrainingHistory"] = {
        "fullRunFinalRecentLoss": 3.3927231252193453,
        "historyPersisted": False,
    }
    return result


def main() -> None:
    started = time.monotonic()
    inputs = {
        "humanData": HUMAN_DATA,
        "physical": PHYSICAL,
        "bp": BP,
        "bpModel": BP_ROOT / "bp128-present/model.safetensors",
        "fixedModel": FIXED_ROOT / "model.safetensors",
        "yeastModel": YEAST_ROOT / "model.safetensors",
        "yeastTargets": YEAST_FIT / "train-targets.npy",
    }
    for name, path in inputs.items():
        if sha256(path) != PINS[name]:
            raise ValueError(f"input drift: {name}")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    # Identity-only sample selection is frozen before either target member opens.
    with np.load(HUMAN_DATA, allow_pickle=False) as archive:
        human_actions = archive["action_ids"].astype(str)
        human_context = archive["context_index"].astype(np.int64)
        human_train = archive["split_train"].astype(np.int64)
        human_context_ids = archive["context_ids"].astype(str)
    human_samples = {
        context: deterministic_gene_sample(
            human_actions[human_train][human_context[human_train] == context],
            count=SAMPLE_GENES,
            seed=SEED,
            label=f"human|{name}",
        )
        for context, name in enumerate(human_context_ids)
    }
    yeast_metadata = _load_npz(YEAST_FIT / "train-metadata.npz")
    yeast_reference = _load_npz(YEAST_ROOT / "reference.npz")
    yeast_samples = {
        context: deterministic_gene_sample(
            yeast_metadata["action_ids"][yeast_metadata["context_index"] == context],
            count=SAMPLE_GENES,
            seed=SEED,
            label=f"yeast|{name}",
        )
        for context, name in enumerate(yeast_reference["context_ids"].astype(str))
    }
    OUTPUT.mkdir(parents=True)
    source = OUTPUT / "source"
    source.mkdir()
    shutil.copy2(Path(__file__), source / Path(__file__).name)
    protocol = {
        "schema": "slp.fitting-convergence-audit-protocol/v1",
        "status": "frozen-before-target-read",
        "purpose": "Distinguish numerical underfit from fitting-to-held generalization failure without new model fitting or held-out scoring.",
        "sample": {
            "rule": "256 identities with smallest SHA256(seed731|species|context|stableGene); metadata only",
            "human": {
                human_context_ids[key]: value.tolist()
                for key, value in human_samples.items()
            },
            "yeast": {
                str(yeast_reference["context_ids"][key]): value.tolist()
                for key, value in yeast_samples.items()
            },
        },
        "metrics": "Frozen training objective on sampled rows plus record/cell-weighted gene profiles; independent query centering. Yeast residual metric subtracts the same exact batch WT anchor.",
        "inputs": {
            name: {"path": str(path), "sha256": PINS[name]}
            for name, path in inputs.items()
        },
        "accessBoundary": {
            "calculationsUseFittingRowsOnly": True,
            "newValidationOrTestScoring": False,
            "humanCompressedMemberCaveat": "The source target member co-packages development rows; ndarray decompression is not row-addressable. Only pre-pinned split_train indices enter calculations.",
        },
        "sourceSha256": sha256(source / Path(__file__).name),
    }
    write_json(OUTPUT / "protocol.json", protocol)
    frozen_protocol_sha = sha256(OUTPUT / "protocol.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    human = audit_human(human_samples, device)
    yeast = audit_yeast(yeast_samples, device)
    split_half = json.loads(
        (
            ROOT
            / "results/slp11-transition/yeast-rna-fitting-split-half-v1/report.json"
        ).read_text()
    )
    split_half_by_context = {
        entry["context"]: entry for entry in split_half["contextMetrics"]
    }
    report = {
        "schema": "slp.fitting-convergence-audit-report/v1",
        "protocolSha256": frozen_protocol_sha,
        "device": str(device),
        "runtimeSeconds": time.monotonic() - started,
        "human": human,
        "yeast": yeast,
        "measurementContext": {
            "yeastSplitHalfReportSha256": sha256(
                ROOT
                / "results/slp11-transition/yeast-rna-fitting-split-half-v1/report.json"
            ),
            "controlIndependentCenteredR": split_half_by_context["Control"][
                "independentlyQueryCenteredSplitHalf"
            ]["meanGeneProfilePearson"],
            "naclIndependentCenteredR": split_half_by_context["NaCl"][
                "independentlyQueryCenteredSplitHalf"
            ]["meanGeneProfilePearson"],
            "interpretation": "Shared-batch fitting split halves are a technical reproducibility diagnostic, not a biological noise ceiling.",
        },
        "savedLossHistory": "No audited run persisted its complete minibatch trajectory. Fixed-basis and yeast reports retain only final-100 means; matched-initializer sample objectives provide a reproducible before/after diagnostic.",
        "limitations": [
            "Development fitting diagnostic only; no new held-gene, test, benchmark, HepG2, Jurkat or SL outcomes are scored.",
            "Sample metrics use a metadata-frozen 256-gene subset per context and are descriptive, not a model-selection objective.",
            "Fitting error, held error and split-half reproducibility do not identify a biological or optimization cause.",
        ],
    }
    write_json(OUTPUT / "report.json", report)
    print(
        json.dumps(
            {
                "report": str(OUTPUT / "report.json"),
                "sha256": sha256(OUTPUT / "report.json"),
                "runtimeSeconds": report["runtimeSeconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
