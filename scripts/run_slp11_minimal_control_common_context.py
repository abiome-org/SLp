#!/usr/bin/env python3
"""Run one matched minimal control-anchor/common-context development pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "modules/slp-1-1-world-transition-v1"
DEFAULT_MODEL_PATH = ROOT / "modules/slp-1-1-control-transition-v2/transition_model.py"
DEFAULT_MODEL_SHA256 = "490a23869cc326f4b2c16d12b43d4aacdbd23f6d44e78008c3a1417d3fc4f46d"
OBJECTIVE_PATH = ROOT / "modules/slp-1-1-control-transition-v2/objective_weighting.py"
sys.path.insert(0, str(HELPERS))
sys.path.insert(0, str(DEFAULT_MODEL_PATH.parent))
from exposure_uncertainty import fit_exposure_uncertainty
from objective_weighting import (
    EQUAL_CONTEXT_GENE_V1,
    SUPPORTED_OBJECTIVES,
    UNIFORM_ROW_V1,
    training_row_weights,
    weighting_summary,
)
from response_queries import fit_query_response_descriptors
from transition_baselines import evaluate
from transition_calibration import (
    fit_grouped_oof_mean,
    fit_grouped_oof_ridge,
)

DATA_SHA256 = "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c"
FEATURE_SHA256 = "a2f3153478c00c191e5a9e218badb3327a180a56948a4c9c6a6926cc506ff02b"
HEPG2_CONTROL_SHA256 = "382626401ee38e8d5084ac9f86ffc44bd10408826fb85a94ede8eb908cdf5b27"
FIXED_CONTEXT_QUERIES = 6_789
EXPECTED_CONTEXTS = (
    "replogle-2022-k562-essential-day-6",
    "replogle-2022-rpe1-essential-day-7",
    "replogle-2022-k562-gwps-day-8",
)
OBSERVED_STATE_AUX_V1 = "uniform-row-observed-state-aux-v1"
LAUNCHER_OBJECTIVES = (*SUPPORTED_OBJECTIVES, OBSERVED_STATE_AUX_V1)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_transition_model(path: str | Path, expected_sha256: str):
    """Load one explicitly pinned compatible transition-model source."""

    source = Path(path).resolve(strict=True)
    actual = sha256_file(source)
    if actual != expected_sha256:
        raise ValueError("transition model source SHA-256 mismatch")
    name = f"slp11_minimal_control_transition_{actual[:12]}"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load control transition model")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for required in ("Config", "MinimalControlTransition", "gaussian_loss"):
        if not hasattr(module, required):
            raise ValueError(f"transition model source lacks {required}")
    return module, source, actual


def write_json(path: Path, value: object) -> None:
    def clean(item: object) -> object:
        if isinstance(item, dict):
            return {str(key): clean(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(entry) for entry in item]
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    path.write_text(
        json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def gene_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    keys: Sequence[tuple[int, str]],
    reference: np.ndarray,
    scale: np.ndarray,
    value_space: str,
) -> dict[str, object]:
    groups: dict[tuple[int, str], list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    reports = [
        evaluate(
            prediction[rows],
            target[rows],
            observed[rows],
            reference,
            scale[rows],
            value_space=value_space,
        )
        for rows in groups.values()
    ]
    result = evaluate(
        prediction, target, observed, reference, scale, value_space=value_space
    )
    for metric in (
        "nll",
        "mse",
        "profile_pearson_mean",
        "profile_centroid_adjusted_pearson_mean",
    ):
        values = [report[metric] for report in reports if np.isfinite(report[metric])]
        result["gene_macro_" + metric] = float(np.mean(values)) if values else None
    result["intervention_genes"] = len(reports)
    return result


def advancement_decision(
    results: dict[str, dict[str, object]],
    original: dict[str, dict[str, object]],
    context_ids: Sequence[str],
) -> dict[str, object]:
    details: dict[str, object] = {}
    for name in context_ids:
        current = results[name]
        prior = original[name]
        world = current["world"]
        old_world = prior["world"]
        assert isinstance(world, dict) and isinstance(old_world, dict)
        world_nll = float(world["gene_macro_nll"])
        old_nll = float(old_world["gene_macro_nll"])
        world_r = float(world["gene_macro_profile_centroid_adjusted_pearson_mean"])
        old_r = float(old_world["gene_macro_profile_centroid_adjusted_pearson_mean"])
        checks = {
            "deltaAgainstMeanAtLeast002": float(current["world_delta_vs_mean"]) >= 0.02,
            "deltaAgainstRidgeAtLeast002": float(current["world_delta_vs_ridge"]) >= 0.02,
            "adjustedPearsonAtLeast010": world_r >= 0.10,
            "noNllRegressionVsOriginal": world_nll <= old_nll,
            "noAdjustedPearsonRegressionVsOriginal": world_r >= old_r,
        }
        details[name] = {
            "checks": checks,
            "originalWorldNll": old_nll,
            "originalWorldAdjustedPearson": old_r,
            "passed": all(checks.values()),
        }
    return {
        "contexts": details,
        "passed": all(bool(details[name]["passed"]) for name in context_ids),
    }


def pooled_delta_amplitude(
    targets: np.ndarray,
    oof_mean: np.ndarray,
    observed: np.ndarray,
    floor: float = 0.05,
) -> np.ndarray:
    """Fit one context-agnostic query amplitude from pooled training rows."""
    values = np.asarray(targets, dtype=np.float64)
    forecast = np.asarray(oof_mean, dtype=np.float64)
    mask = np.asarray(observed)
    if (
        values.shape != forecast.shape
        or mask.shape != values.shape
        or mask.dtype != np.bool_
        or not mask.all()
        or not np.isfinite(values).all()
        or not np.isfinite(forecast).all()
        or not np.isfinite(floor)
        or floor <= 0
    ):
        raise ValueError("pooled decoder amplitude requires complete finite training arrays")
    return np.maximum(
        np.sqrt(np.mean(np.square(values - forecast), axis=0)), floor
    ).astype(np.float32)


def control_identity(
    model: torch.nn.Module,
    query_features: torch.Tensor,
    control_mean: torch.Tensor,
    delta_amplitude: torch.Tensor,
    observation_scale: torch.Tensor,
    basal_features: torch.Tensor,
    basal_values: torch.Tensor,
    basal_mask: torch.Tensor,
) -> dict[str, object]:
    contexts = control_mean.shape[0]
    model.eval()
    with torch.no_grad():
        prediction = model(
            torch.empty(
                contexts,
                0,
                model.config.action_feature_dim,
                dtype=query_features.dtype,
                device=query_features.device,
            ),
            query_features,
            control_mean,
            delta_amplitude,
            observation_scale,
            basal_features,
            basal_values,
            basal_mask,
            action_mask=torch.empty(
                contexts, 0, dtype=torch.bool, device=query_features.device
            ),
        )
    return {
        "meanBitExact": torch.equal(prediction["mean"], control_mean),
        "scaleBitExact": torch.equal(prediction["scale"], observation_scale),
        "molecularDeltaNonzero": int(torch.count_nonzero(prediction["delta"])),
        "latentDeltaNonzero": int(
            torch.count_nonzero(prediction["intervention_delta"])
        ),
        "stateEqualsBasalState": torch.equal(
            prediction["state"], prediction["basal_state"]
        ),
        "contextsChecked": contexts,
        "queriesChecked": control_mean.shape[1],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    started = time.monotonic()
    MODEL, model_path, model_sha256 = load_transition_model(
        args.model_source, args.model_sha256
    )
    data_path = Path(args.data)
    feature_path = Path(args.features)
    comparator_path = Path(args.original_report)
    hepg2_path = Path(args.hepg2_control)
    if sha256_file(data_path) != DATA_SHA256:
        raise ValueError("complete-panel development SHA-256 mismatch")
    if sha256_file(feature_path) != args.feature_sha256:
        raise ValueError("static feature SHA-256 mismatch")
    if sha256_file(hepg2_path) != HEPG2_CONTROL_SHA256:
        raise ValueError("HepG2 control-only descriptor SHA-256 mismatch")
    comparator_sha = sha256_file(comparator_path)
    with comparator_path.open(encoding="utf-8") as stream:
        comparator = json.load(stream)
    with np.load(data_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    with np.load(hepg2_path, allow_pickle=False) as archive:
        hepg2 = {name: archive[name] for name in archive.files}
    if (
        len(data["split_test"])
        or not len(data["split_train"])
        or not len(data["split_validation"])
        or tuple(data["context_ids"].tolist()) != EXPECTED_CONTEXTS
    ):
        raise ValueError("development split or context contract mismatch")
    if (
        data.get("context_basal_observed") is None
        or data["context_basal_observed"].shape != data["context_basal_expression"].shape
        or not np.all(data["context_basal_observed"].sum(1) == FIXED_CONTEXT_QUERIES)
        or hepg2["query_ids"].tolist() != data["query_ids"].tolist()
        or hepg2["context_basal_observed"].shape != (1, len(data["query_ids"]))
        or int(hepg2["context_basal_observed"].sum()) != FIXED_CONTEXT_QUERIES
        or int(hepg2["perturbed_expression_rows_read"]) != 0
        or str(hepg2["context_value_space"].item())
        != str(data["context_value_space"].item())
    ):
        raise ValueError("fixed-panel common-control descriptor contract mismatch")
    train = data["split_train"]
    validation = data["split_validation"]
    context = data["context_index"]
    if set(data["action_ids"][train]) & set(data["action_ids"][validation]):
        raise ValueError("intervention gene crossed train/validation partitions")
    if not data["observed"][train].all():
        raise ValueError("response-query descriptors require complete training values")
    weighting_objective = (
        UNIFORM_ROW_V1
        if args.training_objective == OBSERVED_STATE_AUX_V1
        else args.training_objective
    )
    train_weights = training_row_weights(
        context[train],
        data["action_ids"][train],
        objective=weighting_objective,
    )
    weight_audit = weighting_summary(
        context[train], data["action_ids"][train], train_weights
    )

    with np.load(feature_path, allow_pickle=False) as archive:
        feature_keys = list(
            zip(archive["entity_taxon"].tolist(), archive["entity_id"].tolist())
        )
        feature_rows = archive["feature_values"].astype(np.float32)
    if len(feature_keys) != len(set(feature_keys)) or not np.isfinite(
        feature_rows
    ).all():
        raise ValueError("static feature identity or value contract mismatch")
    lookup = dict(zip(feature_keys, feature_rows))
    try:
        action_values = np.stack(
            [lookup[(9606, str(item))] for item in data["action_ids"]]
        )
        query_values = np.stack(
            [lookup[(9606, str(item))] for item in data["query_ids"]]
        )
    except KeyError as error:
        raise ValueError(f"static feature missing for exact composite key {error}") from error
    targets = data["targets"]
    observed = data["observed"]
    value_space = str(data["target_value_space"].item())
    keys = [(9606, str(item)) for item in data["action_ids"]]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    sources = {
        "launcher.py": Path(__file__),
        "control_transition_model.py": model_path,
        "transition_baselines.py": HELPERS / "transition_baselines.py",
        "transition_calibration.py": HELPERS / "transition_calibration.py",
        "response_queries.py": HELPERS / "response_queries.py",
        "exposure_uncertainty.py": HELPERS / "exposure_uncertainty.py",
        "objective_weighting.py": OBJECTIVE_PATH,
    }
    source_hashes = {}
    for name, path in sources.items():
        shutil.copyfile(path, source / name)
        source_hashes[name] = sha256_file(source / name)
    protocol = {
        "hypothesis": (
            "equal context and intervention-gene training mass aligns optimization with "
            "equal-context gene-macro model selection"
            if args.training_objective == EQUAL_CONTEXT_GENE_V1
            else "a training-only observed-response encoder supplies an auxiliary latent "
            "target that improves control-anchored held-gene forecasts"
            if args.training_objective == OBSERVED_STATE_AUX_V1
            else "a control-anchored residual transition preserves exact no-intervention "
            "identity and improves held-gene molecular forecasts without context regression"
        ),
        "inputs": {
            "developmentSha256": DATA_SHA256,
            "featuresSha256": args.feature_sha256,
            "hepg2ControlOnlyDescriptorSha256": HEPG2_CONTROL_SHA256,
            "originalComparatorReportSha256": comparator_sha,
            "transitionModelSourceSha256": model_sha256,
        },
        "args": vars(args),
        "sourceHashes": source_hashes,
        "splitCounts": {"train": len(train), "validation": len(validation)},
        "queryCount": targets.shape[1],
        "featureDimensions": action_values.shape[1],
        "contexts": list(EXPECTED_CONTEXTS),
        "forecastAnchor": "supplied data.basal_control for the matching context",
        "evaluationCentroid": "context training perturbation mean",
        "decoderAmplitude": (
            "per-query RMS of gene-grouped OOF mean residuals pooled over all "
            "three fitting contexts; floor 0.05; one shared vector"
        ),
        "uncertainty": "separate training-only gene-grouped OOF mean exposure components",
        "context": (
            "64 training-control-selected tokens from the identical 6789-query "
            "control-only fixed panel; HepG2 control metadata verified but unused in fitting"
        ),
        "interaction": "absent; singleton comparison only",
        "trainingObjective": {
            "version": args.training_objective,
            "baseRowWeighting": weighting_objective,
            "weights": weight_audit,
            "minibatchRenormalization": False,
            "auxiliaryWeights": (
                {
                    "forecastGaussianNll": 1.0,
                    "posteriorReconstructionGaussianNll": 0.1,
                    "normalizedLatentMatchMse": 0.1,
                    "latentTeacherStopGradient": True,
                }
                if args.training_objective == OBSERVED_STATE_AUX_V1
                else None
            ),
        },
        "rule": {
            "nllDeltaAgainstMeanAndRidgeEachContext": 0.02,
            "adjustedPearsonEachContext": 0.10,
            "noNllOrAdjustedPearsonRegressionAgainstOriginalSeed731EachContext": True,
        },
        "testAccessed": False,
        "benchmarkAccessed": False,
        "runtime": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "numpy",
                "scipy",
                "scikit-learn",
                "safetensors",
                "threadpoolctl",
            )
        },
    }
    write_json(output / "protocol.json", protocol)
    print(
        json.dumps(
            {
                "event": "protocol-frozen",
                "train": len(train),
                "validation": len(validation),
                "queries": targets.shape[1],
            }
        ),
        flush=True,
    )

    context_count = len(EXPECTED_CONTEXTS)
    means, ridges = [], []
    oof_mean = np.empty_like(targets[train], dtype=np.float64)
    oof_ridge = np.empty_like(targets[train], dtype=np.float64)
    for index in range(context_count):
        rows = train[context[train] == index]
        positions = np.flatnonzero(context[train] == index)
        local_keys = [keys[row] for row in rows]
        mean, local_oof = fit_grouped_oof_mean(
            targets[rows],
            observed[rows],
            local_keys,
            seed=args.seed,
            scale_floor=0.05,
            return_oof=True,
        )
        ridge, local_oof_ridge = fit_grouped_oof_ridge(
            action_values[rows],
            targets[rows],
            observed[rows],
            local_keys,
            args.ridge_alpha,
            seed=args.seed,
            scale_floor=0.05,
            return_oof=True,
        )
        means.append(mean)
        ridges.append(ridge)
        oof_mean[positions] = local_oof
        oof_ridge[positions] = local_oof_ridge
        print(
            json.dumps(
                {
                    "event": "baselines-fit",
                    "context": EXPECTED_CONTEXTS[index],
                    "records": len(rows),
                }
            ),
            flush=True,
        )
    references = np.stack([item.intercept_ for item in means])
    reference_scales = np.stack([item.residual_scale_.values for item in means])
    exposure = {
        label: fit_exposure_uncertainty(
            targets[train] - oof,
            observed[train],
            data["num_cells_filtered"][train],
            context[train],
            control_targets=data["control_targets"],
            control_observed=data["control_observed"],
            control_num_cells=data["control_num_cells_filtered"],
            control_context_index=data["control_context_index"],
            scale_floor=0.05,
        )
        for label, oof in (("mean", oof_mean), ("ridge", oof_ridge))
    }
    exposure_scales = exposure["mean"].scales(
        data["num_cells_filtered"], context
    )
    delta_amplitude = pooled_delta_amplitude(
        targets[train], oof_mean, observed[train], floor=0.05
    )
    descriptors, response_info = fit_query_response_descriptors(
        targets[train],
        context[train],
        references,
        reference_scales,
        rank=args.query_basis_rank,
        seed=args.seed,
    )
    feature_mean = action_values[train].mean(0)
    feature_std = action_values[train].std(0)
    feature_std = np.where(feature_std > 1e-5, feature_std, 1.0)
    query_values = np.concatenate((query_values, descriptors), axis=1)
    query_mean = np.concatenate(
        (feature_mean, np.zeros(args.query_basis_rank, dtype=np.float32))
    )
    query_std = np.concatenate(
        (feature_std, np.ones(args.query_basis_rank, dtype=np.float32))
    )
    basal = data["context_basal_expression"]
    basal_observed = data["context_basal_observed"]
    common_basal = basal_observed.all(0)
    if int(common_basal.sum()) != FIXED_CONTEXT_QUERIES:
        raise ValueError("common basal token count drift")
    variance = basal[:, common_basal].var(0)
    common_indices = np.flatnonzero(common_basal)
    selected = common_indices[
        np.argsort(-variance, kind="stable")[: args.context_tokens]
    ]
    basal_mean = np.asarray(
        [basal[index, common_basal].mean() for index in range(context_count)]
    )[:, None]
    basal_std = np.asarray(
        [basal[index, common_basal].std() for index in range(context_count)]
    )[:, None]
    basal_std = np.maximum(basal_std, 1e-5)
    basal_normalized = np.where(
        basal_observed, (basal - basal_mean) / basal_std, 0.0
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA unavailable; no fallback")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)

    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    action_tensor = tensor((action_values - feature_mean) / feature_std)
    query_tensor = tensor((query_values - query_mean) / query_std)
    target_tensor = tensor(targets)
    observed_tensor = torch.as_tensor(observed, dtype=torch.bool, device=device)
    control_mean_tensor = tensor(data["basal_control"])
    delta_amplitude_tensor = tensor(delta_amplitude)
    basal_feature_tensor = query_tensor[selected]
    basal_value_tensor = tensor(basal_normalized[:, selected])
    basal_mask_tensor = torch.as_tensor(
        basal_observed[:, selected], dtype=torch.bool, device=device
    )
    fixed_exposure_tensor = tensor(exposure_scales)
    train_weight_by_row = np.ones(len(targets), dtype=np.float32)
    train_weight_by_row[train] = train_weights.astype(np.float32)
    train_weight_tensor = tensor(train_weight_by_row)

    config = MODEL.Config(
        action_feature_dim=action_values.shape[1],
        query_feature_dim=query_values.shape[1],
        hidden_dim=args.hidden,
        state_dim=args.state_dim,
        dropout=args.dropout,
    )
    model = MODEL.MinimalControlTransition(config).to(device)
    if args.training_objective == OBSERVED_STATE_AUX_V1 and not hasattr(
        model, "training_loss"
    ):
        raise ValueError("observed-state auxiliary objective requires model.training_loss")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = np.random.default_rng(args.seed)

    def forward(rows: np.ndarray) -> dict[str, torch.Tensor]:
        local_context = context[rows]
        prediction = model(
            action_tensor[rows],
            query_tensor,
            control_mean_tensor[local_context],
            delta_amplitude_tensor,
            fixed_exposure_tensor[rows],
            basal_feature_tensor,
            basal_value_tensor[local_context],
            basal_mask_tensor[local_context],
        )
        return prediction

    def predict(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        predictions, scales = [], []
        with torch.no_grad():
            for offset in range(0, len(rows), args.batch_size):
                selection = rows[offset : offset + args.batch_size]
                result = forward(selection)
                predictions.append(result["mean"].cpu().numpy())
                scales.append(result["scale"].cpu().numpy())
        return np.concatenate(predictions), np.concatenate(scales)

    def metrics_for(
        predictions: np.ndarray, scales: np.ndarray, rows: np.ndarray
    ) -> dict[str, dict[str, object]]:
        reports = {}
        for index, name in enumerate(EXPECTED_CONTEXTS):
            positions = np.flatnonzero(context[rows] == index)
            actual = rows[positions]
            reports[name] = gene_metrics(
                predictions[positions],
                targets[actual],
                observed[actual],
                [keys[row] for row in actual],
                references[index],
                scales[positions],
                value_space,
            )
        return reports

    best_score = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        if time.monotonic() - started >= args.max_seconds:
            break
        model.train()
        order = generator.permutation(train)
        losses = []
        component_losses: dict[str, list[float]] = {}
        for offset in range(0, len(order), args.batch_size):
            rows = order[offset : offset + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = forward(rows)
            if args.training_objective == OBSERVED_STATE_AUX_V1:
                objective = model.training_loss(
                    prediction,
                    query_tensor,
                    target_tensor[rows],
                    observed_tensor[rows],
                    control_mean_tensor[context[rows]],
                    delta_amplitude_tensor,
                )
                expected = {
                    "total",
                    "forecast_nll",
                    "reconstruction_nll",
                    "latent_match",
                }
                if set(objective) != expected:
                    raise ValueError("training_loss returned an incompatible component contract")
                loss = objective["total"]
                for name, value in objective.items():
                    component_losses.setdefault(name, []).append(float(value.detach()))
            elif args.training_objective == UNIFORM_ROW_V1:
                loss = MODEL.gaussian_loss(
                    prediction, target_tensor[rows], observed_tensor[rows]
                )
            else:
                loss = MODEL.gaussian_loss(
                    prediction,
                    target_tensor[rows],
                    observed_tensor[rows],
                    row_weight=train_weight_tensor[rows],
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        predictions, scales = predict(validation)
        reports = metrics_for(predictions, scales, validation)
        score = float(np.mean([report["gene_macro_nll"] for report in reports.values()]))
        epoch_history = {"epoch": epoch, "validation": reports}
        if args.training_objective == OBSERVED_STATE_AUX_V1:
            epoch_history["trainingLossComponents"] = {
                name: float(np.mean(values))
                for name, values in component_losses.items()
            }
        else:
            epoch_history["trainNll"] = float(np.mean(losses))
        history.append(epoch_history)
        if score < best_score - 1e-5:
            best_score = score
            best_epoch = epoch
            stale = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                json.dumps(
                    {
                        "event": "epoch",
                        "epoch": epoch,
                        "nll": score,
                        "adjustedPearson": {
                            name: report[
                                "gene_macro_profile_centroid_adjusted_pearson_mean"
                            ]
                            for name, report in reports.items()
                        },
                        "trainingLossComponents": (
                            epoch_history.get("trainingLossComponents")
                        ),
                        "seconds": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("no checkpoint completed inside the time cap")
    model.load_state_dict(best_state)
    predictions, scales = predict(validation)
    world_results = metrics_for(predictions, scales, validation)
    results: dict[str, dict[str, object]] = {}
    for index, name in enumerate(EXPECTED_CONTEXTS):
        rows = validation[context[validation] == index]
        current = {"world": world_results[name]}
        for label, baseline in (("mean", means[index]), ("ridge", ridges[index])):
            baseline_scale = exposure[label].scales(
                data["num_cells_filtered"][rows], context[rows]
            )
            current[label] = gene_metrics(
                baseline.predict(action_values[rows]),
                targets[rows],
                observed[rows],
                [keys[row] for row in rows],
                references[index],
                baseline_scale,
                value_space,
            )
        current["world_delta_vs_mean"] = (
            current["mean"]["gene_macro_nll"] - current["world"]["gene_macro_nll"]
        )
        current["world_delta_vs_ridge"] = (
            current["ridge"]["gene_macro_nll"] - current["world"]["gene_macro_nll"]
        )
        results[name] = current

    original_results = comparator.get("results")
    if not isinstance(original_results, dict) or any(
        name not in original_results for name in EXPECTED_CONTEXTS
    ):
        raise ValueError("original comparator report lacks the three context results")
    advancement = advancement_decision(results, original_results, EXPECTED_CONTEXTS)
    identity = control_identity(
        model,
        query_tensor,
        control_mean_tensor,
        delta_amplitude_tensor,
        tensor(reference_scales),
        basal_feature_tensor,
        basal_value_tensor,
        basal_mask_tensor,
    )
    if not (
        identity["meanBitExact"]
        and identity["scaleBitExact"]
        and identity["molecularDeltaNonzero"] == 0
        and identity["latentDeltaNonzero"] == 0
        and identity["stateEqualsBasalState"]
    ):
        raise RuntimeError("fitted checkpoint violated the control identity")

    save_file(best_state, str(output / "model.safetensors"))
    write_json(output / "model-config.json", asdict(config))
    np.savez_compressed(
        output / "reference.npz",
        feature_mean=feature_mean,
        feature_std=feature_std,
        query_feature_mean=query_mean,
        query_feature_std=query_std,
        query_features=query_values,
        control_mean=data["basal_control"],
        delta_amplitude=delta_amplitude,
        delta_amplitude_formula=np.asarray(
            "sqrt(mean((target - grouped_oof_context_mean)^2 over pooled training rows)); floor=0.05"
        ),
        evaluation_perturbation_centroid=references,
        context_query_indices=selected,
        context_features=query_values[selected],
        context_values=basal_normalized[:, selected],
        context_mask=basal_observed[:, selected],
        context_value_space=data["context_value_space"],
        context_ids=data["context_ids"],
        query_ids=data["query_ids"],
    )
    np.savez_compressed(
        output / "exposure-uncertainty.npz",
        **{
            f"{label}_{name}": getattr(estimator, name + "_")
            for label, estimator in exposure.items()
            for name in (
                "biological_variance",
                "sampling_variance",
                "residual_counts",
                "control_counts",
                "sampling_from_controls",
            )
        },
    )
    np.savez_compressed(
        output / "development-predictions.npz",
        mean=predictions,
        scale=scales,
        record_ids=data["record_ids"][validation],
        action_ids=data["action_ids"][validation],
        context_index=context[validation],
    )
    report = {
        "protocol": protocol,
        "modelConfig": asdict(config),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainableParameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "decoderAmplitude": {
            "contextIndexed": False,
            "shape": list(delta_amplitude.shape),
            "minimum": float(delta_amplitude.min()),
            "maximum": float(delta_amplitude.max()),
            "formula": (
                "per-query RMS grouped-OOF context-mean residual pooled over "
                "all three training contexts; floor 0.05"
            ),
        },
        "interactionBranch": None,
        "bestEpoch": best_epoch,
        "elapsedSeconds": time.monotonic() - started,
        "responseBasis": response_info,
        "trainingObjective": {
            "version": args.training_objective,
            "baseRowWeighting": weighting_objective,
            "weights": weight_audit,
            "minibatchRenormalization": False,
            "auxiliaryWeights": (
                {
                    "forecastGaussianNll": 1.0,
                    "posteriorReconstructionGaussianNll": 0.1,
                    "normalizedLatentMatchMse": 0.1,
                    "latentTeacherStopGradient": True,
                }
                if args.training_objective == OBSERVED_STATE_AUX_V1
                else None
            ),
        },
        "exposureUncertainty": {
            label: {
                "provenance": estimator.component_provenance,
                "identifiabilityWarning": estimator.identifiability_warning,
                "samplingFromControlsFraction": float(
                    estimator.sampling_from_controls_.mean()
                ),
                "scaleFloor": estimator.scale_floor,
            }
            for label, estimator in exposure.items()
        },
        "controlIdentityAfterCheckpoint": identity,
        "results": results,
        "advancement": advancement,
        "history": history,
        "artifacts": {},
        "testAccessed": False,
        "benchmarkAccessed": False,
    }
    for name in (
        "model.safetensors",
        "model-config.json",
        "reference.npz",
        "exposure-uncertainty.npz",
        "development-predictions.npz",
        "protocol.json",
    ):
        path = output / name
        report["artifacts"][name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    write_json(output / "report.json", report)
    print(
        json.dumps(
            {
                "event": "finished",
                "bestEpoch": best_epoch,
                "advancementPassed": advancement["passed"],
                "controlIdentity": identity,
                "results": {
                    name: {
                        "worldNll": value["world"]["gene_macro_nll"],
                        "ridgeNll": value["ridge"]["gene_macro_nll"],
                        "worldR": value["world"][
                            "gene_macro_profile_centroid_adjusted_pearson_mean"
                        ],
                        "ridgeR": value["ridge"][
                            "gene_macro_profile_centroid_adjusted_pearson_mean"
                        ],
                    }
                    for name, value in results.items()
                },
            }
        ),
        flush=True,
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", required=True)
    result.add_argument("--features", required=True)
    result.add_argument("--feature-sha256", default=FEATURE_SHA256)
    result.add_argument("--hepg2-control", required=True)
    result.add_argument("--original-report", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--epochs", type=int, default=180)
    result.add_argument("--patience", type=int, default=30)
    result.add_argument("--max-seconds", type=int, default=1800)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--context-tokens", type=int, default=64)
    result.add_argument("--query-basis-rank", type=int, default=32)
    result.add_argument("--hidden", type=int, default=128)
    result.add_argument("--state-dim", type=int, default=64)
    result.add_argument("--dropout", type=float, default=0.2)
    result.add_argument("--learning-rate", type=float, default=0.0005)
    result.add_argument("--weight-decay", type=float, default=0.1)
    result.add_argument("--ridge-alpha", type=float, default=10000.0)
    result.add_argument("--seed", type=int, default=731)
    result.add_argument("--cpu-threads", type=int, default=4)
    result.add_argument("--model-source", default=str(DEFAULT_MODEL_PATH))
    result.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    result.add_argument(
        "--training-objective",
        choices=LAUNCHER_OBJECTIVES,
        default=UNIFORM_ROW_V1,
    )
    return result


def main() -> None:
    args = parser().parse_args()
    with threadpool_limits(limits=args.cpu_threads):
        run(args)


if __name__ == "__main__":
    main()
