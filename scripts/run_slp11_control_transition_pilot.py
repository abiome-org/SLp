#!/usr/bin/env python3
"""Run one fixed control-anchored three-context molecular development pilot."""

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
MODEL_PATH = ROOT / "modules/slp-1-1-control-transition-v1/transition_model.py"
sys.path.insert(0, str(HELPERS))
from exposure_uncertainty import fit_exposure_uncertainty
from response_queries import fit_query_response_descriptors
from transition_baselines import evaluate
from transition_calibration import (
    fit_grouped_oof_mean,
    fit_grouped_oof_ridge,
)

MODEL_SPEC = importlib.util.spec_from_file_location("slp11_control_transition", MODEL_PATH)
if MODEL_SPEC is None or MODEL_SPEC.loader is None:
    raise RuntimeError("could not load control transition model")
MODEL = importlib.util.module_from_spec(MODEL_SPEC)
sys.modules[MODEL_SPEC.name] = MODEL
MODEL_SPEC.loader.exec_module(MODEL)

DATA_SHA256 = "006b4bb127a09073a7f409d81a7bccce96bb961879cb5e57dce56b48eb8e664b"
FEATURE_SHA256 = "a2f3153478c00c191e5a9e218badb3327a180a56948a4c9c6a6926cc506ff02b"
EXPECTED_CONTEXTS = (
    "replogle-2022-k562-essential-day-6",
    "replogle-2022-rpe1-essential-day-7",
    "replogle-2022-k562-gwps-day-8",
)


def sha256_file(path: str | Path) -> str:
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


def freeze_singleton_interaction(model: torch.nn.Module) -> None:
    """Disable the pair branch when every fitted intervention is a singleton."""
    with torch.no_grad():
        model.interaction_projection.weight.zero_()
    model.interaction_projection.requires_grad_(False)


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


def control_identity(
    model: torch.nn.Module,
    query_features: torch.Tensor,
    control_mean: torch.Tensor,
    control_scale: torch.Tensor,
    basal_features: torch.Tensor,
    basal_values: torch.Tensor,
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
            control_scale,
            basal_features,
            basal_values,
            torch.ones(
                basal_values.shape, dtype=torch.bool, device=basal_values.device
            ),
            action_mask=torch.empty(
                contexts, 0, dtype=torch.bool, device=query_features.device
            ),
        )
    return {
        "meanBitExact": torch.equal(prediction["mean"], control_mean),
        "scaleBitExact": torch.equal(prediction["scale"], control_scale),
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
    data_path = Path(args.data)
    feature_path = Path(args.features)
    comparator_path = Path(args.original_report)
    if sha256_file(data_path) != DATA_SHA256:
        raise ValueError("complete-panel development SHA-256 mismatch")
    if sha256_file(feature_path) != FEATURE_SHA256:
        raise ValueError("static feature SHA-256 mismatch")
    comparator_sha = sha256_file(comparator_path)
    with comparator_path.open(encoding="utf-8") as stream:
        comparator = json.load(stream)
    with np.load(data_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    if (
        len(data["split_test"])
        or not len(data["split_train"])
        or not len(data["split_validation"])
        or tuple(data["context_ids"].tolist()) != EXPECTED_CONTEXTS
    ):
        raise ValueError("development split or context contract mismatch")
    train = data["split_train"]
    validation = data["split_validation"]
    context = data["context_index"]
    if set(data["action_ids"][train]) & set(data["action_ids"][validation]):
        raise ValueError("intervention gene crossed train/validation partitions")
    if not data["observed"][train].all():
        raise ValueError("response-query descriptors require complete training values")

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
        "control_transition_model.py": MODEL_PATH,
        "transition_baselines.py": HELPERS / "transition_baselines.py",
        "transition_calibration.py": HELPERS / "transition_calibration.py",
        "response_queries.py": HELPERS / "response_queries.py",
        "exposure_uncertainty.py": HELPERS / "exposure_uncertainty.py",
    }
    source_hashes = {}
    for name, path in sources.items():
        shutil.copyfile(path, source / name)
        source_hashes[name] = sha256_file(source / name)
    protocol = {
        "hypothesis": (
            "a control-anchored residual transition preserves exact no-intervention "
            "identity and improves held-gene molecular forecasts without context regression"
        ),
        "inputs": {
            "developmentSha256": DATA_SHA256,
            "featuresSha256": FEATURE_SHA256,
            "originalComparatorReportSha256": comparator_sha,
        },
        "args": vars(args),
        "sourceHashes": source_hashes,
        "splitCounts": {"train": len(train), "validation": len(validation)},
        "queryCount": targets.shape[1],
        "featureDimensions": action_values.shape[1],
        "contexts": list(EXPECTED_CONTEXTS),
        "forecastAnchor": "supplied data.basal_control for the matching context",
        "evaluationCentroid": "context training perturbation mean",
        "uncertainty": "training-only gene-grouped OOF mean exposure components",
        "interaction": "singleton-only fitting; pair projection zeroed and frozen",
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
    selected = np.argsort(-basal.var(0), kind="stable")[: args.context_tokens]
    basal_mean = basal.mean(1, keepdims=True)
    basal_std = np.maximum(basal.std(1, keepdims=True), 1e-5)
    basal_normalized = (basal - basal_mean) / basal_std

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
    control_scale_tensor = tensor(reference_scales)
    basal_feature_tensor = query_tensor[selected]
    basal_value_tensor = tensor(basal_normalized[:, selected])
    fixed_exposure_tensor = tensor(exposure_scales)

    config = MODEL.Config(
        action_feature_dim=action_values.shape[1],
        query_feature_dim=query_values.shape[1],
        assay_feature_dim=0,
        hidden_dim=args.hidden,
        state_dim=args.state_dim,
        dropout=args.dropout,
        learn_scale=False,
    )
    model = MODEL.ControlTransition(config).to(device)
    freeze_singleton_interaction(model)
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
            control_scale_tensor[local_context],
            basal_feature_tensor,
            basal_value_tensor[local_context],
            torch.ones(
                (len(rows), len(selected)), dtype=torch.bool, device=device
            ),
        )
        # Cell count affects likelihood only. It never changes state or mean.
        prediction["scale"] = fixed_exposure_tensor[rows]
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
        for offset in range(0, len(order), args.batch_size):
            rows = order[offset : offset + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = forward(rows)
            loss = MODEL.gaussian_loss(
                prediction, target_tensor[rows], observed_tensor[rows]
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
        history.append(
            {
                "epoch": epoch,
                "trainNll": float(np.mean(losses)),
                "validation": reports,
            }
        )
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
        control_scale_tensor,
        basal_feature_tensor,
        basal_value_tensor,
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
        control_delta_scale=reference_scales,
        evaluation_perturbation_centroid=references,
        context_query_indices=selected,
        context_features=query_values[selected],
        context_values=basal_normalized[:, selected],
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
        "interactionProjection": {
            "parameters": model.interaction_projection.weight.numel(),
            "trained": False,
            "frozenValue": "exact zero",
            "reason": "all fitting interventions are singleton actions",
        },
        "bestEpoch": best_epoch,
        "elapsedSeconds": time.monotonic() - started,
        "responseBasis": response_info,
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
    return result


def main() -> None:
    args = parser().parse_args()
    with threadpool_limits(limits=args.cpu_threads):
        run(args)


if __name__ == "__main__":
    main()
