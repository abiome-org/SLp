#!/usr/bin/env python3
"""Train three context-local Scouter architecture baselines on development data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
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
DEFAULT_MODEL = ROOT / "modules/slp-1-1-scouter-adapted-baseline-v1/scouter_model.py"
DATA_SHA256 = "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c"
FEATURE_SHA256 = "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7"
REFERENCE_SHA256 = "a9f3fd2679b5a52e20dddddd427d8664b2c226f2db91bdae1e44a63e66568562"
EXPOSURE_SHA256 = "9cf5f4a5352dccaa7cb3d6c84e2123b16b190220a1ef9e03c933a887be6c81dd"
COMPARATOR_SHA256 = "8480b1f1b192edb878cb0e25eb9abc57ab9f6b67aa76f85408eab489dfa7a0ca"
SOURCE_COMMIT = "0cfddd000e19b72ff033ba67c8315f7bc3304932"
MISC_COMMIT = "6f2c83e5a32505038060155ca8257fa094732e35"
EXPECTED_CONTEXTS = (
    "replogle-2022-k562-essential-day-6",
    "replogle-2022-rpe1-essential-day-7",
    "replogle-2022-k562-gwps-day-8",
)
RIDGE_INDEPENDENT_R = {
    "replogle-2022-k562-essential-day-6": 0.269269,
    "replogle-2022-rpe1-essential-day-7": 0.315038,
    "replogle-2022-k562-gwps-day-8": 0.084352,
}


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    def clean(item: object) -> object:
        if isinstance(item, dict):
            return {str(key): clean(entry) for key, entry in item.items()}
        if isinstance(item, (tuple, list)):
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


def load_model(source: Path, expected_sha256: str):
    source = source.resolve(strict=True)
    actual = sha256_file(source)
    if actual != expected_sha256:
        raise ValueError("adapted Scouter model source SHA-256 mismatch")
    name = f"slp11_scouter_adapted_{actual[:12]}"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load adapted Scouter source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for required in ("Config", "ScouterAdaptedBaseline", "gaussian_loss"):
        if not hasattr(module, required):
            raise ValueError(f"adapted model source lacks {required}")
    return module, source, actual


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = math.sqrt(float(x @ x) * float(y @ y))
    if denominator <= np.finfo(np.float64).eps:
        return None
    return float((x @ y) / denominator)


def evaluate_rows(
    prediction: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    reference: np.ndarray,
    scale: np.ndarray,
) -> dict[str, object]:
    row_nll, row_mse, ordinary, adjusted = [], [], [], []
    undefined = {"ordinary": 0, "adjusted": 0}
    for row in range(len(prediction)):
        mask = observed[row]
        residual = prediction[row, mask].astype(np.float64) - target[row, mask]
        local_scale = scale[row, mask].astype(np.float64)
        row_nll.append(
            float(
                np.mean(
                    0.5
                    * (
                        math.log(2.0 * math.pi)
                        + 2.0 * np.log(local_scale)
                        + (residual / local_scale) ** 2
                    )
                )
            )
        )
        row_mse.append(float(np.mean(residual**2)))
        value = pearson(prediction[row, mask], target[row, mask])
        if value is None:
            undefined["ordinary"] += 1
        else:
            ordinary.append(value)
        value = pearson(
            prediction[row, mask] - reference[mask],
            target[row, mask] - reference[mask],
        )
        if value is None:
            undefined["adjusted"] += 1
        else:
            adjusted.append(value)
    return {
        "nll": float(np.mean(row_nll)),
        "mse": float(np.mean(row_mse)),
        "profile_pearson_mean": float(np.mean(ordinary)) if ordinary else None,
        "profile_centroid_adjusted_pearson_mean": (
            float(np.mean(adjusted)) if adjusted else None
        ),
        "profile_pearson_undefined": undefined["ordinary"],
        "profile_centroid_adjusted_pearson_undefined": undefined["adjusted"],
        "record_count": len(row_nll),
        "observed_count": int(observed.sum()),
    }


def gene_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    action_ids: Sequence[str],
    reference: np.ndarray,
    scale: np.ndarray,
) -> dict[str, object]:
    row = evaluate_rows(prediction, target, observed, reference, scale)
    reports = []
    for gene in sorted(set(action_ids)):
        selection = np.asarray([item == gene for item in action_ids])
        reports.append(
            evaluate_rows(
                prediction[selection],
                target[selection],
                observed[selection],
                reference,
                scale[selection],
            )
        )
    for metric in ("nll", "mse", "profile_pearson_mean", "profile_centroid_adjusted_pearson_mean"):
        values = [item[metric] for item in reports if item[metric] is not None]
        row["gene_macro_" + metric] = float(np.mean(values)) if values else None
    row["intervention_genes"] = len(reports)
    return row


def independently_centered_gene_correlation(
    prediction: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    action_ids: Sequence[str],
) -> dict[str, object]:
    genes = sorted(set(action_ids))
    pred_group = np.zeros((len(genes), prediction.shape[1]), dtype=np.float64)
    true_group = np.zeros_like(pred_group)
    group_mask = np.zeros_like(pred_group, dtype=np.bool_)
    for index, gene in enumerate(genes):
        rows = np.asarray([item == gene for item in action_ids])
        counts = observed[rows].sum(axis=0)
        group_mask[index] = counts > 0
        pred_group[index] = np.divide(
            np.where(observed[rows], prediction[rows], 0.0).sum(axis=0),
            counts,
            out=np.zeros(prediction.shape[1], dtype=np.float64),
            where=counts > 0,
        )
        true_group[index] = np.divide(
            np.where(observed[rows], target[rows], 0.0).sum(axis=0),
            counts,
            out=np.zeros(prediction.shape[1], dtype=np.float64),
            where=counts > 0,
        )
    query_counts = group_mask.sum(axis=0)
    pred_centroid = np.divide(
        np.where(group_mask, pred_group, 0.0).sum(axis=0),
        query_counts,
        out=np.zeros(prediction.shape[1], dtype=np.float64),
        where=query_counts > 0,
    )
    true_centroid = np.divide(
        np.where(group_mask, true_group, 0.0).sum(axis=0),
        query_counts,
        out=np.zeros(prediction.shape[1], dtype=np.float64),
        where=query_counts > 0,
    )
    correlations = []
    for index in range(len(genes)):
        mask = group_mask[index] & (query_counts > 0)
        value = pearson(
            pred_group[index, mask] - pred_centroid[mask],
            true_group[index, mask] - true_centroid[mask],
        )
        if value is not None:
            correlations.append(value)
    return {
        "correlation": float(np.mean(correlations)) if correlations else None,
        "definedGenes": len(correlations),
        "totalGenes": len(genes),
        "definition": (
            "equal-guide collapse within action gene; prediction and truth query centroids "
            "removed separately across validation genes"
        ),
    }


def gene_macro_weights(action_ids: Sequence[str]) -> np.ndarray:
    counts: dict[str, int] = {}
    for gene in action_ids:
        counts[gene] = counts.get(gene, 0) + 1
    return np.asarray(
        [1.0 / (len(counts) * counts[gene]) for gene in action_ids], dtype=np.float32
    )


def batches(order: np.ndarray, batch_size: int) -> list[np.ndarray]:
    chunks = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
    if len(chunks) > 1 and len(chunks[-1]) == 1:
        chunks[-2] = np.concatenate((chunks[-2], chunks[-1]))
        chunks.pop()
    return chunks


def fixed_scales(
    biological: np.ndarray,
    sampling: np.ndarray,
    context: np.ndarray,
    num_cells: np.ndarray,
) -> np.ndarray:
    variance = biological[context] + sampling[context] / num_cells[:, None]
    return np.sqrt(np.maximum(variance, 0.05**2)).astype(np.float32)


def normalize_basal_controls(
    values: np.ndarray, observed: np.ndarray, expected_common: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float32)
    mask = np.asarray(observed)
    if matrix.ndim != 2 or mask.shape != matrix.shape or mask.dtype != np.bool_:
        raise ValueError("basal control values and masks must be aligned matrices")
    common = mask.all(axis=0)
    if int(common.sum()) != expected_common or not np.isfinite(matrix[mask]).all():
        raise ValueError("fixed common basal panel drift")
    means = np.asarray([row[common].mean() for row in matrix], dtype=np.float32)[:, None]
    standard_deviations = np.asarray(
        [row[common].std() for row in matrix], dtype=np.float32
    )[:, None]
    standard_deviations = np.maximum(standard_deviations, 1e-5)
    normalized = np.where(mask, (matrix - means) / standard_deviations, 0.0).astype(
        np.float32
    )
    return normalized, means, standard_deviations


def profile(args: argparse.Namespace, model_source_sha256: str) -> dict[str, object]:
    model_module, _, _ = load_model(Path(args.model_source), model_source_sha256)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profile requested but unavailable")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    config = model_module.Config(query_dim=7036, action_feature_dim=1156)
    model = model_module.ScouterAdaptedBaseline(config).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    action = torch.randn(args.batch_size, 1, 1156, device=device)
    control = torch.randn(args.batch_size, 7036, device=device)
    target = torch.randn(args.batch_size, 7036, device=device)
    observed = torch.ones_like(target, dtype=torch.bool)
    scale = torch.ones_like(target)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    timings = []
    for iteration in range(args.profile_steps + 1):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = model(action, control)
        loss = model_module.gaussian_loss(output, target, observed, scale)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if iteration:
            timings.append(time.perf_counter() - started)
    report = {
        "schema": "slp.scouter-adapted-profile/v1",
        "modelSourceSha256": model_source_sha256,
        "device": str(device),
        "batchSize": args.batch_size,
        "steps": args.profile_steps,
        "secondsPerStepMedian": float(np.median(timings)),
        "parameters": sum(item.numel() for item in model.parameters()),
        "peakCudaBytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
        ),
    }
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def run(args: argparse.Namespace) -> dict[str, object]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    started = time.monotonic()
    paths = {
        "data": Path(args.data),
        "features": Path(args.features),
        "reference": Path(args.reference),
        "exposure": Path(args.exposure),
        "comparator": Path(args.comparator),
        "source_manifest": Path(args.source_manifest),
    }
    expected = {
        "data": DATA_SHA256,
        "features": FEATURE_SHA256,
        "reference": REFERENCE_SHA256,
        "exposure": EXPOSURE_SHA256,
        "comparator": COMPARATOR_SHA256,
    }
    for label, digest in expected.items():
        if sha256_file(paths[label]) != digest:
            raise ValueError(f"{label} SHA-256 mismatch")
    source_manifest = json.loads(paths["source_manifest"].read_text(encoding="utf-8"))
    if (
        source_manifest["scouter"]["commit"] != SOURCE_COMMIT
        or source_manifest["scouterMisc"]["commit"] != MISC_COMMIT
        or source_manifest["rights"]["trainingAllowed"] is not True
    ):
        raise ValueError("Scouter source or rights manifest mismatch")
    model_module, model_source, model_sha = load_model(
        Path(args.model_source), args.model_sha256
    )
    normalization_config = model_module.Config(query_dim=7036, action_feature_dim=1156)
    if args.variant == "layernorm-v2" and not (
        normalization_config.layer_norm and not normalization_config.batch_norm
    ):
        raise ValueError("layernorm-v2 requires LayerNorm enabled and BatchNorm disabled")
    if args.variant == "batchnorm-v1" and not (
        normalization_config.batch_norm and not normalization_config.layer_norm
    ):
        raise ValueError("batchnorm-v1 requires BatchNorm enabled and LayerNorm disabled")
    with np.load(paths["data"], allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    with np.load(paths["features"], allow_pickle=False) as archive:
        feature_ids = tuple(str(item) for item in archive["entity_id"])
        feature_taxon = archive["entity_taxon"]
        feature_values = archive["feature_values"]
    with np.load(paths["reference"], allow_pickle=False) as archive:
        reference = {name: archive[name] for name in archive.files}
    with np.load(paths["exposure"], allow_pickle=False) as archive:
        exposure = {name: archive[name] for name in archive.files}
    comparator = json.loads(paths["comparator"].read_text(encoding="utf-8"))
    if (
        tuple(str(item) for item in data["context_ids"]) != EXPECTED_CONTEXTS
        or data["targets"].shape != (13_058, 7_036)
        or data["split_train"].shape != (10_719,)
        or data["split_validation"].shape != (2_339,)
        or data["split_test"].size != 0
        or feature_values.shape != (10_231, 1_156)
        or not np.all(feature_taxon == 9606)
        or not np.array_equal(reference["query_ids"], data["query_ids"])
        or not np.array_equal(reference["context_ids"], data["context_ids"])
    ):
        raise ValueError("data, feature, context, or access-boundary contract mismatch")
    feature_row = {gene: row for row, gene in enumerate(feature_ids)}
    action_ids = tuple(str(item) for item in data["action_ids"])
    if any(gene not in feature_row for gene in action_ids):
        raise ValueError("an action lacks a pinned static feature row")
    action_values = np.stack([feature_values[feature_row[gene]] for gene in action_ids])
    standardized_actions = (
        action_values - reference["feature_mean"]
    ) / reference["feature_std"]
    context = data["context_index"].astype(np.int64)
    all_scales = fixed_scales(
        exposure["mean_biological_variance"],
        exposure["mean_sampling_variance"],
        context,
        data["num_cells_filtered"],
    )
    train = data["split_train"].astype(np.int64)
    validation = data["split_validation"].astype(np.int64)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source_dir = output / "source"
    source_dir.mkdir()
    for path in (Path(__file__), model_source, model_source.parent / "CONTRACT.md", model_source.parent / "LICENSE"):
        shutil.copyfile(path, source_dir / path.name)
    hypothesis = (
        "Replacing BatchNorm with the author-supported LayerNorm option removes the "
        "documented constant-pooled-control train/eval mismatch and permits the adapted "
        "Scouter baseline to meet the fixed molecular gates."
        if args.variant == "layernorm-v2"
        else (
            "The Scouter compressor-generator architecture with fixed physical static "
            "action features improves held-gene molecular forecasts over the query-feature decoder."
        )
    )
    protocol = {
        "schema": "slp.scouter-adapted-three-context-protocol/v1",
        "variant": args.variant,
        "status": "frozen-before-training",
        "hypothesis": hypothesis,
        "advancementRule": (
            "In every source: >=0.02 gene-macro NLL gain over mean and full physical ridge; "
            "common-centroid adjusted r >=0.10; no NLL or adjusted-r regression versus the "
            "matched v2 physical128 model; independently centered r no lower than ridge."
        ),
        "architecture": {
            **asdict(normalization_config),
            "contextModels": "three separately trained models, one per source context",
            "actionPooling": "sum",
            "fixedOutputPanel": 7036,
        },
        "optimization": {
            "optimizer": "Adam",
            "learningRate": args.learning_rate,
            "scheduler": "ExponentialLR gamma=0.9 each epoch",
            "batchSize": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "minimumValidationImprovement": args.min_improvement,
            "gradientNormMaximum": 1.0,
            "trainingLoss": "uniform row fixed mean-OOF exposure Gaussian NLL",
            "selection": "context-local validation gene-macro NLL",
            "seed": args.seed,
            "maximumTotalSeconds": args.max_seconds,
            "maximumSecondsPerContext": args.max_context_seconds,
        },
        "inputs": {
            label: {"path": str(paths[label]), "sha256": digest}
            for label, digest in expected.items()
        },
        "source": {
            "model": {"path": str(model_source), "sha256": model_sha},
            "copies": {
                path.name: sha256_file(source_dir / path.name)
                for path in (
                    Path(__file__),
                    model_source,
                    model_source.parent / "CONTRACT.md",
                    model_source.parent / "LICENSE",
                )
            },
            "authorRepository": "https://github.com/PancakeZoy/scouter",
            "authorCommit": SOURCE_COMMIT,
            "reproductionRepository": "https://github.com/PancakeZoy/scouter_misc",
            "reproductionCommit": MISC_COMMIT,
            "paperDoi": "10.1038/s43588-025-00912-8",
            "license": "MIT",
            "manifestPath": str(paths["source_manifest"]),
            "manifestSha256": sha256_file(paths["source_manifest"]),
        },
        "adaptations": [
            "stable static ESM/GO/physical action features replace GenePT descriptions",
            "pooled pseudobulk control descriptors replace sampled single control cells",
            "each control profile is centered and scaled over the 6,789 jointly observed control genes; 247 unsupported fixed-panel positions are zero",
            "fixed exposure-aware Gaussian NLL replaces autofocus direction-aware loss",
            "development pseudobulk targets replace single-cell targets",
        ],
        "accessibleModalities": [
            "training and validation pseudobulk molecular responses",
            "training-only uncertainty references",
            "raw pooled control molecular context",
            "static sequence, GO, and physical action features",
        ],
        "accessBoundary": {
            "testRowsInSnapshot": 0,
            "hepg2OutcomesRead": False,
            "jurkatOutcomesRead": False,
            "syntheticLethalityOutcomesRead": False,
        },
    }
    write_json(output / "protocol.json", protocol)
    print(json.dumps({"event": "protocol-frozen", "output": str(output)}), flush=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA unavailable; no fallback")
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    targets = torch.as_tensor(data["targets"], dtype=torch.float32, device=device)
    observed = torch.as_tensor(data["observed"], dtype=torch.bool, device=device)
    actions = torch.as_tensor(standardized_actions, dtype=torch.float32, device=device)
    scales = torch.as_tensor(all_scales, dtype=torch.float32, device=device)
    basal_observed = data["context_basal_observed"]
    normalized_controls, basal_mean, basal_std = normalize_basal_controls(
        data["context_basal_expression"], basal_observed, 6_789
    )
    controls = torch.as_tensor(normalized_controls, dtype=torch.float32, device=device)
    all_predictions = np.empty((len(validation), 7036), dtype=np.float32)
    model_summaries: dict[str, object] = {}

    for context_index, context_name in enumerate(EXPECTED_CONTEXTS):
        local_train = train[context[train] == context_index]
        local_validation = validation[context[validation] == context_index]
        context_started = time.monotonic()
        if time.monotonic() - started >= args.max_seconds:
            raise TimeoutError("global training cap reached before all context models")
        torch.manual_seed(args.seed)
        config = model_module.Config(query_dim=7036, action_feature_dim=1156)
        active_model = model_module.ScouterAdaptedBaseline(config).to(device)
        optimizer = torch.optim.Adam(active_model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        local_val_weights = torch.as_tensor(
            gene_macro_weights([action_ids[row] for row in local_validation]),
            dtype=torch.float32,
            device=device,
        )
        generator = np.random.default_rng(args.seed)

        def forward(
            rows: np.ndarray,
            local_model: torch.nn.Module = active_model,
            local_context_index: int = context_index,
        ) -> torch.Tensor:
            count = len(rows)
            return local_model(
                actions[rows, None, :],
                controls[local_context_index].expand(count, -1),
            )

        def predict(
            rows: np.ndarray, local_model: torch.nn.Module = active_model
        ) -> np.ndarray:
            local_model.eval()
            values = []
            with torch.no_grad():
                for chunk in batches(rows, args.batch_size):
                    values.append(forward(chunk).cpu().numpy())
            return np.concatenate(values)

        best_score = float("inf")
        best_epoch = 0
        best_state = None
        stale = 0
        history = []
        for epoch in range(1, args.epochs + 1):
            if (
                time.monotonic() - started >= args.max_seconds
                or time.monotonic() - context_started >= args.max_context_seconds
            ):
                break
            active_model.train()
            train_losses = []
            for rows in batches(generator.permutation(local_train), args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                prediction = forward(rows)
                loss = model_module.gaussian_loss(
                    prediction, targets[rows], observed[rows], scales[rows]
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite adapted Scouter training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(active_model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(float(loss.detach()))
            scheduler.step()
            active_model.eval()
            val_row_nll = []
            with torch.no_grad():
                for rows in batches(local_validation, args.batch_size):
                    prediction = forward(rows)
                    mask = observed[rows]
                    safe_target = torch.where(mask, targets[rows], prediction)
                    safe_scale = torch.where(mask, scales[rows], torch.ones_like(scales[rows]))
                    nll = 0.5 * (
                        math.log(2.0 * math.pi)
                        + 2.0 * torch.log(safe_scale)
                        + ((prediction - safe_target) / safe_scale).square()
                    )
                    val_row_nll.append(
                        (torch.where(mask, nll, 0.0).sum(1) / mask.sum(1)).cpu()
                    )
            row_nll = torch.cat(val_row_nll).to(device)
            score = float((row_nll * local_val_weights).sum())
            history.append(
                {
                    "epoch": epoch,
                    "trainingUniformRowNll": float(np.mean(train_losses)),
                    "validationGeneMacroNll": score,
                    "learningRateAfterEpoch": scheduler.get_last_lr()[0],
                }
            )
            if best_state is None or best_score - score > args.min_improvement:
                best_score = score
                best_epoch = epoch
                stale = 0
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in active_model.state_dict().items()
                }
            else:
                stale += 1
            if epoch == 1 or epoch % 10 == 0:
                print(
                    json.dumps(
                        {
                            "event": "epoch",
                            "context": context_name,
                            "epoch": epoch,
                            "validationGeneMacroNll": score,
                            "seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
            if stale >= args.patience:
                break
        if best_state is None:
            raise RuntimeError(f"no checkpoint completed for {context_name}")
        active_model.load_state_dict(best_state)
        prediction = predict(local_validation)
        positions = np.flatnonzero(context[validation] == context_index)
        all_predictions[positions] = prediction
        local_targets = data["targets"][local_validation]
        local_observed = data["observed"][local_validation]
        local_scales = all_scales[local_validation]
        metrics = gene_metrics(
            prediction,
            local_targets,
            local_observed,
            [action_ids[row] for row in local_validation],
            reference["evaluation_perturbation_centroid"][context_index],
            local_scales,
        )
        independent = independently_centered_gene_correlation(
            prediction,
            local_targets,
            local_observed,
            [action_ids[row] for row in local_validation],
        )
        model_path = output / f"model-context-{context_index}.safetensors"
        save_file(best_state, str(model_path))
        model_summaries[context_name] = {
            "trainRows": len(local_train),
            "validationRows": len(local_validation),
            "bestEpoch": best_epoch,
            "bestValidationGeneMacroNll": best_score,
            "epochsCompleted": len(history),
            "elapsedSeconds": time.monotonic() - context_started,
            "metrics": metrics,
            "independentlyCentered": independent,
            "checkpoint": {"path": model_path.name, "sha256": sha256_file(model_path)},
            "history": history,
        }
        del active_model, optimizer, scheduler, best_state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    comparator_results = comparator["results"]
    advancement_contexts = {}
    for context_name in EXPECTED_CONTEXTS:
        candidate = model_summaries[context_name]
        metrics = candidate["metrics"]
        old = comparator_results[context_name]
        candidate_nll = float(metrics["gene_macro_nll"])
        candidate_r = float(metrics["gene_macro_profile_centroid_adjusted_pearson_mean"])
        candidate_independent = float(candidate["independentlyCentered"]["correlation"])
        checks = {
            "meanNllGainAtLeast002": float(old["mean"]["gene_macro_nll"]) - candidate_nll >= 0.02,
            "ridgeNllGainAtLeast002": float(old["ridge"]["gene_macro_nll"]) - candidate_nll >= 0.02,
            "adjustedPearsonAtLeast010": candidate_r >= 0.10,
            "nllNoRegressionVsV2Physical128": candidate_nll <= float(old["world"]["gene_macro_nll"]),
            "adjustedPearsonNoRegressionVsV2Physical128": candidate_r >= float(old["world"]["gene_macro_profile_centroid_adjusted_pearson_mean"]),
            "independentlyCenteredNoRegressionVsRidge": candidate_independent >= RIDGE_INDEPENDENT_R[context_name],
        }
        advancement_contexts[context_name] = {
            "checks": checks,
            "passed": all(checks.values()),
            "candidateNll": candidate_nll,
            "candidateAdjustedPearson": candidate_r,
            "candidateIndependentlyCenteredPearson": candidate_independent,
            "meanNll": old["mean"]["gene_macro_nll"],
            "ridgeNll": old["ridge"]["gene_macro_nll"],
            "v2Nll": old["world"]["gene_macro_nll"],
            "v2AdjustedPearson": old["world"]["gene_macro_profile_centroid_adjusted_pearson_mean"],
            "ridgeIndependentlyCenteredPearson": RIDGE_INDEPENDENT_R[context_name],
        }
    advancement = {
        "contexts": advancement_contexts,
        "passed": all(item["passed"] for item in advancement_contexts.values()),
    }
    np.savez_compressed(
        output / "development-predictions.npz",
        mean=all_predictions,
        scale=all_scales[validation],
        record_ids=data["record_ids"][validation],
        action_ids=data["action_ids"][validation],
        context_index=context[validation],
        query_ids=data["query_ids"],
    )
    np.savez_compressed(
        output / "reference.npz",
        feature_mean=reference["feature_mean"],
        feature_std=reference["feature_std"],
        context_basal_expression=data["context_basal_expression"],
        context_basal_observed=data["context_basal_observed"],
        normalized_context_basal_expression=normalized_controls,
        context_basal_feature_mean=basal_mean,
        context_basal_feature_std=basal_std,
        evaluation_perturbation_centroid=reference["evaluation_perturbation_centroid"],
        query_ids=data["query_ids"],
        context_ids=data["context_ids"],
    )
    report = {
        "schema": "slp.scouter-adapted-three-context-result/v1",
        "variant": args.variant,
        "decision": "advance" if advancement["passed"] else "reject",
        "advancement": advancement,
        "contexts": model_summaries,
        "elapsedSeconds": time.monotonic() - started,
        "parametersPerContext": sum(
            item.numel()
            for item in model_module.ScouterAdaptedBaseline(
                model_module.Config(7036, 1156)
            ).parameters()
        ),
        "adaptedNotPublishedReproduction": True,
        "authorAttribution": {
            "authors": ["Ouyang Zhu", "Jun Li"],
            "doi": "10.1038/s43588-025-00912-8",
            "sourceCommit": SOURCE_COMMIT,
            "license": "MIT",
        },
        "outputs": {
            name: {
                "path": name,
                "sha256": sha256_file(output / name),
            }
            for name in ("development-predictions.npz", "reference.npz", "protocol.json")
        },
        "accessBoundary": protocol["accessBoundary"],
        "limitations": [
            "The full-panel dense decoder is tied to the 7,036-query panel.",
            "Each context has a separate model; this does not test unseen-context transfer.",
            "Pooled pseudobulk controls and Gaussian likelihood differ from published single-cell Scouter training.",
            "Singleton source training does not identify combination interactions.",
        ],
    }
    write_json(output / "report.json", report)
    print(json.dumps({"event": "finished", "decision": report["decision"], "report": str(output / "report.json")}), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", default="data/derived/slp11-human-gwps-fixed-panel-context-v1/replogle-k562-rpe1-gwps-complete-panel-development-v2-fixed-control-context.npz")
    result.add_argument("--features", default="data/derived/slp11-human-physical/direct-experiments700-v1/human-esm-go-physical-features.npz")
    result.add_argument("--reference", default="results/slp11-transition/human-gwps-fixed-context-minimal-control-physical-state128-response32-seed731-v1/model/reference.npz")
    result.add_argument("--exposure", default="results/slp11-transition/human-gwps-fixed-context-minimal-control-physical-state128-response32-seed731-v1/model/exposure-uncertainty.npz")
    result.add_argument("--comparator", default="results/slp11-transition/human-gwps-fixed-context-minimal-control-physical-state128-response32-seed731-v1/summary.json")
    result.add_argument("--source-manifest", default="data/sources/scouter-author-0cfddd0/source-manifest.json")
    result.add_argument("--model-source", default=str(DEFAULT_MODEL))
    result.add_argument("--model-sha256", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--learning-rate", type=float, default=0.001)
    result.add_argument("--epochs", type=int, default=180)
    result.add_argument("--patience", type=int, default=30)
    result.add_argument("--min-improvement", type=float, default=0.001)
    result.add_argument("--max-seconds", type=float, default=1800.0)
    result.add_argument("--max-context-seconds", type=float, default=600.0)
    result.add_argument("--cpu-threads", type=int, default=2)
    result.add_argument("--seed", type=int, default=731)
    result.add_argument("--profile-only", action="store_true")
    result.add_argument("--profile-steps", type=int, default=5)
    result.add_argument(
        "--variant", choices=("batchnorm-v1", "layernorm-v2"), default="batchnorm-v1"
    )
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.profile_only:
        profile(arguments, arguments.model_sha256)
    else:
        with threadpool_limits(limits=arguments.cpu_threads):
            run(arguments)
