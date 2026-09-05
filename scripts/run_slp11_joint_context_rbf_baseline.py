#!/usr/bin/env python3
"""Run a pooled context-conditioned Nyström RBF mean-response baseline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HASHES = {
    "development": "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c",
    "physical": "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7",
    "oldNystrom": "7446d670a1897287e62bf84f74d0f6bc8383a520d1e7b483f4e66753a0dc6da6",
    "frozenPhysicalRidge": "c91d96b724f9b99169536ba17a3cce6f0c8578d603257b830a32a335f7e1c525",
    "minimalControlV2": "501384b600c5f90fbe6ea22918777288f048091e71377ce8963cda6bd105039e",
}
CONTEXTS = (
    "replogle-2022-k562-essential-day-6",
    "replogle-2022-rpe1-essential-day-7",
    "replogle-2022-k562-gwps-day-8",
)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0)
CANDIDATE_ORDER = tuple(f"{alpha:g}" for alpha in ALPHAS) + ("global-mean-limit",)


def load_helper(path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("slp11_joint_rbf_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load generic Nyström helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def normalize_basal(
    values: np.ndarray, observed: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    common = observed.all(axis=0)
    if int(common.sum()) != 6789:
        raise ValueError("context basal common-panel count mismatch")
    selected = values[:, common].astype(np.float32)
    means = selected.mean(axis=1, dtype=np.float64).astype(np.float32)
    scales = selected.std(axis=1, dtype=np.float64).astype(np.float32)
    scales = np.maximum(scales, 1e-5)
    return ((selected - means[:, None]) / scales[:, None]).astype(np.float32), common, means, scales


def fit_context_kernel(normalized: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    if normalized.shape[0] != 3:
        raise ValueError("context kernel requires exactly three training controls")
    left_norm = np.square(normalized).sum(axis=1, dtype=np.float64)
    distances = np.maximum(
        left_norm[:, None] + left_norm[None, :] - 2.0 * (normalized @ normalized.T),
        0.0,
    )
    positive = distances[np.triu_indices(3, k=1)]
    positive = positive[positive > np.finfo(np.float32).eps]
    if len(positive) != 3:
        raise ValueError("three distinct control profiles are required")
    bandwidth = float(np.sqrt(np.median(positive)))
    kernel = np.exp(-distances / (2.0 * bandwidth**2))
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    keep = eigenvalues > 1e-6
    if int(keep.sum()) != 3:
        raise ValueError("context kernel rank drifted below three")
    eigenvalues = eigenvalues[keep]
    eigenvectors = eigenvectors[:, keep]
    basis = (eigenvectors / np.sqrt(eigenvalues)[None, :]).astype(np.float32)
    mapped = (kernel.astype(np.float32) @ basis).astype(np.float32)
    state = {
        "normalized_anchors": normalized,
        "bandwidth": np.asarray(bandwidth, dtype=np.float64),
        "kernel_basis": basis,
        "eigenvalues": eigenvalues,
    }
    report = {
        "bandwidthMedianPositiveDistance": bandwidth,
        "pairwiseSquaredDistances": distances[np.triu_indices(3, k=1)].tolist(),
        "retainedEigenvalues": 3,
        "minimumEigenvalue": float(eigenvalues.min()),
        "maximumEigenvalue": float(eigenvalues.max()),
    }
    return mapped, state, report


def design_matrix(context_basis: np.ndarray, action_basis: np.ndarray) -> np.ndarray:
    if context_basis.shape != (len(action_basis), 3) or action_basis.shape[1] != 512:
        raise ValueError("joint design requires context rank3 and action rank512")
    interaction = np.einsum(
        "ni,nj->nij", context_basis, action_basis, optimize=True
    ).reshape(len(action_basis), 1536)
    return np.concatenate((context_basis, interaction), axis=1).astype(np.float32)


def pad_action_basis(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] > 512:
        raise ValueError("action Nyström output exceeds fixed width")
    if values.shape[1] == 512:
        return values.astype(np.float32, copy=False)
    return np.pad(values, ((0, 0), (0, 512 - values.shape[1]))).astype(np.float32)


def equal_context_gene_weights(contexts: np.ndarray) -> np.ndarray:
    weights = np.empty(len(contexts), dtype=np.float64)
    unique = np.unique(contexts)
    for context in unique:
        selected = contexts == context
        weights[selected] = 1.0 / (len(unique) * int(selected.sum()))
    weights *= len(weights) / weights.sum()
    return weights.astype(np.float32)


def weighted_mean_scale(
    targets: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    total = float(weights.sum(dtype=np.float64))
    mean = np.sum(targets * weights[:, None], axis=0, dtype=np.float64) / total
    variance = np.sum(
        np.square(targets - mean) * weights[:, None], axis=0, dtype=np.float64
    ) / total
    return mean.astype(np.float32), np.maximum(np.sqrt(variance), 0.05).astype(np.float32)


def fit_weighted_ridge(
    features: np.ndarray, targets: np.ndarray, weights: np.ndarray
) -> dict[str, np.ndarray]:
    total = float(weights.sum(dtype=np.float64))
    feature_mean = (
        np.sum(features * weights[:, None], axis=0, dtype=np.float64) / total
    ).astype(np.float32)
    target_mean, _ = weighted_mean_scale(targets, weights)
    centered_x = (features - feature_mean).astype(np.float32)
    centered_y = (targets - target_mean).astype(np.float32)
    root_weight = np.sqrt(weights).astype(np.float32)
    weighted_x = centered_x * root_weight[:, None]
    weighted_y = centered_y * root_weight[:, None]
    gram = (weighted_x.T @ weighted_x).astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    keep = eigenvalues > 1e-8
    eigenvalues = eigenvalues[keep]
    eigenvectors = eigenvectors[:, keep].astype(np.float32)
    rhs = ((weighted_x @ eigenvectors).T @ weighted_y).astype(np.float32)
    return {
        "feature_mean": feature_mean,
        "target_mean": target_mean,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "rhs": rhs,
    }


def predict_weighted_ridge(
    state: dict[str, np.ndarray], features: np.ndarray, label: str
) -> np.ndarray:
    if label == "global-mean-limit":
        return np.broadcast_to(
            state["target_mean"], (len(features), len(state["target_mean"]))
        ).copy()
    rotated = (features - state["feature_mean"]) @ state["eigenvectors"]
    return (
        state["target_mean"]
        + (rotated / (state["eigenvalues"] + float(label))) @ state["rhs"]
    ).astype(np.float32)


def fold_objective(
    prediction: np.ndarray,
    targets: np.ndarray,
    contexts: np.ndarray,
    scale_by_context: dict[int, np.ndarray],
) -> tuple[float, dict[str, float]]:
    scores = {
        str(context): float(
            np.mean(
                np.square(
                    (prediction[contexts == context] - targets[contexts == context])
                    / scale_by_context[int(context)]
                ),
                dtype=np.float64,
            )
        )
        for context in np.unique(contexts)
    }
    return float(np.mean(list(scores.values()))), scores


def build_fold(
    helper: object,
    genes: np.ndarray,
    contexts: np.ndarray,
    action_values: np.ndarray,
    targets: np.ndarray,
    context_basis: np.ndarray,
    held: np.ndarray,
    seed: int,
) -> tuple[dict[str, float], dict[str, object]]:
    fitting = ~held
    fitting_ids = tuple(sorted(set(genes[fitting].astype(str))))
    gene_to_feature = {
        gene: action_values[np.flatnonzero((genes == gene) & fitting)[0]]
        for gene in fitting_ids
    }
    unique_features = np.stack([gene_to_feature[gene] for gene in fitting_ids])
    action_map, action_report = helper.fit_nystrom(fitting_ids, unique_features, seed=seed)
    fitting_phi = pad_action_basis(action_map.transform(action_values[fitting]))
    held_phi = pad_action_basis(action_map.transform(action_values[held]))
    fitting_design = design_matrix(context_basis[contexts[fitting]], fitting_phi)
    held_design = design_matrix(context_basis[contexts[held]], held_phi)
    weights = equal_context_gene_weights(contexts[fitting])
    scale_by_context = {
        int(context): np.maximum(
            targets[fitting & (contexts == context)].std(axis=0, dtype=np.float64),
            0.05,
        ).astype(np.float32)
        for context in np.unique(contexts[held])
    }
    ridge = fit_weighted_ridge(fitting_design, targets[fitting], weights)
    scores: dict[str, float] = {}
    context_scores: dict[str, dict[str, float]] = {}
    for label in CANDIDATE_ORDER:
        prediction = predict_weighted_ridge(ridge, held_design, label)
        scores[label], context_scores[label] = fold_objective(
            prediction, targets[held], contexts[held], scale_by_context
        )
    return scores, {
        "fittingRows": int(fitting.sum()),
        "heldRows": int(held.sum()),
        "fittingUniqueGenes": len(fitting_ids),
        "actionKernel": action_report,
        "actionRetainedDimensions": int(action_map.kernel_basis.shape[1]),
        "actionZeroPaddingDimensions": int(512 - action_map.kernel_basis.shape[1]),
        "ridgeFeatures": int(fitting_design.shape[1]),
        "meanWeight": float(weights.mean()),
        "contextScores": context_scores,
    }


def choose_alpha(
    helper: object,
    genes: np.ndarray,
    contexts: np.ndarray,
    action_values: np.ndarray,
    targets: np.ndarray,
    context_basis: np.ndarray,
    seed: int,
) -> tuple[str, list[dict[str, object]]]:
    reports: list[dict[str, object]] = []
    totals = {label: 0.0 for label in CANDIDATE_ORDER}
    for fold in range(3):
        held = held_gene_mask(helper, genes, fold, seed)
        scores, report = build_fold(
            helper, genes, contexts, action_values, targets, context_basis, held, seed
        )
        for label, score in scores.items():
            totals[label] += score / 3.0
        reports.append({"fold": fold, "objective": scores, **report})
    selected = min(
        CANDIDATE_ORDER,
        key=lambda label: (totals[label], CANDIDATE_ORDER.index(label)),
    )
    reports.append({"meanEqualContextScaledMse": totals, "selected": selected})
    return selected, reports


def held_gene_mask(helper: object, genes: np.ndarray, fold: int, seed: int) -> np.ndarray:
    gene_folds = {
        gene: helper.global_gene_fold(str(gene), seed=seed) for gene in np.unique(genes)
    }
    return np.asarray([gene_folds[gene] == fold for gene in genes])


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    x = left.astype(np.float64) - float(np.mean(left, dtype=np.float64))
    y = right.astype(np.float64) - float(np.mean(right, dtype=np.float64))
    denominator = math.sqrt(float(x @ x) * float(y @ y))
    return None if denominator <= np.finfo(np.float64).eps else float((x @ y) / denominator)


def score_profiles(prediction: np.ndarray, truth: np.ndarray) -> dict[str, object]:
    centered_prediction = prediction - prediction.mean(axis=0, dtype=np.float64)
    centered_truth = truth - truth.mean(axis=0, dtype=np.float64)
    independent = [pearson(x, y) for x, y in zip(centered_prediction, centered_truth, strict=True)]
    ordinary = [pearson(x, y) for x, y in zip(prediction, truth, strict=True)]
    independent = [value for value in independent if value is not None]
    ordinary = [value for value in ordinary if value is not None]
    return {
        "geneProfileMse": float(np.mean(np.square(prediction - truth), dtype=np.float64)),
        "independentlyQueryCenteredPearson": float(np.mean(independent)) if independent else None,
        "ordinaryPearson": float(np.mean(ordinary)) if ordinary else None,
        "genes": len(prediction),
    }


def save_model(
    path: Path,
    action_map: object,
    context_state: dict[str, np.ndarray],
    ridge: dict[str, np.ndarray],
    selected: str,
    query_ids: np.ndarray,
    common_mask: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        action_feature_mean=action_map.feature_mean,
        action_feature_scale=action_map.feature_scale,
        action_bandwidth=np.asarray(action_map.bandwidth),
        action_landmark_ids=np.asarray(action_map.landmark_ids),
        action_standardized_landmarks=action_map.landmarks,
        action_kernel_basis=action_map.kernel_basis,
        context_normalized_anchors=context_state["normalized_anchors"],
        context_bandwidth=context_state["bandwidth"],
        context_kernel_basis=context_state["kernel_basis"],
        context_training_profile_means=context_state["profile_means"],
        context_training_profile_scales=context_state["profile_scales"],
        context_common_mask=common_mask,
        ridge_feature_mean=ridge["feature_mean"],
        target_mean=ridge["target_mean"],
        ridge_eigenvalues=ridge["eigenvalues"],
        ridge_eigenvectors=ridge["eigenvectors"],
        ridge_rhs=ridge["rhs"],
        selected_alpha=np.asarray(selected),
        query_ids=query_ids,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    helper_path = Path(__file__).with_name("run_slp11_nystrom_rbf_baseline.py")
    helper = load_helper(helper_path)
    paths = {
        "development": Path(args.data),
        "physical": Path(args.physical),
        "oldNystrom": Path(args.old_nystrom),
        "frozenPhysicalRidge": Path(args.frozen_ridge),
        "minimalControlV2": Path(args.v2),
    }
    for name, digest in HASHES.items():
        if sha256_file(paths[name]) != digest:
            raise ValueError(f"{name} hash mismatch")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    shutil.copyfile(Path(__file__), source / Path(__file__).name)
    shutil.copyfile(helper_path, source / helper_path.name)
    protocol = {
        "schema": "slp.joint-context-rbf-source-three-protocol/v1",
        "status": "frozen-before-profile-and-fitting",
        "purpose": "stronger shared context-conditioned mean comparator; no world-model or context-transfer claim",
        "hypothesis": "A joint control-context-conditioned RBF mean predictor improves held-gene forecasts in every source context.",
        "advancementRule": "Every context requires >=1% raw equal-gene MSE improvement over both original context-local physical Nyström and frozen physical ridge, independently query-centered r >=0.10, and no r regression versus either.",
        "design": {"actionKernel": "512 landmarks over physical1156; eigenvalues <=1e-6 dropped and retained coordinates deterministically right-zero-padded to width512", "contextKernel": "RBF Nyström over all3 normalized control-only 6789-gene profiles with all3 anchors", "features": "[contextBasis3, Kronecker(contextBasis3,actionBasis512)]", "dimensions": 1539, "latentStandardization": False, "intercept": "global weighted mean"},
        "weighting": "constructs collapsed per context/gene; each context one-third total mass and genes equal within context; weights normalized global mean1",
        "selection": {"folds": 3, "globalGeneHash": "unchanged helper seed731; held gene outcomes excluded across all contexts", "alphasInTieOrder": list(CANDIDATE_ORDER), "objective": "equal-context held-gene MSE scaled by that context's inner-fitting-gene query SD, floor0.05", "ridgeFitObjective": "equal-context/equal-gene weighted raw-target least squares with one shared alpha; no target scaling inside ridge", "inferenceScaleDependency": False, "refitWithinFold": ["action feature statistics", "action bandwidth", "action landmarks/eigensystem", "per-context CV target scales", "output ridge"], "fixedResponseFreeWithinFold": ["three control-only context descriptors"]},
        "inputs": {name: {"path": str(path), "sha256": HASHES[name]} for name, path in paths.items()},
        "source": {"wrapperSha256": sha256_file(source / Path(__file__).name), "helperSha256": sha256_file(source / helper_path.name)},
        "runtime": {"seed": args.seed, "cpuThreads": 2, "maximumSeconds": args.max_seconds, "profileProjectionUnits": 4},
        "accessBoundary": {"developmentTrainAndValidationOnly": True, "testRowsExpected": 0, "HepG2OutcomesRead": False, "JurkatOutcomesRead": False, "benchmarkLabelsRead": False, "likelihoodClaim": False},
    }
    protocol_path = output / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with np.load(paths["development"], allow_pickle=False) as data:
        action_ids = data["action_ids"].astype(str)
        context_index = data["context_index"].astype(np.int64)
        context_ids = data["context_ids"].astype(str)
        query_ids = data["query_ids"].astype(str)
        record_ids = data["record_ids"].astype(str)
        train = data["split_train"].astype(np.int64)
        validation = data["split_validation"].astype(np.int64)
        targets = data["targets"].astype(np.float32)
        observed = data["observed"]
        test = data["split_test"]
        basal = data["context_basal_expression"].astype(np.float32)
        basal_observed = data["context_basal_observed"]
    if tuple(context_ids) != CONTEXTS or len(test) or not observed[train].all() or not observed[validation].all():
        raise ValueError("development contract mismatch")
    if set(action_ids[train]) & set(action_ids[validation]):
        raise ValueError("outer intervention split leakage")
    normalized_basal, common_mask, basal_means, basal_scales = normalize_basal(basal, basal_observed)
    context_basis, context_state, context_report = fit_context_kernel(normalized_basal)
    context_state["profile_means"] = basal_means
    context_state["profile_scales"] = basal_scales
    with np.load(paths["physical"], allow_pickle=False) as pack:
        entity_ids = pack["entity_id"].astype(str)
        feature_values = pack["feature_values"].astype(np.float32)
        if np.any(pack["entity_taxon"] != 9606):
            raise ValueError("physical feature taxonomy mismatch")
    feature_index = {gene: index for index, gene in enumerate(entity_ids)}
    needed = set(action_ids[np.concatenate((train, validation))])
    if needed - set(feature_index):
        raise ValueError("physical feature coverage mismatch")
    action_features = np.stack([feature_values[feature_index[gene]] for gene in action_ids])
    collapsed_gene: list[str] = []
    collapsed_context: list[int] = []
    collapsed_action: list[np.ndarray] = []
    collapsed_target: list[np.ndarray] = []
    for context in range(3):
        rows = train[context_index[train] == context]
        ids, values, responses, _ = helper.collapse_rows(rows, action_ids, action_features, targets)
        collapsed_gene.extend(ids)
        collapsed_context.extend([context] * len(ids))
        collapsed_action.extend(values)
        collapsed_target.extend(responses)
    genes = np.asarray(collapsed_gene)
    contexts = np.asarray(collapsed_context, dtype=np.int64)
    actions = np.asarray(collapsed_action, dtype=np.float32)
    response = np.asarray(collapsed_target, dtype=np.float32)
    folds = np.asarray([helper.global_gene_fold(gene, seed=args.seed) for gene in genes])
    profile_started = time.monotonic()
    profile_scores, profile_details = build_fold(
        helper, genes, contexts, actions, response, context_basis, folds == 0, args.seed
    )
    profile_seconds = time.monotonic() - profile_started
    profile = {
        "schema": "slp.joint-context-rbf-profile/v1",
        "fold": 0,
        "seconds": profile_seconds,
        "projectedFourCvAndRefitUnitsSeconds": profile_seconds * 4,
        "withinBudget": profile_seconds * 4 <= args.max_seconds,
        "objective": profile_scores,
        "details": profile_details,
    }
    (output / "profile.json").write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "profile", "seconds": profile_seconds, "projectedSeconds": profile_seconds * 4, "withinBudget": profile["withinBudget"]}, sort_keys=True), flush=True)
    if not profile["withinBudget"]:
        raise TimeoutError("one-fold profile projects beyond CPU bound")
    with threadpool_limits(limits=2):
        selected, cross_validation = choose_alpha(
            helper, genes, contexts, actions, response, context_basis, args.seed
        )
        unique_ids = tuple(sorted(set(genes.astype(str))))
        unique_values = np.stack([actions[np.flatnonzero(genes == gene)[0]] for gene in unique_ids])
        action_map, action_report = helper.fit_nystrom(unique_ids, unique_values, seed=args.seed)
        action_phi = pad_action_basis(action_map.transform(actions))
        design = design_matrix(context_basis[contexts], action_phi)
        weights = equal_context_gene_weights(contexts)
        ridge = fit_weighted_ridge(design, response, weights)
        validation_prediction = np.empty((len(validation), len(query_ids)), dtype=np.float32)
        reports: dict[str, object] = {}
        with np.load(paths["oldNystrom"], allow_pickle=False) as archive:
            if not np.array_equal(archive["record_ids"].astype(str), record_ids[validation]) or not np.array_equal(archive["query_ids"].astype(str), query_ids):
                raise ValueError("old Nyström comparator identity mismatch")
            old_nystrom = archive["mean"].astype(np.float32)
        with np.load(paths["minimalControlV2"], allow_pickle=False) as archive:
            if not np.array_equal(archive["record_ids"].astype(str), record_ids[validation]) or archive["mean"].shape != (len(validation), len(query_ids)):
                raise ValueError("v2 comparator identity mismatch")
            v2 = archive["mean"].astype(np.float32)
        with np.load(paths["frozenPhysicalRidge"], allow_pickle=False) as archive:
            frozen_ridge = {key: archive[key] for key in archive.files}
        for context, context_id in enumerate(context_ids):
            rows = validation[context_index[validation] == context]
            local = np.flatnonzero(context_index[validation] == context)
            ids, values, _, counts = helper.collapse_rows(rows, action_ids, action_features, targets)
            phi = pad_action_basis(action_map.transform(values))
            candidate_by_gene = predict_weighted_ridge(
                ridge,
                design_matrix(np.broadcast_to(context_basis[context], (len(ids), 3)), phi),
                selected,
            )
            gene_index = {gene: index for index, gene in enumerate(ids)}
            candidate_rows = np.stack([candidate_by_gene[gene_index[gene]] for gene in action_ids[rows]])
            validation_prediction[local] = candidate_rows
            _, candidate_gene, truth_gene = helper.collapse_prediction(candidate_rows, targets[rows], action_ids[rows])
            candidate_score = score_profiles(candidate_gene, truth_gene)
            comparator_rows = {
                "oldContextLocalPhysicalNystrom": old_nystrom[local],
                "originalFrozenPhysicalRidge": frozen_ridge[f"context{context}_physical"],
                "minimalControlV2": v2[local],
            }
            comparator_scores: dict[str, object] = {}
            for name, prediction_rows in comparator_rows.items():
                _, pred_gene, aligned_truth = helper.collapse_prediction(prediction_rows, targets[rows], action_ids[rows])
                if not np.array_equal(aligned_truth, truth_gene):
                    raise ValueError("comparator truth mismatch")
                comparator_scores[name] = score_profiles(pred_gene, truth_gene)
            local_training = contexts == context
            mean_rows = np.broadcast_to(
                response[local_training].mean(axis=0, dtype=np.float64),
                (len(rows), len(query_ids)),
            )
            _, mean_gene, _ = helper.collapse_prediction(mean_rows, targets[rows], action_ids[rows])
            comparator_scores["contextFittingMean"] = score_profiles(mean_gene, truth_gene)
            old_kernel = comparator_scores["oldContextLocalPhysicalNystrom"]
            old_ridge = comparator_scores["originalFrozenPhysicalRidge"]
            candidate_r = candidate_score["independentlyQueryCenteredPearson"]
            checks = {
                "mseAtLeastOnePercentBelowOldContextLocalNystrom": candidate_score["geneProfileMse"] <= 0.99 * old_kernel["geneProfileMse"],
                "mseAtLeastOnePercentBelowOriginalPhysicalRidge": candidate_score["geneProfileMse"] <= 0.99 * old_ridge["geneProfileMse"],
                "independentRAtLeastPoint10": candidate_r >= 0.10,
                "independentRNoRegressionVsOldContextLocalNystrom": candidate_r >= old_kernel["independentlyQueryCenteredPearson"],
                "independentRNoRegressionVsOriginalPhysicalRidge": candidate_r >= old_ridge["independentlyQueryCenteredPearson"],
            }
            reports[context_id] = {
                "candidate": candidate_score,
                "comparators": comparator_scores,
                "validationConstructCounts": {"minimum": int(counts.min()), "maximum": int(counts.max())},
                "checks": checks,
                "passed": all(checks.values()),
            }
            print(json.dumps({"event": "context-finished", "context": context_id, "checks": checks}, sort_keys=True), flush=True)
    model_path = output / "model.npz"
    save_model(model_path, action_map, context_state, ridge, selected, query_ids, common_mask)
    prediction_path = output / "development-predictions.npz"
    np.savez_compressed(
        prediction_path,
        mean=validation_prediction,
        record_ids=record_ids[validation],
        action_ids=action_ids[validation],
        context_index=context_index[validation],
        query_ids=query_ids,
    )
    report = {
        "schema": "slp.joint-context-rbf-source-three-result/v1",
        "decision": "advance" if all(value["passed"] for value in reports.values()) else "reject",
        "selectedAlpha": selected,
        "crossValidation": cross_validation,
        "actionKernel": action_report,
        "actionRetainedDimensions": int(action_map.kernel_basis.shape[1]),
        "actionZeroPaddingDimensions": int(512 - action_map.kernel_basis.shape[1]),
        "contextKernel": context_report,
        "weighting": {"meanWeight": float(weights.mean()), "sumByContext": {str(context): float(weights[contexts == context].sum()) for context in range(3)}},
        "contexts": reports,
        "elapsedSeconds": time.monotonic() - started,
        "profileSha256": sha256_file(output / "profile.json"),
        "protocolSha256": sha256_file(protocol_path),
        "modelSha256": sha256_file(model_path),
        "predictionsSha256": sha256_file(prediction_path),
        "accessBoundary": protocol["accessBoundary"],
        "likelihoodEvaluated": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "decision": report["decision"], "selectedAlpha": selected, "elapsedSeconds": report["elapsedSeconds"], "report": str(report_path)}, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/derived/slp11-human-gwps-fixed-panel-context-v1/replogle-k562-rpe1-gwps-complete-panel-development-v2-fixed-control-context.npz")
    parser.add_argument("--physical", default="data/derived/slp11-human-physical/direct-experiments700-v1/human-esm-go-physical-features.npz")
    parser.add_argument("--old-nystrom", default="results/slp11-transition/human-gwps-nystrom-rbf512-physical-seed731-v1/development-predictions.npz")
    parser.add_argument("--frozen-ridge", default="results/slp11-transition/physical-features-ridge-screen-v1/predictions.npz")
    parser.add_argument("--v2", default="results/slp11-transition/human-gwps-fixed-context-minimal-control-physical-state128-response32-seed731-v1/model/development-predictions.npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
