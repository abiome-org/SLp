#!/usr/bin/env python3
"""Run a frozen context-local Nyström RBF molecular-response baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

DATA_SHA256 = "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c"
FEATURE_SHA256 = "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7"
RIDGE_SHA256 = "c91d96b724f9b99169536ba17a3cce6f0c8578d603257b830a32a335f7e1c525"
RIDGE_REPORT_SHA256 = "736968925a96806e1384cf71663e37ffd84fb70c2c4077ed0f240c4dc7a8c4a3"
V2_SHA256 = "501384b600c5f90fbe6ea22918777288f048091e71377ce8963cda6bd105039e"
CONTEXTS = (
    "replogle-2022-k562-essential-day-6",
    "replogle-2022-rpe1-essential-day-7",
    "replogle-2022-k562-gwps-day-8",
)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
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


def stable_digest(label: str, gene: str, seed: int = 731) -> bytes:
    return hashlib.sha256(f"slp11-nystrom-rbf-v1|{seed}|{label}|9606|{gene}".encode()).digest()


def global_gene_fold(gene: str, folds: int = 3, seed: int = 731) -> int:
    return int.from_bytes(stable_digest("inner-fold", gene, seed)[:8], "big") % folds


def deterministic_selection(
    ids: tuple[str, ...], limit: int, label: str, seed: int = 731
) -> np.ndarray:
    order = sorted(range(len(ids)), key=lambda index: (stable_digest(label, ids[index], seed), ids[index]))
    return np.asarray(order[: min(limit, len(order))], dtype=np.int64)


def squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.square(left).sum(axis=1, dtype=np.float64)
    right_norm = np.square(right).sum(axis=1, dtype=np.float64)
    distance = left_norm[:, None] + right_norm[None, :] - 2.0 * (left @ right.T)
    return np.maximum(distance, 0.0)


@dataclass
class NystromMap:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    bandwidth: float
    landmark_ids: tuple[str, ...]
    landmarks: np.ndarray
    kernel_basis: np.ndarray
    eigenvalues: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        standardized = ((values - self.feature_mean) / self.feature_scale).astype(np.float32)
        distance = squared_distances(standardized, self.landmarks)
        kernel = np.exp(-distance / (2.0 * self.bandwidth**2)).astype(np.float32)
        return (kernel @ self.kernel_basis).astype(np.float32)


def fit_nystrom(
    ids: tuple[str, ...],
    values: np.ndarray,
    *,
    landmarks: int = 512,
    bandwidth_sample: int = 2048,
    eigen_floor: float = 1e-6,
    seed: int = 731,
) -> tuple[NystromMap, dict[str, object]]:
    if values.ndim != 2 or values.shape[0] != len(ids) or values.shape[1] < 1 or len(ids) < landmarks:
        raise ValueError("Nyström fitting requires aligned genes and at least 512 rows")
    feature_mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = np.where(feature_scale > 1e-5, feature_scale, 1.0).astype(np.float32)
    standardized = ((values - feature_mean) / feature_scale).astype(np.float32)
    bandwidth_rows = deterministic_selection(ids, bandwidth_sample, "bandwidth", seed)
    bandwidth_values = standardized[bandwidth_rows]
    distance = squared_distances(bandwidth_values, bandwidth_values)
    positive = distance[np.triu_indices(len(bandwidth_rows), k=1)]
    positive = positive[positive > np.finfo(np.float32).eps]
    if not len(positive):
        raise ValueError("fitting features have no positive pairwise distance")
    bandwidth = float(np.sqrt(np.median(positive)))
    landmark_rows = deterministic_selection(ids, landmarks, "landmark", seed)
    landmark_values = standardized[landmark_rows]
    landmark_distance = squared_distances(landmark_values, landmark_values)
    kernel = np.exp(-landmark_distance / (2.0 * bandwidth**2))
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    keep = eigenvalues > eigen_floor
    if not np.any(keep):
        raise ValueError("Nyström landmark kernel has no eigenvalue above floor")
    eigenvalues = eigenvalues[keep]
    eigenvectors = eigenvectors[:, keep]
    kernel_basis = (eigenvectors / np.sqrt(eigenvalues)[None, :]).astype(np.float32)
    model = NystromMap(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        bandwidth=bandwidth,
        landmark_ids=tuple(ids[index] for index in landmark_rows),
        landmarks=landmark_values,
        kernel_basis=kernel_basis,
        eigenvalues=eigenvalues,
    )
    return model, {
        "fittingGenes": len(ids),
        "bandwidthSampleGenes": len(bandwidth_rows),
        "bandwidthMedianPositiveDistance": bandwidth,
        "landmarks": len(landmark_rows),
        "retainedEigenvalues": int(keep.sum()),
        "droppedEigenvalues": int((~keep).sum()),
        "minimumRetainedEigenvalue": float(eigenvalues.min()),
        "maximumEigenvalue": float(eigenvalues.max()),
        "landmarkIdListSha256": hashlib.sha256(
            "".join(f"{gene}\n" for gene in model.landmark_ids).encode("ascii")
        ).hexdigest(),
    }


def collapse_rows(
    rows: np.ndarray,
    action_ids: np.ndarray,
    action_features: np.ndarray,
    targets: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    genes = tuple(sorted({str(action_ids[row]) for row in rows}))
    features = np.empty((len(genes), action_features.shape[1]), dtype=np.float32)
    responses = np.empty((len(genes), targets.shape[1]), dtype=np.float32)
    counts = np.empty(len(genes), dtype=np.int64)
    for index, gene in enumerate(genes):
        selected = rows[action_ids[rows] == gene]
        if not np.all(action_features[selected] == action_features[selected[0]]):
            raise ValueError("one stable action gene has inconsistent static features")
        features[index] = action_features[selected[0]]
        responses[index] = targets[selected].mean(axis=0, dtype=np.float64)
        counts[index] = len(selected)
    return genes, features, responses, counts


def fit_primal_path(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    scale: np.ndarray,
    alphas: tuple[float, ...] = ALPHAS,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    feature_mean = train_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    target_mean = train_targets.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered_train = (train_features - feature_mean).astype(np.float32)
    gram = (centered_train.T @ centered_train).astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    keep = eigenvalues > 1e-8
    eigenvalues = eigenvalues[keep]
    eigenvectors = eigenvectors[:, keep].astype(np.float32)
    train_rotated = centered_train @ eigenvectors
    validation_rotated = (validation_features - feature_mean) @ eigenvectors
    rhs = train_rotated.T @ (train_targets - target_mean)
    scores = {
        "mean-limit": float(
            np.mean(np.square((target_mean - validation_targets) / scale), dtype=np.float64)
        )
    }
    predictions: dict[str, np.ndarray] = {
        "mean-limit": np.broadcast_to(target_mean, validation_targets.shape).copy()
    }
    for alpha in alphas:
        prediction = target_mean + (validation_rotated / (eigenvalues + alpha)) @ rhs
        label = f"{alpha:g}"
        predictions[label] = prediction.astype(np.float32)
        scores[label] = float(
            np.mean(np.square((prediction - validation_targets) / scale), dtype=np.float64)
        )
    return scores, {
        "feature_mean": feature_mean,
        "target_mean": target_mean,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "rhs": rhs,
        **predictions,
    }


def choose_alpha(
    ids: tuple[str, ...], features: np.ndarray, targets: np.ndarray, seed: int
) -> tuple[str, list[dict[str, object]]]:
    totals = {f"{alpha:g}": 0.0 for alpha in ALPHAS}
    totals["mean-limit"] = 0.0
    total_genes = 0
    reports = []
    fold_ids = np.asarray([global_gene_fold(gene, seed=seed) for gene in ids])
    for fold in range(3):
        held = fold_ids == fold
        fitting = ~held
        if held.sum() == 0 or fitting.sum() < 512:
            raise ValueError("global gene fold lacks fitting or validation support")
        fitting_ids = tuple(gene for gene, selected in zip(ids, fitting, strict=True) if selected)
        kernel, kernel_report = fit_nystrom(fitting_ids, features[fitting], seed=seed)
        train_kernel = kernel.transform(features[fitting])
        held_kernel = kernel.transform(features[held])
        scale = np.maximum(targets[fitting].std(axis=0, dtype=np.float64), 0.05).astype(
            np.float32
        )
        scores, _ = fit_primal_path(
            train_kernel, targets[fitting], held_kernel, targets[held], scale
        )
        for label, score in scores.items():
            totals[label] += score * int(held.sum())
        total_genes += int(held.sum())
        reports.append(
            {
                "fold": fold,
                "fittingGenes": int(fitting.sum()),
                "heldGenes": int(held.sum()),
                "kernel": kernel_report,
                "scaledMse": scores,
            }
        )
    mean_scores = {label: value / total_genes for label, value in totals.items()}
    order = [f"{alpha:g}" for alpha in ALPHAS] + ["mean-limit"]
    selected = min(order, key=lambda label: (mean_scores[label], order.index(label)))
    reports.append({"meanScaledMse": mean_scores, "selected": selected})
    return selected, reports


def fit_final(
    ids: tuple[str, ...],
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    selected: str,
    seed: int,
) -> tuple[np.ndarray, NystromMap, dict[str, np.ndarray], dict[str, object]]:
    kernel, kernel_report = fit_nystrom(ids, train_features, seed=seed)
    train_kernel = kernel.transform(train_features)
    validation_kernel = kernel.transform(validation_features)
    dummy = np.zeros((len(validation_features), train_targets.shape[1]), dtype=np.float32)
    scale = np.ones(train_targets.shape[1], dtype=np.float32)
    _, state = fit_primal_path(
        train_kernel, train_targets, validation_kernel, dummy, scale
    )
    prediction = state[selected]
    return prediction, kernel, state, kernel_report


def collapse_prediction(
    prediction: np.ndarray, targets: np.ndarray, action_ids: np.ndarray
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    genes = tuple(sorted({str(item) for item in action_ids}))
    pred = np.stack([prediction[action_ids == gene].mean(axis=0) for gene in genes])
    truth = np.stack([targets[action_ids == gene].mean(axis=0) for gene in genes])
    return genes, pred, truth


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    x = left.astype(np.float64) - float(np.mean(left, dtype=np.float64))
    y = right.astype(np.float64) - float(np.mean(right, dtype=np.float64))
    denominator = math.sqrt(float(x @ x) * float(y @ y))
    if denominator <= np.finfo(np.float64).eps:
        return None
    return float((x @ y) / denominator)


def score_profiles(
    prediction: np.ndarray, truth: np.ndarray, fitting_centroid: np.ndarray
) -> dict[str, object]:
    pred_centered = prediction - prediction.mean(axis=0, dtype=np.float64)
    true_centered = truth - truth.mean(axis=0, dtype=np.float64)
    independent = [pearson(left, right) for left, right in zip(pred_centered, true_centered, strict=True)]
    common = [
        pearson(left - fitting_centroid, right - fitting_centroid)
        for left, right in zip(prediction, truth, strict=True)
    ]
    ordinary = [pearson(left, right) for left, right in zip(prediction, truth, strict=True)]
    return {
        "geneProfileMse": float(np.mean(np.square(prediction - truth), dtype=np.float64)),
        "independentlyCenteredPearson": float(np.mean([item for item in independent if item is not None])),
        "commonFittingCentroidPearson": float(np.mean([item for item in common if item is not None])),
        "ordinaryPearson": float(np.mean([item for item in ordinary if item is not None])),
        "genes": len(prediction),
    }


def save_model(
    path: Path,
    kernel: NystromMap,
    state: dict[str, np.ndarray],
    selected: str,
    query_ids: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        feature_mean=kernel.feature_mean,
        feature_scale=kernel.feature_scale,
        bandwidth=np.asarray(kernel.bandwidth),
        landmark_ids=np.asarray(kernel.landmark_ids),
        standardized_landmarks=kernel.landmarks,
        kernel_basis=kernel.kernel_basis,
        kernel_eigenvalues=kernel.eigenvalues,
        ridge_feature_mean=state["feature_mean"],
        target_mean=state["target_mean"],
        ridge_eigenvalues=state["eigenvalues"],
        ridge_eigenvectors=state["eigenvectors"],
        ridge_rhs=state["rhs"],
        selected_alpha=np.asarray(selected),
        query_ids=query_ids,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    paths = {
        "development": Path(args.data),
        "features": Path(args.features),
        "fullPhysicalRidgePredictions": Path(args.ridge_predictions),
        "fullPhysicalRidgeReport": Path(args.ridge_report),
        "minimalControlV2Predictions": Path(args.v2_predictions),
    }
    hashes = {
        "development": DATA_SHA256,
        "features": FEATURE_SHA256,
        "fullPhysicalRidgePredictions": RIDGE_SHA256,
        "fullPhysicalRidgeReport": RIDGE_REPORT_SHA256,
        "minimalControlV2Predictions": V2_SHA256,
    }
    for name, digest in hashes.items():
        if sha256_file(paths[name]) != digest:
            raise ValueError(f"{name} SHA-256 mismatch")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    shutil.copyfile(Path(__file__), source / Path(__file__).name)
    protocol = {
        "schema": "slp.nystrom-rbf-three-context-protocol/v1",
        "status": "frozen-before-profile-and-fitting",
        "hypothesis": (
            "A nonlinear RBF map of physical static action features improves held-gene point "
            "forecasts over context-local full physical ridge in every source."
        ),
        "advancementRule": (
            "Every context requires >=1% raw collapsed-gene-profile MSE improvement and no "
            "independently centered Pearson regression versus full physical ridge."
        ),
        "kernel": {
            "family": "Nyström RBF",
            "landmarks": 512,
            "bandwidth": "median positive pairwise distance among at most 2048 fitting vectors",
            "eigenvalueFloor": 1e-6,
            "nonpositiveEigenvalues": "drop",
            "featureStandardization": "fitting genes only, reconstructed inside every fold",
        },
        "selection": {
            "folds": 3,
            "foldIdentity": "seed731 SHA256 of exact taxonomy9606+ENSG; shared across contexts",
            "alphas": [*ALPHAS, "mean-limit"],
            "objective": "gene-macro fitting-SD-scaled MSE, query SD floor 0.05",
            "refitWithinFold": [
                "feature statistics",
                "bandwidth sample",
                "landmarks",
                "kernel eigensystem",
                "response scale",
                "ridge",
            ],
        },
        "fit": "separate context-local models; constructs collapsed equally within gene/context",
        "comparators": ["fitting mean", "full physical ridge alpha10000", "minimal-control-v2"],
        "inputs": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "sourceSha256": sha256_file(source / Path(__file__).name),
        "runtime": {"cpuThreads": 2, "maximumSeconds": args.max_seconds, "seed": args.seed},
        "accessBoundary": {
            "developmentTrainAndValidationOnly": True,
            "testRowsInSnapshot": 0,
            "externalOutcomesRead": False,
            "likelihoodClaim": False,
        },
    }
    write_json(output / "protocol.json", protocol)
    with np.load(paths["development"], allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    if (
        data["split_test"].size
        or tuple(str(item) for item in data["context_ids"]) != CONTEXTS
        or not data["observed"][data["split_train"]].all()
        or not data["observed"][data["split_validation"]].all()
        or set(data["action_ids"][data["split_train"]])
        & set(data["action_ids"][data["split_validation"]])
    ):
        raise ValueError("development access, observed-panel, or held-gene contract mismatch")
    with np.load(paths["features"], allow_pickle=False) as archive:
        feature_row = {
            str(gene): values
            for gene, values in zip(
                archive["entity_id"], archive["feature_values"], strict=True
            )
        }
    action_features = np.stack([feature_row[str(gene)] for gene in data["action_ids"]])
    train = data["split_train"].astype(np.int64)
    validation = data["split_validation"].astype(np.int64)
    context = data["context_index"].astype(np.int64)

    profile_train = train[context[train] == 0]
    profile_ids, profile_x, profile_y, _ = collapse_rows(
        profile_train, data["action_ids"], action_features, data["targets"]
    )
    profile_fold = np.asarray([global_gene_fold(gene, seed=args.seed) for gene in profile_ids])
    profile_started = time.monotonic()
    profile_fit = profile_fold != 0
    profile_held = ~profile_fit
    profile_kernel, profile_kernel_report = fit_nystrom(
        tuple(gene for gene, use in zip(profile_ids, profile_fit, strict=True) if use),
        profile_x[profile_fit],
        seed=args.seed,
    )
    profile_scale = np.maximum(profile_y[profile_fit].std(0, dtype=np.float64), 0.05).astype(np.float32)
    profile_scores, _ = fit_primal_path(
        profile_kernel.transform(profile_x[profile_fit]),
        profile_y[profile_fit],
        profile_kernel.transform(profile_x[profile_held]),
        profile_y[profile_held],
        profile_scale,
    )
    profile_seconds = time.monotonic() - profile_started
    profile = {
        "schema": "slp.nystrom-rbf-inner-fold-profile/v1",
        "context": CONTEXTS[0],
        "fold": 0,
        "seconds": profile_seconds,
        "kernel": profile_kernel_report,
        "scaledMse": profile_scores,
        "projectedTwelveFoldAndRefitUnitsSeconds": profile_seconds * 12.0,
        "withinBudget": profile_seconds * 12.0 <= args.max_seconds,
    }
    write_json(output / "profile.json", profile)
    print(json.dumps({"event": "profile", **profile}), flush=True)
    if not profile["withinBudget"]:
        return {"status": "stopped-after-profile", "profile": profile}

    with np.load(paths["fullPhysicalRidgePredictions"], allow_pickle=False) as archive:
        ridge_predictions = {name: archive[name] for name in archive.files}
    with np.load(paths["minimalControlV2Predictions"], allow_pickle=False) as archive:
        if (
            not np.array_equal(archive["record_ids"], data["record_ids"][validation])
            or not np.array_equal(archive["context_index"], context[validation])
        ):
            raise ValueError("v2 comparator identity order mismatch")
        v2_predictions = archive["mean"]
    predictions = np.empty((len(validation), data["targets"].shape[1]), dtype=np.float32)
    context_reports = {}
    for context_index, context_name in enumerate(CONTEXTS):
        if time.monotonic() - started >= args.max_seconds:
            raise TimeoutError("Nyström CPU runtime cap reached")
        fitting_rows = train[context[train] == context_index]
        validation_positions = np.flatnonzero(context[validation] == context_index)
        validation_rows = validation[validation_positions]
        train_ids, train_x, train_y, train_counts = collapse_rows(
            fitting_rows, data["action_ids"], action_features, data["targets"]
        )
        validation_ids, validation_x, _, validation_counts = collapse_rows(
            validation_rows, data["action_ids"], action_features, data["targets"]
        )
        selected, cv = choose_alpha(train_ids, train_x, train_y, args.seed)
        row_prediction, kernel, state, kernel_report = fit_final(
            train_ids, train_x, train_y, validation_x, selected, args.seed
        )
        row_lookup = {gene: row_prediction[index] for index, gene in enumerate(validation_ids)}
        predictions[validation_positions] = np.stack(
            [row_lookup[str(data["action_ids"][row])] for row in validation_rows]
        )
        save_model(
            output / f"model-context-{context_index}.npz",
            kernel,
            state,
            selected,
            data["query_ids"],
        )
        local_action_ids = data["action_ids"][validation_rows]
        comparator_rows = {
            "nystromRbf": predictions[validation_positions],
            "fullPhysicalRidge": ridge_predictions[f"context{context_index}_physical"],
            "minimalControlV2": v2_predictions[validation_positions],
            "fittingMean": np.broadcast_to(train_y.mean(0), (len(validation_rows), train_y.shape[1])),
        }
        scored = {}
        reference = train_y.mean(axis=0, dtype=np.float64)
        expected_gene_ids = None
        for label, row_values in comparator_rows.items():
            genes, profile_prediction, profile_truth = collapse_prediction(
                row_values, data["targets"][validation_rows], local_action_ids
            )
            if expected_gene_ids is None:
                expected_gene_ids = genes
            elif genes != expected_gene_ids:
                raise RuntimeError("comparator collapsed gene identities disagree")
            scored[label] = score_profiles(profile_prediction, profile_truth, reference)
        nystrom_score = scored["nystromRbf"]
        ridge_score = scored["fullPhysicalRidge"]
        checks = {
            "mseImprovementAtLeastOnePercent": nystrom_score["geneProfileMse"] <= 0.99 * ridge_score["geneProfileMse"],
            "independentlyCenteredPearsonNoRegression": nystrom_score["independentlyCenteredPearson"] >= ridge_score["independentlyCenteredPearson"],
        }
        context_reports[context_name] = {
            "selectedAlpha": selected,
            "trainingGenes": len(train_ids),
            "validationGenes": len(validation_ids),
            "trainingConstructCounts": {"minimum": int(train_counts.min()), "maximum": int(train_counts.max())},
            "validationConstructCounts": {"minimum": int(validation_counts.min()), "maximum": int(validation_counts.max())},
            "crossValidation": cv,
            "finalKernel": kernel_report,
            "scores": scored,
            "checks": checks,
            "passed": all(checks.values()),
            "modelSha256": sha256_file(output / f"model-context-{context_index}.npz"),
        }
        print(
            json.dumps(
                {
                    "event": "context-finished",
                    "context": context_name,
                    "selectedAlpha": selected,
                    "checks": checks,
                    "seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    np.savez_compressed(
        output / "development-predictions.npz",
        mean=predictions,
        record_ids=data["record_ids"][validation],
        action_ids=data["action_ids"][validation],
        context_index=context[validation],
        query_ids=data["query_ids"],
    )
    report = {
        "schema": "slp.nystrom-rbf-three-context-result/v1",
        "decision": (
            "advance" if all(item["passed"] for item in context_reports.values()) else "reject"
        ),
        "contexts": context_reports,
        "elapsedSeconds": time.monotonic() - started,
        "protocolSha256": sha256_file(output / "protocol.json"),
        "profileSha256": sha256_file(output / "profile.json"),
        "predictionsSha256": sha256_file(output / "development-predictions.npz"),
        "likelihoodEvaluated": False,
        "accessBoundary": protocol["accessBoundary"],
    }
    write_json(output / "report.json", report)
    print(json.dumps({"event": "finished", "decision": report["decision"]}), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", default="data/derived/slp11-human-gwps-fixed-panel-context-v1/replogle-k562-rpe1-gwps-complete-panel-development-v2-fixed-control-context.npz")
    result.add_argument("--features", default="data/derived/slp11-human-physical/direct-experiments700-v1/human-esm-go-physical-features.npz")
    result.add_argument("--ridge-predictions", default="results/slp11-transition/physical-features-ridge-screen-v1/predictions.npz")
    result.add_argument("--ridge-report", default="results/slp11-transition/physical-features-ridge-screen-v1/report.json")
    result.add_argument("--v2-predictions", default="results/slp11-transition/human-gwps-fixed-context-minimal-control-physical-state128-response32-seed731-v1/model/development-predictions.npz")
    result.add_argument("--output", required=True)
    result.add_argument("--seed", type=int, default=731)
    result.add_argument("--max-seconds", type=float, default=600.0)
    return result


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        run(parser().parse_args())
