#!/usr/bin/env python3
"""Run a frozen context-local linear ridge screen for static GO BP features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

DATA_SHA256 = "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c"
PHYSICAL_SHA256 = "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7"
BP_SHA256 = "b29cbd70f08e227cddfc013e66cd1032212c8cb62e6e25162965a57101cd1fac"
BP_BASIS_SHA256 = "cc8b8e16176623778b065c92c3eb22e5b28bdd40d6d84594c379c8bab7ae2d9e"
FROZEN_RIDGE_SHA256 = "c91d96b724f9b99169536ba17a3cce6f0c8578d603257b830a32a335f7e1c525"
V2_SHA256 = "501384b600c5f90fbe6ea22918777288f048091e71377ce8963cda6bd105039e"
NYSTROM_SHA256 = "7446d670a1897287e62bf84f74d0f6bc8383a520d1e7b483f4e66753a0dc6da6"
CONTEXTS = (
    "replogle-2022-k562-essential-day-6",
    "replogle-2022-rpe1-essential-day-7",
    "replogle-2022-k562-gwps-day-8",
)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0)
CANDIDATE_ORDER = tuple(f"{alpha:g}" for alpha in ALPHAS) + ("mean-limit",)


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stable_digest(gene: str, seed: int = 731) -> bytes:
    return hashlib.sha256(f"slp11-bp-ridge-v1|{seed}|global-inner-fold|9606|{gene}".encode()).digest()


def global_gene_fold(gene: str, seed: int = 731) -> int:
    return int.from_bytes(stable_digest(gene, seed)[:8], "big") % 3


def collapse_rows(
    rows: np.ndarray,
    action_ids: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    genes = tuple(sorted(set(action_ids[rows].astype(str))))
    x = np.empty((len(genes), features.shape[1]), dtype=np.float32)
    y = np.empty((len(genes), targets.shape[1]), dtype=np.float32)
    counts = np.empty(len(genes), dtype=np.int64)
    for index, gene in enumerate(genes):
        selected = rows[action_ids[rows] == gene]
        if not np.all(features[selected] == features[selected[0]]):
            raise ValueError("static features differ within one intervention gene")
        x[index] = features[selected[0]]
        y[index] = targets[selected].mean(axis=0, dtype=np.float64)
        counts[index] = len(selected)
    return genes, x, y, counts


def standardize_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale > 1e-5, scale, 1.0).astype(np.float32)
    return ((values - mean) / scale).astype(np.float32), mean, scale


def fit_ridge_state(features: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    standardized, feature_mean, feature_scale = standardize_fit(features)
    target_mean = targets.mean(axis=0, dtype=np.float64).astype(np.float32)
    gram = (standardized.T @ standardized).astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    keep = eigenvalues > 1e-8
    eigenvalues = eigenvalues[keep]
    eigenvectors = eigenvectors[:, keep].astype(np.float32)
    rotated = standardized @ eigenvectors
    rhs = rotated.T @ (targets - target_mean)
    return {
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "target_mean": target_mean,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "rhs": rhs.astype(np.float32),
    }


def predict_state(state: dict[str, np.ndarray], values: np.ndarray, label: str) -> np.ndarray:
    if label == "mean-limit":
        return np.broadcast_to(state["target_mean"], (len(values), len(state["target_mean"]))).copy()
    standardized = (values - state["feature_mean"]) / state["feature_scale"]
    rotated = standardized @ state["eigenvectors"]
    prediction = state["target_mean"] + (
        rotated / (state["eigenvalues"] + float(label))
    ) @ state["rhs"]
    return prediction.astype(np.float32)


def choose_alpha(
    ids: tuple[str, ...], features: np.ndarray, targets: np.ndarray, seed: int
) -> tuple[str, list[dict[str, object]]]:
    folds = np.asarray([global_gene_fold(gene, seed) for gene in ids], dtype=np.int64)
    totals = {label: 0.0 for label in CANDIDATE_ORDER}
    total_genes = 0
    reports: list[dict[str, object]] = []
    for fold in range(3):
        held = folds == fold
        fitting = ~held
        if not held.any() or not fitting.any():
            raise ValueError("global fitting fold is empty")
        state = fit_ridge_state(features[fitting], targets[fitting])
        target_scale = np.maximum(
            targets[fitting].std(axis=0, dtype=np.float64), 0.05
        ).astype(np.float32)
        scores: dict[str, float] = {}
        for label in CANDIDATE_ORDER:
            prediction = predict_state(state, features[held], label)
            score = float(
                np.mean(np.square((prediction - targets[held]) / target_scale), dtype=np.float64)
            )
            totals[label] += score * int(held.sum())
            scores[label] = score
        total_genes += int(held.sum())
        reports.append(
            {
                "fold": fold,
                "fittingGenes": int(fitting.sum()),
                "heldGenes": int(held.sum()),
                "featureMeanFloat32Sha256": hashlib.sha256(state["feature_mean"].tobytes()).hexdigest(),
                "featureScaleFloat32Sha256": hashlib.sha256(state["feature_scale"].tobytes()).hexdigest(),
                "scaledMse": scores,
            }
        )
    mean_scores = {label: total / total_genes for label, total in totals.items()}
    selected = min(CANDIDATE_ORDER, key=lambda label: (mean_scores[label], CANDIDATE_ORDER.index(label)))
    reports.append({"meanScaledMse": mean_scores, "selected": selected})
    return selected, reports


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    x = left.astype(np.float64) - float(np.mean(left, dtype=np.float64))
    y = right.astype(np.float64) - float(np.mean(right, dtype=np.float64))
    denominator = math.sqrt(float(x @ x) * float(y @ y))
    if denominator <= np.finfo(np.float64).eps:
        return None
    return float((x @ y) / denominator)


def optional_mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def collapse_prediction(
    prediction: np.ndarray, truth: np.ndarray, action_ids: np.ndarray
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    genes = tuple(sorted(set(action_ids.astype(str))))
    pred = np.stack([prediction[action_ids == gene].mean(axis=0) for gene in genes])
    target = np.stack([truth[action_ids == gene].mean(axis=0) for gene in genes])
    return genes, pred, target


def score_profiles(prediction: np.ndarray, truth: np.ndarray) -> dict[str, object]:
    pred_centered = prediction - prediction.mean(axis=0, dtype=np.float64)
    truth_centered = truth - truth.mean(axis=0, dtype=np.float64)
    independent = [pearson(left, right) for left, right in zip(pred_centered, truth_centered, strict=True)]
    ordinary = [pearson(left, right) for left, right in zip(prediction, truth, strict=True)]
    return {
        "geneProfileMse": float(np.mean(np.square(prediction - truth), dtype=np.float64)),
        "independentlyQueryCenteredPearson": optional_mean(independent),
        "ordinaryPearson": optional_mean(ordinary),
        "genes": len(prediction),
    }


def save_model(
    path: Path,
    state: dict[str, np.ndarray],
    selected: str,
    query_ids: np.ndarray,
    feature_block: str,
) -> None:
    np.savez_compressed(
        path,
        **state,
        selected_alpha=np.asarray(selected),
        query_ids=query_ids,
        feature_block=np.asarray(feature_block),
    )


def validate_v2_identity(
    archive: object,
    expected_records: np.ndarray,
    expected_actions: np.ndarray,
    expected_contexts: np.ndarray,
    query_count: int,
) -> bool:
    return bool(
        np.array_equal(archive["record_ids"].astype(str), expected_records)
        and np.array_equal(archive["action_ids"].astype(str), expected_actions)
        and np.array_equal(archive["context_index"].astype(np.int64), expected_contexts)
        and archive["mean"].shape == (len(expected_records), query_count)
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    paths = {
        "development": Path(args.data),
        "physical": Path(args.physical),
        "bp": Path(args.bp),
        "bpBasis": Path(args.bp_basis),
        "frozenPhysicalRidge": Path(args.frozen_ridge),
        "minimalControlV2": Path(args.v2),
        "frozenNystrom": Path(args.nystrom),
    }
    hashes = {
        "development": DATA_SHA256,
        "physical": PHYSICAL_SHA256,
        "bp": BP_SHA256,
        "bpBasis": BP_BASIS_SHA256,
        "frozenPhysicalRidge": FROZEN_RIDGE_SHA256,
        "minimalControlV2": V2_SHA256,
        "frozenNystrom": NYSTROM_SHA256,
    }
    for name, digest in hashes.items():
        if sha256_file(paths[name]) != digest:
            raise ValueError(f"{name} hash mismatch")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    shutil.copyfile(Path(__file__), source / Path(__file__).name)
    protocol = {
        "schema": "slp.bp-ridge-source-three-protocol/v1",
        "status": "frozen-before-fitting",
        "hypothesis": "GO biological-process descriptors add forecastable functional information beyond protein sequence, MF/CC and direct physical neighbors.",
        "advancementRule": "In every context augmented ridge must lower raw equal-gene MSE by at least 1% versus both matched physical-CV ridge and original frozen physical ridge, reach independently query-centered r >=0.10, and not regress that r versus either comparator.",
        "arms": {"physical1156": 1156, "physical1156_bp128_present1": 1285},
        "fit": "separate context-local ridge; constructs collapsed equally within intervention gene before CV and fitting",
        "selection": {
            "folds": 3,
            "globalFold": "stable taxonomy9606+ENSG SHA256 seed731, shared across contexts and arms",
            "alphasInTieOrder": list(CANDIDATE_ORDER),
            "objective": "equal-gene mean fitting-SD-scaled MSE across all 7036 queries; SD floor 0.05",
            "foldLocal": ["feature mean", "feature scale", "response mean", "response SD", "ridge coefficients"],
            "staticBpBasis": "frozen response-free source-training-roster SVD; not refit within folds, analogous a pretrained static prior",
        },
        "comparators": ["matched physical1156 CV ridge", "original frozen physical ridge", "fitting mean", "minimal-control-v2", "frozen Nyström RBF512"],
        "inputs": {name: {"path": str(paths[name]), "sha256": digest} for name, digest in hashes.items()},
        "runtime": {"seed": args.seed, "cpuThreads": 2, "maximumSeconds": args.max_seconds},
        "accessBoundary": {"developmentTrainAndValidationOnly": True, "testRowsExpected": 0, "benchmarkLabelsRead": False, "externalOutcomesRead": False, "likelihoodClaim": False},
        "sourceSha256": sha256_file(source / Path(__file__).name),
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
    if tuple(context_ids) != CONTEXTS or len(test) or not observed[train].all() or not observed[validation].all():
        raise ValueError("development identity/observation contract mismatch")
    if set(action_ids[train]) & set(action_ids[validation]):
        raise ValueError("outer intervention-gene split leakage")
    with np.load(paths["physical"], allow_pickle=False) as pack:
        physical_ids = pack["entity_id"].astype(str)
        physical_taxon = pack["entity_taxon"].astype(np.int64)
        physical_values = pack["feature_values"].astype(np.float32)
    with np.load(paths["bp"], allow_pickle=False) as pack:
        bp_ids = pack["entity_id"].astype(str)
        bp_taxon = pack["entity_taxon"].astype(np.int64)
        bp_values = pack["feature_values"].astype(np.float32)
        bp_present = pack["annotation_present"].astype(np.float32)
    if np.any(physical_taxon != 9606) or np.any(bp_taxon != 9606):
        raise ValueError("feature taxonomy mismatch")
    physical_index = {gene: index for index, gene in enumerate(physical_ids)}
    bp_index = {gene: index for index, gene in enumerate(bp_ids)}
    needed = set(action_ids[np.concatenate((train, validation))])
    if needed - set(physical_index) or needed - set(bp_index):
        raise ValueError("an intervention gene lacks an explicit static feature row")
    row_physical = np.stack([physical_values[physical_index[gene]] for gene in action_ids])
    row_bp = np.stack([bp_values[bp_index[gene]] for gene in action_ids])
    row_present = np.asarray([bp_present[bp_index[gene]] for gene in action_ids], dtype=np.float32)[:, None]
    arm_features = {
        "physical1156": row_physical,
        "physical1156_bp128_present1": np.concatenate((row_physical, row_bp, row_present), axis=1),
    }
    with np.load(paths["frozenPhysicalRidge"], allow_pickle=False) as frozen_ridge:
        ridge_predictions = {key: frozen_ridge[key] for key in frozen_ridge.files}
    with np.load(paths["minimalControlV2"], allow_pickle=False) as v2:
        if not validate_v2_identity(
            v2,
            record_ids[validation],
            action_ids[validation],
            context_index[validation],
            len(query_ids),
        ):
            raise ValueError("v2 comparator identity mismatch")
        v2_mean = v2["mean"].astype(np.float32)
    with np.load(paths["frozenNystrom"], allow_pickle=False) as nystrom:
        if not np.array_equal(nystrom["record_ids"].astype(str), record_ids[validation]) or not np.array_equal(nystrom["query_ids"].astype(str), query_ids):
            raise ValueError("Nyström comparator identity mismatch")
        nystrom_mean = nystrom["mean"].astype(np.float32)

    all_predictions = {arm: np.empty((len(validation), len(query_ids)), dtype=np.float32) for arm in arm_features}
    context_reports: dict[str, object] = {}
    with threadpool_limits(limits=2):
        for context, context_id in enumerate(context_ids):
            context_train = train[context_index[train] == context]
            context_validation = validation[context_index[validation] == context]
            local_validation = np.flatnonzero(context_index[validation] == context)
            truth_rows = targets[context_validation]
            comparator_rows = {
                "originalFrozenPhysicalRidge": ridge_predictions[f"context{context}_physical"],
                "minimalControlV2": v2_mean[local_validation],
                "frozenNystromRbf512": nystrom_mean[local_validation],
            }
            arm_report: dict[str, object] = {}
            bp_coverage: dict[str, object] = {}
            for arm, values in arm_features.items():
                train_ids, train_x, train_y, train_counts = collapse_rows(context_train, action_ids, values, targets)
                validation_ids, validation_x, _, validation_counts = collapse_rows(context_validation, action_ids, values, targets)
                selected, cross_validation = choose_alpha(train_ids, train_x, train_y, args.seed)
                state = fit_ridge_state(train_x, train_y)
                prediction_by_gene = predict_state(state, validation_x, selected)
                gene_index = {gene: index for index, gene in enumerate(validation_ids)}
                prediction_rows = np.stack([prediction_by_gene[gene_index[gene]] for gene in action_ids[context_validation]])
                all_predictions[arm][local_validation] = prediction_rows
                save_model(output / f"model-{arm}-context-{context}.npz", state, selected, query_ids, arm)
                _, collapsed_pred, collapsed_truth = collapse_prediction(prediction_rows, truth_rows, action_ids[context_validation])
                arm_report[arm] = {
                    "selectedAlpha": selected,
                    "crossValidation": cross_validation,
                    "scores": score_profiles(collapsed_pred, collapsed_truth),
                    "trainingGenes": len(train_ids),
                    "validationGenes": len(validation_ids),
                    "trainingConstructCounts": {"minimum": int(train_counts.min()), "maximum": int(train_counts.max())},
                    "validationConstructCounts": {"minimum": int(validation_counts.min()), "maximum": int(validation_counts.max())},
                    "modelSha256": sha256_file(output / f"model-{arm}-context-{context}.npz"),
                }
                if arm == "physical1156":
                    bp_coverage = {
                        "trainingAnnotatedGenes": int(sum(bp_present[bp_index[gene]] > 0 for gene in train_ids)),
                        "trainingGenes": len(train_ids),
                        "validationAnnotatedGenes": int(sum(bp_present[bp_index[gene]] > 0 for gene in validation_ids)),
                        "validationGenes": len(validation_ids),
                    }
            _, _, collapsed_truth = collapse_prediction(truth_rows, truth_rows, action_ids[context_validation])
            comparator_scores: dict[str, object] = {}
            for name, prediction_rows in comparator_rows.items():
                _, collapsed_pred, comparison_truth = collapse_prediction(prediction_rows, truth_rows, action_ids[context_validation])
                if not np.array_equal(comparison_truth, collapsed_truth):
                    raise ValueError("comparator collapsed truth mismatch")
                comparator_scores[name] = score_profiles(collapsed_pred, collapsed_truth)
            train_ids, _, train_y, _ = collapse_rows(context_train, action_ids, row_physical, targets)
            mean_rows = np.broadcast_to(train_y.mean(axis=0, dtype=np.float64), truth_rows.shape)
            _, collapsed_mean, _ = collapse_prediction(mean_rows, truth_rows, action_ids[context_validation])
            comparator_scores["fittingMean"] = score_profiles(collapsed_mean, collapsed_truth)
            augmented = arm_report["physical1156_bp128_present1"]["scores"]
            matched = arm_report["physical1156"]["scores"]
            frozen = comparator_scores["originalFrozenPhysicalRidge"]
            checks = {
                "mseAtLeastOnePercentBetterThanMatchedPhysicalCv": augmented["geneProfileMse"] <= 0.99 * matched["geneProfileMse"],
                "mseAtLeastOnePercentBetterThanOriginalFrozenPhysical": augmented["geneProfileMse"] <= 0.99 * frozen["geneProfileMse"],
                "independentRAtLeastPoint10": augmented["independentlyQueryCenteredPearson"] >= 0.10,
                "independentRNoRegressionVsMatchedPhysicalCv": augmented["independentlyQueryCenteredPearson"] >= matched["independentlyQueryCenteredPearson"],
                "independentRNoRegressionVsOriginalFrozenPhysical": augmented["independentlyQueryCenteredPearson"] >= frozen["independentlyQueryCenteredPearson"],
            }
            context_reports[context_id] = {
                "arms": arm_report,
                "comparators": comparator_scores,
                "bpCoverage": bp_coverage,
                "checks": checks,
                "passed": all(checks.values()),
            }
            print(json.dumps({"event": "context-finished", "context": context_id, "selectedAlphas": {arm: value["selectedAlpha"] for arm, value in arm_report.items()}, "checks": checks}, sort_keys=True), flush=True)
            if time.monotonic() - started > args.max_seconds:
                raise TimeoutError("fixed CPU time bound exceeded")
    prediction_path = output / "development-predictions.npz"
    np.savez_compressed(
        prediction_path,
        physical1156=all_predictions["physical1156"],
        physical1156_bp128_present1=all_predictions["physical1156_bp128_present1"],
        record_ids=record_ids[validation],
        action_ids=action_ids[validation],
        context_index=context_index[validation],
        query_ids=query_ids,
    )
    report = {
        "schema": "slp.bp-ridge-source-three-result/v1",
        "decision": "advance" if all(value["passed"] for value in context_reports.values()) else "reject",
        "contexts": context_reports,
        "elapsedSeconds": time.monotonic() - started,
        "protocolSha256": sha256_file(protocol_path),
        "predictionsSha256": sha256_file(prediction_path),
        "accessBoundary": protocol["accessBoundary"],
        "likelihoodEvaluated": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "decision": report["decision"], "elapsedSeconds": report["elapsedSeconds"], "report": str(report_path)}, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/derived/slp11-human-gwps-fixed-panel-context-v1/replogle-k562-rpe1-gwps-complete-panel-development-v2-fixed-control-context.npz")
    parser.add_argument("--physical", default="data/derived/slp11-human-physical/direct-experiments700-v1/human-esm-go-physical-features.npz")
    parser.add_argument("--bp", default="data/derived/slp11-human-go-bp/goa-2022-09-19-ensembl108-source3-fit-svd128-v1/human-go-bp-source3-fit-svd128-features.npz")
    parser.add_argument("--bp-basis", default="data/derived/slp11-human-go-bp/goa-2022-09-19-ensembl108-source3-fit-svd128-v1/human-go-bp-source3-fit-svd128-basis.npz")
    parser.add_argument("--frozen-ridge", default="results/slp11-transition/physical-features-ridge-screen-v1/predictions.npz")
    parser.add_argument("--v2", default="results/slp11-transition/human-gwps-fixed-context-minimal-control-physical-state128-response32-seed731-v1/model/development-predictions.npz")
    parser.add_argument("--nystrom", default="results/slp11-transition/human-gwps-nystrom-rbf512-physical-seed731-v1/development-predictions.npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
