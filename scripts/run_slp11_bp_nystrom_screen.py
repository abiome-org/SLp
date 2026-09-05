#!/usr/bin/env python3
"""Run the frozen BP-augmented Nyström mean-forecast screen."""

from __future__ import annotations

import argparse
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
    "bp": "b29cbd70f08e227cddfc013e66cd1032212c8cb62e6e25162965a57101cd1fac",
    "bpBasis": "cc8b8e16176623778b065c92c3eb22e5b28bdd40d6d84594c379c8bab7ae2d9e",
    "oldNystrom": "7446d670a1897287e62bf84f74d0f6bc8383a520d1e7b483f4e66753a0dc6da6",
    "frozenPhysicalRidge": "c91d96b724f9b99169536ba17a3cce6f0c8578d603257b830a32a335f7e1c525",
    "minimalControlV2": "501384b600c5f90fbe6ea22918777288f048091e71377ce8963cda6bd105039e",
    "bpLinear": "f88efe29faccddbe93a7af1c3e95210b615d9235a3f9ad7d6f9de8530fec498f",
}
CONTEXTS = (
    "replogle-2022-k562-essential-day-6",
    "replogle-2022-rpe1-essential-day-7",
    "replogle-2022-k562-gwps-day-8",
)


def load_helper(path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("slp11_nystrom_generic", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load generic Nyström helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    x = left.astype(np.float64) - float(np.mean(left, dtype=np.float64))
    y = right.astype(np.float64) - float(np.mean(right, dtype=np.float64))
    denominator = math.sqrt(float(x @ x) * float(y @ y))
    if denominator <= np.finfo(np.float64).eps:
        return None
    return float((x @ y) / denominator)


def score_profiles(prediction: np.ndarray, truth: np.ndarray) -> dict[str, object]:
    pred_centered = prediction - prediction.mean(axis=0, dtype=np.float64)
    truth_centered = truth - truth.mean(axis=0, dtype=np.float64)
    independent = [pearson(x, y) for x, y in zip(pred_centered, truth_centered, strict=True)]
    ordinary = [pearson(x, y) for x, y in zip(prediction, truth, strict=True)]
    independent = [value for value in independent if value is not None]
    ordinary = [value for value in ordinary if value is not None]
    return {
        "geneProfileMse": float(np.mean(np.square(prediction - truth), dtype=np.float64)),
        "independentlyQueryCenteredPearson": float(np.mean(independent)) if independent else None,
        "ordinaryPearson": float(np.mean(ordinary)) if ordinary else None,
        "genes": len(prediction),
    }


def validate_row_comparator(
    archive: object,
    records: np.ndarray,
    actions: np.ndarray,
    contexts: np.ndarray,
    queries: np.ndarray,
) -> bool:
    return bool(
        np.array_equal(archive["record_ids"].astype(str), records)
        and np.array_equal(archive["action_ids"].astype(str), actions)
        and np.array_equal(archive["context_index"].astype(np.int64), contexts)
        and archive["mean"].shape == (len(records), len(queries))
        and ("query_ids" not in archive.files or np.array_equal(archive["query_ids"].astype(str), queries))
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    helper_path = Path(__file__).with_name("run_slp11_nystrom_rbf_baseline.py")
    helper = load_helper(helper_path)
    paths = {
        "development": Path(args.data),
        "physical": Path(args.physical),
        "bp": Path(args.bp),
        "bpBasis": Path(args.bp_basis),
        "oldNystrom": Path(args.old_nystrom),
        "frozenPhysicalRidge": Path(args.frozen_ridge),
        "minimalControlV2": Path(args.v2),
        "bpLinear": Path(args.bp_linear),
    }
    for name, digest in HASHES.items():
        if helper.sha256_file(paths[name]) != digest:
            raise ValueError(f"{name} hash mismatch")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    shutil.copyfile(Path(__file__), source / Path(__file__).name)
    shutil.copyfile(helper_path, source / helper_path.name)
    protocol = {
        "schema": "slp.bp-nystrom-source-three-protocol/v1",
        "status": "frozen-before-fitting",
        "hypothesis": "Combined static biological-process and physical/MF/CC/protein features improve nonlinear held-gene mean forecasts in every source context.",
        "advancementRule": "Every context requires augmented Nyström raw equal-gene MSE at least 1% below both frozen physical Nyström and original frozen physical ridge, independently query-centered r >=0.10, and no r regression versus either.",
        "candidate": {"featureBlocks": ["physical1156", "frozenBp128", "bpAnnotationPresent1"], "dimensions": 1285},
        "kernel": {"family": "Nyström RBF", "landmarks": 512, "bandwidth": "median positive distance among <=2048 fitting vectors", "eigenvalueFloor": 1e-6},
        "selection": {"folds": 3, "foldIdentity": "unchanged generic-helper global gene hash seed731", "alphas": [*helper.ALPHAS, "mean-limit"], "objective": "equal-gene fitting-SD-scaled MSE, query SD floor0.05", "foldLocal": ["feature standardization", "bandwidth", "landmarks", "kernel eigensystem", "response scale", "ridge"]},
        "bpBasis": "frozen response-free source-training-roster descriptor; no molecular quantitative values used; not refit in inner folds",
        "fitAndEvaluation": "constructs collapsed equally within intervention gene; separate model per context; all7036 queries",
        "comparators": ["frozen physical Nyström", "original frozen physical ridge", "minimal-control-v2 descriptive", "BP linear ridge descriptive", "fitting mean descriptive"],
        "inputs": {name: {"path": str(paths[name]), "sha256": digest} for name, digest in HASHES.items()},
        "source": {"wrapperSha256": helper.sha256_file(source / Path(__file__).name), "genericHelperSha256": helper.sha256_file(source / helper_path.name)},
        "runtime": {"seed": args.seed, "cpuThreads": 2, "maximumSeconds": args.max_seconds},
        "accessBoundary": {"developmentTrainAndValidationOnly": True, "testRowsExpected": 0, "benchmarkLabelsRead": False, "externalOutcomesRead": False, "likelihoodClaim": False},
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
        raise ValueError("development contract mismatch")
    if set(action_ids[train]) & set(action_ids[validation]):
        raise ValueError("outer gene split leakage")
    with np.load(paths["physical"], allow_pickle=False) as pack:
        physical_ids = pack["entity_id"].astype(str)
        physical_values = pack["feature_values"].astype(np.float32)
        if np.any(pack["entity_taxon"] != 9606):
            raise ValueError("physical taxonomy mismatch")
    with np.load(paths["bp"], allow_pickle=False) as pack:
        bp_ids = pack["entity_id"].astype(str)
        bp_values = pack["feature_values"].astype(np.float32)
        bp_present = pack["annotation_present"].astype(np.float32)
        if np.any(pack["entity_taxon"] != 9606):
            raise ValueError("BP taxonomy mismatch")
    physical_index = {gene: index for index, gene in enumerate(physical_ids)}
    bp_index = {gene: index for index, gene in enumerate(bp_ids)}
    needed = set(action_ids[np.concatenate((train, validation))])
    if needed - set(physical_index) or needed - set(bp_index):
        raise ValueError("intervention static feature coverage mismatch")
    physical_rows = np.stack([physical_values[physical_index[gene]] for gene in action_ids])
    bp_rows = np.stack([bp_values[bp_index[gene]] for gene in action_ids])
    present_rows = np.asarray([bp_present[bp_index[gene]] for gene in action_ids], dtype=np.float32)[:, None]
    augmented_rows = np.concatenate((physical_rows, bp_rows, present_rows), axis=1)
    validation_records = record_ids[validation]
    validation_actions = action_ids[validation]
    validation_contexts = context_index[validation]
    row_comparators: dict[str, np.ndarray] = {}
    for name, key in (("oldNystrom", "oldNystrom"), ("minimalControlV2", "minimalControlV2")):
        with np.load(paths[key], allow_pickle=False) as archive:
            if not validate_row_comparator(archive, validation_records, validation_actions, validation_contexts, query_ids):
                raise ValueError(f"{name} identity mismatch")
            row_comparators[name] = archive["mean"].astype(np.float32)
    with np.load(paths["bpLinear"], allow_pickle=False) as archive:
        if not np.array_equal(archive["record_ids"].astype(str), validation_records) or not np.array_equal(archive["query_ids"].astype(str), query_ids):
            raise ValueError("BP linear comparator identity mismatch")
        row_comparators["bpLinear"] = archive["physical1156_bp128_present1"].astype(np.float32)
    with np.load(paths["frozenPhysicalRidge"], allow_pickle=False) as archive:
        frozen_ridge = {key: archive[key] for key in archive.files}

    predictions = np.empty((len(validation), len(query_ids)), dtype=np.float32)
    reports: dict[str, object] = {}
    with threadpool_limits(limits=2):
        for context, context_id in enumerate(context_ids):
            train_rows = train[context_index[train] == context]
            validation_rows = validation[context_index[validation] == context]
            local = np.flatnonzero(validation_contexts == context)
            train_ids, train_x, train_y, train_counts = helper.collapse_rows(train_rows, action_ids, augmented_rows, targets)
            validation_ids, validation_x, _, validation_counts = helper.collapse_rows(validation_rows, action_ids, augmented_rows, targets)
            selected, cross_validation = helper.choose_alpha(train_ids, train_x, train_y, args.seed)
            prediction_by_gene, kernel, state, kernel_report = helper.fit_final(train_ids, train_x, train_y, validation_x, selected, args.seed)
            gene_index = {gene: index for index, gene in enumerate(validation_ids)}
            prediction_rows = np.stack([prediction_by_gene[gene_index[gene]] for gene in action_ids[validation_rows]])
            predictions[local] = prediction_rows
            model_path = output / f"model-context-{context}.npz"
            helper.save_model(model_path, kernel, state, selected, query_ids)
            _, candidate_gene, truth_gene = helper.collapse_prediction(prediction_rows, targets[validation_rows], action_ids[validation_rows])
            candidate_score = score_profiles(candidate_gene, truth_gene)
            comparator_rows = {
                "oldPhysicalNystrom": row_comparators["oldNystrom"][local],
                "originalFrozenPhysicalRidge": frozen_ridge[f"context{context}_physical"],
                "minimalControlV2": row_comparators["minimalControlV2"][local],
                "bpLinearRidge": row_comparators["bpLinear"][local],
            }
            comparator_scores: dict[str, object] = {}
            for name, values in comparator_rows.items():
                _, pred_gene, compare_truth = helper.collapse_prediction(values, targets[validation_rows], action_ids[validation_rows])
                if not np.array_equal(compare_truth, truth_gene):
                    raise ValueError("comparator truth alignment mismatch")
                comparator_scores[name] = score_profiles(pred_gene, truth_gene)
            mean_rows = np.broadcast_to(train_y.mean(axis=0, dtype=np.float64), (len(validation_rows), len(query_ids)))
            _, mean_gene, _ = helper.collapse_prediction(mean_rows, targets[validation_rows], action_ids[validation_rows])
            comparator_scores["fittingMean"] = score_profiles(mean_gene, truth_gene)
            old_nystrom = comparator_scores["oldPhysicalNystrom"]
            old_ridge = comparator_scores["originalFrozenPhysicalRidge"]
            candidate_r = candidate_score["independentlyQueryCenteredPearson"]
            checks = {
                "mseAtLeastOnePercentBelowOldPhysicalNystrom": candidate_score["geneProfileMse"] <= 0.99 * old_nystrom["geneProfileMse"],
                "mseAtLeastOnePercentBelowOriginalPhysicalRidge": candidate_score["geneProfileMse"] <= 0.99 * old_ridge["geneProfileMse"],
                "independentRAtLeastPoint10": candidate_r >= 0.10,
                "independentRNoRegressionVsOldPhysicalNystrom": candidate_r >= old_nystrom["independentlyQueryCenteredPearson"],
                "independentRNoRegressionVsOriginalPhysicalRidge": candidate_r >= old_ridge["independentlyQueryCenteredPearson"],
            }
            reports[context_id] = {
                "selectedAlpha": selected,
                "crossValidation": cross_validation,
                "finalKernel": kernel_report,
                "candidate": candidate_score,
                "comparators": comparator_scores,
                "bpCoverage": {"trainingAnnotatedGenes": int(sum(bp_present[bp_index[gene]] > 0 for gene in train_ids)), "trainingGenes": len(train_ids), "validationAnnotatedGenes": int(sum(bp_present[bp_index[gene]] > 0 for gene in validation_ids)), "validationGenes": len(validation_ids)},
                "trainingConstructCounts": {"minimum": int(train_counts.min()), "maximum": int(train_counts.max())},
                "validationConstructCounts": {"minimum": int(validation_counts.min()), "maximum": int(validation_counts.max())},
                "modelSha256": helper.sha256_file(model_path),
                "checks": checks,
                "passed": all(checks.values()),
            }
            print(json.dumps({"event": "context-finished", "context": context_id, "selectedAlpha": selected, "checks": checks}, sort_keys=True), flush=True)
            if time.monotonic() - started > args.max_seconds:
                raise TimeoutError("CPU time bound exceeded")
    prediction_path = output / "development-predictions.npz"
    np.savez_compressed(prediction_path, mean=predictions, record_ids=validation_records, action_ids=validation_actions, context_index=validation_contexts, query_ids=query_ids)
    report = {
        "schema": "slp.bp-nystrom-source-three-result/v1",
        "decision": "advance" if all(value["passed"] for value in reports.values()) else "reject",
        "contexts": reports,
        "elapsedSeconds": time.monotonic() - started,
        "protocolSha256": helper.sha256_file(protocol_path),
        "predictionsSha256": helper.sha256_file(prediction_path),
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
    parser.add_argument("--old-nystrom", default="results/slp11-transition/human-gwps-nystrom-rbf512-physical-seed731-v1/development-predictions.npz")
    parser.add_argument("--frozen-ridge", default="results/slp11-transition/physical-features-ridge-screen-v1/predictions.npz")
    parser.add_argument("--v2", default="results/slp11-transition/human-gwps-fixed-context-minimal-control-physical-state128-response32-seed731-v1/model/development-predictions.npz")
    parser.add_argument("--bp-linear", default="results/slp11-transition/human-gwps-bp-ridge-source3-seed731-v2/development-predictions.npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
