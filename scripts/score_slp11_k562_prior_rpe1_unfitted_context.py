#!/usr/bin/env python3
"""Freeze source-only transfers, then descriptively score RPE1 fitting genes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "modules/slp-1-1-count-static-ridge-v1/count_static_ridge.py"
K_RIDGE = ROOT / "results/slp11-transition/k562-essential-count-anchored-static-ridge-seed731-v1/model.npz"
K_ROSTER = ROOT / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1/roster-index.npz"
RPE_STATIC = ROOT / "data/derived/slp11-human-rpe1-essential-count-static/ensembl116-esm8m-shared-go-v1/rpe1-essential-count-static577.npz"
RPE_ROSTER = ROOT / "data/derived/slp11-human-rpe1-essential-count-static/ensembl116-esm8m-shared-go-v1/roster-index.npz"
RPE_CONTROL = ROOT / "data/derived/slp11-human-rpe1-essential-count-control/reconstruction-train-nt-gem-v1/gem-control-reference.npz"
PRIOR_FORECAST = ROOT / "results/slp11-transition/k562-count-prior-rpe1-unfitted-context-forecasts-v2/forecasts-before-rpe-fitting-outcomes.npz"
PRIOR_FREEZE = ROOT / "results/slp11-transition/k562-count-prior-rpe1-unfitted-context-forecasts-v2/FORECASTS-FROZEN-BEFORE-RPE-FITTING-OUTCOMES.json"
RPE_MOMENTS = ROOT / "data/derived/slp11-human-rpe1-essential-raw-cells-v1/fitting-action-moments.npz"
RPE_RIDGE = ROOT / "results/slp11-transition/rpe1-essential-count-anchored-static-ridge-seed731-v1/model.npz"
RPE_RIDGE_FREEZE = ROOT / "results/slp11-transition/rpe1-essential-count-anchored-static-ridge-seed731-v1/FROZEN-BEFORE-DEVELOPMENT.json"
OUTPUT = ROOT / "results/slp11-transition/k562-prior-rpe1-unfitted-context-fitting-score-v1"
SOURCE_PINS = {
    CORE: "1032eeff59382fae3874da9a389033192e113e0f5ac2c8d01f09f8441d969e62",
    K_RIDGE: "dbb669d2eb8d844ec9be7c88a2ed21f5592de434d1b2e916412bda4a52fe1cf3",
    K_ROSTER: "f2ee702a0714ca7f11f4fd2aa96f4c1825617c0e4f2bcdac42135cd0ba938d7b",
    RPE_STATIC: "621e1e9f0dffc740ef42382b1b2898f629edd5037e8a02d411e8d30e815ed816",
    RPE_ROSTER: "b9e1b169c2be4ac756e94f465009dc5bef80d06bc0652950c3cf6916d26d1e56",
    RPE_CONTROL: "c0c2eab217d00f9555b6ab5725cd2c49f56b1ecdf34b7af47f303eee9d1b8e20",
    PRIOR_FORECAST: "e6d32b387aeae7d6567ae82368a75c4ae991ba01dc0cadc999d4e88f47d98bdb",
    PRIOR_FREEZE: "83f9a6ef3ee73cc962a427de27befc7df21e1b5816395683283142647300ccc5",
}
SCORING_PINS = {
    RPE_MOMENTS: "d15def86aead06b0bc75ab63c77513735ec7c57d65012bff72f3947bc654895c",
    RPE_RIDGE: "bd144e36b5618c6225828501492edfa5449cef07442041c1d1cc20645b1473bc",
    RPE_RIDGE_FREEZE: "815e81bd9c6abc7e75e5821fd1725d960ffba3ec996eab0c702952398981a8ad",
}
ACTION_COUNT = 1666
RPE_QUERIES = 8749
COMMON_QUERIES = 7226
RPE_ONLY_QUERIES = 1523
SECONDS = 600.0


class TransferScoreError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, array in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, member.getvalue(), compresslevel=9)
    return output.getvalue()


def common_query_indices(k_ids: np.ndarray, rpe_ids: np.ndarray):
    """Map the exact intersection into RPE order and K source order."""
    k = np.asarray(k_ids).astype(str)
    rpe = np.asarray(rpe_ids).astype(str)
    if len(set(k.tolist())) != len(k) or len(set(rpe.tolist())) != len(rpe):
        raise TransferScoreError("query rosters must be unique")
    k_lookup = {gene: row for row, gene in enumerate(k)}
    rpe_rows = np.asarray([row for row, gene in enumerate(rpe) if gene in k_lookup], np.int64)
    k_rows = np.asarray([k_lookup[rpe[row]] for row in rpe_rows], np.int64)
    if len(rpe_rows) != COMMON_QUERIES:
        raise TransferScoreError("expected exact 7,226-query intersection")
    return rpe[rpe_rows], k_rows, rpe_rows


def per_gene_mse(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    target, pred = np.asarray(truth, np.float64), np.asarray(prediction, np.float64)
    if target.shape != pred.shape or target.ndim != 2 or not np.isfinite(target).all() or not np.isfinite(pred).all():
        raise TransferScoreError("finite aligned gene profiles required")
    return np.square(target - pred).mean(1, dtype=np.float64)


def protocol() -> dict[str, object]:
    return {
        "schema": "slp.k562-prior-rpe1-unfitted-context-fitting-score-protocol/v1",
        "purpose": "Descriptive unfitted-context transfer for one fixed K562 checkpoint with source-only comparators; not independent confirmation.",
        "hypothesis": "The fixed K562 prior may retain fitting-response structure under an RPE1 control context beyond source-only K562 ridge, K562 mean, and pure RPE1 control transfers.",
        "selection": "No tuning, filtering, checkpoint selection, architecture selection, or advancement decision. Report every prespecified method, gene stratum, and query subset.",
        "preOutcomeComparatorFreeze": {
            "K562RidgeTransfer": "Apply the frozen K562 ridge feature transform/coefficient/intercept to RPE1 raw action static577, map K output queries to the exact common RPE query order, then add the RPE1 GEM-composition control anchor.",
            "K562MeanTransfer": "Map the frozen K562 fitting residual mean to common RPE queries and add the same RPE1 control anchor.",
            "pureRPE1Control": "The same metadata-GEM-weighted RPE1 control anchor with zero intervention residual.",
            "RPEFittingMomentsRead": False,
        },
        "scoring": {
            "commonQueries": 7226,
            "rpeOnlyQueries": 1523,
            "fullQueries": 8749,
            "geneStrata": {"all": 1666, "k562SeenAction": 1443, "rpe1OnlyAction": 223},
            "metrics": "Equal-gene mean full-query MSE and anchor-subtracted independently query-centered per-gene Pearson, centered independently within each reported gene/query subset using count-static-ridge-v1.",
            "perGeneMse": True,
        },
        "supervisedRPEReferences": "The frozen RPE1 ridge and mean are reported descriptively after comparator freeze. They were fitted on RPE1 fitting moments and therefore have unequal target-context training access.",
        "priorQualification": "Prior forecast construction did not load RPE1 perturbation moments and the checkpoint was never fitted to RPE1. Earlier RPE1 molecular development work exists, so this is not an untouched-context confirmation.",
        "sourceOnlyPins": {str(path.relative_to(ROOT)): expected for path, expected in SOURCE_PINS.items()},
        "postFreezeScoringPins": {str(path.relative_to(ROOT)): expected for path, expected in SCORING_PINS.items()},
        "source": {"runnerSha256": sha256_file(Path(__file__).resolve()), "coreSha256": SOURCE_PINS[CORE]},
        "limits": {"cpuThreads": 2, "wallSeconds": SECONDS},
        "accessBoundary": {"RPEControlAndRoutingMetadata": True, "RPEFittingMomentsOnlyAfterComparatorFreeze": True, "RPERawCellDevelopment": False, "RPERawCellTest": False, "syntheticLethality": False},
    }


def prepare(output: Path):
    if output.exists():
        raise FileExistsError("immutable transfer score output already exists")
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    shutil.copy2(Path(__file__).resolve(), source / "runner.py")
    shutil.copy2(CORE, source / "count_static_ridge.py")
    value = protocol()
    (output / "protocol.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    receipt = {"schema": "slp.k562-prior-rpe1-transfer-score-prepared/v1", "protocolSha256": sha256_file(output / "protocol.json"), "RPEFittingMomentsRead": False, "RPERawCellDevelopmentRead": False, "RPERawCellTestRead": False}
    (output / "PREPARED.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def verify_prepared(output: Path):
    if not (output / "PREPARED.json").exists() or json.loads((output / "protocol.json").read_text()) != protocol():
        raise TransferScoreError("prepared protocol/source drift")
    for path, expected in SOURCE_PINS.items():
        if sha256_file(path) != expected:
            raise TransferScoreError(f"source-only input hash mismatch: {path}")
    if sha256_file(output / "source/runner.py") != sha256_file(Path(__file__).resolve()) or sha256_file(output / "source/count_static_ridge.py") != SOURCE_PINS[CORE]:
        raise TransferScoreError("prepared source copy drift")


def freeze_comparators(output: Path):
    verify_prepared(output)
    path = output / "source-only-comparators-before-rpe-fitting-outcomes.npz"
    if path.exists():
        raise FileExistsError("source-only comparators already frozen")
    started = time.perf_counter()
    core = load_module(output / "source/count_static_ridge.py", "rpe_source_transfer_core")
    k_model, static, roster, prior = map(load_npz, (K_RIDGE, RPE_STATIC, RPE_ROSTER, PRIOR_FORECAST))
    genes, queries = roster["fitting_action_ids"].astype(str), roster["query_ids"].astype(str)
    if not np.array_equal(genes, prior["action_ids"].astype(str)) or not np.array_equal(queries, prior["query_ids"].astype(str)):
        raise TransferScoreError("prior and RPE static axes differ")
    common_ids, k_query_rows, rpe_query_rows = common_query_indices(k_model["query_ids"], queries)
    entities = static["entity_id"].astype(str)
    lookup = {gene: row for row, gene in enumerate(entities)}
    raw_action = static["feature_values"][[lookup[gene] for gene in genes]]
    with threadpool_limits(2):
        k_residual = core.predict_residual(k_model, raw_action, str(k_model["selected_alpha"].item()))
        rpe_anchor = core.control_anchor(load_npz(RPE_CONTROL)["basal_rate"], prior["gem_cell_count"])
        anchor_common = rpe_anchor[:, rpe_query_rows]
        k_ridge = core.absolute_prediction(anchor_common, k_residual[:, k_query_rows])
        k_mean = core.absolute_prediction(anchor_common, np.broadcast_to(k_model["target_mean"][k_query_rows], anchor_common.shape))
    if time.perf_counter() - started > SECONDS:
        raise TimeoutError("source-only comparator freeze exceeded 600 seconds")
    arrays = {
        "schema": np.asarray("slp.k562-source-only-rpe1-context-comparators/v1"),
        "action_ids": genes,
        "common_query_ids": common_ids,
        "rpe_common_query_index": rpe_query_rows,
        "k562_common_query_index": k_query_rows,
        "is_k562_seen_action": prior["is_k562_seen_action"],
        "rpe_control_anchor_common": anchor_common,
        "k562_ridge_transfer_prediction": k_ridge,
        "k562_mean_transfer_prediction": k_mean,
        "k562_ridge_model_sha256": np.asarray(SOURCE_PINS[K_RIDGE]),
        "prior_forecast_sha256": np.asarray(SOURCE_PINS[PRIOR_FORECAST]),
    }
    path.write_bytes(deterministic_npz(arrays))
    receipt = {
        "schema": "slp.k562-source-only-rpe1-context-comparator-freeze/v1",
        "protocolSha256": sha256_file(output / "protocol.json"),
        "comparatorForecastSha256": sha256_file(path),
        "actions": ACTION_COUNT, "commonQueries": COMMON_QUERIES,
        "k562SeenActions": int(prior["is_k562_seen_action"].sum()),
        "rpe1OnlyActions": int((~prior["is_k562_seen_action"]).sum()),
        "RPEFittingMomentsRead": False, "RPERawCellDevelopmentRead": False, "RPERawCellTestRead": False,
        "seconds": time.perf_counter() - started,
    }
    (output / "SOURCE-ONLY-COMPARATORS-FROZEN-BEFORE-RPE-FITTING-OUTCOMES.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def score(output: Path):
    comparator_path = output / "source-only-comparators-before-rpe-fitting-outcomes.npz"
    freeze_path = output / "SOURCE-ONLY-COMPARATORS-FROZEN-BEFORE-RPE-FITTING-OUTCOMES.json"
    if not comparator_path.exists() or not freeze_path.exists() or (output / "report.json").exists():
        raise TransferScoreError("source-only comparator freeze must precede scoring")
    freeze = json.loads(freeze_path.read_text())
    if sha256_file(comparator_path) != freeze["comparatorForecastSha256"] or freeze["RPEFittingMomentsRead"] is not False:
        raise TransferScoreError("source-only comparator freeze mismatch")
    for path, expected in SCORING_PINS.items():
        if sha256_file(path) != expected:
            raise TransferScoreError(f"post-freeze scoring input hash mismatch: {path}")
    started = time.perf_counter()
    core = load_module(output / "source/count_static_ridge.py", "rpe_transfer_scoring_core")
    comparator, prior, moments, rpe_model, static, roster = map(
        load_npz, (comparator_path, PRIOR_FORECAST, RPE_MOMENTS, RPE_RIDGE, RPE_STATIC, RPE_ROSTER)
    )
    genes, query_ids = moments["action_ids"].astype(str), moments["query_ids"].astype(str)
    if not np.array_equal(genes, prior["action_ids"].astype(str)) or not np.array_equal(query_ids, prior["query_ids"].astype(str)) or not np.array_equal(moments["gem_cell_count"], prior["gem_cell_count"]):
        raise TransferScoreError("scoring moments differ from frozen metadata identities/weights")
    truth = core.response_from_cp10k_moments(moments["cp10k_sum"], moments["cell_count"])
    control = load_npz(RPE_CONTROL)
    anchor = core.control_anchor(control["basal_rate"], moments["gem_cell_count"])
    entities = static["entity_id"].astype(str)
    lookup = {gene: row for row, gene in enumerate(entities)}
    raw_action = static["feature_values"][[lookup[gene] for gene in genes]]
    rpe_residual = core.predict_residual(rpe_model, raw_action, str(rpe_model["selected_alpha"].item()))
    rpe_ridge = core.absolute_prediction(anchor, rpe_residual)
    rpe_mean = core.absolute_prediction(anchor, np.broadcast_to(rpe_model["target_mean"], anchor.shape))
    common_rows = comparator["rpe_common_query_index"].astype(np.int64)
    rpe_only_rows = np.flatnonzero(~np.isin(query_ids, comparator["common_query_ids"].astype(str)))
    if len(rpe_only_rows) != RPE_ONLY_QUERIES:
        raise TransferScoreError("RPE-only query stratum drift")
    query_subsets = {
        "common7226": common_rows,
        "rpeOnly1523": rpe_only_rows,
        "full8749": np.arange(RPE_QUERIES, dtype=np.int64),
    }
    action_subsets = {
        "all1666": np.ones(ACTION_COUNT, dtype=np.bool_),
        "k562Seen1443": prior["is_k562_seen_action"].astype(np.bool_),
        "rpe1Only223": ~prior["is_k562_seen_action"].astype(np.bool_),
    }
    methods_full = {
        "k562CountPrior": prior["mean_log1p_cp10k"],
        "pureRpeControl": anchor,
        "rpeTrainedRidgeDescriptive": rpe_ridge,
        "rpeTrainedMeanDescriptive": rpe_mean,
    }
    common_methods = {
        **{name: values[:, common_rows] for name, values in methods_full.items()},
        "k562RidgeTransfer": comparator["k562_ridge_transfer_prediction"],
        "k562MeanTransfer": comparator["k562_mean_transfer_prediction"],
    }
    report_metrics: dict[str, object] = {}
    per_gene: dict[str, np.ndarray] = {
        "schema": np.asarray("slp.k562-prior-rpe1-unfitted-context-per-gene-mse/v1"),
        "action_ids": genes,
        "is_k562_seen_action": prior["is_k562_seen_action"],
    }
    for query_label, query_rows in query_subsets.items():
        truth_subset, anchor_subset = truth[:, query_rows], anchor[:, query_rows]
        methods = common_methods if query_label == "common7226" else {
            name: values[:, query_rows] for name, values in methods_full.items()
        }
        report_metrics[query_label] = {}
        for method, prediction in methods.items():
            per_gene[f"{query_label}_{method}"] = per_gene_mse(truth_subset, prediction)
        for action_label, action_mask in action_subsets.items():
            report_metrics[query_label][action_label] = {
                method: core.centered_landscape_score(
                    truth_subset[action_mask], prediction[action_mask], anchor_subset[action_mask]
                )
                for method, prediction in methods.items()
            }
    if time.perf_counter() - started > SECONDS:
        raise TimeoutError("descriptive scoring exceeded 600 seconds")
    per_gene_path = output / "per-gene-mse.npz"
    per_gene_path.write_bytes(deterministic_npz(per_gene))
    report = {
        "schema": "slp.k562-prior-rpe1-unfitted-context-fitting-score/v1",
        "protocolSha256": sha256_file(output / "protocol.json"),
        "sourceOnlyComparatorForecastSha256": sha256_file(comparator_path),
        "priorForecastSha256": SOURCE_PINS[PRIOR_FORECAST],
        "rpeFittingMomentsSha256": SCORING_PINS[RPE_MOMENTS],
        "rpeTrainedRidgeSha256": SCORING_PINS[RPE_RIDGE],
        "perGeneMseSha256": sha256_file(per_gene_path),
        "metrics": report_metrics,
        "interpretationBoundary": "Descriptive unfitted-context transfer for a fixed K562 checkpoint. RPE-trained ridge/mean have unequal target-context fitting access. Earlier RPE molecular development exists; this is not independent confirmation.",
        "seconds": time.perf_counter() - started,
        "RPEFittingMomentsOpenedOnlyAfterSourceComparatorFreeze": True,
        "RPERawCellDevelopmentRead": False, "RPERawCellTestRead": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=OUTPUT)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--freeze-comparators", action="store_true")
    p.add_argument("--score", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if sum((args.prepare, args.freeze_comparators, args.score)) != 1:
        raise SystemExit("select exactly one phase")
    value = prepare(args.output_dir) if args.prepare else freeze_comparators(args.output_dir) if args.freeze_comparators else score(args.output_dir)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
