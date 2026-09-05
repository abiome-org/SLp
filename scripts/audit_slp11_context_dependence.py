#!/usr/bin/env python3
"""Audit learned measured-context dependence in a fixed human world model."""

from __future__ import annotations

import os

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "4"

import argparse
import hashlib
import importlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

DATA_SHA256 = "88de5164fca4e2504ac5b459ab4226c161eb586dd04700d5784da4bb53048659"
FEATURE_SHA256 = "b3de49e18d3c75676985b8790d1ce85de0d87d526bbd7c0c5b555828a1fb11a0"
VALUE_SPACE = "author-per-gemgroup-core-control-z-score-pseudobulk-mean-v1"
ARMS = ("matched", "swapped", "masked")


class ContextAuditError(ValueError):
    """Raised when the fixed context audit contract is violated."""


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denominator = math.sqrt(float(x @ x) * float(y @ y))
    if denominator <= np.finfo(np.float64).eps:
        return None
    return float((x @ y) / denominator)


def gene_metrics(
    evaluate,
    prediction: np.ndarray,
    truth: np.ndarray,
    observed: np.ndarray,
    action_ids: np.ndarray,
    reference: np.ndarray,
    scale: np.ndarray,
) -> dict[str, object]:
    groups: dict[str, list[int]] = {}
    for row, action in enumerate(action_ids):
        groups.setdefault(str(action), []).append(row)
    reports = [
        evaluate(
            prediction[rows],
            truth[rows],
            observed[rows],
            reference,
            scale[rows],
            value_space=VALUE_SPACE,
        )
        for rows in groups.values()
    ]
    result = evaluate(
        prediction,
        truth,
        observed,
        reference,
        scale,
        value_space=VALUE_SPACE,
    )
    for metric in (
        "nll",
        "mse",
        "profile_pearson_mean",
        "profile_centroid_adjusted_pearson_mean",
    ):
        values = [float(report[metric]) for report in reports if np.isfinite(report[metric])]
        result["gene_macro_" + metric] = float(np.mean(values)) if values else math.nan
    result["intervention_genes"] = len(groups)
    return result


def change_summary(
    candidate: np.ndarray,
    matched: np.ndarray,
    reference: np.ndarray,
    observed: np.ndarray,
) -> dict[str, float | int]:
    candidate_residual = candidate - reference
    matched_residual = matched - reference
    difference = candidate_residual - matched_residual
    row_rms = np.sqrt(
        np.where(observed, np.square(difference), 0.0).sum(axis=1)
        / observed.sum(axis=1)
    )
    return {
        "definition": "candidate-minus-matched predicted residual, after correct context training mean",
        "observedValues": int(observed.sum()),
        "rmsAllObservedValues": float(np.sqrt(np.square(difference[observed]).mean())),
        "recordMacroRms": float(row_rms.mean()),
        "recordMedianRms": float(np.median(row_rms)),
    }


def same_gene_context_difference(
    predictions: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
    validation: np.ndarray,
    references: np.ndarray,
) -> dict[str, object]:
    action_ids = data["action_ids"].astype(str)
    contexts = data["context_index"]
    context_genes = [
        set(action_ids[validation[contexts[validation] == context]]) for context in (0, 1)
    ]
    shared = sorted(context_genes[0] & context_genes[1])
    report: dict[str, object] = {
        "definition": (
            "for each shared validation intervention gene, average duplicate records "
            "within context after subtracting that context training mean, then correlate "
            "K562-minus-RPE1 predicted and observed query profiles"
        ),
        "sharedInterventionGenes": len(shared),
        "k562ValidationGenes": len(context_genes[0]),
        "rpe1ValidationGenes": len(context_genes[1]),
        "arms": {},
    }
    true_profiles: dict[tuple[int, str], np.ndarray] = {}
    masks: dict[tuple[int, str], np.ndarray] = {}
    predicted_profiles: dict[tuple[str, int, str], np.ndarray] = {}
    for context in (0, 1):
        positions = np.flatnonzero(contexts[validation] == context)
        rows = validation[positions]
        local_actions = action_ids[rows]
        for gene in shared:
            selected = local_actions == gene
            mask = data["observed"][rows[selected]].all(axis=0)
            masks[(context, gene)] = mask
            true_profiles[(context, gene)] = (
                data["targets"][rows[selected]].astype(np.float64).mean(axis=0)
                - references[context]
            )
            for arm in ARMS:
                predicted_profiles[(arm, context, gene)] = (
                    predictions[arm][positions[selected]].astype(np.float64).mean(axis=0)
                    - references[context]
                )
    for arm in ARMS:
        correlations = []
        pooled_prediction = []
        pooled_truth = []
        supported_values = 0
        for gene in shared:
            mask = masks[(0, gene)] & masks[(1, gene)]
            predicted_difference = (
                predicted_profiles[(arm, 0, gene)] - predicted_profiles[(arm, 1, gene)]
            )
            true_difference = true_profiles[(0, gene)] - true_profiles[(1, gene)]
            correlation = pearson(predicted_difference[mask], true_difference[mask])
            if correlation is not None:
                correlations.append(correlation)
            pooled_prediction.append(predicted_difference[mask])
            pooled_truth.append(true_difference[mask])
            supported_values += int(mask.sum())
        pooled = pearson(np.concatenate(pooled_prediction), np.concatenate(pooled_truth))
        report["arms"][arm] = {
            "profilePearsonMean": float(np.mean(correlations)) if correlations else None,
            "profilePearsonDefinedGenes": len(correlations),
            "profilePearsonUndefinedGenes": len(shared) - len(correlations),
            "supportedGeneQueryDifferences": supported_values,
            "pooledGeneQueryPearson": pooled,
        }
    return report


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    data_path = args.data.resolve(strict=True)
    feature_path = args.features.resolve(strict=True)
    run_dir = args.run.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise ContextAuditError(f"output directory already exists: {output}")
    if sha256(data_path) != DATA_SHA256 or sha256(feature_path) != FEATURE_SHA256:
        raise ContextAuditError("development data or static feature SHA-256 drift")
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    if (
        protocol["data_sha256"] != DATA_SHA256
        or protocol["features_sha256"] != FEATURE_SHA256
        or protocol.get("test_accessed")
        or protocol["args"].get("reference_kind", "mean") != "mean"
        or (run_dir / "linear-reference.npz").exists()
    ):
        raise ContextAuditError("candidate is not the fixed mean-reference development run")
    with np.load(data_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    if len(data["split_test"]) or str(data["target_value_space"].item()) != VALUE_SPACE:
        raise ContextAuditError("only the corrected development bundle is permitted")
    validation = data["split_validation"]

    with np.load(feature_path, allow_pickle=False) as archive:
        keys = tuple(
            (int(taxon), str(entity))
            for taxon, entity in zip(
                archive["entity_taxon"], archive["entity_id"], strict=True
            )
        )
        feature_values = archive["feature_values"]
    lookup = dict(zip(keys, feature_values, strict=True))
    actions = np.stack(
        [lookup[(9606, str(gene))] for gene in data["action_ids"][validation]]
    )

    source_path = str((run_dir / "source").resolve())
    sys.path.insert(0, source_path)
    Predictor = importlib.import_module("inference").Predictor
    evaluate = importlib.import_module("transition_baselines").evaluate
    predictor = Predictor(run_dir, device="cpu")
    with np.load(run_dir / "reference.npz", allow_pickle=False) as archive:
        saved = {name: archive[name] for name in archive.files}
    if not np.array_equal(saved["query_ids"], data["query_ids"]):
        raise ContextAuditError("candidate and development query identities differ")

    output.mkdir(parents=True, exist_ok=False)
    audit_protocol = {
        "schema": "slp.human-context-dependence-development/v1",
        "scope": "fixed corrected-human development-validation context ablation",
        "hypothesis": (
            "The saved world model uses measured context tokens beyond its fixed "
            "context-specific output reference."
        ),
        "arms": {
            "matched": "measured token values from the record's context",
            "swapped": "measured token values from the other context",
            "masked": "all measured context tokens masked",
        },
        "fixedAcrossArms": [
            "checkpoint", "action features", "query features", "correct context reference",
            "correct context reference scale", "per-record mean-OOF exposure uncertainty",
            "validation rows and outcomes",
        ],
        "inputs": {
            "development": {"path": str(data_path), "sha256": DATA_SHA256},
            "features": {"path": str(feature_path), "sha256": FEATURE_SHA256},
            "candidate": {"path": str(run_dir),
                          "checkpointSha256": sha256(run_dir / "model.safetensors"),
                          "referenceSha256": sha256(run_dir / "reference.npz"),
                          "exposureSha256": sha256(run_dir / "exposure-uncertainty.npz")},
            "auditSourceSha256": sha256(Path(__file__)),
        },
        "modelRefit": False,
        "calibrationRefit": False,
        "testArtifactAccessed": False,
        "benchmarkAccessed": False,
        "device": "cpu",
    }
    write_json(output / "protocol.json", audit_protocol)

    contexts = data["context_index"][validation]
    query_indices = np.arange(len(data["query_ids"]), dtype=np.int64)
    measurement_scale = predictor.measurement_scales(
        data["num_cells_filtered"][validation], contexts, query_indices
    )
    predictions = {
        arm: np.empty((len(validation), len(query_indices)), dtype=np.float32)
        for arm in ARMS
    }
    for start in range(0, len(validation), args.batch_size):
        positions = np.arange(start, min(start + args.batch_size, len(validation)))
        local_context = contexts[positions]
        base = {
            "action_features": actions[positions],
            "query_features": saved["query_features"],
            "reference": saved["reference"][local_context],
            "reference_scale": saved["reference_scale"][local_context],
            "measurement_scale": measurement_scale[positions],
            "context_features": np.broadcast_to(
                saved["context_features"],
                (len(positions), *saved["context_features"].shape),
            ),
        }
        values = {
            "matched": saved["context_values"][local_context],
            "swapped": saved["context_values"][1 - local_context],
            "masked": saved["context_values"][local_context],
        }
        masks = {
            "matched": np.ones(values["matched"].shape, dtype=np.bool_),
            "swapped": np.ones(values["swapped"].shape, dtype=np.bool_),
            "masked": np.zeros(values["masked"].shape, dtype=np.bool_),
        }
        for arm in ARMS:
            result = predictor.predict(
                **base, context_values=values[arm], context_mask=masks[arm]
            )
            if not np.array_equal(result["scale"], measurement_scale[positions]):
                raise ContextAuditError("measurement uncertainty changed across context arms")
            predictions[arm][positions] = result["mean"]

    context_reports: dict[str, object] = {}
    original_report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    reproduction_drift = {}
    for context, context_id_value in enumerate(data["context_ids"]):
        context_id = str(context_id_value)
        selected = contexts == context
        rows = validation[selected]
        reference = saved["reference"][context]
        scale = measurement_scale[selected]
        arm_metrics = {
            arm: gene_metrics(
                evaluate, predictions[arm][selected], data["targets"][rows],
                data["observed"][rows], data["action_ids"][rows], reference, scale,
            )
            for arm in ARMS
        }
        matched = arm_metrics["matched"]
        expected = original_report["results"][context_id]["world"]
        reproduction_drift[context_id] = {
            "geneMacroNll": matched["gene_macro_nll"] - expected["gene_macro_nll"],
            "geneMacroAdjustedPearson": (
                matched["gene_macro_profile_centroid_adjusted_pearson_mean"]
                - expected["gene_macro_profile_centroid_adjusted_pearson_mean"]
            ),
        }
        context_reports[context_id] = {
            "records": int(selected.sum()),
            "interventionGenes": len(set(data["action_ids"][rows].tolist())),
            "arms": {
                arm: {
                    "geneMacroNll": metrics["gene_macro_nll"],
                    "geneMacroMse": metrics["gene_macro_mse"],
                    "geneMacroAdjustedPearson": metrics[
                        "gene_macro_profile_centroid_adjusted_pearson_mean"
                    ],
                    "nllDeltaVsMatched": (
                        matched["gene_macro_nll"] - metrics["gene_macro_nll"]
                    ),
                    "adjustedPearsonDeltaVsMatched": (
                        metrics["gene_macro_profile_centroid_adjusted_pearson_mean"]
                        - matched["gene_macro_profile_centroid_adjusted_pearson_mean"]
                    ),
                }
                for arm, metrics in arm_metrics.items()
            },
            "predictedResidualChange": {
                arm: change_summary(
                    predictions[arm][selected], predictions["matched"][selected],
                    reference, data["observed"][rows],
                )
                for arm in ("swapped", "masked")
            },
        }
    if any(abs(value) > 1e-7 for drift in reproduction_drift.values() for value in drift.values()):
        raise ContextAuditError("matched context arm does not reproduce the saved report")

    report = {
        "protocol": audit_protocol,
        "contexts": context_reports,
        "sameGeneCrossContextDifference": same_gene_context_difference(
            predictions, data, validation, saved["reference"]
        ),
        "matchedReportReproductionDrift": reproduction_drift,
        "interpretationBoundary": (
            "This development ablation measures dependence on two saved measured-context "
            "token profiles. It does not establish transfer to a new context."
        ),
        "elapsedSeconds": time.monotonic() - started,
        "testArtifactAccessed": False,
        "benchmarkAccessed": False,
    }
    write_json(output / "report.json", report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--data", type=Path,
        default=Path("data/derived/slp11-human/replogle-k562-rpe1-author-normalized-development-v2.npz"),
    )
    result.add_argument(
        "--features", type=Path,
        default=Path("data/derived/slp11-human-static-fusion/esm2-t6-plus-go-svd-v1/human-static-esm-go-features.npz"),
    )
    result.add_argument(
        "--run", type=Path,
        default=Path("results/slp11-transition/human-normalized-fusion-response32-exposure-seed731-v1"),
    )
    result.add_argument(
        "--output", type=Path,
        default=Path("results/slp11-transition/human-context-dependence-v1"),
    )
    result.add_argument("--batch-size", type=int, default=32)
    return result


def main() -> int:
    torch.set_num_threads(4)
    report = run(parser().parse_args())
    print(json.dumps({"contexts": report["contexts"],
                      "sameGene": report["sameGeneCrossContextDifference"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
