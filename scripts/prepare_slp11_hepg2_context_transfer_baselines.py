#!/usr/bin/env python3
"""Freeze HepG2 context-transfer baseline forecasts without reading perturbed X."""

from __future__ import annotations

import os

for _variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "2"

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from context_transfer_baselines import (
    control_context_distances,
    equal_source_fitting_centroid,
    population_roster,
    same_gene_source_response_forecast,
)
from transition_baselines import fit_ridge

DATA_SHA256 = "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c"
FEATURE_SHA256 = "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7"
HEPG2_CONTEXT_SHA256 = "382626401ee38e8d5084ac9f86ffc44bd10408826fb85a94ede8eb908cdf5b27"
HEPG2_SOURCE_SHA256 = "e1ad7c3c5a201c861a207a858aa7e59f5e6ac1955674c415f7de0d1dadadb52e"
HEPG2_SOURCE_BYTES = 5_614_460_941
METADATA_AUDIT_SHA256 = "f6f1c459f47a7ea9ba792e177a60dd38843a862c7a4cd5c4b738b7c32ae6f4f7"
FIXED_PANEL_SHA256 = "046891d3ceb0766e3fd09441677d6ae078fa7ac7d81ddb1f1c30866007d0d959"
RIDGE_ALPHA = 10_000.0
SEED = 731
ENSG_RE = re.compile(r"^ENSG[0-9]+$")
BASELINE_NAMES = (
    "zero_control",
    "equal_source_fitting_centroid",
    "equal_source_average_physical_ridge",
    "control_nearest_source_physical_ridge",
    "same_gene_source_response_mean",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values: np.ndarray) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> None:
    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(entry) for entry in item]
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    path.write_text(
        json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )


def categorical(group: h5py.Group, name: str) -> np.ndarray:
    if name not in group or f"__categories/{name}" not in group:
        raise ValueError(f"obs/{name} categorical metadata is absent")
    categories = np.asarray(group[f"__categories/{name}"][...]).astype(str)
    codes = np.asarray(group[name][...], dtype=np.int64)
    if np.any(codes < 0) or np.any(codes >= categories.size):
        raise ValueError(f"obs/{name} category codes are invalid")
    return categories[codes]


def load_hepg2_metadata(source_path: Path, audit_path: Path):
    source = source_path.resolve(strict=True)
    if source.is_symlink() or source.stat().st_size != HEPG2_SOURCE_BYTES:
        raise ValueError("HepG2 source path or byte length drifted")
    if sha256_file(audit_path) != METADATA_AUDIT_SHA256:
        raise ValueError("HepG2 metadata audit SHA-256 drifted")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit["source"]["sha256"] != HEPG2_SOURCE_SHA256
        or audit["source"]["bytes"] != HEPG2_SOURCE_BYTES
        or audit["outcomeAccess"]["matrixValuesRead"]
    ):
        raise ValueError("HepG2 metadata audit source contract drifted")
    # Deliberately never name or index source["X"] in this forecast phase.
    with h5py.File(source, "r") as archive:
        obs = archive["obs"]
        action = categorical(obs, "gene_id")
        population = categorical(obs, "gene_transcript")
        construct = categorical(obs, "sgID_AB")
        transcript = categorical(obs, "transcript")
        source_queries = np.asarray(archive["var/gene_id"][...]).astype(str)
    targeted = np.asarray([ENSG_RE.fullmatch(value) is not None for value in action])
    roster = population_roster(
        action[targeted], population[targeted], construct[targeted], transcript[targeted]
    )
    return roster, source_queries, int(np.count_nonzero(targeted))


def load_npz(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def load_feature_rows(
    path: Path, development_actions: np.ndarray, forecast_actions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if sha256_file(path) != FEATURE_SHA256:
        raise ValueError("physical feature SHA-256 drifted")
    with np.load(path, allow_pickle=False) as archive:
        values = archive["feature_values"]
        taxa = archive["entity_taxon"]
        identifiers = archive["entity_id"]
    if values.ndim != 2 or values.shape[1] != 1156 or not np.isfinite(values).all():
        raise ValueError("physical feature contract drifted")
    keys = [(int(taxon), str(identifier)) for taxon, identifier in zip(taxa, identifiers, strict=True)]
    if len(keys) != len(set(keys)):
        raise ValueError("physical feature identities are duplicated")
    lookup = {key: index for index, key in enumerate(keys)}

    def align(actions: np.ndarray) -> np.ndarray:
        requested = [(9606, str(action)) for action in actions]
        missing = sorted(set(requested) - set(lookup))
        if missing:
            raise ValueError(f"physical features lack requested actions: {missing[:8]}")
        rows = np.asarray([lookup[key] for key in requested], dtype=np.int64)
        return values[rows].astype(np.float64, copy=False)

    return align(development_actions), align(forecast_actions), values.shape


def write_population_prediction(
    path: Path, unique_prediction: np.ndarray, population_action_index: np.ndarray
) -> dict[str, object]:
    shape = (population_action_index.size, unique_prediction.shape[1])
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    for start in range(0, shape[0], 128):
        stop = min(start + 128, shape[0])
        output[start:stop] = unique_prediction[population_action_index[start:stop]]
    output.flush()
    del output
    return {"path": str(path), "shape": list(shape), "dtype": "float32", "sha256": sha256_file(path)}


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    data = load_npz(args.data.resolve(strict=True), DATA_SHA256)
    hepg2 = load_npz(args.hepg2_context.resolve(strict=True), HEPG2_CONTEXT_SHA256)
    roster, source_queries, targeted_cells = load_hepg2_metadata(
        args.hepg2_source, args.metadata_audit.resolve(strict=True)
    )
    if (
        len(data["split_test"]) != 0
        or data["query_ids"].shape != (7036,)
        or not data["observed"][data["split_train"]].all()
        or not np.array_equal(data["query_ids"], hepg2["query_ids"])
        or int(hepg2["perturbed_expression_rows_read"]) != 0
        or str(data["context_value_space"].item()) != str(hepg2["context_value_space"].item())
        or str(hepg2["fixed_panel_query_sha256"].item()) != FIXED_PANEL_SHA256
    ):
        raise ValueError("development/HepG2 control-only contract drifted")
    common_mask = data["context_basal_observed"].all(axis=0)
    if int(common_mask.sum()) != 6789 or not np.array_equal(
        common_mask, hepg2["context_basal_observed"][0]
    ):
        raise ValueError("fixed 6789-query context panel drifted")

    unique_actions = np.asarray(sorted(set(roster.action_ids.tolist())))
    action_lookup = {action: index for index, action in enumerate(unique_actions)}
    population_action_index = np.asarray(
        [action_lookup[str(action)] for action in roster.action_ids], dtype=np.int64
    )
    fitting = data["split_train"]
    fitting_genes = set(data["action_ids"][fitting].astype(str).tolist())
    unique_seen = np.asarray([action in fitting_genes for action in unique_actions])
    population_seen = unique_seen[population_action_index]
    source_query_set = set(source_queries.tolist())
    measured_queries = np.asarray(
        [str(query) in source_query_set for query in data["query_ids"]], dtype=bool
    )
    development_features, forecast_features, feature_shape = load_feature_rows(
        args.features.resolve(strict=True), data["action_ids"], unique_actions
    )
    distances = control_context_distances(
        data["context_basal_expression"], hepg2["context_basal_expression"][0], common_mask
    )
    nearest_context = int(np.argmin(distances))

    output.mkdir(parents=True, exist_ok=False)
    source_output = output / "source"
    prediction_output = output / "predictions"
    source_output.mkdir()
    prediction_output.mkdir()
    source_files = (
        Path(__file__), MODULE / "context_transfer_baselines.py",
        MODULE / "transition_baselines.py",
    )
    source_hashes = {}
    for path in source_files:
        shutil.copyfile(path, source_output / path.name)
        source_hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    roster_path = output / "forecast-roster.npz"
    np.savez_compressed(
        roster_path,
        schema=np.asarray("slp.hepg2-context-transfer-forecast-roster/v1"),
        population_ids=roster.population_ids,
        source_construct_ids=roster.construct_ids,
        source_transcript_labels=roster.transcript_labels,
        action_ids=roster.action_ids,
        fitting_gene_seen=population_seen,
        query_ids=data["query_ids"],
        source_query_measured=measured_queries,
        context_control_query_observed=common_mask,
    )
    roster_hash = sha256_file(roster_path)
    unique_roster_hash = sha256_lines(unique_actions)
    query_roster_hash = sha256_lines(data["query_ids"].astype(str))
    centroid_path = output / "training-centering.npz"
    protocol = {
        "schema": "slp.hepg2-context-transfer-baseline-forecast-protocol/v1",
        "phase": "baseline-forecast-only",
        "frozenBeforeBaselineFitting": True,
        "scientificStatus": (
            "new-context diagnostic only; does not rewrite a failed development gate, "
            "establish promotion, or support a state-of-the-art claim"
        ),
        "candidateRequirement": {
            "family": "minimal-control-v2 physical1156 state128",
            "sameStaticModalitiesAsRidge": True,
            "trainingDataSha256": DATA_SHA256,
            "worldForecastMustBeSavedAndFrozenBeforePerturbedOutcomeAccess": True,
        },
        "strata": {
            "seen": "HepG2 action ENSG occurs in three-context split_train fitting outcomes",
            "unseen": "HepG2 action ENSG has no three-context split_train fitting outcome",
            "selectedFromMetadataAndFittingRosterOnly": True,
            "uniqueGeneCounts": {
                "seen": int(unique_seen.sum()), "unseen": int((~unique_seen).sum())
            },
            "populationCounts": {
                "seen": int(population_seen.sum()), "unseen": int((~population_seen).sum())
            },
        },
        "primaryAdvancementRule": {
            "appliesSeparatelyToBothSeenAndUnseenStrata": True,
            "geneMacroMseImprovementAgainstEveryBaseline": 0.02,
            "independentlyGeneCenteredProfilePearsonMinimum": 0.10,
            "independentlyGeneCenteredProfilePearsonNonregression": (
                "world correlation must be at least every defined nonconstant baseline"
            ),
            "averagesAcrossStrataCannotPassAFailedStratum": True,
        },
        "metrics": {
            "geneBalance": "collapse records to one missing-aware mean profile per stable action ENSG",
            "primaryMse": "mean query MSE per gene, then macro mean over genes",
            "primaryCorrelation": (
                "within each stratum, subtract prediction and truth's own gene-balanced "
                "per-query mean independently, correlate profiles per gene, then gene macro mean"
            ),
            "primaryCenteringUse": "metric computation only; never returned to a forecast",
            "secondary": ["training-centroid-adjusted profile Pearson", "uncentered profile Pearson"],
            "trainingCentroid": (
                "gene-balanced fitting mean inside each source context, then equal average of "
                "the three source centroids"
            ),
            "missingSupport": "never imputed; metrics use explicit query masks",
            "constantPrimaryCorrelation": "undefined and excluded from nonregression comparisons",
            "bootstrap": {
                "samples": 1000, "seed": SEED, "unit": "stable action gene block",
                "intervals": "descriptive percentile 95%; no extra decision threshold",
            },
            "uncertainty": "none; point forecasts only",
        },
        "baselines": {
            "zero_control": "constant zero in the frozen target control-z-score space",
            "equal_source_fitting_centroid": (
                "constant equal-source, within-source gene-balanced fitting centroid"
            ),
            "equal_source_average_physical_ridge": (
                "equal mean of three context-local full physical1156 ridge forecasts"
            ),
            "control_nearest_source_physical_ridge": {
                "forecast": "full physical1156 ridge from one source context",
                "selection": (
                    "Euclidean distance after each context vector is independently centered "
                    "and scaled across the identical 6789 fixed control tokens"
                ),
                "distances": {
                    str(name): float(distance)
                    for name, distance in zip(data["context_ids"], distances, strict=True)
                },
                "selectedContextIndex": nearest_context,
                "selectedContextId": str(data["context_ids"][nearest_context]),
            },
            "same_gene_source_response_mean": (
                "for seen genes, mean records within each available fitting source then equal "
                "mean across sources; unseen genes receive the constant equal-source centroid"
            ),
            "ridgeAlpha": RIDGE_ALPHA,
            "ridgeFitRows": "original split_train rows within each source context only",
        },
        "inputs": {
            "development": {"path": str(args.data), "sha256": DATA_SHA256},
            "physicalFeatures": {
                "path": str(args.features), "sha256": FEATURE_SHA256,
                "shape": list(feature_shape), "dimensions": 1156,
            },
            "hepg2ControlDescriptor": {
                "path": str(args.hepg2_context), "sha256": HEPG2_CONTEXT_SHA256,
                "fixedPanelSha256": FIXED_PANEL_SHA256,
            },
            "hepg2Source": {
                "path": str(args.hepg2_source), "sha256FromPinnedMetadataAudit": HEPG2_SOURCE_SHA256,
                "bytesVerifiedNow": HEPG2_SOURCE_BYTES,
                "fullFileRehashSkippedToAvoidExpressionByteAccess": True,
            },
            "metadataAudit": {"path": str(args.metadata_audit), "sha256": METADATA_AUDIT_SHA256},
        },
        "frozenRosters": {
            "forecastRosterPath": str(roster_path), "forecastRosterSha256": roster_hash,
            "uniqueStableActions": len(unique_actions), "uniqueActionRosterSha256": unique_roster_hash,
            "queries": len(data["query_ids"]), "queryRosterSha256": query_roster_hash,
            "sourceMeasuredQueries": int(measured_queries.sum()),
            "fixedControlTokens": int(common_mask.sum()),
            "exactConstructPopulations": len(roster.population_ids),
            "targetedCellsFromMetadata": targeted_cells,
        },
        "outcomeAccess": {
            "hepg2ExpressionDatasetIndexed": False,
            "hepg2PerturbedValuesRead": 0,
            "hepg2MetricsComputed": False,
            "trainingOutcomesUsedForBaselinesOnly": True,
            "testOrRetiredDevelopmentOutcomesAccessed": False,
        },
        "sourceHashes": source_hashes,
        "seed": SEED,
        "cpuThreads": 2,
    }
    write_json(output / "protocol.json", protocol)
    print(
        json.dumps({
            "event": "protocol-frozen", "populations": len(roster.population_ids),
            "genes": len(unique_actions), "nearestContext": str(data["context_ids"][nearest_context]),
        }), flush=True,
    )

    centroid = equal_source_fitting_centroid(
        data["targets"][fitting], data["observed"][fitting],
        data["context_index"][fitting], data["action_ids"][fitting],
    )
    np.savez_compressed(
        centroid_path,
        schema=np.asarray("slp.context-transfer-training-centroid/v1"),
        query_ids=data["query_ids"],
        equal_source_gene_balanced_fitting_centroid=centroid.astype(np.float32),
    )
    artifacts: dict[str, object] = {}
    zero = np.zeros((len(unique_actions), len(data["query_ids"])), dtype=np.float32)
    artifacts["zero_control"] = write_population_prediction(
        prediction_output / "zero_control.npy", zero, population_action_index
    )
    del zero
    constant = np.broadcast_to(centroid.astype(np.float32), (len(unique_actions), len(centroid)))
    artifacts["equal_source_fitting_centroid"] = write_population_prediction(
        prediction_output / "equal_source_fitting_centroid.npy", constant, population_action_index
    )
    same_gene, same_gene_seen = same_gene_source_response_forecast(
        data["targets"][fitting], data["observed"][fitting],
        data["context_index"][fitting], data["action_ids"][fitting], unique_actions, centroid,
    )
    if not np.array_equal(same_gene_seen, unique_seen):
        raise ValueError("same-gene fitting support disagrees with frozen seen stratum")
    artifacts["same_gene_source_response_mean"] = write_population_prediction(
        prediction_output / "same_gene_source_response_mean.npy",
        same_gene.astype(np.float32), population_action_index,
    )
    del same_gene

    equal_source = np.zeros((len(unique_actions), len(data["query_ids"])), dtype=np.float64)
    nearest = None
    for context, context_name in enumerate(data["context_ids"]):
        rows = fitting[data["context_index"][fitting] == context]
        ridge = fit_ridge(
            development_features[rows], data["targets"][rows], data["observed"][rows],
            RIDGE_ALPHA,
        )
        prediction = ridge.predict(forecast_features)
        equal_source += prediction / len(data["context_ids"])
        if context == nearest_context:
            nearest = prediction.astype(np.float32)
        print(json.dumps({"event": "ridge-fitted", "context": str(context_name)}), flush=True)
    if nearest is None:
        raise RuntimeError("nearest-source prediction was not retained")
    artifacts["equal_source_average_physical_ridge"] = write_population_prediction(
        prediction_output / "equal_source_average_physical_ridge.npy",
        equal_source.astype(np.float32), population_action_index,
    )
    artifacts["control_nearest_source_physical_ridge"] = write_population_prediction(
        prediction_output / "control_nearest_source_physical_ridge.npy",
        nearest, population_action_index,
    )
    if set(artifacts) != set(BASELINE_NAMES):
        raise RuntimeError("baseline forecast set is incomplete")

    manifest = {
        "schema": "slp.hepg2-context-transfer-baseline-forecasts/v1",
        "status": "frozen-before-target-outcome-access",
        "protocol": {"path": str(output / "protocol.json"), "sha256": sha256_file(output / "protocol.json")},
        "roster": {"path": str(roster_path), "sha256": roster_hash},
        "trainingCentering": {"path": str(centroid_path), "sha256": sha256_file(centroid_path)},
        "predictions": artifacts,
        "forecastOrder": "forecast-roster.npz population axis, then exact 7036-query axis",
        "hepg2PerturbedExpressionRowsRead": 0,
        "hepg2MetricsComputed": False,
        "elapsedSeconds": float(time.monotonic() - started),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=(ROOT / "data/derived/slp11-human-gwps-fixed-panel-context-v1/"
                 "replogle-k562-rpe1-gwps-complete-panel-development-v2-fixed-control-context.npz"),
    )
    parser.add_argument(
        "--features", type=Path,
        default=(ROOT / "data/derived/slp11-human-physical/direct-experiments700-v1/"
                 "human-esm-go-physical-features.npz"),
    )
    parser.add_argument(
        "--hepg2-context", type=Path,
        default=(ROOT / "data/derived/slp11-human-gwps-fixed-panel-context-v1/"
                 "nadig-hepg2-fixed-panel-control-context-v1.npz"),
    )
    parser.add_argument(
        "--hepg2-source", type=Path,
        default=(ROOT / "data/sources/nadig-2025-gse264667-hepg2-v1/"
                 "GSE264667_hepg2_raw_singlecell_01.h5ad"),
    )
    parser.add_argument(
        "--metadata-audit", type=Path,
        default=(ROOT / "data/sources/nadig-2025-gse264667-hepg2-v1/h5ad-metadata-audit.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=(ROOT / "results/slp11-transition/hepg2-context-transfer-baseline-forecasts-v1"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        result = run(parse_args())
    print(json.dumps({"event": "complete", "manifest": result["status"]}))
