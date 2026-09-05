#!/usr/bin/env python3
"""Freeze, then later execute, the HepG2 context-transfer scoring contract."""

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
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from context_transfer_scoring import (
    bootstrap_gene_profiles,
    collapse_gene_profiles,
    score_gene_profiles,
)

BASELINE_ROOT = ROOT / "results/slp11-transition/hepg2-context-transfer-baseline-forecasts-v1"
BASELINE_PROTOCOL_SHA256 = "943baee44f25a9be7a9fe99e87bd89a364483902e429a7acbd2e2a92df6e74ed"
BASELINE_MANIFEST_SHA256 = "10c29e56ecb67f61c9398def714c0e84d07a91643d1400ba582dcf781408a4ef"
BASELINE_ROSTER_SHA256 = "89e1819f22568fb9d35b31e84c338e27ea1f13c18c9de18fda266f83ff0e78e0"
TRAINING_CENTER_SHA256 = "fa55ccb81e37db79ee81da10e5f345156a1a84f4f9883cd473f2e334b962268f"
ADAPTER_SOURCE_SHA256 = "bf73d0b62a0ed2fc6cd8d94e081a171d09d89828d8d8f64383f107f0b7cd59e6"
HEPG2_SOURCE_SHA256 = "e1ad7c3c5a201c861a207a858aa7e59f5e6ac1955674c415f7de0d1dadadb52e"
CONTROL_NORMALIZATION_SHA256 = "3f72db203e989cb60d9ecd65874a11d2c83af0772a8011bafcb559a65c459951"
CONTEXT_DESCRIPTOR_SHA256 = "382626401ee38e8d5084ac9f86ffc44bd10408826fb85a94ede8eb908cdf5b27"
EXPECTED_OUTCOME_PATH = (
    ROOT / "data/derived/slp11-human/nadig-hepg2-frozen-context-diagnostic-v1/molecular.npz"
)
BOOTSTRAP_SAMPLES = 1000
SEED = 731
PRIMARY_MSE_IMPROVEMENT = 0.02
PRIMARY_PEARSON_MINIMUM = 0.10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def load_forecast(path: Path, key: str | None = None, *, mmap: bool = True) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False, mmap_mode="r" if mmap else None)
    if path.suffix == ".npz":
        if key is None:
            raise ValueError("an NPZ world forecast requires --world-key")
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive.files:
                raise ValueError(f"world forecast NPZ lacks member {key!r}")
            return archive[key]
    raise ValueError("forecast must be an NPY or NPZ file")


def verified_baseline_inputs() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    expected = {
        BASELINE_ROOT / "protocol.json": BASELINE_PROTOCOL_SHA256,
        BASELINE_ROOT / "manifest.json": BASELINE_MANIFEST_SHA256,
        BASELINE_ROOT / "forecast-roster.npz": BASELINE_ROSTER_SHA256,
        BASELINE_ROOT / "training-centering.npz": TRAINING_CENTER_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"frozen baseline input changed: {path}")
    manifest = load_json(BASELINE_ROOT / "manifest.json")
    for details in manifest["predictions"].values():
        path = Path(details["path"])
        if sha256_file(path) != details["sha256"]:
            raise ValueError(f"baseline forecast changed: {path}")
    return manifest, load_npz(BASELINE_ROOT / "forecast-roster.npz")


def freeze(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"scoring protocol directory already exists: {output}")
    baseline_manifest, roster = verified_baseline_inputs()
    world_path = args.world_forecast.resolve(strict=True)
    world_manifest_path = args.world_manifest.resolve(strict=True)
    world_hash = sha256_file(world_path)
    world_manifest_hash = sha256_file(world_manifest_path)
    world_manifest = load_json(world_manifest_path)
    if (
        world_manifest_hash
        != "7035ba29fbb47a5003233c47d022f3be5b8654b0098e5d6007df6b2c32b6eb97"
        or world_hash
        != "c6d6e6569d8d915886f28aaef024e49d82f55f7f6b219e7fcee5713640d6248d"
        or world_manifest["candidate"]["checkpointSha256"]
        != "b1e55f2bcc8a29b6b2467a92ebedfdc1cc80ff8c343a6ab36916d638b9c48cf3"
        or world_manifest["identity"]["rosterSha256"] != BASELINE_ROSTER_SHA256
        or world_manifest["forecast"]["hepg2PerturbedExpressionRowsRead"] != 0
    ):
        raise ValueError("world forecast manifest or frozen source pins differ")
    world = load_forecast(world_path, args.world_key)
    expected_shape = (roster["population_ids"].size, roster["query_ids"].size)
    if world.shape != expected_shape or not np.isfinite(world).all():
        raise ValueError(f"world forecast must be finite with shape {expected_shape}")
    adapter_source = MODULE / "hepg2_data.py"
    if sha256_file(adapter_source) != ADAPTER_SOURCE_SHA256:
        raise ValueError("HepG2 adapter source changed before scoring freeze")

    output.mkdir(parents=True, exist_ok=False)
    source_output = output / "source"
    source_output.mkdir()
    source_files = (Path(__file__), MODULE / "context_transfer_scoring.py")
    source_hashes = {}
    for path in source_files:
        shutil.copyfile(path, source_output / path.name)
        source_hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    protocol = {
        "schema": "slp.hepg2-context-transfer-final-scoring-protocol/v1",
        "status": "frozen-before-hepg2-perturbed-outcome-access",
        "world": {
            "forecastPath": str(world_path), "forecastSha256": world_hash,
            "forecastMember": args.world_key, "shape": list(expected_shape),
            "manifestPath": str(world_manifest_path), "manifestSha256": world_manifest_hash,
            "requiredFamily": "minimal-control-v2 physical1156 state128",
            "checkpointSha256": "b1e55f2bcc8a29b6b2467a92ebedfdc1cc80ff8c343a6ab36916d638b9c48cf3",
            "runtimeInferenceSha256": "da120d2dd8655d6cf90c684e5dbaa6a6aedd42bfefc1090f8bab121de6cd0d1b",
            "runtimeModelSha256": "fdb4555bd0f7c0a0786539da67048f6985f4ec2f36ef7aa45bd22c7c6bfbb2ef",
        },
        "baselines": {
            "root": str(BASELINE_ROOT),
            "protocolSha256": BASELINE_PROTOCOL_SHA256,
            "manifestSha256": BASELINE_MANIFEST_SHA256,
            "rosterSha256": BASELINE_ROSTER_SHA256,
            "trainingCenterSha256": TRAINING_CENTER_SHA256,
            "predictionHashes": {
                name: details["sha256"]
                for name, details in baseline_manifest["predictions"].items()
            },
        },
        "outcomeContract": {
            "adapterSourcePath": str(adapter_source),
            "adapterSourceSha256": ADAPTER_SOURCE_SHA256,
            "sourceH5adSha256": HEPG2_SOURCE_SHA256,
            "controlNormalizationSha256": CONTROL_NORMALIZATION_SHA256,
            "contextDescriptorSha256": CONTEXT_DESCRIPTOR_SHA256,
            "immutableDestination": str(EXPECTED_OUTCOME_PATH),
            "expectedPopulationAndQueryOrder": "exact forecast-roster.npz order",
            "observed": "must equal query_num_cells_filtered > 0",
            "duplicateExactRecordIds": "reject",
            "outcomeArtifactDoesNotExistOrIsNotOpenedDuringFreeze": True,
        },
        "primaryEstimand": {
            "recordCollapse": (
                "within each stable action gene and query, average prediction and truth "
                "equally over observed exact construct records"
            ),
            "mse": (
                "mean squared error over observed queries of each gene-averaged profile, "
                "then equal mean over genes"
            ),
            "centering": (
                "inside each seen/unseen stratum, separately compute prediction and truth "
                "query-wise means over gene-averaged profiles with equal gene weight"
            ),
            "correlation": (
                "subtract those query-wise centroids, compute ordinary profile Pearson "
                "over observed queries within each gene, then equal mean over defined genes"
            ),
            "constantTolerance": (
                "undefined when centered sum of squares is at most n*(64*float64_eps*"
                "max(1,max_absolute_centered_value))^2"
            ),
            "strata": "actual three-context fitting-gene seen versus unseen, both must pass",
        },
        "secondaryEstimands": {
            "equalGeneMeanConstructMse": (
                "mean query MSE per construct, equal constructs within gene, equal genes"
            ),
            "trainingCentroidAdjustedPearson": (
                "subtract the frozen equal-source gene-balanced fitting centroid from both profiles"
            ),
            "uncenteredPearson": "ordinary per-gene profile Pearson before source centering",
            "globalSplitRoles": "train, validation, and test hash roles reported descriptively",
            "flattenedCorrelation": "omitted to preserve the frozen primary estimand",
        },
        "advancementRule": {
            "eachSeenAndUnseenStratum": {
                "worldGeneAveragedProfileMseImprovementVsEveryBaseline": PRIMARY_MSE_IMPROVEMENT,
                "worldIndependentlyCenteredGeneMacroProfilePearsonMinimum": PRIMARY_PEARSON_MINIMUM,
                "worldPearsonNonregressionVsEveryDefinedNonconstantBaseline": True,
            },
            "averagingAcrossStrataCannotRescueFailure": True,
        },
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES, "seed": SEED, "block": "stable action gene",
            "queryCentroidsRecomputedWithinEveryResampledGenePopulation": True,
            "percentiles": [0.025, 0.5, 0.975], "decisionUse": "descriptive-only",
            "representativeSyntheticTiming": {
                "genes": 2390, "queries": 6789, "secondsForTwoDraws": 0.844,
                "estimatedSixForecasts1000DrawsMinutes": 42.2,
                "bound": "under one hour on current two-thread CPU if timing remains representative",
            },
        },
        "outcomeAccessAtFreeze": {
            "hepg2PerturbedRowsRead": 0, "metricsComputed": False,
            "adapterExecuted": False,
        },
        "interpretationBoundary": (
            "diagnoses joint cross-study, cell-context, and control-cohort normalization "
            "transfer; Replogle core controls differ from Nadig all-non-targeting controls, "
            "and the SLp endpoint is not an author DESeq2 replication"
        ),
        "sourceHashes": source_hashes,
        "cpuThreads": 2,
    }
    write_json(output / "scoring-protocol.json", protocol)
    result = {
        "status": protocol["status"],
        "path": str(output / "scoring-protocol.json"),
        "sha256": sha256_file(output / "scoring-protocol.json"),
        "worldForecastSha256": world_hash,
        "worldManifestSha256": world_manifest_hash,
        "hepg2PerturbedRowsRead": 0,
    }
    write_json(output / "freeze-manifest.json", result)
    return result


def load_and_validate_outcomes(
    path: Path, expected_hash: str, roster: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    if path.resolve() != EXPECTED_OUTCOME_PATH.resolve():
        raise ValueError("HepG2 outcome path differs from the frozen immutable destination")
    if sha256_file(path) != expected_hash:
        raise ValueError("HepG2 outcome artifact SHA-256 does not match the dispatched pin")
    outcomes = load_npz(path)
    required = {
        "targets", "observed", "query_num_cells_filtered", "num_cells_filtered",
        "action_ids", "query_ids", "record_ids", "source_population_ids",
        "source_construct_ids", "source_transcript_labels", "split_role",
        "source_sha256", "control_normalization_sha256", "context_descriptor_sha256",
    }
    if not required <= set(outcomes):
        raise ValueError(f"HepG2 outcome artifact lacks: {sorted(required - set(outcomes))}")
    identity_pairs = (
        ("source_population_ids", "population_ids"),
        ("source_construct_ids", "source_construct_ids"),
        ("source_transcript_labels", "source_transcript_labels"),
        ("action_ids", "action_ids"),
        ("query_ids", "query_ids"),
    )
    if any(not np.array_equal(outcomes[left], roster[right]) for left, right in identity_pairs):
        raise ValueError("HepG2 outcome population or query order differs from frozen roster")
    if len(set(outcomes["record_ids"].astype(str).tolist())) != len(outcomes["record_ids"]):
        raise ValueError("duplicate exact record IDs are forbidden")
    query_counts = outcomes["query_num_cells_filtered"]
    if query_counts.shape != outcomes["targets"].shape or not np.array_equal(
        outcomes["observed"], query_counts > 0
    ):
        raise ValueError("observed must equal positive query-specific contributing cell counts")
    if (
        str(outcomes["source_sha256"].item()) != HEPG2_SOURCE_SHA256
        or str(outcomes["control_normalization_sha256"].item()) != CONTROL_NORMALIZATION_SHA256
        or str(outcomes["context_descriptor_sha256"].item()) != CONTEXT_DESCRIPTOR_SHA256
    ):
        raise ValueError("HepG2 outcome provenance differs from frozen source pins")
    return outcomes


def score_one(
    prediction: np.ndarray,
    outcomes: dict[str, np.ndarray],
    rows: np.ndarray,
    training_centroid: np.ndarray,
    *,
    bootstrap: bool,
) -> dict[str, object]:
    profiles = collapse_gene_profiles(
        prediction[rows], outcomes["targets"][rows], outcomes["observed"][rows],
        outcomes["action_ids"][rows], outcomes["record_ids"][rows],
    )
    report: dict[str, object] = dict(score_gene_profiles(profiles, training_centroid))
    if bootstrap:
        report["bootstrap"] = bootstrap_gene_profiles(
            profiles, samples=BOOTSTRAP_SAMPLES, seed=SEED
        )
    return report


def advancement(primary: dict[str, dict[str, object]]) -> dict[str, object]:
    details = {}
    for stratum, models in primary.items():
        world = models["world"]
        world_mse = float(world["primaryGeneAveragedProfileMse"])
        world_r = world["primaryIndependentlyCenteredGeneMacroProfilePearson"]
        mse_checks = {}
        correlation_checks = {}
        for name, baseline in models.items():
            if name == "world":
                continue
            baseline_mse = float(baseline["primaryGeneAveragedProfileMse"])
            mse_checks[name] = {
                "fractionalImprovement": float((baseline_mse - world_mse) / baseline_mse),
                "passed": bool(world_mse <= (1.0 - PRIMARY_MSE_IMPROVEMENT) * baseline_mse),
            }
            baseline_r = baseline["primaryIndependentlyCenteredGeneMacroProfilePearson"]
            if baseline_r is not None:
                correlation_checks[name] = {
                    "baseline": float(baseline_r),
                    "passed": bool(world_r is not None and float(world_r) >= float(baseline_r)),
                }
        checks = {
            "mseVsEveryBaseline": all(item["passed"] for item in mse_checks.values()),
            "pearsonAtLeast010": bool(world_r is not None and float(world_r) >= 0.10),
            "pearsonNonregressionVsDefinedBaselines": all(
                item["passed"] for item in correlation_checks.values()
            ),
        }
        details[stratum] = {
            "checks": checks, "mseComparisons": mse_checks,
            "definedNonconstantCorrelationComparisons": correlation_checks,
            "passed": all(checks.values()),
        }
    return {"strata": details, "passed": all(item["passed"] for item in details.values())}


def execute_score(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    protocol_dir = args.protocol_dir.resolve(strict=True)
    protocol_path = protocol_dir / "scoring-protocol.json"
    protocol_hash = sha256_file(protocol_path)
    protocol = load_json(protocol_path)
    if protocol["status"] != "frozen-before-hepg2-perturbed-outcome-access":
        raise ValueError("scoring protocol status is not frozen")
    for relative, digest in protocol["sourceHashes"].items():
        if sha256_file(ROOT / relative) != digest:
            raise ValueError(f"scoring source changed after freeze: {relative}")
    baseline_manifest, roster = verified_baseline_inputs()
    outcomes = load_and_validate_outcomes(
        args.outcomes.resolve(strict=True), args.outcome_sha256, roster
    )
    world_path = Path(protocol["world"]["forecastPath"])
    if sha256_file(world_path) != protocol["world"]["forecastSha256"]:
        raise ValueError("world forecast changed after scoring freeze")
    world = load_forecast(world_path, protocol["world"]["forecastMember"])
    center = load_npz(BASELINE_ROOT / "training-centering.npz")[
        "equal_source_gene_balanced_fitting_centroid"
    ]
    predictions: dict[str, np.ndarray] = {"world": world}
    for name, details in baseline_manifest["predictions"].items():
        predictions[name] = np.load(details["path"], allow_pickle=False, mmap_mode="r")

    primary: dict[str, dict[str, object]] = {}
    seen = roster["fitting_gene_seen"]
    for stratum, selected in (("seen", seen), ("unseen", ~seen)):
        rows = np.flatnonzero(selected)
        primary[stratum] = {}
        for name, prediction in predictions.items():
            primary[stratum][name] = score_one(
                prediction, outcomes, rows, center, bootstrap=True
            )
            print(json.dumps({"event": "primary-scored", "stratum": stratum, "model": name}), flush=True)

    secondary_roles: dict[str, object] = {}
    for role in ("train", "validation", "test"):
        role_rows = np.flatnonzero(outcomes["split_role"].astype(str) == role)
        secondary_roles[role] = {
            name: score_one(prediction, outcomes, role_rows, center, bootstrap=False)
            for name, prediction in predictions.items()
        }
    decision = advancement(primary)
    report = {
        "schema": "slp.hepg2-context-transfer-frozen-diagnostic-report/v1",
        "protocol": {"path": str(protocol_path), "sha256": protocol_hash},
        "outcomes": {"path": str(args.outcomes), "sha256": args.outcome_sha256},
        "primary": primary,
        "secondaryGlobalHashRoles": secondary_roles,
        "advancement": decision,
        "bootstrapIntervalsAffectDecision": False,
        "elapsedSeconds": float(time.monotonic() - started),
    }
    report_path = protocol_dir / "report.json"
    if report_path.exists():
        raise ValueError("frozen diagnostic report already exists")
    write_json(report_path, report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--world-forecast", type=Path, required=True)
    freeze_parser.add_argument("--world-manifest", type=Path, required=True)
    freeze_parser.add_argument("--world-key")
    freeze_parser.add_argument("--output", type=Path, required=True)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--protocol-dir", type=Path, required=True)
    score_parser.add_argument("--outcomes", type=Path, required=True)
    score_parser.add_argument("--outcome-sha256", required=True)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    with threadpool_limits(limits=2):
        result = freeze(arguments) if arguments.command == "freeze" else execute_score(arguments)
    print(json.dumps({"event": arguments.command + "-complete", "result": result.get("status", result.get("advancement"))}))
