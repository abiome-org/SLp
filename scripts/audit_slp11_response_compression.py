"""Diagnose response compression versus held-gene predictability on development data."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from response_compression import fit_response_compression, gene_macro_point_metrics

DATA = (
    ROOT
    / "data/derived/slp11-human-gwps-fixed-panel-context-v1"
    / "replogle-k562-rpe1-gwps-complete-panel-development-v2-fixed-control-context.npz"
)
FEATURES = (
    ROOT
    / "data/derived/slp11-human-physical/direct-experiments700-v1"
    / "human-esm-go-physical-features.npz"
)
RIDGE_RUN = ROOT / "results/slp11-transition/physical-features-ridge-screen-v1"
RIDGE_PREDICTIONS = RIDGE_RUN / "predictions.npz"
OUTPUT = ROOT / "results/slp11-transition/response-compression-diagnostic-v1"

EXPECTED = {
    DATA: "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c",
    FEATURES: "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7",
    RIDGE_PREDICTIONS: "c91d96b724f9b99169536ba17a3cce6f0c8578d603257b830a32a335f7e1c525",
    RIDGE_RUN / "protocol.json": "d19aa124c0c313d3b889ee297b71a35a10dba1ecd10868f1ff5f8a700b2f09d2",
}
RANKS = (32, 128)
ALPHA = 10000.0
SEED = 731
SCALE_FLOOR = 0.05


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _assemble_saved_ridge(data: dict[str, np.ndarray], validation: np.ndarray) -> np.ndarray:
    prediction = np.empty((validation.size, data["targets"].shape[1]), dtype=np.float32)
    with np.load(RIDGE_PREDICTIONS, allow_pickle=False) as archive:
        for context in range(data["context_ids"].size):
            local = data["context_index"][validation] == context
            expected = int(local.sum())
            values = archive[f"context{context}_physical"]
            if values.shape != (expected, prediction.shape[1]):
                raise ValueError("saved full physical-ridge prediction shape changed")
            prediction[local] = values
    return prediction


def main() -> None:
    started = time.monotonic()
    for path, expected in EXPECTED.items():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"pinned input drift: {path}: {actual}")
    if OUTPUT.exists():
        raise FileExistsError(f"immutable output already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    source_dir = OUTPUT / "source"
    source_dir.mkdir()
    source_paths = (Path(__file__), MODULE / "response_compression.py", MODULE / "transition_baselines.py")
    for path in source_paths:
        shutil.copyfile(path, source_dir / path.name)

    protocol = {
        "schema": "slp.response-compression-development-diagnostic/v1",
        "hypothesis": (
            "A rank-128 training response basis preserves measured held-gene response structure "
            "even when static physical features cannot predict the corresponding latent scores."
        ),
        "fixedQuestion": {
            "oracleRank128GeneMacroSourceCentroidAdjustedProfilePearsonAtLeast": 0.8,
            "forecastRank128GeneMacroSourceCentroidAdjustedProfilePearsonBelow": 0.4,
            "application": "required independently in every one of the three contexts",
        },
        "data": {
            "path": str(DATA.relative_to(ROOT)),
            "sha256": EXPECTED[DATA],
            "targetValueSpace": "author-control-normalized molecular response; exact value copied from artifact",
            "targetObservedIdentityWith006bSnapshot": {
                "verifiedBeforeProtocol": True,
                "targetsArraySha256": "cc2ba733e8f4f9c4974e8af2b8933322a634201644825cbd930afef6b3617bd2",
                "observedArraySha256": "b34170e17ca6d215313b222a3365570f2b8c8ebd161442b825f0401d758798c8",
                "splitTrainArraySha256": "9bc54d5d3f5ea0b546ca33eb255227a65efcb2e88a216f35bf57d567365ed9ae",
                "splitValidationArraySha256": "bed6e4ac50d8755cdf122fb603a4f9d78c6427f6a55af7b0123d6d1d1db7736a",
            },
        },
        "features": {
            "path": str(FEATURES.relative_to(ROOT)),
            "sha256": EXPECTED[FEATURES],
            "dimensions": 1156,
            "modalities": "static ESM2 plus GO plus direct physical-neighbor annotations",
        },
        "fit": {
            "records": "split_train only",
            "interventions": "global fitting genes only; zero overlap with validation genes",
            "contextMean": "per-query arithmetic mean among fitting records within each source context",
            "queryScale": (
                "pooled fitting residual RMS after context-mean subtraction, per query, "
                f"floored at {SCALE_FLOOR} outcome units"
            ),
            "basis": {
                "method": "sklearn randomized_svd of standardized fitting response residuals",
                "maximumRank": 128,
                "nestedReportedRanks": list(RANKS),
                "seed": SEED,
                "nOversamples": 10,
                "nIter": 3,
            },
            "forecast": (
                "feature-linear ridge alpha 10000 maps standardized fitting-only physical features "
                "to fitting latent scores; it never consumes validation outcomes"
            ),
        },
        "oracle": (
            "validation measured residuals are projected into the fitting-only response basis and decoded; "
            "this consumes each validation outcome and is only a measurement-compression diagnostic, never a forecast"
        ),
        "comparators": {
            "mean": "the same per-context fitting response centroid",
            "fullPhysicalRidge": {
                "predictions": str(RIDGE_PREDICTIONS.relative_to(ROOT)),
                "sha256": EXPECTED[RIDGE_PREDICTIONS],
                "fit": "alpha 10000, context-local training rows, physical1156",
                "compatibility": "saved 006b targets/splits are byte-identical arrays in the 55def context-descriptor snapshot",
            },
        },
        "metrics": {
            "weighting": "per-record query-profile metric, mean within intervention gene, then equal mean over genes",
            "original": "MSE and ordinary molecular-profile Pearson",
            "sourceCentroidAdjusted": (
                "subtract the fitting per-context query centroid from prediction and truth before profile Pearson; "
                "MSE is reported under both labels and is algebraically unchanged"
            ),
        },
        "scope": "exploratory development validation only; no internal test, HepG2, Jurkat, SL, or benchmark outcomes",
        "protectedOutcomesAccessed": False,
        "compute": {"device": "CPU", "BLASThreads": 2, "timeLimitMinutes": 10},
        "sourceHashes": {path.name: sha256(source_dir / path.name) for path in source_paths},
    }
    write_json(OUTPUT / "protocol.json", protocol)
    print(json.dumps({"event": "protocol-frozen", "path": str(OUTPUT / "protocol.json")}), flush=True)

    with np.load(DATA, allow_pickle=False) as archive:
        required = (
            "action_ids", "context_ids", "context_index", "observed", "query_ids",
            "split_test", "split_train", "split_validation", "target_value_space", "targets",
        )
        data = {name: archive[name] for name in required}
    train = data["split_train"]
    validation = data["split_validation"]
    if data["split_test"].size:
        raise ValueError("development snapshot unexpectedly contains an internal test partition")
    fitting_actions = set(data["action_ids"][train].tolist())
    validation_actions = set(data["action_ids"][validation].tolist())
    if fitting_actions & validation_actions:
        raise ValueError("fitting and validation intervention genes overlap")
    if not np.all(data["observed"][train]) or not np.all(data["observed"][validation]):
        raise ValueError("fixed complete-query response diagnostic requires complete panels")

    with np.load(FEATURES, allow_pickle=False) as archive:
        lookup = {
            (int(taxon), str(entity)): values
            for taxon, entity, values in zip(
                archive["entity_taxon"], archive["entity_id"], archive["feature_values"], strict=True
            )
        }
    action_features = np.stack([lookup[(9606, str(action))] for action in data["action_ids"]])
    model = fit_response_compression(
        action_features[train],
        data["targets"][train],
        data["observed"][train],
        data["context_index"][train],
        maximum_rank=max(RANKS),
        alpha=ALPHA,
        seed=SEED,
        scale_floor=SCALE_FLOOR,
    )
    validation_truth = data["targets"][validation]
    validation_observed = data["observed"][validation]
    validation_context = data["context_index"][validation]
    forecast_by_rank = {
        rank: model.predict(action_features[validation], validation_context, rank).astype(np.float32)
        for rank in RANKS
    }
    oracle_by_rank = {
        rank: model.oracle_reconstruct(
            validation_truth, validation_observed, validation_context, rank
        ).astype(np.float32)
        for rank in RANKS
    }
    mean_prediction = model.context_means_[validation_context].astype(np.float32)
    full_ridge_prediction = _assemble_saved_ridge(data, validation)

    results: dict[str, object] = {}
    decisions: list[bool] = []
    for context, context_name in enumerate(data["context_ids"]):
        local = validation_context == context
        truth = validation_truth[local]
        observed = validation_observed[local]
        actions = data["action_ids"][validation[local]].tolist()
        centroid = model.context_means_[context]
        arms: dict[str, object] = {
            "mean": gene_macro_point_metrics(mean_prediction[local], truth, observed, actions, centroid),
            "full_feature_linear_ridge_physical1156": gene_macro_point_metrics(
                full_ridge_prediction[local], truth, observed, actions, centroid
            ),
        }
        for rank in RANKS:
            arms[f"rank{rank}_oracle_measurement_compression"] = gene_macro_point_metrics(
                oracle_by_rank[rank][local], truth, observed, actions, centroid
            )
            arms[f"rank{rank}_feature_linear_latent_forecast"] = gene_macro_point_metrics(
                forecast_by_rank[rank][local], truth, observed, actions, centroid
            )
        oracle_r = arms["rank128_oracle_measurement_compression"][
            "gene_macro_profile_source_centroid_adjusted_pearson_mean"
        ]
        forecast_r = arms["rank128_feature_linear_latent_forecast"][
            "gene_macro_profile_source_centroid_adjusted_pearson_mean"
        ]
        passed = bool(oracle_r is not None and oracle_r >= 0.8 and forecast_r is not None and forecast_r < 0.4)
        decisions.append(passed)
        results[str(context_name)] = {
            "validationRecords": int(local.sum()),
            "validationInterventionGenes": len(set(actions)),
            "arms": arms,
            "fixedQuestionPassed": passed,
        }
        print(
            json.dumps(
                {
                    "event": "context-scored",
                    "context": str(context_name),
                    "oracleRank128AdjustedR": oracle_r,
                    "forecastRank128AdjustedR": forecast_r,
                    "fixedQuestionPassed": passed,
                }
            ),
            flush=True,
        )

    predictions_path = OUTPUT / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        validation_indices=validation,
        rank32_oracle=oracle_by_rank[32],
        rank128_oracle=oracle_by_rank[128],
        rank32_forecast=forecast_by_rank[32],
        rank128_forecast=forecast_by_rank[128],
        mean=mean_prediction,
        full_physical_ridge=full_ridge_prediction,
    )
    report = {
        "schema": "slp.response-compression-development-diagnostic-report/v1",
        "results": results,
        "retainedTrainingStandardizedResidualVariance": {
            f"rank{rank}": model.retained_training_variance(rank) for rank in RANKS
        },
        "fixedQuestionPassedEveryContext": all(decisions),
        "interpretation": (
            "The oracle arms consume validation responses and quantify only compression in a fitting-derived basis. "
            "Only the feature-linear latent arms and saved full ridge are held-gene forecasts."
        ),
        "predictionArtifact": {
            "path": str(predictions_path.relative_to(ROOT)),
            "sha256": sha256(predictions_path),
        },
        "protocolSha256": sha256(OUTPUT / "protocol.json"),
        "elapsedSeconds": time.monotonic() - started,
        "protectedOutcomesAccessed": False,
    }
    write_json(OUTPUT / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
