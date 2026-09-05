"""Add count-stratified and context-matched response-compression diagnostics."""

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
MAIN = ROOT / "results/slp11-transition/response-compression-diagnostic-v1"
EXPOSURE = (
    ROOT
    / "results/slp11-transition/human-gwps-fixed-context-minimal-control-physical-state128-response32-seed731-v1"
    / "model/exposure-uncertainty.npz"
)
OUTPUT = ROOT / "results/slp11-transition/response-compression-count-bin-supplement-v1"

EXPECTED = {
    DATA: "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c",
    FEATURES: "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7",
    MAIN / "protocol.json": "7898252301d5488fabd45fea1be5739acc6417e99c5ead4165b301435dfd6b9a",
    MAIN / "report.json": "525cae361e2d7888a08ef70b4b5543bf13834b2a0467f55cff8d1c5fd638b51b",
    MAIN / "predictions.npz": "c569caf5e63437cf9d0ff1112edc44a870a4df3b08c1b3bfa9f4bf5e732558df",
    EXPOSURE: "9cf5f4a5352dccaa7cb3d6c84e2123b16b190220a1ef9e03c933a887be6c81dd",
}
BINS = ((1.0, 20.0), (20.0, 100.0), (100.0, 500.0), (500.0, np.inf))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def bin_name(lower: float, upper: float) -> str:
    return f"[{int(lower)},{int(upper)})" if np.isfinite(upper) else f"[{int(lower)},infinity)"


def selected_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    observed: np.ndarray,
    actions: list[str],
    centroid: np.ndarray,
) -> dict[str, float | int | None]:
    metrics = gene_macro_point_metrics(prediction, truth, observed, actions, centroid)
    return {
        key: metrics[key]
        for key in (
            "gene_macro_mse",
            "gene_macro_source_centroid_adjusted_mse",
            "gene_macro_profile_source_centroid_adjusted_pearson_mean",
            "profile_source_centroid_adjusted_pearson_defined_genes",
        )
    }


def main() -> None:
    started = time.monotonic()
    for path, expected in EXPECTED.items():
        if sha256(path) != expected:
            raise ValueError(f"pinned input drift: {path}")
    if OUTPUT.exists():
        raise FileExistsError(f"immutable output already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    source_dir = OUTPUT / "source"
    source_dir.mkdir()
    for path in (Path(__file__), MODULE / "response_compression.py", MODULE / "transition_baselines.py"):
        shutil.copyfile(path, source_dir / path.name)

    protocol = {
        "schema": "slp.response-compression-count-bin-supplement/v1",
        "scope": "descriptive development-validation supplement; original frozen report remains unchanged",
        "inputs": {str(path.relative_to(ROOT)): digest for path, digest in EXPECTED.items()},
        "countField": "num_cells_filtered per source perturbation construct record",
        "fixedBins": [bin_name(lower, upper) for lower, upper in BINS],
        "binWeighting": (
            "within each context and count bin, average record profile metrics within intervention gene, "
            "then weight genes equally"
        ),
        "nonIndependence": (
            "a gene represented by multiple constructs can occur in multiple bins; bins are descriptive, "
            "not independent samples and not a replicate-based measurement ceiling"
        ),
        "matchedForecastCorrection": {
            "reason": (
                "the original latent forecast pooled its coefficient ridge across contexts while the full "
                "physical ridge comparator was fitted separately within context"
            ),
            "method": (
                "deterministically recreate the unchanged fitting-only standardized response basis, then "
                "project saved context-local full-ridge forecasts into its rank-32 and rank-128 spans"
            ),
            "equivalence": (
                "for fixed design, penalty, and context, ridge is linear in the response matrix; the focused "
                "synthetic test proves projection equals an explicit context-local latent-score ridge fit"
            ),
            "alpha": 10000.0,
            "newSvdFit": False,
            "basisRecovery": "same data, split, standardization, seed 731, and deterministic randomized-SVD contract",
        },
        "samplingVarianceEnergy": {
            "source": "frozen control-derived mean_sampling_variance from exposure artifact",
            "formula": (
                "for each split/context/query, mean over records of sampling_variance[context,query] "
                "/ num_cells_filtered[record]"
            ),
            "interpretation": (
                "descriptive fitted control-noise contribution only; it is not an estimate of true noise ceiling"
            ),
        },
        "interpretationLimit": (
            "observed response compression loss can include sampling and other measurement noise. It does not "
            "by itself establish a biological representation deficit or justify increasing state dimension."
        ),
        "protectedOutcomesAccessed": False,
        "sourceHashes": {path.name: sha256(path) for path in source_dir.glob("*.py")},
    }
    write_json(OUTPUT / "protocol.json", protocol)
    print(json.dumps({"event": "supplement-protocol-frozen"}), flush=True)

    with np.load(DATA, allow_pickle=False) as archive:
        names = (
            "action_ids", "context_ids", "context_index", "num_cells_filtered", "observed",
            "split_test", "split_train", "split_validation", "targets",
        )
        data = {name: archive[name] for name in names}
    if data["split_test"].size:
        raise ValueError("development snapshot unexpectedly contains internal test records")
    train = data["split_train"]
    validation = data["split_validation"]
    with np.load(FEATURES, allow_pickle=False) as archive:
        lookup = {
            (int(taxon), str(entity)): values
            for taxon, entity, values in zip(
                archive["entity_taxon"], archive["entity_id"], archive["feature_values"], strict=True
            )
        }
    features = np.stack([lookup[(9606, str(action))] for action in data["action_ids"]])
    basis = fit_response_compression(
        features[train], data["targets"][train], data["observed"][train],
        data["context_index"][train], maximum_rank=128, alpha=10000.0, seed=731, scale_floor=0.05
    )
    with np.load(MAIN / "predictions.npz", allow_pickle=False) as archive:
        saved = {name: archive[name] for name in archive.files}
    if not np.array_equal(saved["validation_indices"], validation):
        raise ValueError("main prediction validation ordering changed")
    recreated_pooled = basis.predict(features[validation], data["context_index"][validation], 128)
    pooled_recreation_max_abs = float(
        np.max(np.abs(recreated_pooled - saved["rank128_forecast"].astype(np.float64)))
    )
    if pooled_recreation_max_abs > 1e-6:
        raise ValueError("deterministically recreated response basis does not reproduce pooled forecast")
    matched = {
        rank: basis.project_forecast(
            saved["full_physical_ridge"], data["context_index"][validation], rank
        ).astype(np.float32)
        for rank in (32, 128)
    }
    matched_path = OUTPUT / "matched-context-local-forecasts.npz"
    np.savez_compressed(
        matched_path,
        validation_indices=validation,
        rank32_context_local_latent_forecast=matched[32],
        rank128_context_local_latent_forecast=matched[128],
    )

    context_results: dict[str, object] = {}
    validation_context = data["context_index"][validation]
    counts = data["num_cells_filtered"][validation]
    for context, context_name in enumerate(data["context_ids"]):
        bins: dict[str, object] = {}
        for lower, upper in BINS:
            local = (validation_context == context) & (counts >= lower) & (counts < upper)
            rows = np.flatnonzero(local)
            actions = data["action_ids"][validation[rows]].tolist()
            if not rows.size:
                bins[bin_name(lower, upper)] = {"records": 0, "interventionGenes": 0, "arms": {}}
                continue
            truth = data["targets"][validation[rows]]
            observed = data["observed"][validation[rows]]
            centroid = basis.context_means_[context]
            arms = {
                "rank128_oracle_measurement_compression": saved["rank128_oracle"][rows],
                "rank128_pooled_context_latent_forecast": saved["rank128_forecast"][rows],
                "rank128_matched_context_local_latent_forecast": matched[128][rows],
                "full_feature_linear_ridge_physical1156": saved["full_physical_ridge"][rows],
            }
            bins[bin_name(lower, upper)] = {
                "records": int(rows.size),
                "interventionGenes": len(set(actions)),
                "cellCountMin": float(counts[rows].min()),
                "cellCountMax": float(counts[rows].max()),
                "arms": {
                    name: selected_metrics(prediction, truth, observed, actions, centroid)
                    for name, prediction in arms.items()
                },
            }
        all_rows = np.flatnonzero(validation_context == context)
        actions = data["action_ids"][validation[all_rows]].tolist()
        truth = data["targets"][validation[all_rows]]
        observed = data["observed"][validation[all_rows]]
        centroid = basis.context_means_[context]
        context_results[str(context_name)] = {
            "bins": bins,
            "pooledVersusMatchedAllValidation": {
                "rank128_pooled_context_latent_forecast": selected_metrics(
                    saved["rank128_forecast"][all_rows], truth, observed, actions, centroid
                ),
                "rank128_matched_context_local_latent_forecast": selected_metrics(
                    matched[128][all_rows], truth, observed, actions, centroid
                ),
            },
        }

    with np.load(EXPOSURE, allow_pickle=False) as archive:
        sampling_variance = archive["mean_sampling_variance"].astype(np.float64)
    sampling_energy = np.empty((2, data["context_ids"].size, data["targets"].shape[1]))
    sampling_summary: dict[str, object] = {}
    for split_number, (split_name, split) in enumerate((('train', train), ('validation', validation))):
        split_result: dict[str, object] = {}
        for context, context_name in enumerate(data["context_ids"]):
            local = split[data["context_index"][split] == context]
            inverse_count_mean = float(np.mean(1.0 / data["num_cells_filtered"][local]))
            values = sampling_variance[context] * inverse_count_mean
            sampling_energy[split_number, context] = values
            split_result[str(context_name)] = {
                "records": int(local.size),
                "meanAcrossQueries": float(np.mean(values)),
                "medianAcrossQueries": float(np.median(values)),
                "p10AcrossQueries": float(np.quantile(values, 0.1)),
                "p90AcrossQueries": float(np.quantile(values, 0.9)),
                "meanInverseCellCount": inverse_count_mean,
            }
        sampling_summary[split_name] = split_result
    sampling_path = OUTPUT / "per-query-sampling-variance-energy.npz"
    np.savez_compressed(
        sampling_path,
        split_ids=np.asarray(["train", "validation"]),
        context_ids=data["context_ids"],
        per_query_sampling_variance_energy=sampling_energy,
    )

    report = {
        "schema": "slp.response-compression-count-bin-supplement-report/v1",
        "contexts": context_results,
        "samplingVarianceEnergy": sampling_summary,
        "pooledForecastRecreationMaxAbs": pooled_recreation_max_abs,
        "matchedForecastArtifact": {
            "path": str(matched_path.relative_to(ROOT)), "sha256": sha256(matched_path)
        },
        "samplingEnergyArtifact": {
            "path": str(sampling_path.relative_to(ROOT)), "sha256": sha256(sampling_path)
        },
        "protocolSha256": sha256(OUTPUT / "protocol.json"),
        "elapsedSeconds": time.monotonic() - started,
        "interpretationLimit": protocol["interpretationLimit"],
        "protectedOutcomesAccessed": False,
    }
    write_json(OUTPUT / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
