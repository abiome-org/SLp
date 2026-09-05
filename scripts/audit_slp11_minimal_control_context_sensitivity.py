#!/usr/bin/env python3
"""Audit basal-context sensitivity of the frozen physical/state128 candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "runtime_checkpoint": "b1e55f2bcc8a29b6b2467a92ebedfdc1cc80ff8c343a6ab36916d638b9c48cf3",
    "runtime_config": "f9504ab419cc783ca3c3565d38a4ae073139cb091845097f8dbef571f9b6b7bd",
    "runtime_reference": "bb47b189b2010a2f497c0bed207bf4294e0df5c4ec592c302672ac2449c7fb8d",
    "runtime_model": "fdb4555bd0f7c0a0786539da67048f6985f4ec2f36ef7aa45bd22c7c6bfbb2ef",
    "runtime_inference": "da120d2dd8655d6cf90c684e5dbaa6a6aedd42bfefc1090f8bab121de6cd0d1b",
    "features": "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7",
    "roster": "89e1819f22568fb9d35b31e84c338e27ea1f13c18c9de18fda266f83ff0e78e0",
    "fixed_context": "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c",
    "hepg2_control": "382626401ee38e8d5084ac9f86ffc44bd10408826fb85a94ede8eb908cdf5b27",
    "hepg2_forecast": "c6d6e6569d8d915886f28aaef024e49d82f55f7f6b219e7fcee5713640d6248d",
    "candidate_report": "49333ade99f04d96e9d4c4ccc2fc01c002170b38f02d10f88fdc8559d274203d",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_npz(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(names) - set(archive.files)
        if missing:
            raise ValueError(f"{path} lacks {sorted(missing)}")
        return {name: archive[name] for name in names}


def unique_first(values: np.ndarray) -> np.ndarray:
    seen: set[str] = set()
    indices = []
    for index, value in enumerate(values.tolist()):
        if value not in seen:
            seen.add(value)
            indices.append(index)
    return np.asarray(indices, dtype=np.int64)


def flat_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_mean = float(np.mean(left, dtype=np.float64))
    right_mean = float(np.mean(right, dtype=np.float64))
    centered_left = left.astype(np.float64) - left_mean
    centered_right = right.astype(np.float64) - right_mean
    denominator = np.sqrt(
        np.sum(np.square(centered_left)) * np.sum(np.square(centered_right))
    )
    return float(np.sum(centered_left * centered_right) / denominator)


def row_correlation_summary(
    left: np.ndarray, right: np.ndarray, *, batch_size: int = 128
) -> dict[str, float]:
    correlations = []
    for start in range(0, len(left), batch_size):
        x = left[start : start + batch_size].astype(np.float64)
        y = right[start : start + batch_size].astype(np.float64)
        x -= x.mean(1, keepdims=True)
        y -= y.mean(1, keepdims=True)
        denominator = np.sqrt(np.square(x).sum(1) * np.square(y).sum(1))
        valid = denominator > 0
        correlations.extend((x[valid] * y[valid]).sum(1) / denominator[valid])
    values = np.asarray(correlations, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "definedGenes": len(values),
    }


def context_summary(values: np.ndarray) -> dict[str, float]:
    centroid = values.mean(0, dtype=np.float64)
    total_energy = float(np.mean(np.square(values), dtype=np.float64))
    centroid_energy = float(np.mean(np.square(centroid), dtype=np.float64))
    residual_energy = max(total_energy - centroid_energy, 0.0)
    return {
        "totalRms": float(np.sqrt(total_energy)),
        "commonAcrossGeneProfileRms": float(np.sqrt(centroid_energy)),
        "geneCenteredForecastRms": float(np.sqrt(residual_energy)),
        "geneCenteredForecastVariance": residual_energy,
        "commonProfileEnergyFraction": (
            centroid_energy / total_energy if total_energy > 0 else 0.0
        ),
        "geneSpecificEnergyFraction": (
            residual_energy / total_energy if total_energy > 0 else 0.0
        ),
    }


def pairwise_decomposition(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("pairwise forecasts must share [genes,queries] shape")
    difference = left.astype(np.float64) - right.astype(np.float64)
    common = difference.mean(0)
    total_energy = float(np.mean(np.square(difference)))
    common_energy = float(np.mean(np.square(common)))
    residual_energy = max(total_energy - common_energy, 0.0)
    left_centroid = left.mean(0, dtype=np.float64)
    right_centroid = right.mean(0, dtype=np.float64)
    return {
        "overallDifferenceRms": float(np.sqrt(total_energy)),
        "commonAcrossGeneDifferenceRms": float(np.sqrt(common_energy)),
        "geneSpecificResidualDifferenceRms": float(np.sqrt(residual_energy)),
        "commonProfileDifferenceEnergyFraction": (
            common_energy / total_energy if total_energy > 0 else 0.0
        ),
        "geneSpecificDifferenceEnergyFraction": (
            residual_energy / total_energy if total_energy > 0 else 0.0
        ),
        "flattenedForecastCorrelation": flat_correlation(left, right),
        "flattenedGeneSpecificResidualCorrelation": flat_correlation(
            left - left_centroid, right - right_centroid
        ),
        "perGeneProfileCorrelation": row_correlation_summary(left, right),
    }


def across_context_decomposition(values: list[np.ndarray]) -> dict[str, float]:
    if len(values) < 2 or any(value.shape != values[0].shape for value in values):
        raise ValueError("at least two aligned context forecasts are required")
    context_mean = sum(value.astype(np.float64) for value in values) / len(values)
    centroids = [value.mean(0, dtype=np.float64) for value in values]
    centroid_mean = sum(centroids) / len(centroids)
    total_energy = sum(
        float(np.mean(np.square(value - context_mean), dtype=np.float64))
        for value in values
    ) / len(values)
    common_energy = sum(
        float(np.mean(np.square(centroid - centroid_mean), dtype=np.float64))
        for centroid in centroids
    ) / len(centroids)
    residual_energy = max(total_energy - common_energy, 0.0)
    within_gene_energy = np.mean(
        [context_summary(value)["geneCenteredForecastVariance"] for value in values]
    )
    return {
        "contextSensitivityRms": float(np.sqrt(total_energy)),
        "commonProfileSensitivityRms": float(np.sqrt(common_energy)),
        "geneSpecificSensitivityRms": float(np.sqrt(residual_energy)),
        "commonProfileSensitivityEnergyFraction": (
            common_energy / total_energy if total_energy > 0 else 0.0
        ),
        "geneSpecificSensitivityEnergyFraction": (
            residual_energy / total_energy if total_energy > 0 else 0.0
        ),
        "contextSensitivityToWithinContextGeneRmsRatio": (
            float(np.sqrt(total_energy / within_gene_energy))
            if within_gene_energy > 0
            else 0.0
        ),
    }


def import_runtime(path: Path):
    spec = importlib.util.spec_from_file_location(
        "context_sensitivity_minimal_control_runtime", path / "inference.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load portable inference runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(args: argparse.Namespace) -> dict[str, object]:
    runtime_dir = Path(args.runtime)
    paths = {
        "runtime_checkpoint": runtime_dir / "model.safetensors",
        "runtime_config": runtime_dir / "model-config.json",
        "runtime_reference": runtime_dir / "runtime-reference.npz",
        "runtime_model": runtime_dir / "transition_model.py",
        "runtime_inference": runtime_dir / "inference.py",
        "features": Path(args.features),
        "roster": Path(args.roster),
        "fixed_context": Path(args.fixed_context),
        "hepg2_control": Path(args.hepg2_control),
        "hepg2_forecast": Path(args.hepg2_forecast),
        "candidate_report": Path(args.candidate_report),
    }
    for label, path in paths.items():
        actual = sha256_file(path)
        if actual != EXPECTED[label]:
            raise ValueError(f"{label} SHA-256 mismatch: {actual}")
    with paths["candidate_report"].open(encoding="utf-8") as stream:
        candidate = json.load(stream)
    if candidate.get("advancement", {}).get("passed") is not False:
        raise ValueError("audit requires the frozen failed development candidate")

    roster = selected_npz(
        paths["roster"],
        (
            "population_ids",
            "action_ids",
            "query_ids",
            "context_control_query_observed",
        ),
    )
    first = unique_first(roster["action_ids"])
    if len(first) != 2390 or roster["query_ids"].shape != (7036,):
        raise ValueError("expected exact 2390-action/7036-query diagnostic axes")
    features = selected_npz(
        paths["features"], ("feature_values", "entity_taxon", "entity_id")
    )
    keys = list(zip(features["entity_taxon"].tolist(), features["entity_id"].tolist()))
    lookup = {key: row for row, key in enumerate(keys)}
    action_ids = roster["action_ids"][first]
    action_features = np.stack(
        [features["feature_values"][lookup[(9606, str(item))]] for item in action_ids]
    ).astype(np.float32)
    fixed = selected_npz(
        paths["fixed_context"],
        (
            "context_ids",
            "query_ids",
            "basal_control",
            "context_basal_expression",
            "context_basal_observed",
        ),
    )
    hepg2 = selected_npz(
        paths["hepg2_control"],
        (
            "context_ids",
            "query_ids",
            "context_basal_expression",
            "context_basal_observed",
            "perturbed_expression_rows_read",
        ),
    )
    if (
        int(hepg2["perturbed_expression_rows_read"]) != 0
        or not np.array_equal(fixed["query_ids"], roster["query_ids"])
        or not np.array_equal(hepg2["query_ids"], roster["query_ids"])
        or not np.array_equal(
            hepg2["context_basal_observed"][0], roster["context_control_query_observed"]
        )
    ):
        raise ValueError("control-only context/query contract mismatch")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA unavailable; no silent fallback")
    runtime_module = import_runtime(runtime_dir)
    runtime = runtime_module.PortableMinimalControl(runtime_dir, device=args.device)

    names = [str(item) for item in fixed["context_ids"].tolist()] + [
        str(hepg2["context_ids"][0])
    ]
    means: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    for index, name in enumerate(names):
        context_values = (
            fixed["context_basal_expression"][index]
            if index < 3
            else hepg2["context_basal_expression"][0]
        )
        context_mask = (
            fixed["context_basal_observed"][index]
            if index < 3
            else hepg2["context_basal_observed"][0]
        )
        control_mean = fixed["basal_control"][index] if index < 3 else np.zeros(7036, np.float32)
        mean = np.empty((2390, 7036), dtype=np.float32)
        delta = np.empty_like(mean)
        for start in range(0, 2390, args.batch_size):
            stop = min(start + args.batch_size, 2390)
            prediction = runtime.predict(
                action_features[start:stop],
                context_values,
                context_mask,
                control_mean,
                query_ids=roster["query_ids"],
            )
            if prediction["uncertainty_calibrated"] or "scale" in prediction:
                raise RuntimeError("diagnostic unexpectedly emitted uncertainty")
            mean[start:stop] = prediction["mean"]
            delta[start:stop] = prediction["delta"]
        means.append(mean)
        deltas.append(delta)

    frozen_hepg2 = np.load(paths["hepg2_forecast"], mmap_mode="r", allow_pickle=False)
    frozen_match_max_abs = float(np.max(np.abs(means[-1] - frozen_hepg2[first])))
    if frozen_match_max_abs > 2e-6:
        raise RuntimeError("regenerated HepG2 forecasts differ from frozen output")

    def analyze(values: list[np.ndarray]) -> dict[str, object]:
        per_context = {
            name: context_summary(value) for name, value in zip(names, values)
        }
        pairwise = {
            f"{names[left]}__vs__{names[right]}": pairwise_decomposition(
                values[left], values[right]
            )
            for left, right in combinations(range(4), 2)
        }
        return {
            "withinContext": per_context,
            "pairwise": pairwise,
            "acrossAllContexts": across_context_decomposition(values),
        }

    mean_analysis = analyze(means)
    delta_analysis = analyze(deltas)
    delta_global = delta_analysis["acrossAllContexts"]
    residual_fraction = float(delta_global["geneSpecificSensitivityEnergyFraction"])
    common_fraction = float(delta_global["commonProfileSensitivityEnergyFraction"])
    ratio = float(delta_global["contextSensitivityToWithinContextGeneRmsRatio"])
    delta_pairs = delta_analysis["pairwise"]
    closest_pair = min(delta_pairs, key=lambda key: delta_pairs[key]["overallDifferenceRms"])
    farthest_pair = max(delta_pairs, key=lambda key: delta_pairs[key]["overallDifferenceRms"])
    assessment = {
        "basis": "intervention delta isolates basal-context response from supplied control mean",
        "contextSensitivityRelativeToWithinContextGeneVariation": ratio,
        "commonProfileDifferenceEnergyFraction": common_fraction,
        "geneSpecificDifferenceEnergyFraction": residual_fraction,
        "closestContextPair": {
            "pair": closest_pair,
            "differenceRms": delta_pairs[closest_pair]["overallDifferenceRms"],
            "flattenedCorrelation": delta_pairs[closest_pair]["flattenedForecastCorrelation"],
        },
        "farthestContextPair": {
            "pair": farthest_pair,
            "differenceRms": delta_pairs[farthest_pair]["overallDifferenceRms"],
            "flattenedCorrelation": delta_pairs[farthest_pair]["flattenedForecastCorrelation"],
        },
        "description": (
            "Context sensitivity RMS is reported as an exact ratio to within-context "
            "action-specific RMS. Difference energy is split between a common query "
            "profile and action-specific residuals, with neither component alone "
            "accounting for all sensitivity. Pairwise results identify strong "
            "heterogeneity: the closest controls can be nearly invariant while the "
            "farthest context produces both common and action-specific changes."
        ),
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source_dir = output / "source"
    source_dir.mkdir()
    shutil.copyfile(Path(__file__), source_dir / Path(__file__).name)
    report: dict[str, object] = {
        "schema": "slp.minimal-control-basal-context-sensitivity/v1",
        "status": "descriptive-target-free-architecture-audit-failed-candidate",
        "candidateAdvancementPassed": False,
        "identity": {
            "ncbiTaxon": 9606,
            "actionNamespace": "Ensembl-gene",
            "uniqueActions": 2390,
            "queries": 7036,
            "actionOrder": "first occurrence of each exact action ID in pinned forecast roster",
            "queryOrder": "exact pinned forecast roster query axis",
        },
        "contexts": names,
        "definitions": {
            "commonProfile": "mean forecast difference across the 2390 actions for each query",
            "geneSpecificResidual": "pairwise difference minus its common profile",
            "geneCenteredForecastVariance": "mean squared forecast residual after subtracting each query's across-action mean",
            "energyFractions": "orthogonal squared-RMS fractions; common plus residual equals total up to floating point",
        },
        "molecularMean": mean_analysis,
        "interventionDelta": delta_analysis,
        "assessment": assessment,
        "verification": {
            "frozenHepg2ForecastMaxAbsError": frozen_match_max_abs,
            "allForecastsFinite": all(np.isfinite(value).all() for value in means + deltas),
            "hepg2PerturbedExpressionRowsRead": 0,
            "hepg2OutcomesRead": False,
            "uncertaintyComputed": False,
            "fittingPerformed": False,
        },
        "inputs": {
            label: {"path": str(path), "sha256": EXPECTED[label]}
            for label, path in paths.items()
        },
        "source": {
            "path": f"source/{Path(__file__).name}",
            "sha256": sha256_file(source_dir / Path(__file__).name),
        },
    }
    write_json(output / "report.json", report)
    report["artifacts"] = {
        "reportSha256": sha256_file(output / "report.json"),
        "sourceSha256": sha256_file(source_dir / Path(__file__).name),
    }
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--runtime", required=True)
    result.add_argument("--features", required=True)
    result.add_argument("--roster", required=True)
    result.add_argument("--fixed-context", required=True)
    result.add_argument("--hepg2-control", required=True)
    result.add_argument("--hepg2-forecast", required=True)
    result.add_argument("--candidate-report", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--batch-size", type=int, default=256)
    return result


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parser().parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "event": "context-sensitivity-audit-complete",
                "output": args.output,
                "reportSha256": report["artifacts"]["reportSha256"],
                "assessment": report["assessment"],
            }
        )
    )


if __name__ == "__main__":
    main()
