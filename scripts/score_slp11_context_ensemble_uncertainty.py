"""Descriptive gene bootstrap for already frozen context-expansion ensembles."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ENSEMBLE_SHA = "4ca976498710d3a1678c8b4384fd3f1822da693a7b02e6830ba3cf5e5db902b7"
BASELINE_SHA = "0c40ed63c336d5fb1795466693c733711150ec6de84d9fc21585f1d38fe57bc0"
CONTEXTS = ("K562-essential", "RPE1-essential", "K562-GWPS", "HepG2-adaptive")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def gene_mse(prediction: np.ndarray, truth: np.ndarray, observed: np.ndarray) -> np.ndarray:
    if prediction.shape != truth.shape or observed.shape != truth.shape or observed.dtype != bool:
        raise ValueError("prediction, truth and Boolean mask must align")
    if truth.ndim != 2 or np.any(observed.sum(1) == 0):
        raise ValueError("each gene needs observed queries")
    if not np.isfinite(prediction[observed]).all() or not np.isfinite(truth[observed]).all():
        raise ValueError("observed values must be finite")
    delta = np.where(observed, prediction, 0).astype(np.float64) - np.where(observed, truth, 0)
    return (delta * delta).sum(1) / observed.sum(1)


def paired_interval(candidate: np.ndarray, reference: np.ndarray, seed: int = 731,
                    samples: int = 2000) -> dict:
    if candidate.ndim != 1 or candidate.shape != reference.shape or len(candidate) < 2:
        raise ValueError("paired gene losses must align and contain at least two genes")
    if samples < 1 or not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("finite losses and positive sample count required")
    if np.any(candidate < 0) or np.any(reference <= 0):
        raise ValueError("candidate loss must be nonnegative and reference loss positive")
    draws = np.random.default_rng(seed).integers(0, len(candidate), (samples, len(candidate)))
    gain = 100 * (1 - candidate[draws].mean(1) / reference[draws].mean(1))
    return {"mseImprovementPercent": float(100 * (1 - candidate.mean() / reference.mean())),
            "pairedGeneBootstrap95PercentileInterval": np.quantile(gain, [.025, .975]).tolist(),
            "genes": len(candidate), "samples": samples, "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.ensemble) != ENSEMBLE_SHA or sha256(args.baseline) != BASELINE_SHA:
        raise ValueError("frozen prediction checksum mismatch")
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {"schema": "slp11-context-ensemble-descriptive-uncertainty-v1",
              "claimLimit": "Post-hoc descriptive intervals conditional on frozen models and this adaptive development population; no training-seed, source, biological-replicate or selection uncertainty; no decision-rule changes or multiple-comparison correction.",
              "estimand": "Ratio of equal-gene mean observed-query MSE; positive improvement favors source4.",
              "inputs": {"ensembleSha256": ENSEMBLE_SHA, "baselineSha256": BASELINE_SHA,
                         "scoringSourceSha256": sha256(Path(__file__))}, "contexts": {}}
    with np.load(args.ensemble, allow_pickle=False) as predictions, np.load(args.baseline, allow_pickle=False) as baseline:
        for index, name in enumerate(CONTEXTS):
            key = f"context{index}"
            expected = baseline[f"{key}_action_ids"]
            for arm in ("source3", "source4"):
                if not np.array_equal(expected, predictions[f"{arm}_{key}_action_ids"]):
                    raise ValueError("gene IDs must be identically ordered")
            truth, observed = baseline[f"{key}_truth"], baseline[f"{key}_observed"]
            losses = {arm: gene_mse(predictions[f"{arm}_{key}_world"], truth, observed)
                      for arm in ("source3", "source4")}
            losses["ridge"] = gene_mse(baseline[f"{key}_ridge"], truth, observed)
            report["contexts"][name] = {
                "source4VsSource3": paired_interval(losses["source4"], losses["source3"]),
                "source4VsRidge": paired_interval(losses["source4"], losses["ridge"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["contexts"], indent=2))


if __name__ == "__main__":
    main()
