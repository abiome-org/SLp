"""Descriptive paired gene-resampling intervals from saved molecular errors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def paired_intervals(candidate, comparator, *, seed=731, draws=10000):
    candidate, comparator = np.asarray(candidate, float), np.asarray(comparator, float)
    if (candidate.ndim != 1 or candidate.shape != comparator.shape or len(candidate) < 2
            or not np.isfinite(candidate).all() or not np.isfinite(comparator).all()
            or (candidate < 0).any() or (comparator <= 0).any()):
        raise ValueError("paired finite gene MSEs and positive comparator required")
    rng = np.random.default_rng(seed)
    differences, gains = [], []
    for left in range(0, draws, 500):
        index = rng.integers(len(candidate), size=(min(500, draws-left), len(candidate)))
        a, b = candidate[index].mean(1), comparator[index].mean(1)
        differences.append(a-b)
        gains.append(1-a/b)
    return {
        "candidateMinusComparatorMse": float(candidate.mean()-comparator.mean()),
        "candidateRelativeMseGain": float(1-candidate.mean()/comparator.mean()),
        "pairedGeneResampling95PercentMseDifference": np.quantile(np.concatenate(differences), [.025, .975]).tolist(),
        "pairedGeneResampling95PercentRelativeGain": np.quantile(np.concatenate(gains), [.025, .975]).tolist(),
        "genesWithLowerCandidateMse": int(np.sum(candidate < comparator)),
        "genes": len(candidate),
    }


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "sourceArtifact": str(args.artifact.resolve()), "seed": 731, "draws": 10000,
        "comparisons": [["meanAux", "staticRidge"], ["meanAux", "anchoredMean"], ["meanAux", "countOnly"], ["countOnly", "staticRidge"]],
        "method": "Pair-resample saved per-gene full-query MSE vectors with replacement; percentile95 intervals; candidate minus comparator and 1-candidate/comparator; no raw-count access or model selection.",
        "limitation": "Conditional empirical gene-resampling uncertainty in this adaptive cohort; genes may share biology. Not biological-replicate, source, species, seed or prospective confirmation uncertainty. Correlation intervals are omitted because resampling would change query centering.",
        "scriptSha256": digest(Path(__file__)),
    }
    path = args.output / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("frozen interval protocol changed")
    else:
        path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.prepare:
        return
    if (args.output / "report.json").exists():
        raise FileExistsError("immutable interval report exists")
    source_report = json.loads((args.artifact / "report.json").read_text())
    data_path = args.artifact / "development-per-gene-metrics.npz"
    if digest(data_path) != source_report["development"]["perGeneMetrics"]["sha256"]:
        raise ValueError("per-gene metrics differ from final evaluation report")
    with np.load(data_path, allow_pickle=False) as data:
        results = {f"{a}_vs_{b}": paired_intervals(data[a+"_mse"], data[b+"_mse"])
                   for a, b in protocol["comparisons"]}
    report = {"protocolSha256": digest(path), "evaluationReportSha256": digest(args.artifact / "report.json"), "comparisons": results}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
