#!/usr/bin/env python3
"""Create a descriptive comparison of frozen paired PCA and AE reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PCA = ROOT / "results/slp11-transition/frangieh-paired-pca128-latent-ridge-seed731-v1"
AE = ROOT / "results/slp11-transition/frangieh-cell-state-ae-latent-ridge-seed731-v1"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run() -> None:
    pca = json.loads((PCA / "report.json").read_text(encoding="utf-8"))
    ae = json.loads((AE / "report.json").read_text(encoding="utf-8"))
    with np.load(PCA / "pca-forecast.npz", allow_pickle=False) as pca_artifact, np.load(
        AE / "reference.npz", allow_pickle=False,
    ) as ae_reference:
        stats_equal = {
            f"{head}_{name}": bool(
                np.array_equal(pca_artifact[f"{head}_{name}"], ae_reference[f"{head}_{name}"])
            )
            for head in ("rna", "protein")
            for name in ("mean", "sd")
        }
    contexts = {}
    for context, pca_context in pca["forecast"]["contexts"].items():
        contexts[context] = {}
        for head, pca_head in pca_context["heads"].items():
            pca_metrics = pca_head["pca"]
            ae_metrics = ae["forecast"]["contexts"][context]["heads"][head]["world"]
            contexts[context][head] = {
                "pcaRawMse": pca_metrics["raw_mse"],
                "aeRawMse": ae_metrics["raw_mse"],
                "pcaFractionalMseImprovementVsAe": (
                    1 - pca_metrics["raw_mse"] / ae_metrics["raw_mse"]
                ),
                "pcaCenteredProfilePearson": (
                    pca_metrics["query_centroid_adjusted_profile_pearson"]
                ),
                "aeCenteredProfilePearson": (
                    ae_metrics["query_centroid_adjusted_profile_pearson"]
                ),
                "centeredProfilePearsonDifference": (
                    pca_metrics["query_centroid_adjusted_profile_pearson"]
                    - ae_metrics["query_centroid_adjusted_profile_pearson"]
                ),
            }
    report = {
        "schema": "slp.frangieh-paired-pca-vs-ae-descriptive/v1",
        "meaning": "post-run descriptive comparison of two frozen assay-specific representations; no selection, new gate, or joint-supervision attribution",
        "inputs": {
            "pcaReportSha256": sha256(PCA / "report.json"),
            "pcaPredictionsSha256": sha256(PCA / "predictions.npz"),
            "aeReportSha256": sha256(AE / "report.json"),
            "aePredictionsSha256": sha256(AE / "predictions.npz"),
        },
        "normalizationArraysBitExact": stats_equal,
        "reconstruction": {
            "pca": pca["reconstruction"],
            "ae": ae["reconstruction"]["validation"],
        },
        "forecast": contexts,
        "fixedDecisionsUnchanged": True,
    }
    path = PCA / "pca-vs-ae-descriptive.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"path": str(path), "sha256": sha256(path)}))


if __name__ == "__main__":
    run()
