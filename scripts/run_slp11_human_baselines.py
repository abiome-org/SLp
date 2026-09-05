"""Run fitting-only human context null-baseline development diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from human_baselines import (
    duplicate_guide_consistency,
    evaluate_context_baselines,
    fit_context_references,
    randomized_pca_explained_variance,
)

EXPECTED_DEVELOPMENT_SHA256 = (
    "82904b7b52ab34d71e94abb2311c93a420321697d53eab12dabae5b247376f75"
)
REQUIRED_FIELDS = {
    "targets",
    "observed",
    "action_ids",
    "query_ids",
    "context_index",
    "context_ids",
    "basal_control",
    "record_ids",
    "split_train",
    "split_validation",
    "split_test",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_development(path: Path) -> dict[str, np.ndarray]:
    if _sha256(path) != EXPECTED_DEVELOPMENT_SHA256:
        raise ValueError("development bundle SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != REQUIRED_FIELDS:
            raise ValueError("development bundle fields drifted")
        bundle = {name: archive[name] for name in archive.files}
    if len(bundle["split_test"]):
        raise ValueError("development bundle unexpectedly contains test indices")
    train_actions = set(bundle["action_ids"][bundle["split_train"]].tolist())
    validation_actions = set(bundle["action_ids"][bundle["split_validation"]].tolist())
    if train_actions & validation_actions:
        raise ValueError("training and validation action genes overlap")
    if not np.all(bundle["observed"]):
        raise ValueError("v1 diagnostics require the expected complete shared panel")
    return bundle


def run(development_path: Path, output_dir: Path) -> dict[str, object]:
    """Create development-only null references and diagnostic report."""

    bundle = _load_development(development_path)
    references = fit_context_references(
        bundle["targets"],
        bundle["observed"],
        bundle["context_index"],
        bundle["action_ids"],
        bundle["split_train"],
        bundle["basal_control"],
        bundle["context_ids"],
    )
    validation = evaluate_context_baselines(
        references,
        bundle["targets"],
        bundle["observed"],
        bundle["context_index"],
        bundle["action_ids"],
        bundle["split_validation"],
        bundle["basal_control"],
    )

    training_diagnostics: dict[str, object] = {}
    for context, context_id_value in enumerate(bundle["context_ids"]):
        context_id = str(context_id_value)
        rows = bundle["split_train"][
            bundle["context_index"][bundle["split_train"]] == context
        ]
        residuals = (
            bundle["targets"][rows].astype(np.float64)
            - references.perturbation_mean[context]
        )
        training_diagnostics[context_id] = {
            "records": len(rows),
            "actions": len(set(bundle["action_ids"][rows].tolist())),
            "pcaResidualDefinition": (
                "fixed-transform target minus per-query context training mean"
            ),
            "top10Pca": randomized_pca_explained_variance(residuals),
            "duplicateGuideConsistency": duplicate_guide_consistency(
                bundle["targets"][rows],
                bundle["action_ids"][rows],
                bundle["basal_control"][context],
            ),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    references_path = output_dir / "context-references.npz"
    np.savez_compressed(
        references_path,
        context_ids=np.asarray(references.context_ids),
        perturbation_mean=references.perturbation_mean.astype(np.float32),
        perturbation_scale=references.perturbation_scale.astype(np.float32),
        basal_mean=references.basal_mean.astype(np.float32),
        basal_scale=references.basal_scale.astype(np.float32),
        scale_counts=references.scale_counts,
    )
    report: dict[str, object] = {
        "schema": "slp.human-null-baseline-development-diagnostics/v1",
        "label": "development diagnostics",
        "candidateSelectionResult": False,
        "testOnlyArtifactAccessed": False,
        "source": {
            "developmentPath": development_path.as_posix(),
            "developmentSha256": EXPECTED_DEVELOPMENT_SHA256,
            "sourceManifest": "sources/human-perturbation-development-v1.yaml",
            "ncbiTaxon": 9606,
        },
        "valueSpace": "log2(1 + 10000*x/sum(x_shared_7226))",
        "counts": {
            "queries": int(bundle["targets"].shape[1]),
            "developmentRecords": int(bundle["targets"].shape[0]),
            "trainingRecords": len(bundle["split_train"]),
            "validationRecords": len(bundle["split_validation"]),
            "trainingActions": len(
                set(bundle["action_ids"][bundle["split_train"]].tolist())
            ),
            "validationActions": len(
                set(bundle["action_ids"][bundle["split_validation"]].tolist())
            ),
        },
        "scaleCalibration": {
            "provenance": references.scale_provenance,
            "folds": references.folds,
            "seed": references.seed,
            "actionGrouping": "same taxon-9606 Ensembl action uses one fold across contexts",
            "perturbationMeanPrediction": "held action-fold context/query mean",
            "basalPrediction": (
                "fixed matched-control mean; independent of perturbation training rows"
            ),
            "floor": 1e-3,
        },
        "validationBySourceContext": validation,
        "trainingSignalStructure": training_diagnostics,
        "referenceArtifact": {
            "path": references_path.as_posix(),
            "bytes": references_path.stat().st_size,
            "sha256": _sha256(references_path),
        },
        "limitations": [
            "These are development diagnostics from train and validation rows only.",
            "Duplicate rows are guide-level per-perturbation cell-mean summaries, not biological replicates or a noise ceiling.",
            "K562 and RPE1 differ in cell line, sampling day, and screened action population.",
            "PCA uses training residuals only and a deterministic randomized approximation.",
        ],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    report["outputs"] = {
        "references": report["referenceArtifact"],
        "report": {
            "path": report_path.as_posix(),
            "bytes": report_path.stat().st_size,
            "sha256": _sha256(report_path),
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development",
        type=Path,
        default=ROOT
        / "data"
        / "derived"
        / "slp11-human"
        / "replogle-k562-rpe1-development-v1.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "slp11-transition" / "human-null-baselines-v1",
    )
    args = parser.parse_args()
    report = run(args.development, args.output)
    print(json.dumps(report["outputs"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
