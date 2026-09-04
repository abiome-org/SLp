from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from omf.sdk import ProtocolRequest, ProtocolResult, main


def _validation_outputs() -> dict[str, object]:
    return {
        "evaluationSummary": {
            "schema": "slp.molecular-evaluation-summary/v1",
            "validationOnly": True,
        },
        "evaluationReportSha256": "0" * 64,
        "molecularTargets": 0,
        "profileCount": 0,
        "ordinaryPearson": 0.0,
        "perturbedCentroidPearson": 0.0,
        "perturbedCentroidCosine": 0.0,
        "centroidAccuracyCommonPanel": 0.0,
        "minimumSpeciesPerturbedCentroidPearson": 0.0,
        "benchmarkLabelRecords": 0,
        "heldInterventionOverlap": 0,
        "passed": False,
        "compatibilityPassed": False,
    }


def validate(_request: ProtocolRequest) -> ProtocolResult:
    return ProtocolResult(status="ok", outputs=_validation_outputs())


def run(request: ProtocolRequest) -> ProtocolResult:
    from evaluator import evaluate_molecular_predictions, resolve_literal_omf_artifact

    reference_path, reference_artifact = resolve_literal_omf_artifact(
        request.inputs["molecularReference"], "molecularReference"
    )
    prediction_path, prediction_artifact = resolve_literal_omf_artifact(
        request.inputs["molecularPredictions"], "molecularPredictions"
    )
    if reference_artifact == prediction_artifact:
        raise ValueError("molecular reference and prediction artifacts must be distinct")

    report = evaluate_molecular_predictions(
        reference_path,
        prediction_path,
        minimum_reference_perturbations=int(
            request.config.get("minimumReferencePerturbations", 2)
        ),
        minimum_profile_readouts=int(request.config.get("minimumProfileReadouts", 2)),
        max_line_bytes=int(request.config.get("maxLineBytes", 16 * 1024 * 1024)),
        maximum_absolute_log_scale=float(
            request.config.get("maximumAbsoluteLogScale", 20.0)
        ),
    )
    report["inputs"]["reference"]["omfArtifactManifestDigest"] = reference_artifact
    report["inputs"]["predictions"]["omfArtifactManifestDigest"] = prediction_artifact
    output_dir = Path(os.environ["OMF_RESULT_FILE"]).parent
    report_path = output_dir / "molecular-evaluation-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    overall = report["overall"]
    ordinary = overall["ordinary"]
    specific = overall["perturbationSpecific"]
    species_pearson = [
        species["perturbationSpecific"]["perturbedCentroidPearson"]
        for species in report["species"].values()
    ]
    summary = {
        "schema": "slp.molecular-evaluation-summary/v1",
        "reportSha256": report_digest,
        "modelCheckpointSha256": report["inputs"]["predictions"][
            "modelCheckpointSha256"
        ],
        "referenceManifestSha256": report["inputs"]["reference"]["manifestSha256"],
        "predictionManifestSha256": report["inputs"]["predictions"]["manifestSha256"],
        "referenceArtifactManifestDigest": reference_artifact,
        "predictionArtifactManifestDigest": prediction_artifact,
        "methodClass": report["method"]["class"],
        "decisionScope": report["decision"]["scope"],
        "decisionThresholds": report["decision"]["thresholds"],
        "compatibilityScope": report["decision"]["compatibilityScope"],
    }
    outputs = {
        "evaluationSummary": summary,
        "evaluationReportSha256": report_digest,
        "molecularTargets": ordinary["targets"],
        "profileCount": specific["profiles"],
        "ordinaryPearson": ordinary["pearson"],
        "perturbedCentroidPearson": specific["perturbedCentroidPearson"],
        "perturbedCentroidCosine": specific["perturbedCentroidCosine"],
        "centroidAccuracyCommonPanel": specific["centroidAccuracyCommonPanel"],
        "minimumSpeciesPerturbedCentroidPearson": min(species_pearson),
        "benchmarkLabelRecords": report["audit"]["benchmarkLabelRecords"],
        "heldInterventionOverlap": report["audit"]["heldInterventionOverlap"],
        "passed": report["decision"]["passed"],
        "compatibilityPassed": report["decision"]["compatibilityPassed"],
    }
    return ProtocolResult(
        status="ok",
        outputs=outputs,
        metrics={
            "ordinary_pearson": outputs["ordinaryPearson"],
            "perturbed_centroid_pearson": outputs["perturbedCentroidPearson"],
            "perturbed_centroid_cosine": outputs["perturbedCentroidCosine"],
            "centroid_accuracy_common_panel": outputs["centroidAccuracyCommonPanel"],
        },
        artifacts=[
            {
                "name": "molecularEvaluationReport",
                "kind": "evaluation",
                "path": report_path.name,
            }
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main({"validate": validate, "run": run}))
