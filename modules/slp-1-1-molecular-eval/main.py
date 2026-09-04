from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from omf.sdk import ProtocolRequest, ProtocolResult, main


def _validation_outputs() -> dict[str, object]:
    return {
        "evaluationSummary": {
            "schema": "slp.molecular-evaluation-summary/v2",
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
        "centeringHeldInterventionOverlap": 0,
        "strictCorpusAuditPassed": False,
        "heldRosterValidationMatch": False,
        "exactTargetFreeQueryManifest": False,
        "exactProfilePanelJoin": False,
        "diagnosticPassed": False,
        "compatibilityPassed": False,
    }


def validate(_request: ProtocolRequest) -> ProtocolResult:
    return ProtocolResult(status="ok", outputs=_validation_outputs())


def run(request: ProtocolRequest) -> ProtocolResult:
    from evaluator import (
        evaluate_molecular_predictions,
        resolve_literal_omf_artifact,
        resolve_pinned_query_input,
    )

    prediction_path, prediction_artifact = resolve_literal_omf_artifact(
        request.inputs["molecularPredictions"], "molecularPredictions"
    )
    if not Path(prediction_path).is_file():
        raise ValueError("molecularPredictions must be the file-valued canonical tar artifact")
    query_input = resolve_pinned_query_input(
        request.inputs["molecularQuery"],
        expected_resource=request.config["expectedQueryResource"],
        expected_manifest_digest=request.config["expectedQueryDatasetManifestDigest"],
    )

    report = evaluate_molecular_predictions(
        request.inputs["molecularCenteringReference"],
        prediction_path,
        request.inputs["molecularTruth"],
        query_input,
        request.inputs["corpusAudit"],
        request.inputs["heldRoster"],
        request.inputs["modelCheckpoint"],
        minimum_reference_perturbations=int(
            request.config.get("minimumReferencePerturbations", 2)
        ),
        minimum_profile_readouts=int(request.config.get("minimumProfileReadouts", 2)),
        max_line_bytes=int(request.config.get("maxLineBytes", 16 * 1024 * 1024)),
        maximum_absolute_log_scale=float(
            request.config.get("maximumAbsoluteLogScale", 20.0)
        ),
    )
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
        "schema": "slp.molecular-evaluation-summary/v2",
        "reportSha256": report_digest,
        "modelCheckpointContentSha256": report["inputs"]["modelCheckpoint"]["contentSha256"],
        "modelCheckpointResource": report["inputs"]["modelCheckpoint"]["resource"],
        "centeringManifestSha256": report["inputs"]["centeringReference"]["manifestSha256"],
        "predictionManifestSha256": report["inputs"]["predictions"]["manifestSha256"],
        "truthManifestSha256": report["inputs"]["heldTruth"]["manifestSha256"],
        "queryResource": report["inputs"]["molecularQuery"]["resource"],
        "queryDatasetManifestDigest": report["inputs"]["molecularQuery"]["datasetManifestDigest"],
        "queryManifestSha256": report["inputs"]["molecularQuery"]["queryManifestSha256"],
        "predictionArtifactManifestDigest": prediction_artifact,
        "centeringDatasetManifestDigest": report["inputs"]["centeringReference"]["datasetManifestDigest"],
        "truthDatasetManifestDigest": report["inputs"]["heldTruth"]["datasetManifestDigest"],
        "corpusAuditDatasetManifestDigest": report["inputs"]["corpusAudit"]["datasetManifestDigest"],
        "heldRosterDatasetManifestDigest": report["inputs"]["heldRoster"]["datasetManifestDigest"],
        "methodClass": report["method"]["class"],
        "diagnosticScope": report["diagnostic"]["scope"],
        "diagnosticThresholds": report["diagnostic"]["thresholds"],
        "compatibilityScope": report["diagnostic"]["compatibilityScope"],
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
        "centeringHeldInterventionOverlap": report["audit"]["centeringHeldInterventionOverlap"],
        "strictCorpusAuditPassed": report["audit"]["strictCorpusAuditPassed"],
        "heldRosterValidationMatch": report["audit"]["heldRosterValidationMatch"],
        "exactTargetFreeQueryManifest": report["audit"]["exactTargetFreeQueryManifest"],
        "exactProfilePanelJoin": report["audit"]["exactProfilePanelJoin"],
        "diagnosticPassed": report["diagnostic"]["diagnosticPassed"],
        "compatibilityPassed": report["diagnostic"]["compatibilityPassed"],
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
