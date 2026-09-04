from __future__ import annotations

import hashlib
import os
from pathlib import Path

from omf.sdk import ProtocolRequest, ProtocolResult, main


def _validation_outputs() -> dict[str, object]:
    return {
        "baselineSummary": {
            "schema": "slp.molecular-baseline-report/v1",
            "validationOnly": True,
        },
        "baselineReportSha256": "0" * 64,
        "trainingManifestSha256": "0" * 64,
        "referenceManifestSha256": "0" * 64,
        "contextOnlyManifestSha256": "0" * 64,
        "txpertMeanAdditiveManifestSha256": "0" * 64,
        "referenceProfiles": 0,
        "contextOnlyPredictedValues": 0,
        "txpertMeanAdditivePredictedValues": 0,
        "evaluationCompatibility": {
            "status": "contract-blocked",
            "reasonCode": "prediction-log-scale-not-defined",
        },
        "featureBilinearRidge": {
            "status": "protocol-required-contract-blocked",
            "reasonCode": "feature-vectors-absent",
        },
    }


def validate(_request: ProtocolRequest) -> ProtocolResult:
    return ProtocolResult(status="ok", outputs=_validation_outputs())


def run(request: ProtocolRequest) -> ProtocolResult:
    from baselines import Limits, build_baselines, resolve_pinned_dataset

    training = resolve_pinned_dataset(request.inputs["molecularTraining"], "molecularTraining")
    reference = resolve_pinned_dataset(
        request.inputs["molecularReference"], "molecularReference"
    )
    if training.resource == reference.resource or training.path == reference.path:
        raise ValueError("molecular training and reference snapshots must be distinct")
    limits = Limits(
        max_shards=int(request.config.get("maxShards", 256)),
        max_records=int(request.config.get("maxRecords", 2_000_000)),
        max_readouts_per_record=int(
            request.config.get("maxReadoutsPerRecord", 100_000)
        ),
        max_line_bytes=int(request.config.get("maxLineBytes", 16 * 1024 * 1024)),
    )
    result_parent = Path(os.environ["OMF_RESULT_FILE"]).parent
    output_root = result_parent / "molecular-baselines"
    report = build_baselines(training.path, reference.path, output_root, limits)
    report["omfInputs"] = {
        "training": {
            "resource": training.resource,
            "revision": training.revision,
            "omfManifestDigest": training.manifest_digest,
        },
        "reference": {
            "resource": reference.resource,
            "revision": reference.revision,
            "omfManifestDigest": reference.manifest_digest,
        },
    }
    from baselines import _write_json

    report_path = output_root / "baseline-report.json"
    _write_json(report_path, report)
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    context = report["baselines"]["context-only"]
    txpert = report["baselines"]["txpert-mean-additive"]
    outputs = {
        "baselineSummary": {
            "schema": report["schema"],
            "taskName": report["taskName"],
            "trainingManifestSha256": report["training"]["manifestSha256"],
            "referenceManifestSha256": report["reference"]["manifestSha256"],
            "evaluationCompatibility": report["evaluationCompatibility"],
            "featureBilinearRidge": report["featureBilinearRidge"],
        },
        "baselineReportSha256": report_sha,
        "trainingManifestSha256": report["training"]["manifestSha256"],
        "referenceManifestSha256": report["reference"]["manifestSha256"],
        "contextOnlyManifestSha256": context["predictionManifestSha256"],
        "txpertMeanAdditiveManifestSha256": txpert["predictionManifestSha256"],
        "referenceProfiles": context["profiles"],
        "contextOnlyPredictedValues": context["predictedValues"],
        "txpertMeanAdditivePredictedValues": txpert["predictedValues"],
        "evaluationCompatibility": report["evaluationCompatibility"],
        "featureBilinearRidge": report["featureBilinearRidge"],
    }
    return ProtocolResult(
        status="ok",
        outputs=outputs,
        metrics={
            "context_only_predicted_values": outputs["contextOnlyPredictedValues"],
            "txpert_mean_additive_predicted_values": outputs[
                "txpertMeanAdditivePredictedValues"
            ],
        },
        artifacts=[
            {
                "name": "contextOnlyPredictions",
                "kind": "dataset",
                "path": str(Path("molecular-baselines") / "context-only"),
            },
            {
                "name": "txpertMeanAdditivePredictions",
                "kind": "dataset",
                "path": str(Path("molecular-baselines") / "txpert-mean-additive"),
            },
            {
                "name": "molecularBaselineReport",
                "kind": "evaluation",
                "path": str(Path("molecular-baselines") / "baseline-report.json"),
            },
        ],
    )

if __name__ == "__main__":
    raise SystemExit(main({"validate": validate, "run": run}))
