"""OMF production boundary for target-free sparse world-model training."""

from __future__ import annotations

import os
from pathlib import Path

from omf.sdk import ProtocolRequest, ProtocolResult, main

STATE_SCHEMA = "slp.world-sparse-state/v2"


def _validation_outputs() -> dict[str, object]:
    return {
        "modelState": {"schema": STATE_SCHEMA, "validationOnly": True},
        "parameterCount": 0,
        "checkpointContentSha256": "0" * 64,
        "predictionContentSha256": "0" * 64,
        "trainingReportContentSha256": "0" * 64,
        "predictionRecords": 0,
        "predictionQueries": 0,
        "predictionsTargetFree": True,
        "heldTruthAccessible": False,
        "omfPriorAdmissionRequired": True,
        "environmentAttestedNotPortable": True,
        "releasePortable": False,
    }


def validate(_request: ProtocolRequest) -> ProtocolResult:
    outputs = _validation_outputs()
    return ProtocolResult(status="ok", outputs=outputs, state=outputs["modelState"])


def run(request: ProtocolRequest) -> ProtocolResult:
    from slp_sparse_architecture import MODEL_FORMAT
    from slp_sparse_artifacts import (
        CHECKPOINT_FORMAT,
        RELEASE_BLOCKERS,
        attested_dataset_identity,
        attested_query_identity,
        build_artifact_report,
        validate_admitted_training_evidence,
        write_canonical_report,
        write_sparse_checkpoint,
        write_target_free_predictions,
    )
    from slp_sparse_corpus import CorpusIndex, PredictionQueryIndex, pinned_dataset_path
    from slp_sparse_training import TrainingConfig, train_sparse_world

    required = {
        "pretrain", "molecularPredictionQuery", "corpusAuditEvidence",
        "heldRosterEvidence",
    }
    if set(request.inputs) != required:
        raise ValueError(
            "training requires only pretrain, target-free query, and admitted audit/roster evidence"
        )
    pretrain_input = request.inputs["pretrain"]
    query_input = request.inputs["molecularPredictionQuery"]
    if not isinstance(pretrain_input, dict) or not isinstance(query_input, dict):
        raise ValueError("model inputs must be materialized DatasetSnapshots")
    if pretrain_input.get("resource") == query_input.get("resource"):
        raise ValueError("pretrain and prediction-query resources must be distinct")

    pretrain = CorpusIndex.load(pinned_dataset_path(pretrain_input, "pretrain"))
    query = PredictionQueryIndex.load(
        pinned_dataset_path(query_input, "molecularPredictionQuery"), pretrain
    )
    if not isinstance(request.config, dict):
        raise ValueError("training config must be an object")
    config_value = dict(request.config)
    expected_audit_digest = config_value.pop(
        "expectedCorpusAuditArtifactManifestDigest", None
    )
    evidence = validate_admitted_training_evidence(
        pretrain,
        query,
        pretrain_input,
        query_input,
        request.inputs["corpusAuditEvidence"],
        request.inputs["heldRosterEvidence"],
        expected_corpus_audit_manifest_digest=expected_audit_digest,
    )
    config = _training_config(config_value, TrainingConfig)
    outcome = train_sparse_world(pretrain, config)

    result_file = os.environ.get("OMF_RESULT_FILE")
    if not result_file:
        raise ValueError("OMF_RESULT_FILE is required for artifact placement")
    output_directory = Path(result_file).absolute().parent
    checkpoint_path, checkpoint_sha256 = write_sparse_checkpoint(
        output_directory, outcome.model, outcome.report, evidence
    )
    prediction_path, prediction_sha256, prediction_records, prediction_queries = (
        write_target_free_predictions(
            output_directory,
            outcome.model,
            query,
            query_input,
            checkpoint_sha256,
            batch_size=config.prediction_batch_size,
        )
    )
    report = build_artifact_report(
        outcome.report,
        evidence=evidence,
        molecular_query=query,
        molecular_query_input=query_input,
        checkpoint_content_sha256=checkpoint_sha256,
        prediction_content_sha256=prediction_sha256,
        prediction_records=prediction_records,
        prediction_queries=prediction_queries,
    )
    report_path, report_sha256 = write_canonical_report(output_directory, report)

    state = {
        "schema": STATE_SCHEMA,
        "modelFormat": MODEL_FORMAT,
        "checkpointFormat": CHECKPOINT_FORMAT,
        "selection": "fixed-final-epoch",
        "modelConfig": report["modelConfig"],
        "pretrain": attested_dataset_identity(pretrain, pretrain_input),
        "molecularPredictionQuery": attested_query_identity(query, query_input),
        "admissionEvidence": evidence.as_dict(),
        "modelParameterSha256": report["modelParameterSha256"],
        "checkpointContentSha256": checkpoint_sha256,
        "predictionContentSha256": prediction_sha256,
        "trainingReportContentSha256": report_sha256,
        "predictionsTargetFree": True,
        "heldTruthAccessible": False,
        "omfPriorAdmissionRequired": True,
        "environmentAttestedNotPortable": True,
        "releasePortable": False,
        "releaseBlockers": RELEASE_BLOCKERS,
    }
    outputs = {
        "modelState": state,
        "parameterCount": report["parameterCount"],
        "checkpointContentSha256": checkpoint_sha256,
        "predictionContentSha256": prediction_sha256,
        "trainingReportContentSha256": report_sha256,
        "predictionRecords": prediction_records,
        "predictionQueries": prediction_queries,
        "predictionsTargetFree": True,
        "heldTruthAccessible": False,
        "omfPriorAdmissionRequired": True,
        "environmentAttestedNotPortable": True,
        "releasePortable": False,
    }
    return ProtocolResult(
        status="ok",
        outputs=outputs,
        state=state,
        metrics={
            "pretrain_final_epoch_mean_record_nll": report["training"][
                "epochOptimizationMeanRecordNll"
            ][-1]
        },
        artifacts=[
            {"name": "worldCheckpoint", "kind": "checkpoint", "path": checkpoint_path.name},
            {"name": "molecularValidationPredictions", "kind": "prediction", "path": prediction_path.name},
            {"name": "trainingReport", "kind": "evaluation", "path": report_path.name},
        ],
    )


def _training_config(value: object, config_type: type) -> object:
    if not isinstance(value, dict):
        raise ValueError("training config must be an object")
    mapping = {
        "seed": "seed", "epochs": "epochs", "drawsPerEpoch": "draws_per_epoch",
        "batchSize": "batch_size", "learningRate": "learning_rate",
        "weightDecay": "weight_decay", "gradientClipNorm": "gradient_clip_norm",
        "predictionBatchSize": "prediction_batch_size", "dModel": "d_model",
        "nhead": "nhead", "encoderLayers": "encoder_layers",
        "decoderLayers": "decoder_layers", "ffnMultiplier": "ffn_multiplier",
        "dropout": "dropout",
    }
    unexpected = set(value) - set(mapping)
    if unexpected:
        raise ValueError("unsupported training config fields: " + ", ".join(sorted(unexpected)))
    return config_type(**{mapping[name]: item for name, item in value.items()})


if __name__ == "__main__":
    raise SystemExit(main({"validate": validate, "run": run}))
