"""Immutable evidence, target-free predictions, and deterministic checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import tarfile
from typing import Any

import numpy as np
import torch

from slp_sparse_architecture import MODEL_FORMAT, SparseTypedWorldModel, WorldConfig
from slp_sparse_corpus import CorpusIndex, PredictionQueryIndex, pinned_dataset_path
from slp_sparse_training import TrainingConfig, iter_sparse_predictions, model_parameter_sha256


AUDIT_SCHEMA = "slp.corpus-audit/v1.1"
CHECKPOINT_FORMAT = "slp.world-sparse-checkpoint/v1"
CHECKPOINT_MAGIC = b"SLP-WORLD-SPARSE-CHECKPOINT-V1\n"
PREDICTION_SCHEMA = "slp.molecular-evaluation/v2"
ARTIFACT_REPORT_SCHEMA = "slp.sparse-training-artifacts/v2"
MAX_EVIDENCE_BYTES = 4 * 1024 * 1024
MAX_CHECKPOINT_HEADER_BYTES = 8 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024
MAX_MODEL_PARAMETERS = 250_000_000
MAX_TENSORS = 10_000
RELEASE_BLOCKERS = [
    "hash-pinned-offline-wheelhouse-required",
    "omf-1.0-artifact-to-inference-adapter-gap",
    "omf-corpus-audit-producer-lineage-policy-required",
]
ROSTER_ASSIGNMENT_DOMAIN = b"slp-1.1-yeast-global-held-v1\x00"
AUDIT_DATASET_ROLES = {
    "pretrain": "pretrain",
    "molecularReward": "molecular-reward",
    "molecularValidation": "molecular-validation",
    "molecularFinal": "molecular-final",
}


@dataclass(frozen=True)
class EvidenceBinding:
    held_roster_dataset_manifest_digest: str
    held_roster_payload_sha256: str
    validation_genes_sha256: str
    final_genes_sha256: str
    held_union_genes_sha256: str
    validation_gene_count: int
    final_gene_count: int
    held_union_gene_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "corpusAuditAdmissionValidated": True,
            "heldRosterDatasetManifestDigest": self.held_roster_dataset_manifest_digest,
            "heldRosterPayloadSha256": self.held_roster_payload_sha256,
            "validationGenesSha256": self.validation_genes_sha256,
            "finalGenesSha256": self.final_genes_sha256,
            "heldUnionGenesSha256": self.held_union_genes_sha256,
            "validationGeneCount": self.validation_gene_count,
            "finalGeneCount": self.final_gene_count,
            "heldUnionGeneCount": self.held_union_gene_count,
        }


def canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def compact_corpus_identity(corpus: CorpusIndex) -> dict[str, object]:
    return {
        "datasetId": corpus.dataset_id,
        "version": corpus.version,
        "role": corpus.role,
        "contentDigest": corpus.content_digest,
        "trajectoryGeneCount": len(corpus.trajectory_genes),
        "trajectoryGeneSetSha256": canonical_sha256(sorted(corpus.trajectory_genes)),
        "records": sum(item.records for item in corpus.shards),
        "targetValues": sum(item.target_values for item in corpus.shards),
    }


def attested_dataset_identity(corpus: CorpusIndex, dataset_input: object) -> dict[str, object]:
    _validate_dataset_input_shape(dataset_input, "DatasetSnapshot")
    assert isinstance(dataset_input, dict)
    return {
        "resource": dataset_input["resource"],
        "manifestDigest": dataset_input["manifestDigest"],
        **compact_corpus_identity(corpus),
    }


def attested_query_identity(
    query: PredictionQueryIndex, dataset_input: object,
) -> dict[str, object]:
    _validate_dataset_input_shape(dataset_input, "molecularPredictionQuery")
    assert isinstance(dataset_input, dict)
    return {
        "resource": dataset_input["resource"],
        "manifestDigest": dataset_input["manifestDigest"],
        "datasetId": query.dataset_id,
        "version": query.version,
        "role": "molecular-validation-query",
        "queryManifestSha256": query.query_manifest_sha256,
        "records": sum(item.records for item in query.shards),
        "speciesTaxa": list(query.species_taxa),
        "sourceIds": list(query.sources),
    }


def _audit_dataset_identity(corpus: CorpusIndex, dataset_input: object) -> dict[str, object]:
    _validate_dataset_input_shape(dataset_input, "audit DatasetSnapshot")
    assert isinstance(dataset_input, dict)
    return {
        "resource": dataset_input["resource"],
        "revision": str(dataset_input["resource"]).rpartition("@")[2],
        "manifestDigest": dataset_input["manifestDigest"],
        "datasetId": corpus.dataset_id,
        "version": corpus.version,
        "role": corpus.role,
        "corpusManifestSha256": corpus.corpus_manifest_sha256,
        "contentDigest": corpus.content_digest,
        "trajectoryGenesSha256": corpus.trajectory_genes_sha256,
        "trajectoryGeneSetSha256": canonical_sha256(sorted(corpus.trajectory_genes)),
        "trajectoryGeneCount": len(corpus.trajectory_genes),
        "records": sum(item.records for item in corpus.shards),
        "targetValues": sum(item.target_values for item in corpus.shards),
        "modalities": list(corpus.modalities),
        "sourceIds": list(corpus.sources),
        "speciesTaxa": list(corpus.species_taxa),
    }


def _validate_audit_dataset_identity(value: object, role: str, name: str) -> None:
    expected = {
        "resource", "revision", "manifestDigest", "datasetId", "version", "role",
        "corpusManifestSha256", "contentDigest", "trajectoryGenesSha256",
        "trajectoryGeneSetSha256", "trajectoryGeneCount", "records", "targetValues",
        "modalities", "sourceIds", "speciesTaxa",
    }
    if not isinstance(value, dict) or set(value) != expected or value["role"] != role:
        raise ValueError(f"corpus audit {name} identity fields/role are invalid")
    if (
        not _is_revisioned_dataset_resource(value["resource"])
        or value["revision"] != str(value["resource"]).rpartition("@")[2]
        or not _is_prefixed_sha256(value["manifestDigest"])
        or any(
            not _is_sha256(value[field])
            for field in (
                "corpusManifestSha256", "contentDigest", "trajectoryGenesSha256",
                "trajectoryGeneSetSha256",
            )
        )
        or any(type(value[field]) is not int or value[field] < 0 for field in ("trajectoryGeneCount", "targetValues"))
        or type(value["records"]) is not int or value["records"] <= 0
        or not isinstance(value["datasetId"], str) or ":" not in value["datasetId"]
        or not isinstance(value["version"], str) or not value["version"]
        or not isinstance(value["modalities"], list) or not value["modalities"]
        or not isinstance(value["sourceIds"], list) or not value["sourceIds"]
        or not isinstance(value["speciesTaxa"], list) or not value["speciesTaxa"]
    ):
        raise ValueError(f"corpus audit {name} identity provenance is invalid")
    if name in {"molecularValidation", "molecularFinal"} and value["trajectoryGeneCount"] <= 0:
        raise ValueError("corpus audit validation/final intervention populations must be non-empty")


def validate_admitted_training_evidence(
    pretrain: CorpusIndex,
    molecular_query: PredictionQueryIndex,
    pretrain_input: object,
    molecular_query_input: object,
    corpus_audit_artifact: object,
    held_roster_snapshot: object,
    *,
    expected_corpus_audit_manifest_digest: str,
) -> EvidenceBinding:
    """Require exact admitted evidence and recheck global held-gene isolation."""

    if not _is_prefixed_sha256(expected_corpus_audit_manifest_digest):
        raise ValueError("expected corpus-audit artifact manifest digest must be sha256-pinned")
    audit_path, audit_manifest_digest = _resolve_literal_artifact(
        corpus_audit_artifact, "corpusAuditEvidence"
    )
    if audit_manifest_digest != expected_corpus_audit_manifest_digest:
        raise ValueError("corpusAuditEvidence does not match the frozen artifact manifest digest")
    roster_path, coverage_path, roster_manifest_digest = _resolve_roster_snapshot(
        held_roster_snapshot
    )
    audit = _read_json_bounded(audit_path, "corpus audit")
    required = {
        "schema", "auditPassed", "strictInterventionIsolation", "leakageViolations",
        "leakedTrajectoryGenes", "benchmarkLabelRecords", "omfPriorAdmissionRequired",
        "datasets", "heldRoster",
    }
    if not isinstance(audit, dict) or set(audit) != required:
        raise ValueError("corpus audit must be an exact admitted sparse-training attestation")
    if (
        audit["schema"] != AUDIT_SCHEMA
        or audit["auditPassed"] is not True
        or audit["strictInterventionIsolation"] is not True
        or audit["leakageViolations"] != 0
        or audit["leakedTrajectoryGenes"] != []
        or audit["benchmarkLabelRecords"] != 0
        or audit["omfPriorAdmissionRequired"] is not True
    ):
        raise ValueError("prior-admitted zero-leakage, zero-benchmark audit is required")
    datasets = audit["datasets"]
    if not isinstance(datasets, dict) or set(datasets) != set(AUDIT_DATASET_ROLES):
        raise ValueError("corpus audit must bind exactly four quantitative corpora")
    for name, role in AUDIT_DATASET_ROLES.items():
        _validate_audit_dataset_identity(datasets[name], role, name)
    if datasets["pretrain"] != _audit_dataset_identity(pretrain, pretrain_input):
        raise ValueError("corpus audit does not exactly bind the optimizer pretrain input")

    roster_pretrain, validation, final = _read_held_roster(roster_path)
    held_union = validation | final
    held = audit["heldRoster"]
    held_fields = {
        "resource", "revision", "manifestDigest", "rosterSha256", "coverageSha256",
        "assignmentDomainHex", "bucketRule", "identityMappingId",
        "identityMappingSha256", "sourceInventories", "intersectionSize",
        "pretrainGeneCount", "validationGeneSetSha256", "validationGeneCount",
        "finalGeneSetSha256", "finalGeneCount", "unionGeneSetSha256", "unionGeneCount",
    }
    if not isinstance(held, dict) or set(held) != held_fields:
        raise ValueError("corpus audit heldRoster fields do not match v1.1")
    expected_held = {
        "rosterSha256": _sha256_file(roster_path),
        "coverageSha256": _sha256_file(coverage_path),
        "assignmentDomainHex": ROSTER_ASSIGNMENT_DOMAIN.hex(),
        "bucketRule": "int(first-16-lowercase-hex,16) mod 100",
        "intersectionSize": len(roster_pretrain | validation | final),
        "pretrainGeneCount": len(roster_pretrain),
        "validationGeneSetSha256": canonical_sha256(sorted(validation)),
        "validationGeneCount": len(validation),
        "finalGeneSetSha256": canonical_sha256(sorted(final)),
        "finalGeneCount": len(final),
        "unionGeneSetSha256": canonical_sha256(sorted(held_union)),
        "unionGeneCount": len(held_union),
    }
    if any(held.get(name) != value for name, value in expected_held.items()):
        raise ValueError("corpus audit held-roster population binding is mismatched")
    if (
        not _is_revisioned_dataset_resource(held["resource"])
        or not isinstance(held_roster_snapshot, dict)
        or held["resource"] != held_roster_snapshot["resource"]
        or held["revision"] != str(held["resource"]).rpartition("@")[2]
        or held["manifestDigest"] != held_roster_snapshot["manifestDigest"]
        or not _is_prefixed_sha256(held["manifestDigest"])
        or not _is_sha256(held["coverageSha256"])
        or not isinstance(held["identityMappingId"], str) or not held["identityMappingId"]
        or not _is_sha256(held["identityMappingSha256"])
    ):
        raise ValueError("corpus audit held-roster provenance is invalid")
    _validate_source_inventories(held)
    if (
        datasets["molecularValidation"]["trajectoryGeneSetSha256"]
        != held["validationGeneSetSha256"]
        or datasets["molecularValidation"]["trajectoryGeneCount"]
        != held["validationGeneCount"]
        or datasets["molecularFinal"]["trajectoryGeneSetSha256"]
        != held["finalGeneSetSha256"]
        or datasets["molecularFinal"]["trajectoryGeneCount"]
        != held["finalGeneCount"]
    ):
        raise ValueError("validation/final corpora do not match the held-roster populations")
    overlap = sorted(pretrain.trajectory_genes & held_union)
    if overlap:
        raise ValueError(
            "held validation/final genes occur in pretrain quantitative trajectories: "
            + ", ".join(overlap)
        )
    uncovered_validation = set(validation)
    for shard_index in range(len(molecular_query.shards)):
        for record in molecular_query.iter_records(shard_index):
            for identifier in record["interventionIds"]:
                if identifier not in validation:
                    raise ValueError(
                        "prediction-query intervention is outside the admitted validation roster: "
                        + str(identifier)
                    )
                uncovered_validation.discard(str(identifier))
    if uncovered_validation:
        first_missing = min(uncovered_validation)
        raise ValueError(
            "prediction-query intervention domain does not exactly equal the admitted "
            f"validation roster; missing {len(uncovered_validation)} identifiers, first={first_missing}"
        )
    _validate_model_contract_compatibility(pretrain, molecular_query)
    return EvidenceBinding(
        roster_manifest_digest,
        _sha256_file(roster_path),
        str(held["validationGeneSetSha256"]),
        str(held["finalGeneSetSha256"]),
        str(held["unionGeneSetSha256"]),
        len(validation),
        len(final),
        len(held_union),
    )


def write_sparse_checkpoint(
    output_directory: Path,
    model: SparseTypedWorldModel,
    training_report: dict[str, Any],
    evidence: EvidenceBinding,
) -> tuple[Path, str]:
    """Write timestamp-free deterministic bytes and return path and content hash."""

    _validate_core_training_report(training_report)
    if training_report.get("modelParameterSha256") != model_parameter_sha256(model):
        raise ValueError("training report does not describe the checkpoint model")
    if training_report.get("modelConfig") != model.config.as_dict():
        raise ValueError("training report model config does not match the model")
    output_directory = _existing_output_directory(output_directory)
    tensors: list[tuple[dict[str, object], np.ndarray]] = []
    payload_bytes = 0
    for name, tensor in sorted(model.state_dict().items()):
        if tensor.dtype != torch.float32:
            raise ValueError(f"checkpoint tensor {name} is not float32")
        array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        view = memoryview(array).cast("B")
        entry = {
            "name": name, "dtype": "float32-le", "shape": list(array.shape),
            "bytes": view.nbytes, "sha256": hashlib.sha256(view).hexdigest(),
        }
        tensors.append((entry, array))
        payload_bytes += view.nbytes
    if payload_bytes // 4 > MAX_MODEL_PARAMETERS:
        raise ValueError("checkpoint parameter bound exceeded")
    header = {
        "format": CHECKPOINT_FORMAT,
        "modelFormat": MODEL_FORMAT,
        "modelConfig": model.config.as_dict(),
        "trainingConfig": training_report["config"],
        "corpora": training_report["corpora"],
        "admissionEvidence": evidence.as_dict(),
        "coreTrainingReportSha256": training_report["reportSha256"],
        "modelParameterSha256": training_report["modelParameterSha256"],
        "payloadBytes": payload_bytes,
        "tensors": [entry for entry, _ in tensors],
    }
    header_bytes = canonical_json_bytes(header)
    total_bytes = len(CHECKPOINT_MAGIC) + 8 + len(header_bytes) + payload_bytes
    if len(header_bytes) > MAX_CHECKPOINT_HEADER_BYTES or total_bytes > MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint byte bound exceeded")
    partial = output_directory / ".slp-world-sparse-checkpoint.partial"
    if partial.exists():
        raise ValueError("checkpoint staging path already exists")
    try:
        with partial.open("xb") as stream:
            stream.write(CHECKPOINT_MAGIC)
            stream.write(struct.pack(">Q", len(header_bytes)))
            stream.write(header_bytes)
            for _, array in tensors:
                stream.write(memoryview(array).cast("B"))
        digest = _sha256_file(partial)
        final = output_directory / f"slp-world-sparse-{digest}.slpc"
        _install_content_addressed_file(partial, final, digest)
    finally:
        if partial.exists():
            partial.unlink()
    return final, digest


def load_sparse_checkpoint(
    path: str | Path, *, expected_sha256: str
) -> tuple[SparseTypedWorldModel, dict[str, object]]:
    """Validate all bounds and exact sizes before model or payload allocation."""

    if not _is_sha256(expected_sha256):
        raise ValueError("expected checkpoint content digest must be a lowercase SHA-256")
    checkpoint = Path(path).absolute()
    _reject_symlink_components(checkpoint)
    if not checkpoint.is_file():
        raise ValueError("checkpoint must be a regular non-symlink file")
    size = checkpoint.stat().st_size
    if size <= len(CHECKPOINT_MAGIC) + 8 or size > MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint file-size bound is invalid")
    digest = _sha256_file(checkpoint)
    if digest != expected_sha256:
        raise ValueError("checkpoint content digest mismatch")
    with checkpoint.open("rb") as stream:
        if stream.read(len(CHECKPOINT_MAGIC)) != CHECKPOINT_MAGIC:
            raise ValueError("unsupported checkpoint magic")
        encoded = stream.read(8)
        if len(encoded) != 8:
            raise ValueError("truncated checkpoint header length")
        header_size = struct.unpack(">Q", encoded)[0]
        if header_size <= 0 or header_size > MAX_CHECKPOINT_HEADER_BYTES:
            raise ValueError("checkpoint header byte bound is invalid")
        header_bytes = stream.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError("truncated checkpoint header")
        try:
            header = json.loads(header_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("checkpoint header is not canonical JSON") from error
        if canonical_json_bytes(header) != header_bytes:
            raise ValueError("checkpoint header is not canonical JSON")
        _validate_checkpoint_header(header)
        if size != len(CHECKPOINT_MAGIC) + 8 + header_size + header["payloadBytes"]:
            raise ValueError("checkpoint file size does not exactly match its payload")
        config = _bounded_world_config(header["modelConfig"])
        if header["payloadBytes"] // 4 > MAX_MODEL_PARAMETERS:
            raise ValueError("checkpoint parameter bound exceeded")

        previous_rng = torch.random.get_rng_state()
        try:
            model = SparseTypedWorldModel(config).cpu()
        finally:
            torch.random.set_rng_state(previous_rng)
        expected_state = model.state_dict()
        entries = header["tensors"]
        if [item["name"] for item in entries] != sorted(expected_state):
            raise ValueError("checkpoint tensor inventory does not match the model")
        loaded: dict[str, torch.Tensor] = {}
        for entry in entries:
            name = entry["name"]
            expected_tensor = expected_state[name]
            expected_shape = list(expected_tensor.shape)
            expected_bytes = expected_tensor.numel() * 4
            if (
                entry["dtype"] != "float32-le"
                or entry["shape"] != expected_shape
                or entry["bytes"] != expected_bytes
            ):
                raise ValueError(f"checkpoint tensor metadata mismatch for {name}")
            array = np.empty(expected_tensor.numel(), dtype="<f4")
            view = memoryview(array).cast("B")
            offset = 0
            while offset < expected_bytes:
                count = stream.readinto(view[offset:])
                if not count:
                    raise ValueError(f"checkpoint tensor payload is truncated for {name}")
                offset += count
            if hashlib.sha256(view).hexdigest() != entry["sha256"]:
                raise ValueError(f"checkpoint tensor digest mismatch for {name}")
            loaded[name] = torch.from_numpy(array.reshape(expected_shape))
        if stream.read(1):
            raise ValueError("checkpoint contains trailing bytes")
    model.load_state_dict(loaded, strict=True)
    if model_parameter_sha256(model) != header["modelParameterSha256"]:
        raise ValueError("loaded model does not match the recorded parameter digest")
    metadata = {key: value for key, value in header.items() if key != "tensors"}
    metadata["checkpointContentSha256"] = digest
    return model, metadata


def write_target_free_predictions(
    output_directory: Path,
    model: SparseTypedWorldModel,
    molecular_query: PredictionQueryIndex,
    molecular_query_input: object,
    checkpoint_sha256: str,
    *,
    batch_size: int,
) -> tuple[Path, str, int, int]:
    """Emit one deterministic tar file containing manifest plus target-free JSONL."""

    _validate_dataset_input_shape(molecular_query_input, "molecularPredictionQuery")
    if not _is_sha256(checkpoint_sha256):
        raise ValueError("model checkpoint digest is invalid")
    assert isinstance(molecular_query_input, dict)
    output_directory = _existing_output_directory(output_directory)
    partial_records = output_directory / ".slp-world-sparse-predictions.records.partial"
    partial_bundle = output_directory / ".slp-world-sparse-predictions.tar.partial"
    if partial_records.exists() or partial_bundle.exists():
        raise ValueError("prediction staging path already exists")
    record_count = query_count = 0
    try:
        with partial_records.open("xb") as stream:
            for batch in iter_sparse_predictions(model, molecular_query, batch_size=batch_size):
                for row, profile_id in enumerate(batch.profile_id):
                    width = len(batch.readout_ids[row])
                    if width <= 0 or not bool(batch.query_mask[row, :width].all().item()):
                        raise ValueError("prediction does not cover the full declared panel")
                    if bool(batch.query_mask[row, width:].any().item()):
                        raise ValueError("prediction mask exceeds the declared panel")
                    values = batch.parameters[row, :width]
                    if not torch.isfinite(values).all():
                        raise ValueError("prediction contains a non-finite value")
                    distribution_types: list[str] = []
                    prediction_parameters: list[dict[str, float]] = []
                    for offset in range(width):
                        likelihood = int(batch.likelihood_type[row, offset].item())
                        first, second = (float(item) for item in values[offset])
                        if likelihood == 0:
                            distribution_types.append("gaussian")
                            prediction_parameters.append({"mean": first, "logScale": second})
                        elif likelihood == 1:
                            distribution_types.append("negative-binomial")
                            prediction_parameters.append(
                                {"logMean": first, "logInverseDispersion": second}
                            )
                        else:
                            raise ValueError("prediction likelihood type is invalid")
                    record = {
                        "profileId": profile_id,
                        "speciesTaxon": batch.species_taxon[row],
                        "sourceId": batch.source_id[row],
                        "centeringGroup": batch.centering_group[row],
                        "perturbationId": batch.perturbation_id[row],
                        "interventionIds": list(batch.intervention_ids[row]),
                        "readoutIds": list(batch.readout_ids[row]),
                        "distributionTypes": distribution_types,
                        "predictionParameters": prediction_parameters,
                    }
                    stream.write(canonical_json_bytes(record, newline=True))
                    record_count += 1
                    query_count += width
        shard_digest = _sha256_file(partial_records)
        manifest = {
            "schema": PREDICTION_SCHEMA,
            "datasetId": molecular_query.dataset_id,
            "version": molecular_query.version,
            "role": "molecular-validation-predictions",
            "labelClass": "none",
            "benchmarkLabelsPresent": False,
            "valueSpace": molecular_query.value_space,
            "speciesTaxa": list(molecular_query.species_taxa),
            "sourceIds": list(molecular_query.sources),
            "modelCheckpointContentSha256": checkpoint_sha256,
            "queryResource": molecular_query_input["resource"],
            "queryDatasetManifestDigest": molecular_query_input["manifestDigest"],
            "queryManifestSha256": molecular_query.query_manifest_sha256,
            "targetValuesPresent": False,
            "observedMaskPresent": False,
            "shards": [{
                "path": "profiles-000.jsonl", "sha256": shard_digest,
                "bytes": partial_records.stat().st_size, "records": record_count,
            }],
        }
        manifest_bytes = canonical_json_bytes(manifest, newline=True)
        with tarfile.open(partial_bundle, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            _add_canonical_tar_member(
                archive, "evaluation.json", len(manifest_bytes), io.BytesIO(manifest_bytes)
            )
            with partial_records.open("rb") as records:
                _add_canonical_tar_member(
                    archive, "profiles-000.jsonl", partial_records.stat().st_size, records
                )
        digest = _sha256_file(partial_bundle)
        final = output_directory / f"molecular-predictions-{digest}.tar"
        if final.exists():
            if _sha256_file(final) != digest:
                raise ValueError("existing content-hashed prediction artifact is corrupt")
        else:
            os.replace(partial_bundle, final)
    finally:
        if partial_records.exists():
            partial_records.unlink()
        if partial_bundle.exists():
            partial_bundle.unlink()
    return final, digest, record_count, query_count


def _add_canonical_tar_member(
    archive: tarfile.TarFile,
    name: str,
    size: int,
    source: object,
) -> None:
    if name not in {"evaluation.json", "profiles-000.jsonl"} or size <= 0:
        raise ValueError("prediction tar member is invalid")
    member = tarfile.TarInfo(name)
    member.size = size
    member.mode = 0o644
    member.mtime = 0
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    archive.addfile(member, source)


def build_artifact_report(
    core_report: dict[str, Any], *, evidence: EvidenceBinding,
    molecular_query: PredictionQueryIndex, molecular_query_input: object,
    checkpoint_content_sha256: str, prediction_content_sha256: str,
    prediction_records: int, prediction_queries: int,
) -> dict[str, Any]:
    _validate_core_training_report(core_report)
    _validate_dataset_input_shape(molecular_query_input, "molecularPredictionQuery")
    if not _is_sha256(checkpoint_content_sha256) or not _is_sha256(prediction_content_sha256):
        raise ValueError("artifact content digest is invalid")
    if type(prediction_records) is not int or prediction_records <= 0:
        raise ValueError("prediction record count is invalid")
    if type(prediction_queries) is not int or prediction_queries < prediction_records:
        raise ValueError("prediction query count is invalid")
    report = json.loads(json.dumps(core_report, allow_nan=False))
    core_hash = report.pop("reportSha256")
    report["checkpointProduced"] = True
    report["artifactBoundary"] = {
        "schema": ARTIFACT_REPORT_SCHEMA,
        "selection": "fixed-final-epoch",
        "heldTruthAccessible": False,
        "predictionQueryUsedForOptimization": False,
        "coreTrainingReportSha256": core_hash,
        "admissionEvidence": evidence.as_dict(),
        "molecularPredictionQuery": attested_query_identity(molecular_query, molecular_query_input),
        "checkpointContentSha256": checkpoint_content_sha256,
        "predictionContentSha256": prediction_content_sha256,
        "predictionRecords": prediction_records,
        "predictionQueries": prediction_queries,
        "predictionsTargetFree": True,
        "advancementEvaluation": "requires-independent-protected-truth-join",
        "omfPriorAdmissionRequired": True,
        "environmentAttestedNotPortable": True,
        "releasePortable": False,
        "releaseBlockers": RELEASE_BLOCKERS,
    }
    report["reportSha256"] = canonical_sha256(report)
    return report


def write_canonical_report(output_directory: Path, report: dict[str, Any]) -> tuple[Path, str]:
    output_directory = _existing_output_directory(output_directory)
    logical = dict(report)
    logical_digest = logical.pop("reportSha256", None)
    if not _is_sha256(logical_digest) or canonical_sha256(logical) != logical_digest:
        raise ValueError("artifact report canonical digest is invalid")
    data = canonical_json_bytes(report, newline=True)
    digest = hashlib.sha256(data).hexdigest()
    path = output_directory / f"training-report-{digest}.json"
    if path.exists() and path.read_bytes() != data:
        raise ValueError("existing content-hashed training report is corrupt")
    if not path.exists():
        path.write_bytes(data)
    return path, digest


def _validate_checkpoint_header(value: object) -> None:
    required = {
        "format", "modelFormat", "modelConfig", "trainingConfig", "corpora",
        "admissionEvidence", "coreTrainingReportSha256", "modelParameterSha256",
        "payloadBytes", "tensors",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("checkpoint header fields do not match")
    if value["format"] != CHECKPOINT_FORMAT or value["modelFormat"] != MODEL_FORMAT:
        raise ValueError("unsupported checkpoint format")
    _bounded_world_config(value["modelConfig"])
    if not isinstance(value["trainingConfig"], dict) or set(value["trainingConfig"]) != {
        item.name for item in fields(TrainingConfig)
    }:
        raise ValueError("checkpoint training config fields do not match")
    TrainingConfig(**value["trainingConfig"])
    corpora = value["corpora"]
    if not isinstance(corpora, dict) or set(corpora) != {"pretrain"}:
        raise ValueError("checkpoint may bind only the pretrain corpus")
    identity = corpora["pretrain"]
    expected_identity = {
        "datasetId", "version", "role", "contentDigest", "trajectoryGeneCount",
        "trajectoryGeneSetSha256",
    }
    if (
        not isinstance(identity, dict) or set(identity) != expected_identity
        or identity["role"] != "pretrain" or not _is_sha256(identity["contentDigest"])
        or not _is_sha256(identity["trajectoryGeneSetSha256"])
        or type(identity["trajectoryGeneCount"]) is not int
        or identity["trajectoryGeneCount"] < 0
    ):
        raise ValueError("checkpoint pretrain corpus identity is invalid")
    evidence = value["admissionEvidence"]
    expected_evidence = {
        "corpusAuditAdmissionValidated",
        "heldRosterDatasetManifestDigest", "heldRosterPayloadSha256",
        "validationGenesSha256", "finalGenesSha256", "heldUnionGenesSha256",
        "validationGeneCount", "finalGeneCount", "heldUnionGeneCount",
    }
    if not isinstance(evidence, dict) or set(evidence) != expected_evidence:
        raise ValueError("checkpoint admission evidence fields do not match")
    if evidence["corpusAuditAdmissionValidated"] is not True:
        raise ValueError("checkpoint lacks prior corpus-audit admission")
    for name in ("heldRosterDatasetManifestDigest",):
        if not isinstance(evidence[name], str) or not evidence[name].startswith("sha256:") or not _is_sha256(evidence[name][7:]):
            raise ValueError("checkpoint evidence manifest digest is invalid")
    for name in (
        "heldRosterPayloadSha256", "validationGenesSha256", "finalGenesSha256",
        "heldUnionGenesSha256",
    ):
        if not _is_sha256(evidence[name]):
            raise ValueError("checkpoint evidence content digest is invalid")
    for name in ("validationGeneCount", "finalGeneCount", "heldUnionGeneCount"):
        if type(evidence[name]) is not int or evidence[name] <= 0:
            raise ValueError("checkpoint evidence gene count is invalid")
    for name in ("coreTrainingReportSha256", "modelParameterSha256"):
        if not _is_sha256(value[name]):
            raise ValueError(f"checkpoint {name} is not a SHA-256")
    entries = value["tensors"]
    if not isinstance(entries, list) or not entries or len(entries) > MAX_TENSORS:
        raise ValueError("checkpoint tensor inventory is invalid")
    names: list[str] = []
    total = 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"name", "dtype", "shape", "bytes", "sha256"}:
            raise ValueError("checkpoint tensor entry fields do not match")
        shape = entry["shape"]
        if (
            not isinstance(entry["name"], str) or not entry["name"]
            or entry["dtype"] != "float32-le"
            or not isinstance(shape, list) or len(shape) > 8
            or any(type(item) is not int or item < 0 or item > 1_000_000 for item in shape)
            or type(entry["bytes"]) is not int or entry["bytes"] < 0
            or entry["bytes"] != math.prod(shape) * 4
            or not _is_sha256(entry["sha256"])
        ):
            raise ValueError("checkpoint tensor metadata is invalid")
        names.append(entry["name"])
        total += entry["bytes"]
        if total > MAX_CHECKPOINT_BYTES:
            raise ValueError("checkpoint tensor payload bound exceeded")
    if names != sorted(set(names)) or total != value["payloadBytes"]:
        raise ValueError("checkpoint tensor inventory is not canonical")


def _bounded_world_config(value: object) -> WorldConfig:
    expected = {item.name for item in fields(WorldConfig)}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("checkpoint model config fields do not match")
    integer_limits = {
        "entity_feature_dim": 8192, "species_feature_dim": 4096,
        "entity_types": 4096, "context_types": 4096, "action_types": 4096,
        "readout_types": 4096, "record_covariate_dim": 1024,
        "context_covariate_dim": 1024, "action_covariate_dim": 1024,
        "observation_covariate_dim": 1024, "d_model": 2048, "nhead": 256,
        "encoder_layers": 48, "decoder_layers": 48, "ffn_multiplier": 16,
    }
    for name, limit in integer_limits.items():
        item = value[name]
        minimum = 0 if name.endswith("covariate_dim") else 1
        if type(item) is not int or item < minimum or item > limit:
            raise ValueError(f"checkpoint model config {name} violates its bound")
    config = WorldConfig(**value)
    d = config.d_model
    dense_upper = (
        32 * (config.encoder_layers + config.decoder_layers + 1)
        * config.ffn_multiplier * d * d
        + 8 * d * (
            config.entity_feature_dim + config.species_feature_dim
            + config.record_covariate_dim + config.context_covariate_dim
            + config.action_covariate_dim + config.observation_covariate_dim
            + config.entity_types + config.context_types + config.action_types
            + config.readout_types + 8
        )
    )
    if dense_upper > MAX_MODEL_PARAMETERS:
        raise ValueError("checkpoint model parameter upper bound exceeded")
    return config


def _validate_core_training_report(report: object) -> None:
    if not isinstance(report, dict) or "reportSha256" not in report:
        raise ValueError("core training report must be canonical")
    logical = dict(report)
    digest = logical.pop("reportSha256")
    isolation = report.get("isolation")
    if (
        not _is_sha256(digest) or canonical_sha256(logical) != digest
        or report.get("checkpointProduced") is not False
        or "validation" in report or "molecularValidation" in json.dumps(report)
        or not isinstance(isolation, dict)
        or isolation.get("benchmarkLabelsPresent") is not False
        or isolation.get("heldTruthAccessible") is not False
        or isolation.get("predictionQueryUsedForOptimization") is not False
        or isolation.get("selection") != "fixed-final-epoch"
    ):
        raise ValueError("core training report violates the production information boundary")


def _validate_model_contract_compatibility(
    pretrain: CorpusIndex, query: PredictionQueryIndex,
) -> None:
    if pretrain.role != "pretrain" or query.feature_corpus is not pretrain:
        raise ValueError("query features must resolve only through the optimizer pretrain corpus")


def _validate_source_inventories(held: dict[str, object]) -> None:
    expected = {
        "resource", "revision", "artifactManifestDigest",
        "sourceId", "sourceRelease", "identityMappingId", "identityMappingSha256",
        "manifestSha256", "records", "duplicateRecords", "uniqueInterventions",
        "qcPassing", "qcFailed", "intersectionCoverage",
    }
    values = held["sourceInventories"]
    if not isinstance(values, list) or not 2 <= len(values) <= 4096:
        raise ValueError("corpus audit requires at least two held-roster source inventories")
    source_ids: list[str] = []
    resources: list[str] = []
    artifact_digests: list[str] = []
    for value in values:
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("corpus audit source-inventory fields do not match")
        source_id = value["sourceId"]
        source_ids.append(str(source_id))
        resources.append(str(value["resource"]))
        artifact_digests.append(str(value["artifactManifestDigest"]))
        if (
            not _is_revisioned_dataset_resource(value["resource"])
            or value["revision"] != str(value["resource"]).rpartition("@")[2]
            or not _is_prefixed_sha256(value["artifactManifestDigest"])
            or not isinstance(source_id, str) or ":" not in source_id
            or not isinstance(value["sourceRelease"], str) or not value["sourceRelease"]
            or value["identityMappingId"] != held["identityMappingId"]
            or value["identityMappingSha256"] != held["identityMappingSha256"]
            or not _is_sha256(value["manifestSha256"])
            or any(
                type(value[name]) is not int or value[name] < 0
                for name in (
                    "records", "duplicateRecords", "uniqueInterventions",
                    "qcPassing", "qcFailed", "intersectionCoverage",
                )
            )
            or value["intersectionCoverage"] != held["intersectionSize"]
            or value["records"] != value["uniqueInterventions"] + value["duplicateRecords"]
            or value["uniqueInterventions"] != value["qcPassing"] + value["qcFailed"]
            or value["intersectionCoverage"] > value["qcPassing"]
        ):
            raise ValueError("corpus audit source-inventory provenance is invalid")
    if (
        source_ids != sorted(set(source_ids))
        or len(resources) != len(set(resources))
        or len(artifact_digests) != len(set(artifact_digests))
    ):
        raise ValueError("corpus audit source inventories must be sorted and independently pinned")


def _read_held_roster(
    path: Path,
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    if path.stat().st_size > MAX_EVIDENCE_BYTES:
        raise ValueError("held roster exceeds its byte bound")
    validation: set[str] = set()
    final: set[str] = set()
    pretrain: set[str] = set()
    previous = b""
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            if not raw.endswith(b"\n") or len(raw) > 4096:
                raise ValueError(f"held roster line {number} is not canonical")
            line_bytes = raw[:-1]
            if not line_bytes or line_bytes <= previous or b"\r" in line_bytes:
                raise ValueError("held roster must be non-empty, sorted, and duplicate-free")
            previous = line_bytes
            try:
                parts = line_bytes.decode("ascii").split("\t")
            except UnicodeDecodeError as error:
                raise ValueError("held roster identifiers must be canonical ASCII") from error
            if len(parts) != 3 or ":" not in parts[0]:
                raise ValueError(f"held roster line {number} is invalid")
            gene, role, assignment_digest = parts
            expected_digest = hashlib.sha256(
                ROSTER_ASSIGNMENT_DOMAIN + gene.encode("ascii")
            ).hexdigest()
            bucket = int(expected_digest[:16], 16) % 100
            expected_role = (
                "molecular-final" if bucket < 10
                else "molecular-validation" if bucket < 30
                else "pretrain"
            )
            if assignment_digest != expected_digest or role != expected_role:
                raise ValueError("held roster role/digest contradicts the frozen assignment")
            if role == "molecular-validation":
                validation.add(gene)
            elif role == "molecular-final":
                final.add(gene)
            elif role == "pretrain":
                pretrain.add(gene)
            else:
                raise ValueError(f"held roster line {number} has an invalid role")
    if not validation or not final or validation & final:
        raise ValueError("global held roster requires disjoint validation and final genes")
    return frozenset(pretrain), frozenset(validation), frozenset(final)


def _resolve_roster_snapshot(value: object) -> tuple[Path, Path, str]:
    _validate_dataset_input_shape(value, "heldRosterEvidence")
    assert isinstance(value, dict)
    root = Path(pinned_dataset_path(value, "heldRosterEvidence"))
    path = root / "held-intervention-roster.tsv"
    coverage = root / "coverage.json"
    _reject_symlink_components(path)
    _reject_symlink_components(coverage)
    if not path.is_file() or not coverage.is_file():
        raise ValueError("heldRosterEvidence must contain its roster and coverage files")
    _require_exact_root_file_set(
        root,
        {"held-intervention-roster.tsv", "coverage.json"},
        "heldRosterEvidence",
    )
    return path, coverage, str(value["manifestDigest"])


def _resolve_literal_artifact(value: object, name: str) -> tuple[Path, str]:
    expected = {"resource", "kind", "artifacts", "paths", "path"}
    if not isinstance(value, dict) or set(value) != expected or value["kind"] != "artifact":
        raise ValueError(f"{name} must be an exact materialized admitted OMF artifact")
    artifacts, paths, path_value = value["artifacts"], value["paths"], value["path"]
    if not isinstance(artifacts, dict) or set(artifacts) != {"payload"}:
        raise ValueError(f"{name}.artifacts must contain only payload")
    manifest_digest = artifacts["payload"]
    if not isinstance(manifest_digest, str) or not manifest_digest.startswith("sha256:") or not _is_sha256(manifest_digest[7:]):
        raise ValueError(f"{name} payload must be an admitted artifact digest")
    if value["resource"] != f"artifact:{manifest_digest}" or paths != {"payload": path_value}:
        raise ValueError(f"{name} artifact identity/path binding is inconsistent")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{name}.path must be a non-empty string")
    path = Path(path_value).absolute()
    _reject_symlink_components(path)
    path = path.resolve(strict=True)
    if not path.is_file() or path.name != "payload" or path.parent.name != "payload":
        raise ValueError(
            f"{name} must use OMF file artifact semantics .../payload/payload"
        )
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_EVIDENCE_BYTES:
        raise ValueError(f"{name} payload byte size is outside bounds")
    return path, manifest_digest


def _require_exact_root_file_set(
    root: Path, expected_names: set[str], name: str,
) -> None:
    """Validate a tiny flat artifact without an unbounded recursive accumulation."""

    seen: set[str] = set()
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if len(seen) >= len(expected_names):
                    raise ValueError(f"{name} contains undeclared entries")
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise ValueError(f"{name} contains a non-regular or nested entry")
                if entry.name not in expected_names or entry.name in seen:
                    raise ValueError(f"{name} contains undeclared entries")
                seen.add(entry.name)
    except OSError as error:
        raise ValueError(f"could not inspect {name} artifact directory") from error
    if seen != expected_names:
        raise ValueError(f"{name} is missing required files")


def _validate_dataset_input_shape(value: object, name: str) -> None:
    required = {"manifestDigest", "mode", "path", "resource"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{name} must be an exact materialized OMF DatasetSnapshot")
    if value["mode"] != "copy" or not isinstance(value["manifestDigest"], str) or not value["manifestDigest"].startswith("sha256:") or not _is_sha256(value["manifestDigest"][7:]):
        raise ValueError(f"{name} must be an immutable admission-pinned copy")
    if not isinstance(value["resource"], str) or not value["resource"].startswith("omf://") or "/datasetsnapshot/" not in value["resource"]:
        raise ValueError(f"{name}.resource must be a revisioned DatasetSnapshot URI")
    _, separator, revision = value["resource"].rpartition("@")
    if not separator or not revision.startswith("sha256:") or not _is_sha256(revision[7:]):
        raise ValueError(f"{name}.resource must be a revisioned DatasetSnapshot URI")


def _read_json_bounded(path: Path, name: str) -> object:
    if path.stat().st_size > MAX_EVIDENCE_BYTES:
        raise ValueError(f"{name} exceeds its byte bound")
    try:
        return json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error


def _existing_output_directory(path: Path) -> Path:
    value = Path(path).absolute()
    _reject_symlink_components(value)
    if not value.is_dir():
        raise ValueError("artifact output directory must be a regular directory")
    return value


def _install_content_addressed_file(partial: Path, final: Path, digest: str) -> None:
    if final.exists():
        if _sha256_file(final) != digest:
            raise ValueError("existing content-hashed checkpoint is corrupt")
        partial.unlink()
    else:
        os.replace(partial, final)


def _tree_sha256(root: Path) -> str:
    entries = [
        {"path": path.relative_to(root).as_posix(), "sha256": _sha256_file(path)}
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path.is_file()
    ]
    return canonical_sha256(entries)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _is_prefixed_sha256(value: object) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and _is_sha256(value[7:])


def _is_revisioned_dataset_resource(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("omf://") or "/datasetsnapshot/" not in value:
        return False
    identity, separator, revision = value.rpartition("@")
    return bool(separator and identity.rpartition("/")[2] and _is_prefixed_sha256(revision))


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"symlink path component is forbidden: {cursor}")
