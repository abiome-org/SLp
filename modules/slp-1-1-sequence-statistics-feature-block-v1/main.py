"""OMF entry point for the frozen sequence-statistics feature block."""

from __future__ import annotations

import os
from pathlib import Path

from omf.sdk import ProtocolRequest, ProtocolResult, main

ZERO = "0" * 64
REQUIRED_INPUTS = frozenset(
    {
        "staticEntityUniverse",
        "sgdProteinSequences",
        "sgdCurrentOrfs",
        "sgdMappingManifest",
    }
)
BOUND_DEFAULTS = {
    "maxManifestBytes": 1_048_576,
    "maxLineBytes": 16_384,
    "maxFastaBytes": 16 * 1024 * 1024,
    "maxSequenceLength": 100_000,
    "maxRecords": 20_000,
    "maxArchiveBytes": 64 * 1024 * 1024,
}


def _validation_outputs() -> dict[str, object]:
    return {
        "auditSummary": {
            "schema": "slp.sequence-statistics-feature-block-validation/v1",
            "validationOnly": True,
        },
        "archiveSha256": ZERO,
        "auditSha256": ZERO,
        "manifestSha256": ZERO,
        "entityRowsSha256": ZERO,
        "valuesNpySha256": ZERO,
        "presentNpySha256": ZERO,
        "sequenceProvenanceSha256": ZERO,
        "excludedNonCurrentSha256": ZERO,
        "featureDefinitionSha256": ZERO,
        "entityKeySetSha256": ZERO,
        "rows": 0,
        "featureDimension": 21,
        "presentValues": 0,
        "excludedNonCurrentSequences": 0,
        "currentOrfsOutsideUniverse": 0,
        "multiTargetProteinConsensus": 0,
    }


def _validate_request_surface(request: ProtocolRequest, operation: str) -> None:
    if request.operation != operation:
        raise ValueError(f"{operation} handler requires operation={operation}")
    if set(request.inputs) != REQUIRED_INPUTS:
        raise ValueError(
            f"{operation} requires exactly staticEntityUniverse, "
            "sgdProteinSequences, sgdCurrentOrfs, and sgdMappingManifest; "
            "held rosters, quantitative outcomes, rewards, labels, and "
            "benchmarks are forbidden"
        )
    if request.state:
        raise ValueError("sequence-statistics feature-block v1 is stateless")
    unexpected_config = set(request.config) - set(BOUND_DEFAULTS)
    if unexpected_config:
        raise ValueError(
            "unsupported sequence-statistics feature-block config keys: "
            + ", ".join(sorted(unexpected_config))
        )


def _bounds_from_config(request: ProtocolRequest, bounds_type: type) -> object:
    config = request.config
    return bounds_type(
        max_manifest_bytes=config.get(
            "maxManifestBytes", BOUND_DEFAULTS["maxManifestBytes"]
        ),
        max_line_bytes=config.get("maxLineBytes", BOUND_DEFAULTS["maxLineBytes"]),
        max_fasta_bytes=config.get(
            "maxFastaBytes", BOUND_DEFAULTS["maxFastaBytes"]
        ),
        max_sequence_length=config.get(
            "maxSequenceLength", BOUND_DEFAULTS["maxSequenceLength"]
        ),
        max_records=config.get("maxRecords", BOUND_DEFAULTS["maxRecords"]),
        max_archive_bytes=config.get(
            "maxArchiveBytes", BOUND_DEFAULTS["maxArchiveBytes"]
        ),
    )


def validate(request: ProtocolRequest) -> ProtocolResult:
    from feature_block import Bounds

    _validate_request_surface(request, "validate")
    _bounds_from_config(request, Bounds)
    return ProtocolResult(status="ok", outputs=_validation_outputs())


def run(request: ProtocolRequest) -> ProtocolResult:
    from feature_block import (
        Bounds,
        build_sequence_feature_block,
        resolve_literal_artifact,
        resolve_pinned_dataset,
    )

    _validate_request_surface(request, "run")
    universe = resolve_pinned_dataset(
        request.inputs["staticEntityUniverse"], "staticEntityUniverse"
    )
    sequences = resolve_pinned_dataset(
        request.inputs["sgdProteinSequences"], "sgdProteinSequences"
    )
    current = resolve_literal_artifact(
        request.inputs["sgdCurrentOrfs"], "sgdCurrentOrfs"
    )
    mapping = resolve_literal_artifact(
        request.inputs["sgdMappingManifest"], "sgdMappingManifest"
    )
    bounds = _bounds_from_config(request, Bounds)
    result_file = os.environ.get("OMF_RESULT_FILE")
    if not result_file:
        raise ValueError("OMF_RESULT_FILE is required for artifact placement")
    output_root = Path(result_file).parent
    result = build_sequence_feature_block(
        universe,
        sequences,
        current,
        mapping,
        output_root / "sequence-statistics-feature-block-v1",
        bounds,
    )
    output_keys = (
        "archiveSha256", "auditSha256", "manifestSha256", "entityRowsSha256",
        "valuesNpySha256", "presentNpySha256", "sequenceProvenanceSha256",
        "excludedNonCurrentSha256", "featureDefinitionSha256", "entityKeySetSha256",
        "rows", "featureDimension", "presentValues", "excludedNonCurrentSequences",
        "currentOrfsOutsideUniverse", "multiTargetProteinConsensus",
    )
    outputs = {
        "auditSummary": {
            "schema": "slp.sequence-statistics-feature-block-summary/v1",
            "validationOnly": False,
            "inputNames": sorted(REQUIRED_INPUTS),
            "heldRosterConsumed": False,
            "quantitativeOutcomesConsumed": False,
            "benchmarkDataConsumed": False,
            "rows": result["rows"],
            "featureDimension": result["featureDimension"],
            "presentValues": result["presentValues"],
            "excludedNonCurrentSequences": result["excludedNonCurrentSequences"],
            "currentOrfsOutsideUniverse": result["currentOrfsOutsideUniverse"],
            "multiTargetProteinConsensus": result["multiTargetProteinConsensus"],
        },
        **{key: result[key] for key in output_keys},
    }
    return ProtocolResult(
        status="ok",
        outputs=outputs,
        metrics={
            "rows": outputs["rows"],
            "feature_dimension": outputs["featureDimension"],
            "present_values": outputs["presentValues"],
            "excluded_non_current_sequences": outputs["excludedNonCurrentSequences"],
            "multi_target_protein_consensus": outputs["multiTargetProteinConsensus"],
        },
        artifacts=[
            {
                "name": "sequenceStatisticsFeatureBlock",
                "kind": "dataset",
                "path": "sequence-statistics-feature-block-v1/sequence-feature-block.tar",
            },
            {
                "name": "sequenceStatisticsFeatureBlockAudit",
                "kind": "audit",
                "path": "sequence-statistics-feature-block-v1/sequence-feature-block-audit.json",
            },
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main({"validate": validate, "run": run}))
