"""OMF protocol entry point for composite-keyed proteome corpus v1.2."""

from __future__ import annotations

import os
from pathlib import Path

from omf.sdk import ProtocolRequest, ProtocolResult, main

ZERO = "0" * 64
REQUIRED_INPUTS = frozenset(
    {"observations", "staticFeatures", "heldInterventionRoster"}
)
BOUND_DEFAULTS = {
    "maxManifestBytes": 2 * 1024 * 1024,
    "maxLineBytes": 64 * 1024,
    "maxArchiveBytes": 256 * 1024 * 1024,
    "maxRecords": 100_000,
    "maxEntities": 100_000,
    "maxReadouts": 100_000,
    "maxTargetValues": 20_000_000,
}


def _validate_surface(request: ProtocolRequest, operation: str) -> None:
    if request.operation != operation:
        raise ValueError(f"{operation} handler requires operation={operation}")
    if set(request.inputs) != REQUIRED_INPUTS:
        raise ValueError(
            "composer requires exactly fitting observations, static features, and the "
            "outcome-blind held roster; protected outcomes, rewards, and benchmarks are forbidden"
        )
    if request.state:
        raise ValueError("proteome corpus composer is stateless")
    extra = set(request.config) - set(BOUND_DEFAULTS)
    if extra:
        raise ValueError(
            "unsupported composer config keys: " + ", ".join(sorted(extra))
        )


def _bounds(request: ProtocolRequest):
    from composer import Bounds

    return Bounds(
        max_manifest_bytes=request.config.get(
            "maxManifestBytes", BOUND_DEFAULTS["maxManifestBytes"]
        ),
        max_line_bytes=request.config.get(
            "maxLineBytes", BOUND_DEFAULTS["maxLineBytes"]
        ),
        max_archive_bytes=request.config.get(
            "maxArchiveBytes", BOUND_DEFAULTS["maxArchiveBytes"]
        ),
        max_records=request.config.get("maxRecords", BOUND_DEFAULTS["maxRecords"]),
        max_entities=request.config.get("maxEntities", BOUND_DEFAULTS["maxEntities"]),
        max_readouts=request.config.get("maxReadouts", BOUND_DEFAULTS["maxReadouts"]),
        max_target_values=request.config.get(
            "maxTargetValues", BOUND_DEFAULTS["maxTargetValues"]
        ),
    )


def _validation_outputs() -> dict[str, object]:
    return {
        "auditSummary": {
            "schema": "slp.proteome-corpus-compose-validation/v1",
            "validationOnly": True,
        },
        "archiveSha256": ZERO,
        "auditSha256": ZERO,
        "manifestSha256": ZERO,
        "featurePackSha256": ZERO,
        "entityKeySetSha256": ZERO,
        "featureEntityKeySetSha256": ZERO,
        "targetValueBytesSha256": ZERO,
        "entities": 0,
        "featureRows": 0,
        "contexts": 0,
        "queries": 0,
        "panels": 0,
        "trajectoryInterventions": 0,
        "records": 0,
        "targetValues": 0,
        "shards": 0,
    }


def validate(request: ProtocolRequest) -> ProtocolResult:
    _validate_surface(request, "validate")
    _bounds(request)
    return ProtocolResult(status="ok", outputs=_validation_outputs())


def run(request: ProtocolRequest) -> ProtocolResult:
    from composer import build_composite_corpus, resolve_pinned_dataset

    _validate_surface(request, "run")
    result_file = os.environ.get("OMF_RESULT_FILE")
    if not result_file:
        raise ValueError("OMF_RESULT_FILE is required for artifact placement")
    output_root = Path(result_file).parent
    result = build_composite_corpus(
        resolve_pinned_dataset(request.inputs["observations"], "observations"),
        resolve_pinned_dataset(request.inputs["staticFeatures"], "staticFeatures"),
        resolve_pinned_dataset(
            request.inputs["heldInterventionRoster"], "heldInterventionRoster"
        ),
        output_root / "proteome-corpus-compose-v1",
        _bounds(request),
    )
    public_result_keys = {
        "archiveSha256",
        "auditSha256",
        "manifestSha256",
        "featurePackSha256",
        "entityKeySetSha256",
        "featureEntityKeySetSha256",
        "targetValueBytesSha256",
        "entities",
        "featureRows",
        "contexts",
        "queries",
        "panels",
        "trajectoryInterventions",
        "records",
        "targetValues",
        "shards",
    }
    outputs = {
        "auditSummary": {
            "schema": "slp.proteome-corpus-compose-summary/v1",
            "validationOnly": False,
            "inputNames": sorted(REQUIRED_INPUTS),
            "role": "pretrain",
            "identityKey": ["ncbiTaxon", "entityId"],
            "protectedInterventionOverlap": 0,
            "benchmarkLabelsConsumed": False,
            "rewardDataConsumed": False,
            "targetValuesBytePreserved": True,
        },
        **{key: result[key] for key in public_result_keys},
    }
    return ProtocolResult(
        status="ok",
        outputs=outputs,
        metrics={
            "entities": result["entities"],
            "feature_rows": result["featureRows"],
            "records": result["records"],
            "target_values": result["targetValues"],
            "protected_intervention_overlap": 0,
        },
        artifacts=[
            {
                "name": "proteomeCompositeCorpus",
                "kind": "dataset",
                "path": "proteome-corpus-compose-v1/corpus-v1-2.tar",
            },
            {
                "name": "proteomeCompositeCorpusAudit",
                "kind": "audit",
                "path": "proteome-corpus-compose-v1/corpus-compose-audit.json",
            },
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main({"validate": validate, "run": run}))
