"""OMF protocol entry point for the pinned SGD stable-ID mapping module."""

from __future__ import annotations

from pathlib import Path

from omf.sdk import ProtocolRequest, ProtocolResult, main


def _validation_outputs() -> dict[str, object]:
    return {
        "mappingSummary": {
            "schema": "slp.sgd-stable-id-mapping/v1",
            "validationOnly": True,
        },
        "identityMappingId": "slp-sgd-map:2026-08-28-object-set-v1",
        "identityMappingSha256": "0" * 64,
        "mappingManifestSha256": "0" * 64,
        "currentOrfCount": 0,
        "typedExternalRelationCount": 0,
        "oneToManyExternalRelationCount": 0,
        "retiredQuarantineCount": 0,
        "retiredIrregularCount": 0,
    }


def validate(_request: ProtocolRequest) -> ProtocolResult:
    return ProtocolResult(status="ok", outputs=_validation_outputs())


def run(request: ProtocolRequest) -> ProtocolResult:
    from mapper import (
        MapBounds,
        SourceProvenance,
        normalize_sgd_snapshot,
        resolve_pinned_dataset_input,
    )

    if set(request.inputs) != {"rawSgdMapping"}:
        raise ValueError("run requires exactly one rawSgdMapping DatasetSnapshot input")
    resolved = resolve_pinned_dataset_input(
        request.inputs["rawSgdMapping"], "rawSgdMapping"
    )
    config = request.config
    bounds = MapBounds(
        max_feature_records=config.get("maxFeatureRecords", 20_000),
        max_external_records=config.get("maxExternalRecords", 300_000),
        max_retired_physical_lines=config.get("maxRetiredPhysicalLines", 256),
        max_line_bytes=config.get("maxLineBytes", 4_096),
        max_targets_per_external_key=config.get("maxTargetsPerExternalKey", 1_024),
        max_assertions_per_external_target=config.get(
            "maxAssertionsPerExternalTarget", 64
        ),
        max_display_aliases_per_orf=config.get("maxDisplayAliasesPerOrf", 128),
    )
    result = normalize_sgd_snapshot(
        resolved.path,
        Path(request.outputs_dir) / "sgd-map",
        bounds,
        source_provenance=SourceProvenance(
            resource=resolved.resource,
            revision=resolved.revision,
            manifest_digest=resolved.manifest_digest,
        ),
    )
    return ProtocolResult(
        status="ok",
        outputs={
            "mappingSummary": result["mappingManifest"],
            "identityMappingId": result["identityMappingId"],
            "identityMappingSha256": result["identityMappingSha256"],
            "mappingManifestSha256": result["mappingManifestSha256"],
            "currentOrfCount": result["currentOrfCount"],
            "typedExternalRelationCount": result["typedExternalRelationCount"],
            "oneToManyExternalRelationCount": result[
                "oneToManyExternalRelationCount"
            ],
            "retiredQuarantineCount": result["retiredQuarantineCount"],
            "retiredIrregularCount": result["retiredIrregularCount"],
        },
        metrics={
            "current_orfs": result["currentOrfCount"],
            "typed_external_relations": result["typedExternalRelationCount"],
            "one_to_many_external_relations": result[
                "oneToManyExternalRelationCount"
            ],
            "retired_quarantine_records": result["retiredQuarantineCount"],
        },
        artifacts=[
            {
                "name": "sgdCurrentOrfs",
                "kind": "dataset",
                "path": "sgd-map/current-orfs.jsonl",
            },
            {
                "name": "sgdExternalAccessionRelations",
                "kind": "dataset",
                "path": "sgd-map/external-accessions.jsonl",
            },
            {
                "name": "sgdRetiredMergedQuarantine",
                "kind": "audit",
                "path": "sgd-map/retired-merged-quarantine.jsonl",
            },
            {
                "name": "sgdMappingManifest",
                "kind": "audit",
                "path": "sgd-map/mapping-manifest.json",
            },
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main({"validate": validate, "run": run}))
