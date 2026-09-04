"""OMF entry point for the outcome-blind static entity universe."""

from __future__ import annotations

import os
from pathlib import Path

from omf.sdk import ProtocolRequest, ProtocolResult, main


def _validation_outputs() -> dict[str, object]:
    return {
        "auditSummary": {
            "schema": "slp.static-entity-universe-validation/v1",
            "validationOnly": True,
        },
        "archiveSha256": "0" * 64,
        "auditSha256": "0" * 64,
        "manifestSha256": "0" * 64,
        "entitySetSha256": "0" * 64,
        "relationSetSha256": "0" * 64,
        "actionIdSetSha256": "0" * 64,
        "proteinIdSetSha256": "0" * 64,
        "relationEdgeSetSha256": "0" * 64,
        "fullEntityIdSetSha256": "0" * 64,
        "fullEntityKeySetSha256": "0" * 64,
        "actionEntities": 0,
        "readoutQueryEntities": 0,
        "totalEntities": 0,
        "relationRecords": 0,
        "relationEdges": 0,
        "relationTargetGenes": 0,
        "relationTargetsInUniverse": 0,
        "relationSupportOnly": 0,
        "oneToManyRelations": 0,
    }


def validate(_request: ProtocolRequest) -> ProtocolResult:
    return ProtocolResult(status="ok", outputs=_validation_outputs())


def run(request: ProtocolRequest) -> ProtocolResult:
    from universe import Bounds, build_entity_universe, resolve_pinned_dataset

    if set(request.inputs) != {"interventionInventory", "proteinRelations"}:
        raise ValueError(
            "run requires exactly interventionInventory and proteinRelations; "
            "held rosters, outcomes, labels, and numeric features are forbidden"
        )
    intervention = resolve_pinned_dataset(
        request.inputs["interventionInventory"], "interventionInventory"
    )
    relations = resolve_pinned_dataset(
        request.inputs["proteinRelations"], "proteinRelations"
    )
    config = request.config
    bounds = Bounds(
        max_manifest_bytes=config.get("maxManifestBytes", 1_048_576),
        max_line_bytes=config.get("maxLineBytes", 16_384),
        max_intervention_records=config.get("maxInterventionRecords", 10_000),
        max_relation_records=config.get("maxRelationRecords", 10_000),
        max_archive_bytes=config.get("maxArchiveBytes", 64 * 1024 * 1024),
    )
    output_root = Path(os.environ["OMF_RESULT_FILE"]).parent
    result = build_entity_universe(
        intervention,
        relations,
        output_root / "static-entity-universe-v1",
        bounds,
    )
    outputs = {
        "auditSummary": result["audit"],
        **{key: result[key] for key in (
            "archiveSha256", "auditSha256", "manifestSha256", "entitySetSha256",
            "relationSetSha256", "actionEntities", "readoutQueryEntities",
            "actionIdSetSha256", "proteinIdSetSha256", "relationEdgeSetSha256",
            "fullEntityIdSetSha256",
            "fullEntityKeySetSha256",
            "totalEntities", "relationRecords", "relationEdges", "oneToManyRelations",
            "relationTargetGenes", "relationTargetsInUniverse", "relationSupportOnly",
        )},
    }
    return ProtocolResult(
        status="ok",
        outputs=outputs,
        metrics={
            "action_entities": outputs["actionEntities"],
            "readout_query_entities": outputs["readoutQueryEntities"],
            "relation_records": outputs["relationRecords"],
            "relation_support_only": outputs["relationSupportOnly"],
            "one_to_many_relations": outputs["oneToManyRelations"],
        },
        artifacts=[
            {
                "name": "staticEntityUniverse",
                "kind": "dataset",
                "path": "static-entity-universe-v1/entity-universe.tar",
            },
            {
                "name": "staticEntityUniverseAudit",
                "kind": "audit",
                "path": "static-entity-universe-v1/entity-universe-audit.json",
            },
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main({"validate": validate, "run": run}))
