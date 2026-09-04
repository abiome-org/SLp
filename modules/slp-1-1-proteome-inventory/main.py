"""OMF entry point for outcome-blind proteome identity inventories."""

from __future__ import annotations

import os
from pathlib import Path

from omf.sdk import ProtocolRequest, ProtocolResult, main


def _validation_outputs() -> dict[str, object]:
    return {
        "inventorySummary": {"schema": "slp.proteome-identity-audit/v1", "validationOnly": True},
        "auditSha256": "0" * 64,
        "inventoryManifestSha256": "0" * 64,
        "proteinManifestSha256": "0" * 64,
        "eligibleInterventions": 0,
        "quarantineRows": 0,
        "proteinRecords": 0,
        "oneToManyProteinRelations": 0,
    }


def validate(_request: ProtocolRequest) -> ProtocolResult:
    return ProtocolResult(status="ok", outputs=_validation_outputs())


def run(request: ProtocolRequest) -> ProtocolResult:
    from inventory import (
        Bounds,
        MAPPING_ARTIFACT_DIGESTS,
        SourceProvenance,
        build_inventory,
        resolve_literal_artifact,
        resolve_pinned_raw_dataset,
    )

    expected = {"rawProteome", *MAPPING_ARTIFACT_DIGESTS}
    if set(request.inputs) != expected:
        raise ValueError("run requires one raw proteome snapshot and four exact SGD artifacts")
    raw = resolve_pinned_raw_dataset(request.inputs["rawProteome"])
    artifacts = {
        name: resolve_literal_artifact(request.inputs[name], name, digest)
        for name, digest in sorted(MAPPING_ARTIFACT_DIGESTS.items())
    }
    config = request.config
    bounds = Bounds(
        max_metadata_rows=config.get("maxMetadataRows", 6_000),
        max_protein_rows=config.get("maxProteinRows", 2_000),
        max_mapping_records=config.get("maxMappingRecords", 300_000),
        max_line_bytes=config.get("maxLineBytes", 2_097_152),
        max_quarantine_rows=config.get("maxQuarantineRows", 512),
    )
    output_root = Path(os.environ["OMF_RESULT_FILE"]).parent
    result = build_inventory(
        raw.path,
        {name: artifact.path for name, artifact in artifacts.items()},
        output_root / "proteome-inventory",
        bounds,
        provenance=SourceProvenance(
            resource=raw.resource,
            revision=raw.revision,
            manifest_digest=raw.manifest_digest,
            mapping_artifacts={
                name: artifact.artifact_manifest_digest
                for name, artifact in artifacts.items()
            },
        ),
    )
    outputs = {
        "inventorySummary": result["audit"],
        "auditSha256": result["auditSha256"],
        "inventoryManifestSha256": result["inventoryManifestSha256"],
        "proteinManifestSha256": result["proteinManifestSha256"],
        "eligibleInterventions": result["eligibleInterventions"],
        "quarantineRows": result["quarantineRows"],
        "proteinRecords": result["proteinRecords"],
        "oneToManyProteinRelations": result["oneToManyProteinRelations"],
    }
    return ProtocolResult(
        status="ok",
        outputs=outputs,
        metrics={
            "eligible_interventions": outputs["eligibleInterventions"],
            "quarantined_intervention_rows": outputs["quarantineRows"],
            "protein_relations": outputs["proteinRecords"],
            "one_to_many_protein_relations": outputs["oneToManyProteinRelations"],
        },
        artifacts=[
            {
                "name": "proteomeInterventionInventory",
                "kind": "dataset",
                "path": "proteome-inventory/intervention-inventory/inventory.json",
            },
            {
                "name": "proteomeInterventionRecords",
                "kind": "dataset",
                "path": "proteome-inventory/intervention-inventory/interventions.jsonl",
            },
            {
                "name": "proteomeProteinRelations",
                "kind": "dataset",
                "path": "proteome-inventory/protein-relations/manifest.json",
            },
            {
                "name": "proteomeProteinRelationRecords",
                "kind": "dataset",
                "path": "proteome-inventory/protein-relations/relations.jsonl",
            },
            {
                "name": "proteomeIdentityAudit",
                "kind": "audit",
                "path": "proteome-inventory/audit.json",
            },
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main({"validate": validate, "run": run}))
