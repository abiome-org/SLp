"""OMF entry point for the outcome-blind atlas genotype inventory."""

from __future__ import annotations

import os
from pathlib import Path

from omf_protocol import ProtocolRequest, ProtocolResult, main


def _validation_outputs() -> dict[str, object]:
    return {
        "inventorySummary": {
            "schema": "slp.atlas-genotype-identity-audit/v1",
            "validationOnly": True,
        },
        "auditSha256": "0" * 64,
        "inventoryManifestSha256": "0" * 64,
        "evidenceManifestSha256": "0" * 64,
        "candidateNonWildTypeIntersection": 0,
        "uniqueCurrentInterventions": 0,
        "retiredOrMergedCandidateAssignments": 0,
        "unmatchedCandidateAssignments": 0,
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

    expected = {"rawAtlasSummary", *MAPPING_ARTIFACT_DIGESTS}
    if set(request.inputs) != expected:
        raise ValueError("run requires one raw atlas summary and four exact SGD artifacts")
    raw = resolve_pinned_raw_dataset(request.inputs["rawAtlasSummary"])
    artifacts = {
        name: resolve_literal_artifact(request.inputs[name], name, digest)
        for name, digest in sorted(MAPPING_ARTIFACT_DIGESTS.items())
    }
    config = request.config
    bounds = Bounds(
        max_frame_rows=config.get("maxFrameRows", 4_000),
        max_assignment_bytes=config.get("maxAssignmentBytes", 256),
        max_mapping_records=config.get("maxMappingRecords", 300_000),
        max_mapping_artifact_bytes=config.get("maxMappingArtifactBytes", 134_217_728),
        max_line_bytes=config.get("maxLineBytes", 2_097_152),
        max_evidence_records=config.get("maxEvidenceRecords", 4_000),
    )
    output_root = Path(os.environ["OMF_RESULT_FILE"]).parent
    result = build_inventory(
        raw.path,
        {name: artifact.path for name, artifact in artifacts.items()},
        output_root / "atlas-genotype-inventory",
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
        "evidenceManifestSha256": result["evidenceManifestSha256"],
        "candidateNonWildTypeIntersection": result["candidateNonWildTypeIntersection"],
        "uniqueCurrentInterventions": result["uniqueCurrentInterventions"],
        "retiredOrMergedCandidateAssignments": result[
            "retiredOrMergedCandidateAssignments"
        ],
        "unmatchedCandidateAssignments": result["unmatchedCandidateAssignments"],
    }
    return ProtocolResult(
        status="ok",
        outputs=outputs,
        metrics={
            "candidate_non_wild_type_intersection": outputs[
                "candidateNonWildTypeIntersection"
            ],
            "unique_current_interventions": outputs["uniqueCurrentInterventions"],
            "retired_or_merged_candidate_assignments": outputs[
                "retiredOrMergedCandidateAssignments"
            ],
            "unmatched_candidate_assignments": outputs[
                "unmatchedCandidateAssignments"
            ],
        },
        artifacts=[
            {
                "name": "atlasInterventionInventory",
                "kind": "dataset",
                "path": "atlas-genotype-inventory/intervention-inventory/inventory.json",
            },
            {
                "name": "atlasInterventionRecords",
                "kind": "dataset",
                "path": "atlas-genotype-inventory/intervention-inventory/interventions.jsonl",
            },
            {
                "name": "atlasGenotypeIdentityEvidence",
                "kind": "audit",
                "path": "atlas-genotype-inventory/identity-evidence/manifest.json",
            },
            {
                "name": "atlasGenotypeIdentityEvidenceRecords",
                "kind": "audit",
                "path": "atlas-genotype-inventory/identity-evidence/evidence.jsonl",
            },
            {
                "name": "atlasGenotypeIdentityQuarantineRecords",
                "kind": "audit",
                "path": "atlas-genotype-inventory/identity-evidence/quarantine.jsonl",
            },
            {
                "name": "atlasGenotypeIdentityAudit",
                "kind": "audit",
                "path": "atlas-genotype-inventory/audit.json",
            },
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main({"validate": validate, "run": run}))
