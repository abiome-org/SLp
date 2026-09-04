"""OMF entry point for the fail-closed clean-training corpus audit v1.3."""

from __future__ import annotations

import os
from pathlib import Path

from omf.sdk import ProtocolRequest, ProtocolResult, main

from audit import AUDIT_SCHEMA, AuditBounds, write_training_audit_artifact


def _validation_outputs() -> dict[str, object]:
    return {
        "auditSummary": {"schema": AUDIT_SCHEMA, "validationOnly": True},
        "auditSha256": "0" * 64,
        "rewardEnabled": False,
        "auditPassed": 0,
        "leakageViolations": 0,
        "benchmarkLabelRecords": 0,
        "pretrainRecords": 0,
        "protectedTruthInputsPresent": False,
        "custodianSignatureVerified": False,
        "protectedInventorySources": 0,
    }


def validate(_request: ProtocolRequest) -> ProtocolResult:
    return ProtocolResult(status="ok", outputs=_validation_outputs())


def run(request: ProtocolRequest) -> ProtocolResult:
    if request.config.get("rewardEnabled") is not False:
        raise ValueError(
            "training corpus-audit v1.3 requires rewardEnabled=false"
        )
    fixed_inputs = {"pretrain", "heldRoster", "custodianBoundaryAttestation"}
    protected_inputs = {
        name: value
        for name, value in request.inputs.items()
        if name.startswith("protectedInventory")
        and len(name) > len("protectedInventory")
    }
    if (
        set(request.inputs) != fixed_inputs | set(protected_inputs)
        or not 2 <= len(protected_inputs) <= 64
    ):
        raise ValueError(
            "inputs require exactly pretrain, heldRoster, "
            "custodianBoundaryAttestation, and two to 64 "
            "protectedInventory* DatasetSnapshots"
        )
    config = request.config
    bounds = AuditBounds(
        max_manifest_bytes=config.get("maxManifestBytes", 4 * 1024 * 1024),
        max_coverage_bytes=config.get("maxCoverageBytes", 64 * 1024 * 1024),
        max_files_per_corpus=config.get("maxFilesPerCorpus", 256),
        max_file_bytes=config.get("maxFileBytes", 8 * 1024 * 1024 * 1024),
        max_total_bytes_per_corpus=config.get(
            "maxTotalBytesPerCorpus", 64 * 1024 * 1024 * 1024
        ),
        max_records_per_corpus=config.get("maxRecordsPerCorpus", 20_000_000),
        max_trajectory_genes=config.get("maxTrajectoryGenes", 2_000_000),
        max_line_bytes=config.get("maxLineBytes", 4_096),
        max_sources=config.get("maxSources", 4_096),
        max_species=config.get("maxSpecies", 128),
        max_entities=config.get("maxEntities", 2_000_000),
        max_identity_array_bytes=config.get(
            "maxIdentityArrayBytes", 2 * 1024 * 1024 * 1024
        ),
        max_roster_records=config.get("maxRosterRecords", 2_000_000),
        max_coverage_exclusions=config.get(
            "maxCoverageExclusions", 8_000_000
        ),
        max_npz_members=config.get("maxNpzMembers", 256),
        max_inventory_files_per_source=config.get(
            "maxInventoryFilesPerSource", 32
        ),
        max_inventory_records_per_source=config.get(
            "maxInventoryRecordsPerSource", 2_000_000
        ),
    )
    output_root = Path(os.environ["OMF_RESULT_FILE"]).parent
    audit, audit_sha256 = write_training_audit_artifact(
        request.inputs["pretrain"],
        request.inputs["heldRoster"],
        protected_inputs,
        request.inputs["custodianBoundaryAttestation"],
        output_root / "training-corpus-audit",
        bounds,
        reward_enabled=False,
        recipient_factory_identity=config["recipientFactoryIdentity"],
        challenge_nonce=config["challengeNonce"],
    )
    outputs = {
        "auditSummary": audit,
        "auditSha256": audit_sha256,
        "rewardEnabled": False,
        "auditPassed": int(audit["auditPassed"]),
        "leakageViolations": audit["leakageViolations"],
        "benchmarkLabelRecords": audit["benchmarkLabelRecords"],
        "pretrainRecords": audit["datasets"]["pretrain"]["records"],
        "protectedTruthInputsPresent": False,
        "custodianSignatureVerified": True,
        "protectedInventorySources": len(
            audit["heldRoster"]["sourceInventories"]
        ),
    }
    return ProtocolResult(
        status="ok",
        outputs=outputs,
        metrics={
            "leakage_violations": 0,
            "benchmark_label_records": 0,
            "pretrain_records": outputs["pretrainRecords"],
            "protected_inventory_sources": outputs[
                "protectedInventorySources"
            ],
        },
        artifacts=[
            {
                "name": "trainingCorpusAudit",
                "kind": "audit",
                "path": "training-corpus-audit/corpus-audit.json",
            }
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main({"validate": validate, "run": run}))
