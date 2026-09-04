"""OMF entry point for the fail-closed corpus audit v1.2."""

from __future__ import annotations

import os
from pathlib import Path

from omf.sdk import ProtocolRequest, ProtocolResult, main

from audit import AUDIT_SCHEMA, AuditBounds, EXPECTED_ROLES, write_audit_artifact


def _validation_outputs() -> dict[str, object]:
    return {
        "auditSummary": {"schema": AUDIT_SCHEMA, "validationOnly": True},
        "auditSha256": "0" * 64,
        "rewardEnabled": False,
        "auditPassed": 0,
        "leakageViolations": 0,
        "benchmarkLabelRecords": 0,
        "pretrainRecords": 0,
        "rewardRecords": 0,
        "validationRecords": 0,
        "finalRecords": 0,
    }


def validate(_request: ProtocolRequest) -> ProtocolResult:
    return ProtocolResult(status="ok", outputs=_validation_outputs())


def run(request: ProtocolRequest) -> ProtocolResult:
    if request.config.get("rewardEnabled") is not False:
        raise ValueError(
            "corpus-audit v1.2 requires rewardEnabled=false; molecular reward "
            "needs a new versioned contract"
        )
    fixed_inputs = {*EXPECTED_ROLES, "heldRoster"}
    protected_inputs = {
        name: value
        for name, value in request.inputs.items()
        if name.startswith("protectedInventory") and len(name) > len("protectedInventory")
    }
    if set(request.inputs) != fixed_inputs | set(protected_inputs) or len(protected_inputs) < 2:
        raise ValueError(
            "inputs require exactly pretrain, molecularValidation, molecularFinal, "
            "heldRoster, and at least two "
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
        max_coverage_exclusions=config.get("maxCoverageExclusions", 8_000_000),
        max_npz_members=config.get("maxNpzMembers", 256),
        max_inventory_files_per_source=config.get("maxInventoryFilesPerSource", 32),
        max_inventory_records_per_source=config.get(
            "maxInventoryRecordsPerSource", 2_000_000
        ),
    )
    output_root = Path(os.environ["OMF_RESULT_FILE"]).parent
    audit, audit_sha256 = write_audit_artifact(
        {name: request.inputs[name] for name in EXPECTED_ROLES},
        request.inputs["heldRoster"],
        protected_inputs,
        output_root / "corpus-audit",
        bounds,
        reward_enabled=request.config["rewardEnabled"],
    )
    records = {
        name: audit["datasets"][name]["records"] for name in EXPECTED_ROLES
    }
    outputs = {
        "auditSummary": audit,
        "auditSha256": audit_sha256,
        "rewardEnabled": False,
        "auditPassed": int(audit["auditPassed"]),
        "leakageViolations": audit["leakageViolations"],
        "benchmarkLabelRecords": audit["benchmarkLabelRecords"],
        "pretrainRecords": records["pretrain"],
        "rewardRecords": 0,
        "validationRecords": records["molecularValidation"],
        "finalRecords": records["molecularFinal"],
    }
    return ProtocolResult(
        status="ok",
        outputs=outputs,
        metrics={
            "leakage_violations": 0,
            "benchmark_label_records": 0,
            "pretrain_records": outputs["pretrainRecords"],
            "reward_records": outputs["rewardRecords"],
            "validation_records": outputs["validationRecords"],
            "final_records": outputs["finalRecords"],
        },
        artifacts=[
            {
                "name": "corpusAudit",
                "kind": "audit",
                "path": "corpus-audit/corpus-audit.json",
            }
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main({"validate": validate, "run": run}))
