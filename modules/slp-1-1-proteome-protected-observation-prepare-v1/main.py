"""OMF entry point for one protected proteome truth role per run."""

from __future__ import annotations

import os
from pathlib import Path

from omf.sdk import ProtocolRequest, ProtocolResult, main


PROTECTED_AUDIT_SCHEMA = "slp.proteome-protected-observation-preparation-audit/v1"
ROLES = frozenset({"molecular-validation", "molecular-final"})


def _outputs(role: str) -> dict[str, object]:
    return {
        "preparationSummary": {
            "schema": PROTECTED_AUDIT_SCHEMA,
            "role": role,
            "validationOnly": True,
        },
        "observationArchiveSha256": "0" * 64,
        "auditSha256": "0" * 64,
        "role": role,
        "records": 0,
        "interventionGenes": 0,
        "targetValues": 0,
        "missingValues": 0,
        "unselectedRowsNotDecoded": 0,
    }


def validate(request: ProtocolRequest) -> ProtocolResult:
    role = request.config.get("role")
    if role not in ROLES or set(request.config) != {"role"}:
        raise ValueError("protected preparation requires exactly one governed held role")
    return ProtocolResult(status="ok", outputs=_outputs(role))


def run(request: ProtocolRequest) -> ProtocolResult:
    from protected_prepare import (
        Bounds,
        PRODUCTION_ARTIFACTS,
        PRODUCTION_DATASETS,
        PRODUCTION_EXPECTED_COUNTS,
        PRODUCTION_PROTECTED_ROLES,
        PreparationProvenance,
        build_protected_observations,
        resolve_literal_artifact,
        resolve_pinned_dataset_input,
    )

    expected_inputs = {*PRODUCTION_DATASETS, *PRODUCTION_ARTIFACTS}
    if set(request.inputs) != expected_inputs:
        raise ValueError(
            "protected proteome preparation requires exactly four datasets and two SGD artifacts"
        )
    datasets = {
        name: resolve_pinned_dataset_input(request.inputs[name], name, contract)
        for name, contract in sorted(PRODUCTION_DATASETS.items())
    }
    artifacts = {
        name: resolve_literal_artifact(request.inputs[name], name, digest)
        for name, digest in sorted(PRODUCTION_ARTIFACTS.items())
    }
    role = request.config.get("role")
    if role not in PRODUCTION_PROTECTED_ROLES or set(request.config) != {"role"}:
        raise ValueError("protected role or config surface is not governed")

    output_root = Path(os.environ["OMF_RESULT_FILE"]).parent
    directory = f"proteome-observation-{role}-v1"
    result = build_protected_observations(
        datasets["rawProteome"].path,
        datasets["interventionInventory"].path,
        datasets["proteinRelations"].path,
        datasets["heldRoster"].path,
        artifacts["sgdCurrentOrfs"].path,
        artifacts["sgdMappingManifest"].path,
        output_root / directory,
        role=role,
        role_contract=PRODUCTION_PROTECTED_ROLES[role],
        expected=PRODUCTION_EXPECTED_COUNTS,
        bounds=Bounds(),
        provenance=PreparationProvenance(datasets=datasets, artifacts=artifacts),
    )
    outputs = {
        "preparationSummary": result["audit"],
        **{key: result[key] for key in _outputs(role) if key != "preparationSummary"},
    }
    role_name = "MolecularValidation" if role == "molecular-validation" else "MolecularFinal"
    return ProtocolResult(
        status="ok",
        outputs=outputs,
        metrics={
            "protected_records": outputs["records"],
            "protected_intervention_genes": outputs["interventionGenes"],
            "protected_observed_values": outputs["targetValues"],
            "unselected_rows_not_decoded": outputs["unselectedRowsNotDecoded"],
        },
        artifacts=[
            {
                "name": f"proteome{role_name}Observation",
                "kind": "dataset",
                "path": f"{directory}/observation-corpus.tar",
            },
            {
                "name": f"proteome{role_name}PreparationAudit",
                "kind": "audit",
                "path": f"{directory}/preparation-audit.json",
            },
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main({"validate": validate, "run": run}))
