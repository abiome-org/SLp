"""OMF entry point for leakage-safe proteome pretraining observations."""

from __future__ import annotations

import os
from pathlib import Path

from omf.sdk import ProtocolRequest, ProtocolResult, main


def _validation_outputs() -> dict[str, object]:
    return {
        "preparationSummary": {
            "schema": "slp.proteome-observation-preparation-audit/v1",
            "role": "pretrain",
            "validationOnly": True,
        },
        "observationArchiveSha256": "0" * 64,
        "basalArchiveSha256": "0" * 64,
        "auditSha256": "0" * 64,
        "records": 0,
        "interventionGenes": 0,
        "targetValues": 0,
        "missingValues": 0,
        "basalControls": 0,
        "basalObservedValues": 0,
        "basalSupportedReadouts": 0,
        "excludedValidationRows": 0,
        "excludedFinalRows": 0,
        "excludedQuarantineRows": 0,
        "excludedQcRows": 0,
    }


def validate(_request: ProtocolRequest) -> ProtocolResult:
    return ProtocolResult(status="ok", outputs=_validation_outputs())


def run(request: ProtocolRequest) -> ProtocolResult:
    from observation_prepare import (
        Bounds,
        ExpectedCounts,
        PRODUCTION_ARTIFACTS,
        PRODUCTION_DATASETS,
        PRODUCTION_EXPECTED_COUNTS,
        PreparationProvenance,
        build_pretrain_observations,
        resolve_literal_artifact,
        resolve_pinned_dataset_input,
    )

    expected_inputs = {*PRODUCTION_DATASETS, *PRODUCTION_ARTIFACTS}
    if set(request.inputs) != expected_inputs:
        raise ValueError("proteome observation preparation requires exactly four datasets and two SGD artifacts")
    datasets = {
        name: resolve_pinned_dataset_input(request.inputs[name], name, contract)
        for name, contract in sorted(PRODUCTION_DATASETS.items())
    }
    artifacts = {
        name: resolve_literal_artifact(request.inputs[name], name, digest)
        for name, digest in sorted(PRODUCTION_ARTIFACTS.items())
    }
    config = request.config
    if config["role"] != "pretrain":
        raise ValueError("v1 prepares only the pretraining role")
    expected = ExpectedCounts(
        metadata_rows=config["expectedMetadataRows"],
        eligible_rows=config["expectedEligibleRows"],
        eligible_genes=config["expectedEligibleGenes"],
        pretrain_records=config["expectedPretrainRecords"],
        pretrain_genes=config["expectedPretrainGenes"],
        target_values=config["expectedTargetValues"],
        missing_values=config["expectedMissingValues"],
        trajectory_genes_sha256=config["expectedTrajectoryGenesSha256"],
        trajectory_gene_set_sha256=config["expectedTrajectoryGeneSetSha256"],
        protein_readouts=config["expectedProteinReadouts"],
        basal_controls=config["expectedBasalControls"],
        basal_observed_values=config["expectedBasalObservedValues"],
        basal_supported_readouts=config["expectedBasalSupportedReadouts"],
        validation_genes=config["expectedValidationGenes"],
        validation_rows=config["expectedValidationRows"],
        final_genes=config["expectedFinalGenes"],
        final_rows=config["expectedFinalRows"],
        quarantine_rows=config["expectedQuarantineRows"],
        qc_rows=config["expectedQcRows"],
    )
    if expected != PRODUCTION_EXPECTED_COUNTS:
        raise ValueError("frozen production counts or digests were loosened")
    output_root = Path(os.environ["OMF_RESULT_FILE"]).parent
    result = build_pretrain_observations(
        datasets["rawProteome"].path,
        datasets["interventionInventory"].path,
        datasets["proteinRelations"].path,
        datasets["heldRoster"].path,
        artifacts["sgdCurrentOrfs"].path,
        artifacts["sgdMappingManifest"].path,
        output_root / "proteome-observation-pretrain-v1",
        expected=expected,
        bounds=Bounds(),
        provenance=PreparationProvenance(datasets=datasets, artifacts=artifacts),
    )
    outputs = {
        "preparationSummary": result["audit"],
        **{key: result[key] for key in _validation_outputs() if key != "preparationSummary"},
    }
    return ProtocolResult(
        status="ok",
        outputs=outputs,
        metrics={
            "pretrain_records": outputs["records"],
            "pretrain_intervention_genes": outputs["interventionGenes"],
            "pretrain_observed_values": outputs["targetValues"],
            "protected_rows_not_decoded": outputs["excludedValidationRows"] + outputs["excludedFinalRows"],
            "basal_supported_readouts": outputs["basalSupportedReadouts"],
        },
        artifacts=[
            {
                "name": "proteomeObservationCorpus",
                "kind": "dataset",
                "path": "proteome-observation-pretrain-v1/observation-corpus.tar",
            },
            {
                "name": "proteomeBasalControl",
                "kind": "dataset",
                "path": "proteome-observation-pretrain-v1/basal-control.tar",
            },
            {
                "name": "proteomePreparationAudit",
                "kind": "audit",
                "path": "proteome-observation-pretrain-v1/preparation-audit.json",
            },
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main({"validate": validate, "run": run}))
