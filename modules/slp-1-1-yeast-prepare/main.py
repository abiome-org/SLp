from __future__ import annotations

import json
import os
from pathlib import Path

from omf.sdk import ProtocolRequest, ProtocolResult, main


def _validation_outputs() -> dict[str, object]:
    return {
        "preparation": {"schema": "slp.yeast-preparation/v1", "validationOnly": True},
        "contentSha256": "0" * 64,
        "archiveSha256": "0" * 64,
        "records": 0,
        "shards": 0,
        "trajectoryGenes": 0,
        "ncbiTaxon": 4932,
        "trainingAllowed": 1,
        "redistributionAllowed": 0,
    }


def validate(_request: ProtocolRequest) -> ProtocolResult:
    return ProtocolResult(status="ok", outputs=_validation_outputs())


def run(request: ProtocolRequest) -> ProtocolResult:
    source = request.inputs.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("path"), str):
        raise ValueError("source must be an OMF dataset input with a materialized path")
    from prepare import prepare_yeast_snapshot, write_deterministic_tar

    output_root = Path(os.environ["OMF_RESULT_FILE"]).parent
    corpus_path = output_root / "prepared-corpus"
    report = prepare_yeast_snapshot(source["path"], corpus_path, request.config)
    archive_path = output_root / "prepared-corpus.tar"
    archive_digest = write_deterministic_tar(corpus_path, archive_path)
    report_path = output_root / "preparation-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    redistribution = report["rights"]["redistributionAllowed"] is True
    outputs = {
        "preparation": report,
        "contentSha256": report["contentSha256"],
        "archiveSha256": archive_digest,
        "records": report["records"],
        "shards": len(report["generatedFiles"]) - 2,
        "trajectoryGenes": report["trajectoryGenes"],
        "ncbiTaxon": 4932,
        "trainingAllowed": 1,
        "redistributionAllowed": int(redistribution),
    }
    return ProtocolResult(
        status="ok",
        outputs=outputs,
        metrics={
            "records": outputs["records"],
            "shards": outputs["shards"],
            "trajectory_genes": outputs["trajectoryGenes"],
        },
        artifacts=[
            {"name": "preparedCorpus", "kind": "dataset", "path": archive_path.name},
            {"name": "preparationReport", "kind": "audit", "path": report_path.name},
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main({"validate": validate, "run": run}))
