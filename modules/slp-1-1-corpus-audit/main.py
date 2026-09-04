from __future__ import annotations

import json
import os
from pathlib import Path

from omf.sdk import ProtocolRequest, ProtocolResult, main

from audit import audit_corpora


def _outputs(audit: dict[str, object]) -> dict[str, object]:
    records = audit.get("records", {})
    species = audit.get("speciesTaxa", [])
    return {
        "audit": audit,
        "auditPassed": int(bool(audit.get("auditPassed"))),
        "leakageViolations": int(audit.get("leakageViolations", 0)),
        "benchmarkLabelRecords": int(audit.get("benchmarkLabelRecords", 0)),
        "pretrainRecords": int(records.get("pretrain", 0)),
        "validationRecords": int(records.get("molecularValidation", 0)),
        "rewardRecords": int(records.get("molecularReward", 0)),
        "speciesCount": len(species),
    }


def validate(_request: ProtocolRequest) -> ProtocolResult:
    return ProtocolResult(
        status="ok",
        outputs=_outputs(
            {
                "schema": "slp.corpus-audit/v1",
                "validationOnly": True,
                "auditPassed": True,
            }
        ),
    )


def run(request: ProtocolRequest) -> ProtocolResult:
    paths = {
        name: request.inputs[name]["path"]
        for name in ("pretrain", "molecularValidation", "molecularReward")
    }
    audit = audit_corpora(
        paths,
        strict=bool(request.config.get("strictInterventionIsolation", True)),
    )
    destination = Path(os.environ["OMF_RESULT_FILE"]).parent / "corpus-audit.json"
    destination.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ProtocolResult(
        status="ok",
        outputs=_outputs(audit),
        metrics={
            "leakage_violations": audit["leakageViolations"],
            "benchmark_label_records": audit["benchmarkLabelRecords"],
        },
        artifacts=[{"name": "auditReport", "kind": "audit", "path": destination.name}],
    )


if __name__ == "__main__":
    raise SystemExit(main({"validate": validate, "run": run}))
