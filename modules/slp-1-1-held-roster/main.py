from __future__ import annotations

import json
import os
from pathlib import Path

from omf.sdk import ProtocolRequest, ProtocolResult, main


def _validation_outputs() -> dict[str, object]:
    return {
        "rosterSummary": {
            "schema": "slp.held-intervention-roster-report/v1",
            "validationOnly": True,
        },
        "rosterSha256": "0" * 64,
        "coverageSha256": "0" * 64,
        "sourceCount": 0,
        "intersectionSize": 0,
        "pretrainCount": 0,
        "validationCount": 0,
        "finalCount": 0,
    }


def validate(_request: ProtocolRequest) -> ProtocolResult:
    return ProtocolResult(status="ok", outputs=_validation_outputs())


def run(request: ProtocolRequest) -> ProtocolResult:
    from roster import RosterBounds, build_held_roster, resolve_pinned_dataset_input

    if len(request.inputs) < 2:
        raise ValueError("at least two protected inventory dataset inputs are required")
    resolved_inputs = [
        resolve_pinned_dataset_input(value, name)
        for name, value in sorted(request.inputs.items())
    ]
    paths = [item.path for item in resolved_inputs]

    config = request.config
    bounds = RosterBounds(
        minimum_intersection_size=config["minimumIntersectionSize"],
        max_sources=config.get("maxSources", 16),
        max_files_per_source=config.get("maxFilesPerSource", 32),
        max_records_per_source=config.get("maxRecordsPerSource", 200_000),
        max_line_bytes=config.get("maxLineBytes", 4_096),
        expected_intersection_size=config["expectedIntersectionSize"],
        expected_pretrain_count=config["expectedPretrainCount"],
        expected_validation_count=config["expectedValidationCount"],
        expected_final_count=config["expectedFinalCount"],
        expected_roster_sha256=config["expectedRosterSha256"],
    )
    output_root = Path(os.environ["OMF_RESULT_FILE"]).parent
    roster_root = output_root / "held-roster"
    report = build_held_roster(paths, roster_root, bounds)
    role_counts = report["roleCounts"]
    outputs = {
        "rosterSummary": report,
        "rosterSha256": report["rosterSha256"],
        "coverageSha256": report["coverageSha256"],
        "sourceCount": report["sourceCount"],
        "intersectionSize": report["intersectionSize"],
        "pretrainCount": role_counts["pretrain"],
        "validationCount": role_counts["molecular-validation"],
        "finalCount": role_counts["molecular-final"],
    }
    summary_path = output_root / "held-roster-summary.json"
    summary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return ProtocolResult(
        status="ok",
        outputs=outputs,
        metrics={
            "protected_sources": outputs["sourceCount"],
            "intersection_size": outputs["intersectionSize"],
            "validation_interventions": outputs["validationCount"],
            "final_interventions": outputs["finalCount"],
        },
        artifacts=[
            {
                "name": "heldInterventionRoster",
                "kind": "dataset",
                "path": "held-roster/held-intervention-roster.tsv",
            },
            {
                "name": "heldRosterCoverage",
                "kind": "audit",
                "path": "held-roster/coverage.json",
            },
            {
                "name": "heldRosterSummary",
                "kind": "audit",
                "path": summary_path.name,
            },
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main({"validate": validate, "run": run}))
