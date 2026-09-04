from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from omf.sdk import ProtocolRequest, ProtocolResult, main


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validation_outputs() -> dict[str, object]:
    return {
        "modelState": {"schema": "slp.world/v1.1", "validationOnly": True},
        "checkpointSha256": "0" * 64,
        "validationNll": 0.0,
        "validationBaselineNll": 0.0,
        "validationNllDelta": 0.0,
        "validationNllImprovement": 0.0,
        "validationEffectPearson": 0.0,
        "minimumSpeciesNllDelta": 0.0,
        "minimumSpeciesNllImprovement": 0.0,
        "rlRetained": 0,
        "parameterCount": 0,
    }


def validate(_request: ProtocolRequest) -> ProtocolResult:
    return ProtocolResult(status="ok", outputs=_validation_outputs())


def run(request: ProtocolRequest) -> ProtocolResult:
    audit = request.inputs.get("corpusAudit")
    roots = {
        name: request.inputs[name]["path"]
        for name in ("pretrain", "molecularValidation", "molecularReward")
    }
    from trainer import train_world

    import torch

    model, report, baselines = train_world(roots, request.config, audit)
    output_dir = Path(os.environ["OMF_RESULT_FILE"]).parent
    checkpoint = output_dir / "slp-1-1-world.pt"
    torch.save(
        {
            "format": "slp.world/v1.1",
            "config": report["modelConfig"],
            "corpora": report["corpora"],
            "corpusAudit": audit,
            "readoutTypes": report["readoutTypes"],
            "baselineMean": baselines.mean,
            "baselineLogScale": baselines.mean_log_scale,
            "linearBaselineWeight": baselines.linear_weight,
            "linearBaselineLogScale": baselines.linear_log_scale,
            "stateDict": model.state_dict(),
        },
        checkpoint,
    )
    digest = _sha256(checkpoint)
    report_path = output_dir / "training-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected = report["selected"]
    model_state = {
        "schema": "slp.world/v1.1",
        "checkpointSha256": digest,
        "modelConfig": report["modelConfig"],
        "readoutTypes": report["readoutTypes"],
        "corpora": report["corpora"],
    }
    outputs = {
        "modelState": model_state,
        "checkpointSha256": digest,
        "validationNll": selected["nll"],
        "validationBaselineNll": selected["baselineNll"],
        "validationNllDelta": selected["nllDelta"],
        "validationNllImprovement": selected["nllImprovement"],
        "validationEffectPearson": selected["effectPearson"],
        "minimumSpeciesNllDelta": selected["minimumSpeciesNllDelta"],
        "minimumSpeciesNllImprovement": selected["minimumSpeciesNllImprovement"],
        "rlRetained": int(report["reinforcementRetained"]),
        "parameterCount": report["parameterCount"],
    }
    return ProtocolResult(
        status="ok",
        outputs=outputs,
        state=model_state,
        metrics={
            "validation_nll": selected["nll"],
            "validation_nll_delta": selected["nllDelta"],
            "validation_nll_improvement": selected["nllImprovement"],
            "validation_effect_pearson": selected["effectPearson"],
        },
        artifacts=[
            {"name": "checkpoint", "kind": "checkpoint", "path": checkpoint.name},
            {"name": "trainingReport", "kind": "evaluation", "path": report_path.name},
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main({"validate": validate, "run": run}))
