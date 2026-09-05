"""Superseding score of frozen predictions with stable query centering."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/slp11-transition/yeast-raw-count-batch-ridge-v1/report.json"
OUTPUT = ROOT / "results/slp11-transition/yeast-raw-count-batch-ridge-roundoff-scoring-v1"
RUNNER = ROOT / "scripts/run_slp11_yeast_batch_ridge.py"
SOURCE_SHA = "e15c9b14dc37b4eae01ef1e5bc847860a2d39273c76c930cb12030e622488824"


def main():
    spec = importlib.util.spec_from_file_location("yeast_batch_scoring", RUNNER)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    if runner.sha256(SOURCE) != SOURCE_SHA or OUTPUT.exists():
        raise ValueError("source drift or existing immutable output")
    report = json.loads(SOURCE.read_text())
    OUTPUT.mkdir()
    protocol = {
        "schema": "slp.yeast-batch-ridge-roundoff-scoring-protocol/v1",
        "source_report_sha256": SOURCE_SHA,
        "scorer_sha256": runner.sha256(Path(__file__)),
        "metrics_source_sha256": runner.sha256(RUNNER),
        "change": "Subtract first gene row before across-gene centering to eliminate common-profile reduction residue.",
        "fitting": False,
        "predictions_changed": False,
        "advancement_rule_changed": False,
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for entry in report["validation"].values():
        path = Path(entry["predictionPath"])
        if runner.sha256(path) != entry["predictionSha256"]:
            raise ValueError("frozen prediction drift")
        with np.load(path, allow_pickle=False) as archive:
            for arm in ("pooled", "batch", "pooledMean", "batchMean"):
                for view, suffix in (("raw", ""), ("batchMeanSubtracted", "_batch_mean_subtracted")):
                    new = runner.metrics(archive[f"truth{suffix}"], archive[f"prediction_{arm}{suffix}"])
                    old = entry["metrics"][arm][view]
                    if new["geneProfileMse"] != old["geneProfileMse"]:
                        raise ValueError("MSE changed during correlation-only correction")
                    if not (arm == "pooledMean" and view == "raw"):
                        a, b = old["independentlyQueryCenteredPearson"], new["independentlyQueryCenteredPearson"]
                        if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-12):
                            raise ValueError("unexpected meaningful correlation change")
                    entry["metrics"][arm][view] = new
        if entry["metrics"]["pooledMean"]["raw"]["independentlyQueryCenteredPearson"] is not None:
            raise ValueError("constant pooled mean retains false perturbation signal")
        # All decision inputs above are bit-identical MSE or correlation within
        # 1e-12; explicitly verify the only relative correlation comparison.
        batch_r = entry["metrics"]["batch"]["batchMeanSubtracted"]["independentlyQueryCenteredPearson"]
        pooled_r = entry["metrics"]["pooled"]["batchMeanSubtracted"]["independentlyQueryCenteredPearson"]
        if (batch_r >= pooled_r) != entry["gate"]["batchResidualCenteredRNonregressionVsPooled"]:
            raise ValueError("decision changed")
    report["scoringCorrection"] = protocol
    report["scoringCorrection"]["protocol_sha256"] = runner.sha256(OUTPUT / "protocol.json")
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"report_sha256": runner.sha256(OUTPUT / "report.json"), "passesAllContexts": report["passesAllContexts"]}))


if __name__ == "__main__":
    main()
