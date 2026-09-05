"""One frozen response-query-width experiment using the v2 numerical core."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/run_slp11_nonlinear_decoder.py"
SPEC = importlib.util.spec_from_file_location("response128_shared_scoring", HELPER)
SHARED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHARED)
MODEL = SHARED.COMPARATOR_V2.parent / "source/control_transition_model.py"
OUTPUT = ROOT / "results/slp11-transition/human-gwps-control-v2-response128-seed731-v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    SHARED.verify_inputs()
    if sha(MODEL) != "fdb4555bd0f7c0a0786539da67048f6985f4ec2f36ef7aa45bd22c7c6bfbb2ef":
        raise ValueError("frozen v2 numerical core changed")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(SHARED.LAUNCHER), "--data", str(SHARED.DATA),
               "--features", str(SHARED.FEATURES), "--feature-sha256", SHARED.EXPECTED[SHARED.FEATURES],
               "--hepg2-control", str(SHARED.HEPG2), "--original-report", str(SHARED.COMPARATOR_V2),
               "--model-source", str(MODEL), "--model-sha256", sha(MODEL),
               "--training-objective", "uniform-row-v1", "--output", str(OUTPUT / "model"),
               "--device", "cuda", "--epochs", "180", "--patience", "30", "--max-seconds", "1800",
               "--batch-size", "64", "--context-tokens", "64", "--query-basis-rank", "128",
               "--hidden", "128", "--state-dim", "128", "--dropout", "0.2",
               "--learning-rate", "0.0005", "--weight-decay", "0.1", "--ridge-alpha", "10000",
               "--seed", "731", "--cpu-threads", "2"]
    protocol = {
        "hypothesis": "The fitting-derived rank32 query descriptor limits molecular response prediction; rank128 improves all source contexts with unchanged global v2 core and features.",
        "motivation": "Previously frozen fitting-only response compression and matched context-local ridge diagnostics favored a richer response basis.",
        "advancement": "Existing .02 NLL gains over mean/ridge and adjusted r>=.10 in each source, no NLL/r regression against v2 response32, plus independently centered gene-profile r nonregression against full ridge in each source.",
        "accessible_modalities": "Human static sequence/GO/physical features; source fitting RNA endpoints and controls; development validation only for fixed early stopping; HepG2 control-only engineering input, no target outcomes.",
        "command": command,
        "inputs": {str(p.relative_to(ROOT)): h for p, h in SHARED.EXPECTED.items()},
        "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in
                          (MODEL, HELPER, SHARED.SCORING, Path(__file__))},
        "bound": "Same measured v2 topology, query encoder input expands by96; prior v2 training430s. Explicit CUDA only,1800-second training cap, no new executor.",
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    source = OUTPUT / "source"
    source.mkdir()
    for path in (MODEL, HELPER, SHARED.SCORING, Path(__file__)):
        shutil.copyfile(path, source / path.name)
    subprocess.run(command, cwd=ROOT, check=True)
    report = json.loads((OUTPUT / "model/report.json").read_text())
    centered = SHARED.independently_centered_metrics(OUTPUT / "model")
    result = {"report_sha256": sha(OUTPUT / "model/report.json"),
              "independently_centered_metrics": centered,
              "additional_gate_passed": all(v["worldNonregressionVsFullPhysicalRidge"] for v in centered.values()),
              "model_report": report}
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"summary_sha256": sha(OUTPUT / "summary.json"),
                      "additional_gate_passed": result["additional_gate_passed"]}))


if __name__ == "__main__":
    main()
