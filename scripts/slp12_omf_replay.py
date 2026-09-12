"""Run zero-optimization artifact replay/export through the pinned Linux OpenFoundry.

Run on a Linux host with OpenFoundry's real network-namespace capability. The RunPod
training container lacks that capability; this campaign uses the existing local
Linux VM after collecting the bundle. No GPU or training corpus is required.
"""

import argparse
import json
from pathlib import Path
import platform
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    if platform.system() != "Linux":
        raise RuntimeError("Use the pinned OpenFoundry runtime on Linux")
    bundle = args.bundle.resolve().relative_to(ROOT)
    if not (ROOT / bundle / "manifest.json").is_file():
        raise FileNotFoundError("A complete standalone bundle is required")
    args.output.mkdir(parents=True, exist_ok=False)
    lock = (
        "requirements-linux-arm64-cpu.lock"
        if platform.machine() == "aarch64"
        else "requirements-linux-cpu.lock"
    )
    definition = (
        (ROOT / "experiment-slp12-replay.yaml")
        .read_text()
        .replace("results/slp12-joint-142m-r1-bundle", bundle.as_posix())
        .replace("requirements-linux-cpu.lock", lock)
    )
    (args.output / "definition.yaml").write_text(definition)
    command = ["bash", str(ROOT / "scripts/openfoundry.sh"), "--output", "json"]

    def invoke(arguments, receipt):
        result = subprocess.run(
            command + arguments, cwd=ROOT, capture_output=True, text=True, timeout=1200
        )
        (args.output / (receipt + ".json")).write_text(result.stdout)
        (args.output / (receipt + ".stderr")).write_text(result.stderr)
        result.check_returncode()
        return json.loads(result.stdout)

    # OpenFoundry resolves all source paths relative to its definition directory and
    # requires repository-relative paths. Capture a temporary root definition.
    with tempfile.NamedTemporaryFile(
        "w", prefix="experiment-slp12-", suffix=".generated.yaml", dir=ROOT
    ) as f:
        f.write(definition)
        f.flush()
        result = invoke(
            ["experiment", "run", Path(f.name).name, "--candidate", "joint-world"], "run"
        )
    if result.get("state") != "succeeded":
        raise RuntimeError("OpenFoundry replay did not succeed; inspect the saved run receipt")
    exported = invoke(
        ["experiment", "export", result["runId"], "--to", str(args.output / "export")], "export"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "run_id": result["runId"],
                "executor": "local",
                "architecture": platform.machine(),
                "optimization_steps": 0,
                "export": exported,
                "scores": result.get("scores"),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
