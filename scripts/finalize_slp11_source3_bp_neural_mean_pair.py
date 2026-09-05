#!/usr/bin/env python3
"""Finalize frozen source3 BP checkpoints after a stale verifier rejected them."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "results/slp11-transition/human-source3-bp-neural-mean-pair-seed731-v2"
OUTPUT = ROOT / "results/slp11-transition/human-source3-bp-neural-mean-pair-seed731-v2-finalization-v1"
RUNNER = ROOT / "scripts/run_slp11_source3_bp_neural_mean_pair.py"
ARMS = ("masked-bp-control", "bp128-present")
CONTEXTS = (
    "replogle-2022-k562-essential-day-6",
    "replogle-2022-rpe1-essential-day-7",
    "replogle-2022-k562-gwps-day-8",
)
PINS = {
    "pairProtocol": "7f0c33d5ee6bd465a8685fae45c189a69377aae5b089ca50ca48a06256638d52",
    "maskedModel": "690ed2d627aac7e17d81fdc35064aaa45bef065110377d26c00c218ba7ca6d14",
    "bpModel": "f1e0acf79c5326d4553ee77f45ccaa0d02628042672413d7089a17991e5d99fc",
    "runner": "fe007480adc485f71e6e8dfe0af8787bf68d5706b8ed02f2815912da9c0a6de8",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: object) -> None:
    def clean(item: object) -> object:
        if isinstance(item, dict):
            return {str(key): clean(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(entry) for entry in item]
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def select_probe_rows(split_train: np.ndarray, context_index: np.ndarray) -> np.ndarray:
    """Select two fitting rows in every context, failing closed on missing contexts."""
    selected = []
    for context in range(len(CONTEXTS)):
        rows = split_train[context_index[split_train] == context]
        if len(rows) < 2:
            raise ValueError(f"context {context} has fewer than two fitting rows")
        selected.extend(rows[:2].tolist())
    return np.asarray(selected, dtype=np.int64)


def verify_parent() -> dict[str, str]:
    if sha256(PARENT / "protocol.json") != PINS["pairProtocol"]:
        raise ValueError("parent protocol drift")
    if sha256(RUNNER) != PINS["runner"]:
        raise ValueError("executing runner differs from frozen training source")
    expected = {"masked-bp-control": PINS["maskedModel"], "bp128-present": PINS["bpModel"]}
    for arm, digest in expected.items():
        if sha256(PARENT / arm / "model.safetensors") != digest:
            raise ValueError(f"parent checkpoint drift: {arm}")
    marker = json.loads((PARENT / "FROZEN-BEFORE-VALIDATION.json").read_text())
    if marker["validationEvaluations"] != 0 or marker["models"] != expected:
        raise ValueError("checkpoint freeze marker drift")
    return expected


def prepare(output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    model_hashes = verify_parent()
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    for path in sorted((PARENT / "source").glob("*.py")):
        if path.name == "verify_artifact.py":
            continue
        shutil.copy2(path, source / path.name)
    shutil.copy2(Path(__file__), source / "finalizer.py")
    shutil.copy2(ROOT / "scripts/verify_slp11_source3_bp_neural_arm.py", source / "verify_artifact.py")
    for arm in ARMS:
        destination = output / arm
        destination.mkdir()
        shutil.copy2(PARENT / arm / "model.safetensors", destination / "model.safetensors")
        shutil.copy2(PARENT / arm / "reference.npz", destination / "reference.npz")
    protocol = {
        "schema": "slp.source3-bp-neural-mean-finalization-protocol/v1",
        "purpose": "Package and score two already-frozen source3 checkpoints after the inherited four-context verifier rejected a three-context artifact.",
        "parent": {
            "path": str(PARENT.relative_to(ROOT)),
            "protocolSha256": PINS["pairProtocol"],
            "freezeMarkerSha256": sha256(PARENT / "FROZEN-BEFORE-VALIDATION.json"),
            "models": model_hashes,
        },
        "correction": "Use exactly two fitting-only raw-action probes per each of the three source3 contexts; require fresh CPU mean replay <=1e-5 and exact empty identity in all three contexts.",
        "training": "No fitting, optimizer update, checkpoint selection, or validation-derived choice occurs in this finalization.",
        "decisionRule": "Unchanged parent rule: BP arm in each context must improve raw gene-profile MSE >=2% versus both matched masked-BP control and BP ridge, have finite independently query-centered r >=0.10, and not regress r versus either comparator.",
        "validationTiming": "One development-validation evaluation only after both checkpoints, this protocol, and both target-free packages are frozen and verified.",
        "sourceHashes": {path.name: sha256(path) for path in sorted(source.glob("*.py"))},
        "hepg2OutcomesUsed": False,
        "jurkatAccessed": False,
        "testAccessed": False,
        "benchmarkAccessed": False,
    }
    write_json(output / "protocol.json", protocol)
    prepared = {
        "protocolSha256": sha256(output / "protocol.json"),
        "modelSha256": model_hashes,
        "validationEvaluations": 0,
    }
    write_json(output / "PREPARED-BEFORE-VALIDATION.json", prepared)
    return prepared


def package_arm(arm: str, runner, data, actions, references, output: Path, device: torch.device) -> dict[str, object]:
    arm_path = output / arm
    model_module = load(output / "source/control_transition_model.py", f"bp_finalize_model_{arm}")
    model, _ = runner.initialize_extended(model_module)
    model.load_state_dict(load_file(arm_path / "model.safetensors"))
    model = model.to(device).eval()
    probe = select_probe_rows(data["split_train"], data["context_index"])
    expected = runner.predict(model, actions[arm][probe], data["context_index"][probe], references[arm], device)
    np.savez_compressed(
        arm_path / "target-free-probe.npz",
        raw_action_features=actions[arm][probe],
        context_index=data["context_index"][probe],
        expected_mean=expected,
    )
    source_hashes = {
        f"../source/{path.name}": sha256(path) for path in sorted((output / "source").glob("*.py"))
    }
    manifest = {
        "schema": "slp.source3-bp-neural-arm-artifact/v2",
        "sha256": {
            "model.safetensors": sha256(arm_path / "model.safetensors"),
            "reference.npz": sha256(arm_path / "reference.npz"),
            "target-free-probe.npz": sha256(arm_path / "target-free-probe.npz"),
            **source_hashes,
        },
    }
    write_json(arm_path / "artifact-manifest.json", manifest)
    verification = subprocess.run(
        [sys.executable, str(output / "source/verify_artifact.py"), str(arm_path)],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    return {"model": model, "verification": json.loads(verification.stdout)}


def execute(output: Path, device_name: str) -> dict[str, object]:
    prepared = json.loads((output / "PREPARED-BEFORE-VALIDATION.json").read_text())
    if sha256(output / "protocol.json") != prepared["protocolSha256"]:
        raise ValueError("finalization protocol drift")
    protocol = json.loads((output / "protocol.json").read_text())
    for name, digest in protocol["sourceHashes"].items():
        if sha256(output / "source" / name) != digest:
            raise ValueError(f"finalization source drift: {name}")
    verify_parent()
    runner = load(RUNNER, "bp_frozen_pair_finalizer_runner")
    data, actions, references, audit = runner.load_data()
    device = torch.device(device_name)
    packaged = {
        arm: package_arm(arm, runner, data, actions, references, output, device) for arm in ARMS
    }
    write_json(
        output / "VERIFIED-BEFORE-VALIDATION.json",
        {
            "protocolSha256": prepared["protocolSha256"],
            "portableReload": {arm: packaged[arm]["verification"] for arm in ARMS},
            "validationEvaluations": 0,
        },
    )
    # This is the sole quantitative development-validation access in this finalization.
    metrics = load(output / "source/four_context_baselines.py", "bp_finalize_metrics")
    reports = {}
    for arm in ARMS:
        arm_metrics = runner.score_arm(
            arm,
            packaged[arm]["model"],
            data,
            actions[arm],
            references[arm],
            output,
            device,
            metrics,
        )
        reports[arm] = {
            "metrics": arm_metrics,
            "portableReload": packaged[arm]["verification"],
            "artifacts": {
                name: sha256(output / arm / name)
                for name in ("model.safetensors", "reference.npz", "predictions.npz", "artifact-manifest.json")
            },
        }
        write_json(output / arm / "report.json", reports[arm])
    decision = runner.decide(reports)
    result = {
        "schema": "slp.source3-bp-neural-mean-pair-finalized-result/v1",
        "arms": reports,
        "decision": decision,
        "inputAudit": audit,
        "parentTrainingSeconds": {"masked-bp-control": 125.437, "bp128-present": 126.969},
        "protocolSha256": prepared["protocolSha256"],
        "validationEvaluations": 1,
        "checkpointUpdatesAfterFreeze": 0,
        "hepg2OutcomesUsed": False,
        "jurkatAccessed": False,
        "testAccessed": False,
        "benchmarkAccessed": False,
    }
    write_json(output / "report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = prepare(args.output) if args.prepare else execute(args.output, args.device)
    print(json.dumps(result, sort_keys=True, default=lambda value: value.item()))


if __name__ == "__main__":
    main()
