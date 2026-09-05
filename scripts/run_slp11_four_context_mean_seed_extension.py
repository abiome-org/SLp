#!/usr/bin/env python3
"""Run seeds 732/733 and a fixed three-seed ensemble of the mean-objective pair."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PAIR_SCRIPT = ROOT / "scripts/run_slp11_four_context_mean_objective_pair.py"
PAIR731 = ROOT / (
    "results/slp11-transition/"
    "human-source3-vs-four-context-mean-objective-seed731-v2"
)
OUTPUT = ROOT / (
    "results/slp11-transition/"
    "human-source3-vs-four-context-mean-objective-ensemble731-733-v1"
)
KERNEL = ROOT / (
    "results/slp11-transition/human-gwps-nystrom-rbf512-physical-seed731-v1/report.json"
)
ORIGINAL_V2 = ROOT / (
    "results/slp11-transition/"
    "human-gwps-fixed-context-minimal-control-physical-state128-response32-seed731-v1/"
    "model/report.json"
)
BP_KERNEL = ROOT / (
    "results/slp11-transition/human-gwps-bp-nystrom-rbf512-seed731-v1/report.json"
)
NEW_SEEDS = (732, 733)
ARMS = {"source3": 3, "source4": 4}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    def clean(item: object) -> object:
        if isinstance(item, dict):
            return {str(key): clean(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(entry) for entry in item]
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    path.write_text(
        json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_pair_module(path: Path = PAIR_SCRIPT):
    spec = importlib.util.spec_from_file_location("slp11_mean_pair_extension", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load fixed mean-pair trainer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def arithmetic_ensemble(members: list[np.ndarray]) -> np.ndarray:
    """Average aligned member means with float64 accumulation and float32 storage."""

    if len(members) != 3:
        raise ValueError("the frozen ensemble requires exactly three members")
    arrays = [np.asarray(member) for member in members]
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("ensemble member prediction axes do not align")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("ensemble members must be finite")
    return np.mean(np.stack(arrays).astype(np.float64), axis=0).astype(np.float32)


def source_paths(pair) -> dict[str, Path]:
    return {
        "run_pair.py": PAIR_SCRIPT,
        "control_transition_model.py": pair.MODEL,
        "objective_weighting.py": pair.OBJECTIVE,
        "mean_objective.py": pair.WORLD / "mean_objective.py",
        "four_context_baselines.py": pair.WORLD / "four_context_baselines.py",
        "four_context_mean_inference.py": pair.WORLD / "four_context_mean_inference.py",
        "verify_artifact.py": ROOT / "scripts/verify_slp11_four_context_mean_artifact.py",
        "run_seed_extension.py": Path(__file__),
    }


def prepare(output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    pair = load_pair_module()
    data, reference, _ = pair.load_inputs()
    audit = pair.frozen_reference_audit(data, reference)
    audit.pop("contextValues")
    prior_hashes = {
        "protocol": sha256(PAIR731 / "protocol.json"),
        "report": sha256(PAIR731 / "report.json"),
        "reference": sha256(PAIR731 / "frozen-reference.npz"),
        **{
            f"{arm}_{artifact}": sha256(PAIR731 / f"arm-{arm}" / filename)
            for arm in ARMS
            for artifact, filename in {
                "report": "report.json",
                "model": "model.safetensors",
                "predictions": "predictions.npz",
                "manifest": "artifact-manifest.json",
            }.items()
        },
    }
    expected_prior = {
        "protocol": "9f2858397c8589d0c3149b968adddfb6e9deae1204622d2938057705bfdfb580",
        "report": "34978aac3f366deccd927c3bda11cda1c4e1107ea388ed432d848cf71b02e010",
        "reference": "54cac4bc2e2ee02a6d78f812d5646cf3988154d5ae4f371265b24751f03c99b1",
        "source3_report": "ac5e05603365f9eb2c8888cc8344a99d68c5d209bbfc7fac6df0d073ea75508f",
        "source3_model": "74b9e9c7e59a544a01ccc814cc2812240e5db6b220cb230a7ed0faf5d5ae753e",
        "source3_predictions": "5cebfc97112088c019754b9937763b035fe3b1f903d4799340873d986e5459db",
        "source3_manifest": "b6562ac04a9a9cad76ae8204254f3ce806b9c37b74ad795781a5337718478435",
        "source4_report": "4f1cc1fa020cc1da281bee670897bc5f098580441700c7e1127acd5d3f1fe59f",
        "source4_model": "e20706b8e357042e5910cd523f16d6136ecb748a18d3c5c4968cc781c6a1fdcf",
        "source4_predictions": "d52d577012ebea0f1090078b5cc291c715c0c3514e12cd6e8833a9bd18fa0c0d",
        "source4_manifest": "8f700553c698ec615dd4ee049ac4bb04ae4062c546089c6624c85f359a92d61e",
    }
    if prior_hashes != expected_prior:
        raise ValueError("seed731 pair drifted")

    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    source_hashes = {}
    workspace_sources = {}
    for name, path in source_paths(pair).items():
        shutil.copy2(path, source / name)
        source_hashes[name] = sha256(source / name)
        workspace_sources[str(path.resolve())] = source_hashes[name]
    shutil.copy2(PAIR731 / "frozen-reference.npz", output / "frozen-reference.npz")
    frontier = {
        "originalV2Report": {"path": str(ORIGINAL_V2), "sha256": sha256(ORIGINAL_V2)},
        "physicalKernelReport": {"path": str(KERNEL), "sha256": sha256(KERNEL)},
        "bpPhysicalKernelReport": {"path": str(BP_KERNEL), "sha256": sha256(BP_KERNEL)},
    }
    protocol = {
        "schema": "slp.source3-source4-mean-seed-extension-protocol/v1",
        "hypothesis": (
            "the adaptive HepG2 gain survives seed variation and equal-weight averaging "
            "improves point accuracy without source3 regression"
        ),
        "newSeeds": list(NEW_SEEDS),
        "ensembleSeeds": [731, 732, 733],
        "ensemble": "arithmetic mean of all three aligned point predictions; no member selection",
        "training": {
            "contract": "exact seed731 mean-objective protocol; seed is the only changed argument",
            "steps": 12_000,
            "batchSize": 64,
            "queries": 7_036,
            "finalCheckpointOnly": True,
            "perArmHardCapSeconds": 1200,
        },
        "fixedRule": json.loads((PAIR731 / "protocol.json").read_text())["fixedRule"],
        "seed731": {
            "path": str(PAIR731),
            "hashes": prior_hashes,
            "retrained": False,
        },
        "inputs": json.loads((PAIR731 / "protocol.json").read_text())["inputs"],
        "frozenReference": {
            "path": "frozen-reference.npz",
            "sha256": sha256(output / "frozen-reference.npz"),
        },
        "sourceHashes": source_hashes,
        "workspaceSourceHashesRequiredAtLaunch": workspace_sources,
        "frontierComparators": frontier,
        "inputAudit": audit,
        "claimLimit": (
            "adaptive development seed stability only; ridge is the fixed gate, while "
            "original v2, physical kernel and BP kernel describe the broader source3 "
            "baseline frontier and prevent a world-winner claim when stronger baselines remain"
        ),
        "jurkatAccessed": False,
        "testAccessed": False,
        "benchmarkAccessed": False,
        "uncertaintyClaim": False,
    }
    write_json(output / "protocol.json", protocol)
    arm_hashes = {}
    train = data["split_train"]
    for seed in NEW_SEEDS:
        for arm, count in ARMS.items():
            name = f"seed{seed}-{arm}"
            arm_path = output / f"arm-{name}"
            arm_path.mkdir()
            arm_protocol = {
                "schema": "slp.four-context-mean-seed-extension-arm/v1",
                "seed": seed,
                "arm": arm,
                "fittingContexts": list(pair.CONTEXTS[:count]),
                "fittingRows": int(np.sum(data["context_index"][train] < count)),
                "optimizerSteps": 12_000,
                "batchSize": 64,
                "parentProtocol": "../protocol.json",
                "validationEvaluations": 1,
                "testAccessed": False,
            }
            write_json(arm_path / "protocol.json", arm_protocol)
            arm_hashes[name] = sha256(arm_path / "protocol.json")
    prepared = {
        "schema": "slp.source3-source4-mean-seed-extension-prepared/v1",
        "protocolSha256": sha256(output / "protocol.json"),
        "armProtocolSha256": arm_hashes,
        "sourceHashes": source_hashes,
        "workspaceSourceHashesRequiredAtLaunch": workspace_sources,
        "frozenReferenceSha256": sha256(output / "frozen-reference.npz"),
    }
    write_json(output / "PREPARED.json", prepared)
    return prepared


def validate_prepared(output: Path) -> tuple[dict[str, object], object]:
    prepared = json.loads((output / "PREPARED.json").read_text())
    if sha256(output / "protocol.json") != prepared["protocolSha256"]:
        raise ValueError("extension protocol changed after freezing")
    for name, expected in prepared["armProtocolSha256"].items():
        if sha256(output / f"arm-{name}/protocol.json") != expected:
            raise ValueError(f"arm protocol drift: {name}")
    for name, expected in prepared["sourceHashes"].items():
        if sha256(output / "source" / name) != expected:
            raise ValueError(f"snapshotted source drift: {name}")
    for name, expected in prepared["workspaceSourceHashesRequiredAtLaunch"].items():
        if sha256(name) != expected:
            raise ValueError(f"effective imported source differs from snapshot: {name}")
    if sha256(output / "frozen-reference.npz") != prepared["frozenReferenceSha256"]:
        raise ValueError("frozen reference drift")
    return prepared, load_pair_module()


def load_arm_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path / "predictions.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def score_ensembles(output: Path, pair, arm_reports: dict[str, dict]) -> dict[str, object]:
    member_roots = {
        731: {
            arm: PAIR731 / f"arm-{arm}" for arm in ARMS
        },
        **{
            seed: {
                arm: output / f"arm-seed{seed}-{arm}" for arm in ARMS
            }
            for seed in NEW_SEEDS
        },
    }
    members = {
        seed: {arm: load_arm_predictions(root) for arm, root in arms.items()}
        for seed, arms in member_roots.items()
    }
    with np.load(pair.BASELINE_PREDICTIONS, allow_pickle=False) as baseline:
        baseline_values = {name: baseline[name] for name in baseline.files}
    ensemble_arrays = {}
    ensemble_metrics = {arm: {} for arm in ARMS}
    alignment = {}
    for arm in ARMS:
        for index, context in enumerate(pair.CONTEXTS):
            key_ids = f"context{index}_action_ids"
            ids = [members[seed][arm][key_ids] for seed in (731, 732, 733)]
            if not all(np.array_equal(ids[0], item) for item in ids[1:]):
                raise ValueError(f"member gene alignment drift: {arm}/{context}")
            if not np.array_equal(ids[0], baseline_values[key_ids]):
                raise ValueError(f"baseline gene alignment drift: {arm}/{context}")
            forecasts = [
                members[seed][arm][f"context{index}_world"]
                for seed in (731, 732, 733)
            ]
            mean = arithmetic_ensemble(forecasts)
            ensemble_arrays[f"{arm}_context{index}_action_ids"] = ids[0]
            ensemble_arrays[f"{arm}_context{index}_world"] = mean
            ensemble_metrics[arm][context] = pair.point_metrics(
                mean,
                baseline_values[f"context{index}_truth"],
                baseline_values[f"context{index}_observed"],
                baseline_values[f"context{index}_fitting_query_scale"],
                baseline_values[f"context{index}_fitting_target_centroid"],
            )
            alignment[f"{arm}/{context}"] = {
                "threeMemberGeneIdsExact": True,
                "baselineGeneIdsExact": True,
                "queries": 7036,
                "genes": len(ids[0]),
            }
    np.savez_compressed(output / "ensemble-predictions.npz", **ensemble_arrays)
    reports_for_decision = {
        arm: {"validationMetrics": ensemble_metrics[arm]} for arm in ARMS
    }
    baseline_report = json.loads(pair.BASELINE_REPORT.read_text())
    ensemble_decision = pair.decide(reports_for_decision, baseline_report)

    seed_decisions = {
        "731": json.loads((PAIR731 / "report.json").read_text())["decision"]
    }
    for seed in NEW_SEEDS:
        pair_reports = {
            arm: arm_reports[f"seed{seed}-{arm}"] for arm in ARMS
        }
        seed_decisions[str(seed)] = pair.decide(pair_reports, baseline_report)

    kernel = json.loads(KERNEL.read_text())
    bp_kernel = json.loads(BP_KERNEL.read_text())
    frontier = {}
    for context in pair.CONTEXTS[:3]:
        frontier[context] = {
            "ridge": baseline_report["contexts"][context]["ridge"],
            "originalMinimalControlV2Descriptive": kernel["contexts"][context]["scores"]["minimalControlV2"],
            "physicalNystromDescriptive": kernel["contexts"][context]["scores"]["nystromRbf"],
            "bpPhysicalNystromDescriptive": bp_kernel["contexts"][context]["candidate"],
            "source3MeanObjectiveEnsemble": ensemble_metrics["source3"][context],
        }
    return {
        "ensembleMetrics": ensemble_metrics,
        "ensembleDecision": ensemble_decision,
        "perSeedDecisions": seed_decisions,
        "alignment": alignment,
        "frontier": frontier,
        "ensemblePredictions": {
            "path": "ensemble-predictions.npz",
            "sha256": sha256(output / "ensemble-predictions.npz"),
        },
    }


def execute(output: Path, device_name: str) -> dict[str, object]:
    prepared, pair = validate_prepared(output)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no fallback")
    data, frozen, actions = pair.load_inputs()
    audit = pair.frozen_reference_audit(data, frozen)
    train = data["split_train"]
    scale = pair.context_query_sd(
        data["targets"], data["observed"], data["context_index"], train, 4, floor=0.05
    )
    reference = pair.build_reference(data, frozen, scale, audit.pop("contextValues"))
    with np.load(output / "frozen-reference.npz", allow_pickle=False) as saved:
        if saved.files != list(reference) or any(
            not np.array_equal(saved[name], reference[name]) for name in saved.files
        ):
            raise ValueError("effective frozen reference differs from prepared payload")
    reports = {}
    device = torch.device(device_name)
    for seed in NEW_SEEDS:
        for arm, count in ARMS.items():
            name = f"seed{seed}-{arm}"
            reports[name] = pair.fit_arm(
                name,
                count,
                data,
                actions,
                reference,
                device,
                output,
                seed=seed,
            )
            if not reports[name]["complete"]:
                result = {"complete": False, "failedArm": name, "reports": reports}
                write_json(output / "report.json", result)
                return result
    scoring = score_ensembles(output, pair, reports)
    result = {
        "schema": "slp.source3-source4-mean-seed-extension-result/v1",
        "complete": True,
        **scoring,
        "newArms": {
            name: {
                "report": f"arm-{name}/report.json",
                "reportSha256": sha256(output / f"arm-{name}/report.json"),
                "modelSha256": reports[name]["artifacts"]["model"]["sha256"],
                "elapsedSeconds": reports[name]["elapsedSeconds"],
            }
            for name in reports
        },
        "protocol": {"path": "protocol.json", "sha256": prepared["protocolSha256"]},
        "claimLimit": (
            "three-seed adaptive development point means; no uncertainty, SOTA, "
            "new-context, confirmation or release claim"
        ),
        "jurkatAccessed": False,
        "testAccessed": False,
        "benchmarkAccessed": False,
    }
    write_json(output / "report.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = prepare(args.output.resolve()) if args.prepare_only else execute(args.output.resolve(strict=True), args.device)
    print(json.dumps(result, sort_keys=True, default=lambda value: value.item()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
