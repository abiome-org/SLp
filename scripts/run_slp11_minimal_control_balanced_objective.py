"""Run the frozen equal-context/equal-gene objective pilot on minimal-v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_slp11_minimal_control_common_context.py"
OBJECTIVE = ROOT / "modules/slp-1-1-control-transition-v2/objective_weighting.py"
MODEL = ROOT / "modules/slp-1-1-control-transition-v2/transition_model.py"
DATA_SHA256 = "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c"
FEATURE_SHA256 = "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7"
HEPG2_CONTROL_SHA256 = "382626401ee38e8d5084ac9f86ffc44bd10408826fb85a94ede8eb908cdf5b27"
COMPARATOR_SHA256 = "49333ade99f04d96e9d4c4ccc2fc01c002170b38f02d10f88fdc8559d274203d"
LAUNCHER_SHA256 = "140212ebbd02fd8de9e2970271aabe11ab0c36bcdc4057531bbdb8d339140e1c"
OBJECTIVE_SHA256 = "2f54e3a3e6ef4e84b4d7ca63d62fd38bd0751a1f7e8aaf4769f9a2c505352c38"
MODEL_SHA256 = "490a23869cc326f4b2c16d12b43d4aacdbd23f6d44e78008c3a1417d3fc4f46d"
OBJECTIVE_VERSION = "equal-context-gene-v1"
OUTPUT = ROOT / "results/slp11-transition" / (
    "human-gwps-fixed-context-minimal-control-physical-state128-"
    "balanced-objective-seed731-v1"
)


class BalancedObjectiveError(ValueError):
    """Raised when the frozen balanced-objective experiment drifts."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def validate_inputs(args: argparse.Namespace) -> dict[str, object]:
    paths = {
        Path(args.data).resolve(strict=True): DATA_SHA256,
        Path(args.features).resolve(strict=True): FEATURE_SHA256,
        Path(args.hepg2_control).resolve(strict=True): HEPG2_CONTROL_SHA256,
        Path(args.comparator).resolve(strict=True): COMPARATOR_SHA256,
        LAUNCHER: LAUNCHER_SHA256,
        OBJECTIVE: OBJECTIVE_SHA256,
        MODEL: MODEL_SHA256,
    }
    for path, expected in paths.items():
        if sha256(path) != expected:
            raise BalancedObjectiveError(f"pinned input SHA-256 drift: {path}")
    with np.load(Path(args.data), allow_pickle=False) as data:
        train = data["split_train"]
        validation = data["split_validation"]
        context = data["context_index"]
        actions = data["action_ids"].astype(str)
        if (
            train.size != 10_719
            or validation.size != 2_339
            or data["split_test"].size
            or data["targets"].shape != (13_058, 7_036)
            or not data["observed"].all()
            or not np.all(data["context_basal_observed"].sum(1) == 6_789)
            or set(actions[train]) & set(actions[validation])
        ):
            raise BalancedObjectiveError("development-only split contract drift")
        row_counts = [int(np.count_nonzero(context[train] == index)) for index in range(3)]
        gene_counts = [
            len(set(actions[train][context[train] == index].tolist())) for index in range(3)
        ]
    if row_counts != [1_522, 1_759, 7_438] or gene_counts != [1_443, 1_666, 6_864]:
        raise BalancedObjectiveError("frozen context/gene weighting populations drifted")
    with np.load(Path(args.features), allow_pickle=False) as features:
        if features["feature_values"].shape != (10_231, 1_156):
            raise BalancedObjectiveError("physical feature shape drift")
    with np.load(Path(args.hepg2_control), allow_pickle=False) as control:
        if int(control["perturbed_expression_rows_read"]) != 0:
            raise BalancedObjectiveError("HepG2 perturbation outcomes were accessed")
    comparator = json.loads(Path(args.comparator).read_text(encoding="utf-8"))
    if (
        comparator["modelConfig"]["action_feature_dim"] != 1_156
        or comparator["modelConfig"]["state_dim"] != 128
        or comparator["testAccessed"]
        or comparator["benchmarkAccessed"]
    ):
        raise BalancedObjectiveError("minimal-v2 physical/state128 comparator drift")
    return {
        "records": 13_058,
        "trainRecords": 10_719,
        "validationRecords": 2_339,
        "testRecords": 0,
        "queries": 7_036,
        "contextTrainRows": row_counts,
        "contextTrainGenes": gene_counts,
        "legacyGradientRowFractions": [value / 10_719 for value in row_counts],
        "balancedContextWeightFraction": [1 / 3, 1 / 3, 1 / 3],
        "objectiveVersion": OBJECTIVE_VERSION,
        "modelFamily": "minimal-control-v2 physical1156 state128",
    }


def command(args: argparse.Namespace, output: Path) -> list[str]:
    return [
        sys.executable,
        str(LAUNCHER),
        "--data",
        str(Path(args.data).resolve()),
        "--features",
        str(Path(args.features).resolve()),
        "--feature-sha256",
        FEATURE_SHA256,
        "--hepg2-control",
        str(Path(args.hepg2_control).resolve()),
        "--original-report",
        str(Path(args.comparator).resolve()),
        "--output",
        str(output),
        "--model-source",
        str(MODEL),
        "--model-sha256",
        MODEL_SHA256,
        "--training-objective",
        OBJECTIVE_VERSION,
        "--device",
        args.device,
        "--epochs",
        "180",
        "--patience",
        "30",
        "--max-seconds",
        "1800",
        "--query-basis-rank",
        "32",
        "--hidden",
        "128",
        "--state-dim",
        "128",
        "--dropout",
        "0.2",
        "--learning-rate",
        "0.0005",
        "--weight-decay",
        "0.1",
        "--ridge-alpha",
        "10000",
        "--seed",
        "731",
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    audit = validate_inputs(args)
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    copies = {}
    for path in (Path(__file__), LAUNCHER, OBJECTIVE, MODEL):
        destination = source / path.name
        if destination.exists():
            destination = source / f"{path.parent.name}-{path.name}"
        shutil.copy2(path, destination)
        copies[destination.relative_to(output).as_posix()] = sha256(destination)
    protocol = {
        "schema": "slp.minimal-v2-balanced-objective-protocol/v1",
        "hypothesis": (
            "fixed equal-context and equal-intervention-gene training mass improves "
            "equal-context gene-macro molecular metrics"
        ),
        "fixedRule": {
            "eachContext": {
                "geneMacroNllGainAgainstMean": 0.02,
                "geneMacroNllGainAgainstFullPhysicalRidge": 0.02,
                "geneMacroCentroidAdjustedPearson": 0.10,
                "noNllRegressionVsMinimalV2PhysicalState128": True,
                "noAdjustedPearsonRegressionVsMinimalV2PhysicalState128": True,
            },
            "allThreeContextsRequired": True,
        },
        "singleChange": (
            "training row NLL weight N/(3 * unique genes in context * records for gene in context)"
        ),
        "weighting": {
            "version": OBJECTIVE_VERSION,
            "globalMean": 1.0,
            "minibatchRenormalization": False,
            "computedFrom": "split_train context_index and stable action_ids only",
        },
        "training": {
            "seed": 731,
            "hidden": 128,
            "stateDim": 128,
            "queryBasisRank": 32,
            "epochs": 180,
            "patience": 30,
            "maxSeconds": 1800,
            "learningRate": 0.0005,
            "weightDecay": 0.1,
            "ridgeAlpha": 10000.0,
            "rowOrderAndRng": "unchanged seed731 permutation",
            "otherFittingArtifacts": "unchanged baselines, OOF calibration, response basis, query amplitude and features",
            "trials": 1,
            "sweep": False,
        },
        "inputs": {
            "development": {"path": str(Path(args.data).resolve()), "sha256": DATA_SHA256},
            "features": {"path": str(Path(args.features).resolve()), "sha256": FEATURE_SHA256},
            "comparator": {
                "path": str(Path(args.comparator).resolve()),
                "sha256": COMPARATOR_SHA256,
            },
            "hepg2ControlMetadataOnly": {
                "path": str(Path(args.hepg2_control).resolve()),
                "sha256": HEPG2_CONTROL_SHA256,
                "usedForFitting": False,
            },
        },
        "sourceHashes": copies,
        "inputAudit": audit,
        "modelVersion": "minimal-control-v2",
        "hepg2OrJurkatPerturbedOutcomesAccessed": False,
        "testAccessed": False,
        "benchmarkAccessed": False,
    }
    write_json(output / "protocol.json", protocol)
    subprocess.run(command(args, output / "model"), cwd=ROOT, check=True)
    model_report_path = output / "model/report.json"
    model_report = json.loads(model_report_path.read_text(encoding="utf-8"))
    summary = {
        "schema": "slp.minimal-v2-balanced-objective-result/v1",
        "advancement": model_report["advancement"],
        "results": model_report["results"],
        "bestEpoch": model_report["bestEpoch"],
        "trainingObjective": model_report["trainingObjective"],
        "elapsedSeconds": time.monotonic() - started,
        "modelReport": {"path": "model/report.json", "sha256": sha256(model_report_path)},
        "checkpoint": {
            "path": "model/model.safetensors",
            "sha256": sha256(output / "model/model.safetensors"),
        },
        "singleNumericalChange": "training row objective weights",
        "hepg2OrJurkatPerturbedOutcomesAccessed": False,
        "testAccessed": False,
        "benchmarkAccessed": False,
    }
    write_json(output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / (
            "data/derived/slp11-human-gwps-fixed-panel-context-v1/"
            "replogle-k562-rpe1-gwps-complete-panel-development-v2-fixed-control-context.npz"
        ),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / (
            "data/derived/slp11-human-physical/direct-experiments700-v1/"
            "human-esm-go-physical-features.npz"
        ),
    )
    parser.add_argument(
        "--hepg2-control",
        type=Path,
        default=ROOT / (
            "data/derived/slp11-human-gwps-fixed-panel-context-v1/"
            "nadig-hepg2-fixed-panel-control-context-v1.npz"
        ),
    )
    parser.add_argument(
        "--comparator",
        type=Path,
        default=ROOT / (
            "results/slp11-transition/"
            "human-gwps-fixed-context-minimal-control-physical-state128-response32-seed731-v1/"
            "model/report.json"
        ),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_only:
        print(json.dumps(validate_inputs(args), sort_keys=True))
        return 0
    print(json.dumps(run(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
