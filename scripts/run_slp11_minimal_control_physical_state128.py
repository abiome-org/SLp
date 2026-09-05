"""Run the frozen physical/state-128 minimal-control candidate synthesis."""

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
DATA_SHA256 = "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c"
PHYSICAL_FEATURE_SHA256 = (
    "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7"
)
BASE_FEATURE_SHA256 = "a2f3153478c00c191e5a9e218badb3327a180a56948a4c9c6a6926cc506ff02b"
HEPG2_CONTROL_SHA256 = "382626401ee38e8d5084ac9f86ffc44bd10408826fb85a94ede8eb908cdf5b27"
STATE64_REPORT_SHA256 = "665edc70ce283df3187ea0a16485e36f2b3c061fdb5d933065c5b1f85cddf3f9"
LAUNCHER_SHA256 = "8e035a77ad1efe6886d6fd77596a17ec9ee897d4e735c82102b32ee0bddf1635"
OUTPUT = ROOT / "results/slp11-transition" / (
    "human-gwps-fixed-context-minimal-control-physical-state128-response32-seed731-v1"
)


class CandidateSynthesisError(ValueError):
    """The frozen candidate-synthesis contract was violated."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def training_command(args: argparse.Namespace, model_output: Path) -> list[str]:
    """Return the explicit frozen invocation of the shared focused launcher."""

    return [
        sys.executable,
        str(LAUNCHER),
        "--data",
        str(Path(args.data).resolve()),
        "--features",
        str(Path(args.features).resolve()),
        "--feature-sha256",
        PHYSICAL_FEATURE_SHA256,
        "--hepg2-control",
        str(Path(args.hepg2_control).resolve()),
        "--original-report",
        str(Path(args.state64_report).resolve()),
        "--output",
        str(model_output),
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


def validate_inputs(args: argparse.Namespace) -> dict[str, object]:
    data = Path(args.data).resolve(strict=True)
    physical = Path(args.features).resolve(strict=True)
    base = Path(args.base_features).resolve(strict=True)
    hepg2 = Path(args.hepg2_control).resolve(strict=True)
    state64 = Path(args.state64_report).resolve(strict=True)
    expected = {
        data: DATA_SHA256,
        physical: PHYSICAL_FEATURE_SHA256,
        base: BASE_FEATURE_SHA256,
        hepg2: HEPG2_CONTROL_SHA256,
        state64: STATE64_REPORT_SHA256,
        LAUNCHER: LAUNCHER_SHA256,
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise CandidateSynthesisError(f"pinned input SHA-256 drift: {path}")
    with np.load(data, allow_pickle=False) as archive:
        if (
            len(archive["split_train"]) != 10_719
            or len(archive["split_validation"]) != 2_339
            or len(archive["split_test"])
            or archive["targets"].shape != (13_058, 7_036)
            or not archive["observed"].all()
            or not np.all(archive["context_basal_observed"].sum(1) == 6_789)
        ):
            raise CandidateSynthesisError("common-context development contract drifted")
        if set(archive["action_ids"][archive["split_train"]].tolist()) & set(
            archive["action_ids"][archive["split_validation"]].tolist()
        ):
            raise CandidateSynthesisError("intervention identity crosses development splits")
        contexts = archive["context_ids"].astype(str).tolist()
    with np.load(hepg2, allow_pickle=False) as archive:
        if int(archive["perturbed_expression_rows_read"]) != 0:
            raise CandidateSynthesisError("HepG2 perturbed outcomes were read")
        if int(archive["context_basal_observed"].sum()) != 6_789:
            raise CandidateSynthesisError("HepG2 common-control descriptor support drifted")
    with np.load(base, allow_pickle=False) as archive:
        base_ids = archive["entity_id"]
        base_taxon = archive["entity_taxon"]
        base_values = archive["feature_values"]
    with np.load(physical, allow_pickle=False) as archive:
        physical_ids = archive["entity_id"]
        physical_taxon = archive["entity_taxon"]
        physical_values = archive["feature_values"]
    if (
        physical_values.shape != (10_231, 1_156)
        or physical_values.dtype != np.float32
        or not np.array_equal(base_ids, physical_ids)
        or not np.array_equal(base_taxon, physical_taxon)
        or not np.array_equal(base_values, physical_values[:, :577])
        or not np.isfinite(physical_values).all()
    ):
        raise CandidateSynthesisError("physical static feature extension contract drifted")
    previous = json.loads(state64.read_text(encoding="utf-8"))
    if previous["modelConfig"]["action_feature_dim"] != 577 or previous["modelConfig"][
        "state_dim"
    ] != 64:
        raise CandidateSynthesisError("minimal-control state64 comparator contract drifted")
    return {
        "records": 13_058,
        "trainRecords": 10_719,
        "validationRecords": 2_339,
        "testRecords": 0,
        "queries": 7_036,
        "contexts": contexts,
        "commonControlQueriesPerContext": 6_789,
        "featureShape": [10_231, 1_156],
        "baseColumnsVerifiedExact": 577,
        "hepg2PerturbedRowsRead": 0,
        "twoCombinedChanges": ["physical feature dimensions 577 to 1156", "state_dim 64 to 128"],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"immutable candidate output already exists: {output}")
    audit = validate_inputs(args)
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    wrapper_copy = source / Path(__file__).name
    launcher_copy = source / LAUNCHER.name
    shutil.copy2(Path(__file__), wrapper_copy)
    shutil.copy2(LAUNCHER, launcher_copy)
    protocol = {
        "schema": "slp.minimal-control-physical-state128-candidate-synthesis-protocol/v1",
        "hypothesis": (
            "The minimal-control-v2 architecture with physical features and state 128 improves "
            "held-gene molecular forecasts while preserving exact control-only inference."
        ),
        "purpose": (
            "Match stronger baseline modalities and provide a relevant candidate for a later frozen "
            "new-context diagnostic. Passing that diagnostic cannot rewrite failed development gates."
        ),
        "fixedRule": {
            "eachContext": {
                "geneMacroNllGainAgainstMean": 0.02,
                "geneMacroNllGainAgainstFullPhysicalRidge": 0.02,
                "geneMacroCentroidAdjustedPearson": 0.10,
                "noNllRegressionVsMinimalControl577State64": True,
                "noAdjustedPearsonRegressionVsMinimalControl577State64": True,
            },
            "allThreeContextsRequired": True,
        },
        "candidateSynthesis": {
            "combinedChanges": audit["twoCombinedChanges"],
            "isolatedCausalAblation": False,
        },
        "inputs": {
            "commonContextDevelopment": {
                "path": str(Path(args.data).resolve()),
                "sha256": DATA_SHA256,
            },
            "physicalFeatures": {
                "path": str(Path(args.features).resolve()),
                "sha256": PHYSICAL_FEATURE_SHA256,
            },
            "minimalControl577State64Report": {
                "path": str(Path(args.state64_report).resolve()),
                "sha256": STATE64_REPORT_SHA256,
            },
            "hepg2ControlOnlyDescriptor": {
                "path": str(Path(args.hepg2_control).resolve()),
                "sha256": HEPG2_CONTROL_SHA256,
                "perturbedRowsRead": 0,
            },
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
            "queryAmplitude": "one shared vector fitted from pooled training-only grouped-OOF residuals",
            "trials": 1,
            "hyperparameterSweep": False,
        },
        "inputAudit": audit,
        "source": {
            "launcherSha256": LAUNCHER_SHA256,
            "copies": {
                wrapper_copy.relative_to(output).as_posix(): sha256(wrapper_copy),
                launcher_copy.relative_to(output).as_posix(): sha256(launcher_copy),
            },
        },
        "hepg2PerturbedOrTestOutcomesAccessed": False,
        "slBenchmarkAccessed": False,
    }
    write_json(output / "protocol.json", protocol)
    model_output = output / "model"
    subprocess.run(training_command(args, model_output), cwd=ROOT, check=True)
    report_path = model_output / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = {
        "schema": "slp.minimal-control-physical-state128-candidate-synthesis-result/v1",
        "advancement": report["advancement"],
        "results": report["results"],
        "bestEpoch": report["bestEpoch"],
        "parameters": report["parameters"],
        "controlIdentityAfterCheckpoint": report["controlIdentityAfterCheckpoint"],
        "elapsedSeconds": time.monotonic() - started,
        "modelReport": {"path": "model/report.json", "sha256": sha256(report_path)},
        "checkpoint": {
            "path": "model/model.safetensors",
            "sha256": sha256(model_output / "model.safetensors"),
        },
        "candidateSynthesisNotCausalAblation": True,
        "priorFailedGatesRemainFailed": True,
        "hepg2PerturbedOrTestOutcomesAccessed": False,
        "slBenchmarkAccessed": False,
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
        "--base-features",
        type=Path,
        default=ROOT / (
            "data/derived/slp11-human-gwps-static/ensembl116-goa2022-fixed-basis-v1/"
            "gwps-extended-static-esm-go-features.npz"
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
        "--state64-report",
        type=Path,
        default=ROOT / (
            "results/slp11-transition/"
            "human-gwps-fixed-context-minimal-control-response32-seed731-v1/report.json"
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
