"""Run one frozen state-128 decoder-capacity pilot on physical features."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).parent


def _load_script(name: str, module_name: str):
    path = SCRIPT_DIR / name
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = _load_script("run_slp11_gwps_genome_scale_pilot.py", "state128_common")
PHYSICAL = _load_script("run_slp11_gwps_physical_neural_pilot.py", "state128_physical")

DATA_SHA256 = "006b4bb127a09073a7f409d81a7bccce96bb961879cb5e57dce56b48eb8e664b"
BASE_FEATURE_SHA256 = "a2f3153478c00c191e5a9e218badb3327a180a56948a4c9c6a6926cc506ff02b"
PHYSICAL_FEATURE_SHA256 = (
    "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7"
)
STATE64_SUMMARY_SHA256 = "344bb7dab606d496b6f7533e1407eae1f0f894b0ca2cf47e1931d07a00248950"
DECODER_AUDIT_REPORT_SHA256 = (
    "bde60c82d506303dfe33adeb3329a612ff355811cfe8e72f065fbef6f1ea83d8"
)
DECODER_AUDIT_PROTOCOL_SHA256 = (
    "bdaf1e2380ea7a050d213fb64b8b8e474339ac00bd1330f8b50a427dd3238a6f"
)
OUTPUT = ROOT / "results/slp11-transition/human-gwps-physical-fusion-response32-state128-seed731-v1"


class State128PilotError(ValueError):
    """The frozen state-128 decoder pilot contract was violated."""


def evaluate_rules(
    results: dict[str, object], state64_results: dict[str, object]
) -> dict[str, object]:
    """Apply the unchanged primary rule and strict state-64 no-regression rule."""

    contexts: dict[str, object] = {}
    for context, result in results.items():
        world = result["world"]
        state64 = state64_results[context]["world"]
        nll_gain = state64["gene_macro_nll"] - world["gene_macro_nll"]
        r_gain = (
            world["gene_macro_profile_centroid_adjusted_pearson_mean"]
            - state64["gene_macro_profile_centroid_adjusted_pearson_mean"]
        )
        contexts[context] = {
            "primaryRulePassed": result["development_rule_passed"],
            "noRegressionVsPhysicalState64": {
                "nll": nll_gain >= 0.0,
                "adjustedPearson": r_gain >= 0.0,
                "passed": nll_gain >= 0.0 and r_gain >= 0.0,
                "nllGainState64MinusState128": nll_gain,
                "adjustedPearsonState128MinusState64": r_gain,
            },
        }
    primary = all(value["primaryRulePassed"] for value in contexts.values())
    no_regression = all(
        value["noRegressionVsPhysicalState64"]["passed"] for value in contexts.values()
    )
    return {
        "contexts": contexts,
        "primaryRulePassedAllContexts": primary,
        "noRegressionPassedAllContexts": no_regression,
        "hypothesisPassed": primary and no_regression,
    }


def validate_inputs(args: argparse.Namespace) -> dict[str, object]:
    data = Path(args.data).resolve(strict=True)
    base = Path(args.base_features).resolve(strict=True)
    old = Path(args.old_features).resolve(strict=True)
    physical = Path(args.features).resolve(strict=True)
    state64 = Path(args.state64_summary).resolve(strict=True)
    audit_report = Path(args.decoder_audit_report).resolve(strict=True)
    audit_protocol = Path(args.decoder_audit_protocol).resolve(strict=True)
    base_audit = COMMON.validate_inputs(data, base, old)
    expected = {
        physical: PHYSICAL_FEATURE_SHA256,
        state64: STATE64_SUMMARY_SHA256,
        audit_report: DECODER_AUDIT_REPORT_SHA256,
        audit_protocol: DECODER_AUDIT_PROTOCOL_SHA256,
    }
    for path, digest in expected.items():
        if COMMON.sha256(path) != digest:
            raise State128PilotError(f"pinned input SHA-256 drift: {path}")
    with np.load(base, allow_pickle=False) as archive:
        base_taxon = archive["entity_taxon"]
        base_ids = archive["entity_id"]
        base_values = archive["feature_values"]
    with np.load(physical, allow_pickle=False) as archive:
        physical_taxon = archive["entity_taxon"]
        physical_ids = archive["entity_id"]
        physical_values = archive["feature_values"]
    PHYSICAL.validate_feature_extension(
        base_taxon,
        base_ids,
        base_values,
        physical_taxon,
        physical_ids,
        physical_values,
    )
    return {
        **base_audit,
        "physicalFeatureRows": len(physical_ids),
        "physicalFeatureDimensions": physical_values.shape[1],
        "baseColumnsVerifiedExact": 577,
        "onlyArchitectureChange": "state_dim 64 to 128",
    }


def training_command(
    data: Path, features: Path, output: Path, device: str
) -> list[str]:
    """Return the fully frozen training invocation."""

    return [
        sys.executable,
        str(ROOT / "modules/slp-1-1-world-transition-v1/train_human.py"),
        "--data",
        str(data),
        "--data-sha256",
        DATA_SHA256,
        "--features",
        str(features),
        "--output",
        str(output),
        "--device",
        device,
        "--epochs",
        "180",
        "--patience",
        "30",
        "--max-seconds",
        "1800",
        "--query-basis-rank",
        "32",
        "--exposure-aware",
        "--reference-kind",
        "mean",
        "--hidden",
        "128",
        "--state-dim",
        "128",
        "--dropout",
        "0.2",
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
        raise FileExistsError(f"immutable state-128 output already exists: {output}")
    input_audit = validate_inputs(args)
    data = Path(args.data).resolve(strict=True)
    features = Path(args.features).resolve(strict=True)
    state64_path = Path(args.state64_summary).resolve(strict=True)
    audit_report = Path(args.decoder_audit_report).resolve(strict=True)
    audit_protocol = Path(args.decoder_audit_protocol).resolve(strict=True)
    state64 = json.loads(state64_path.read_text(encoding="utf-8"))

    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    source_copies = {}
    for script in (
        Path(__file__),
        SCRIPT_DIR / "run_slp11_gwps_physical_neural_pilot.py",
        SCRIPT_DIR / "run_slp11_gwps_genome_scale_pilot.py",
    ):
        destination = source / script.name
        shutil.copy2(script, destination)
        source_copies[destination.relative_to(output).as_posix()] = COMMON.sha256(destination)
    protocol = {
        "schema": "slp.gwps-physical-state128-development-protocol/v1",
        "hypothesis": (
            "Doubling the latent decoder state from 64 to 128 closes the diagnosed "
            "representational loss and improves held-gene molecular forecasts."
        ),
        "fixedRule": {
            "primaryEachContext": {
                "geneMacroNllGainAgainstContextMean": 0.02,
                "geneMacroNllGainAgainstFull1156FeatureRidgeAlpha10000": 0.02,
                "geneMacroCentroidAdjustedPearson": 0.10,
            },
            "noRegressionEachContextVsPhysicalState64": {
                "geneMacroNll": "less than or equal to state64",
                "geneMacroCentroidAdjustedPearson": "greater than or equal to state64",
            },
            "allThreeContextsRequired": True,
        },
        "inputs": {
            "development": {"path": str(data), "sha256": DATA_SHA256},
            "physicalFeatures": {"path": str(features), "sha256": PHYSICAL_FEATURE_SHA256},
            "physicalState64Summary": {
                "path": str(state64_path),
                "sha256": STATE64_SUMMARY_SHA256,
            },
            "decoderSpanAudit": {
                "report": str(audit_report),
                "reportSha256": DECODER_AUDIT_REPORT_SHA256,
                "protocol": str(audit_protocol),
                "protocolSha256": DECODER_AUDIT_PROTOCOL_SHA256,
                "role": "geometry diagnostic only; not an oracle or fitted target",
            },
        },
        "inputAudit": input_audit,
        "training": {
            "seed": 731,
            "featureDimensions": 1156,
            "queryBasisRank": 32,
            "hidden": 128,
            "stateDim": 128,
            "dropout": 0.2,
            "weightDecay": 0.1,
            "epochs": 180,
            "patience": 30,
            "maxSeconds": 1800,
            "ridgeAlpha": 10000.0,
            "referenceKind": "mean",
            "exposureAware": True,
            "trials": 1,
            "hyperparameterSweep": False,
            "onlyChangeVsParent": "stateDim 64 to 128",
        },
        "parameterCountIsEvidenceOfGain": False,
        "physicalState64ParentRemainsImmutable": True,
        "testArtifactAccessed": False,
        "slBenchmarkAccessed": False,
        "sourceCopies": source_copies,
    }
    COMMON.write_json(output / "protocol.json", protocol)
    model_output = output / "model"
    subprocess.run(
        training_command(data, features, model_output, args.device), cwd=ROOT, check=True
    )
    model_report_path = model_output / "report.json"
    model_report = json.loads(model_report_path.read_text(encoding="utf-8"))
    decision = evaluate_rules(model_report["results"], state64["results"])
    summary = {
        "schema": "slp.gwps-physical-state128-development-result/v1",
        "decision": decision,
        "results": model_report["results"],
        "bestEpoch": model_report["best_epoch"],
        "parameters": model_report["parameters"],
        "elapsedSeconds": time.monotonic() - started,
        "modelReport": {"path": "model/report.json", "sha256": COMMON.sha256(model_report_path)},
        "checkpoint": {
            "path": "model/model.safetensors",
            "sha256": COMMON.sha256(model_output / "model.safetensors"),
        },
        "physicalState64ParentRemainsImmutable": True,
        "testArtifactAccessed": False,
        "slBenchmarkAccessed": False,
    }
    COMMON.write_json(output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/derived/slp11-human-gwps/complete-panel-v1/development.npz",
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
        "--old-features",
        type=Path,
        default=ROOT / (
            "data/derived/slp11-human-static-fusion/esm2-t6-plus-go-svd-v1/"
            "human-static-esm-go-features.npz"
        ),
    )
    parser.add_argument(
        "--state64-summary",
        type=Path,
        default=ROOT / (
            "results/slp11-transition/human-gwps-physical-fusion-response32-seed731-v1/"
            "summary.json"
        ),
    )
    parser.add_argument(
        "--decoder-audit-report",
        type=Path,
        default=ROOT / "results/slp11-transition/gwps-decoder-span-audit-v1/report.json",
    )
    parser.add_argument(
        "--decoder-audit-protocol",
        type=Path,
        default=ROOT / "results/slp11-transition/gwps-decoder-span-audit-v1/protocol.json",
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
