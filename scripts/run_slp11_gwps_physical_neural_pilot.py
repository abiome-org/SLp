"""Run one frozen neural pilot with direct physical-neighbor static features."""

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
COMMON_PATH = Path(__file__).with_name("run_slp11_gwps_genome_scale_pilot.py")
COMMON_SPEC = importlib.util.spec_from_file_location("gwps_pilot_contract", COMMON_PATH)
assert COMMON_SPEC is not None and COMMON_SPEC.loader is not None
COMMON = importlib.util.module_from_spec(COMMON_SPEC)
COMMON_SPEC.loader.exec_module(COMMON)

DATA_SHA256 = "006b4bb127a09073a7f409d81a7bccce96bb961879cb5e57dce56b48eb8e664b"
BASE_FEATURE_SHA256 = "a2f3153478c00c191e5a9e218badb3327a180a56948a4c9c6a6926cc506ff02b"
PHYSICAL_FEATURE_SHA256 = (
    "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7"
)
PRIOR_SUMMARY_SHA256 = "203f0f015505b67e30edee2ab4eeaeb7428ac4c1e9ebedd58c110b052f1cb37e"
PHYSICAL_SCREEN_SHA256 = "736968925a96806e1384cf71663e37ffd84fb70c2c4077ed0f240c4dc7a8c4a3"
OUTPUT = ROOT / "results/slp11-transition/human-gwps-physical-fusion-response32-seed731-v1"


class PhysicalNeuralPilotError(ValueError):
    """A frozen physical-neighbor pilot contract was violated."""


def validate_feature_extension(
    base_taxon: np.ndarray,
    base_ids: np.ndarray,
    base_values: np.ndarray,
    physical_taxon: np.ndarray,
    physical_ids: np.ndarray,
    physical_values: np.ndarray,
) -> None:
    """Require identical identities and exact preservation of the first 577 columns."""

    if physical_values.shape != (10_231, 1_156) or physical_values.dtype != np.float32:
        raise PhysicalNeuralPilotError("physical feature shape or dtype drifted")
    if not np.array_equal(base_taxon, physical_taxon) or not np.array_equal(
        base_ids, physical_ids
    ):
        raise PhysicalNeuralPilotError("physical feature entity identities drifted")
    if not np.array_equal(base_values, physical_values[:, :577]):
        raise PhysicalNeuralPilotError("physical pack changed a frozen base feature value")
    if not np.isfinite(physical_values).all():
        raise PhysicalNeuralPilotError("physical feature values are nonfinite")


def evaluate_rules(
    results: dict[str, object], prior_results: dict[str, object]
) -> dict[str, object]:
    """Apply the frozen primary and per-context previous-world no-regression rules."""

    contexts: dict[str, object] = {}
    for context, result in results.items():
        world = result["world"]
        prior = prior_results[context]["world"]
        no_regression = {
            "nll": world["gene_macro_nll"] <= prior["gene_macro_nll"],
            "adjustedPearson": (
                world["gene_macro_profile_centroid_adjusted_pearson_mean"]
                >= prior["gene_macro_profile_centroid_adjusted_pearson_mean"]
            ),
        }
        contexts[context] = {
            "primaryRulePassed": result["development_rule_passed"],
            "noRegressionVsPreviousWorld": {
                **no_regression,
                "passed": all(no_regression.values()),
                "nllGainPreviousMinusPhysical": (
                    prior["gene_macro_nll"] - world["gene_macro_nll"]
                ),
                "adjustedPearsonPhysicalMinusPrevious": (
                    world["gene_macro_profile_centroid_adjusted_pearson_mean"]
                    - prior["gene_macro_profile_centroid_adjusted_pearson_mean"]
                ),
            },
        }
    return {
        "contexts": contexts,
        "primaryRulePassedAllContexts": all(
            value["primaryRulePassed"] for value in contexts.values()
        ),
        "noRegressionPassedAllContexts": all(
            value["noRegressionVsPreviousWorld"]["passed"]
            for value in contexts.values()
        ),
        "hypothesisPassed": all(
            value["primaryRulePassed"]
            and value["noRegressionVsPreviousWorld"]["passed"]
            for value in contexts.values()
        ),
    }


def validate_inputs(args: argparse.Namespace) -> dict[str, object]:
    data = Path(args.data).resolve(strict=True)
    base = Path(args.base_features).resolve(strict=True)
    physical = Path(args.features).resolve(strict=True)
    old = Path(args.old_features).resolve(strict=True)
    prior_summary = Path(args.prior_summary).resolve(strict=True)
    physical_screen = Path(args.physical_screen).resolve(strict=True)
    base_audit = COMMON.validate_inputs(data, base, old)
    if COMMON.sha256(physical) != PHYSICAL_FEATURE_SHA256:
        raise PhysicalNeuralPilotError("physical feature SHA-256 drift")
    if COMMON.sha256(prior_summary) != PRIOR_SUMMARY_SHA256:
        raise PhysicalNeuralPilotError("previous world summary SHA-256 drift")
    if COMMON.sha256(physical_screen) != PHYSICAL_SCREEN_SHA256:
        raise PhysicalNeuralPilotError("physical ridge screen SHA-256 drift")
    with np.load(base, allow_pickle=False) as archive:
        base_taxon = archive["entity_taxon"]
        base_ids = archive["entity_id"]
        base_values = archive["feature_values"]
    with np.load(physical, allow_pickle=False) as archive:
        physical_taxon = archive["entity_taxon"]
        physical_ids = archive["entity_id"]
        physical_values = archive["feature_values"]
    validate_feature_extension(
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
        "appendedPhysicalColumns": 579,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"immutable physical neural output already exists: {output}")
    audit = validate_inputs(args)
    data = Path(args.data).resolve(strict=True)
    features = Path(args.features).resolve(strict=True)
    base_features = Path(args.base_features).resolve(strict=True)
    prior_summary_path = Path(args.prior_summary).resolve(strict=True)
    physical_screen_path = Path(args.physical_screen).resolve(strict=True)
    prior_summary = json.loads(prior_summary_path.read_text(encoding="utf-8"))
    physical_screen = json.loads(physical_screen_path.read_text(encoding="utf-8"))

    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    runner_copy = source / Path(__file__).name
    common_copy = source / COMMON_PATH.name
    shutil.copy2(Path(__file__), runner_copy)
    shutil.copy2(COMMON_PATH, common_copy)
    protocol = {
        "schema": "slp.gwps-physical-neural-development-protocol/v1",
        "hypothesis": (
            "Nonlinear use of direct physical-neighbor features improves held-gene molecular "
            "prediction beyond both the full 1,156-feature ridge and the previous full-GWPS world."
        ),
        "fixedRule": {
            "primaryEachContext": {
                "geneMacroNllGainAgainstContextMean": 0.02,
                "geneMacroNllGainAgainstFull1156FeatureRidgeAlpha10000": 0.02,
                "geneMacroCentroidAdjustedPearson": 0.10,
            },
            "noRegressionEachContextVsPreviousSingleSeedWorld": {
                "geneMacroNll": "less than or equal to previous",
                "geneMacroCentroidAdjustedPearson": "greater than or equal to previous",
            },
            "allThreeContextsRequired": True,
        },
        "inputs": {
            "development": {"path": str(data), "sha256": DATA_SHA256},
            "physicalFeatures": {"path": str(features), "sha256": PHYSICAL_FEATURE_SHA256},
            "baseFeatures": {"path": str(base_features), "sha256": BASE_FEATURE_SHA256},
            "previousWorldSummary": {
                "path": str(prior_summary_path),
                "sha256": PRIOR_SUMMARY_SHA256,
            },
            "physicalRidgeScreen": {
                "path": str(physical_screen_path),
                "sha256": PHYSICAL_SCREEN_SHA256,
            },
        },
        "inputAudit": audit,
        "training": {
            "seed": 731,
            "queryBasisRank": 32,
            "hidden": 128,
            "stateDim": 64,
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
        },
        "degreeOnlyBaseline": {
            "role": "point-prediction diagnostic copied from the pinned ridge screen",
            "uncertaintyComparable": False,
            "results": {
                context: values["degree"]
                for context, values in physical_screen["results"].items()
            },
        },
        "screenPromotion": False,
        "testArtifactAccessed": False,
        "slBenchmarkAccessed": False,
        "sourceCopies": {
            runner_copy.relative_to(output).as_posix(): COMMON.sha256(runner_copy),
            common_copy.relative_to(output).as_posix(): COMMON.sha256(common_copy),
        },
    }
    COMMON.write_json(output / "protocol.json", protocol)
    model_output = output / "model"
    command = [
        sys.executable,
        str(ROOT / "modules/slp-1-1-world-transition-v1/train_human.py"),
        "--data",
        str(data),
        "--data-sha256",
        DATA_SHA256,
        "--features",
        str(features),
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
        "--exposure-aware",
        "--reference-kind",
        "mean",
        "--hidden",
        "128",
        "--state-dim",
        "64",
        "--dropout",
        "0.2",
        "--weight-decay",
        "0.1",
        "--ridge-alpha",
        "10000",
        "--seed",
        "731",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    model_report_path = model_output / "report.json"
    model_report = json.loads(model_report_path.read_text(encoding="utf-8"))
    decision = evaluate_rules(model_report["results"], prior_summary["results"])
    summary = {
        "schema": "slp.gwps-physical-neural-development-result/v1",
        "decision": decision,
        "results": model_report["results"],
        "degreeOnlyPointBaseline": protocol["degreeOnlyBaseline"],
        "bestEpoch": model_report["best_epoch"],
        "elapsedSeconds": time.monotonic() - started,
        "modelReport": {"path": "model/report.json", "sha256": COMMON.sha256(model_report_path)},
        "checkpoint": {
            "path": "model/model.safetensors",
            "sha256": COMMON.sha256(model_output / "model.safetensors"),
        },
        "previousPilotRemainsImmutable": True,
        "physicalRidgeScreenPromoted": False,
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
        "--prior-summary",
        type=Path,
        default=ROOT / (
            "results/slp11-transition/human-gwps-complete-panel-fusion-response32-seed731-v1/"
            "summary.json"
        ),
    )
    parser.add_argument(
        "--physical-screen",
        type=Path,
        default=ROOT / "results/slp11-transition/physical-features-ridge-screen-v1/report.json",
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
