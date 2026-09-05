#!/usr/bin/env python3
"""Append-only numerical finalization and one development score for K562 counts."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "results/slp11-transition/k562-essential-count-latent-state-seed731-v1"
OUTPUT = ROOT / "results/slp11-transition/k562-essential-count-latent-state-seed731-portable-finalization-v2"
INFERENCE = ROOT / "modules/slp-1-1-count-latent-inference-v2/inference.py"
VERIFIER = ROOT / "scripts/verify_slp11_k562_count_latent_artifact_v2.py"
EXPECTED = {
    "protocol.json": "a85d2ab7cb83760a818614f20ab28d2936c3604c4f9236293c18b355391b89e7",
    "artifact-manifest.json": "7f0151d7af61782613407cad22de111df997a12f60e4723cc4c8faaeeb0e24b5",
    "model.safetensors": "c7cc6a369f8b63d936c535f7cc59439fec38033202d4b98616b02270df74f3f8",
    "reference.npz": "8020753e9e2597b08cb94c5351772be05986b286f61e0f7a26be26fbfabae4f6",
    "target-free-probe.npz": "952089fc28040e6504c7d76a558e0bbd9d2e89a47c804fb0ac506c2ff74113e1",
    "loss-history.json": "d4c024f9b270b193e5b8a4969690eb877c19c243c41d75ce208d97fa01d97fef",
    "source/runner.py": "9d6668ceb61a3bb0b9dc540a42430b523632b86ddcf547ec2175bfb2fe155920",
    "source/count_latent_state.py": "75df347a82151074c0ce6f4c732106e70ed17126aff07d017294894421d30bac",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run() -> dict[str, object]:
    if OUTPUT.exists():
        raise FileExistsError("immutable finalization output exists")
    for relative, expected in EXPECTED.items():
        if sha(ORIGINAL / relative) != expected:
            raise ValueError(f"original frozen artifact drift: {relative}")
    OUTPUT.mkdir(parents=True)
    source = OUTPUT / "source"
    source.mkdir()
    shutil.copy2(INFERENCE, source / "inference.py")
    shutil.copy2(VERIFIER, source / "verify.py")
    shutil.copy2(Path(__file__).resolve(), source / "finalize.py")
    amendment = {
        "schema": "slp.k562-count-latent-portable-finalization-protocol/v2",
        "originalProtocolSha256": EXPECTED["protocol.json"],
        "originalModelSha256": EXPECTED["model.safetensors"],
        "originalReferenceSha256": EXPECTED["reference.npz"],
        "originalCompletedUpdates": 12_000,
        "reason": "The original pre-development check used a 1e-5 absolute CP10k CPU/GPU tolerance. It observed max absolute drift 3.0517578125e-5 at large rates while relative CP10k drift was 2.921336016548005e-7 and absolute ln1p drift was 2.0741753448128009e-7. Empty and repeated CPU means were exact. This amendment changes only the numerical portability criterion; model, reference, forecasts and scientific gate remain frozen.",
        "revisedNumericalGate": {
            "maximumRelativeCp10kDifferenceWithUnitFloor": 1e-6,
            "maximumAbsoluteLog1pDifference": 1e-6,
            "repeatedCpuMean": "bit-exact",
            "emptyMean": "bit-exact",
            "execution": "new isolated CPU subprocess using artifact-only saved sources",
        },
        "developmentCountMembersOpenedAtFreeze": False,
        "testOpened": False,
        "supersedes": {
            "path": "results/slp11-transition/k562-essential-count-latent-state-seed731-portable-finalization-v1",
            "reason": "The first continuation passed isolated CPU replay, then stopped before held/development access because the saved runner derived the repository root from its relocated artifact path. This continuation explicitly binds that unchanged frozen runner to the actual workspace data roots.",
        },
        "source": {
            "inference.py": sha(source / "inference.py"),
            "verify.py": sha(source / "verify.py"),
            "finalize.py": sha(source / "finalize.py"),
        },
    }
    write(OUTPUT / "protocol-amendment.json", amendment)
    replay = subprocess.run(
        [sys.executable, str(source / "verify.py"), str(ORIGINAL), str(source / "inference.py")],
        check=True, capture_output=True, text=True, timeout=120,
    )
    verification = json.loads(replay.stdout)
    write(OUTPUT / "isolated-cpu-verification.json", verification)
    freeze = {
        "schema": "slp.k562-count-latent-final-checkpoint-freeze/v2",
        "originalProtocolSha256": EXPECTED["protocol.json"],
        "amendmentProtocolSha256": sha(OUTPUT / "protocol-amendment.json"),
        "modelSha256": EXPECTED["model.safetensors"],
        "referenceSha256": EXPECTED["reference.npz"],
        "isolatedCpuVerificationSha256": sha(OUTPUT / "isolated-cpu-verification.json"),
        "isolatedCpuVerification": verification,
        "developmentCountMembersOpened": False,
        "testOpened": False,
    }
    write(OUTPUT / "FROZEN-BEFORE-DEVELOPMENT.json", freeze)
    write(ORIGINAL / "FROZEN-BEFORE-DEVELOPMENT-V3.json", {
        **freeze,
        "authoritativeContinuation": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
    })

    runner = load(ORIGINAL / "source/runner.py", "frozen_count_latent_runner")
    # The numerical runner is frozen byte-for-byte. Its path constants were
    # originally derived from scripts/, so bind those constants explicitly
    # after relocation into the artifact source directory.
    runner.ROOT = ROOT
    runner.CORE = ORIGINAL / "source/count_latent_state.py"
    runner.INFERENCE = source / "inference.py"
    runner.RAW_CELL_DIR = ROOT / "data/derived/slp11-human-k562-essential-raw-cells-v2"
    runner.TRAINING_MMAP_DIR = ROOT / "data/derived/slp11-human-k562-essential-count-latent-training-mmap-v1"
    runner.STATIC_DIR = ROOT / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1"
    runner.STATIC_PATH = runner.STATIC_DIR / "k562-essential-count-static577.npz"
    runner.ROSTER_PATH = runner.STATIC_DIR / "roster-index.npz"
    runner.CONTROL_PATH = ROOT / "data/derived/slp11-human-k562-essential-count-control/reconstruction-train-nt-gem-v1/gem-control-reference.npz"
    runner.ROUTING_PATH = ROOT / "data/derived/slp11-human-k562-essential-singlecell-metadata-v1/cell-routing-metadata.npz"
    runner.BASELINE_DIR = ROOT / "results/slp11-transition/k562-essential-count-anchored-static-ridge-seed731-v1"
    core = load(ORIGINAL / "source/count_latent_state.py", "frozen_count_latent_core")
    resources = runner.load_training_resources()
    model = core.CountLatentState(core.Config(**runner.MODEL_CONFIG)).cuda().eval()
    model.load_state_dict(load_file(str(ORIGINAL / "model.safetensors")))
    fitting = runner.evaluate_fitting_reconstruction(core, model, resources)
    write(OUTPUT / "fitting-reconstruction-diagnostic.json", fitting)

    metadata = runner.validation_metadata(resources)
    baseline_core = load(
        runner.BASELINE_DIR / "source/count_static_ridge.py", "finalize_count_ridge"
    )
    baseline = runner.load_npz(runner.BASELINE_DIR / "model.npz")
    anchor = baseline_core.control_anchor(
        baseline["basal_rate"], metadata["gem_cell_count"]
    )
    ridge_residual = baseline_core.predict_residual(
        baseline, metadata["raw_action_features"], str(baseline["selected_alpha"])
    )
    ridge = baseline_core.absolute_prediction(anchor, ridge_residual)
    mean = baseline_core.absolute_prediction(
        anchor, np.broadcast_to(baseline["target_mean"], anchor.shape)
    )
    inference = load(source / "inference.py", "finalize_count_inference")
    predictor = inference.Predictor(ORIGINAL, device="cuda")
    neural = predictor.predict(
        metadata["raw_action_features"], metadata["gem_weights"], chunk_size=1024
    )["mean_log1p_cp10k"]
    forecast_path = OUTPUT / "development-forecasts-before-outcomes.npz"
    np.savez_compressed(
        forecast_path,
        schema=np.asarray("slp.k562-count-latent-frozen-development-forecasts/v1"),
        gene_ids=metadata["gene_ids"], query_ids=resources["registered"]["query_ids"],
        gem_group_ids=resources["registered"]["gem_group_ids"],
        gem_cell_count=metadata["gem_cell_count"], cell_count=metadata["cell_count"],
        anchor=anchor, control_prediction=anchor, anchored_mean_prediction=mean,
        static_ridge_prediction=ridge, neural_prediction=neural,
    )
    forecast_freeze = {
        "schema": "slp.k562-count-latent-development-forecast-freeze/v1",
        "forecastSha256": sha(forecast_path),
        "genes": 305, "queries": 8563,
        "cellsRepresentedByMetadata": int(metadata["cell_count"].sum()),
        "baselineModelSha256": runner.HASHES["baselineModel"],
        "modelSha256": EXPECTED["model.safetensors"],
        "developmentCountMembersOpened": False,
        "testOpened": False,
    }
    write(OUTPUT / "FORECASTS-FROZEN-BEFORE-DEVELOPMENT.json", forecast_freeze)

    truth, count_diagnostic = runner.aggregate_validation_truth(metadata["gene_ids"])
    metrics = {
        name: runner.profile_metrics(prediction, truth, anchor)
        for name, prediction in (
            ("neural", neural), ("staticRidge", ridge),
            ("anchoredMean", mean), ("pureControl", anchor),
        )
    }
    candidate, ridge_metric, mean_metric = (
        metrics["neural"], metrics["staticRidge"], metrics["anchoredMean"]
    )
    candidate_r = candidate["independentlyQueryCenteredPearson"]
    ridge_r = ridge_metric["independentlyQueryCenteredPearson"]
    gate = {
        "mseAtLeastOnePercentBelowStaticRidge": (
            candidate["geneProfileMse"] <= 0.99 * ridge_metric["geneProfileMse"]
        ),
        "mseAtLeastOnePercentBelowAnchoredMean": (
            candidate["geneProfileMse"] <= 0.99 * mean_metric["geneProfileMse"]
        ),
        "centeredRAtLeastPoint10": candidate_r is not None and candidate_r >= 0.10,
        "centeredRNonregressionVsStaticRidge": (
            candidate_r is not None and ridge_r is not None and candidate_r >= ridge_r
        ),
    }
    gate["passed"] = all(gate.values())
    loss_history = json.loads((ORIGINAL / "loss-history.json").read_text())
    report = {
        "schema": "slp.k562-essential-count-latent-state-portable-finalized-result/v1",
        "originalProtocolSha256": EXPECTED["protocol.json"],
        "amendmentProtocolSha256": sha(OUTPUT / "protocol-amendment.json"),
        "training": loss_history,
        "fittingReconstruction": fitting,
        "portableVerification": verification,
        "forecastFreeze": forecast_freeze,
        "development": {
            "metrics": metrics, "countDiagnostic": count_diagnostic,
            "negativePredictionFraction": float(np.mean(neural < 0)), "gate": gate,
        },
        "interpretation": "Adaptive held-gene development evidence for a conditional aggregate-mean count model. It is not a validated single-cell generator, identified latent biology, test result, or benchmark claim.",
        "developmentEvaluations": 1,
        "testAccessed": False,
        "benchmarkAccessed": False,
        "artifacts": {
            "model.safetensors": EXPECTED["model.safetensors"],
            "reference.npz": EXPECTED["reference.npz"],
            "development-forecasts-before-outcomes.npz": sha(forecast_path),
            "fitting-reconstruction-diagnostic.json": sha(
                OUTPUT / "fitting-reconstruction-diagnostic.json"
            ),
        },
    }
    write(OUTPUT / "report.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
