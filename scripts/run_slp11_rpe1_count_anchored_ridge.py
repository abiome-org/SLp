#!/usr/bin/env python3
"""Freeze the fitting-only RPE1 control-anchored static577 ridge baseline."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "modules/slp-1-1-count-static-ridge-v1/count_static_ridge.py"
OUTPUT = ROOT / "results/slp11-transition/rpe1-essential-count-anchored-static-ridge-seed731-v1"
STATIC = ROOT / "data/derived/slp11-human-rpe1-essential-count-static/ensembl116-esm8m-shared-go-v1/rpe1-essential-count-static577.npz"
ROSTER = ROOT / "data/derived/slp11-human-rpe1-essential-count-static/ensembl116-esm8m-shared-go-v1/roster-index.npz"
NORMALIZERS = ROOT / "data/derived/slp11-human-rpe1-essential-count-static/ensembl116-esm8m-shared-go-v1/fitting-action-normalizers.npz"
CONTROL = ROOT / "data/derived/slp11-human-rpe1-essential-count-control/reconstruction-train-nt-gem-v1/gem-control-reference.npz"
MOMENTS = ROOT / "data/derived/slp11-human-rpe1-essential-raw-cells-v1/fitting-action-moments.npz"
SOURCE_SHA256 = "9b05ef1f81526216fa008d677e9e0d03dce9a2f7a95499a4fb81e505e9d88ef1"
ROUTING_SHA256 = "10f3d313a5671122bde10a9bd586e3a2808d6f9b554f737ddcbbc28becc5e2f2"
PINS = {
    CORE: "1032eeff59382fae3874da9a389033192e113e0f5ac2c8d01f09f8441d969e62",
    STATIC: "621e1e9f0dffc740ef42382b1b2898f629edd5037e8a02d411e8d30e815ed816",
    ROSTER: "b9e1b169c2be4ac756e94f465009dc5bef80d06bc0652950c3cf6916d26d1e56",
    NORMALIZERS: "d397dfdb08973ccf9884d504a1279042cc470cba2ff5341c770443d0c7915951",
    CONTROL: "c0c2eab217d00f9555b6ab5725cd2c49f56b1ecdf34b7af47f303eee9d1b8e20",
    MOMENTS: "d15def86aead06b0bc75ab63c77513735ec7c57d65012bff72f3947bc654895c",
}
QUERY_COUNT = 8749
ACTION_COUNT = 1666
GEM_COUNT = 56
SECONDS = 600.0


class RpeRidgeError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, array in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, member.getvalue(), compresslevel=9)
    return output.getvalue()


def scalar_text(values: dict[str, np.ndarray], key: str) -> str:
    if key not in values or values[key].ndim != 0:
        raise RpeRidgeError(f"missing scalar {key}")
    return str(values[key].item())


def frozen_protocol() -> dict[str, object]:
    return {
        "schema": "slp.rpe1-essential-count-anchored-static-ridge-protocol/v1",
        "purpose": "Matched aggregate-mean RPE1 baseline for a future raw-count conditional-prior experiment; no cell-generation or transfer claim.",
        "hypothesis": "Static intervention descriptors predict fitting-held residual molecular means beyond a GEM-composition-matched control anchor.",
        "endpoint": "For each fitting action, ln1p(equal-cell mean raw CP10k) across all 8,749 source queries.",
        "anchor": "ln1p of the gene-cell-weighted mixture over 56 GEM-specific reconstruction-training NT control CP10k rates, smoothed by 0.5 in full-panel count space.",
        "model": "Fit endpoint minus anchor from raw static577 with the unchanged count-static-ridge-v1 core, exact unpenalized residual intercept, and ridge-penalized feature coefficients.",
        "selection": {
            "folds": "three exact global folds SHA256(slp11-bp-ridge-v1|731|global-inner-fold|9606|ENSG)",
            "candidates": ["0.1", "1", "10", "100", "1000", "10000", "100000", "1e+06", "mean-limit"],
            "criterion": "equal-gene raw MSE across every 8,749 query",
            "featureNormalization": "fold-local float64 population mean/SD over unique fitting genes; SD<=1e-5 uses scale1; final state refit on all 1,666 fitting genes",
        },
        "accessibleData": "RPE1 fitting-action aggregate moments, reconstruction-training NT control moments, stable metadata rosters, and frozen static sequence/GO only.",
        "outcomeBoundary": {"fittingMoments": True, "reconstructionHeld": False, "developmentValidation": False, "test": False, "syntheticLethality": False},
        "inputs": {str(path.relative_to(ROOT)): digest for path, digest in PINS.items()},
        "source": {"runnerSha256": sha256_file(Path(__file__).resolve()), "coreSha256": PINS[CORE]},
        "limits": {"cpuThreads": 2, "wallSeconds": SECONDS},
    }


def prepare(output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError("immutable RPE1 ridge output already exists")
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    shutil.copy2(Path(__file__).resolve(), source / "runner.py")
    shutil.copy2(CORE, source / "count_static_ridge.py")
    protocol = frozen_protocol()
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt = {
        "schema": "slp.rpe1-essential-count-anchored-static-ridge-prepared/v1",
        "protocolSha256": sha256_file(output / "protocol.json"),
        "runnerSha256": sha256_file(source / "runner.py"),
        "coreSha256": sha256_file(source / "count_static_ridge.py"),
        "fittingMomentsOpened": False,
        "developmentValidationOpened": False,
        "testOpened": False,
    }
    (output / "PREPARED.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def load_inputs():
    for path, expected in PINS.items():
        if sha256_file(path) != expected:
            raise RpeRidgeError(f"frozen input hash mismatch: {path}")
    static, roster, normalizers, control, moments = map(
        load_npz, (STATIC, ROSTER, NORMALIZERS, CONTROL, MOMENTS)
    )
    if (
        scalar_text(control, "source_sha256") != SOURCE_SHA256
        or scalar_text(control, "routing_sha256") != ROUTING_SHA256
        or scalar_text(moments, "source_sha256") != SOURCE_SHA256
        or scalar_text(moments, "routing_sha256") != ROUTING_SHA256
    ):
        raise RpeRidgeError("control/fitting lineage mismatch")
    query_ids = roster["query_ids"].astype(str)
    genes = roster["fitting_action_ids"].astype(str)
    if (
        not np.array_equal(query_ids, control["query_ids"].astype(str))
        or not np.array_equal(query_ids, moments["query_ids"].astype(str))
        or not np.array_equal(genes, moments["action_ids"].astype(str))
        or query_ids.shape != (QUERY_COUNT,)
        or genes.shape != (ACTION_COUNT,)
        or not np.array_equal(control["gem_group"], moments["gem_group"])
        or not np.array_equal(control["gem_group"], np.arange(1, GEM_COUNT + 1))
    ):
        raise RpeRidgeError("RPE1 query/action/GEM axes disagree")
    gem_count = np.asarray(moments["gem_cell_count"], dtype=np.int64)
    cell_count = np.asarray(moments["cell_count"], dtype=np.int64)
    cp10k_sum = np.asarray(moments["cp10k_sum"], dtype=np.float64)
    if (
        gem_count.shape != (ACTION_COUNT, GEM_COUNT)
        or cell_count.shape != (ACTION_COUNT,)
        or cp10k_sum.shape != (ACTION_COUNT, QUERY_COUNT)
        or not np.array_equal(gem_count.sum(1), cell_count)
        or int(cell_count.sum()) != 142601
        or np.any(cell_count <= 0)
        or np.any(cp10k_sum < 0)
        or not np.isfinite(cp10k_sum).all()
    ):
        raise RpeRidgeError("RPE1 fitting moments violate the frozen contract")
    entities = static["entity_id"].astype(str)
    lookup = {gene: row for row, gene in enumerate(entities)}
    indices = np.asarray([lookup[gene] for gene in genes], dtype=np.int64)
    if not np.array_equal(indices, roster["fitting_action_entity_index"]):
        raise RpeRidgeError("RPE1 fitting static indices disagree")
    features = np.asarray(static["feature_values"][indices], dtype=np.float32)
    if features.shape != (ACTION_COUNT, 577) or not np.isfinite(features).all():
        raise RpeRidgeError("RPE1 fitting features are invalid")
    if not np.array_equal(normalizers["rpe1_fitting_action_ids"].astype(str), genes):
        raise RpeRidgeError("RPE1 normalizer roster differs")
    return roster, normalizers, control, moments, features


def verify_artifact(output: Path) -> dict[str, object]:
    manifest = json.loads((output / "artifact-manifest-before-development.json").read_text())
    for relative, expected in manifest["hashes"].items():
        if sha256_file(output / relative) != expected:
            raise RpeRidgeError(f"saved artifact hash drift: {relative}")
    core = load_module(output / "source/count_static_ridge.py", "rpe_saved_count_ridge")
    model, probe = load_npz(output / "model.npz"), load_npz(output / "target-free-probe.npz")
    residual = core.predict_residual(model, probe["feature_values"], str(model["selected_alpha"].item()))
    anchor = core.control_anchor(model["basal_rate"], probe["gem_cell_count"])
    absolute = core.absolute_prediction(anchor, residual)
    maximum = max(
        float(np.max(np.abs(residual - probe["expected_residual"]))),
        float(np.max(np.abs(anchor - probe["expected_anchor"]))),
        float(np.max(np.abs(absolute - probe["expected_absolute"]))),
    )
    result = {"maximumAbsoluteDifference": maximum, "tolerance": 1e-10, "passes": maximum <= 1e-10}
    if not result["passes"]:
        raise RpeRidgeError("saved artifact replay failed")
    return result


def execute(output: Path) -> dict[str, object]:
    if not (output / "PREPARED.json").exists() or (output / "model.npz").exists():
        raise RpeRidgeError("prepare exactly once before fitting")
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol != frozen_protocol():
        raise RpeRidgeError("prepared protocol or executable source drifted")
    source = output / "source"
    if sha256_file(source / "runner.py") != sha256_file(Path(__file__).resolve()) or sha256_file(source / "count_static_ridge.py") != PINS[CORE]:
        raise RpeRidgeError("prepared source drifted")
    started = time.perf_counter()
    roster, normalizers, control, moments, features = load_inputs()
    core = load_module(CORE, "rpe_execute_count_ridge")
    genes = roster["fitting_action_ids"].astype(str)
    with threadpool_limits(2):
        anchor = core.control_anchor(control["basal_rate"], moments["gem_cell_count"])
        truth = core.response_from_cp10k_moments(moments["cp10k_sum"], moments["cell_count"])
        residual = truth - anchor
        selected, cv_score, folds = core.choose_alpha(genes, features, residual, seed=731)
        state = core.fit_state(features, residual)
    elapsed = time.perf_counter() - started
    if elapsed > SECONDS:
        raise TimeoutError("RPE1 ridge fit exceeded 600 seconds")
    if (
        not np.array_equal(state["feature_mean"], normalizers["rpe1_feature_mean"])
        or not np.array_equal(state["feature_sd"], normalizers["rpe1_feature_sd"])
        or not np.array_equal(state["feature_scale"], normalizers["rpe1_feature_scale"])
    ):
        raise RpeRidgeError("final ridge normalizer differs from the frozen RPE1 stats")
    model = {
        **state,
        "schema": np.asarray("slp.rpe1-essential-count-anchored-static-ridge/v1"),
        "selected_alpha": np.asarray(selected),
        "query_ids": roster["query_ids"].astype(str),
        "gem_group": np.asarray(control["gem_group"], dtype=np.int16),
        "basal_rate": np.asarray(control["basal_rate"], dtype=np.float32),
        "static_sha256": np.asarray(PINS[STATIC]),
        "roster_sha256": np.asarray(PINS[ROSTER]),
        "normalizers_sha256": np.asarray(PINS[NORMALIZERS]),
        "control_sha256": np.asarray(PINS[CONTROL]),
        "fitting_moments_sha256": np.asarray(PINS[MOMENTS]),
        "core_sha256": np.asarray(PINS[CORE]),
    }
    model_path = output / "model.npz"
    model_path.write_bytes(deterministic_npz(model))
    probe_features = features[:2]
    probe_gems = np.asarray(moments["gem_cell_count"][:2], dtype=np.int64)
    expected_residual = core.predict_residual(state, probe_features, selected)
    expected_anchor = core.control_anchor(control["basal_rate"], probe_gems)
    probe = {
        "feature_values": probe_features,
        "gem_cell_count": probe_gems,
        "expected_residual": expected_residual,
        "expected_anchor": expected_anchor,
        "expected_absolute": core.absolute_prediction(expected_anchor, expected_residual),
    }
    probe_path = output / "target-free-probe.npz"
    probe_path.write_bytes(deterministic_npz(probe))
    mean_residual = residual.mean(0, dtype=np.float64)
    report = {
        "schema": "slp.rpe1-essential-count-anchored-static-ridge-fitting-report/v1",
        "selectedAlpha": selected,
        "crossValidationRawAllQueryMse": cv_score,
        "folds": folds,
        "pureControlAnchorFittingMseDescriptive": float(np.mean(np.square(residual))),
        "anchoredMeanFittingMseDescriptive": float(np.mean(np.square(residual - mean_residual))),
        "finalFittingGenes": len(genes), "finalFittingCells": int(np.asarray(moments["cell_count"]).sum()),
        "queries": QUERY_COUNT, "gemGroups": GEM_COUNT, "elapsedSeconds": elapsed,
        "developmentScored": False, "reconstructionHeldRead": False, "testAccess": False,
    }
    report_path = output / "fitting-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    hashes = {
        "model.npz": sha256_file(model_path), "target-free-probe.npz": sha256_file(probe_path),
        "fitting-report.json": sha256_file(report_path), "source/runner.py": sha256_file(source / "runner.py"),
        "source/count_static_ridge.py": sha256_file(source / "count_static_ridge.py"),
    }
    artifact = {"protocolSha256": sha256_file(output / "protocol.json"), "hashes": hashes}
    artifact_path = output / "artifact-manifest-before-development.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    replay = subprocess.run(
        [sys.executable, str(source / "runner.py"), "--verify-artifact", str(output)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    freeze = {
        "schema": "slp.rpe1-essential-count-anchored-static-ridge-freeze/v1",
        "protocolSha256": artifact["protocolSha256"],
        "artifactManifestSha256": sha256_file(artifact_path),
        "modelSha256": hashes["model.npz"],
        "freshSavedSourceReplay": json.loads(replay.stdout),
        "developmentValidationOpened": False, "reconstructionHeldOpened": False, "testOpened": False,
    }
    (output / "FROZEN-BEFORE-DEVELOPMENT.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    return {**freeze, "fittingReport": report}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=OUTPUT)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--verify-artifact", type=Path)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    selected = sum((args.prepare, args.execute, args.verify_artifact is not None))
    if selected != 1:
        raise SystemExit("select exactly one of --prepare, --execute, --verify-artifact")
    if args.prepare:
        value = prepare(args.output_dir)
    elif args.execute:
        value = execute(args.output_dir)
    else:
        value = verify_artifact(args.verify_artifact)
    print(json.dumps(value, indent=2, sort_keys=True))
