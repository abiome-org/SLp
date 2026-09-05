#!/usr/bin/env python3
"""Freeze a fitting-only control-anchored static ridge count baseline."""

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
SOURCE_SHA256 = "3e5a63a9e892b21029bb55fca4e12517a49aad7af6c14133ca63d12cf68c6cee"
ROUTING_SHA256 = "47c89c5082c0a9d4008c6b567407c530933a36fb7603621c37cbe913143f15ad"
STATIC_SHA256 = "6706f8867adedef8822897bc275ea90680584f84afd24771e4beb3c8ecf07659"
ROSTER_SHA256 = "f2ee702a0714ca7f11f4fd2aa96f4c1825617c0e4f2bcdac42135cd0ba938d7b"
SECONDS = 600.0


class CountRidgeRunError(ValueError):
    """Raised when the fitting-only baseline contract fails."""


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CountRidgeRunError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
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
        raise CountRidgeRunError(f"missing scalar {key}")
    return str(values[key].item())


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def load_inputs(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], ...]:
    expected = {
        args.static: STATIC_SHA256,
        args.roster: ROSTER_SHA256,
        args.control: args.control_sha256,
        args.fitting_moments: args.fitting_moments_sha256,
    }
    for path, digest in expected.items():
        actual = sha256_file(path)
        if actual != digest:
            raise CountRidgeRunError(f"input hash mismatch for {path}: {actual}")
    static, roster, control, moments = map(
        load_npz, (args.static, args.roster, args.control, args.fitting_moments)
    )
    if scalar_text(control, "source_sha256") != SOURCE_SHA256 or scalar_text(
        control, "routing_sha256"
    ) != ROUTING_SHA256:
        raise CountRidgeRunError("control reference lineage mismatch")
    if scalar_text(moments, "source_sha256") != SOURCE_SHA256 or scalar_text(
        moments, "routing_sha256"
    ) != ROUTING_SHA256:
        raise CountRidgeRunError("fitting moment lineage mismatch")
    query_ids = roster["query_ids"].astype(str)
    fitting_ids = roster["fitting_action_ids"].astype(str)
    if (
        not np.array_equal(query_ids, control["query_ids"].astype(str))
        or not np.array_equal(query_ids, moments["query_ids"].astype(str))
        or not np.array_equal(fitting_ids, moments["action_ids"].astype(str))
        or len(query_ids) != 8563
        or len(fitting_ids) != 1443
    ):
        raise CountRidgeRunError("query/action moment alignment mismatch")
    if not np.array_equal(control["gem_group"], moments["gem_group"]):
        raise CountRidgeRunError("GEM alignment mismatch")
    gem_cells = np.asarray(moments["gem_cell_count"], dtype=np.int64)
    cell_count = np.asarray(moments["cell_count"], dtype=np.int64)
    cp10k_sum = np.asarray(moments["cp10k_sum"], dtype=np.float64)
    if (
        gem_cells.shape != (1443, 48)
        or cell_count.shape != (1443,)
        or cp10k_sum.shape != (1443, 8563)
        or not np.array_equal(gem_cells.sum(1), cell_count)
        or np.any(cell_count <= 0)
        or not np.isfinite(cp10k_sum).all()
        or np.any(cp10k_sum < 0)
    ):
        raise CountRidgeRunError("invalid fitting sufficient statistics")
    entity_ids = static["entity_id"].astype(str)
    entity_index = {entity: index for index, entity in enumerate(entity_ids)}
    expected_indices = np.asarray([entity_index[item] for item in fitting_ids], np.int64)
    if not np.array_equal(expected_indices, roster["fitting_action_entity_index"]):
        raise CountRidgeRunError("static fitting row index mismatch")
    features = np.asarray(static["feature_values"][expected_indices], dtype=np.float32)
    if features.shape != (1443, 577) or not np.isfinite(features).all():
        raise CountRidgeRunError("invalid fitting static features")
    return static, roster, control, moments, features


def frozen_protocol(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema": "slp.k562-essential-count-anchored-static-ridge-protocol/v1",
        "purpose": "A matched aggregate-mean baseline for the experimental K562 raw-count latent model; no single-cell-generation claim.",
        "hypothesis": "Static intervention descriptors predict held-gene residual molecular means beyond a GEM-composition-matched control anchor.",
        "endpoint": "For each fitting gene, y=ln1p(sum over reconstruction-training cells of per-cell raw CP10k divided by gene cell count), on all 8,563 source queries.",
        "anchor": "a=ln1p(sum_gem((gene cells in gem)/(gene cells total) * reconstruction-training NT smoothed control CP10k rate in gem)).",
        "model": "Fit y-a from raw static577 with an unpenalized residual intercept and ridge-penalized feature coefficients.",
        "selection": {
            "folds": "three global gene folds from SHA256(slp11-bp-ridge-v1|731|global-inner-fold|9606|ENSG)",
            "candidates": ["0.1", "1", "10", "100", "1000", "10000", "100000", "1e+06", "mean-limit"],
            "criterion": "equal-gene mean raw squared error across every one of 8,563 queries",
            "featureNormalization": "population mean/SD on unique inner-fitting genes only; SD<=1e-5 maps to scale1; final transform refit on all 1,443 fitting genes",
        },
        "comparators": {
            "pureControlAnchor": "zero residual",
            "anchoredMean": "inner-fitting residual mean / final fitting residual mean",
            "staticRidge": "selected candidate, including prespecified mean limit",
        },
        "developmentBoundary": "Fit/CV/model freeze uses only reconstruction-training cells from 1,443 fitting interventions and reconstruction-training NT control moments. Development validation moments are not opened until this artifact and the neural model are frozen. Test cells remain unopened.",
        "futureScore": "Primary raw log-space gene-profile MSE. Landscape correlation subtracts the same GEM-matched anchor from truth and prediction, then independently centers held-gene query columns with a first-row numerical anchor.",
        "inputs": {
            "static": {"path": args.static.as_posix(), "sha256": STATIC_SHA256},
            "roster": {"path": args.roster.as_posix(), "sha256": ROSTER_SHA256},
            "control": {"path": args.control.as_posix(), "sha256": args.control_sha256},
            "fittingMoments": {
                "path": args.fitting_moments.as_posix(),
                "sha256": args.fitting_moments_sha256,
            },
        },
        "source": {
            "runnerSha256": sha256_file(Path(__file__).resolve()),
            "coreSha256": sha256_file(CORE),
        },
        "limits": {"cpuThreads": 2, "wallSeconds": SECONDS},
        "outcomeAccess": {
            "fittingCountMoments": True,
            "developmentValidation": False,
            "test": False,
            "syntheticLethality": False,
        },
    }


def verify_artifact(output: Path) -> dict[str, object]:
    manifest = json.loads((output / "artifact-manifest-before-development.json").read_text())
    for relative, expected in manifest["hashes"].items():
        actual = sha256_file(output / relative)
        if actual != expected:
            raise CountRidgeRunError(f"saved artifact hash drift: {relative}")
    core = load_module(output / "source/count_static_ridge.py", "_saved_count_static_ridge")
    model = load_npz(output / "model.npz")
    probe = load_npz(output / "target-free-probe.npz")
    residual = core.predict_residual(
        model, probe["feature_values"], str(model["selected_alpha"].item())
    )
    anchor = core.control_anchor(model["basal_rate"], probe["gem_cell_count"])
    absolute = core.absolute_prediction(anchor, residual)
    maximum = max(
        float(np.max(np.abs(residual - probe["expected_residual"]))),
        float(np.max(np.abs(anchor - probe["expected_anchor"]))),
        float(np.max(np.abs(absolute - probe["expected_absolute"]))),
    )
    result = {"maximumAbsoluteDifference": maximum, "tolerance": 1e-10, "passes": maximum <= 1e-10}
    if not result["passes"]:
        raise CountRidgeRunError("fresh saved-source replay failed")
    return result


def execute(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_dir = args.output_dir / "source"
    source_dir.mkdir()
    shutil.copy2(Path(__file__).resolve(), source_dir / "runner.py")
    shutil.copy2(CORE, source_dir / "count_static_ridge.py")
    protocol = frozen_protocol(args)
    protocol_path = args.output_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    _, roster, control, moments, features = load_inputs(args)
    core = load_module(CORE, "_execute_count_static_ridge")
    genes = roster["fitting_action_ids"].astype(str)
    with threadpool_limits(2):
        anchor = core.control_anchor(control["basal_rate"], moments["gem_cell_count"])
        truth = core.response_from_cp10k_moments(
            moments["cp10k_sum"], moments["cell_count"]
        )
        residual = truth - anchor
        selected, cv_score, fold_reports = core.choose_alpha(
            genes, features, residual, seed=731
        )
        state = core.fit_state(features, residual)
    elapsed = time.perf_counter() - started
    if elapsed > SECONDS:
        raise TimeoutError(f"fitting exceeded frozen {SECONDS:g}s cap")
    model = {
        **state,
        "schema": np.asarray("slp.k562-essential-count-anchored-static-ridge/v1"),
        "selected_alpha": np.asarray(selected),
        "query_ids": roster["query_ids"].astype(str),
        "gem_group": np.asarray(control["gem_group"], dtype=np.int16),
        "basal_rate": np.asarray(control["basal_rate"], dtype=np.float32),
        "static_sha256": np.asarray(STATIC_SHA256),
        "control_sha256": np.asarray(args.control_sha256),
        "fitting_moments_sha256": np.asarray(args.fitting_moments_sha256),
        "core_sha256": np.asarray(sha256_file(CORE)),
    }
    model_path = args.output_dir / "model.npz"
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
    probe_path = args.output_dir / "target-free-probe.npz"
    probe_path.write_bytes(deterministic_npz(probe))
    fitting_report = {
        "schema": "slp.k562-essential-count-anchored-static-ridge-fitting-report/v1",
        "selectedAlpha": selected,
        "crossValidationRawAllQueryMse": cv_score,
        "folds": fold_reports,
        "pureControlAnchorFittingMseDescriptive": float(np.mean(np.square(residual))),
        "finalFittingGenes": len(genes),
        "finalFittingCells": int(np.asarray(moments["cell_count"]).sum()),
        "queries": residual.shape[1],
        "elapsedSeconds": elapsed,
        "developmentScored": False,
        "testAccess": False,
    }
    report_path = args.output_dir / "fitting-report.json"
    report_path.write_text(json.dumps(fitting_report, indent=2, sort_keys=True) + "\n")
    hashes = {
        "model.npz": sha256_file(model_path),
        "target-free-probe.npz": sha256_file(probe_path),
        "fitting-report.json": sha256_file(report_path),
        "source/runner.py": sha256_file(source_dir / "runner.py"),
        "source/count_static_ridge.py": sha256_file(source_dir / "count_static_ridge.py"),
    }
    artifact_manifest = {"protocolSha256": sha256_file(protocol_path), "hashes": hashes}
    artifact_path = args.output_dir / "artifact-manifest-before-development.json"
    artifact_path.write_text(json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n")
    replay = subprocess.run(
        [sys.executable, str(source_dir / "runner.py"), "--verify-artifact", str(args.output_dir)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    replay_value = json.loads(replay.stdout)
    freeze = {
        "schema": "slp.k562-essential-count-anchored-static-ridge-freeze/v1",
        "protocolSha256": sha256_file(protocol_path),
        "artifactManifestSha256": sha256_file(artifact_path),
        "modelSha256": hashes["model.npz"],
        "freshSavedSourceReplay": replay_value,
        "developmentValidationOpened": False,
        "testOpened": False,
    }
    freeze_path = args.output_dir / "FROZEN-BEFORE-DEVELOPMENT.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    return {**freeze, "fittingReport": fitting_report}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--static", type=Path, default=ROOT / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1/k562-essential-count-static577.npz")
    result.add_argument("--roster", type=Path, default=ROOT / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1/roster-index.npz")
    result.add_argument("--control", type=Path)
    result.add_argument("--control-sha256")
    result.add_argument("--fitting-moments", type=Path)
    result.add_argument("--fitting-moments-sha256")
    result.add_argument("--output-dir", type=Path, default=ROOT / "results/slp11-transition/k562-essential-count-anchored-static-ridge-seed731-v1")
    result.add_argument("--verify-artifact", type=Path)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.verify_artifact is not None:
        print(json.dumps(verify_artifact(arguments.verify_artifact), sort_keys=True))
    else:
        required = (arguments.control, arguments.control_sha256, arguments.fitting_moments, arguments.fitting_moments_sha256)
        if any(value is None for value in required):
            raise SystemExit("control and fitting moment paths/hashes are required")
        print(json.dumps(execute(arguments), indent=2, sort_keys=True))
