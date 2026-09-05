#!/usr/bin/env python3
"""Build a compact local research bundle for frozen rank32 response models."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "results/slp11-transition/human-essential-count-response-rank32-seed731-v1"
OUTPUT = PARENT / "local-research-inference-v1"
INFERENCE = ROOT / "modules/slp-1-1-reduced-rank-response-inference-v1/inference.py"
RESPONSE = ROOT / "modules/slp-1-1-reduced-rank-response-v1/response_model.py"
REGISTRY_DIR = ROOT / "data/derived/slp11-human-essential-joint-training-registry-v1"
SHARED = ROOT / "results/slp11-transition/human-essential-count-shared-context-seed731-v1"
PATHS = {
    "k562": {
        "model": PARENT / "model-k562.npz",
        "reference": SHARED / "reference-k562.npz",
        "forecast": PARENT / "development-forecast-k562.npz",
        "index": REGISTRY_DIR / "k562-index.npz",
    },
    "rpe1": {
        "model": PARENT / "model-rpe1.npz",
        "reference": SHARED / "reference-rpe1.npz",
        "forecast": PARENT / "development-forecast-rpe1.npz",
        "index": REGISTRY_DIR / "rpe1-index.npz",
    },
}
STATIC = REGISTRY_DIR / "shared-static577.npz"
REGISTRY = REGISTRY_DIR / "registry.json"
PINS = {
    PATHS["k562"]["model"]: "6267584a4a69dc30899b18d0c9660e0c73d2b8383a1e4911571295a1ea57ae44",
    PATHS["rpe1"]["model"]: "ff864e96d02fb81b64baadc36c164de61a01d9e7d31a2609f78b64d48107be70",
    PATHS["k562"]["reference"]: "63cf7d7b6ce807cbc5b7383ecb0466affccb4a54a1be69bc0ddc1b5ef93413ec",
    PATHS["rpe1"]["reference"]: "2d7c594cac20ccad80c1e78fc35960aa3350d4f837cdf5dec379fd26e78c8bd0",
    PATHS["k562"]["forecast"]: "53d147035d3f04569ea7ca7a9956ebf9704f3cefe6ed30caeda0a6fa513b04b1",
    PATHS["rpe1"]["forecast"]: "8f7819fc5e45744fe73ea971e2745404ddba1a86b015fa2354ea6fe692afe58c",
    PATHS["k562"]["index"]: "d1e79ec3c7c15a8142e3eaff6e1f5854fc8cd890af594f7c18d6dbe7772aeef1",
    PATHS["rpe1"]["index"]: "261b00867092fd228921f9f7069ddc32102ff422edecd7aaf0f02145cfa83878",
    STATIC: "e29d2f32e143cc95c89e8ad636f75de1fe18f8ccdb2e85fa0225405b1015cb7f",
    REGISTRY: "4de798e53a4d8149c200088e054caa4c9b71ecea91e6c00c68ecd3a6c938127c",
    RESPONSE: "da5989fef73891a2fe79a5802858bc583721e3c7e8f9c72e5fde971df3eb92bb",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def protocol():
    pins = dict(PINS)
    pins[INFERENCE] = sha256(INFERENCE)
    return {
        "schema": "slp.rank32-local-research-inference-build-protocol/v1",
        "purpose": "Compact local research inference for unchanged passing rank32 native-panel response weights; no upload, release, or OMF portability claim.",
        "prediction": "Require caller-supplied weights over one exact native GEM axis; mix frozen positive control rates in CP10k units, apply log1p, then add the signed fitted residual without clipping or renormalization.",
        "features": "Numerical API accepts raw static577 action features. Convenience stable-ID lookup uses a frozen union cache of source action rosters and does not add learned IDs.",
        "scope": "K562 and RPE1 native measured panels only. No default GEM mixture, new-context inference, unmeasured-query inference, count generation, or dynamics claim.",
        "verification": "Chosen metadata-only development feature/GEM rows must reproduce the already frozen forecasts; an isolated subprocess is launched from a temporary working directory without training-corpus paths.",
        "pins": {str(path.relative_to(ROOT)): value for path, value in pins.items()},
        "builderSha256": sha256(Path(__file__).resolve()),
    }


def prepare(output=OUTPUT):
    for path, expected in PINS.items():
        if sha256(path) != expected:
            raise ValueError(f"frozen input changed: {path}")
    output.mkdir(parents=True, exist_ok=True)
    value = protocol()
    path = output / "BUILD-PROTOCOL.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("local inference build protocol changed")
    else:
        write_json(path, value)
    return value


def static_cache(output):
    ids = []
    for source in ("k562", "rpe1"):
        index = load_npz(PATHS[source]["index"])
        ids.extend(index["action_ids"].astype(str).tolist())
    action_ids = np.asarray(sorted(set(ids)))
    static = load_npz(STATIC)
    lookup = {value: row for row, value in enumerate(static["entity_id"].astype(str))}
    features = np.asarray(
        static["feature_values"][[lookup[value] for value in action_ids]], np.float32
    )
    path = output / "static-actions.npz"
    np.savez_compressed(
        path,
        schema=np.asarray("slp.rank32-local-static-action-cache/v1"),
        entity_taxon=np.full(len(action_ids), 9606, np.int64),
        entity_id=action_ids,
        feature_values=features,
    )
    return path


def reference(output, source):
    source_values = load_npz(PATHS[source]["reference"])
    index = load_npz(PATHS[source]["index"])
    if not np.array_equal(source_values["query_ids"], index["query_ids"]):
        raise ValueError("reference query order differs from source index")
    path = output / f"reference-{source}.npz"
    np.savez_compressed(
        path,
        schema=np.asarray("slp.rank32-local-native-control-reference/v1"),
        source_id=np.asarray(source),
        query_ids=source_values["query_ids"],
        context_ids=source_values["context_ids"],
        gem_group_ids=index["gem_group"],
        basal_rate=source_values["basal_rate"],
    )
    return path


def manifest(output):
    files = [
        output / "model-k562.npz",
        output / "model-rpe1.npz",
        output / "reference-k562.npz",
        output / "reference-rpe1.npz",
        output / "static-actions.npz",
        output / "source/response_model.py",
        output / "source/inference.py",
        output / "README.txt",
    ]
    value = {
        "schema": "slp.rank32-local-research-inference-bundle/v1",
        "buildProtocolSha256": sha256(output / "BUILD-PROTOCOL.json"),
        "sources": {
            source: {
                "model": f"model-{source}.npz",
                "reference": f"reference-{source}.npz",
            }
            for source in ("k562", "rpe1")
        },
        "sha256": {
            str(path.relative_to(output)).replace("\\", "/"): sha256(path)
            for path in files
        },
        "localResearchOnly": True,
        "omfPortableOrRelease": False,
        "trainingCorpusRequiredAtInference": False,
    }
    write_json(output / "manifest.json", value)
    return value


def verify(output, module):
    arrays = {}
    results = {}
    for source in ("k562", "rpe1"):
        forecast = load_npz(PATHS[source]["forecast"])
        genes = forecast["gene_ids"].astype(str)[:7]
        weights = forecast["gem_cell_count"][:7].astype(np.float64)
        before = weights.copy()
        predictor = module.ResearchPredictor(output, source)
        actual = predictor.predict_genes(genes, weights)
        expected = forecast["rank32_prediction"][:7]
        difference = float(np.max(np.abs(actual["mean_log1p_cp10k"] - expected)))
        if difference > 1e-12 or not np.array_equal(weights, before):
            raise RuntimeError(f"{source} frozen forecast replay failed")
        arrays[f"{source}_gene_ids"] = genes
        arrays[f"{source}_gem_weights"] = weights
        arrays[f"{source}_expected"] = expected
        results[source] = {
            "rows": len(genes),
            "queries": expected.shape[1],
            "maximumAbsoluteDifference": difference,
            "callerWeightsUnchanged": True,
        }
    probe = output / "frozen-forecast-replay.npz"
    np.savez_compressed(probe, **arrays)
    k = module.ResearchPredictor(output, "k562")
    weights = ",".join(["1"] + ["0"] * (len(k.gem_group_ids) - 1))
    with tempfile.TemporaryDirectory(prefix="slp-rank32-isolated-") as directory:
        process = subprocess.run(
            [
                sys.executable,
                str((output / "source/inference.py").resolve()),
                "--bundle",
                str(output.resolve()),
                "--source",
                "k562",
                "--gene",
                str(k.action_ids[0]),
                "--gem-weights",
                weights,
                "--query-index",
                "0",
            ],
            cwd=directory,
            capture_output=True,
            check=False,
            text=True,
            timeout=120,
            env={**os.environ, "PYTHONPATH": ""},
        )
    if process.returncode:
        raise RuntimeError(f"isolated inference failed: {process.stderr}")
    isolated = json.loads(process.stdout)
    return results, {
        "path": probe.name,
        "sha256": sha256(probe),
        "isolatedWorkingDirectoryOutsideWorkspace": True,
        "isolatedQueryIds": isolated["queryIds"],
    }


def build(output=OUTPUT):
    prepare(output)
    if (output / "COMPLETE.json").exists():
        raise FileExistsError("immutable local inference bundle is complete")
    (output / "source").mkdir(exist_ok=True)
    shutil.copyfile(RESPONSE, output / "source/response_model.py")
    shutil.copyfile(INFERENCE, output / "source/inference.py")
    for source in ("k562", "rpe1"):
        shutil.copyfile(PATHS[source]["model"], output / f"model-{source}.npz")
        reference(output, source)
    static_cache(output)
    (output / "README.txt").write_text(
        "Local research-only rank32 molecular inference.\n"
        "Run: python source/inference.py --help\n"
        "Example: python source/inference.py --bundle . --source k562 --gene ENSG... "
        "--gem-weights <48 comma-separated weights> --query-index 0\n"
        "Weights are required in gem_group_ids order from reference-k562.npz or reference-rpe1.npz.\n"
        "Output is a signed log1p molecular profile, not generated counts or an unseen-context prediction.\n",
        encoding="utf-8",
    )
    manifest(output)
    module = load_module(output / "source/inference.py", "rank32_local_bundle_inference")
    replay, isolated = verify(output, module)
    write_json(output / "verification.json", {"forecastReplay": replay, "isolated": isolated})
    complete = {
        "schema": "slp.rank32-local-research-inference-complete/v1",
        "manifestSha256": sha256(output / "manifest.json"),
        "verificationSha256": sha256(output / "verification.json"),
        "forecastReplaySha256": isolated["sha256"],
        "unchangedModelSha256": {
            source: sha256(output / f"model-{source}.npz")
            for source in ("k562", "rpe1")
        },
        "localResearchOnly": True,
        "uploaded": False,
        "releaseClaim": False,
    }
    write_json(output / "COMPLETE.json", complete)
    print(json.dumps(complete))
    return complete


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "build"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    {"prepare": prepare, "build": build}[args.mode](args.output)
