#!/usr/bin/env python3
"""Verify an exported OMF2 rank-32 model through its standalone CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "data/derived/slp11-omf2-response-v1/development"
REFERENCE = ROOT / "results/slp11-transition/human-essential-count-response-rank32-seed731-v1"
CONTEXTS = ("k562", "rpe1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def model_directory(export: Path) -> Path:
    candidate = export.resolve() / "artifacts/model"
    if not candidate.is_dir():
        raise FileNotFoundError(f"exported model directory is absent: {candidate}")
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    # Resolving a venv executable's symlink would select the base interpreter.
    child_environment = {**os.environ, "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1"}

    if args.output.exists():
        raise FileExistsError("verification output must be new")
    args.output.mkdir(parents=True)
    model = model_directory(args.export)
    inference = model / "inference.py"
    manifest_path = model / "manifest.json"
    if not inference.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("export lacks inference.py or manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    environment_probe = subprocess.run(
        [str(args.python.absolute()), "-c", (
            "import importlib.util,json,os,sys,numpy; "
            "print(json.dumps({'executable':sys.executable,'python':sys.version,"
            "'numpy':numpy.__version__,'cwd':os.getcwd(),"
            "'omfInstalled':importlib.util.find_spec('omf') is not None}))"
        )], check=True, text=True,
        capture_output=True, cwd=args.output,
        env=child_environment,
    )
    report = {
        "schema": "slp.omf2-response-export-verification/v1",
        "python": str(args.python.absolute()),
        "childEnvironment": json.loads(environment_probe.stdout),
        "inferenceSha256": sha256(inference),
        "manifestSha256": sha256(manifest_path),
        "contexts": {},
    }
    for context in CONTEXTS:
        prepared_path = PREPARED / f"{context}.npz"
        reference_path = REFERENCE / f"development-forecast-{context}.npz"
        with np.load(prepared_path, allow_pickle=False) as archive:
            prepared = {name: np.asarray(archive[name]) for name in
                        ('gene_ids', 'features', 'control_prediction')}
        reference = load_npz(reference_path)
        if not np.array_equal(prepared["gene_ids"].astype(str), reference["gene_ids"].astype(str)):
            raise ValueError(f"{context} prepared/reference intervention identity mismatch")
        request_path = args.output / f"request-{context}.npz"
        result_path = args.output / f"prediction-{context}.npz"
        np.savez_compressed(
            request_path,
            features=prepared["features"],
            basal_anchor=prepared["control_prediction"],
        )
        subprocess.run(
            [
                str(args.python.absolute()), str(inference), "--model", str(model),
                "--context", context, "--input", str(request_path),
                "--output", str(result_path),
            ],
            check=True, cwd=args.output, env=child_environment,
        )
        actual = load_npz(result_path)
        expected = np.asarray(reference["rank32_prediction"], np.float64)
        predicted = np.asarray(actual["predictions"], np.float64)
        query_identity = bool(np.array_equal(
            actual["query_ids"].astype(str), reference["query_ids"].astype(str)
        ))
        shape_identity = bool(predicted.shape == expected.shape)
        finite = bool(np.isfinite(predicted).all())
        maximum_error = (
            float(np.max(np.abs(predicted - expected)))
            if shape_identity and predicted.size else float("inf")
        )
        model_name = manifest["contexts"][context]["model"]
        model_path = model / model_name
        report["contexts"][context] = {
            "predictionShape": list(predicted.shape),
            "referenceShape": list(expected.shape),
            "queryIdentity": query_identity,
            "shapeIdentity": shape_identity,
            "finite": finite,
            "maximumAbsoluteError": maximum_error,
            "modelSha256": sha256(model_path),
            "manifestModelSha256": manifest["contexts"][context]["sha256"],
            "referenceForecastSha256": sha256(reference_path),
        }
        if (
            not query_identity or not shape_identity or not finite
            or maximum_error > 1e-6
            or report["contexts"][context]["modelSha256"]
            != report["contexts"][context]["manifestModelSha256"]
        ):
            raise RuntimeError(f"{context} exported inference verification failed")

    report["passed"] = True
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
