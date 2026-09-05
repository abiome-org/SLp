"""Verify frozen Nyström models by reconstructing development forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = (
        np.square(left).sum(1, dtype=np.float64)[:, None]
        + np.square(right).sum(1, dtype=np.float64)[None, :]
        - 2.0 * (left @ right.T)
    )
    return np.maximum(result, 0.0)


def reload_model(model_path: Path, features: np.ndarray) -> np.ndarray:
    with np.load(model_path, allow_pickle=False) as model:
        standardized = (features - model["feature_mean"]) / model["feature_scale"]
        kernel = np.exp(
            -squared_distances(standardized, model["standardized_landmarks"])
            / (2.0 * float(model["bandwidth"]) ** 2)
        ).astype(np.float32)
        mapped = kernel @ model["kernel_basis"]
        centered = mapped - model["ridge_feature_mean"]
        alpha_text = str(model["selected_alpha"])
        if alpha_text == "mean-limit":
            return np.broadcast_to(model["target_mean"], (len(features), len(model["query_ids"]))).copy()
        alpha = float(alpha_text)
        rotated = centered @ model["ridge_eigenvectors"]
        coefficients = rotated / (model["ridge_eigenvalues"] + alpha)
        return (model["target_mean"] + coefficients @ model["ridge_rhs"]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    inputs = protocol["inputs"]
    for item in inputs.values():
        path = Path(item["path"])
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"input hash mismatch for {path}: {actual}")

    development_path = Path(inputs["development"]["path"])
    feature_path = Path(inputs["features"]["path"])
    with np.load(development_path, allow_pickle=False) as data:
        action_ids = data["action_ids"].astype(str)
        context_index = data["context_index"].astype(np.int64)
        validation_rows = data["split_validation"].astype(np.int64)
        record_ids = data["record_ids"].astype(str)
        query_ids = data["query_ids"].astype(str)
        context_ids = data["context_ids"].astype(str)
    with np.load(feature_path, allow_pickle=False) as pack:
        feature_ids = pack["entity_id"].astype(str)
        feature_taxon = pack["entity_taxon"].astype(np.int64)
        feature_values = pack["feature_values"].astype(np.float32)
    if np.any(feature_taxon != 9606) or len(np.unique(feature_ids)) != len(feature_ids):
        raise RuntimeError("feature identity contract failed")
    feature_index = {gene: index for index, gene in enumerate(feature_ids)}

    with np.load(output / "development-predictions.npz", allow_pickle=False) as saved:
        saved_mean = saved["mean"]
        if not np.array_equal(saved["record_ids"].astype(str), record_ids[validation_rows]):
            raise RuntimeError("record identity mismatch")
        if not np.array_equal(saved["action_ids"].astype(str), action_ids[validation_rows]):
            raise RuntimeError("action identity mismatch")
        if not np.array_equal(saved["context_index"].astype(np.int64), context_index[validation_rows]):
            raise RuntimeError("context identity mismatch")
        if not np.array_equal(saved["query_ids"].astype(str), query_ids):
            raise RuntimeError("query identity mismatch")

    maxima: dict[str, float] = {}
    for context, context_id in enumerate(context_ids):
        local = np.flatnonzero(context_index[validation_rows] == context)
        genes = action_ids[validation_rows[local]]
        matrix = np.stack([feature_values[feature_index[gene]] for gene in genes])
        reloaded = reload_model(output / f"model-context-{context}.npz", matrix)
        difference = float(np.max(np.abs(reloaded - saved_mean[local])))
        maxima[context_id] = difference
        if difference > 1e-4:
            raise RuntimeError(f"reload mismatch for {context_id}: {difference}")

    report = {
        "schema": "slp.nystrom-rbf-three-context-verification/v1",
        "status": "pass",
        "checks": {
            "allPinnedInputHashes": True,
            "featureIdentityTaxonomy9606Unique": True,
            "savedPredictionIdentityExact": True,
            "modelReloadMaximumAbsoluteErrorAtMost1e-4": True,
        },
        "maximumAbsoluteErrorByContext": maxima,
        "protocolSha256": sha256_file(protocol_path),
        "predictionsSha256": sha256_file(output / "development-predictions.npz"),
        "verifierSourceSha256": sha256_file(Path(__file__)),
    }
    verifier_copy = output / "source" / Path(__file__).name
    if verifier_copy.exists() and sha256_file(verifier_copy) != report["verifierSourceSha256"]:
        raise RuntimeError("a different verifier source is already frozen")
    if not verifier_copy.exists():
        shutil.copyfile(Path(__file__), verifier_copy)
    destination = output / "verification.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
