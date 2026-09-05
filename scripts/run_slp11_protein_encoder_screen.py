"""Compare frozen protein encoders on source-development molecular outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules/slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))
from context_transfer_scoring import collapse_gene_profiles, score_gene_profiles
from train import gene_metrics
from transition_baselines import fit_ridge

DATA = ROOT / "data/derived/slp11-human-gwps/complete-panel-v1/development.npz"
DATA_SHA = "006b4bb127a09073a7f409d81a7bccce96bb961879cb5e57dce56b48eb8e664b"
BASE_SHA = "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7"
ARMS = ("esm8m_physical", "esm650m_pca320_physical", "esm650m_full_physical")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def aligned_features(path: Path, actions: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as pack:
        taxa, genes, values = pack["entity_taxon"], pack["entity_id"], pack["feature_values"]
    if values.ndim != 2 or len(taxa) != len(values) or len(genes) != len(values):
        raise ValueError("invalid feature axes")
    keys = [(int(taxon), str(gene)) for taxon, gene in zip(taxa, genes, strict=True)]
    if len(set(keys)) != len(keys) or not np.isfinite(values).all():
        raise ValueError("duplicate feature identities or nonfinite features")
    lookup = dict(zip(keys, range(len(keys)), strict=True))
    return values[[lookup[(9606, str(gene))] for gene in actions]]


def run(manifest_path: Path, output: Path) -> dict:
    started = time.monotonic()
    manifest = json.loads(manifest_path.read_text())
    if set(manifest["arms"]) != set(ARMS) or sha(DATA) != DATA_SHA:
        raise ValueError("encoder arms or source development snapshot changed")
    if manifest["arms"][ARMS[0]]["sha256"] != BASE_SHA:
        raise ValueError("frozen 8M physical comparator changed")
    for details in manifest["arms"].values():
        if sha(Path(details["path"])) != details["sha256"]:
            raise ValueError("feature checksum mismatch")
    with np.load(DATA, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    train, validation = data["split_train"], data["split_validation"]
    if len(data["split_test"]) or set(data["action_ids"][train]) & set(data["action_ids"][validation]):
        raise ValueError("source fitting and evaluation intervention identities overlap")
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    for path in (Path(__file__), *MODULE.glob("*.py")):
        shutil.copyfile(path, source / path.name)
    protocol = {
        "hypothesis": "650M static protein representations improve held-gene molecular point prediction over 8M with unchanged GO and physical graph",
        "primaryArm": ARMS[1],
        "secondaryArm": ARMS[2],
        "primaryInterpretation": "PCA320 matches the previous sequence width; PCA uses source fitting-gene static vectors only",
        "secondaryInterpretation": "full1280 changes representation width as well as encoder, so does not isolate capacity",
        "rule": "primary PCA320 arm reduces gene-macro MSE at least1% and does not regress fitted-centroid-adjusted profile Pearson against 8M in every source context",
        "secondaryCannotRescuePrimaryFailure": True,
        "secondaryCorrelation": "gene-averaged profiles with prediction and truth independently query-centered inside each source validation context; descriptive only",
        "ridgeAlpha": 10000.0,
        "data": {"path": str(DATA), "sha256": DATA_SHA},
        "features": manifest,
        "featureManifestSha256": sha(manifest_path),
        "modalities": "protein sequence, archived GO MF/CC, direct physical neighbors",
        "fitting": "original source development split_train within each context only",
        "evaluation": "source split_validation only; no HepG2, Jurkat, retired confirmation or SL outcomes",
        "likelihoodEvaluated": False,
        "cpuThreads": 2,
        "sourceHashes": {path.name: sha(path) for path in source.glob("*.py")},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    forecasts, results = {}, {str(context): {} for context in data["context_ids"]}
    for arm in ARMS:
        features = aligned_features(Path(manifest["arms"][arm]["path"]), data["action_ids"])
        for index, name in enumerate(data["context_ids"]):
            fitting = train[data["context_index"][train] == index]
            scoring = validation[data["context_index"][validation] == index]
            ridge = fit_ridge(features[fitting], data["targets"][fitting], data["observed"][fitting], 10000.0)
            prediction = ridge.predict(features[scoring])
            metrics = gene_metrics(
                prediction, data["targets"][scoring], data["observed"][scoring],
                [(9606, str(gene)) for gene in data["action_ids"][scoring]], ridge.intercept_,
                np.ones_like(prediction), value_space=str(data["target_value_space"].item()),
            )
            results[str(name)][arm] = {key: metrics[key] for key in (
                "gene_macro_mse", "gene_macro_profile_centroid_adjusted_pearson_mean")}
            profiles = collapse_gene_profiles(
                prediction, data["targets"][scoring], data["observed"][scoring],
                data["action_ids"][scoring], data["record_ids"][scoring])
            strict = score_gene_profiles(profiles, ridge.intercept_)
            results[str(name)][arm]["secondary_independently_centered_profile_pearson"] = strict[
                "primaryIndependentlyCenteredGeneMacroProfilePearson"]
            forecasts[f"context{index}_{arm}"] = prediction.astype(np.float32)
            print(json.dumps({"context": str(name), "arm": arm, "metrics": results[str(name)][arm]}), flush=True)
    for context in results.values():
        base, candidate = context[ARMS[0]], context[ARMS[1]]
        context["primaryPassed"] = bool(
            candidate["gene_macro_mse"] <= .99 * base["gene_macro_mse"]
            and candidate["gene_macro_profile_centroid_adjusted_pearson_mean"] >=
            base["gene_macro_profile_centroid_adjusted_pearson_mean"])
    np.savez_compressed(output / "predictions.npz", **forecasts)
    report = {"results": results, "primaryPassed": all(item["primaryPassed"] for item in results.values()),
              "elapsedSeconds": time.monotonic() - started, "predictionsSha256": sha(output / "predictions.npz"),
              "protocolSha256": sha(output / "protocol.json")}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.feature_manifest, args.output)
