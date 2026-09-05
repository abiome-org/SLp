#!/usr/bin/env python3
"""Run fixed SLIM bilinear comparators on SLp's retained native panels."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules/slp-1-1-slim-baseline-v1/slim_native.py"
DATA = ROOT / "data/derived/slp11-omf2-response-v1"
OUTPUT = ROOT / "results/slp11-transition/slim-native-v1"
UPSTREAM = ROOT / "data/tooling/slim-5a7e9ade"
RANK32_REPORT = ROOT / "results/slp11-transition/human-essential-count-response-rank32-seed731-v1/report.json"
ARMS = {"publishedDefaultK10": 10, "developmentK32": 32}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("slp_slim_native", MODULE)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def score(truth, prediction, anchor):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    anchor = np.asarray(anchor, np.float64)
    mse_by_gene = np.square(truth - prediction).mean(axis=1)
    x = truth - anchor
    z = prediction - anchor
    x = x - x.mean(axis=0, keepdims=True)
    z = z - z.mean(axis=0, keepdims=True)
    x = x - x.mean(axis=1, keepdims=True)
    z = z - z.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(z, axis=1)
    valid = denominator > 1e-12
    correlations = np.full(len(x), np.nan)
    correlations[valid] = np.sum(x[valid] * z[valid], axis=1) / denominator[valid]
    return {
        "geneProfileMse": float(mse_by_gene.mean()),
        "independentlyQueryCenteredResidualPearson": float(np.nanmean(correlations)),
        "finiteCorrelationGenes": int(valid.sum()),
    }, mse_by_gene, correlations


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def save_model(path: Path, model):
    np.savez_compressed(
        path,
        schema=np.asarray("slp.slim-native-model/v1"),
        query_basis=model.query_basis,
        weight=model.weight,
        bias=model.bias,
        rank=np.asarray(model.rank),
        lambda_reg=np.asarray(model.lambda_reg),
    )


def main(output: Path):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    module = load_module()
    protocol = {
        "schema": "slp.slim-native-protocol/v1",
        "hypothesis": "Published-default SLIM bilinear factorization using the same static577 intervention descriptors improves intervention-cold development prediction beyond static ridge and approaches the retained rank32 response model in both K562 and RPE1.",
        "primary": {"rank": 10, "lambdaReg": 0.1, "basis": "PCA", "bias": "training perturbation mean"},
        "developmentDiagnostic": {"rank": 32, "lambdaReg": 0.1, "selectionUse": False},
        "featureTransform": "None; raw static577 rows replace SLIM's published STRING embeddings.",
        "endpointAdaptation": "SLp control-anchored residual targets; add each held gene's precomputed control anchor for molecular scoring.",
        "split": "Existing seed731 disjoint fitting/development intervention rosters; protected tests unopened.",
        "upstream": {"repository": "https://github.com/RasmussenLab/SLIM.git", "revision": "5a7e9ade5d0a6b6331e6dbc81181450605047bcc"},
        "pins": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in [
                MODULE,
                UPSTREAM / "src/slim/model.py",
                UPSTREAM / "src/slim/basis.py",
                UPSTREAM / "src/slim/bias.py",
                DATA / "training/k562.npz",
                DATA / "training/rpe1.npz",
                DATA / "development/k562.npz",
                DATA / "development/rpe1.npz",
                RANK32_REPORT,
            ]
        },
        "benchmarkAccessed": False,
        "protectedTestOpened": False,
    }
    write_json(output / "protocol.json", protocol)
    fitted = {}
    model_receipts = {}
    for source in ("k562", "rpe1"):
        train = load_npz(DATA / f"training/{source}.npz")
        fitted[source] = {}
        model_receipts[source] = {}
        for arm, rank in ARMS.items():
            model = module.fit(train["features"], train["residual_targets"], rank=rank, lambda_reg=0.1)
            path = output / f"model-{source}-{arm}.npz"
            save_model(path, model)
            fitted[source][arm] = model
            model_receipts[source][arm] = {"path": path.name, "sha256": sha256(path), "rank": model.rank}
    freeze = {
        "schema": "slp.slim-native-model-freeze/v1",
        "protocolSha256": sha256(output / "protocol.json"),
        "models": model_receipts,
        "developmentOpened": False,
        "protectedTestOpened": False,
    }
    write_json(output / "MODELS-FROZEN-BEFORE-DEVELOPMENT.json", freeze)
    metrics = {}
    per_gene = {}
    rank32 = json.loads(RANK32_REPORT.read_text(encoding="utf-8"))["metrics"]
    for source in ("k562", "rpe1"):
        development = load_npz(DATA / f"development/{source}.npz")
        metrics[source] = {
            "staticRidge": score(development["truth"], development["static_ridge_prediction"], development["control_prediction"])[0],
            "retainedRank32": rank32[source]["rank32_prediction"],
        }
        per_gene[f"{source}_gene_ids"] = development["gene_ids"]
        for arm, model in fitted[source].items():
            residual = model.predict_residual(development["features"])
            prediction = np.maximum(development["control_prediction"] + residual, 0.0)
            summary, mse, correlation = score(development["truth"], prediction, development["control_prediction"])
            metrics[source][arm] = summary
            per_gene[f"{source}_{arm}_mse"] = mse
            per_gene[f"{source}_{arm}_centered_pearson"] = correlation
    np.savez_compressed(output / "development-per-gene-metrics.npz", **per_gene)
    for source in metrics:
        for arm in ARMS:
            current = metrics[source][arm]
            for comparator in ("staticRidge", "retainedRank32"):
                baseline = metrics[source][comparator]
                current[f"mseChangeVs{comparator}"] = current["geneProfileMse"] / baseline["geneProfileMse"] - 1.0
                current[f"centeredPearsonChangeVs{comparator}"] = current["independentlyQueryCenteredResidualPearson"] - baseline["independentlyQueryCenteredResidualPearson"]
    report = {
        "schema": "slp.slim-native-report/v1",
        "protocolSha256": sha256(output / "protocol.json"),
        "modelFreezeSha256": sha256(output / "MODELS-FROZEN-BEFORE-DEVELOPMENT.json"),
        "metrics": metrics,
        "primaryArm": "publishedDefaultK10",
        "developmentDiagnosticArm": "developmentK32",
        "seconds": time.perf_counter() - started,
        "interpretation": "Matched-feature native-panel adaptation of published SLIM algebra, not a reproduction of author-reported canonical GEARS benchmark scores. K32 is descriptive development evidence and does not replace the fixed K10 primary result.",
        "benchmarkAccessed": False,
        "protectedTestOpened": False,
    }
    write_json(output / "report.json", report)
    print(json.dumps(report["metrics"], sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with threadpool_limits(2):
        main(args.output)
