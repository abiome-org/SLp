#!/usr/bin/env python3
"""Fit and evaluate fixed rank-32 native-panel response models."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/slp11-transition/human-essential-count-response-rank32-seed731-v1"
MODEL = ROOT / "modules/slp-1-1-reduced-rank-response-v1/response_model.py"
RIDGE = ROOT / "modules/slp-1-1-count-static-ridge-v1/count_static_ridge.py"
AUDIT = ROOT / "scripts/audit_slp11_count_response_rank.py"
AUDIT_REPORT = ROOT / "results/slp11-transition/human-essential-count-response-rank-audit-v1/report.json"
SHARED = ROOT / "results/slp11-transition/human-essential-count-shared-context-seed731-v1"
EVALUATION = ROOT / "results/slp11-transition/human-essential-count-shared-context-development-evaluation-v2"
EVALUATOR = ROOT / "modules/slp-1-1-count-world-evaluation-v1/evaluator.py"
CONTEXTS = {
    "k562": {
        "moments": ROOT / "data/derived/slp11-human-k562-essential-fitting-action-moments-v1/fitting-action-moments.npz",
        "static": ROOT / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1/k562-essential-count-static577.npz",
        "baseline": ROOT / "results/slp11-transition/k562-essential-count-anchored-static-ridge-seed731-v1/model.npz",
        "metadata_forecast": SHARED / "development-forecasts-k562.npz",
        "truth": EVALUATION / "development-truth-k562.npz",
    },
    "rpe1": {
        "moments": ROOT / "data/derived/slp11-human-rpe1-essential-raw-cells-v1/fitting-action-moments.npz",
        "static": ROOT / "data/derived/slp11-human-rpe1-essential-count-static/ensembl116-esm8m-shared-go-v1/rpe1-essential-count-static577.npz",
        "baseline": ROOT / "results/slp11-transition/rpe1-essential-count-anchored-static-ridge-seed731-v1/model.npz",
        "metadata_forecast": SHARED / "development-forecasts-rpe1.npz",
        "truth": EVALUATION / "development-truth-rpe1.npz",
    },
}
PINS = {
    RIDGE: "1032eeff59382fae3874da9a389033192e113e0f5ac2c8d01f09f8441d969e62",
    AUDIT: "7cb64ed7cb4b657d29bffcca00249567bde8fd44b684427a37ca972bdd016b3f",
    AUDIT_REPORT: "8adcd2b064d327a4d7b8df1ff82027263b535bf6334fe93fea674ec8c906e6b6",
    EVALUATOR: "c4120a8a9c8c768ce518339afee2fcfdabfe6dc925dbeb6a9dd54562c58004c9",
    CONTEXTS["k562"]["moments"]: "a1f44a15a42c5b56e4ce897fde6ebba97298fc296105c6c870ee0e740331694e",
    CONTEXTS["k562"]["static"]: "6706f8867adedef8822897bc275ea90680584f84afd24771e4beb3c8ecf07659",
    CONTEXTS["k562"]["baseline"]: "dbb669d2eb8d844ec9be7c88a2ed21f5592de434d1b2e916412bda4a52fe1cf3",
    CONTEXTS["k562"]["metadata_forecast"]: "12baf98b1de1c9f724e1af34707192f16465faceb310478dc5300339b2a53389",
    CONTEXTS["k562"]["truth"]: "abe9fafc8df755e9a90f8e544ef1737ed799db7332cef864afd08ae4e1c99588",
    CONTEXTS["rpe1"]["moments"]: "d15def86aead06b0bc75ab63c77513735ec7c57d65012bff72f3947bc654895c",
    CONTEXTS["rpe1"]["static"]: "621e1e9f0dffc740ef42382b1b2898f629edd5037e8a02d411e8d30e815ed816",
    CONTEXTS["rpe1"]["baseline"]: "bd144e36b5618c6225828501492edfa5449cef07442041c1d1cc20645b1473bc",
    CONTEXTS["rpe1"]["metadata_forecast"]: "5c3ade1d023a73a024eef1ccbb528a4f92ebc73f18f19cc6e52d35c08424a6c9",
    CONTEXTS["rpe1"]["truth"]: "a8b6df1dd24863a76ba8e7bac740110c81008ab65de91d7671ba59b708f08d93",
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
    pins[MODEL] = sha256(MODEL)
    return {
        "schema": "slp.human-essential-count-response-rank32-protocol/v1",
        "hypothesis": "The fitting-selected rank-32 regularized response map improves held-gene development MSE by at least 1% over full static ridge in both K562 and RPE1 while preserving independently query-centered residual correlation.",
        "advancementRule": "In each source, rank32 MSE must be at least 1% below the frozen full static ridge, independently query-centered anchor-residual profile Pearson must be at least .10 and no lower than ridge.",
        "model": "Separate source-native exact rank-constrained feature-linear ridge maps; rank32 and alpha1000 fixed by the prior fitting-only audit. Intercept is unpenalized.",
        "fit": "All fitting genes separately in K562 and RPE1; raw static577 features; ln1p equal-cell mean CP10k minus fitting-cell-GEM-weighted reconstruction-training NT anchor; no sweep, seed, early stopping, or development feedback.",
        "portableState": "A 32-dimensional action state and native-panel fitted query loading. It has no learned action IDs and makes no unmeasured-query, cell-generative, or identified-dynamics claim.",
        "executionOrder": "Fit and freeze both models, target-free reload, and both metadata-only development forecasts before opening the two already frozen development truth bundles once.",
        "rank": 32,
        "alpha": 1000.0,
        "compute": {"cpuThreads": 2, "maximumSeconds": 600},
        "pins": {str(path.relative_to(ROOT)): value for path, value in pins.items()},
        "runnerSha256": sha256(Path(__file__).resolve()),
        "developmentOpened": False,
        "testOpened": False,
    }


def prepare(output=OUTPUT):
    for path, expected in PINS.items():
        if sha256(path) != expected:
            raise ValueError(f"frozen input changed: {path}")
    output.mkdir(parents=True, exist_ok=True)
    value = protocol()
    path = output / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("frozen rank32 protocol changed")
    else:
        write_json(path, value)
    return value


def raw_features(static, ids):
    entity = static["entity_id"].astype(str)
    lookup = {value: row for row, value in enumerate(entity)}
    try:
        return np.asarray(static["feature_values"][[lookup[value] for value in ids]], np.float32)
    except KeyError as error:
        raise ValueError(f"stable action lacks static feature row: {error}") from error


def fit_and_freeze(output, model_module, ridge):
    models, receipts = {}, {}
    (output / "source").mkdir(exist_ok=True)
    shutil.copyfile(MODEL, output / "source/response_model.py")
    shutil.copyfile(Path(__file__).resolve(), output / "source/runner.py")
    for source, paths in CONTEXTS.items():
        moments = load_npz(paths["moments"])
        static = load_npz(paths["static"])
        baseline = load_npz(paths["baseline"])
        genes = moments["action_ids"].astype(str)
        query = moments["query_ids"].astype(str)
        if not np.array_equal(query, baseline["query_ids"].astype(str)):
            raise ValueError(f"{source} fitting query axis drift")
        features = raw_features(static, genes)
        target = ridge.response_from_cp10k_moments(
            moments["cp10k_sum"], moments["cell_count"]
        ) - ridge.control_anchor(baseline["basal_rate"], moments["gem_cell_count"])
        model = model_module.fit(features, target, rank=32, alpha=1000.0)
        path = output / f"model-{source}.npz"
        model_module.save(path, model, query_ids=query, source_id=source)
        restored = model_module.load(path)
        difference = float(np.max(np.abs(model.predict(features[:16]) - restored.predict(features[:16]))))
        if difference != 0:
            raise RuntimeError(f"{source} target-free model reload is not exact")
        models[source] = restored
        receipts[source] = {
            "path": path.name,
            "sha256": sha256(path),
            "genes": len(genes),
            "queries": len(query),
            "reloadMaximumAbsoluteDifference": difference,
            "stateProjectionShape": list(model.state_projection.shape),
            "queryLoadingShape": list(model.query_loading.shape),
        }
    manifest = {
        "schema": "slp.human-essential-count-response-rank32-model-freeze/v1",
        "protocolSha256": sha256(output / "protocol.json"),
        "models": receipts,
        "source": {
            "modelSha256": sha256(output / "source/response_model.py"),
            "runnerSha256": sha256(output / "source/runner.py"),
        },
        "developmentOpened": False,
        "testOpened": False,
    }
    write_json(output / "MODELS-FROZEN-BEFORE-DEVELOPMENT.json", manifest)
    return models, manifest


def freeze_forecasts(output, models, ridge):
    receipts = {}
    for source, paths in CONTEXTS.items():
        metadata = load_npz(paths["metadata_forecast"])
        static = load_npz(paths["static"])
        gene_ids = metadata["gene_ids"].astype(str)
        residual = models[source].predict(raw_features(static, gene_ids))
        prediction = ridge.absolute_prediction(metadata["control_prediction"], residual)
        if not np.isfinite(prediction).all():
            raise ValueError("rank32 forecast is nonfinite")
        path = output / f"development-forecast-{source}.npz"
        np.savez_compressed(
            path,
            schema=np.asarray("slp.human-essential-count-response-rank32-development-forecast/v1"),
            source_id=metadata["source_id"],
            context_id=metadata["context_id"],
            gene_ids=metadata["gene_ids"],
            query_ids=metadata["query_ids"],
            gem_group_ids=metadata["gem_group_ids"],
            cell_count=metadata["cell_count"],
            gem_cell_count=metadata["gem_cell_count"],
            control_prediction=metadata["control_prediction"],
            anchored_mean_prediction=metadata["anchored_mean_prediction"],
            static_ridge_prediction=metadata["static_ridge_prediction"],
            rank32_prediction=prediction,
        )
        receipts[source] = {
            "path": path.name,
            "sha256": sha256(path),
            "genes": len(gene_ids),
            "queries": len(metadata["query_ids"]),
        }
    model_freeze = sha256(output / "MODELS-FROZEN-BEFORE-DEVELOPMENT.json")
    freeze = {
        "schema": "slp.human-essential-count-response-rank32-forecast-freeze/v1",
        "modelsFreezeSha256": model_freeze,
        "models": {
            source: {
                "path": f"model-{source}.npz",
                "sha256": sha256(output / f"model-{source}.npz"),
            }
            for source in CONTEXTS
        },
        "forecasts": receipts,
        "forecastsFrozenBeforeDevelopmentTruthAccess": True,
        "developmentOpened": False,
        "testOpened": False,
    }
    write_json(output / "FORECASTS-FROZEN-BEFORE-DEVELOPMENT.json", freeze)
    return freeze


def score(output, evaluator):
    if not (output / "FORECASTS-FROZEN-BEFORE-DEVELOPMENT.json").is_file():
        raise ValueError("forecast freeze must precede truth access")
    metrics, arrays = {}, {}
    for source, paths in CONTEXTS.items():
        forecast = load_npz(output / f"development-forecast-{source}.npz")
        truth = load_npz(paths["truth"])
        if (
            not np.array_equal(forecast["gene_ids"], truth["gene_ids"])
            or not np.array_equal(forecast["query_ids"], truth["query_ids"])
            or not np.array_equal(forecast["cell_count"], truth["cell_count"])
            or not np.array_equal(forecast["gem_cell_count"], truth["gem_cell_count"])
        ):
            raise ValueError(f"{source} frozen truth identity mismatch")
        target = truth["truth_log1p_mean_cp10k"]
        metrics[source] = {}
        for name in ("anchored_mean_prediction", "static_ridge_prediction", "rank32_prediction"):
            summary, mse, correlation = evaluator.score_prediction(
                target, forecast[name], forecast["control_prediction"]
            )
            metrics[source][name] = summary
            arrays[f"{source}_{name}_mse"] = mse
            arrays[f"{source}_{name}_centered_pearson"] = correlation
        arrays[f"{source}_gene_ids"] = forecast["gene_ids"]
    per_gene = output / "development-per-gene-metrics.npz"
    np.savez_compressed(per_gene, **arrays)
    gate = {}
    for source in CONTEXTS:
        rank = metrics[source]["rank32_prediction"]
        ridge = metrics[source]["static_ridge_prediction"]
        gate[f"{source}MseOnePercentBelowRidge"] = bool(
            rank["geneProfileMse"] <= 0.99 * ridge["geneProfileMse"]
        )
        gate[f"{source}PearsonAtLeastPoint10"] = bool(
            rank["independentlyQueryCenteredResidualPearson"] is not None
            and rank["independentlyQueryCenteredResidualPearson"] >= 0.1
        )
        gate[f"{source}PearsonNonregressionVsRidge"] = bool(
            rank["independentlyQueryCenteredResidualPearson"] is not None
            and ridge["independentlyQueryCenteredResidualPearson"] is not None
            and rank["independentlyQueryCenteredResidualPearson"]
            >= ridge["independentlyQueryCenteredResidualPearson"]
        )
    gate["passes"] = bool(all(gate.values()))
    return metrics, gate, {"path": per_gene.name, "sha256": sha256(per_gene)}


def run(output=OUTPUT):
    prepare(output)
    if (output / "report.json").exists():
        raise FileExistsError("immutable rank32 result already exists")
    started = time.perf_counter()
    model_module = load_module(MODEL, "rank32_response_model")
    ridge = load_module(RIDGE, "rank32_response_ridge")
    evaluator = load_module(EVALUATOR, "rank32_response_evaluator")
    models, model_freeze = fit_and_freeze(output, model_module, ridge)
    forecast_freeze = freeze_forecasts(output, models, ridge)
    metrics, gate, per_gene = score(output, evaluator)
    seconds = time.perf_counter() - started
    if seconds > 600:
        raise TimeoutError("rank32 fixed run exceeded CPU cap")
    report = {
        "schema": "slp.human-essential-count-response-rank32-report/v1",
        "protocolSha256": sha256(output / "protocol.json"),
        "modelsFreezeSha256": sha256(output / "MODELS-FROZEN-BEFORE-DEVELOPMENT.json"),
        "forecastsFreezeSha256": sha256(output / "FORECASTS-FROZEN-BEFORE-DEVELOPMENT.json"),
        "modelFreeze": model_freeze,
        "forecastFreeze": forecast_freeze,
        "metrics": metrics,
        "gate": gate,
        "perGeneMetrics": per_gene,
        "seconds": seconds,
        "interpretation": "Adaptive development test of a fitting-selected panel-specific rank32 feature-linear response model. Native fitted query loadings do not establish unmeasured-query prediction, a cell generator, nonlinear dynamics, or independent confirmation.",
        "developmentEvaluations": 1,
        "testOpened": False,
        "benchmarkAccessed": False,
    }
    write_json(output / "report.json", report)
    print(json.dumps({"metrics": metrics, "gate": gate, "seconds": seconds}))
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with threadpool_limits(2):
        {"prepare": prepare, "run": run}[args.mode](args.output)
