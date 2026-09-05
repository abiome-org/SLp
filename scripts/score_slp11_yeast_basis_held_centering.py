"""No-refit held-cohort centering supplement for frozen yeast response bases."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "results/slp11-transition/yeast-rna-fitting-crosscov-basis-rank32-v1"
STATS = ROOT / "results/slp11-transition/yeast-rna-fitting-split-half-v1/split-half-sufficient-statistics.npz"
OUTPUT = ROOT / "results/slp11-transition/yeast-rna-basis-held-centering-supplement-v1"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def center(values):
    delta = np.asarray(values, dtype=np.float64) - values[:1]
    return delta - delta.mean(axis=0, keepdims=True)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run():
    started = time.monotonic()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    report_path = PRIOR / "report.json"
    if digest(report_path) != "4aebdcb26bcaa14026a3a44fbed83b6e49160580406715314167f1948b52d6c7":
        raise ValueError("prior report drift")
    prior = json.loads(report_path.read_text())
    paths = {"statistics": STATS, "priorReport": report_path,
             "basisHelper": ROOT / "scripts/run_slp11_yeast_crosscov_response_basis.py",
             "correlationHelper": ROOT / "scripts/run_slp11_yeast_batch_ridge.py",
             "scorer": Path(__file__)}
    for context, details in prior["contexts"].items():
        for fold in details["folds"]:
            path = PRIOR / fold["basis"]["path"]
            if digest(path) != fold["basis"]["sha256"]:
                raise ValueError("basis drift")
            paths[f"{context}-{fold['fold']}"] = path
    OUTPUT.mkdir(parents=True)
    source = OUTPUT / "source"
    source.mkdir()
    for name in ("basisHelper", "correlationHelper", "scorer"):
        shutil.copy2(paths[name], source / paths[name].name)
    write(OUTPUT / "protocol.json", {
        "schema": "slp.yeast-basis-held-centering-protocol/v1",
        "question": "Does the saved representation advantage persist after removing each held fold's own separate A/B query centroid?",
        "method": "No refitting; separately anchored-center held A and B before fixed projection. Equal-gene aggregate of fold-local metrics; retain rank32 and original +.02 crosscov-minus-PCA rule descriptively.",
        "access": "Fitting-only sufficient statistics and saved fold bases; no raw count, development validation, test or SL outcome access.",
        "inputs": {name: {"path": str(path), "sha256": digest(path)} for name, path in paths.items()},
        "limitSeconds": 180, "cpuThreads": 2,
    })
    basis = load(paths["basisHelper"], "held_center_basis")
    scorer = load(paths["correlationHelper"], "held_center_metric")
    statistics = basis.load_statistics(STATS)
    contexts = {}
    for context_index, context in enumerate(statistics["contexts"].tolist()):
        genes, a, b = basis.environment_profiles(statistics, context_index)
        lookup = {gene: row for row, gene in enumerate(genes.tolist())}
        aggregates = {name: [] for name in ("raw", "pca", "crosscov")}
        folds = []
        for entry in prior["contexts"][context]["folds"]:
            if time.monotonic() - started > 180:
                raise TimeoutError("supplement budget exceeded")
            with np.load(PRIOR / entry["basis"]["path"], allow_pickle=False) as archive:
                held = archive["held_gene_ids"].astype(str)
                if set(held) & set(archive["fit_gene_ids"].astype(str)):
                    raise ValueError("fold overlap")
                rows = np.asarray([lookup[gene] for gene in held])
                ac, bc = center(a[rows]), center(b[rows])
                predictions = {"raw": (ac, bc)}
                for arm, key in (("pca", "pca_basis"), ("crosscov", "cross_basis")):
                    coordinates = archive[key]
                    predictions[arm] = ((ac @ coordinates) @ coordinates.T,
                                        (bc @ coordinates) @ coordinates.T)
            full_trace = float(np.mean(np.sum(ac * bc, axis=1)))
            fold_result = {"fold": entry["fold"], "genes": len(rows), "metrics": {}}
            for arm, (left, right) in predictions.items():
                correlations = scorer._row_corr(left, right)
                mse = np.mean((left - right) ** 2, axis=1)
                trace = np.sum(left * right, axis=1)
                aggregates[arm].append((correlations, mse, trace))
                fold_result["metrics"][arm] = {
                    "centeredPearson": float(np.nanmean(correlations)),
                    "mse": float(mse.mean()),
                    "traceFraction": float(trace.mean() / full_trace),
                }
            folds.append(fold_result)
        summary = {}
        denominator = np.concatenate([item[2] for item in aggregates["raw"]]).mean()
        for arm, entries in aggregates.items():
            correlations = np.concatenate([item[0] for item in entries])
            summary[arm] = {
                "centeredPearson": float(np.nanmean(correlations)),
                "undefinedGenes": int(np.isnan(correlations).sum()),
                "mse": float(np.concatenate([item[1] for item in entries]).mean()),
                "traceFraction": float(np.concatenate([item[2] for item in entries]).mean() / denominator),
            }
        difference = summary["crosscov"]["centeredPearson"] - summary["pca"]["centeredPearson"]
        contexts[context] = {"folds": folds, "aggregate": summary,
                             "crosscovMinusPca": difference, "passesOriginalPoint02": difference >= .02}
    result = {"schema": "slp.yeast-basis-held-centering-report/v1", "contexts": contexts,
              "seconds": time.monotonic() - started,
              "limitations": "Projection reproducibility only; shared batch/clone effects remain. No intervention forecasting evidence.",
              "protocolSha256": digest(OUTPUT / "protocol.json")}
    write(OUTPUT / "report.json", result)
    print(json.dumps({context: value["aggregate"] for context, value in contexts.items()}, indent=2))
    print(digest(OUTPUT / "report.json"))


if __name__ == "__main__":
    run()
