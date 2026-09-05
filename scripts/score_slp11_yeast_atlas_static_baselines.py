#!/usr/bin/env python3
"""Correct and stratify frozen yeast baseline metrics without refitting."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "modules/slp-1-1-yeast-static-baseline-v1/static_baseline.py"
RUNNER_PATH = ROOT / "scripts/run_slp11_yeast_atlas_static_baselines.py"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_python(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def finite(value: object) -> object:
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run(args: argparse.Namespace) -> None:
    fit = args.fit.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    for path in (MODULE_PATH, Path(__file__)):
        shutil.copyfile(path, source / path.name)
    baseline = load_python(MODULE_PATH, "slp11_yeast_static_baseline_scoring")
    runner = load_python(RUNNER_PATH, "slp11_yeast_static_baseline_runner")
    fit_report_path = fit / "report.json"
    fit_protocol_path = fit / "protocol.json"
    prediction_path = fit / "validation-predictions.npz"
    fit_report = json.loads(fit_report_path.read_text(encoding="utf-8"))
    with np.load(args.corpus, allow_pickle=False) as item:
        target = item["targets"].astype(np.float64)
        observed = item["observed"]
        action_ids = item["action_ids"].astype(str)
        context_index = item["context_index"]
        context_ids = item["context_ids"].astype(str)
        validation = item["split_validation"]
    with np.load(prediction_path, allow_pickle=False) as item:
        if not np.array_equal(item["validation_indices"], validation):
            raise ValueError("frozen validation prediction roster mismatch")
        predictions = {name: item[name].astype(np.float64) for name in ("mean", "linear", "nystrom")}
    features, _ = runner.aligned_static_features(action_ids, args.esm, args.go)
    supported = features[:, -2].astype(bool)
    contexts = {}
    for context, context_name in enumerate(context_ids):
        global_rows = validation[context_index[validation] == context]
        positions = np.flatnonzero(context_index[validation] == context)
        with np.load(fit / f"{context_name}-mean.npz", allow_pickle=False) as mean_file:
            centroid = mean_file["intercept"]
            scale = mean_file["training_scale"]
        metrics = {}
        for method, prediction in predictions.items():
            metrics[method] = {
                "all": baseline.evaluate_gene_profiles(
                    prediction[positions], target[global_rows], observed[global_rows], centroid, scale,
                ),
                "static_supported": baseline.evaluate_gene_profiles(
                    prediction[positions][supported[global_rows]], target[global_rows][supported[global_rows]],
                    observed[global_rows][supported[global_rows]], centroid, scale,
                ),
                "static_unsupported": baseline.evaluate_gene_profiles(
                    prediction[positions][~supported[global_rows]], target[global_rows][~supported[global_rows]],
                    observed[global_rows][~supported[global_rows]], centroid, scale,
                ),
            }
        primary = {method: item["all"] for method, item in metrics.items()}
        rbf_mse = primary["nystrom"]["gene_macro_mse"]
        mean_mse = primary["mean"]["gene_macro_mse"]
        linear_mse = primary["linear"]["gene_macro_mse"]
        rbf_r = primary["nystrom"]["gene_macro_independent_query_centered_profile_pearson"]
        linear_r = primary["linear"]["gene_macro_independent_query_centered_profile_pearson"]
        gate = {
            "mseImprovementVsMean": 1.0 - rbf_mse / mean_mse,
            "mseImprovementVsLinear": 1.0 - rbf_mse / linear_mse,
            "msePass": rbf_mse <= 0.98 * mean_mse and rbf_mse <= 0.98 * linear_mse,
            "correlationThresholdPass": rbf_r is not None and rbf_r >= 0.10,
            "correlationNonregressionPass": linear_r is None or (rbf_r is not None and rbf_r >= linear_r),
        }
        gate["passed"] = all(bool(gate[key]) for key in ("msePass", "correlationThresholdPass", "correlationNonregressionPass"))
        contexts[str(context_name)] = {
            "staticSupportedGenes": int(supported[global_rows].sum()),
            "staticUnsupportedGenes": int((~supported[global_rows]).sum()),
            "selectionReusedWithoutRefit": fit_report["contexts"][str(context_name)]["selection"],
            "metrics": metrics,
            "gate": gate,
        }
    protocol = {
        "schema": "slp.nadal-ribelles-yeast-static-baseline-score-correction/v2",
        "supersedesForMetrics": str(fit_report_path),
        "reason": "The v1 mean comparator's independently query-centered correlation was numerically nonconstant at roundoff scale. V2 applies a scale-aware variance tolerance so mathematically constant profiles are undefined.",
        "fitsPredictionsAndSelectionChanged": False,
        "primaryPopulationChanged": False,
        "primaryPopulation": "all validation genes including static-unsupported genes",
        "staticSupportStrata": "descriptive only; exact presence on the older 7,037-entity yeast static axis; not used for selection or advancement",
        "fit": {
            "protocolSha256": sha256(fit_protocol_path),
            "reportSha256": sha256(fit_report_path),
            "predictionsSha256": sha256(prediction_path),
        },
        "inputs": {
            "corpus": {"path": str(args.corpus), "sha256": sha256(args.corpus)},
            "esm": {"path": str(args.esm), "sha256": sha256(args.esm)},
            "go": {"path": str(args.go), "sha256": sha256(args.go)},
        },
        "sourceHashes": {path.name: sha256(path) for path in source.iterdir()},
        "protectedOutcomesAccessed": False,
        "developmentTestOutcomesAccessed": False,
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n",
    )
    report = {
        "schema": "slp.nadal-ribelles-yeast-static-baseline-report/v2",
        "protocolSha256": sha256(output / "protocol.json"),
        "contexts": contexts,
        "advancementPassed": all(item["gate"]["passed"] for item in contexts.values()),
        "fitMaximumTargetFreeReloadDrift": fit_report["maximumTargetFreeReloadDrift"],
        "fitsPredictionsAndSelectionChanged": False,
    }
    (output / "report.json").write_text(
        json.dumps(finite(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"report": str(output / "report.json"), "advancementPassed": False}))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--fit", type=Path, required=True)
    value.add_argument("--corpus", type=Path, required=True)
    value.add_argument("--esm", type=Path, required=True)
    value.add_argument("--go", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
