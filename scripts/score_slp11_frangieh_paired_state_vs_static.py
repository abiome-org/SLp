"""Score frozen Frangieh paired-state predictions against frozen static forecasts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRIC_PATH = ROOT / "modules/slp-1-1-world-transition-v1/frangieh_basal_ridge.py"
SPEC = importlib.util.spec_from_file_location("frangieh_basal_ridge", METRIC_PATH)
METRICS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(METRICS)

CONTEXT_KEYS = {
    "Co-culture": "Co_culture",
    "Control": "Control",
    "IFNγ": "IFNgamma",
}
HEAD_KEYS = {"rna": "rna", "protein": "adt"}
BASELINES = ("mean", "base577", "physical1156")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def score_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float | int | None]:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("prediction and truth must be aligned matrices")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("prediction and truth must be finite")
    adjusted_r, per_gene = METRICS.query_centroid_adjusted_profile_pearson(prediction, truth)
    return {
        "raw_mse": float(np.mean((prediction - truth) ** 2)),
        "query_centroid_adjusted_profile_pearson": (
            float(adjusted_r) if np.isfinite(adjusted_r) else None
        ),
        "query_centroid_adjusted_profile_pearson_undefined_genes": int(
            np.sum(~np.isfinite(per_gene))
        ),
        "ordinary_pearson": METRICS.ordinary_pearson(prediction, truth),
    }


def evaluate_gates(world: dict, baselines: dict[str, dict]) -> dict:
    world_mse = float(world["raw_mse"])
    world_r_value = world["query_centroid_adjusted_profile_pearson"]
    world_r = float(world_r_value) if world_r_value is not None else float("nan")
    comparisons = {}
    for name in BASELINES:
        baseline = baselines[name]
        baseline_mse = float(baseline["raw_mse"])
        baseline_r_value = baseline["query_centroid_adjusted_profile_pearson"]
        baseline_r = float(baseline_r_value) if baseline_r_value is not None else float("nan")
        improvement = (baseline_mse - world_mse) / baseline_mse
        r_defined = bool(np.isfinite(baseline_r))
        comparisons[name] = {
            "fractional_raw_mse_improvement": float(improvement),
            "mse_improvement_at_least_one_percent": bool(improvement >= 0.01),
            "baseline_adjusted_r_defined": r_defined,
            "adjusted_r_no_regression": bool(
                (not r_defined) or (np.isfinite(world_r) and world_r >= baseline_r - 1e-12)
            ),
        }
    world_r_pass = bool(np.isfinite(world_r) and world_r >= 0.1)
    passed = bool(
        world_r_pass
        and all(item["mse_improvement_at_least_one_percent"] for item in comparisons.values())
        and all(item["adjusted_r_no_regression"] for item in comparisons.values())
    )
    return {
        "world_adjusted_r_at_least_point_one": world_r_pass,
        "comparisons": comparisons,
        "pass": passed,
    }


def paired_mse_bootstrap(
    world: np.ndarray,
    baseline: np.ndarray,
    truth: np.ndarray,
    samples: np.ndarray,
) -> dict[str, float]:
    world = np.asarray(world, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if world.shape != baseline.shape or world.shape != truth.shape:
        raise ValueError("bootstrap matrices do not align")
    per_gene_world = np.mean((world - truth) ** 2, axis=1)
    per_gene_baseline = np.mean((baseline - truth) ** 2, axis=1)
    world_draw = np.mean(per_gene_world[samples], axis=1)
    baseline_draw = np.mean(per_gene_baseline[samples], axis=1)
    delta = baseline_draw - world_draw
    improvement = np.divide(
        delta,
        baseline_draw,
        out=np.full_like(delta, np.nan),
        where=baseline_draw > 0,
    )
    return {
        "raw_mse_difference_baseline_minus_world_ci025": float(np.quantile(delta, 0.025)),
        "raw_mse_difference_baseline_minus_world_ci975": float(np.quantile(delta, 0.975)),
        "fractional_raw_mse_improvement_ci025": float(np.nanquantile(improvement, 0.025)),
        "fractional_raw_mse_improvement_ci975": float(np.nanquantile(improvement, 0.975)),
    }


def _require_equal(label: str, left: np.ndarray, right: np.ndarray) -> None:
    if not np.array_equal(left, right):
        raise ValueError(f"{label} mismatch")


def run(
    neural_predictions: Path,
    static_predictions: Path,
    mean_predictions: Path,
    development_data: Path,
    output_dir: Path,
    *,
    bootstrap_seed: int = 731,
    bootstrap_replicates: int = 1000,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (
        np.load(neural_predictions, allow_pickle=False) as neural,
        np.load(static_predictions, allow_pickle=False) as static,
        np.load(mean_predictions, allow_pickle=False) as mean,
        np.load(development_data, allow_pickle=False) as data,
    ):
        _require_equal("RNA query IDs", neural["rna_query_ids"].astype(str), data["rna_query_ids"].astype(str))
        _require_equal(
            "protein query IDs",
            neural["protein_query_ids"].astype(str),
            data["protein_channel_ids"].astype(str),
        )
        contexts = neural["context_names"].astype(str)
        if tuple(contexts) != tuple(CONTEXT_KEYS):
            raise ValueError("unexpected neural context order")
        if neural["action_ids"].shape != (129,) or neural["context_index"].shape != (129,):
            raise ValueError("unexpected neural validation axes")

        rng = np.random.default_rng(bootstrap_seed)
        samples = rng.integers(0, 43, size=(bootstrap_replicates, 43))
        report = {
            "schema": "slp.frangieh-paired-state-vs-static-scoring/v1",
            "fixed_rule": (
                "In every context and head, paired-state raw MSE must improve by at least one "
                "percent over mean, base577 and physical1156; adjusted profile r must be at least "
                "0.10 and must not regress against each comparator whose adjusted r is defined."
            ),
            "bootstrap": {
                "unit": "intervention gene with all queries retained",
                "replicates": bootstrap_replicates,
                "seed": bootstrap_seed,
                "interval": "paired percentile 2.5--97.5 percent",
            },
            "contexts": {},
        }
        decisions = []
        for context_index, context in enumerate(contexts):
            neural_rows = np.flatnonzero(neural["context_index"] == context_index)
            if len(neural_rows) != 43:
                raise ValueError(f"{context}: expected 43 neural validation genes")
            context_key = CONTEXT_KEYS[context]
            context_report = {"heads": {}}
            for neural_head, static_head in HEAD_KEYS.items():
                prefix = f"{context_key}_{static_head}"
                static_ids = static[f"{prefix}_action_ids"].astype(str)
                mean_ids = mean[f"{prefix}_action_ids"].astype(str)
                neural_ids = neural["action_ids"][neural_rows].astype(str)
                _require_equal(f"{context}/{neural_head} action IDs", neural_ids, static_ids)
                _require_equal(f"{context}/{neural_head} mean action IDs", neural_ids, mean_ids)

                truth = np.asarray(neural[f"{neural_head}_truth"][neural_rows], dtype=np.float32)
                _require_equal(f"{context}/{neural_head} static truth", truth, static[f"{prefix}_truth"])
                _require_equal(f"{context}/{neural_head} mean truth", truth, mean[f"{prefix}_truth"])
                world = np.asarray(neural[f"{neural_head}_prediction"][neural_rows], dtype=np.float32)
                forecasts = {
                    "mean": np.asarray(mean[f"{prefix}_mean"], dtype=np.float32),
                    "base577": np.asarray(static[f"{prefix}_base577"], dtype=np.float32),
                    "physical1156": np.asarray(static[f"{prefix}_physical1156"], dtype=np.float32),
                }
                world_metrics = score_metrics(world, truth)
                baseline_metrics = {name: score_metrics(value, truth) for name, value in forecasts.items()}
                gates = evaluate_gates(world_metrics, baseline_metrics)
                bootstrap = {
                    name: paired_mse_bootstrap(world, value, truth, samples)
                    for name, value in forecasts.items()
                }
                context_report["heads"][neural_head] = {
                    "genes": len(neural_ids),
                    "queries": truth.shape[1],
                    "world": world_metrics,
                    "baselines": baseline_metrics,
                    "gates": gates,
                    "paired_gene_mse_bootstrap": bootstrap,
                }
                decisions.append(gates["pass"])
            report["contexts"][context] = context_report
        report["decision"] = {
            "advance": bool(all(decisions)),
            "passed_strata": int(sum(decisions)),
            "total_strata": len(decisions),
        }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural-predictions", type=Path, required=True)
    parser.add_argument("--static-predictions", type=Path, required=True)
    parser.add_argument("--mean-predictions", type=Path, required=True)
    parser.add_argument("--development-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    checks = {
        "neural_predictions": args.neural_predictions,
        "static_predictions": args.static_predictions,
        "mean_predictions": args.mean_predictions,
        "development_data": args.development_data,
    }
    for name, path in checks.items():
        if digest(path) != protocol["inputs"][name]["sha256"]:
            raise ValueError(f"{name} hash mismatch")
    print(json.dumps(run(**checks, output_dir=args.output_dir), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
