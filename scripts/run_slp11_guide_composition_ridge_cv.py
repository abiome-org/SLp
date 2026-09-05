"""Fitting-only guide-composition ridge ablation for K562 and RPE1 counts."""
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
OUTPUT = ROOT / "results/slp11-transition/replogle-guide-composition-ridge-fitting-cv-v1"
GUIDES = ROOT / "data/derived/slp11-human-replogle-guide-library-v2/guide-pair-metadata.npz"
GUIDE_MANIFEST = ROOT / "data/derived/slp11-human-replogle-guide-library-v2/manifest.json"
RIDGE_CORE = ROOT / "modules/slp-1-1-count-static-ridge-v1/count_static_ridge.py"
GUIDE_CORE = ROOT / "modules/slp-1-1-guide-composition-v1/guide_composition.py"

CONTEXTS = {
    "k562": {
        "moments": ROOT / "data/derived/slp11-human-k562-essential-fitting-action-moments-v1/fitting-action-moments.npz",
        "rows": ROOT / "data/derived/slp11-human-k562-essential-count-latent-training-mmap-v1/rows.npz",
        "static": ROOT / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1/k562-essential-count-static577.npz",
        "baseline": ROOT / "results/slp11-transition/k562-essential-count-anchored-static-ridge-seed731-v1/model.npz",
    },
    "rpe1": {
        "moments": ROOT / "data/derived/slp11-human-rpe1-essential-raw-cells-v1/fitting-action-moments.npz",
        "rows": ROOT / "data/derived/slp11-human-rpe1-essential-raw-cells-v1/reconstruction-train-row-metadata.npz",
        "static": ROOT / "data/derived/slp11-human-rpe1-essential-count-static/ensembl116-esm8m-shared-go-v1/rpe1-essential-count-static577.npz",
        "baseline": ROOT / "results/slp11-transition/rpe1-essential-count-anchored-static-ridge-seed731-v1/model.npz",
    },
}

PINS = {
    GUIDES: "c3e5e2167ac9fa0517caf7d8bbeed80d19212d9935fed991816954dcfea16b1c",
    GUIDE_MANIFEST: "9311788f6992fc470e8f3c04d863bedbcfe8ff83aac560c2debbedf7719ac5e2",
    RIDGE_CORE: "1032eeff59382fae3874da9a389033192e113e0f5ac2c8d01f09f8441d969e62",
    GUIDE_CORE: "42ee694b0239e9d8efbdf989f5b2f1b1ef569930ea8e7a084ee6fa1c7423dded",
    CONTEXTS["k562"]["moments"]: "a1f44a15a42c5b56e4ce897fde6ebba97298fc296105c6c870ee0e740331694e",
    CONTEXTS["k562"]["rows"]: "5d8631e50b3dcabc9448eaa112eb94bc1335967e5b9098b6e278b6340a9a226b",
    CONTEXTS["k562"]["static"]: "6706f8867adedef8822897bc275ea90680584f84afd24771e4beb3c8ecf07659",
    CONTEXTS["k562"]["baseline"]: "dbb669d2eb8d844ec9be7c88a2ed21f5592de434d1b2e916412bda4a52fe1cf3",
    CONTEXTS["rpe1"]["moments"]: "d15def86aead06b0bc75ab63c77513735ec7c57d65012bff72f3947bc654895c",
    CONTEXTS["rpe1"]["rows"]: "b7b035798415ce2bc55361b12a52d13739cb2555621456342f75cf1e7a15339a",
    CONTEXTS["rpe1"]["static"]: "621e1e9f0dffc740ef42382b1b2898f629edd5037e8a02d411e8d30e815ed816",
    CONTEXTS["rpe1"]["baseline"]: "bd144e36b5618c6225828501492edfa5449cef07442041c1d1cc20645b1473bc",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {name: np.asarray(values[name]) for name in values.files}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def protocol() -> dict[str, object]:
    return {
        "schema": "slp.replogle-guide-composition-ridge-fitting-cv-protocol/v1",
        "hypothesis": "Cell-proportion-aggregated dual-guide sequence composition adds intervention-transfer signal beyond static577 protein/GO features for aggregate molecular means.",
        "advancementRule": "The augmented static577+guide71 ridge must reduce three-fold fitting-gene OOF full-query MSE by at least 1% relative to static577 ridge in both K562 and RPE1. No development model is fitted or evaluated if either context fails.",
        "population": "All 1,443 K562 and 1,666 RPE1 fitting action genes; exact reconstruction-training intervention cells only; equal weight per gene and query.",
        "endpoint": "ln1p(equal-cell mean CP10k) minus the gene's fitting-cell-GEM-weighted measured-control log1p anchor; same absolute MSE as adding the anchor back.",
        "guideDescriptor": {
            "dimensions": 71,
            "definition": "unordered and independently reverse-complement-invariant: pair mean32 and absolute difference32 of canonical RC 3-mer frequencies; GC mean/absolute difference2; homopolymer min/max2; base-entropy mean/absolute difference2; minimum normalized Hamming(A,B or RC(B))1",
            "aggregation": "average exact pair descriptors by reconstruction-training cell guide-pair proportions within each fitting gene and context",
            "excluded": ["guide identity", "absolute genomic coordinate", "outcome-derived efficacy"],
            "missing": "fail closed on any fitting-cell guide pair without one exact 20-nt sequence join",
        },
        "models": {
            "static577": "raw ESM8M320 + protein-presence1 + shared GO MF/CC256",
            "static577Guide71": "the identical raw static577 concatenated with the fixed 71D aggregate guide descriptor",
        },
        "crossValidation": "Same global_gene_fold seed731, three folds, fold-local feature normalization, exact unpenalized residual intercept, alpha grid .1,1,10,100,1000,10000,100000,1e6 plus mean-limit; alpha selected separately for each context/model by equal-gene all-query OOF MSE.",
        "compute": {"cpuThreads": 2, "maximumSecondsPerContext": 600, "maximumRssBytes": 6 * 2**30},
        "pins": {str(path.relative_to(ROOT)): value for path, value in PINS.items()},
        "runnerSha256": sha256(Path(__file__).resolve()),
        "developmentOpened": False,
        "testOpened": False,
        "benchmarkAccessed": False,
    }


def prepare(output: Path = OUTPUT) -> dict[str, object]:
    for path, expected in PINS.items():
        if sha256(path) != expected:
            raise ValueError(f"frozen input mismatch: {path}")
    output.mkdir(parents=True, exist_ok=True)
    frozen = protocol()
    path = output / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != frozen:
            raise ValueError("frozen guide-composition protocol changed")
    else:
        write_json(path, frozen)
    return frozen


def cross_validated_predictions(
    core,
    genes: np.ndarray,
    features: np.ndarray,
    target: np.ndarray,
) -> tuple[str, dict[str, float], list[dict[str, object]], np.ndarray, np.ndarray]:
    """Fit each fold once, select alpha globally, and return exact OOF errors."""
    gene_ids = np.asarray(genes).astype(str)
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(gene_ids) != len(x):
        raise ValueError("gene, feature, and target rows must align")
    folds = np.asarray([core.global_gene_fold(gene, 731) for gene in gene_ids], np.int8)
    totals = {candidate: 0.0 for candidate in core.ALPHAS}
    states, reports = [], []
    for fold in range(3):
        fitting, held = folds != fold, folds == fold
        if not fitting.any() or not held.any():
            raise ValueError("each global fold requires fitting and held genes")
        state = core.fit_state(x[fitting], y[fitting])
        scores = core.candidate_mse(state, x[held], y[held])
        for candidate, score in scores.items():
            totals[candidate] += score * int(held.sum())
        reports.append(
            {
                "fold": fold,
                "fittingGenes": int(fitting.sum()),
                "heldGenes": int(held.sum()),
                "rawAllQueryMse": scores,
                "featureMeanSha256": hashlib.sha256(state["feature_mean"].tobytes()).hexdigest(),
                "featureScaleSha256": hashlib.sha256(state["feature_scale"].tobytes()).hexdigest(),
            }
        )
        states.append(state)
    scores = {candidate: total / len(gene_ids) for candidate, total in totals.items()}
    selected = min(core.ALPHAS, key=lambda item: (scores[item], core.ALPHAS.index(item)))
    prediction = np.empty_like(y)
    mean_prediction = np.empty_like(y)
    for fold, state in enumerate(states):
        held = folds == fold
        prediction[held] = core.predict_residual(state, x[held], selected)
        mean_prediction[held] = core.predict_residual(state, x[held], "mean-limit")
    return selected, scores, reports, prediction, mean_prediction


def context_inputs(name: str, guide_core, guide_library) -> dict[str, object]:
    paths = CONTEXTS[name]
    rows = load_npz(paths["rows"])
    selected = ~np.asarray(rows["is_control"], dtype=np.bool_)
    actions = rows["action_ids"].astype(str)[selected]
    pairs = rows["guide_pair_ids"].astype(str)[selected]
    with np.load(paths["moments"], allow_pickle=False) as values:
        genes = values["action_ids"].astype(str)
        query_ids = values["query_ids"].astype(str)
        cell_count = np.asarray(values["cell_count"], np.int64)
        cp10k_sum = np.asarray(values["cp10k_sum"], np.float64)
        gem_cell_count = np.asarray(values["gem_cell_count"], np.int64)
    guide_values, guide_cells, guide_pairs = guide_core.aggregate_gene_descriptors(
        genes,
        actions,
        pairs,
        guide_library["guide_pair_ids"],
        guide_library["targeting_sequence_a"],
        guide_library["targeting_sequence_b"],
    )
    if not np.array_equal(guide_cells, cell_count):
        raise ValueError(f"{name} guide-cell aggregation does not match fitting moments")
    static = load_npz(paths["static"])
    lookup = {value: row for row, value in enumerate(static["entity_id"].astype(str))}
    entity = np.asarray([lookup[gene] for gene in genes], np.int64)
    raw_static = np.asarray(static["feature_values"][entity], np.float32)
    baseline = load_npz(paths["baseline"])
    if not np.array_equal(query_ids, baseline["query_ids"].astype(str)):
        raise ValueError(f"{name} baseline and moment query axes differ")
    return {
        "genes": genes,
        "query_ids": query_ids,
        "cell_count": cell_count,
        "gem_cell_count": gem_cell_count,
        "guide_features": guide_values,
        "guide_pair_count": guide_pairs,
        "raw_static": raw_static,
        "target": np.log1p(cp10k_sum / cell_count[:, None]),
        "basal_rate": np.asarray(baseline["basal_rate"], np.float64),
    }


def run(output: Path = OUTPUT) -> dict[str, object]:
    prepare(output)
    if (output / "report.json").exists():
        raise FileExistsError("immutable guide-composition CV already complete")
    ridge = load_module(RIDGE_CORE, "guide_composition_ridge")
    guide = load_module(GUIDE_CORE, "guide_composition_features")
    guide_library = load_npz(GUIDES)
    reports, arrays = {}, {}
    for name in ("k562", "rpe1"):
        started = time.perf_counter()
        values = context_inputs(name, guide, guide_library)
        anchor = ridge.control_anchor(values["basal_rate"], values["gem_cell_count"])
        residual = values["target"] - anchor
        augmented = np.concatenate(
            (values["raw_static"], values["guide_features"].astype(np.float32)), axis=1
        )
        model_reports, predictions = {}, {}
        for model, features in (
            ("static577", values["raw_static"]),
            ("static577Guide71", augmented),
        ):
            selected, scores, folds, prediction, mean_prediction = cross_validated_predictions(
                ridge, values["genes"], features, residual
            )
            model_reports[model] = {
                "selectedAlpha": selected,
                "oofRawAllQueryMse": scores[selected],
                "candidateOofRawAllQueryMse": scores,
                "folds": folds,
            }
            predictions[model] = prediction
            if "mean-limit" not in predictions:
                predictions["mean-limit"] = mean_prediction
        seconds = time.perf_counter() - started
        if seconds > 600:
            raise TimeoutError(f"{name} exceeded frozen 600-second CPU cap")
        raw_mse = np.mean(np.square(predictions["static577"] - residual), axis=1)
        augmented_mse = np.mean(
            np.square(predictions["static577Guide71"] - residual), axis=1
        )
        mean_mse = np.mean(np.square(predictions["mean-limit"] - residual), axis=1)
        gain = 1.0 - model_reports["static577Guide71"]["oofRawAllQueryMse"] / model_reports["static577"]["oofRawAllQueryMse"]
        reports[name] = {
            "genes": len(values["genes"]),
            "queries": len(values["query_ids"]),
            "fittingCells": int(values["cell_count"].sum()),
            "distinctGuidePairsPerGene": {
                "minimum": int(values["guide_pair_count"].min()),
                "median": float(np.median(values["guide_pair_count"])),
                "maximum": int(values["guide_pair_count"].max()),
            },
            "guideFeatureSha256": hashlib.sha256(
                values["guide_features"].astype("<f8").tobytes()
            ).hexdigest(),
            "models": model_reports,
            "augmentedGainOverStatic577": gain,
            "passesOnePercentGain": bool(gain >= 0.01),
            "seconds": seconds,
        }
        arrays[f"{name}_gene_ids"] = values["genes"]
        arrays[f"{name}_fold"] = np.asarray(
            [ridge.global_gene_fold(gene, 731) for gene in values["genes"]], np.int8
        )
        arrays[f"{name}_static577_mse"] = raw_mse
        arrays[f"{name}_static577_guide71_mse"] = augmented_mse
        arrays[f"{name}_mean_limit_mse"] = mean_mse
        arrays[f"{name}_cell_count"] = values["cell_count"]
        arrays[f"{name}_guide_pair_count"] = values["guide_pair_count"]
        print(json.dumps({"context": name, **reports[name]}), flush=True)
    per_gene_path = output / "oof-per-gene-errors.npz"
    np.savez_compressed(per_gene_path, **arrays)
    gate = {
        "k562OnePercentGain": reports["k562"]["passesOnePercentGain"],
        "rpe1OnePercentGain": reports["rpe1"]["passesOnePercentGain"],
    }
    gate["passes"] = bool(all(gate.values()))
    (output / "source").mkdir(exist_ok=True)
    for source, destination in (
        (Path(__file__).resolve(), output / "source/runner.py"),
        (RIDGE_CORE, output / "source/count_static_ridge.py"),
        (GUIDE_CORE, output / "source/guide_composition.py"),
    ):
        shutil.copyfile(source, destination)
    report = {
        "schema": "slp.replogle-guide-composition-ridge-fitting-cv-report/v1",
        "protocolSha256": sha256(output / "protocol.json"),
        "contexts": reports,
        "gate": gate,
        "oofPerGeneErrors": {
            "path": per_gene_path.name,
            "sha256": sha256(per_gene_path),
            "definition": "equal-query OOF squared error per fitting gene for each selected model and the fold-local mean limit",
        },
        "source": {
            "runnerSha256": sha256(output / "source/runner.py"),
            "ridgeSha256": sha256(output / "source/count_static_ridge.py"),
            "guideSha256": sha256(output / "source/guide_composition.py"),
        },
        "decision": "fit no development model; fitting-CV gate failed" if not gate["passes"] else "fitting-CV gate passed; any development model requires a separately frozen protocol",
        "developmentOpened": False,
        "testOpened": False,
        "benchmarkAccessed": False,
    }
    write_json(output / "report.json", report)
    write_json(output / "execution-receipt.json", {
        "reportSha256": sha256(output / "report.json"),
        "perGeneErrorsSha256": sha256(per_gene_path),
        "decision": "advance" if gate["passes"] else "reject",
    })
    print(json.dumps({"report": str(output / "report.json"), "gate": gate}))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with threadpool_limits(2):
        {"prepare": prepare, "run": run}[args.mode](args.output)
