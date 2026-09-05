"""Fitting-only control-coexpression feature ridge ablation for K562/RPE1."""
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
OUTPUT = ROOT / "results/slp11-transition/replogle-control-coexpression-ridge-fitting-cv-v1"
RIDGE_CORE = ROOT / "modules/slp-1-1-count-static-ridge-v1/count_static_ridge.py"
COEXPRESSION_DIR = ROOT / "data/derived/slp11-human-control-coexpression/static577-gaussian64-leave-self-out-v1"
COEXPRESSION_MANIFEST = COEXPRESSION_DIR / "manifest.json"
COEXPRESSION_REPORT = ROOT / "results/slp11-transition/human-control-coexpression-reliability-v1/report.json"
COEXPRESSION_REPLAY = ROOT / "results/slp11-transition/human-control-coexpression-reliability-v1/coverage-and-replay.json"

CONTEXTS = {
    "k562": {
        "moments": ROOT / "data/derived/slp11-human-k562-essential-fitting-action-moments-v1/fitting-action-moments.npz",
        "static": ROOT / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1/k562-essential-count-static577.npz",
        "baseline": ROOT / "results/slp11-transition/k562-essential-count-anchored-static-ridge-seed731-v1/model.npz",
        "coexpression": COEXPRESSION_DIR / "k562-control-coexpression64.npz",
    },
    "rpe1": {
        "moments": ROOT / "data/derived/slp11-human-rpe1-essential-raw-cells-v1/fitting-action-moments.npz",
        "static": ROOT / "data/derived/slp11-human-rpe1-essential-count-static/ensembl116-esm8m-shared-go-v1/rpe1-essential-count-static577.npz",
        "baseline": ROOT / "results/slp11-transition/rpe1-essential-count-anchored-static-ridge-seed731-v1/model.npz",
        "coexpression": COEXPRESSION_DIR / "rpe1-control-coexpression64.npz",
    },
}

PINS = {
    RIDGE_CORE: "1032eeff59382fae3874da9a389033192e113e0f5ac2c8d01f09f8441d969e62",
    COEXPRESSION_MANIFEST: "2f600edd401cdf964b31995e7ef8c2566864f3a78af3db6c0e0f3fbd8840a7df",
    COEXPRESSION_REPORT: "6f90802b9090fefae5775002328170d8aee85206549fa29f669ff9937303f473",
    COEXPRESSION_REPLAY: "dcb1e566074cb3ede5532769c842431d2823e6600ffd2bb827b7133b28545d59",
    CONTEXTS["k562"]["moments"]: "a1f44a15a42c5b56e4ce897fde6ebba97298fc296105c6c870ee0e740331694e",
    CONTEXTS["k562"]["static"]: "6706f8867adedef8822897bc275ea90680584f84afd24771e4beb3c8ecf07659",
    CONTEXTS["k562"]["baseline"]: "dbb669d2eb8d844ec9be7c88a2ed21f5592de434d1b2e916412bda4a52fe1cf3",
    CONTEXTS["k562"]["coexpression"]: "dc4932ef22733619ba2a4aa07b1f469598d28e2178aabbbedda038d068c10912",
    CONTEXTS["rpe1"]["moments"]: "d15def86aead06b0bc75ab63c77513735ec7c57d65012bff72f3947bc654895c",
    CONTEXTS["rpe1"]["static"]: "621e1e9f0dffc740ef42382b1b2898f629edd5037e8a02d411e8d30e815ed816",
    CONTEXTS["rpe1"]["baseline"]: "bd144e36b5618c6225828501492edfa5449cef07442041c1d1cc20645b1473bc",
    CONTEXTS["rpe1"]["coexpression"]: "2875081a0e5417b57698632b096af261fbed7e86e279ba94b293b187af75706d",
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


def join_action_coexpression(
    gene_ids: np.ndarray,
    action_ids: np.ndarray,
    action_features: np.ndarray,
    action_query_present: np.ndarray,
) -> np.ndarray:
    """Join 64 control-only coordinates plus one measured-query flag."""
    genes = np.asarray(gene_ids).astype(str)
    actions = np.asarray(action_ids).astype(str)
    features = np.asarray(action_features, dtype=np.float32)
    present = np.asarray(action_query_present)
    if (
        genes.ndim != 1
        or actions.ndim != 1
        or features.shape != (len(actions), 64)
        or present.shape != (len(actions),)
        or present.dtype != np.bool_
        or len(set(genes.tolist())) != len(genes)
        or len(set(actions.tolist())) != len(actions)
        or not np.isfinite(features).all()
        or np.any(features[~present] != 0)
    ):
        raise ValueError("invalid control-coexpression action pack")
    lookup = {value: row for row, value in enumerate(actions)}
    missing = [gene for gene in genes if gene not in lookup]
    if missing:
        raise ValueError(f"fitting gene absent from action roster: {missing[0]}")
    rows = np.asarray([lookup[gene] for gene in genes], np.int64)
    result = np.concatenate(
        (features[rows], present[rows, None].astype(np.float32)), axis=1
    )
    if result.shape != (len(genes), 65) or not np.isfinite(result).all():
        raise AssertionError("control-coexpression feature join drift")
    return result


def cross_validated_predictions(core, genes, features, target):
    """Return separately selected alpha, all scores, folds, and OOF errors."""
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
    for fold, state in enumerate(states):
        held = folds == fold
        prediction[held] = core.predict_residual(state, x[held], selected)
    per_gene_mse = np.mean(np.square(prediction - y), axis=1)
    if not np.isclose(per_gene_mse.mean(), scores[selected], rtol=1e-12, atol=1e-15):
        raise AssertionError("expanded OOF prediction does not match selected score")
    return selected, scores, reports, folds, per_gene_mse


def protocol() -> dict[str, object]:
    return {
        "schema": "slp.replogle-control-coexpression-ridge-fitting-cv-protocol/v1",
        "hypothesis": "A control-only coexpression fingerprint adds held-fitting-gene aggregate-mean signal beyond static577 protein/GO features.",
        "advancementRule": "Static577+control-coexpression65 must reduce three-fold fitting-gene OOF full-query MSE by at least 1% relative to static577 in both K562 and RPE1. If either fails, fit no development model and perform no neural feature tuning.",
        "population": "All 1,443 K562 and 1,666 RPE1 fitting action genes; equal weight per gene and native query.",
        "endpoint": "ln1p(equal-cell mean CP10k) minus each gene's fitting-cell-GEM-weighted reconstruction-training NT control anchor; identical to the frozen static577 ridge endpoint.",
        "features": {
            "controlCoexpression65": "64 leave-self-out Pearson coordinates computed only from reconstruction-training NT cells, plus one exact measured-query presence flag; missing measured action queries are zero with presence false",
            "static577": "raw ESM8M320, protein-presence1, and shared GO MF/CC256",
            "augmented642": "static577 concatenated with control-coexpression65; no guide features, gene ID, perturbed outcome, cell count, library size, or efficacy feature",
        },
        "crossValidation": "Identical seed731 global three-fold assignment, fold-local feature normalization, unpenalized residual intercept, and alpha grid .1 through 1e6 plus mean-limit; select alpha separately for each context/model by equal-gene all-query OOF MSE.",
        "compute": {"cpuThreads": 2, "maximumSecondsPerContext": 600},
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
            raise ValueError("frozen control-coexpression CV protocol changed")
    else:
        write_json(path, frozen)
    return frozen


def context_inputs(name: str, ridge) -> dict[str, np.ndarray]:
    paths = CONTEXTS[name]
    with np.load(paths["moments"], allow_pickle=False) as values:
        genes = values["action_ids"].astype(str)
        query_ids = values["query_ids"].astype(str)
        cell_count = np.asarray(values["cell_count"], np.int64)
        target = ridge.response_from_cp10k_moments(
            values["cp10k_sum"], cell_count
        )
        gem_cell_count = np.asarray(values["gem_cell_count"], np.int64)
    static = load_npz(paths["static"])
    static_lookup = {
        value: row for row, value in enumerate(static["entity_id"].astype(str))
    }
    raw_static = np.asarray(
        static["feature_values"][[static_lookup[gene] for gene in genes]], np.float32
    )
    coexpression = load_npz(paths["coexpression"])
    control_features = join_action_coexpression(
        genes,
        coexpression["action_ids"],
        coexpression["action_features"],
        coexpression["action_query_present"],
    )
    baseline = load_npz(paths["baseline"])
    if not np.array_equal(query_ids, baseline["query_ids"].astype(str)):
        raise ValueError(f"{name} query axes differ")
    anchor = ridge.control_anchor(baseline["basal_rate"], gem_cell_count)
    return {
        "genes": genes,
        "query_ids": query_ids,
        "cell_count": cell_count,
        "raw_static": raw_static,
        "control_features": control_features,
        "residual": target - anchor,
    }


def run(output: Path = OUTPUT) -> dict[str, object]:
    prepare(output)
    if (output / "report.json").exists():
        raise FileExistsError("immutable control-coexpression CV already complete")
    ridge = load_module(RIDGE_CORE, "control_coexpression_ridge")
    reports, arrays = {}, {}
    for name in ("k562", "rpe1"):
        started = time.perf_counter()
        values = context_inputs(name, ridge)
        augmented = np.concatenate(
            (values["raw_static"], values["control_features"]), axis=1
        )
        model_reports = {}
        for model, features in (
            ("static577", values["raw_static"]),
            ("static577ControlCoexpression65", augmented),
        ):
            selected, scores, folds, assignment, per_gene = cross_validated_predictions(
                ridge, values["genes"], features, values["residual"]
            )
            model_reports[model] = {
                "selectedAlpha": selected,
                "oofRawAllQueryMse": scores[selected],
                "candidateOofRawAllQueryMse": scores,
                "folds": folds,
            }
            arrays[f"{name}_{model}_mse"] = per_gene
            arrays[f"{name}_fold"] = assignment
        seconds = time.perf_counter() - started
        if seconds > 600:
            raise TimeoutError(f"{name} exceeded frozen CPU cap")
        gain = 1.0 - model_reports["static577ControlCoexpression65"]["oofRawAllQueryMse"] / model_reports["static577"]["oofRawAllQueryMse"]
        present = values["control_features"][:, -1].astype(bool)
        reports[name] = {
            "genes": len(values["genes"]),
            "queries": len(values["query_ids"]),
            "fittingCells": int(values["cell_count"].sum()),
            "controlCoexpressionPresentGenes": int(present.sum()),
            "controlCoexpressionAbsentGenes": int((~present).sum()),
            "controlFeatureSha256": hashlib.sha256(
                values["control_features"].astype("<f4").tobytes()
            ).hexdigest(),
            "models": model_reports,
            "augmentedGainOverStatic577": gain,
            "passesOnePercentGain": bool(gain >= 0.01),
            "seconds": seconds,
        }
        arrays[f"{name}_gene_ids"] = values["genes"]
        arrays[f"{name}_cell_count"] = values["cell_count"]
        arrays[f"{name}_control_coexpression_present"] = present
        print(json.dumps({"context": name, **reports[name]}), flush=True)
    path = output / "oof-per-gene-errors.npz"
    np.savez_compressed(path, **arrays)
    gate = {
        "k562OnePercentGain": reports["k562"]["passesOnePercentGain"],
        "rpe1OnePercentGain": reports["rpe1"]["passesOnePercentGain"],
    }
    gate["passes"] = bool(all(gate.values()))
    (output / "source").mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__).resolve(), output / "source/runner.py")
    shutil.copyfile(RIDGE_CORE, output / "source/count_static_ridge.py")
    report = {
        "schema": "slp.replogle-control-coexpression-ridge-fitting-cv-report/v1",
        "protocolSha256": sha256(output / "protocol.json"),
        "contexts": reports,
        "gate": gate,
        "oofPerGeneErrors": {
            "path": path.name,
            "sha256": sha256(path),
            "definition": "equal-query selected-model OOF squared error per fitting gene, with shared global fold assignments",
        },
        "source": {
            "runnerSha256": sha256(output / "source/runner.py"),
            "ridgeSha256": sha256(output / "source/count_static_ridge.py"),
        },
        "decision": "fit no development model; fitting-CV gate failed" if not gate["passes"] else "fitting-CV gate passed; a development model requires a separate frozen protocol",
        "developmentOpened": False,
        "testOpened": False,
        "benchmarkAccessed": False,
    }
    write_json(output / "report.json", report)
    write_json(output / "execution-receipt.json", {
        "reportSha256": sha256(output / "report.json"),
        "perGeneErrorsSha256": sha256(path),
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
