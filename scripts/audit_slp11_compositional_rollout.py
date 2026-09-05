"""Post-hoc audit of locked compositional-state rollout checkpoints.

This script trains and tunes nothing.  It freezes every secondary forecast
before opening held-combination outcomes for descriptive scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules/slp-1-1-compositional-state-v1"
DEFAULT_RUN = ROOT / "results/slp11-transition/norman-observed-composition-seeds731-733-v2"
DATA = ROOT / "data/derived/slp11-norman-author-normalized-v2/norman-2019-author-normalized-development-v2.npz"
FEATURES = ROOT / "data/derived/slp11-norman-static/ensembl116-goa2022-fixed-basis-v1/norman-extended-static-esm-go-features.npz"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def row_pearson(a, b):
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    return np.sum(a * b, 1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)


def metrics(pred, truth, additive):
    return {"mse": float(np.mean((pred - truth) ** 2)),
            "nonadditivePearson": float(row_pearson(pred - additive, truth - additive).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run / "secondary-rollout-audit-v1"
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    dm = load_module("rollout_audit_data", MODULE / "data.py")
    core = load_module("rollout_audit_core", MODULE / "operator.py")
    data = dm.load_compositional_data(DATA, FEATURES)
    protocol = json.loads((args.run / "protocol.json").read_text())
    config = core.Config(**protocol["config"])
    support = data.observed[data.single_rows].all(0)
    pair_count, q = len(data.combination_rows), int(support.sum())
    names = ("operator_predicted_additive", "operator_simultaneous_joint",
             "operator_autonomous", "operator_autonomous_parent_swap",
             "endpoint_predicted_additive", "endpoint_direct_joint",
             "observed_projected_additive")
    forecasts = {f"{name}_{seed}": np.empty((pair_count, q), np.float64)
                 for name in names for seed in protocol["seeds"]}
    action_pairs = np.asarray([data.canonical_actions[row] for row in data.combination_rows])
    for fold in range(3):
        test = np.flatnonzero(data.combination_fold == fold)
        held = data.combination_rows[test]
        left, right = data.combination_single_rows[test].T
        with np.load(args.run / f"fold{fold}-basis.npz", allow_pickle=False) as pack:
            basis, scale = pack["basis"].astype(np.float64), pack["zscale"].astype(np.float64)
            fmean, fscale = pack["feature_mean"].astype(np.float64), pack["feature_scale"].astype(np.float64)
        decoder = scale[:, None] * basis
        features = ((data.gene_features.astype(np.float64) - fmean) / fscale).astype(np.float32)
        actions = features[np.maximum(data.action_feature_index, 0)]
        actions[~data.action_mask] = 0
        actions = torch.from_numpy(actions)
        masks = torch.from_numpy(data.action_mask)
        yleft = data.y[left][:, support].astype(np.float64)
        yright = data.y[right][:, support].astype(np.float64)
        za = torch.from_numpy(((yleft @ basis.T) / scale).astype(np.float32))
        zb = torch.from_numpy(((yright @ basis.T) / scale).astype(np.float32))
        zero = torch.zeros((len(test), config.state_dim))
        observed_projected = (za + zb).numpy() @ decoder
        for seed in protocol["seeds"]:
            models = {}
            for arm in ("endpoint", "observed_operator"):
                model = core.CompositionalStateOperator(config)
                model.load_state_dict(load_file(str(args.run / f"fold{fold}-{arm}-seed{seed}.safetensors"), device="cpu"))
                models[arm] = model.eval()
            with torch.no_grad():
                operator, endpoint = models["observed_operator"], models["endpoint"]
                osa = operator(zero, actions[left], masks[left])
                osb = operator(zero, actions[right], masks[right])
                esa = endpoint(zero, actions[left], masks[left])
                esb = endpoint(zero, actions[right], masks[right])
                values = {
                    "operator_predicted_additive": (osa + osb).numpy() @ decoder,
                    "operator_simultaneous_joint": operator(zero, actions[held], masks[held]).numpy() @ decoder,
                    "operator_autonomous": (0.5 * (operator(osa, actions[right], masks[right]) + operator(osb, actions[left], masks[left]))).numpy() @ decoder,
                    "operator_autonomous_parent_swap": (0.5 * (operator(osa.roll(1, 0), actions[right], masks[right]) + operator(osb.roll(1, 0), actions[left], masks[left]))).numpy() @ decoder,
                    "endpoint_predicted_additive": (esa + esb).numpy() @ decoder,
                    "endpoint_direct_joint": endpoint(zero, actions[held], masks[held]).numpy() @ decoder,
                    "observed_projected_additive": observed_projected,
                }
            for name, value in values.items():
                forecasts[f"{name}_{seed}"][test] = value
    for name in names:
        forecasts[f"{name}_ensemble"] = np.mean([forecasts[f"{name}_{seed}"] for seed in protocol["seeds"]], axis=0)
    frozen = output / "frozen-secondary-forecasts.npz"
    np.savez_compressed(frozen, **forecasts)
    freeze = {"schema": "slp11-compositional-rollout-secondary-freeze/v1",
              "postHoc": True, "trainingOrTuning": False,
              "primaryDecisionPreserved": json.loads((args.run / "report.json").read_text())["decision"],
              "sourceProtocolSHA256": sha(args.run / "protocol.json"),
              "sourceForecastSHA256": sha(args.run / "frozen-forecasts.npz"),
              "forecastSHA256": sha(frozen), "names": sorted(forecasts)}
    write(output / "forecast-freeze.json", freeze)

    # Held outcomes are opened only after the secondary forecast artifact is frozen.
    saved = np.load(frozen)
    truth = data.y[data.combination_rows][:, support].astype(np.float64)
    primary = np.load(args.run / "frozen-forecasts.npz")
    additive = primary["additive"]
    scored = {name: metrics(saved[name], truth, additive) for name in saved.files}
    folds = {str(f): {name: metrics(saved[name][data.combination_fold == f], truth[data.combination_fold == f], additive[data.combination_fold == f]) for name in saved.files} for f in range(3)}
    # Cluster-style gene bootstrap: resampled gene multiplicities weight both
    # constituents of each fixed pair.  It is descriptive because genes and
    # folds are shared and this audit was motivated by the observed result.
    genes = np.unique(action_pairs)
    gene_index = {gene: i for i, gene in enumerate(genes)}
    pi = np.asarray([[gene_index[x] for x in pair] for pair in action_pairs])
    rng = np.random.default_rng(947)
    contrasts = {"autonomous_vs_operator_predicted_additive": [],
                 "autonomous_vs_endpoint_direct_joint": []}
    auto_error = np.mean((saved["operator_autonomous_ensemble"] - truth) ** 2, 1)
    comparisons = {"autonomous_vs_operator_predicted_additive": np.mean((saved["operator_predicted_additive_ensemble"] - truth) ** 2, 1),
                   "autonomous_vs_endpoint_direct_joint": np.mean((saved["endpoint_direct_joint_ensemble"] - truth) ** 2, 1)}
    for _ in range(10000):
        count = np.bincount(rng.integers(0, len(genes), len(genes)), minlength=len(genes))
        weight = count[pi[:, 0]] + count[pi[:, 1]]
        if weight.sum():
            for name, baseline_error in comparisons.items():
                contrasts[name].append(1 - np.average(auto_error, weights=weight) / np.average(baseline_error, weights=weight))
    summary = {}
    for key, baseline in (("operatorPredictedAdditive", "operator_predicted_additive_ensemble"),
                          ("endpointDirectJoint", "endpoint_direct_joint_ensemble"),
                          ("endpointPredictedAdditive", "endpoint_predicted_additive_ensemble")):
        summary[key] = 1 - scored["operator_autonomous_ensemble"]["mse"] / scored[baseline]["mse"]
    report = {"schema": "slp11-compositional-rollout-secondary-report/v1",
              "interpretation": "Descriptive post-hoc mechanism audit; preserves the primary rejection and creates no advancement claim.",
              "primaryDecision": freeze["primaryDecisionPreserved"], "relativeMSEGain": summary,
              "metrics": scored, "folds": folds,
              "geneConditionalBootstrap95": {name: np.quantile(value, [0.025, 0.975]).tolist() for name, value in contrasts.items()},
              "bootstrapLimitation": "Gene-multiplicity weighted descriptive bootstrap; shared pair endpoints and folds remain dependent.",
              "forecastSHA256": freeze["forecastSHA256"]}
    write(output / "report.json", report)
    print(json.dumps({"primaryDecision": report["primaryDecision"], "relativeMSEGain": summary,
                      "autonomousMSE": scored["operator_autonomous_ensemble"]["mse"]}, sort_keys=True))


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
