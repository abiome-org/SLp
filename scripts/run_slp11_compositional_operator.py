"""Fixed fitting-only test of observed-background molecular composition."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules/slp-1-1-compositional-state-v1"
DATA = ROOT / "data/derived/slp11-norman-author-normalized-v2/norman-2019-author-normalized-development-v2.npz"
FEATURES = ROOT / "data/derived/slp11-norman-static/ensembl116-goa2022-fixed-basis-v1/norman-extended-static-esm-go-features.npz"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def ridge(x, y, xnew, alpha=100.0):
    mean_x, mean_y = x.mean(0), y.mean(0)
    xc = x - mean_x
    return (xnew - mean_x) @ xc.T @ np.linalg.solve(xc @ xc.T + alpha * np.eye(len(x)), y - mean_y) + mean_y


def pearson_rows(a, b):
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    return np.sum(a * b, 1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)


def score(prediction, truth, additive):
    residual_pred, residual_true = prediction - additive, truth - additive
    return {
        "mse": float(np.mean((prediction - truth) ** 2)),
        "nonadditivePearson": float(pearson_rows(residual_pred, residual_true).mean()),
        "centeredNonadditivePearson": float(pearson_rows(residual_pred - residual_pred.mean(0), residual_true - residual_true.mean(0)).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dm = load("composition_data", MODULE / "data.py")
    core = load("composition_operator", MODULE / "operator.py")
    data = dm.load_compositional_data(DATA, FEATURES)
    # Support is determined by the original singles, not by held double values.
    support = data.observed[data.single_rows].all(0)
    if not np.all(data.observed[:, support]):
        raise ValueError("pilot requires a common observed panel")
    config = core.Config()
    protocol = {
        "hypothesis": "Training observed single-state to double-state edges teaches a reusable background-dependent molecular action operator beyond simultaneous endpoint fitting.",
        "advancement": "Three-seed mean observed-background forecast must reduce pooled MSE >=5% versus each fixed baseline including the matched endpoint model, not regress versus the best pooled baseline in any fold, have positive centered nonadditive correlation, and worsen >=1% when conditioning parents are cyclically swapped while the additive anchor remains correct.",
        "scope": "Known-gene held-combination interpolation in K562 CRISPRa, not held-gene transfer, time-course dynamics, SL, or independent confirmation.",
        "source": "GEO:GSE133344 Norman2019, Homo sapiens NCBI:9606, day5 endpoint",
        "data": {"path": str(DATA.relative_to(ROOT)), "sha256": dm.DATA_SHA256},
        "features": {"path": str(FEATURES.relative_to(ROOT)), "sha256": dm.FEATURE_SHA256},
        "modalities": "Static ESM320 + protein presence1 + GO256; observed standardized RNA pseudobulk states; no SL labels.",
        "rights": "rights/norman-2019-geo-public-molecular.yaml",
        "target": data.target_value_space,
        "aggregation": data.aggregation,
        "singles": len(data.single_rows), "pairs": len(data.combination_rows), "queries": int(support.sum()),
        "folds": data.combination_fold.tolist(), "seeds": [731, 732, 733],
        "config": asdict(config), "updates": 1000, "learningRate": 0.001,
        "optimizer": "AdamW weight_decay=0.01; clip_grad_norm=1; full-batch; fixed last checkpoint",
        "objective": "Equal single and pair class weight; operator pair loss averages simultaneous endpoint and both observed-parent edges. Endpoint arm uses simultaneous pairs only.",
        "basis": "Uncentered rank32 SVD of fold-fitting singles and doubles, per-coordinate RMS scaling; zero is matched control.",
        "readout": "Observed additive singles plus model-predicted nonadditive difference, decoded through fold-fitting basis. Full autonomous rollout reported separately.",
        "baselines": ["additive observed singles", "training mean nonadditive correction", "scalar-weighted additive", "ridge symmetric single-state features", "matched v1-style endpoint attention"],
        "budgetSeconds": 2700, "nativeResearchOnly": True,
        "code": {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), MODULE / "data.py", MODULE / "operator.py"]},
        "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "mhaFastpath": False},
    }
    write(args.output / "protocol.json", protocol)
    # No adaptive epoch or hyperparameter selection; all OOF forecasts precede scoring.
    torch.set_num_threads(4)
    # PyTorch 2.11 fused eval attention diverged from CPU by 4.6e-4 on
    # a trained fitting-state probe; the unfused path agreed within 1.2e-6.
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no silent executor fallback")
    device = "cuda"
    features = data.gene_features.astype(np.float64)
    fmean = features.mean(0)
    fscale = np.maximum(features.std(0), 0.05)
    features = ((features - fmean) / fscale).astype(np.float32)
    actions = features[np.maximum(data.action_feature_index, 0)]
    actions[~data.action_mask] = 0
    all_actions = torch.tensor(actions, device=device)
    all_mask = torch.tensor(data.action_mask, device=device)
    predictions = {}
    pair_count, query_count = len(data.combination_rows), int(support.sum())
    for name in ["additive", "mean_residual", "weighted_additive", "state_ridge"]:
        predictions[name] = np.empty((pair_count, query_count))
    for seed in protocol["seeds"]:
        for arm in ("endpoint", "observed_operator"):
            predictions[f"{arm}_{seed}"] = np.empty((pair_count, query_count))
        predictions[f"swapped_{seed}"] = np.empty((pair_count, query_count))
        predictions[f"autonomous_{seed}"] = np.empty((pair_count, query_count))
    training = []
    started = time.monotonic()
    for fold in range(3):
        fit, held = data.fold_rows(fold)
        train_pair = np.flatnonzero(data.combination_fold != fold)
        test_pair = np.flatnonzero(data.combination_fold == fold)
        fit_pairs = data.combination_rows[train_pair]
        fit_y = data.y[fit][:, support].astype(np.float64)
        _, _, vt = np.linalg.svd(fit_y, full_matrices=False)
        basis = vt[:32]
        zscale = np.maximum(np.sqrt(np.mean((fit_y @ basis.T) ** 2, 0)), 1e-4)
        decoder = zscale[:, None] * basis
        # Only fitting outcome rows are projected for training.
        z = np.zeros((len(data.y), 32), dtype=np.float32)
        z[fit] = ((fit_y @ basis.T) / zscale).astype(np.float32)
        zt = torch.tensor(z, device=device)
        zero_s = torch.zeros((len(data.single_rows), 32), device=device)
        zero_p = torch.zeros((len(fit_pairs), 32), device=device)
        pleft, pright = data.combination_single_rows[train_pair].T
        hleft, hright = data.combination_single_rows[test_pair].T
        singles_y = data.y[data.single_rows][:, support].astype(np.float64)
        yleft = data.y[hleft][:, support].astype(np.float64)
        yright = data.y[hright][:, support].astype(np.float64)
        additive = yleft + yright
        train_additive = data.y[pleft][:, support] + data.y[pright][:, support]
        pair_truth = data.y[fit_pairs][:, support].astype(np.float64)
        residual = pair_truth - train_additive
        predictions["additive"][test_pair] = additive
        predictions["mean_residual"][test_pair] = additive + residual.mean(0)
        weight = np.sum(train_additive * pair_truth) / np.maximum(np.sum(train_additive ** 2), 1e-12)
        predictions["weighted_additive"][test_pair] = weight * additive
        def symmetric(left, right):
            return np.concatenate((z[left] + z[right], z[left] * z[right], np.abs(z[left] - z[right])), 1).astype(np.float64)
        predictions["state_ridge"][test_pair] = additive + ridge(symmetric(pleft, pright), residual, symmetric(hleft, hright))
        np.savez_compressed(args.output / f"fold{fold}-basis.npz", basis=basis, zscale=zscale, feature_mean=fmean, feature_scale=fscale, query_ids=data.query_ids[support], fit_rows=fit, held_rows=held)
        for seed in protocol["seeds"]:
            for arm in ("endpoint", "observed_operator"):
                torch.manual_seed(seed)
                model = core.CompositionalStateOperator(config).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
                model.train()
                tick = time.monotonic()
                updates = 30 if args.profile else protocol["updates"]
                for step in range(updates):
                    if time.monotonic() - started > protocol["budgetSeconds"]:
                        raise TimeoutError("fixed pilot compute ceiling reached")
                    ps = model(zero_s, all_actions[data.single_rows], all_mask[data.single_rows])
                    pp = model(zero_p, all_actions[fit_pairs], all_mask[fit_pairs])
                    single_loss = torch.mean((ps - zt[data.single_rows]) ** 2)
                    pair_loss = torch.mean((pp - zt[fit_pairs]) ** 2)
                    if arm == "observed_operator":
                        ab = model(zt[pleft], all_actions[pright], all_mask[pright])
                        ba = model(zt[pright], all_actions[pleft], all_mask[pleft])
                        pair_loss = (pair_loss + torch.mean((ab - zt[fit_pairs]) ** 2) + torch.mean((ba - zt[fit_pairs]) ** 2)) / 3
                    loss = 0.5 * (single_loss + pair_loss)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                torch.cuda.synchronize()
                elapsed = time.monotonic() - tick
                entry = {"fold": fold, "seed": seed, "arm": arm, "seconds": elapsed, "updates": updates, "loss": float(loss.detach()), "parameters": sum(p.numel() for p in model.parameters())}
                training.append(entry)
                print(json.dumps(entry), flush=True)
                if args.profile:
                    if arm == "observed_operator":
                        write(args.output / "profile.json", {"arms": training, "projectedFullSeconds": sum(x["seconds"] for x in training) * 1000 / 30 * 9})
                        return
                    continue
                model.eval()
                save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, args.output / f"fold{fold}-{arm}-seed{seed}.safetensors")
                with torch.no_grad():
                    hzero = torch.zeros((len(held), 32), device=device)
                    sa = model(hzero, all_actions[hleft], all_mask[hleft])
                    sb = model(hzero, all_actions[hright], all_mask[hright])
                    if arm == "endpoint":
                        joint = model(hzero, all_actions[held], all_mask[held])
                        interaction = joint - sa - sb
                    else:
                        ab = model(zt[hleft], all_actions[hright], all_mask[hright])
                        ba = model(zt[hright], all_actions[hleft], all_mask[hleft])
                        interaction = 0.5 * (ab - zt[hleft] - sb + ba - zt[hright] - sa)
                        swap_a, swap_b = zt[hleft].roll(1, 0), zt[hright].roll(1, 0)
                        swapped = 0.5 * (model(swap_a, all_actions[hright], all_mask[hright]) - swap_a - sb + model(swap_b, all_actions[hleft], all_mask[hleft]) - swap_b - sa)
                        predictions[f"swapped_{seed}"][test_pair] = additive + swapped.cpu().numpy() @ decoder
                        auto = 0.5 * (model(sa, all_actions[hright], all_mask[hright]) + model(sb, all_actions[hleft], all_mask[hleft]))
                        predictions[f"autonomous_{seed}"][test_pair] = auto.cpu().numpy() @ decoder
                    predictions[f"{arm}_{seed}"][test_pair] = additive + interaction.cpu().numpy() @ decoder
                # Saved tensor replay is checked on actual fitting states, in CPU isolation from CUDA.
                cpu = core.CompositionalStateOperator(config)
                from safetensors.torch import load_file
                cpu.load_state_dict(load_file(str(args.output / f"fold{fold}-{arm}-seed{seed}.safetensors")))
                cpu.eval()
                with torch.no_grad():
                    actual = cpu(zt[data.single_rows[:3]].cpu(), all_actions[data.single_rows[:3]].cpu(), all_mask[data.single_rows[:3]].cpu())
                    expected = model(zt[data.single_rows[:3]], all_actions[data.single_rows[:3]], all_mask[data.single_rows[:3]]).cpu()
                    entry["cpuReplayMaxError"] = float((actual - expected).abs().max())
                    if entry["cpuReplayMaxError"] > 1e-4:
                        raise RuntimeError("CPU artifact replay failed")
                del model, optimizer, cpu
    for arm in ("endpoint", "observed_operator", "swapped", "autonomous"):
        predictions[f"{arm}_ensemble"] = np.mean([predictions[f"{arm}_{seed}"] for seed in protocol["seeds"]], axis=0)
    np.savez_compressed(args.output / "frozen-forecasts.npz", **predictions)
    write(args.output / "forecast-freeze.json", {"protocolSHA256": sha(args.output / "protocol.json"), "forecastsSHA256": sha(args.output / "frozen-forecasts.npz"), "training": training})
    # First held-combination scoring occurs after every forecast is fixed.
    truth = data.y[data.combination_rows][:, support].astype(np.float64)
    additive = predictions["additive"]
    metrics = {name: score(value, truth, additive) for name, value in predictions.items()}
    folds = {str(fold): {name: score(value[data.combination_fold == fold], truth[data.combination_fold == fold], additive[data.combination_fold == fold]) for name, value in predictions.items()} for fold in range(3)}
    baselines = ["additive", "mean_residual", "weighted_additive", "state_ridge", "endpoint_ensemble"]
    best = min(baselines, key=lambda name: metrics[name]["mse"])
    candidate = "observed_operator_ensemble"
    gain = 1 - metrics[candidate]["mse"] / metrics[best]["mse"]
    swap_regression = metrics["swapped_ensemble"]["mse"] / metrics[candidate]["mse"] - 1
    passed = gain >= 0.05 and all(folds[str(f)][candidate]["mse"] <= folds[str(f)][best]["mse"] for f in range(3)) and metrics[candidate]["centeredNonadditivePearson"] > 0 and swap_regression >= 0.01
    rng = np.random.default_rng(731)
    baseline_errors = np.mean((predictions[best] - truth) ** 2, axis=1)
    candidate_errors = np.mean((predictions[candidate] - truth) ** 2, axis=1)
    samples = rng.integers(0, pair_count, size=(10000, pair_count))
    gains = 1 - candidate_errors[samples].mean(1) / baseline_errors[samples].mean(1)
    report = {"decision": "advance" if passed else "reject", "bestBaseline": best, "relativeMSEGain": gain, "conditioningSwapMSERegression": swap_regression, "pairBootstrap95": np.quantile(gains, [0.025, 0.975]).tolist(), "bootstrapLimitation": "Conditional descriptive interval; shared genes and fitting folds induce dependence.", "metrics": metrics, "folds": folds, "training": training, "seconds": time.monotonic() - started, "protocolSHA256": sha(args.output / "protocol.json"), "forecastsSHA256": sha(args.output / "frozen-forecasts.npz")}
    write(args.output / "report.json", report)
    print(json.dumps({k: report[k] for k in ("decision", "bestBaseline", "relativeMSEGain", "conditioningSwapMSERegression", "seconds")}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
