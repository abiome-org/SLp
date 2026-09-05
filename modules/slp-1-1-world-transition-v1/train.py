"""Explicit local exploratory training; original protected snapshots are absent.

Outputs are development artifacts, not an OMF release. Every run uses a fresh
output directory, records its source/input hashes, and leaves the internal test
partition unscored unless explicitly requested after candidate lock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from threadpoolctl import threadpool_limits
from transition_baselines import compare_paired_nll, evaluate, fit_mean, fit_ridge
from transition_calibration import fit_grouped_oof_mean, fit_grouped_oof_ridge
from transition_data import load_corpus, split_by_gene
from transition_model import Config, TransitionWorld, gaussian_loss


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(x) for x in item]
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    Path(path).write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def feature_values(corpus, path):
    if path is None:
        return corpus["entity_features"], {"kind": "sequence-composition-21"}
    with np.load(path, allow_pickle=False) as archive:
        keys = list(zip(archive["entity_taxon"].tolist(), archive["entity_id"].tolist()))
        values = archive["feature_values"].astype(np.float32)
    if len(keys) != len(set(keys)) or values.ndim != 2 or values.shape[0] != len(keys):
        raise ValueError("feature identity/shape mismatch")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite static features")
    lookup = dict(zip(keys, values))
    required = set(corpus["action_index"].tolist()) | set(corpus["query_entity_index"].tolist())
    result = np.zeros((len(corpus["entity_keys"]), values.shape[1]), np.float32)
    for index in required:
        key = corpus["entity_keys"][index]
        if key not in lookup:
            raise ValueError(f"required composite feature identity missing: {key}")
        result[index] = lookup[key]
    return result, {"path": str(Path(path).resolve()), "sha256": sha(path), "dimensions": values.shape[1]}


def collapse_genes(prediction, truth, observed, keys, scale):
    """Average record diagnostics within gene; retain scale per original record."""
    groups = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    # Per-gene scores aggregate original records, without pretending a replicate
    # average has the same measurement variance as a single record.
    return groups


def gene_metrics(pred, target, mask, keys, reference, scale, *, value_space="log2"):
    reports = []
    scale = np.broadcast_to(scale, pred.shape)
    for rows in collapse_genes(pred, target, mask, keys, scale).values():
        reports.append(evaluate(pred[rows], target[rows], mask[rows], reference, scale[rows], value_space=value_space))
    result = evaluate(pred, target, mask, reference, scale, value_space=value_space)
    for metric in ("nll", "mse", "profile_pearson_mean", "profile_centroid_adjusted_pearson_mean"):
        values = [r[metric] for r in reports if np.isfinite(r[metric])]
        result["gene_macro_" + metric] = float(np.mean(values)) if values else None
    result["intervention_genes"] = len(reports)
    return result


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    started = time.monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    corpus = load_corpus(args.corpus)
    indices = split_by_gene(corpus["action_keys"], seed=731)
    features, feature_info = feature_values(corpus, args.features)
    x = features[corpus["action_index"]]
    q = features[corpus["query_entity_index"]]
    y, observed = corpus["targets"], corpus["observed"]
    train, validation = indices["train"], indices["validation"]
    source_dir = output / "source"
    source_dir.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(path, source_dir / path.name)
    source_hashes = {p.name: sha(p) for p in sorted(source_dir.glob("*.py"))}
    protocol = {
        "scope": "fitting-corpus-only exploratory development; not release certification",
        "hypothesis": "nonlinear transition beats fitting mean and feature-linear ridge on unseen development intervention genes",
        "rule": {"nll_delta_against_each": 0.02, "centroid_adjusted_pearson": 0.10},
        "args": vars(args), "source_hashes": source_hashes, "corpus_sha256": sha(args.corpus),
        "features": feature_info, "split_seed": 731,
        "split_counts": {k: len(v) for k, v in indices.items()},
        "split_index_hashes": {k: hashlib.sha256(np.asarray(v, dtype="<i8").tobytes()).hexdigest() for k, v in indices.items()},
        "torch": torch.__version__, "numpy": np.__version__,
        "protected_original_truth_accessed": False, "internal_test_scored": args.evaluate_test,
        "scale_calibration": args.scale_calibration,
    }
    write_json(output / "protocol.json", protocol)
    print(json.dumps({"event": "protocol-frozen", "output": str(output), "split_counts": protocol["split_counts"], "features": features.shape[1]}), flush=True)
    training_keys = [corpus["action_keys"][i] for i in train]
    if args.scale_calibration == "oof":
        mean = fit_grouped_oof_mean(y[train], observed[train], training_keys, scale_floor=0.05)
    else:
        mean = fit_mean(y[train], observed[train], scale_floor=0.05)
    baseline_reports = {}
    fitted_ridges = []
    for alpha in args.ridge_alphas:
        if args.scale_calibration == "oof":
            ridge = fit_grouped_oof_ridge(x[train], y[train], observed[train], training_keys, alpha, scale_floor=0.05)
        else:
            ridge = fit_ridge(x[train], y[train], observed[train], alpha, scale_floor=0.05)
        prediction = ridge.predict(x[validation])
        metrics = gene_metrics(prediction, y[validation], observed[validation], [corpus["action_keys"][i] for i in validation], mean.intercept_, ridge.residual_scale_.values)
        baseline_reports[str(alpha)] = metrics
        fitted_ridges.append((metrics["gene_macro_nll"], ridge))
        print(json.dumps({"event": "ridge", "alpha": alpha, "nll": metrics["gene_macro_nll"], "pearson": metrics["gene_macro_profile_centroid_adjusted_pearson_mean"]}), flush=True)
    ridge = min(fitted_ridges, key=lambda item: item[0])[1]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no executor fallback")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    # Normalization is fit on training action features; held outcomes never enter.
    feature_mean = x[train].mean(0)
    feature_std = x[train].std(0)
    feature_std = np.where(feature_std > 1e-5, feature_std, 1.0)
    def tensor(array):
        return torch.as_tensor(array, device=device, dtype=torch.float32)
    xt = tensor((x - feature_mean) / feature_std)
    qt = tensor((q - feature_mean) / feature_std)
    yt = tensor(y)
    mt = torch.as_tensor(observed, device=device)
    reference = tensor(mean.intercept_)
    reference_scale = tensor(mean.residual_scale_.values)
    config = Config(features.shape[1], hidden=args.hidden, state_dim=args.state_dim,
                    covariance_rank=args.covariance_rank, dropout=args.dropout,
                    learn_scale=not args.fixed_scale)
    model = TransitionWorld(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = np.random.default_rng(args.seed)
    best = float("inf")
    best_state, best_epoch, stale = None, 0, 0
    history = []

    def predict(rows):
        model.eval()
        means, scales = [], []
        with torch.no_grad():
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start:start + args.batch_size]
                pred = model(xt[batch], qt, reference, reference_scale)
                variance = pred["scale"].square()
                if "factor" in pred:
                    variance = variance + pred["factor"].square().sum(-1)
                means.append(pred["mean"].cpu().numpy())
                scales.append(variance.sqrt().cpu().numpy())
        return np.concatenate(means), np.concatenate(scales)

    for epoch in range(1, args.epochs + 1):
        if time.monotonic() - started > args.max_seconds:
            print(json.dumps({"event": "wall-time-cap", "epoch": epoch}), flush=True)
            break
        model.train()
        losses = []
        order = generator.permutation(train)
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xt[batch], qt, reference, reference_scale)
            loss = gaussian_loss(prediction, yt[batch], mt[batch], joint=args.covariance_rank > 0)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        prediction, scale = predict(validation)
        metrics = gene_metrics(prediction, y[validation], observed[validation], [corpus["action_keys"][i] for i in validation], mean.intercept_, scale)
        score = metrics["gene_macro_nll"]
        history.append({"epoch": epoch, "train_nll": float(np.mean(losses)), "validation": metrics})
        if score < best - 1e-5:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({"event": "epoch", "epoch": epoch, "nll": score, "pearson": metrics["gene_macro_profile_centroid_adjusted_pearson_mean"], "seconds": round(time.monotonic() - started, 1)}), flush=True)
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("no trained checkpoint within wall-time cap")
    model.load_state_dict(best_state)
    save_file(best_state, str(output / "model.safetensors"))
    np.savez_compressed(output / "reference.npz", feature_mean=feature_mean, feature_std=feature_std,
                        reference=mean.intercept_, reference_scale=mean.residual_scale_.values,
                        ridge_coef=ridge.coef_, ridge_intercept=ridge.intercept_,
                        ridge_feature_mean=ridge.feature_mean_, ridge_feature_scale=ridge.feature_scale_,
                        ridge_scale=ridge.residual_scale_.values)
    report = {"protocol": protocol, "model_config": asdict(config), "parameters": sum(p.numel() for p in model.parameters()),
              "best_epoch": best_epoch, "ridge_alpha": ridge.alpha, "ridge_validation_grid": baseline_reports,
              "history": history, "results": {}, "elapsed_seconds": time.monotonic() - started}
    partitions = ["validation"] + (["test"] if args.evaluate_test else [])
    for partition in partitions:
        rows = indices[partition]
        prediction, scale = predict(rows)
        mean_pred, ridge_pred = mean.predict(x[rows]), ridge.predict(x[rows])
        keys = [corpus["action_keys"][i] for i in rows]
        report["results"][partition] = {
            "world": gene_metrics(prediction, y[rows], observed[rows], keys, mean.intercept_, scale),
            "mean": gene_metrics(mean_pred, y[rows], observed[rows], keys, mean.intercept_, mean.residual_scale_.values),
            "ridge": gene_metrics(ridge_pred, y[rows], observed[rows], keys, mean.intercept_, ridge.residual_scale_.values),
            "world_vs_mean": compare_paired_nll(prediction, mean_pred, y[rows], observed[rows], scale, mean.residual_scale_.values),
            "world_vs_ridge": compare_paired_nll(prediction, ridge_pred, y[rows], observed[rows], scale, ridge.residual_scale_.values),
        }
        result = report["results"][partition]
        world_nll = result["world"]["gene_macro_nll"]
        adjusted = result["world"]["gene_macro_profile_centroid_adjusted_pearson_mean"]
        result["development_rule"] = {
            "mean_delta_nats": result["mean"]["gene_macro_nll"] - world_nll,
            "ridge_delta_nats": result["ridge"]["gene_macro_nll"] - world_nll,
            "minimum_delta_nats": 0.02,
            "minimum_adjusted_pearson": 0.10,
            "passed": bool(adjusted is not None and adjusted >= 0.10 and all(
                result[name]["gene_macro_nll"] - world_nll >= 0.02 for name in ("mean", "ridge")
            )),
            "program_or_launch_advancement": False,
        }
    report["checkpoint_sha256"] = sha(output / "model.safetensors")
    write_json(output / "report.json", report)
    write_json(output / "model-config.json", asdict(config))
    print(json.dumps({"event": "finished", "output": str(output), "best_epoch": best_epoch, "results": report["results"]}, default=str), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--features")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--covariance-rank", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fixed-scale", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ridge-alphas", type=float, nargs="+", default=[100.0, 1000.0, 10000.0])
    parser.add_argument("--scale-calibration", choices=["oof", "fitting"], default="oof")
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args)


if __name__ == "__main__":
    main()
