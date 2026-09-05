"""Human molecular development with explicit measured basal state.

Consumes the train/validation bundle only. Test-only data and SL benchmarks are
not accepted by this entry point. The learned network remains query-based.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from exposure_uncertainty import fit_exposure_uncertainty
from response_queries import fit_query_response_descriptors
from safetensors.torch import save_file
from threadpoolctl import threadpool_limits
from train import gene_metrics, sha, write_json
from transition_calibration import fit_grouped_oof_mean, fit_grouped_oof_ridge
from transition_model import Config, TransitionWorld, gaussian_loss


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    started = time.monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if sha(args.data) != args.data_sha256:
        raise ValueError("development artifact checksum mismatch")
    with np.load(args.data, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if len(data["split_test"]) or not len(data["split_train"]) or not len(data["split_validation"]):
        raise ValueError("only development train/validation bundle is allowed")
    with np.load(args.features, allow_pickle=False) as archive:
        feature_keys = list(zip(archive["entity_taxon"].tolist(), archive["entity_id"].tolist()))
        feature_rows = archive["feature_values"].astype(np.float32)
    if len(feature_keys) != len(set(feature_keys)) or not np.isfinite(feature_rows).all():
        raise ValueError("static feature identities or values invalid")
    lookup = dict(zip(feature_keys, feature_rows))
    x = np.stack([lookup[(9606, str(gene))] for gene in data["action_ids"]])
    query = np.stack([lookup[(9606, str(gene))] for gene in data["query_ids"]])
    y, mask = data["targets"], data["observed"]
    value_space = str(data["target_value_space"].item()) if "target_value_space" in data else "log2"
    context = data["context_index"]
    train, validation = data["split_train"], data["split_validation"]
    if set(data["action_ids"][train]) & set(data["action_ids"][validation]):
        raise ValueError("intervention overlap across development partitions")
    keys = [(9606, str(gene)) for gene in data["action_ids"]]
    source_dir = output / "source"
    source_dir.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(path, source_dir / path.name)
    protocol = {
        "scope": "human held-gene development in recorded assay contexts; no test access or launch claim",
        "data_sha256": sha(args.data), "features_sha256": sha(args.features),
        "args": vars(args), "source_hashes": {p.name: sha(p) for p in source_dir.glob("*.py")},
        "split_counts": {"train": len(train), "validation": len(validation)},
        "feature_dimensions": x.shape[1], "query_count": y.shape[1],
        "value_space": value_space,
        "context": "measured core-control expression when available; panel selected by control difference only",
        "scale_calibration": "gene-grouped OOF fitting residuals",
        "rule": {"delta_nats_against_mean_and_ridge": 0.02, "adjusted_pearson": 0.10,
                 "protected_context_nonregression": True},
        "test_accessed": False, "benchmark_accessed": False,
        "runtime": {name:importlib.metadata.version(name) for name in
                    ("torch","numpy","scipy","scikit-learn","safetensors","h5py","threadpoolctl")},
    }
    write_json(output / "protocol.json", protocol)
    print(json.dumps({"event": "protocol-frozen", "rows": len(y), "queries": y.shape[1], "features": x.shape[1]}), flush=True)
    contexts = range(len(data["context_ids"]))
    means, ridges = [], []
    oof_mean, oof_ridge = np.empty_like(y[train], dtype=np.float64), np.empty_like(y[train], dtype=np.float64)
    for c in contexts:
        rows = train[context[train] == c]
        positions = np.flatnonzero(context[train] == c)
        context_keys = [keys[i] for i in rows]
        mean, oof_mean[positions] = fit_grouped_oof_mean(y[rows], mask[rows], context_keys, scale_floor=0.05, return_oof=True)
        ridge, oof_ridge[positions] = fit_grouped_oof_ridge(x[rows], y[rows], mask[rows], context_keys, args.ridge_alpha, scale_floor=0.05, return_oof=True)
        means.append(mean)
        ridges.append(ridge)
        print(json.dumps({"event": "baselines-fit", "context": str(data["context_ids"][c]), "records": len(rows)}), flush=True)
    exposure = {}
    if args.exposure_aware:
        control_args = {"control_targets":data["control_targets"], "control_observed":data["control_observed"],
                        "control_num_cells":data["control_num_cells_filtered"], "control_context_index":data["control_context_index"]}
        for name, oof in (("mean",oof_mean),("ridge",oof_ridge)):
            exposure[name] = fit_exposure_uncertainty(y[train]-oof, mask[train],
                data["num_cells_filtered"][train], context[train], **control_args, scale_floor=0.05)
    # Retain exact fitted centroids for scoring: rounding the reference alone
    # produces spurious correlations for the identically-zero mean residual.
    references = np.stack([model.intercept_ for model in means])
    reference_scales = np.stack([model.residual_scale_.values for model in means])
    forecasts = references[context].copy()
    if args.reference_kind == "ridge":
        for c in contexts:
            rows = np.flatnonzero(context == c)
            forecasts[rows] = ridges[c].predict(x[rows])
        # Training corrections target unseen-gene forecasts, not in-sample
        # ridge residuals. Validation/inference uses the full training ridge.
        forecasts[train] = oof_ridge
        reference_scales = np.stack([model.residual_scale_.values for model in ridges])
    fmean, fstd = x[train].mean(0), x[train].std(0)
    fstd = np.where(fstd > 1e-5, fstd, 1.0)
    query_mean, query_std = fmean.copy(), fstd.copy()
    response_info = None
    if args.query_basis_rank:
        if not mask[train].all():
            raise ValueError("response-query SVD currently requires complete training observations")
        descriptors, response_info = fit_query_response_descriptors(
            y[train], context[train], references, reference_scales,
            rank=args.query_basis_rank, seed=args.seed)
        query = np.concatenate((query, descriptors), axis=1)
        query_mean = np.concatenate((query_mean, np.zeros(args.query_basis_rank)))
        query_std = np.concatenate((query_std, np.ones(args.query_basis_rank)))
    # The same measured context is supplied in fitting and inference.
    basal = data["context_basal_expression"] if "context_basal_expression" in data else data["basal_control"]
    selected = np.argsort(-basal.var(0), kind="stable")[:args.context_tokens]
    basal_normalized = (basal - basal.mean(1, keepdims=True)) / np.maximum(basal.std(1, keepdims=True), 1e-5)
    if not torch.cuda.is_available() and args.device == "cuda":
        raise RuntimeError("requested CUDA unavailable; no fallback")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float32, device=device)
    xt, qt, yt = tensor((x-fmean)/fstd), tensor((query-query_mean)/query_std), tensor(y)
    mt = torch.as_tensor(mask, device=device)
    rt, st = tensor(forecasts), tensor(reference_scales)
    context_values = tensor(basal_normalized[:, selected])
    context_features = qt[selected]
    exposure_scales = tensor(exposure[args.reference_kind].scales(data["num_cells_filtered"], context)) if exposure else None
    config = Config(x.shape[1], hidden=args.hidden, state_dim=args.state_dim,
                    covariance_rank=args.covariance_rank, dropout=args.dropout, learn_scale=False,
                    query_feature_dim=query.shape[1])
    model = TransitionWorld(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = np.random.default_rng(args.seed)

    def forward(rows):
        c = context[rows]
        prediction = model(xt[rows], qt, rt[rows], st[c],
                     context_features=context_features[None].expand(len(rows), -1, -1),
                     context_values=context_values[c],
                     context_mask=torch.ones((len(rows), len(selected)), dtype=torch.bool, device=device))
        # Measurement exposure affects likelihood only, never the mean/state.
        if exposure_scales is not None:
            prediction["scale"] = exposure_scales[rows]
        return prediction

    def predict(rows):
        model.eval()
        predictions, scales = [], []
        with torch.no_grad():
            for offset in range(0, len(rows), args.batch_size):
                pred = forward(rows[offset:offset+args.batch_size])
                variance = pred["scale"].square()
                if "factor" in pred:
                    variance = variance + pred["factor"].square().sum(-1)
                predictions.append(pred["mean"].cpu().numpy())
                scales.append(variance.sqrt().cpu().numpy())
        return np.concatenate(predictions), np.concatenate(scales)

    def metrics_for(predictions, scales, rows):
        reports = {}
        for c in contexts:
            select = np.flatnonzero(context[rows] == c)
            actual = rows[select]
            reports[str(data["context_ids"][c])] = gene_metrics(
                predictions[select], y[actual], mask[actual], [keys[i] for i in actual],
                references[c], scales[select], value_space=value_space,
            )
        return reports

    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, args.epochs+1):
        if time.monotonic()-started >= args.max_seconds:
            break
        model.train()
        order = generator.permutation(train)
        losses = []
        for offset in range(0, len(order), args.batch_size):
            rows = order[offset:offset+args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            pred = forward(rows)
            loss = gaussian_loss(pred, yt[rows], mt[rows], joint=args.covariance_rank > 0)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        pred, scales = predict(validation)
        reports = metrics_for(pred, scales, validation)
        score = float(np.mean([r["gene_macro_nll"] for r in reports.values()]))
        history.append({"epoch":epoch, "train_nll":float(np.mean(losses)), "validation":reports})
        if score < best-1e-5:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({"event":"epoch", "epoch":epoch, "nll":score,
                              "adjusted_pearson":{k:v["gene_macro_profile_centroid_adjusted_pearson_mean"] for k,v in reports.items()},
                              "seconds":round(time.monotonic()-started,1)}), flush=True)
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("no checkpoint within time cap")
    model.load_state_dict(best_state)
    pred, scales = predict(validation)
    results = metrics_for(pred, scales, validation)
    for c in contexts:
        name = str(data["context_ids"][c])
        rows = validation[context[validation] == c]
        result = {"world": results[name]}
        for label, baseline in (("mean",means[c]),("ridge",ridges[c])):
            baseline_scale = exposure[label].scales(data["num_cells_filtered"][rows], context[rows]) if exposure else baseline.residual_scale_.values
            result[label] = gene_metrics(baseline.predict(x[rows]), y[rows], mask[rows], [keys[i] for i in rows], references[c], baseline_scale, value_space=value_space)
        result["world_delta_vs_mean"] = result["mean"]["gene_macro_nll"] - result["world"]["gene_macro_nll"]
        result["world_delta_vs_ridge"] = result["ridge"]["gene_macro_nll"] - result["world"]["gene_macro_nll"]
        result["development_rule_passed"] = (
            min(result["world_delta_vs_mean"], result["world_delta_vs_ridge"]) >= 0.02
            and (result["world"]["gene_macro_profile_centroid_adjusted_pearson_mean"] or 0) >= 0.10)
        results[name] = result
    save_file(best_state, str(output/"model.safetensors"))
    np.savez_compressed(output/"development-predictions.npz", mean=pred, scale=scales,
                        record_ids=data["record_ids"][validation], action_ids=data["action_ids"][validation],
                        context_index=context[validation])
    write_json(output/"model-config.json", asdict(config))
    if exposure:
        components = {
            f"{name}_{component}":getattr(estimator, component+"_")
            for name, estimator in exposure.items() for component in ("biological_variance", "sampling_variance")}
        for component in ("biological_variance", "sampling_variance"):
            components["world_"+component] = getattr(exposure[args.reference_kind], component+"_")
        np.savez_compressed(output/"exposure-uncertainty.npz", **components)
    if args.reference_kind == "ridge":
        np.savez_compressed(output/"linear-reference.npz", coefficient=np.stack([r.coef_ for r in ridges]),
                            feature_mean=np.stack([r.feature_mean_ for r in ridges]),
                            feature_std=np.stack([r.feature_scale_ for r in ridges]),
                            intercept=np.stack([r.intercept_ for r in ridges]))
    np.savez_compressed(output/"reference.npz", feature_mean=fmean, feature_std=fstd,
                        query_feature_mean=query_mean, query_feature_std=query_std,
                        query_features=query,
                        reference=references, reference_scale=reference_scales,
                        context_query_indices=selected, context_features=query[selected],
                        context_values=basal_normalized[:,selected], context_ids=data["context_ids"],
                        query_ids=data["query_ids"])
    report = {"protocol":protocol,"model_config":asdict(config),"best_epoch":best_epoch,
              "parameters":sum(p.numel() for p in model.parameters()), "results":results,
              "history":history,"response_basis":response_info,"elapsed_seconds":time.monotonic()-started,
              "development_rule_passed":all(r["development_rule_passed"] for r in results.values()),
              "exposure_uncertainty":{name:{"provenance":est.component_provenance,
                  "identifiability_warning":est.identifiability_warning,
                  "sampling_from_controls_fraction":float(est.sampling_from_controls_.mean()),
                  "scale_floor":est.scale_floor} for name,est in exposure.items()},
              "checkpoint_sha256":sha(output/"model.safetensors")}
    write_json(output/"report.json",report)
    print(json.dumps({"event":"finished","best_epoch":best_epoch,"results":{
        k:{"world_nll":v["world"]["gene_macro_nll"],"ridge_nll":v["ridge"]["gene_macro_nll"],
           "world_r":v["world"]["gene_macro_profile_centroid_adjusted_pearson_mean"],
           "ridge_r":v["ridge"]["gene_macro_profile_centroid_adjusted_pearson_mean"]} for k,v in results.items()}}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",required=True)
    parser.add_argument("--data-sha256",default="82904b7b52ab34d71e94abb2311c93a420321697d53eab12dabae5b247376f75")
    parser.add_argument("--features",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--epochs",type=int,default=180)
    parser.add_argument("--patience",type=int,default=30)
    parser.add_argument("--max-seconds",type=int,default=1800)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--context-tokens",type=int,default=64)
    parser.add_argument("--query-basis-rank",type=int,default=0)
    parser.add_argument("--exposure-aware",action="store_true")
    parser.add_argument("--reference-kind",choices=("mean","ridge"),default="mean")
    parser.add_argument("--hidden",type=int,default=128)
    parser.add_argument("--state-dim",type=int,default=64)
    parser.add_argument("--covariance-rank",type=int,default=0)
    parser.add_argument("--dropout",type=float,default=0.2)
    parser.add_argument("--learning-rate",type=float,default=0.0005)
    parser.add_argument("--weight-decay",type=float,default=0.1)
    parser.add_argument("--ridge-alpha",type=float,default=10000)
    parser.add_argument("--seed",type=int,default=731)
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        run(args)


if __name__=="__main__":
    main()
