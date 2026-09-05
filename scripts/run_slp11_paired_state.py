"""One fixed, point-prediction pilot on paired Frangieh development profiles."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE = ROOT / "modules/slp-1-1-paired-state-v1/paired_model.py"
INFERENCE_SOURCE = MODEL_SOURCE.with_name("inference.py")
HELPER_SOURCE = ROOT / "modules/slp-1-1-world-transition-v1/frangieh_basal_ridge.py"
DATA_SHA = "4bbb1eec9ede66211f1316b2841bb0037032ef975cd6c92d34aba0adb5fed744"
SEED = 731
SETTINGS = {
    "epochs": 180,
    "patience_evaluations": 6,
    "evaluate_every": 5,
    "batch_size": 32,
    "rna_queries_per_step": 1024,
    "basal_rna_tokens": 128,
    "learning_rate": 0.0005,
    "weight_decay": 0.1,
    "max_seconds": 1200,
    "loss_scale_floor": 0.05,
    "feature_clip": 10.0,
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def json_write(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {k: clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return None
        return item

    Path(path).write_text(
        json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def prepare(data_path, feature_path, helper):
    with (
        np.load(data_path, allow_pickle=False) as data,
        np.load(feature_path, allow_pickle=False) as static,
    ):
        if not np.all(data["action_taxon"] == 9606) or not np.all(
            static["entity_taxon"] == 9606
        ):
            raise ValueError("pilot requires species-native human inputs")
        if not np.all(data["rna_observed"]) or not np.all(data["protein_observed"]):
            raise ValueError("pilot requires the complete paired development panel")
        action, context, rna, counts = helper.collapse_gene_profiles(
            data["action_ids"], data["context_ids"], data["rna_targets"]
        )
        pa, pc, protein, pcounts = helper.collapse_gene_profiles(
            data["action_ids"], data["context_ids"], data["protein_targets"]
        )
        if not (
            np.array_equal(action, pa)
            and np.array_equal(context, pc)
            and np.array_equal(counts, pcounts)
        ):
            raise ValueError("paired axes differ")
        split = np.asarray([helper.development_split(gene) for gene in action])
        fitting = np.flatnonzero(split == "train")
        validation = np.flatnonzero(split == "validation")
        if len(set(action[fitting])) != 151 or len(set(action[validation])) != 43:
            raise ValueError("unexpected frozen development population")
        expected_fit_records = set(np.asarray(data["action_ids"])[data["split_train"]])
        expected_val_records = set(
            np.asarray(data["action_ids"])[data["split_validation"]]
        )
        if (
            set(action[fitting]) != expected_fit_records
            or set(action[validation]) != expected_val_records
        ):
            raise ValueError(
                "recomputed gene split disagrees with admitted development indices"
            )
        if set(action[fitting]) & set(action[validation]):
            raise ValueError("intervention split overlap")
        context_names = np.asarray(sorted(set(context)))
        ci = np.asarray([int(np.flatnonzero(context_names == x)[0]) for x in context])
        for c in range(len(context_names)):
            if np.sum(ci[fitting] == c) != 151 or np.sum(ci[validation] == c) != 43:
                raise ValueError(
                    "pilot expects identical gene populations in three contexts"
                )
        control_context_ids = np.asarray(data["control_context_ids"], dtype=str)
        if len(set(control_context_ids)) != len(control_context_ids):
            raise ValueError("control context identities must be unique")
        if set(control_context_ids) != set(context_names):
            raise ValueError("control and perturbation contexts must match exactly")
        control_order = [
            int(np.flatnonzero(control_context_ids == x)[0]) for x in context_names
        ]
        controls = {
            "rna": data["control_rna_targets"][control_order],
            "protein": data["control_protein_targets"][control_order],
        }
        query_ids = {
            "rna": data["rna_query_ids"].astype(str),
            "protein": data["protein_channel_ids"].astype(str),
        }
        feature_ids = static["entity_id"].astype(str)
        if len(set(feature_ids)) != len(feature_ids):
            raise ValueError("duplicate static identity")
        lookup = {x: i for i, x in enumerate(feature_ids)}
        if any(x not in lookup for x in set(action) | set(query_ids["rna"])):
            raise ValueError(
                "new static pack must explicitly represent all action and RNA identities"
            )
        values = np.asarray(static["feature_values"], dtype=np.float32)
        if values.shape[1] != 1156 or not np.isfinite(values).all():
            raise ValueError("unexpected static feature shape or values")
        raw_actions = values[[lookup[x] for x in action]]
        unique_fit_genes = sorted(set(action[fitting]))
        fit_features = values[[lookup[x] for x in unique_fit_genes]].astype(np.float64)
        feature_mean = fit_features.mean(0)
        feature_scale = fit_features.std(0)
        feature_scale[feature_scale < 1e-8] = 1.0

        def standardized(x):
            return np.clip(
                (x - feature_mean) / feature_scale,
                -SETTINGS["feature_clip"],
                SETTINGS["feature_clip"],
            ).astype(np.float32)

        actions = standardized(raw_actions)
        query_features = {
            "rna": standardized(values[[lookup[x] for x in query_ids["rna"]]]),
            "protein": np.eye(len(query_ids["protein"]), dtype=np.float32),
        }
        control_variance = np.var(controls["rna"].astype(np.float64), axis=0)
        # Stable identifiers break exact ties without introducing learned ID features.
        basal_index = np.lexsort((query_ids["rna"], -control_variance))[
            : SETTINGS["basal_rna_tokens"]
        ]
        basal_indices = {
            "rna": basal_index,
            "protein": np.arange(len(query_ids["protein"])),
        }
        targets = {"rna": rna, "protein": protein}
        amplitude, scales, means, basal_values, basal_stats = {}, {}, {}, {}, {}
        for name, y in targets.items():
            residual = y[fitting].astype(np.float64) - controls[name][ci[fitting]]
            amplitude[name] = np.maximum(
                np.sqrt(np.mean(residual**2, axis=0)), 0.05
            ).astype(np.float32)
            scales[name] = np.stack(
                [
                    np.maximum(
                        np.std(y[fitting[ci[fitting] == c]].astype(np.float64), axis=0),
                        0.05,
                    )
                    for c in range(len(context_names))
                ]
            ).astype(np.float32)
            means[name] = np.stack(
                [
                    np.mean(y[fitting[ci[fitting] == c]], axis=0, dtype=np.float64)
                    for c in range(len(context_names))
                ]
            ).astype(np.float32)
            mean = float(controls[name].mean(dtype=np.float64))
            scale = max(float(controls[name].std(dtype=np.float64)), 0.05)
            basal_values[name] = (
                (controls[name][:, basal_indices[name]] - mean) / scale
            ).astype(np.float32)
            basal_stats[name] = np.asarray([mean, scale])
        return {
            "action_ids": action,
            "context_names": context_names,
            "context_index": ci,
            "fitting": fitting,
            "validation": validation,
            "guide_counts": counts,
            "actions": actions,
            "query_features": query_features,
            "query_ids": query_ids,
            "controls": controls,
            "targets": targets,
            "amplitude": amplitude,
            "scales": scales,
            "means": means,
            "basal_indices": basal_indices,
            "basal_values": basal_values,
            "basal_stats": basal_stats,
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
        }


def to_device(prepared):
    def tensor(value):
        return torch.as_tensor(value, device="cuda")

    d = {k: tensor(prepared[k]) for k in ("actions", "context_index")}
    for key in (
        "query_features",
        "controls",
        "targets",
        "amplitude",
        "scales",
        "basal_values",
    ):
        if key in prepared:
            d[key] = {name: tensor(values) for name, values in prepared[key].items()}
    d["basal_features"] = {
        name: d["query_features"][name][tensor(prepared["basal_indices"][name])]
        for name in prepared["query_features"]
    }
    return d


def encoded_batch(model, tensors, rows, *, empty=False):
    ci = tensors["context_index"][rows]
    controls = {
        name: {
            "features": tensors["basal_features"][name],
            "values": values[ci],
            "observed": torch.ones_like(values[ci], dtype=torch.bool),
        }
        for name, values in tensors["basal_values"].items()
    }
    actions = tensors["actions"][rows, None]
    mask = torch.full(actions.shape[:2], not empty, device="cuda", dtype=torch.bool)
    return model.encode(actions, mask, controls), ci


def training_step(model, tensors, rows, rng, module):
    encoded, ci = encoded_batch(model, tensors, rows)
    components = {}
    for name in ("rna", "protein"):
        q = tensors["query_features"][name]
        count = (
            min(SETTINGS["rna_queries_per_step"], len(q)) if name == "rna" else len(q)
        )
        index = torch.as_tensor(
            rng.choice(len(q), size=count, replace=False), device="cuda"
        )
        predicted = model.observe(
            encoded,
            name,
            q[index],
            tensors["controls"][name][ci][:, index],
            tensors["amplitude"][name][index],
        )["mean"]
        target = tensors["targets"][name][rows][:, index]
        scale = tensors["scales"][name][ci][:, index]
        components[name] = module.scaled_mse(
            predicted, target, torch.ones_like(target, dtype=torch.bool), scale
        )
    return (components["rna"] + components["protein"]) / 2


@torch.no_grad()
def forecast(model, tensors, rows, *, empty=False):
    model.eval()
    result = {name: [] for name in ("rna", "protein")}
    for start in range(0, len(rows), 32):
        batch = torch.as_tensor(rows[start : start + 32], device="cuda")
        encoded, ci = encoded_batch(model, tensors, batch, empty=empty)
        for name, parts in result.items():
            q = tensors["query_features"][name]
            chunks = [
                model.observe(
                    encoded,
                    name,
                    q[j : j + 1024],
                    tensors["controls"][name][ci, j : j + 1024],
                    tensors["amplitude"][name][j : j + 1024],
                )["mean"]
                .cpu()
                .numpy()
                for j in range(0, len(q), 1024)
            ]
            parts.append(np.concatenate(chunks, axis=1))
    return {name: np.concatenate(parts) for name, parts in result.items()}


def selection_loss(prediction, prepared):
    val = prepared["validation"]
    ci = prepared["context_index"][val]
    return float(
        np.mean(
            [
                np.mean(
                    (
                        (
                            prediction[name].astype(np.float64)
                            - prepared["targets"][name][val]
                        )
                        / prepared["scales"][name][ci]
                    )
                    ** 2
                )
                for name in prediction
            ]
        )
    )


def save_reference(path, prepared):
    arrays = {
        k: prepared[k] for k in ("context_names", "feature_mean", "feature_scale")
    }
    arrays["feature_clip"] = np.asarray(SETTINGS["feature_clip"], dtype=np.float64)
    for key in (
        "query_features",
        "query_ids",
        "controls",
        "amplitude",
        "scales",
        "means",
        "basal_indices",
        "basal_values",
        "basal_stats",
    ):
        arrays.update({f"{name}_{key}": value for name, value in prepared[key].items()})
    np.savez_compressed(path, **arrays)


def load_inference_inputs(reference_path, feature_path, action_ids, context_index):
    """Rebuild target-free inference inputs from frozen references and raw features."""
    action_ids = np.asarray(action_ids, dtype=str)
    context_index = np.asarray(context_index)
    if context_index.shape != (len(action_ids),) or not np.issubdtype(
        context_index.dtype, np.integer
    ):
        raise ValueError("context indices must be an integer vector aligned to actions")
    with (
        np.load(reference_path, allow_pickle=False) as reference,
        np.load(feature_path, allow_pickle=False) as static,
    ):
        context_names = np.asarray(reference["context_names"], dtype=str)
        if np.any(context_index < 0) or np.any(context_index >= len(context_names)):
            raise ValueError("context index outside frozen reference")
        feature_mean = np.asarray(reference["feature_mean"], dtype=np.float64)
        feature_scale = np.asarray(reference["feature_scale"], dtype=np.float64)
        feature_clip = np.asarray(reference["feature_clip"], dtype=np.float64)
        if (
            feature_clip.shape != ()
            or not np.isfinite(feature_clip)
            or feature_clip <= 0
        ):
            raise ValueError("feature clip must be a finite positive scalar")
        if (
            feature_mean.shape != feature_scale.shape
            or feature_mean.ndim != 1
            or not np.isfinite(feature_mean).all()
            or not np.isfinite(feature_scale).all()
            or not (feature_scale > 0).all()
        ):
            raise ValueError("invalid frozen feature standardization")

        feature_ids = np.asarray(static["entity_id"], dtype=str)
        feature_values = np.asarray(static["feature_values"], dtype=np.float32)
        if (
            len(set(feature_ids)) != len(feature_ids)
            or feature_values.shape != (len(feature_ids), len(feature_mean))
            or not np.isfinite(feature_values).all()
        ):
            raise ValueError("raw static feature pack disagrees with frozen reference")
        lookup = {identity: i for i, identity in enumerate(feature_ids)}

        def standardized(identities):
            if any(identity not in lookup for identity in identities):
                raise ValueError("identity absent from raw static feature pack")
            raw = feature_values[[lookup[identity] for identity in identities]]
            return np.clip(
                (raw - feature_mean) / feature_scale,
                -float(feature_clip),
                float(feature_clip),
            ).astype(np.float32)

        actions = standardized(action_ids)
        query_ids = {
            name: np.asarray(reference[f"{name}_query_ids"], dtype=str)
            for name in ("rna", "protein")
        }
        query_features = {
            name: np.asarray(reference[f"{name}_query_features"], dtype=np.float32)
            for name in ("rna", "protein")
        }
        rebuilt_rna_queries = standardized(query_ids["rna"])
        if not np.array_equal(query_features["rna"], rebuilt_rna_queries):
            raise ValueError("frozen RNA query features disagree with raw feature pack")
        expected_protein = np.eye(len(query_ids["protein"]), dtype=np.float32)
        if not np.array_equal(query_features["protein"], expected_protein):
            raise ValueError(
                "frozen protein query features are not the fixed assay one-hot basis"
            )

        controls, amplitude, basal_indices, basal_values = {}, {}, {}, {}
        for name in ("rna", "protein"):
            controls[name] = np.asarray(reference[f"{name}_controls"], dtype=np.float32)
            amplitude[name] = np.asarray(
                reference[f"{name}_amplitude"], dtype=np.float32
            )
            basal_indices[name] = np.asarray(
                reference[f"{name}_basal_indices"], dtype=np.int64
            )
            basal_values[name] = np.asarray(
                reference[f"{name}_basal_values"], dtype=np.float32
            )
            q = len(query_features[name])
            if controls[name].shape != (len(context_names), q):
                raise ValueError(f"invalid frozen {name} controls")
            if amplitude[name].shape != (q,) or not (amplitude[name] > 0).all():
                raise ValueError(f"invalid frozen {name} amplitude")
            if (
                basal_indices[name].ndim != 1
                or np.any(basal_indices[name] < 0)
                or np.any(basal_indices[name] >= q)
            ):
                raise ValueError(f"invalid frozen {name} basal indices")
            if basal_values[name].shape != (
                len(context_names),
                len(basal_indices[name]),
            ):
                raise ValueError(f"invalid frozen {name} basal values")
            if not all(
                np.isfinite(x).all()
                for x in (
                    controls[name],
                    amplitude[name],
                    query_features[name],
                    basal_values[name],
                )
            ):
                raise ValueError(f"nonfinite frozen {name} inference values")
        return {
            "actions": actions,
            "context_index": context_index.astype(np.int64),
            "query_features": query_features,
            "controls": controls,
            "amplitude": amplitude,
            "basal_indices": basal_indices,
            "basal_values": basal_values,
        }


def runtime_projection(
    seconds_per_step, validation_seconds, fitting_rows, validation_rows
):
    """Return the fixed profiled wall-time contract used before GPU training."""
    if (
        min(seconds_per_step, validation_seconds) <= 0
        or min(fitting_rows, validation_rows) <= 0
    ):
        raise ValueError("runtime measurements and row counts must be positive")
    steps_per_epoch = math.ceil(fitting_rows / SETTINGS["batch_size"])
    epoch_seconds = seconds_per_step * steps_per_epoch
    evaluation_count = SETTINGS["epochs"] // SETTINGS["evaluate_every"]
    full_training = epoch_seconds * SETTINGS["epochs"]
    full_evaluations = validation_seconds * evaluation_count
    first_evaluation = epoch_seconds * SETTINGS["evaluate_every"] + validation_seconds
    # Final selected-checkpoint forecast, frozen-source reload forecast, tiny empty-action
    # check, and a conservative fixed allowance for source/checkpoint loading.
    final_verification = validation_seconds * (2 + min(1.0, 3 / validation_rows)) + 30.0
    bounded_total = (
        min(full_training + full_evaluations, SETTINGS["max_seconds"])
        + final_verification
    )
    return {
        "seconds_per_epoch": epoch_seconds,
        "full_training_projection_seconds": full_training,
        "full_validation_projection_seconds": full_evaluations,
        "first_complete_evaluation_projection_seconds": first_evaluation,
        "final_verification_projection_seconds": final_verification,
        "bounded_total_projection_seconds": bounded_total,
    }


def projected_to_next_evaluation(epoch, epoch_seconds, validation_seconds):
    """Profiled time required before starting an epoch and reaching its next evaluation."""
    epochs = SETTINGS["evaluate_every"] - ((epoch - 1) % SETTINGS["evaluate_every"])
    return epochs * epoch_seconds + validation_seconds


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("explicit CUDA executor unavailable; no fallback")
    if digest(args.data) != DATA_SHA or digest(args.features) != args.features_sha256:
        raise ValueError("input digest mismatch")
    if args.output.exists():
        raise FileExistsError("immutable experiment output already exists")
    helper = load_source(HELPER_SOURCE, "paired_pilot_helper")
    module = load_source(MODEL_SOURCE, "paired_pilot_model")
    prepared = prepare(args.data, args.features, helper)
    config = module.Config(1156, 1156, 20)
    args.output.mkdir(parents=True)
    protocol = {
        "schema": "slp.paired-state-pilot/v1",
        "seed": SEED,
        "settings": SETTINGS,
        "config": asdict(config),
        "hypothesis": "A shared intervention state with RNA and ADT observation heads generalizes to unseen intervention genes in each of three molecular environments.",
        "advancement": "Every context/head: raw MSE at least 1% below fitting mean and each base577/new-physical1156 static baseline; query-centroid-adjusted gene-profile Pearson >=0.10 and no regression against each defined static baseline correlation. This is paired endpoint pilot advancement only.",
        "comparators": "Separate context-local ridge heads with fitting-gene 3-fold CV seed731, alpha grid .1/1/10/100/1000/10000/100000/1000000 plus exact mean limit; both base577 and new physical1156. All six context/head strata must pass; no comparator selected by validation.",
        "selection": "Minimum full development mean of RNA/ADT per-query fitting-SD-scaled MSE; equal genes and contexts. Validation every five epochs; fixed 30-epoch patience.",
        "wall_stop_rule": "Before starting an epoch, reserve profiled training time through the next required full validation plus that validation time within the 1200-second training phase. Preflight requires one complete evaluation and projects the capped phase plus final selected-checkpoint and frozen-source verification below one hour.",
        "population": {
            "fitting_genes": 151,
            "validation_genes": 43,
            "contexts": prepared["context_names"].tolist(),
            "guide_aggregation": "equal guide pseudobulks per gene/context",
        },
        "accessible_modalities": "Static specieswide protein/GO/physical features; quantitative RNA and ADT controls; fitting gene RNA and ADT endpoints. Assay components use fixed one-hot descriptors; no gene-ID embedding.",
        "inputs": {
            "data": {"path": str(args.data.resolve()), "sha256": DATA_SHA},
            "features": {
                "path": str(args.features.resolve()),
                "sha256": args.features_sha256,
            },
        },
        "sources": {
            p.relative_to(ROOT).as_posix(): digest(p)
            for p in (MODEL_SOURCE, INFERENCE_SOURCE, HELPER_SOURCE, Path(__file__))
        },
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
        "limitations": "One adaptive-development seed; endpoint means only; no calibrated uncertainty, identified dynamics, unseen assay component transfer or benefit attributable specifically to joint supervision.",
    }
    json_write(args.output / "protocol.json", protocol)
    code = args.output / "source"
    code.mkdir()
    for source in (MODEL_SOURCE, INFERENCE_SOURCE, HELPER_SOURCE, Path(__file__)):
        shutil.copyfile(source, code / source.name)
    save_reference(args.output / "reference.npz", prepared)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tensors = to_device(prepared)
    model = module.PairedStateModel(config).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=SETTINGS["learning_rate"],
        weight_decay=SETTINGS["weight_decay"],
    )
    rng = np.random.default_rng(SEED)
    batch = torch.as_tensor(
        prepared["fitting"][: SETTINGS["batch_size"]], device="cuda"
    )
    model.train()
    profile_start = time.perf_counter()
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        training_step(model, tensors, batch, rng, module).backward()
        optimizer.step()
    torch.cuda.synchronize()
    seconds_per_step = (time.perf_counter() - profile_start) / 10
    validation_start = time.perf_counter()
    forecast(model, tensors, prepared["validation"])
    torch.cuda.synchronize()
    validation_seconds = time.perf_counter() - validation_start
    projection = runtime_projection(
        seconds_per_step,
        validation_seconds,
        len(prepared["fitting"]),
        len(prepared["validation"]),
    )
    profile = {
        "seconds_per_step": seconds_per_step,
        "full_validation_seconds": validation_seconds,
        **projection,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "profile_updates_discarded": True,
    }
    json_write(args.output / "profile.json", profile)
    if (
        projection["first_complete_evaluation_projection_seconds"]
        > SETTINGS["max_seconds"]
    ):
        raise RuntimeError(
            "profile cannot reach one required evaluation inside the training bound"
        )
    if projection["bounded_total_projection_seconds"] > 3600:
        raise RuntimeError(
            "profile cannot support the bounded local run and final verification"
        )
    del optimizer, model
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = module.PairedStateModel(config).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=SETTINGS["learning_rate"],
        weight_decay=SETTINGS["weight_decay"],
    )
    started = time.perf_counter()
    best, best_epoch, best_state, stale = float("inf"), None, None, 0
    history = []
    for epoch in range(1, SETTINGS["epochs"] + 1):
        elapsed = time.perf_counter() - started
        required = projected_to_next_evaluation(
            epoch,
            projection["seconds_per_epoch"],
            validation_seconds,
        )
        if elapsed + required > SETTINGS["max_seconds"]:
            break
        model.train()
        order = rng.permutation(prepared["fitting"])
        losses = []
        for offset in range(0, len(order), SETTINGS["batch_size"]):
            rows = torch.as_tensor(
                order[offset : offset + SETTINGS["batch_size"]], device="cuda"
            )
            optimizer.zero_grad(set_to_none=True)
            loss = training_step(model, tensors, rows, rng, module)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss ends experiment")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append((float(loss.detach().cpu()), len(rows)))
        elapsed = time.perf_counter() - started
        if (
            epoch % SETTINGS["evaluate_every"] == 0
            or elapsed >= SETTINGS["max_seconds"]
        ):
            predicted = forecast(model, tensors, prepared["validation"])
            score = selection_loss(predicted, prepared)
            if not np.isfinite(score):
                raise RuntimeError("nonfinite selection score ends experiment")
            entry = {
                "epoch": epoch,
                "validation_scaled_mse": score,
                "training_sampled_scaled_mse": sum(x * n for x, n in losses)
                / sum(n for _, n in losses),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(entry)
            with (args.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)
            if score < best:
                best, best_epoch = score, epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stale >= SETTINGS["patience_evaluations"]:
                break
        if time.perf_counter() - started >= SETTINGS["max_seconds"]:
            break
    if best_state is None:
        raise RuntimeError("no complete evaluated checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    save_file(best_state, str(args.output / "model.safetensors"))
    predicted = forecast(model, tensors, prepared["validation"])
    reloaded_source = load_source(code / "paired_model.py", "paired_frozen_reload")
    reloaded = reloaded_source.PairedStateModel(
        reloaded_source.Config(**asdict(config))
    ).cuda()
    reloaded.load_state_dict(load_file(str(args.output / "model.safetensors")))
    val = prepared["validation"]
    if digest(args.features) != args.features_sha256:
        raise RuntimeError("raw feature pack changed before frozen reload verification")
    inference = load_inference_inputs(
        args.output / "reference.npz",
        args.features,
        prepared["action_ids"][val],
        prepared["context_index"][val],
    )
    inference_tensors = to_device(inference)
    inference_rows = np.arange(len(val))
    repeated = forecast(reloaded, inference_tensors, inference_rows)
    drift = max(float(np.max(np.abs(predicted[k] - repeated[k]))) for k in predicted)
    if drift > 1e-6:
        raise RuntimeError("source/weight reload differs")
    empty = forecast(reloaded, inference_tensors, inference_rows[:3], empty=True)
    ci_empty = inference["context_index"][:3]
    if any(
        not np.array_equal(empty[k], inference["controls"][k][ci_empty]) for k in empty
    ):
        raise RuntimeError("empty intervention identity failed")
    arrays = {
        "action_ids": prepared["action_ids"][prepared["validation"]],
        "context_index": prepared["context_index"][prepared["validation"]],
        "context_names": prepared["context_names"],
    }
    report = {
        "schema": "slp.paired-state-pilot-result/v1",
        "best_epoch": best_epoch,
        "selected_scaled_mse": best,
        "training_seconds": time.perf_counter() - started,
        "parameters": sum(x.numel() for x in model.parameters()),
        "source_reload_max_abs_error": drift,
        "target_free_reference_reload": True,
        "empty_intervention_exact": True,
        "history": history,
        "contexts": {},
        "decision": "requires separately pinned comparison to completed static baseline; no advancement yet",
    }
    for index, context in enumerate(prepared["context_names"]):
        rows = np.flatnonzero(arrays["context_index"] == index)
        report["contexts"][str(context)] = {}
        for name in predicted:
            truth = prepared["targets"][name][val[rows]]
            mean = np.broadcast_to(prepared["means"][name][index], truth.shape).copy()
            world_metrics = helper.metrics(
                predicted[name][rows], truth, prepared["scales"][name][index]
            )
            report["contexts"][str(context)][name] = {
                "world": world_metrics,
                "fitting_mean": helper.metrics(
                    mean, truth, prepared["scales"][name][index]
                ),
            }
    for name in predicted:
        arrays[f"{name}_prediction"] = predicted[name]
        arrays[f"{name}_truth"] = prepared["targets"][name][val]
        arrays[f"{name}_query_ids"] = prepared["query_ids"][name]
    np.savez_compressed(args.output / "predictions.npz", **arrays)
    report["artifacts"] = {
        name: digest(args.output / name)
        for name in (
            "protocol.json",
            "profile.json",
            "model.safetensors",
            "reference.npz",
            "predictions.npz",
        )
    }
    json_write(args.output / "report.json", report)
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "report_sha256": digest(args.output / "report.json"),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
