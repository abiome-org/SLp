"""Matched count-only and molecular-mean continuations of the K562 count pilot."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "results/slp11-transition/k562-essential-count-latent-state-seed731-v1"
ORIGINAL_FINAL = ROOT / "results/slp11-transition/k562-essential-count-latent-state-seed731-portable-finalization-v2"
BASELINE = ROOT / "results/slp11-transition/k562-essential-count-anchored-static-ridge-seed731-v1"
MOMENTS = ROOT / "data/derived/slp11-human-k562-essential-fitting-action-moments-v1/fitting-action-moments.npz"
STATIC = ROOT / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1/k562-essential-count-static577.npz"
BASE_RUNNER = ROOT / "scripts/run_slp11_k562_count_latent_state.py"
OBJECTIVE = ROOT / "modules/slp-1-1-molecular-mean-objective-v1/molecular_mean_objective.py"
INFERENCE = ROOT / "modules/slp-1-1-count-latent-continuation-inference-v1/inference.py"
VERIFY = ROOT / "scripts/verify_slp11_k562_count_latent_continuation.py"
OUTPUT = ROOT / "results/slp11-transition/k562-essential-count-latent-mean-aux-continuation-seed1731-v1"

SEED = 1731
MEAN_SEED = 1732
UPDATES = 4000
BATCH = 128
MEAN_GENES = 16
MEAN_WEIGHT = 0.1
MAX_SECONDS_PER_ARM = 900
TRAINING_ANCHORED_MEAN_MSE = 0.004324449194506417
MODEL_CONFIG = {
    "feature_dim": 577,
    "hidden_dim": 128,
    "state_dim": 32,
    "key_dim": 64,
    "dropout": 0.1,
}
ARMS = ("count-only", "mean-aux")

PINS = {
    ORIGINAL / "model.safetensors": "c7cc6a369f8b63d936c535f7cc59439fec38033202d4b98616b02270df74f3f8",
    ORIGINAL / "reference.npz": "8020753e9e2597b08cb94c5351772be05986b286f61e0f7a26be26fbfabae4f6",
    ORIGINAL / "source/count_latent_state.py": "75df347a82151074c0ce6f4c732106e70ed17126aff07d017294894421d30bac",
    ORIGINAL / "FROZEN-BEFORE-DEVELOPMENT-V3.json": "a8936f7e6a8f1ebed65a91ce4b91d8f375c5cf61b1fe300ddb1c8b681eb57208",
    ORIGINAL_FINAL / "report.json": "62a8cb6a766ac3eb0b8767d8905178c407cfe87cc53e7e153954572e51470bbb",
    BASE_RUNNER: "9d6668ceb61a3bb0b9dc540a42430b523632b86ddcf547ec2175bfb2fe155920",
    OBJECTIVE: "f9dc1fc1d7c6f1071f5bdb98e45a5140116cb583975bf3a76892814883989cd9",
    MOMENTS: "a1f44a15a42c5b56e4ce897fde6ebba97298fc296105c6c870ee0e740331694e",
    STATIC: "6706f8867adedef8822897bc275ea90680584f84afd24771e4beb3c8ecf07659",
    BASELINE / "model.npz": "dbb669d2eb8d844ec9be7c88a2ed21f5592de434d1b2e916412bda4a52fe1cf3",
    BASELINE / "FROZEN-BEFORE-DEVELOPMENT.json": "a57e4d406be62f1ad3c41736f119cde780c20d30c4e7b02e465f265c36fb296f",
    BASELINE / "protocol.json": "9ec8520d7c47ecb37f40b4f06f8a54f13f05a34fdced6b2ecf359ac88fa30f0b",
    BASELINE / "source/count_static_ridge.py": "1032eeff59382fae3874da9a389033192e113e0f5ac2c8d01f09f8441d969e62",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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


def verify_pins() -> None:
    for path, expected in PINS.items():
        if sha256(path) != expected:
            raise ValueError(f"frozen input mismatch: {path}")


def draw_mean_rows(generator: np.random.Generator, genes: int) -> np.ndarray:
    """Draw 16 unique fitting genes uniformly, independent of count sampling."""
    if genes < MEAN_GENES:
        raise ValueError("mean objective requires at least 16 fitting genes")
    return np.asarray(generator.choice(genes, size=MEAN_GENES, replace=False), np.int64)


def anchored_mean_mse(
    cp10k_sum: np.ndarray,
    cell_count: np.ndarray,
    gem_cell_count: np.ndarray,
    basal_rate: np.ndarray,
    target_mean: np.ndarray,
) -> float:
    """Return the all-fitting-gene scalar used only to normalize mean loss."""
    totals = np.asarray(cp10k_sum, dtype=np.float64)
    cells = np.asarray(cell_count, dtype=np.float64)
    weights = np.asarray(gem_cell_count, dtype=np.float64)
    basal = np.asarray(basal_rate, dtype=np.float64)
    offset = np.asarray(target_mean, dtype=np.float64)
    if (
        totals.ndim != 2
        or cells.shape != (len(totals),)
        or weights.ndim != 2
        or weights.shape[0] != len(totals)
        or basal.shape != (weights.shape[1], totals.shape[1])
        or offset.shape != (totals.shape[1],)
        or not np.isfinite(totals).all()
        or not np.isfinite(cells).all()
        or not np.isfinite(weights).all()
        or not np.isfinite(basal).all()
        or not np.isfinite(offset).all()
        or np.any(cells <= 0)
        or np.any(weights < 0)
        or np.any(weights.sum(1) <= 0)
        or np.any(basal <= 0)
    ):
        raise ValueError("invalid anchored-mean fitting arrays")
    weights = weights / weights.sum(1, keepdims=True)
    truth = np.log1p(totals / cells[:, None])
    prediction = np.log1p(weights @ basal) + offset
    return float(np.mean(np.square(prediction - truth)))


def per_gene_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    anchor: np.ndarray,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return gene MSE and stable independently query-centered profile r."""
    pred = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    reference = np.asarray(anchor, dtype=np.float64)
    if pred.shape != target.shape or reference.shape != pred.shape or pred.ndim != 2:
        raise ValueError("prediction, truth, and anchor must share [G,Q]")
    if not np.isfinite(pred).all() or not np.isfinite(target).all() or not np.isfinite(reference).all():
        raise ValueError("profile arrays must be finite")
    mse = np.mean(np.square(pred - target), axis=1)
    pred = pred - reference
    target = target - reference
    pred = pred - pred[:1]
    target = target - target[:1]
    pred = pred - pred.mean(0, keepdims=True)
    target = target - target.mean(0, keepdims=True)
    pred = pred - pred.mean(1, keepdims=True)
    target = target - target.mean(1, keepdims=True)
    numerator = np.sum(pred * target, axis=1)
    denominator = np.sqrt(np.sum(pred * pred, axis=1) * np.sum(target * target, axis=1))
    correlation = np.full(len(pred), np.nan, dtype=np.float64)
    defined = denominator > tolerance
    correlation[defined] = numerator[defined] / denominator[defined]
    return mse, correlation


def reconstruction_loss(report: dict[str, object]) -> float:
    """Read the frozen nested reconstruction schema without silent fallback."""
    try:
        value = float(report["strata"]["all"]["lossMean"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid fitting-reconstruction report schema") from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError("reconstruction loss must be positive finite")
    return value


def protocol() -> dict[str, object]:
    return {
        "schema": "slp.k562-count-latent-mean-aux-continuation-protocol/v1",
        "hypothesis": "A fitting-only molecular aggregate-mean loss closes the frozen count prior's fitting gap while preserving its reconstruction likelihood, and improves held-gene molecular means over the matched anchored mean and static ridge.",
        "advancementRule": "For the mean-aux arm over all 305 development genes and 8,563 queries: MSE at least 1% below both anchored mean and static ridge; independently query-centered anchor-residual profile Pearson at least .10 and not below ridge; reconstruction-held four-antithetic ELBO no more than 1% above both the original checkpoint and matched count-only continuation. The count-only arm is a matched additional-training control.",
        "accessibleModalities": [
            "reconstruction-training fitting/control integer counts and source-panel libraries",
            "fitting-only gene aggregate CP10k sums, cell counts, and GEM composition",
            "reconstruction-training NT control rates by GEM",
            "fitting-gene-normalized static577 action/query features",
            "development identity and GEM composition only until all forecasts freeze",
        ],
        "excluded": ["development outcomes until the common forecast freeze", "excluded test counts", "benchmark outcomes"],
        "modelConfig": MODEL_CONFIG,
        "continuation": {
            "initialCheckpointSha256": PINS[ORIGINAL / "model.safetensors"],
            "arms": {
                "count-only": "unchanged beta-one normalized count ELBO",
                "mean-aux": "same count ELBO plus 0.1 times fitting aggregate profile MSE divided by the fixed all-fitting anchored-mean MSE",
            },
            "updatesPerArm": UPDATES,
            "batch": BATCH,
            "countSampling": "64 GEM-uniform controls and 64 fitting-gene/sgID_AB-population-uniform targets; independent identical RNG seed 1731 in both arms",
            "meanSampling": "16 unique uniformly sampled fitting genes per update using separate seed 1732; all 48 GEM controls and all 8,563 queries",
            "optimizer": "new AdamW per arm, lr .0005, weight decay .01, gradient clip1",
            "checkpoint": "exact final update only; no early stopping or tuning",
            "maximumSecondsPerArm": MAX_SECONDS_PER_ARM,
        },
        "meanObjective": {
            "endpoint": "ln1p of GEM-cell-weighted conditional-prior expected CP10k, matched to each fitting gene's ln1p equal-cell mean CP10k",
            "modelMode": "eval mode with gradients for the auxiliary forecast, then restore training mode; no dropout in the molecular endpoint",
            "weight": MEAN_WEIGHT,
            "normalizer": TRAINING_ANCHORED_MEAN_MSE,
            "normalizerDefinition": "exact full-fitting 1,443-gene x 8,563-query MSE of frozen all-fitting anchored mean; not cross-validation and not minibatch-renormalized",
        },
        "fittingDiagnostic": "fixed 128 fitting genes by SHA256(slp11-count-prior-fit-audit-v1|ENSG), evaluated before development and never used to select a checkpoint",
        "development": "freeze both arm forecasts, shared anchor, anchored mean, and ridge before opening development count members; aggregate truth once; score arms together; persist per-gene MSE and centered profile r",
        "reconstructionDiagnostic": "same four deterministic antithetic draws over the 21,900 reconstruction-held fitting/control cells used for the original checkpoint diagnostic",
        "seeds": {"trainingAndCountRows": SEED, "meanGenes": MEAN_SEED},
        "pins": {str(path.relative_to(ROOT)): value for path, value in PINS.items()},
        "runnerSha256": sha256(Path(__file__).resolve()),
        "developmentOpened": False,
        "testOpened": False,
    }


def prepare(output: Path = OUTPUT) -> dict[str, object]:
    verify_pins()
    output.mkdir(parents=True, exist_ok=True)
    frozen = protocol()
    path = output / "protocol.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != frozen:
            original_runner = existing.get("runnerSha256")
            amended_runner = frozen.get("runnerSha256")
            left, right = dict(existing), dict(frozen)
            left.pop("runnerSha256", None)
            right.pop("runnerSha256", None)
            if left != right:
                raise ValueError("frozen scientific continuation protocol changed")
            amendment = {
                "schema": "slp.k562-count-latent-mean-aux-operational-amendment/v1",
                "originalProtocolSha256": sha256(path),
                "originalRunnerSha256": original_runner,
                "amendedRunnerSha256": amended_runner,
                "changes": [
                    "bound isolated CPU replay to two intra-op threads and one inter-op thread",
                    "require an in-memory model versus reloaded CUDA artifact forecast check before development",
                ],
                "scientificProtocolChanged": False,
                "developmentOpened": False,
                "testOpened": False,
            }
            amendment_path = output / "execution-amendment.json"
            if amendment_path.exists():
                if json.loads(amendment_path.read_text(encoding="utf-8")) != amendment:
                    raise ValueError("continuation operational amendment changed")
            else:
                write_json(amendment_path, amendment)
    else:
        write_json(path, frozen)
    return frozen


def load_mean_resources(resources: dict[str, object]) -> dict[str, object]:
    moments = load_npz(MOMENTS)
    baseline = load_npz(BASELINE / "model.npz")
    static = resources["static"]
    genes = moments["action_ids"].astype(str)
    if not np.array_equal(moments["query_ids"].astype(str), resources["registered"]["query_ids"]):
        raise ValueError("fitting moments query roster mismatch")
    lookup = {value: row for row, value in enumerate(static["entity_id"].astype(str))}
    entity = np.asarray([lookup[gene] for gene in genes], np.int64)
    normalizer = anchored_mean_mse(
        moments["cp10k_sum"], moments["cell_count"], moments["gem_cell_count"],
        baseline["basal_rate"], baseline["target_mean"],
    )
    if normalizer != TRAINING_ANCHORED_MEAN_MSE:
        raise ValueError("training anchored-mean MSE drift")
    weights = moments["gem_cell_count"].astype(np.float32)
    weights /= weights.sum(1, keepdims=True)
    return {
        "gene_ids": genes,
        "target": np.log1p(
            moments["cp10k_sum"] / moments["cell_count"][:, None]
        ).astype(np.float32),
        "gem_weights": weights,
        "normalized_action_features": np.asarray(
            static["normalized_feature_values"][entity], np.float32
        ),
        "raw_action_features": np.asarray(static["feature_values"][entity], np.float32),
        "baseline": baseline,
    }


def _parameter_norms(model) -> dict[str, float]:
    groups: dict[str, float] = {}
    for name, value in model.named_parameters():
        group = name.split(".", 1)[0]
        groups[group] = groups.get(group, 0.0) + float(value.detach().square().sum())
    return {name: value**0.5 for name, value in groups.items()}


def train_arm(base, core, objective, resources, mean, arm: str):
    if arm not in ARMS:
        raise ValueError(arm)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    model = core.CountLatentState(core.Config(**MODEL_CONFIG)).to(device)
    model.load_state_dict(load_file(str(ORIGINAL / "model.safetensors")))
    model.train()
    initial_norms = _parameter_norms(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    registered = resources["registered"]
    query = torch.as_tensor(registered["query_features"], device=device)
    basal = torch.as_tensor(registered["basal_rate"], device=device)
    basal_mask = torch.as_tensor(registered["basal_observed"], device=device)
    normalized = np.asarray(resources["static"]["normalized_feature_values"], np.float32)
    observed = torch.ones((BATCH, 8563), dtype=torch.bool, device=device)
    mean_target = torch.as_tensor(mean["target"], device=device)
    mean_weights = torch.as_tensor(mean["gem_weights"], device=device)
    mean_actions = torch.as_tensor(mean["normalized_action_features"], device=device)
    count_rng = np.random.default_rng(SEED)
    mean_rng = np.random.default_rng(MEAN_SEED)
    count_trace = hashlib.sha256()
    mean_trace = hashlib.sha256()
    history, window = [], []
    started = time.perf_counter()
    for update in range(1, UPDATES + 1):
        if time.perf_counter() - started > MAX_SECONDS_PER_ARM:
            raise TimeoutError(f"{arm} exceeded frozen 900-second cap")
        rows = base.draw_balanced_rows(resources["sampling"], count_rng)
        count_trace.update(np.asarray(rows, dtype="<i8").tobytes())
        count_array = np.asarray(resources["counts"][rows], dtype=np.float32)
        libraries = np.asarray(resources["rows"]["library_size"][rows], dtype=np.float32)
        if not np.array_equal(count_array.astype(np.int64).sum(1), libraries.astype(np.int64)):
            raise ValueError("random-access count/library mismatch")
        actions, action_mask = base.action_batch(normalized, resources["actionEntityIndex"], rows)
        gem_index = resources["gemIndex"][rows]
        unique_gem, inverse = np.unique(gem_index, return_inverse=True)
        optimizer.zero_grad(set_to_none=True)
        contexts = model.encode_context(
            query,
            basal[torch.as_tensor(unique_gem, device=device)],
            basal_mask[torch.as_tensor(unique_gem, device=device)],
        )
        prior = model.prior_from_context(
            torch.as_tensor(actions, device=device),
            torch.as_tensor(action_mask, device=device),
            contexts[torch.as_tensor(inverse, device=device)],
        )
        result = model.elbo(
            torch.as_tensor(count_array, device=device), observed,
            torch.as_tensor(libraries, device=device), query,
            basal[torch.as_tensor(gem_index, device=device)], prior,
        )
        count_loss = result["loss_per_cell"].mean()
        raw_mean_loss = torch.zeros((), device=device)
        normalized_mean_loss = torch.zeros((), device=device)
        if arm == "mean-aux":
            selected = draw_mean_rows(mean_rng, len(mean["gene_ids"]))
            mean_trace.update(np.asarray(selected, dtype="<i8").tobytes())
            index = torch.as_tensor(selected, device=device)
            model.eval()
            all_contexts = model.encode_context(query, basal, basal_mask)
            action = mean_actions[index, None, :]
            expanded_action = action[:, None].expand(-1, len(basal), -1, -1)
            expanded_action = expanded_action.reshape(MEAN_GENES * len(basal), 1, -1)
            expanded_mask = torch.ones(
                (MEAN_GENES * len(basal), 1), dtype=torch.bool, device=device
            )
            mean_prior = model.prior_from_context(
                expanded_action,
                expanded_mask,
                all_contexts.repeat(MEAN_GENES, 1),
            )
            expanded_basal = basal[None].expand(MEAN_GENES, -1, -1)
            rates = model.population_mean(
                mean_prior, query, expanded_basal.reshape(MEAN_GENES * len(basal), -1)
            ).reshape(MEAN_GENES, len(basal), -1)
            prediction = objective.population_log1p_mean(rates, mean_weights[index])
            raw_mean_loss = torch.mean(torch.square(prediction - mean_target[index]))
            normalized_mean_loss = objective.normalized_profile_mse(
                prediction, mean_target[index], TRAINING_ANCHORED_MEAN_MSE
            )
            model.train()
        loss = count_loss + MEAN_WEIGHT * normalized_mean_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite {arm} loss at update {update}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"nonfinite {arm} gradient at update {update}")
        optimizer.step()
        diagnostic = base.latent_diagnostics(core, result["posterior"], prior)
        window.append({
            "totalLoss": float(loss.detach()),
            "countElbo": float(count_loss.detach()),
            "reconstructionPerQuery": float(result["reconstruction_per_query"].mean().detach()),
            "aggregateRawMse": float(raw_mean_loss.detach()),
            "aggregateNormalizedMse": float(normalized_mean_loss.detach()),
            "gradientNormBeforeClip": float(gradient_norm.detach()),
            **diagnostic,
        })
        if update % 100 == 0:
            item = {"update": update, "elapsedSeconds": time.perf_counter() - started}
            for name in window[0]:
                item[name] = float(np.mean([entry[name] for entry in window]))
            history.append(item)
            print(json.dumps({"event": "training", "arm": arm, **item}), flush=True)
            window.clear()
    model.eval()
    return model, {
        "arm": arm,
        "updates": UPDATES,
        "seconds": time.perf_counter() - started,
        "history": history,
        "countRowTraceSha256": count_trace.hexdigest(),
        "meanGeneTraceSha256": mean_trace.hexdigest() if arm == "mean-aux" else None,
        "initialParameterNorms": initial_norms,
        "finalParameterNorms": _parameter_norms(model),
    }


def fitting_prior_diagnostic(base, models, resources, mean) -> dict[str, object]:
    genes = mean["gene_ids"]
    selected = np.asarray(
        sorted(
            range(len(genes)),
            key=lambda row: hashlib.sha256(
                ("slp11-count-prior-fit-audit-v1|" + genes[row]).encode()
            ).digest(),
        )[:128],
        np.int64,
    )
    truth = mean["target"][selected].astype(np.float64)
    baseline_core = load_module(BASELINE / "source/count_static_ridge.py", "continuation_ridge")
    baseline = mean["baseline"]
    counts = load_npz(MOMENTS)["gem_cell_count"][selected]
    anchor = baseline_core.control_anchor(baseline["basal_rate"], counts)
    ridge = baseline_core.absolute_prediction(
        anchor,
        baseline_core.predict_residual(
            baseline, mean["raw_action_features"][selected], str(baseline["selected_alpha"])
        ),
    )
    anchored_mean = baseline_core.absolute_prediction(
        anchor, np.broadcast_to(baseline["target_mean"], anchor.shape)
    )
    values = {"ridge": ridge, "anchoredMean": anchored_mean, "control": anchor}
    for arm, model in models.items():
        _, prediction = base.direct_population_prediction(
            model,
            mean["normalized_action_features"][selected, None, :],
            np.ones((len(selected), 1), np.bool_),
            mean["gem_weights"][selected],
            resources["registered"]["query_features"],
            resources["registered"]["basal_rate"],
            next(model.parameters()).device,
        )
        values[arm] = prediction
    return {
        "schema": "slp.k562-count-latent-continuation-fitting-diagnostic/v1",
        "selection": "first 128 by SHA256(slp11-count-prior-fit-audit-v1|ENSG)",
        "geneIds": genes[selected].tolist(),
        "metrics": {name: base.profile_metrics(value, truth, anchor) for name, value in values.items()},
        "developmentOpened": False,
        "testOpened": False,
    }


def save_artifact(output: Path, models, training, resources) -> dict[str, object]:
    (output / "source").mkdir(exist_ok=True)
    for arm in ARMS:
        (output / "arms" / arm).mkdir(parents=True, exist_ok=True)
        save_file(models[arm].state_dict(), str(output / "arms" / arm / "model.safetensors"))
        write_json(output / "arms" / arm / "loss-history.json", training[arm])
    shutil.copyfile(ORIGINAL / "reference.npz", output / "reference.npz")
    copies = {
        ORIGINAL / "source/count_latent_state.py": output / "source/count_latent_state.py",
        OBJECTIVE: output / "source/molecular_mean_objective.py",
        INFERENCE: output / "source/inference.py",
        VERIFY: output / "source/verify.py",
        Path(__file__).resolve(): output / "source/runner.py",
    }
    for source, destination in copies.items():
        shutil.copyfile(source, destination)
    hashes = {
        str(path.relative_to(output)).replace("\\", "/"): sha256(path)
        for path in output.rglob("*")
        if path.is_file() and path.name not in {"artifact-manifest.json", "protocol.json"}
    }
    manifest = {
        "schema": "slp.k562-count-latent-mean-aux-continuation-artifact/v1",
        "protocolSha256": sha256(output / "protocol.json"),
        "parentModelSha256": PINS[ORIGINAL / "model.safetensors"],
        "referenceInheritedBitExact": hashes["reference.npz"] == PINS[ORIGINAL / "reference.npz"],
        "arms": {arm: {"modelPath": f"arms/{arm}/model.safetensors"} for arm in ARMS},
        "sha256": hashes,
        "developmentOpened": False,
        "testOpened": False,
    }
    write_json(output / "artifact-manifest.json", manifest)
    return manifest


def target_free_probe(output: Path, mean, base, models, resources) -> dict[str, object]:
    inference = load_module(output / "source/inference.py", "continuation_inference_probe")
    raw = mean["raw_action_features"][:4]
    weights = mean["gem_weights"][:4]
    arrays: dict[str, np.ndarray] = {
        "raw_action_features": raw,
        "gem_weights": weights,
    }
    metrics = {}
    for arm in ARMS:
        predictor = inference.Predictor(output, arm, device="cuda")
        first = predictor.predict(raw, weights)
        direct_cp10k, direct_log1p = base.direct_population_prediction(
            models[arm],
            mean["normalized_action_features"][:4, None, :],
            np.ones((len(raw), 1), np.bool_),
            weights,
            resources["registered"]["query_features"],
            resources["registered"]["basal_rate"],
            torch.device("cuda"),
        )
        cp_difference = np.abs(
            first["mean_cp10k"].astype(np.float64) - direct_cp10k.astype(np.float64)
        )
        relative = cp_difference / np.maximum(np.abs(direct_cp10k), 1.0)
        log_difference = np.abs(
            first["mean_log1p_cp10k"].astype(np.float64)
            - direct_log1p.astype(np.float64)
        )
        if np.max(relative) > 1e-6 or np.max(log_difference) > 1e-6:
            raise RuntimeError(f"in-memory/reloaded CUDA forecast mismatch: {arm}")
        empty = predictor.predict(
            raw, weights, action_mask=np.zeros((len(raw), 1), np.bool_)
        )
        arrays[f"{arm}_mean_cp10k"] = first["mean_cp10k"]
        arrays[f"{arm}_empty_cp10k"] = empty["mean_cp10k"]
        expected_empty = weights @ predictor.reference["basal_rate"]
        metrics[arm] = {
            "emptyMaximumAbsoluteDifference": float(np.max(np.abs(empty["mean_cp10k"] - expected_empty))),
            "inMemoryReloadMaximumRelativeCp10kDifference": float(np.max(relative)),
            "inMemoryReloadMaximumAbsoluteLog1pDifference": float(np.max(log_difference)),
            "finite": bool(np.isfinite(first["mean_cp10k"]).all()),
        }
    path = output / "target-free-probe.npz"
    np.savez_compressed(path, **arrays)
    return {"path": path.name, "sha256": sha256(path), "gpu": metrics}


def freeze_development_forecasts(base, output, models, resources, metadata, mean):
    baseline_core = load_module(BASELINE / "source/count_static_ridge.py", "continuation_forecast_ridge")
    baseline = mean["baseline"]
    anchor = baseline_core.control_anchor(baseline["basal_rate"], metadata["gem_cell_count"])
    ridge = baseline_core.absolute_prediction(
        anchor,
        baseline_core.predict_residual(
            baseline, metadata["raw_action_features"], str(baseline["selected_alpha"])
        ),
    )
    anchored_mean = baseline_core.absolute_prediction(
        anchor, np.broadcast_to(baseline["target_mean"], anchor.shape)
    )
    arrays = {
        "schema": np.asarray("slp.k562-count-latent-continuation-frozen-development-forecasts/v1"),
        "gene_ids": metadata["gene_ids"],
        "query_ids": resources["registered"]["query_ids"],
        "gem_group_ids": resources["registered"]["gem_group_ids"],
        "gem_cell_count": metadata["gem_cell_count"],
        "cell_count": metadata["cell_count"],
        "anchor": anchor,
        "control_prediction": anchor,
        "anchored_mean_prediction": anchored_mean,
        "static_ridge_prediction": ridge,
    }
    for arm, model in models.items():
        _, prediction = base.direct_population_prediction(
            model,
            metadata["normalized_action_features"][:, None, :],
            np.ones((len(metadata["gene_ids"]), 1), np.bool_),
            metadata["gem_weights"],
            resources["registered"]["query_features"],
            resources["registered"]["basal_rate"],
            next(model.parameters()).device,
        )
        arrays[f"{arm}_prediction"] = prediction
    path = output / "development-forecasts-before-outcomes.npz"
    np.savez_compressed(path, **arrays)
    receipt = {
        "schema": "slp.k562-count-latent-continuation-development-forecast-freeze/v1",
        "forecastSha256": sha256(path),
        "modelSha256": {
            arm: sha256(output / "arms" / arm / "model.safetensors") for arm in ARMS
        },
        "genes": 305,
        "queries": 8563,
        "developmentCountMembersOpened": False,
        "testOpened": False,
    }
    write_json(output / "FORECASTS-FROZEN-BEFORE-DEVELOPMENT.json", receipt)
    return receipt


def score_development(base, output: Path, truth: np.ndarray, count_diagnostic):
    forecasts = load_npz(output / "development-forecasts-before-outcomes.npz")
    mappings = {
        "countOnly": "count-only_prediction",
        "meanAux": "mean-aux_prediction",
        "staticRidge": "static_ridge_prediction",
        "anchoredMean": "anchored_mean_prediction",
        "pureControl": "control_prediction",
    }
    aggregate = {
        name: base.profile_metrics(forecasts[key], truth, forecasts["anchor"])
        for name, key in mappings.items()
    }
    arrays: dict[str, np.ndarray] = {"gene_ids": forecasts["gene_ids"]}
    for name, key in mappings.items():
        mse, correlation = per_gene_metrics(forecasts[key], truth, forecasts["anchor"])
        arrays[f"{name}_mse"] = mse
        arrays[f"{name}_centered_pearson"] = correlation
    path = output / "development-per-gene-metrics.npz"
    np.savez_compressed(path, **arrays)
    return aggregate, {
        "sha256": sha256(path),
        "genes": len(forecasts["gene_ids"]),
        "definition": "per-gene raw profile MSE and row Pearson after anchor subtraction, anchored-first-gene translation removal, independent query centering, and row centering",
    }, count_diagnostic


def profile(output: Path = OUTPUT) -> dict[str, object]:
    prepare(output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    core = load_module(ORIGINAL / "source/count_latent_state.py", "continuation_profile_core")
    objective = load_module(OBJECTIVE, "continuation_profile_objective")
    with np.load(ORIGINAL / "reference.npz", allow_pickle=False) as values:
        query_values = np.asarray(values["query_features"], np.float32)
        basal_values = np.asarray(values["basal_rate"], np.float32)
    device = torch.device("cuda")
    model = core.CountLatentState(core.Config(**MODEL_CONFIG)).to(device)
    model.load_state_dict(load_file(str(ORIGINAL / "model.safetensors")))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    query = torch.as_tensor(query_values, device=device)
    basal = torch.as_tensor(basal_values, device=device)
    basal_mask = torch.ones_like(basal, dtype=torch.bool)
    rng = np.random.default_rng(731)
    counts = torch.as_tensor(rng.poisson(1.0, (BATCH, 8563)).astype(np.float32), device=device)
    library = counts.sum(1).clamp_min(1)
    actions = torch.as_tensor(rng.normal(size=(BATCH, 1, 577)).astype(np.float32), device=device)
    action_mask = torch.ones((BATCH, 1), dtype=torch.bool, device=device)
    gem = torch.as_tensor(rng.integers(0, 48, BATCH), device=device)
    mean_actions = torch.as_tensor(rng.normal(size=(MEAN_GENES, 1, 577)).astype(np.float32), device=device)
    mean_weights = torch.as_tensor(rng.dirichlet(np.ones(48), MEAN_GENES).astype(np.float32), device=device)
    mean_target = torch.as_tensor(rng.random((MEAN_GENES, 8563)).astype(np.float32), device=device)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    repeats = 10
    for _ in range(repeats):
        optimizer.zero_grad(set_to_none=True)
        contexts = model.encode_context(query, basal, basal_mask)
        prior = model.prior_from_context(actions, action_mask, contexts[gem])
        result = model.elbo(counts, torch.ones_like(counts, dtype=torch.bool), library, query, basal[gem], prior)
        model.eval()
        expanded = mean_actions[:, None].expand(-1, 48, -1, -1).reshape(MEAN_GENES * 48, 1, 577)
        mean_prior = model.prior_from_context(
            expanded, torch.ones((MEAN_GENES * 48, 1), dtype=torch.bool, device=device),
            contexts.repeat(MEAN_GENES, 1),
        )
        rates = model.population_mean(
            mean_prior, query, basal[None].expand(MEAN_GENES, -1, -1).reshape(MEAN_GENES * 48, -1)
        ).reshape(MEAN_GENES, 48, -1)
        mean_loss = objective.normalized_profile_mse(
            objective.population_log1p_mean(rates, mean_weights),
            mean_target,
            TRAINING_ANCHORED_MEAN_MSE,
        )
        model.train()
        (result["loss_per_cell"].mean() + MEAN_WEIGHT * mean_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - started) / repeats
    report = {
        "secondsPerMeanAuxUpdate": seconds,
        "projectedSecondsPerArm": seconds * UPDATES,
        "peakAllocatedBytes": torch.cuda.max_memory_allocated(),
        "peakReservedBytes": torch.cuda.max_memory_reserved(),
        "fitsCaps": seconds * UPDATES < MAX_SECONDS_PER_ARM and torch.cuda.max_memory_reserved() < 10 * 2**30,
        "biologicalOutcomesRead": False,
    }
    write_json(output / "cuda-profile.json", report)
    print(json.dumps(report))
    return report


def run(output: Path = OUTPUT) -> dict[str, object]:
    prepare(output)
    if (output / "report.json").exists():
        raise FileExistsError("immutable continuation already complete")
    profile_report = json.loads((output / "cuda-profile.json").read_text(encoding="utf-8"))
    if not profile_report["fitsCaps"]:
        raise RuntimeError("target-free CUDA profile does not fit frozen caps")
    base = load_module(BASE_RUNNER, "count_latent_continuation_base")
    core = load_module(ORIGINAL / "source/count_latent_state.py", "count_latent_continuation_core")
    objective = load_module(OBJECTIVE, "count_latent_continuation_objective")
    resources = base.load_training_resources()
    mean = load_mean_resources(resources)
    models, training = {}, {}
    for arm in ARMS:
        models[arm], training[arm] = train_arm(base, core, objective, resources, mean, arm)
    if training["count-only"]["countRowTraceSha256"] != training["mean-aux"]["countRowTraceSha256"]:
        raise AssertionError("paired count-row streams differ")
    save_artifact(output, models, training, resources)
    target_free_probe(output, mean, base, models, resources)
    process = subprocess.run(
        [sys.executable, str(output / "source/verify.py"), str(output)],
        check=False, capture_output=True, text=True, timeout=300,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    if process.returncode != 0:
        raise RuntimeError(f"isolated CPU replay failed: {process.stderr[-2000:]}")
    replay = json.loads(process.stdout.strip().splitlines()[-1])
    write_json(output / "isolated-cpu-verification.json", replay)
    if not replay["passes"]:
        raise RuntimeError("isolated CPU replay exceeds frozen numerical tolerance")
    fitting_reconstruction = {
        arm: base.evaluate_fitting_reconstruction(core, model, resources)
        for arm, model in models.items()
    }
    write_json(output / "fitting-reconstruction-diagnostic.json", fitting_reconstruction)
    fitting_prior = fitting_prior_diagnostic(base, models, resources, mean)
    write_json(output / "fitting-prior-diagnostic.json", fitting_prior)
    metadata = base.validation_metadata(resources)
    forecast_freeze = freeze_development_forecasts(base, output, models, resources, metadata, mean)
    write_json(output / "FROZEN-BEFORE-DEVELOPMENT.json", {
        "schema": "slp.k562-count-latent-mean-aux-continuation-freeze/v1",
        "protocolSha256": sha256(output / "protocol.json"),
        "artifactManifestSha256": sha256(output / "artifact-manifest.json"),
        "executionAmendmentSha256": sha256(output / "execution-amendment.json"),
        "modelSha256": forecast_freeze["modelSha256"],
        "forecastSha256": forecast_freeze["forecastSha256"],
        "isolatedCpuVerificationSha256": sha256(output / "isolated-cpu-verification.json"),
        "targetFreeProbeSha256": sha256(output / "target-free-probe.npz"),
        "fittingReconstructionSha256": sha256(output / "fitting-reconstruction-diagnostic.json"),
        "fittingPriorSha256": sha256(output / "fitting-prior-diagnostic.json"),
        "developmentCountMembersOpened": False,
        "testOpened": False,
    })
    truth, count_diagnostic = base.aggregate_validation_truth(metadata["gene_ids"])
    metrics, per_gene, count_diagnostic = score_development(base, output, truth, count_diagnostic)
    original_reconstruction = reconstruction_loss(
        json.loads((ORIGINAL_FINAL / "report.json").read_text())["fittingReconstruction"]
    )
    mean_aux = metrics["meanAux"]
    ridge = metrics["staticRidge"]
    anchored_mean = metrics["anchoredMean"]
    recon_aux = reconstruction_loss(fitting_reconstruction["mean-aux"])
    recon_control = reconstruction_loss(fitting_reconstruction["count-only"])
    gate = {
        "mseOnePercentBelowRidge": bool(mean_aux["geneProfileMse"] <= 0.99 * ridge["geneProfileMse"]),
        "mseOnePercentBelowAnchoredMean": bool(mean_aux["geneProfileMse"] <= 0.99 * anchored_mean["geneProfileMse"]),
        "centeredPearsonAtLeastPoint10": bool(
            mean_aux["independentlyQueryCenteredPearson"] is not None
            and mean_aux["independentlyQueryCenteredPearson"] >= 0.1
        ),
        "centeredPearsonNoLowerThanRidge": bool(
            mean_aux["independentlyQueryCenteredPearson"] is not None
            and ridge["independentlyQueryCenteredPearson"] is not None
            and mean_aux["independentlyQueryCenteredPearson"] >= ridge["independentlyQueryCenteredPearson"]
        ),
        "reconstructionWithinOnePercentOriginal": bool(recon_aux <= 1.01 * original_reconstruction),
        "reconstructionWithinOnePercentCountContinuation": bool(recon_aux <= 1.01 * recon_control),
    }
    gate["passes"] = bool(all(gate.values()))
    report = {
        "schema": "slp.k562-count-latent-mean-aux-continuation-report/v1",
        "protocolSha256": sha256(output / "protocol.json"),
        "artifactManifestSha256": sha256(output / "artifact-manifest.json"),
        "freezeSha256": sha256(output / "FROZEN-BEFORE-DEVELOPMENT.json"),
        "training": training,
        "portableVerification": replay,
        "fittingReconstruction": fitting_reconstruction,
        "fittingPrior": fitting_prior,
        "forecastFreeze": forecast_freeze,
        "development": {
            "metrics": metrics,
            "perGeneMetrics": per_gene,
            "countDiagnostic": count_diagnostic,
            "gate": gate,
            "negativePredictionFraction": {
                name: float(np.mean(load_npz(output / "development-forecasts-before-outcomes.npz")[f"{arm}_prediction"] < 0))
                for name, arm in (("countOnly", "count-only"), ("meanAux", "mean-aux"))
            },
        },
        "interpretation": "Matched adaptive K562 development continuations testing an added fitting aggregate-mean objective. This is not a single-cell generator validation, identified biological state, independent confirmation, test result, or benchmark claim.",
        "developmentEvaluations": 1,
        "testOpened": False,
        "benchmarkAccessed": False,
    }
    write_json(output / "report.json", report)
    write_json(output / "execution-receipt.json", {
        "reportSha256": sha256(output / "report.json"),
        "perGeneMetricsSha256": per_gene["sha256"],
        "decision": "advance" if gate["passes"] else "reject",
    })
    print(json.dumps({"report": str(output / "report.json"), "gate": gate, "metrics": metrics}))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "profile", "run"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with threadpool_limits(2):
        {"prepare": prepare, "profile": profile, "run": run}[args.mode](args.output)
