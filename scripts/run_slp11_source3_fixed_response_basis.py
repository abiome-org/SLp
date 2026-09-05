#!/usr/bin/env python3
"""Prepare the fixed shared-response-basis source3 neural diagnostic."""

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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
PAIR = ROOT / "results/slp11-transition/human-source3-bp-neural-mean-pair-seed731-v2-finalization-v1"
BASIS_ROOT = ROOT / "data/derived/slp11-human-response-basis/source3-shared-rank128-fitting-v1"
CORE = ROOT / "modules/slp-1-1-fixed-query-transition-v1/transition_model.py"
OLD_CORE = PAIR / "source/control_transition_model.py"
BP_HELPER = ROOT / "scripts/run_slp11_source3_bp_neural_mean_pair.py"
OBJECTIVE = PAIR / "source/mean_objective.py"
WEIGHTING = PAIR / "source/objective_weighting.py"
METRICS = PAIR / "source/four_context_baselines.py"
BP_PAIR_REPORT = PAIR / "report.json"
BP_RIDGE_REPORT = ROOT / "results/slp11-transition/human-gwps-bp-ridge-source3-seed731-v2/report.json"
BP_KERNEL_REPORT = ROOT / "results/slp11-transition/human-gwps-bp-nystrom-rbf512-seed731-v1/report.json"
VERIFIER = ROOT / "scripts/verify_slp11_source3_fixed_response_basis.py"
OUTPUT = ROOT / "results/slp11-transition/human-source3-bp-fixed-response-basis-seed731-v2"
PINS = {
    "basis": "a31bb27db40d542dfb541daccebe056415080b457803ec7eb222c303add0b1ee",
    "basisReport": "817c74653a248cd350fc659ab495072ada1b3efe43db5f3c2ad2879d0c532d25",
    "bpReference": "aea0c407fcc0e3199b2926eb9f639cc8fe4d8f3abd5eb3dd1055abc065c0dfef",
    "bpModel": "f1e0acf79c5326d4553ee77f45ccaa0d02628042672413d7089a17991e5d99fc",
    "oldCore": "fdb4555bd0f7c0a0786539da67048f6985f4ec2f36ef7aa45bd22c7c6bfbb2ef",
    "bpHelper": "fe007480adc485f71e6e8dfe0af8787bf68d5706b8ed02f2815912da9c0a6de8",
    "bpPairReport": "5d7c14fff561f02e1a46c353c54fefda5aec8c778edaacea4c5bca5849441060",
    "bpRidgeReport": "8a3d1ba2265dc09bf6856c97c7a791775ef3282594beed269f708f353d895a0a",
    "bpKernelReport": "d8259c864460a21f9a13718b2190aad926ca58dc01409c0fab1220a6fbbd276c",
}
SEED = 731
BATCH = 64
STEPS = 12_000


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: object) -> None:
    def clean(item: object) -> object:
        if isinstance(item, dict):
            return {str(key): clean(entry) for key, entry in item.items()}
        if isinstance(item, (tuple, list)):
            return [clean(entry) for entry in item]
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def initialize_fixed(fixed_module, old_module, helper):
    """Copy every shared seed-731 tensor from the matched BP neural initializer."""
    old, physical = helper.initialize_extended(old_module, seed=SEED)
    torch.manual_seed(SEED)
    fixed = fixed_module.FixedQueryTransition(
        fixed_module.Config(1285, 1188, state_dim=128, hidden_dim=128, dropout=0.2)
    )
    with torch.no_grad():
        for name, target in fixed.state_dict().items():
            source = old.state_dict()[name]
            if target.shape != source.shape:
                raise ValueError(f"shared tensor shape drift: {name}")
            target.copy_(source)
    old_weight = old.state_dict()["action_encoder.0.weight"]
    physical_weight = physical.state_dict()["action_encoder.0.weight"]
    if not torch.equal(old_weight[:, :1156], physical_weight) or torch.count_nonzero(
        old_weight[:, 1156:]
    ):
        raise ValueError("matched extended initializer does not preserve a zero BP tail")
    return fixed, old


def cpu_shape_profile(fixed_module, old_module, helper, query_coordinates: np.ndarray) -> dict[str, object]:
    model, old = initialize_fixed(fixed_module, old_module, helper)
    model.train()
    actions = torch.zeros((2, 1285))
    control = torch.zeros((2, 7036))
    amplitude = torch.ones(7036)
    scale = torch.ones((2, 7036))
    basal_features = torch.zeros((2, 64, 1188))
    basal_values = torch.zeros((2, 64))
    basal_mask = torch.ones((2, 64), dtype=torch.bool)
    started = time.monotonic()
    result = model(
        actions,
        torch.as_tensor(query_coordinates),
        control,
        amplitude,
        scale,
        basal_features,
        basal_values,
        basal_mask,
    )
    result["mean"].square().mean().backward()
    seconds = time.monotonic() - started
    return {
        "device": "cpu",
        "targetFreeSynthetic": True,
        "batch": 2,
        "queries": 7036,
        "actionFeatures": 1285,
        "basalFeatures": 1188,
        "state": 128,
        "fixedParameters": sum(value.numel() for value in model.parameters()),
        "matchedLearnedQueryParameters": sum(value.numel() for value in old.parameters()),
        "removedQueryEncoderParameters": sum(value.numel() for value in old.query_encoder.parameters()),
        "forwardBackwardSeconds": seconds,
        "gpuRuntimeUpperReferenceSeconds": 126.969,
        "gpuRuntimeUpperReferenceMeaning": "completed matched BP learned-query arm; fixed-basis arm removes query-encoder compute but has not been GPU-profiled",
    }


def prepare(output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    paths = {
        "basis": BASIS_ROOT / "basis.npz",
        "basisReport": BASIS_ROOT / "report.json",
        "basisProtocol": BASIS_ROOT / "protocol.json",
        "bpReference": PAIR / "bp128-present/reference.npz",
        "bpModel": PAIR / "bp128-present/model.safetensors",
        "oldCore": OLD_CORE,
        "fixedCore": CORE,
        "bpHelper": BP_HELPER,
        "objective": OBJECTIVE,
        "weighting": WEIGHTING,
        "metrics": METRICS,
        "bpPairReport": BP_PAIR_REPORT,
        "bpRidgeReport": BP_RIDGE_REPORT,
        "bpKernelReport": BP_KERNEL_REPORT,
    }
    for name in (
        "basis",
        "basisReport",
        "bpReference",
        "bpModel",
        "oldCore",
        "bpHelper",
        "bpPairReport",
        "bpRidgeReport",
        "bpKernelReport",
    ):
        if sha256(paths[name]) != PINS[name]:
            raise ValueError(f"input drift: {name}")
    with np.load(paths["basis"], allow_pickle=False) as archive:
        coordinates = archive["query_coordinates"].astype(np.float32)
        if coordinates.shape != (7036, 128):
            raise ValueError("fixed query coordinate shape drift")
    fixed_module = load(CORE, "fixed_basis_prepare_core")
    old_module = load(OLD_CORE, "fixed_basis_prepare_old_core")
    helper = load(BP_HELPER, "fixed_basis_prepare_helper")
    fixed, old = initialize_fixed(fixed_module, old_module, helper)
    shared_exact = all(
        torch.equal(value, old.state_dict()[name]) for name, value in fixed.state_dict().items()
    )
    if not shared_exact:
        raise ValueError("shared initialization is not bit exact")
    profile = cpu_shape_profile(fixed_module, old_module, helper, coordinates)
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    for name, path in {
        "transition_model.py": CORE,
        "old_transition_model.py": OLD_CORE,
        "mean_objective.py": OBJECTIVE,
        "objective_weighting.py": WEIGHTING,
        "four_context_baselines.py": METRICS,
        "verify_artifact.py": VERIFIER,
        "trainer.py": Path(__file__),
    }.items():
        shutil.copy2(path, source / name)
    protocol = {
        "schema": "slp.source3-fixed-response-basis-neural-protocol/v2",
        "status": "prepared-before-biological-fitting",
        "hypothesis": "The feasible shared fitting-only response basis improves unseen-gene molecular landscapes because the learned response-query decoder is misaligned with source3 variation.",
        "modelChange": "Replace only query_encoder output with fixed shared rank128 query coordinates=components.T*sqrt(128). Keep action/context encoders, transition, trainable mean_state, state128, hidden128, dropout.2 and frozen delta_amplitude.",
        "initialization": "Pinned helper.initialize_extended creates the matched seed731 network from physical1156 then appends129 exact-zero action columns. Every fixed-model shared tensor is copied bit-exact from that initializer; no query encoder parameters remain.",
        "training": "Source3 only; real BP128+presence action tail; 12000 deterministic updates,B64,seed731,global equal-context/equal-gene weights,all7036queries,masked standardized mean MSE,AdamW lr.0005 decay.1,clip1,final checkpoint only,no validation before final.",
        "decisionRule": "Diagnostic: each context requires >=1% raw gene-profile MSE improvement and no independently query-centered r regression versus the matched BP learned-query neural arm. World advancement remains distinct: >=2% versus BP ridge, finite r>=.10 and no r regression. BP Nyström kernel is reported as an additional frontier comparator.",
        "portableVerification": "After fitting and before validation, save raw-action/context probes and in-memory CUDA means; a fresh CPU process must load only packaged source/model/reference, verify hashes, replay <=1e-5, and prove exact empty identity across3contexts.",
        "supersedesPreparedOnly": {"path": "results/slp11-transition/human-source3-bp-fixed-response-basis-seed731-v1", "reason": "v1 compared against a directly initialized1285-input model instead of the pinned physical1156-to1285 zero-tail initializer; it was never fitted"},
        "basisProvenance": "Quantitative fitting-derived shared source3 response basis; not a static query prior and not available for an unmeasured assay panel.",
        "inputs": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in paths.items()},
        "sharedInitializationBitExact": shared_exact,
        "profile": profile,
        "validationEvaluations": 0,
        "testAccessed": False,
        "benchmarkAccessed": False,
    }
    write_json(output / "protocol.json", protocol)
    prepared = {
        "protocolSha256": sha256(output / "protocol.json"),
        "sourceHashes": {path.name: sha256(path) for path in sorted(source.glob("*.py"))},
        "profile": profile,
        "validationEvaluations": 0,
    }
    write_json(output / "PREPARED.json", prepared)
    return prepared


def make_runtime(
    raw_actions: np.ndarray,
    reference: dict[str, np.ndarray],
    coordinates: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    normalized = ((raw_actions - reference["feature_mean"]) / reference["feature_std"]).astype(
        np.float32
    )
    query = ((reference["query_features"] - reference["query_feature_mean"]) / reference[
        "query_feature_std"
    ]).astype(np.float32)
    selected = reference["context_query_indices"]
    return {
        "actions": torch.as_tensor(normalized, device=device),
        "coordinates": torch.as_tensor(coordinates, device=device),
        "control": torch.as_tensor(reference["control_mean"], device=device),
        "amplitude": torch.as_tensor(reference["delta_amplitude"], device=device),
        "scale": torch.as_tensor(reference["objective_query_scale"], device=device),
        "basal_features": torch.as_tensor(query[selected], device=device),
        "basal_values": torch.as_tensor(reference["context_values"], device=device),
        "basal_mask": torch.as_tensor(reference["context_mask"], dtype=torch.bool, device=device),
    }


def forward_rows(
    model,
    rows: np.ndarray,
    contexts: np.ndarray,
    runtime: dict[str, torch.Tensor],
    device: torch.device,
):
    row = torch.as_tensor(rows, dtype=torch.int64, device=device)
    context = torch.as_tensor(contexts, dtype=torch.int64, device=device)
    return model(
        runtime["actions"][row],
        runtime["coordinates"],
        runtime["control"][context],
        runtime["amplitude"],
        runtime["scale"][context],
        runtime["basal_features"],
        runtime["basal_values"][context],
        runtime["basal_mask"][context],
    )


def predict(model, raw_actions, contexts, reference, coordinates, device):
    model.eval()
    runtime = make_runtime(raw_actions, reference, coordinates, device)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(raw_actions), 256):
            rows = np.arange(start, min(start + 256, len(raw_actions)))
            chunks.append(forward_rows(model, rows, contexts[rows], runtime, device)["mean"].cpu().numpy())
    return np.concatenate(chunks)


def score(model, data, raw_actions, reference, coordinates, device, metrics):
    validation = data["split_validation"]
    rows = predict(
        model,
        raw_actions[validation],
        data["context_index"][validation],
        reference,
        coordinates,
        device,
    )
    result = {}
    arrays = {}
    for context, name in enumerate(data["context_ids"].astype(str)):
        local = data["context_index"][validation] == context
        genes, prediction, mask, _ = metrics.collapse_equal_records(
            data["action_ids"][validation][local], rows[local], data["observed"][validation][local]
        )
        target_genes, truth, truth_mask, _ = metrics.collapse_equal_records(
            data["action_ids"][validation][local],
            data["targets"][validation][local],
            data["observed"][validation][local],
        )
        if not np.array_equal(genes, target_genes) or not np.array_equal(mask, truth_mask):
            raise ValueError("validation collapse drift")
        train = data["split_train"]
        fitting_centroid = data["targets"][train][data["context_index"][train] == context].mean(0)
        result[name] = metrics.point_metrics(
            prediction,
            truth,
            mask,
            reference["objective_query_scale"][context],
            fitting_centroid,
        )
        arrays[f"context{context}_action_ids"] = genes
        arrays[f"context{context}_prediction"] = prediction.astype(np.float32)
        arrays[f"context{context}_truth"] = truth.astype(np.float32)
        arrays[f"context{context}_observed"] = mask
    return result, arrays


def execute(output: Path, device_name: str) -> dict[str, object]:
    prepared = json.loads((output / "PREPARED.json").read_text())
    if sha256(output / "protocol.json") != prepared["protocolSha256"]:
        raise ValueError("protocol drift")
    if sha256(Path(__file__)) != prepared["sourceHashes"]["trainer.py"]:
        raise ValueError("executing trainer differs from prepared source")
    for name, digest in prepared["sourceHashes"].items():
        if sha256(output / "source" / name) != digest:
            raise ValueError(f"source drift: {name}")
    for name, path in {
        "basis": BASIS_ROOT / "basis.npz",
        "bpReference": PAIR / "bp128-present/reference.npz",
        "bpHelper": BP_HELPER,
        "bpPairReport": BP_PAIR_REPORT,
        "bpRidgeReport": BP_RIDGE_REPORT,
        "bpKernelReport": BP_KERNEL_REPORT,
    }.items():
        if sha256(path) != PINS[name]:
            raise ValueError(f"execution input drift: {name}")
    torch.use_deterministic_algorithms(True)
    helper = load(BP_HELPER, "fixed_basis_data_helper")
    fixed_module = load(output / "source/transition_model.py", "fixed_basis_train_core")
    old_module = load(output / "source/old_transition_model.py", "fixed_basis_train_old_core")
    objective = load(output / "source/mean_objective.py", "fixed_basis_objective")
    weighting = load(output / "source/objective_weighting.py", "fixed_basis_weighting")
    metrics = load(output / "source/four_context_baselines.py", "fixed_basis_metrics")
    data, actions, references, audit = helper.load_data()
    raw_actions = actions["bp128-present"]
    reference = references["bp128-present"]
    with np.load(BASIS_ROOT / "basis.npz", allow_pickle=False) as archive:
        coordinates = archive["query_coordinates"].astype(np.float32)
        if not np.array_equal(archive["query_ids"], data["query_ids"]):
            raise ValueError("basis query identity drift")
    device = torch.device(device_name)
    model, matched_initializer = initialize_fixed(fixed_module, old_module, helper)
    if not all(
        torch.equal(value, matched_initializer.state_dict()[name])
        for name, value in model.state_dict().items()
    ):
        raise ValueError("execution shared initialization drift")
    model = model.to(device)
    train = data["split_train"]
    contexts = data["context_index"][train]
    weights = weighting.training_row_weights(
        contexts,
        data["action_ids"][train],
        objective=weighting.EQUAL_CONTEXT_GENE_V1,
    ).astype(np.float32)
    target = torch.as_tensor(data["targets"][train], device=device)
    observed = torch.as_tensor(data["observed"][train], device=device)
    scale = torch.as_tensor(reference["objective_query_scale"], device=device)
    row_weight = torch.as_tensor(weights, device=device)
    runtime = make_runtime(raw_actions[train], reference, coordinates, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    batches = objective.deterministic_shuffled_batches(
        np.arange(len(train)), batch_size=BATCH, steps=STEPS, seed=SEED
    )
    losses = []
    started = time.monotonic()
    model.train()
    for step, rows in enumerate(batches, 1):
        optimizer.zero_grad(set_to_none=True)
        local = contexts[rows]
        prediction = forward_rows(model, rows, local, runtime, device)
        loss = objective.masked_standardized_mse(
            prediction["mean"], target[rows], observed[rows], scale[local], row_weight[rows]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 1000 == 0:
            print(json.dumps({"step": step, "recentLoss": float(np.mean(losses[-100:])), "seconds": time.monotonic() - started}), flush=True)
        if time.monotonic() - started > 270:
            raise TimeoutError("fixed-basis training exceeded 270 seconds")
    fit_seconds = time.monotonic() - started
    model_path = output / "model.safetensors"
    save_file({name: value.detach().cpu() for name, value in model.state_dict().items()}, str(model_path))
    np.savez_compressed(output / "reference.npz", **reference, fixed_query_coordinates=coordinates)
    # Freeze the final checkpoint before any validation scoring.
    write_json(
        output / "FROZEN-BEFORE-VALIDATION.json",
        {
            "modelSha256": sha256(model_path),
            "referenceSha256": sha256(output / "reference.npz"),
            "updates": STEPS,
            "validationEvaluations": 0,
        },
    )
    probe = []
    for context in range(3):
        local = train[data["context_index"][train] == context]
        probe.extend(local[:2].tolist())
    probe = np.asarray(probe, dtype=np.int64)
    expected = predict(model, raw_actions[probe], data["context_index"][probe], reference, coordinates, device)
    np.savez_compressed(
        output / "target-free-probe.npz",
        raw_action_features=raw_actions[probe],
        context_index=data["context_index"][probe],
        expected_mean=expected,
    )
    source_hashes = {
        f"source/{path.name}": sha256(path) for path in sorted((output / "source").glob("*.py"))
    }
    write_json(
        output / "artifact-manifest.json",
        {
            "schema": "slp.source3-fixed-response-basis-artifact/v1",
            "sha256": {
                "model.safetensors": sha256(model_path),
                "reference.npz": sha256(output / "reference.npz"),
                "target-free-probe.npz": sha256(output / "target-free-probe.npz"),
                **source_hashes,
            },
        },
    )
    verification = subprocess.run(
        [sys.executable, str(output / "source/verify_artifact.py"), str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    portable = json.loads(verification.stdout)
    validation_metrics, predictions = score(
        model, data, raw_actions, reference, coordinates, device, metrics
    )
    np.savez_compressed(output / "predictions.npz", **predictions)
    old_report = json.loads(BP_PAIR_REPORT.read_text())
    ridge_report = json.loads(BP_RIDGE_REPORT.read_text())
    kernel_report = json.loads(BP_KERNEL_REPORT.read_text())
    decisions = {}
    passed = True
    for name, candidate in validation_metrics.items():
        learned = old_report["arms"]["bp128-present"]["metrics"][name]
        ridge = ridge_report["contexts"][name]["arms"]["physical1156_bp128_present1"]["scores"]
        candidate_r = candidate["independently_query_centered_profile_pearson"]
        learned_r = learned["independently_query_centered_profile_pearson"]
        ridge_r = ridge["independentlyQueryCenteredPearson"]
        finite = all(np.isfinite(value) for value in (candidate_r, learned_r, ridge_r))
        primary = {
            "mseAtLeastOnePercentBelowLearnedQuery": 1 - candidate["gene_profile_raw_mse"] / learned["gene_profile_raw_mse"] >= 0.01,
            "allRequiredRFinite": finite,
            "rNonregressionVsLearnedQuery": finite and candidate_r >= learned_r,
        }
        prior = {
            "mseAtLeastTwoPercentBelowBpRidge": 1 - candidate["gene_profile_raw_mse"] / ridge["geneProfileMse"] >= 0.02,
            "rAtLeastPoint10": finite and candidate_r >= 0.1,
            "rNonregressionVsBpRidge": finite and candidate_r >= ridge_r,
        }
        context_pass = all(primary.values())
        passed &= context_pass
        kernel = kernel_report["contexts"][name]["candidate"]
        decisions[name] = {
            "primary": primary,
            "primaryPassed": context_pass,
            "worldAdvancementRule": prior,
            "bpKernelComparator": kernel,
            "beatsBpKernelMse": candidate["gene_profile_raw_mse"] < kernel["geneProfileMse"],
        }
    result = {
        "schema": "slp.source3-fixed-response-basis-neural-result/v1",
        "training": {"updates": STEPS, "fitSeconds": fit_seconds, "totalSeconds": time.monotonic() - started, "finalRecentLoss": float(np.mean(losses[-100:]))},
        "validationMetrics": validation_metrics,
        "decision": {"contexts": decisions, "passed": bool(passed)},
        "portableReload": portable,
        "artifacts": {"model": sha256(model_path), "reference": sha256(output / "reference.npz"), "predictions": sha256(output / "predictions.npz")},
        "inputAudit": audit,
        "protocolSha256": prepared["protocolSha256"],
        "validationEvaluations": 1,
        "testAccessed": False,
        "benchmarkAccessed": False,
    }
    write_json(output / "report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = prepare(args.output) if args.prepare_only else execute(args.output, args.device)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
