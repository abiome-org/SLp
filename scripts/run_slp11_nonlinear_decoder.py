#!/usr/bin/env python3
"""Profile and run the isolated v4 nonlinear observation-decoder pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_slp11_minimal_control_common_context.py"
MODEL = ROOT / "modules/slp-1-1-control-transition-v4/transition_model.py"
CONTRACT = MODEL.with_name("CONTRACT.md")
SCORING = ROOT / "modules/slp-1-1-world-transition-v1/context_transfer_scoring.py"
DATA = (
    ROOT
    / "data/derived/slp11-human-gwps-fixed-panel-context-v1"
    / "replogle-k562-rpe1-gwps-complete-panel-development-v2-fixed-control-context.npz"
)
FEATURES = (
    ROOT
    / "data/derived/slp11-human-physical/direct-experiments700-v1"
    / "human-esm-go-physical-features.npz"
)
HEPG2 = (
    ROOT
    / "data/derived/slp11-human-gwps-fixed-panel-context-v1"
    / "nadig-hepg2-fixed-panel-control-context-v1.npz"
)
COMPARATOR_V2 = (
    ROOT
    / "results/slp11-transition/human-gwps-fixed-context-minimal-control-physical-state128-response32-seed731-v1/model/report.json"
)
COMPARATOR_V3 = (
    ROOT
    / "results/slp11-transition/human-gwps-fixed-context-state-difference-physical-state128-seed731-v1/model/report.json"
)
REFERENCE_V2 = COMPARATOR_V2.parent / "reference.npz"
RIDGE = ROOT / "results/slp11-transition/physical-features-ridge-screen-v1/predictions.npz"

EXPECTED = {
    LAUNCHER: "6222c4df4ad898220f3e22aaa17a5e7e6e848ad19b2d7ef3d5983daa13aa8c47",
    DATA: "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c",
    FEATURES: "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7",
    HEPG2: "382626401ee38e8d5084ac9f86ffc44bd10408826fb85a94ede8eb908cdf5b27",
    COMPARATOR_V2: "49333ade99f04d96e9d4c4ccc2fc01c002170b38f02d10f88fdc8559d274203d",
    COMPARATOR_V3: "1e8b3b9ce951a3b4164a8f187577760ccf1721c8c6b4e721754cbd3cb9e4600e",
    REFERENCE_V2: "a9f3fd2679b5a52e20dddddd427d8664b2c226f2db91bdae1e44a63e66568562",
    RIDGE: "c91d96b724f9b99169536ba17a3cce6f0c8578d603257b830a32a335f7e1c525",
}
BATCH_SIZE = 64


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not load module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def verify_inputs() -> None:
    for path, expected in EXPECTED.items():
        if sha256(path) != expected:
            raise ValueError(f"pinned input drift: {path}")


def command(output: Path, batch_size: int) -> list[str]:
    return [
        sys.executable,
        str(LAUNCHER),
        "--data", str(DATA),
        "--features", str(FEATURES),
        "--feature-sha256", EXPECTED[FEATURES],
        "--hepg2-control", str(HEPG2),
        "--original-report", str(COMPARATOR_V2),
        "--model-source", str(MODEL),
        "--model-sha256", sha256(MODEL),
        "--training-objective", "uniform-row-v1",
        "--output", str(output / "model"),
        "--device", "cuda",
        "--epochs", "180",
        "--patience", "30",
        "--max-seconds", "1800",
        "--batch-size", str(batch_size),
        "--context-tokens", "64",
        "--query-basis-rank", "32",
        "--hidden", "128",
        "--state-dim", "128",
        "--dropout", "0.2",
        "--learning-rate", "0.0005",
        "--weight-decay", "0.1",
        "--ridge-alpha", "10000",
        "--seed", "731",
        "--cpu-threads", "4",
    ]


def centered_nonregression(candidate: float, full_ridge: float) -> bool:
    """Additional fixed gate for independently centered profile correlation."""

    if not np.isfinite(candidate) or not np.isfinite(full_ridge):
        raise ValueError("centered correlations must be finite")
    return bool(candidate >= full_ridge)


def profile(output: Path) -> dict[str, object]:
    """Run one actual-shape forward/backward memory and timing profile."""

    verify_inputs()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU profiling fallback")
    output.mkdir(parents=True, exist_ok=False)
    module = load_module(MODEL, "slp11_nonlinear_decoder_profile")
    with np.load(DATA, allow_pickle=False) as archive:
        train = archive["split_train"][:BATCH_SIZE]
        action_ids = archive["action_ids"][train]
        context = archive["context_index"][train]
        targets = archive["targets"][train]
        observed = archive["observed"][train]
        basal_control = archive["basal_control"]
    with np.load(FEATURES, allow_pickle=False) as archive:
        lookup = dict(
            zip(
                zip(archive["entity_taxon"].tolist(), archive["entity_id"].tolist(), strict=True),
                archive["feature_values"],
                strict=True,
            )
        )
    with np.load(REFERENCE_V2, allow_pickle=False) as archive:
        reference = {name: archive[name] for name in archive.files}
    actions = np.stack([lookup[(9606, str(item))] for item in action_ids])
    actions = ((actions - reference["feature_mean"]) / reference["feature_std"]).astype(np.float32)
    queries = (
        (reference["query_features"] - reference["query_feature_mean"])
        / reference["query_feature_std"]
    ).astype(np.float32)
    device = torch.device("cuda")
    torch.manual_seed(731)
    torch.cuda.manual_seed_all(731)
    model = module.MinimalControlTransition(
        module.Config(1156, 1188, hidden_dim=128, state_dim=128, dropout=0.2)
    ).to(device)
    model.train()

    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    action_tensor = tensor(actions)
    query_tensor = tensor(queries)
    target_tensor = tensor(targets)
    observed_tensor = torch.as_tensor(observed, dtype=torch.bool, device=device)
    control_tensor = tensor(basal_control[context])
    amplitude = tensor(reference["delta_amplitude"])
    scale = torch.ones((BATCH_SIZE, targets.shape[1]), dtype=torch.float32, device=device)
    basal_features = query_tensor[reference["context_query_indices"]]
    basal_values = tensor(reference["context_values"][context])
    basal_mask = torch.as_tensor(reference["context_mask"][context], dtype=torch.bool, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    prediction = model(
        action_tensor,
        query_tensor,
        control_tensor,
        amplitude,
        scale,
        basal_features,
        basal_values,
        basal_mask,
    )
    loss = module.gaussian_loss(prediction, target_tensor, observed_tensor)
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "schema": "slp.nonlinear-decoder-actual-shape-profile/v1",
        "batchSize": BATCH_SIZE,
        "queries": int(targets.shape[1]),
        "actionFeatureDim": 1156,
        "queryFeatureDim": 1188,
        "stateDim": 128,
        "decoderHiddenDim": 64,
        "forwardBackwardSeconds": elapsed,
        "peakAllocatedBytes": int(torch.cuda.max_memory_allocated(device)),
        "peakReservedBytes": int(torch.cuda.max_memory_reserved(device)),
        "device": torch.cuda.get_device_name(device),
        "finiteLoss": bool(torch.isfinite(loss)),
        "batch64Accepted": True,
        "modelSha256": sha256(MODEL),
        "launcherSha256": EXPECTED[LAUNCHER],
        "dataSha256": EXPECTED[DATA],
        "featureSha256": EXPECTED[FEATURES],
        "developmentTrainRowsOnly": True,
        "testOrExternalOutcomesAccessed": False,
    }
    path = output / "profile.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["profileSha256"] = sha256(path)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def postrun_reload(model_dir: Path) -> dict[str, object]:
    module = load_module(MODEL, "slp11_nonlinear_decoder_postrun")
    config = module.Config(**json.loads((model_dir / "model-config.json").read_text()))
    model = module.MinimalControlTransition(config).eval()
    model.load_state_dict(load_file(model_dir / "model.safetensors", device="cpu"))
    with np.load(model_dir / "reference.npz", allow_pickle=False) as archive:
        reference = {name: archive[name] for name in archive.files}
    with np.load(model_dir / "development-predictions.npz", allow_pickle=False) as archive:
        prediction = {name: archive[name] for name in archive.files}
    with np.load(DATA, allow_pickle=False) as archive:
        data = {name: archive[name] for name in ("record_ids", "action_ids", "context_index", "basal_control")}
    with np.load(FEATURES, allow_pickle=False) as archive:
        lookup = dict(
            zip(
                zip(archive["entity_taxon"].tolist(), archive["entity_id"].tolist(), strict=True),
                archive["feature_values"],
                strict=True,
            )
        )
    row_lookup = {str(item): row for row, item in enumerate(data["record_ids"])}
    chosen = np.asarray([np.flatnonzero(prediction["context_index"] == index)[0] for index in range(3)])
    rows = np.asarray([row_lookup[str(item)] for item in prediction["record_ids"][chosen]])
    actions = np.stack([lookup[(9606, str(item))] for item in data["action_ids"][rows]])
    actions = ((actions - reference["feature_mean"]) / reference["feature_std"]).astype(np.float32)
    queries = (
        (reference["query_features"] - reference["query_feature_mean"])
        / reference["query_feature_std"]
    ).astype(np.float32)
    contexts = data["context_index"][rows]
    with torch.no_grad():
        result = model(
            torch.from_numpy(actions),
            torch.from_numpy(queries),
            torch.from_numpy(data["basal_control"][contexts]),
            torch.from_numpy(reference["delta_amplitude"]),
            torch.ones((3, len(queries))),
            torch.from_numpy(queries[reference["context_query_indices"]]),
            torch.from_numpy(reference["context_values"][contexts]),
            torch.from_numpy(reference["context_mask"][contexts]),
        )
        query_state = model.query_encoder(torch.from_numpy(queries))
        direct = reference["delta_amplitude"][None, :] * (
            model.decode_state_queries(result["state"], query_state)
            - model.decode_state_queries(result["basal_state"], query_state)
        ).numpy()
        empty = model(
            torch.empty((3, 0, config.action_feature_dim)),
            torch.from_numpy(queries),
            torch.from_numpy(data["basal_control"]),
            torch.from_numpy(reference["delta_amplitude"]),
            torch.ones((3, len(queries))),
            torch.from_numpy(queries[reference["context_query_indices"]]),
            torch.from_numpy(reference["context_values"]),
            torch.from_numpy(reference["context_mask"]),
            action_mask=torch.empty((3, 0), dtype=torch.bool),
        )
    reload_error = float(np.max(np.abs(result["mean"].numpy() - prediction["mean"][chosen])))
    difference_error = float(np.max(np.abs(result["delta"].numpy() - direct)))
    empty_identity = bool(
        torch.equal(empty["mean"], torch.from_numpy(data["basal_control"]))
        and torch.count_nonzero(empty["delta"]) == 0
        and torch.equal(empty["state"], empty["basal_state"])
    )
    if reload_error > 2e-6 or difference_error > 2e-6 or not empty_identity:
        raise RuntimeError("v4 fitted source reload failed")
    return {
        "knownContextSourceReloadMaxAbsError": reload_error,
        "decodedStateDifferenceMaxAbsError": difference_error,
        "emptyActionIdentityExact": empty_identity,
    }


def independently_centered_metrics(model_dir: Path) -> dict[str, object]:
    scoring = load_module(SCORING, "slp11_nonlinear_decoder_scoring")
    with np.load(DATA, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    train, validation = data["split_train"], data["split_validation"]
    with np.load(model_dir / "development-predictions.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["record_ids"], data["record_ids"][validation]):
            raise ValueError("v4 validation forecast ordering drift")
        world = archive["mean"]
    results = {}
    with np.load(RIDGE, allow_pickle=False) as ridge:
        for context, name in enumerate(data["context_ids"]):
            positions = np.flatnonzero(data["context_index"][validation] == context)
            rows = validation[positions]
            centroid = data["targets"][train[data["context_index"][train] == context]].mean(
                axis=0, dtype=np.float64
            )
            arms = {}
            for label, prediction in (
                ("world", world[positions]),
                ("fullPhysicalRidge", ridge[f"context{context}_physical"]),
            ):
                profiles = scoring.collapse_gene_profiles(
                    prediction,
                    data["targets"][rows],
                    data["observed"][rows],
                    data["action_ids"][rows],
                    data["record_ids"][rows],
                )
                arms[label] = scoring.score_gene_profiles(profiles, centroid)
            world_r = arms["world"]["primaryIndependentlyCenteredGeneMacroProfilePearson"]
            ridge_r = arms["fullPhysicalRidge"]["primaryIndependentlyCenteredGeneMacroProfilePearson"]
            results[str(name)] = {
                "arms": arms,
                "worldNonregressionVsFullPhysicalRidge": centered_nonregression(world_r, ridge_r),
            }
    return results


def run(output: Path, profile_path: Path) -> dict[str, object]:
    verify_inputs()
    profile_result = json.loads(profile_path.read_text(encoding="utf-8"))
    if (
        not profile_result.get("batch64Accepted")
        or profile_result.get("batchSize") != BATCH_SIZE
        or profile_result.get("queries") != 7036
        or profile_result.get("modelSha256") != sha256(MODEL)
    ):
        raise ValueError("actual-shape batch-64 profile does not authorize frozen run")
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    for path in (Path(__file__), MODEL, CONTRACT, LAUNCHER, SCORING):
        shutil.copyfile(path, source / path.name)
    protocol = {
        "schema": "slp.nonlinear-state-query-decoder-experiment/v1",
        "hypothesis": (
            "Replacing the rank-limited bilinear observation with a nonlinear basal-state-aware "
            "state/query difference decoder improves held-intervention molecular landscapes."
        ),
        "isolatedChange": (
            "D(basal_state + intervention_delta, query) - D(basal_state, query), using a 64-unit "
            "GELU decoder with no decoder dropout"
        ),
        "primaryRule": (
            "in every source context: world gene-macro NLL improves at least 0.02 nats over mean "
            "and full physical ridge, training-centroid-adjusted profile Pearson is at least 0.10, "
            "and NLL and adjusted Pearson do not regress versus v2 report49333"
        ),
        "additionalRequiredGate": (
            "independently prediction/truth-centered gene-macro profile Pearson does not regress "
            "versus full physical ridge in every source context"
        ),
        "descriptiveComparator": {"v3Report": str(COMPARATOR_V3), "sha256": EXPECTED[COMPARATOR_V3]},
        "inputs": {str(path.relative_to(ROOT)): digest for path, digest in EXPECTED.items()},
        "modelSource": {"path": str(MODEL), "sha256": sha256(MODEL)},
        "configuration": {
            "trainingObjective": "uniform-row-v1",
            "hidden": 128,
            "stateDim": 128,
            "decoderHiddenDim": 64,
            "dropout": 0.2,
            "decoderDropout": 0.0,
            "queryBasisRank": 32,
            "contextTokens": 64,
            "learningRate": 0.0005,
            "weightDecay": 0.1,
            "seed": 731,
            "epochs": 180,
            "patience": 30,
            "maxSeconds": 1800,
            "batchSize": BATCH_SIZE,
        },
        "profile": {"path": str(profile_path), "sha256": sha256(profile_path), **profile_result},
        "interpretation": "nonlinear molecular measurement-decoder experiment; not identified dynamics",
        "excluded": "no response encoder, auxiliary loss, learned gene ID, or molecular action-presence gate",
        "testHepg2JurkatFrangiehOrBenchmarkOutcomesAccessed": False,
        "command": command(output, BATCH_SIZE),
        "sourceHashes": {path.name: sha256(path) for path in source.iterdir()},
    }
    protocol_path = output / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "protocol-frozen", "sha256": sha256(protocol_path)}), flush=True)
    subprocess.run(command(output, BATCH_SIZE), cwd=ROOT, check=True)
    model_dir = output / "model"
    reload_checks = postrun_reload(model_dir)
    centered = independently_centered_metrics(model_dir)
    report = json.loads((model_dir / "report.json").read_text(encoding="utf-8"))
    v3_report = json.loads(COMPARATOR_V3.read_text(encoding="utf-8"))["results"]
    v3_comparison = {}
    for context, values in report["results"].items():
        current = values["world"]
        prior = v3_report[context]["world"]
        v3_comparison[context] = {
            "geneMacroNllCurrentMinusV3": float(current["gene_macro_nll"])
            - float(prior["gene_macro_nll"]),
            "adjustedPearsonCurrentMinusV3": float(
                current["gene_macro_profile_centroid_adjusted_pearson_mean"]
            )
            - float(prior["gene_macro_profile_centroid_adjusted_pearson_mean"]),
        }
    centered_passed = all(
        item["worldNonregressionVsFullPhysicalRidge"] for item in centered.values()
    )
    summary = {
        "schema": "slp.nonlinear-state-query-decoder-summary/v1",
        "status": (
            "development-rule-passed"
            if report["advancement"]["passed"] and centered_passed
            else "development-rule-failed"
        ),
        "launcherAdvancement": report["advancement"],
        "additionalCenteredGatePassed": centered_passed,
        "independentlyCenteredMetrics": centered,
        "results": report["results"],
        "descriptiveV3Comparison": v3_comparison,
        "bestEpoch": report["bestEpoch"],
        "elapsedSeconds": report["elapsedSeconds"],
        "reloadChecks": reload_checks,
        "protocolSha256": sha256(protocol_path),
        "modelReportSha256": sha256(model_dir / "report.json"),
        "checkpointSha256": sha256(model_dir / "model.safetensors"),
        "testHepg2JurkatFrangiehOrBenchmarkOutcomesAccessed": False,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "nonlinear-decoder-finished", "summary": summary}), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--profile-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-report", type=Path)
    args = parser.parse_args()
    if args.profile_only:
        profile(args.output)
    else:
        if args.profile_report is None:
            parser.error("--run requires --profile-report")
        run(args.output, args.profile_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
