#!/usr/bin/env python3
"""Run the frozen observed-state auxiliary representation-learning pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_slp11_minimal_control_common_context.py"
MODEL = ROOT / "modules/slp-1-1-observed-state-transition-v1/transition_model.py"
CONTRACT = MODEL.with_name("CONTRACT.md")
DATA = ROOT / "data/derived/slp11-human-gwps-fixed-panel-context-v1/replogle-k562-rpe1-gwps-complete-panel-development-v2-fixed-control-context.npz"
FEATURES = ROOT / "data/derived/slp11-human-physical/direct-experiments700-v1/human-esm-go-physical-features.npz"
HEPG2 = ROOT / "data/derived/slp11-human-gwps-fixed-panel-context-v1/nadig-hepg2-fixed-panel-control-context-v1.npz"
COMPARATOR_V2 = ROOT / "results/slp11-transition/human-gwps-fixed-context-minimal-control-physical-state128-response32-seed731-v1/model/report.json"
COMPARATOR_V3 = ROOT / "results/slp11-transition/human-gwps-fixed-context-state-difference-physical-state128-seed731-v1/model/report.json"
HASHES = {
    "launcher": "6222c4df4ad898220f3e22aaa17a5e7e6e848ad19b2d7ef3d5983daa13aa8c47",
    "model": "42abbffde432270e48a02b9a54db7938dedca10951c672ecad65f865434653c4",
    "data": "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c",
    "features": "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7",
    "hepg2": "382626401ee38e8d5084ac9f86ffc44bd10408826fb85a94ede8eb908cdf5b27",
    "comparator_v2": "49333ade99f04d96e9d4c4ccc2fc01c002170b38f02d10f88fdc8559d274203d",
    "comparator_v3": "1e8b3b9ce951a3b4164a8f187577760ccf1721c8c6b4e721754cbd3cb9e4600e",
}
OBJECTIVE = "uniform-row-observed-state-aux-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(output: Path) -> list[str]:
    return [
        sys.executable,
        str(LAUNCHER),
        "--data", str(DATA),
        "--features", str(FEATURES),
        "--feature-sha256", HASHES["features"],
        "--hepg2-control", str(HEPG2),
        "--original-report", str(COMPARATOR_V2),
        "--model-source", str(MODEL),
        "--model-sha256", HASHES["model"],
        "--training-objective", OBJECTIVE,
        "--output", str(output / "model"),
        "--device", "cuda",
        "--epochs", "180",
        "--patience", "30",
        "--max-seconds", "1800",
        "--batch-size", "64",
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


def load_model_source():
    spec = importlib.util.spec_from_file_location("slp11_observed_aux_postrun", MODEL)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen observed-state model")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def postrun_checks(model_dir: Path) -> dict[str, object]:
    module = load_model_source()
    with (model_dir / "model-config.json").open(encoding="utf-8") as stream:
        config = module.Config(**json.load(stream))
    model = module.MinimalControlTransition(config).eval()
    model.load_state_dict(load_file(model_dir / "model.safetensors", device="cpu"))
    with np.load(model_dir / "reference.npz", allow_pickle=False) as archive:
        reference = {name: archive[name] for name in archive.files}
    with np.load(model_dir / "development-predictions.npz", allow_pickle=False) as archive:
        prediction = {name: archive[name] for name in archive.files}
    with np.load(DATA, allow_pickle=False) as archive:
        data = {
            name: archive[name]
            for name in ("record_ids", "action_ids", "context_index", "basal_control")
        }
    with np.load(FEATURES, allow_pickle=False) as archive:
        keys = list(zip(archive["entity_taxon"].tolist(), archive["entity_id"].tolist()))
        values = archive["feature_values"]
    lookup = dict(zip(keys, values, strict=True))
    row_by_record = {str(item): row for row, item in enumerate(data["record_ids"])}
    chosen = np.asarray(
        [int(np.flatnonzero(prediction["context_index"] == index)[0]) for index in range(3)]
    )
    rows = np.asarray(
        [row_by_record[str(item)] for item in prediction["record_ids"][chosen]]
    )
    action = np.stack(
        [lookup[(9606, str(item))] for item in data["action_ids"][rows]]
    ).astype(np.float32)
    action = ((action - reference["feature_mean"]) / reference["feature_std"]).astype(np.float32)
    queries = (
        (reference["query_features"] - reference["query_feature_mean"])
        / reference["query_feature_std"]
    ).astype(np.float32)
    contexts = data["context_index"][rows]

    # A fitted inference reload must remain independent of the training-only encoder.
    def forbidden(*_args, **_kwargs):
        raise RuntimeError("training-only response encoder called by inference")

    model.response_keys.forward = forbidden
    model.response_state.forward = forbidden
    with torch.no_grad():
        result = model(
            torch.from_numpy(action),
            torch.from_numpy(queries),
            torch.from_numpy(data["basal_control"][contexts]),
            torch.from_numpy(reference["delta_amplitude"]),
            torch.ones((3, len(queries)), dtype=torch.float32),
            torch.from_numpy(queries[reference["context_query_indices"]]),
            torch.from_numpy(reference["context_values"][contexts]),
            torch.from_numpy(reference["context_mask"][contexts]),
        )
        empty = model(
            torch.empty((3, 0, config.action_feature_dim), dtype=torch.float32),
            torch.from_numpy(queries),
            torch.from_numpy(data["basal_control"]),
            torch.from_numpy(reference["delta_amplitude"]),
            torch.ones((3, len(queries)), dtype=torch.float32),
            torch.from_numpy(queries[reference["context_query_indices"]]),
            torch.from_numpy(reference["context_values"]),
            torch.from_numpy(reference["context_mask"]),
            action_mask=torch.empty((3, 0), dtype=torch.bool),
        )
    reload_error = float(np.max(np.abs(result["mean"].numpy() - prediction["mean"][chosen])))
    empty_identity = bool(
        torch.equal(empty["mean"], torch.from_numpy(data["basal_control"]))
        and torch.count_nonzero(empty["delta"]) == 0
        and torch.count_nonzero(empty["intervention_delta"]) == 0
        and torch.equal(empty["state"], empty["basal_state"])
    )
    if reload_error > 2e-6 or not empty_identity:
        raise RuntimeError("observed-state fitted inference contract failed")
    return {
        "knownContextSourceReloadMaxAbsError": reload_error,
        "emptyActionIdentityExact": empty_identity,
        "trainingOnlyResponseEncoderForbiddenDuringReload": True,
        "knownContextRecords": 3,
    }


def descriptive_v3_comparison(results: dict[str, object]) -> dict[str, object]:
    with COMPARATOR_V3.open(encoding="utf-8") as stream:
        prior = json.load(stream)["results"]
    comparison = {}
    for context, values in results.items():
        world = values["world"]
        old = prior[context]["world"]
        comparison[context] = {
            "geneMacroNllDifferenceCurrentMinusV3": float(world["gene_macro_nll"])
            - float(old["gene_macro_nll"]),
            "adjustedPearsonDifferenceCurrentMinusV3": float(
                world["gene_macro_profile_centroid_adjusted_pearson_mean"]
            )
            - float(old["gene_macro_profile_centroid_adjusted_pearson_mean"]),
        }
    return comparison


def run(output: Path) -> dict[str, object]:
    inputs = {
        "launcher": LAUNCHER,
        "model": MODEL,
        "data": DATA,
        "features": FEATURES,
        "hepg2": HEPG2,
        "comparator_v2": COMPARATOR_V2,
        "comparator_v3": COMPARATOR_V3,
    }
    for label, path in inputs.items():
        if sha256_file(path) != HASHES[label]:
            raise ValueError(f"{label} source drift")
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    for path in (Path(__file__), MODEL, CONTRACT, LAUNCHER):
        shutil.copyfile(path, source / path.name)
    protocol = {
        "schema": "slp.observed-state-auxiliary-experiment/v1",
        "hypothesis": "A training-only observed-response state target improves control-anchored held-intervention-gene point forecasts",
        "model": "v3 state-difference forecast topology plus a training-only observed-response encoder",
        "objective": {
            "name": OBJECTIVE,
            "formula": "forecast Gaussian NLL + 0.1 * posterior reconstruction Gaussian NLL + 0.1 * normalized latent-state MSE",
            "latentTeacherStopGradient": True,
            "epsilon": 1e-6,
            "validationAndEarlyStopping": "forecast gene-macro NLL only",
        },
        "primaryRule": "in every source context: at least 0.02 nats gene-macro NLL gain versus mean and full physical ridge, adjusted Pearson at least 0.10, and no NLL or adjusted-Pearson regression versus v2 physical/state128",
        "descriptiveComparator": {"path": str(COMPARATOR_V3), "sha256": HASHES["comparator_v3"]},
        "inputs": {label: {"path": str(path), "sha256": HASHES[label]} for label, path in inputs.items()},
        "configuration": {"hidden": 128, "stateDim": 128, "dropout": 0.2, "queryBasisRank": 32, "contextTokens": 64, "learningRate": 0.0005, "weightDecay": 0.1, "seed": 731, "epochs": 180, "patience": 30, "maxSeconds": 1800, "batchSize": 64},
        "inferenceUsesPerturbedResponse": False,
        "hepg2OutcomesRead": False,
        "jurkatOutcomesRead": False,
        "benchmarkOutcomesRead": False,
        "claimsExcluded": ["time dynamics", "single-cell generation", "causal latent-state recovery"],
        "command": command(output),
        "sourceHashes": {path.name: sha256_file(source / path.name) for path in (Path(__file__), MODEL, CONTRACT, LAUNCHER)},
    }
    protocol_path = output / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "protocol-frozen", "sha256": sha256_file(protocol_path)}), flush=True)
    subprocess.run(command(output), cwd=ROOT, check=True)
    checks = postrun_checks(output / "model")
    with (output / "model/report.json").open(encoding="utf-8") as stream:
        report = json.load(stream)
    best_history = next(item for item in report["history"] if item["epoch"] == report["bestEpoch"])
    components = best_history.get("trainingLossComponents")
    if not isinstance(components, dict) or set(components) != {
        "total", "forecast_nll", "reconstruction_nll", "latent_match"
    }:
        raise RuntimeError("auxiliary loss components missing from frozen report")
    summary = {
        "schema": "slp.observed-state-auxiliary-summary/v1",
        "status": "development-rule-passed" if report["advancement"]["passed"] else "development-rule-failed",
        "advancement": report["advancement"],
        "results": report["results"],
        "descriptiveV3Comparison": descriptive_v3_comparison(report["results"]),
        "bestEpoch": report["bestEpoch"],
        "bestEpochTrainingLossComponents": components,
        "elapsedSeconds": report["elapsedSeconds"],
        "parameters": report["parameters"],
        "controlIdentityAfterCheckpoint": report["controlIdentityAfterCheckpoint"],
        "numericalChecks": checks,
        "protocolSha256": sha256_file(protocol_path),
        "modelReportSha256": sha256_file(output / "model/report.json"),
        "checkpointSha256": sha256_file(output / "model/model.safetensors"),
        "hepg2OutcomesRead": False,
        "jurkatOutcomesRead": False,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "observed-state-auxiliary-finished", "summary": summary}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
