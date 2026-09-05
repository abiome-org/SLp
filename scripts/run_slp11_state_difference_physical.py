#!/usr/bin/env python3
"""Run the frozen v3 state-difference decoder experiment as one isolated change."""

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
MODEL = ROOT / "modules/slp-1-1-control-transition-v3/transition_model.py"
DATA = ROOT / "data/derived/slp11-human-gwps-fixed-panel-context-v1/replogle-k562-rpe1-gwps-complete-panel-development-v2-fixed-control-context.npz"
FEATURES = ROOT / "data/derived/slp11-human-physical/direct-experiments700-v1/human-esm-go-physical-features.npz"
HEPG2 = ROOT / "data/derived/slp11-human-gwps-fixed-panel-context-v1/nadig-hepg2-fixed-panel-control-context-v1.npz"
COMPARATOR = ROOT / "results/slp11-transition/human-gwps-fixed-context-minimal-control-physical-state128-response32-seed731-v1/model/report.json"
HASHES = {
    "launcher": "140212ebbd02fd8de9e2970271aabe11ab0c36bcdc4057531bbdb8d339140e1c",
    "model": "75a487046d30d399000bd50dbe7bf2c642fb1acf297a1c78e1d65b5adbb5a832",
    "data": "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c",
    "features": "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7",
    "hepg2": "382626401ee38e8d5084ac9f86ffc44bd10408826fb85a94ede8eb908cdf5b27",
    "comparator": "49333ade99f04d96e9d4c4ccc2fc01c002170b38f02d10f88fdc8559d274203d",
}


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
        "--original-report", str(COMPARATOR),
        "--model-source", str(MODEL),
        "--model-sha256", HASHES["model"],
        "--training-objective", "uniform-row-v1",
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
    spec = importlib.util.spec_from_file_location("slp11_state_difference_postrun", MODEL)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen v3 model")
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
            for name in (
                "record_ids",
                "action_ids",
                "context_index",
                "basal_control",
            )
        }
    with np.load(FEATURES, allow_pickle=False) as archive:
        keys = list(zip(archive["entity_taxon"].tolist(), archive["entity_id"].tolist()))
        values = archive["feature_values"]
    lookup = dict(zip(keys, values, strict=True))
    row_by_record = {str(item): row for row, item in enumerate(data["record_ids"])}
    chosen_prediction = np.asarray(
        [int(np.flatnonzero(prediction["context_index"] == index)[0]) for index in range(3)],
        dtype=np.int64,
    )
    source_rows = np.asarray(
        [row_by_record[str(item)] for item in prediction["record_ids"][chosen_prediction]],
        dtype=np.int64,
    )
    action = np.stack(
        [lookup[(9606, str(item))] for item in data["action_ids"][source_rows]]
    ).astype(np.float32)
    action = ((action - reference["feature_mean"]) / reference["feature_std"]).astype(np.float32)
    queries = (
        (reference["query_features"] - reference["query_feature_mean"])
        / reference["query_feature_std"]
    ).astype(np.float32)
    contexts = data["context_index"][source_rows]
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
        query_state = model.query_encoder(torch.from_numpy(queries))
        direct_delta = reference["delta_amplitude"][None, :] * (
            model.mean_state(result["intervention_delta"]) @ query_state.T
            / np.sqrt(config.state_dim)
        ).numpy()
        decoded_difference = reference["delta_amplitude"][None, :] * (
            (
                model.mean_state(result["state"])
                - model.mean_state(result["basal_state"])
            )
            @ query_state.T
            / np.sqrt(config.state_dim)
        ).numpy()
    reload_error = float(
        np.max(np.abs(result["mean"].numpy() - prediction["mean"][chosen_prediction]))
    )
    direct_error = float(np.max(np.abs(result["delta"].numpy() - direct_delta)))
    difference_error = float(np.max(np.abs(result["delta"].numpy() - decoded_difference)))
    empty_actions = torch.empty((3, 0, config.action_feature_dim), dtype=torch.float32)
    with torch.no_grad():
        empty = model(
            empty_actions,
            torch.from_numpy(queries),
            torch.from_numpy(data["basal_control"]),
            torch.from_numpy(reference["delta_amplitude"]),
            torch.ones((3, len(queries)), dtype=torch.float32),
            torch.from_numpy(queries[reference["context_query_indices"]]),
            torch.from_numpy(reference["context_values"]),
            torch.from_numpy(reference["context_mask"]),
            action_mask=torch.empty((3, 0), dtype=torch.bool),
        )
    empty_identity = bool(
        torch.equal(empty["mean"], torch.from_numpy(data["basal_control"]))
        and torch.count_nonzero(empty["delta"]) == 0
        and torch.count_nonzero(empty["intervention_delta"]) == 0
        and torch.equal(empty["state"], empty["basal_state"])
    )
    if reload_error > 2e-6 or direct_error > 1e-7 or difference_error > 2e-6 or not empty_identity:
        raise RuntimeError("v3 postrun numerical contract failed")
    return {
        "knownContextSourceReloadMaxAbsError": reload_error,
        "directInterventionDeltaDecodeMaxAbsError": direct_error,
        "decodedStateDifferenceMaxAbsError": difference_error,
        "emptyActionIdentityExact": empty_identity,
        "knownContextRecords": 3,
    }


def run(output: Path) -> dict[str, object]:
    for label, path in (
        ("launcher", LAUNCHER),
        ("model", MODEL),
        ("data", DATA),
        ("features", FEATURES),
        ("hepg2", HEPG2),
        ("comparator", COMPARATOR),
    ):
        if sha256_file(path) != HASHES[label]:
            raise ValueError(f"{label} source drift")
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    shutil.copyfile(Path(__file__), source / Path(__file__).name)
    protocol = {
        "schema": "slp.state-difference-physical-state128-experiment/v1",
        "hypothesis": "A decoder of the latent intervention-state difference fixes zero-effect semantics and preserves or improves held-intervention-gene molecular performance",
        "isolatedChange": "v3 decodes intervention_delta rather than total state with an action-presence output gate",
        "primaryRule": "in every source context: at least 0.02 nats gene-macro NLL gain versus mean and full physical ridge, adjusted Pearson at least 0.10, and no NLL or adjusted-Pearson regression versus v2 physical/state128",
        "data": {"path": str(DATA), "sha256": HASHES["data"], "panelQueries": 7036, "commonControlTokens": 6789},
        "features": {"path": str(FEATURES), "sha256": HASHES["features"], "dimensions": 1156, "encoder": "ESM2-8M plus GO plus direct physical neighbors"},
        "comparator": {"path": str(COMPARATOR), "sha256": HASHES["comparator"], "model": "v2 physical/state128 seed731"},
        "modelSource": {"path": str(MODEL), "sha256": HASHES["model"]},
        "launcherSource": {"path": str(LAUNCHER), "sha256": HASHES["launcher"]},
        "hepg2ControlOnly": {"path": str(HEPG2), "sha256": HASHES["hepg2"], "perturbedRowsRead": 0},
        "configuration": {"trainingObjective": "uniform-row-v1", "hidden": 128, "stateDim": 128, "dropout": 0.2, "queryBasisRank": 32, "contextTokens": 64, "learningRate": 0.0005, "weightDecay": 0.1, "seed": 731, "epochs": 180, "patience": 30, "maxSeconds": 1800},
        "excludedChanges": ["ESM2-650M features", "balanced loss", "6517-query panel"],
        "hepg2OutcomesRead": False,
        "jurkatOutcomesRead": False,
        "benchmarkOutcomesRead": False,
        "command": command(output),
        "sourceSha256": sha256_file(source / Path(__file__).name),
    }
    protocol_path = output / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "protocol-frozen", "sha256": sha256_file(protocol_path)}), flush=True)
    subprocess.run(command(output), cwd=ROOT, check=True)
    checks = postrun_checks(output / "model")
    with (output / "model/report.json").open(encoding="utf-8") as stream:
        report = json.load(stream)
    summary = {
        "schema": "slp.state-difference-physical-state128-summary/v1",
        "status": "development-rule-passed" if report["advancement"]["passed"] else "development-rule-failed",
        "advancement": report["advancement"],
        "results": report["results"],
        "bestEpoch": report["bestEpoch"],
        "elapsedSeconds": report["elapsedSeconds"],
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
    print(json.dumps({"event": "state-difference-finished", "summary": summary}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
