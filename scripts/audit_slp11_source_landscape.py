"""Rescore saved source forecasts after removing their separate average profiles."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules/slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))
from context_transfer_scoring import collapse_gene_profiles, score_gene_profiles

DATA = ROOT / "data/derived/slp11-human-gwps/complete-panel-v1/development.npz"
DATA_SHA = "006b4bb127a09073a7f409d81a7bccce96bb961879cb5e57dce56b48eb8e664b"
RUNS = ROOT / "results/slp11-transition"
MODELS = {
    "minimal_control_v2": RUNS / "human-gwps-fixed-context-minimal-control-physical-state128-response32-seed731-v1/model/development-predictions.npz",
    "state_difference_v3": RUNS / "human-gwps-fixed-context-state-difference-physical-state128-seed731-v1/model/development-predictions.npz",
    "observed_state_auxiliary": RUNS / "human-gwps-fixed-context-observed-state-auxiliary-physical-state128-seed731-v1/model/development-predictions.npz",
}
RIDGE = RUNS / "physical-features-ridge-screen-v1/predictions.npz"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(output: Path) -> None:
    if sha(DATA) != DATA_SHA:
        raise ValueError("source development snapshot changed")
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    for path in (Path(__file__), MODULE / "context_transfer_scoring.py"):
        shutil.copyfile(path, source / path.name)
    inputs = {"development": DATA, "full_physical_ridge": RIDGE, **MODELS}
    protocol = {
        "question": "Do saved world forecasts exceed full physical ridge after independent prediction/truth query-centroid removal?",
        "scope": "descriptive adaptive source-development diagnostic; no fit or change to frozen advancement decisions",
        "primary": "average constructs into one profile per intervention gene; center prediction and truth separately per query within each source; compute profile Pearson per gene and macro average",
        "secondary": "same collapsed gene profiles with common source-fitting centroid and ordinary correlation; equal-gene MSE",
        "inputs": {name: {"path": str(path), "sha256": sha(path)} for name, path in inputs.items()},
        "sourceHashes": {path.name: sha(path) for path in source.iterdir()},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    with np.load(DATA, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    train, validation = data["split_train"], data["split_validation"]
    if len(data["split_test"]) or set(data["action_ids"][train]) & set(data["action_ids"][validation]):
        raise ValueError("requires intervention-disjoint development data")
    context = data["context_index"]
    means = {}
    for name, path in MODELS.items():
        with np.load(path, allow_pickle=False) as archive:
            if not np.array_equal(archive["record_ids"], data["record_ids"][validation]):
                raise ValueError(f"{name} forecast record order mismatch")
            if not np.array_equal(archive["context_index"], context[validation]):
                raise ValueError(f"{name} forecast context order mismatch")
            means[name] = archive["mean"]
    results = {}
    with np.load(RIDGE, allow_pickle=False) as ridge:
        for index, name in enumerate(data["context_ids"]):
            positions = np.flatnonzero(context[validation] == index)
            rows = validation[positions]
            reference = data["targets"][train[context[train] == index]].mean(0, dtype=np.float64)
            forecasts = {label: mean[positions] for label, mean in means.items()}
            forecasts["full_physical_ridge"] = ridge[f"context{index}_physical"]
            result = {}
            for label, prediction in forecasts.items():
                profiles = collapse_gene_profiles(prediction, data["targets"][rows],
                    data["observed"][rows], data["action_ids"][rows], data["record_ids"][rows])
                result[label] = score_gene_profiles(profiles, reference)
            results[str(name)] = result
            print(json.dumps({"context": str(name), "results": result}), flush=True)
    report = {"results": results, "protocolSha256": sha(output / "protocol.json")}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
