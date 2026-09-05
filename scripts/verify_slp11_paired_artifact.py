"""Reload a paired artifact on CPU using only static actions and controls."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(artifact):
    target = artifact / "portable-verification.json"
    if target.exists():
        raise FileExistsError(target)
    source = artifact / "source/inference.py"
    protocol = json.loads((artifact / "protocol.json").read_text())
    if sha(source) != protocol["sources"]["modules/slp-1-1-paired-state-v1/inference.py"]:
        raise ValueError("runtime source changed")
    spec = importlib.util.spec_from_file_location("paired_artifact_verification", source)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    torch.set_num_threads(2)
    predictor = runtime.PairedEndpointPredictor(artifact, device="cpu")
    features_path = Path(protocol["inputs"]["features"]["path"])
    if sha(features_path) != protocol["inputs"]["features"]["sha256"]:
        raise ValueError("raw static feature pack changed")
    with np.load(artifact / "predictions.npz", allow_pickle=False) as saved, np.load(features_path, allow_pickle=False) as features:
        chosen = np.asarray([np.flatnonzero(saved["context_index"] == c)[0] for c in range(3)])
        contexts = saved["context_index"][chosen]
        lookup = {str(gene): row for row, gene in enumerate(features["entity_id"])}
        actions = features["feature_values"][[lookup[str(gene)] for gene in saved["action_ids"][chosen]]]
        controls = {name: predictor.reference[f"{name}_controls"][contexts] for name in predictor.query_ids}
        actual = predictor.predict(actions, controls, chunk_size=1024)
        small = predictor.predict(actions, controls, chunk_size=137)
        errors = {name: float(np.max(np.abs(actual[name]["mean"] - saved[f"{name}_prediction"][chosen])))
                  for name in controls}
        chunk_errors = {name: float(np.max(np.abs(actual[name]["mean"] - small[name]["mean"]))) for name in controls}
        empty = predictor.predict(actions, controls, action_mask=np.zeros((len(actions), 1), dtype=bool))
        identity = all(np.array_equal(empty[name]["mean"], control) for name, control in controls.items())
        if max(errors.values()) > 2e-6 or max(chunk_errors.values()) > 2e-6 or not identity:
            raise ValueError("artifact inference verification failed")
    result = {"device": "cpu", "checked_validation_profiles": chosen.tolist(),
              "forecast_reload_max_abs_error": errors, "query_chunk_max_abs_error": chunk_errors,
              "empty_intervention_exact": identity, "outcomes_used_as_inference_input": False,
              "input_transformations_loaded_from_artifact": True,
              "source_hash": sha(Path(__file__)), "artifact_protocol_hash": sha(artifact / "protocol.json"),
              "inference_runtime_hash": sha(source)}
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    verify(parser.parse_args().artifact)
