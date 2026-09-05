import hashlib
import importlib.util
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

DIRECTORY = Path(__file__).parents[1] / "modules/slp-1-1-paired-state-v1"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def artifact(tmp_path):
    module = load("paired_inference_fixture_model", DIRECTORY / "paired_model.py")
    config = module.Config(5, 5, 3, hidden=8, state=4, decoder_hidden=6)
    torch.manual_seed(731)
    model = module.PairedStateModel(config).eval()
    (tmp_path / "source").mkdir()
    shutil.copyfile(DIRECTORY / "paired_model.py", tmp_path / "source/paired_model.py")
    protocol = {
        "config": asdict(config),
        "settings": {"feature_clip": 10.0},
        "sources": {
            "modules/slp-1-1-paired-state-v1/paired_model.py": digest(
                DIRECTORY / "paired_model.py"
            )
        },
    }
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
    rng = np.random.default_rng(731)
    reference = {
        "feature_mean": np.arange(5) * 0.1,
        "feature_scale": np.arange(5) + 1.0,
    }
    for name, q, features in (("rna", 7, 5), ("protein", 3, 3)):
        reference[f"{name}_query_ids"] = np.asarray([f"{name}-{i}" for i in range(q)])
        reference[f"{name}_query_features"] = rng.normal(size=(q, features)).astype(
            np.float32
        )
        reference[f"{name}_basal_indices"] = np.arange(q)
        reference[f"{name}_basal_stats"] = np.asarray([0.3, 2.0])
        reference[f"{name}_amplitude"] = np.ones(q, dtype=np.float32)
    np.savez_compressed(tmp_path / "reference.npz", **reference)
    report = {
        "artifacts": {
            name: digest(tmp_path / name)
            for name in ("protocol.json", "reference.npz", "model.safetensors")
        }
    }
    (tmp_path / "report.json").write_text(json.dumps(report))
    runtime = load("paired_inference_fixture_runtime", DIRECTORY / "inference.py")
    controls = {
        "rna": rng.normal(size=(2, 7)).astype(np.float32),
        "protein": rng.normal(size=(2, 3)).astype(np.float32),
    }
    return runtime, tmp_path, controls, rng.normal(size=(2, 5)).astype(np.float32)


def test_reference_runtime_query_subset_and_chunk_invariance(artifact):
    runtime, path, controls, actions = artifact
    predictor = runtime.PairedEndpointPredictor(path)
    whole = predictor.predict(actions, controls)
    subset = {"rna": np.array([6, 1, 3]), "protein": np.array([2, 0])}
    partial = predictor.predict(actions, controls, query_indices=subset, chunk_size=1)
    for name in subset:
        np.testing.assert_allclose(
            partial[name]["mean"], whole[name]["mean"][:, subset[name]], atol=1e-7
        )
    np.testing.assert_array_equal(partial["state"], whole["state"])


def test_runtime_empty_action_matches_new_caller_controls(artifact):
    runtime, path, controls, actions = artifact
    predictor = runtime.PairedEndpointPredictor(path)
    actions[:] = np.nan
    out = predictor.predict(actions, controls, action_mask=np.zeros((2, 1), dtype=bool))
    for name in controls:
        np.testing.assert_array_equal(out[name]["mean"], controls[name])
    shifted = {name: values + 2 for name, values in controls.items()}
    changed = predictor.predict(
        actions, shifted, action_mask=np.zeros((2, 1), dtype=bool)
    )
    assert not np.array_equal(changed["state"], out["state"])
    for name in shifted:
        np.testing.assert_array_equal(changed[name]["mean"], shifted[name])


def test_runtime_detects_reference_corruption(artifact):
    runtime, path, _, _ = artifact
    with (path / "reference.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="reference.npz"):
        runtime.PairedEndpointPredictor(path)
