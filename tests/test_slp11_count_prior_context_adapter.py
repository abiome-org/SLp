import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules/slp-1-1-count-prior-context-adapter-v1"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ADAPTER = load(MODULE / "inference.py", "count_prior_context_adapter_test")
CORE = load(MODULE / "count_latent_state.py", "count_prior_context_core_test")
RUNNER = load(
    ROOT / "scripts/prepare_slp11_k562_prior_rpe1_context_forecasts.py",
    "count_prior_context_runner_test",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(tmp_path):
    root = tmp_path / "artifact"
    (root / "source").mkdir(parents=True)
    shutil.copy2(MODULE / "count_latent_state.py", root / "source/count_latent_state.py")
    config = {"feature_dim": 4, "hidden_dim": 8, "state_dim": 3, "key_dim": 5, "dropout": 0.0}
    torch.manual_seed(731)
    model = CORE.CountLatentState(CORE.Config(**config)).eval()
    save_file(model.state_dict(), str(root / "model.safetensors"))
    np.savez(
        root / "reference.npz",
        feature_mean=np.asarray([1, 2, 3, 4], np.float32),
        feature_scale=np.asarray([2, 3, 4, 5], np.float32),
        feature_clip=np.asarray(8, np.float32),
    )
    (root / "protocol.json").write_text(json.dumps({"modelConfig": config}) + "\n")
    hashes = {
        item: sha(root / item)
        for item in ("model.safetensors", "reference.npz", "source/count_latent_state.py")
    }
    (root / "artifact-manifest.json").write_text(
        json.dumps({"protocolSha256": sha(root / "protocol.json"), "sha256": hashes}) + "\n"
    )
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "modelSha256": hashes["model.safetensors"],
                "referenceSha256": hashes["reference.npz"],
                "originalProtocolSha256": sha(root / "protocol.json"),
                "developmentCountMembersOpened": False,
                "testOpened": False,
            }
        )
        + "\n"
    )
    return root, freeze


def inputs():
    rng = np.random.default_rng(731)
    actions = rng.normal(size=(3, 4)).astype(np.float32)
    queries = rng.normal(size=(7, 4)).astype(np.float32)
    ids = np.asarray([f"q{i}" for i in range(7)])
    basal = rng.uniform(0.1, 20, size=(3, 7)).astype(np.float32)
    observed = np.ones((3, 7), np.bool_)
    weights = rng.uniform(size=(3, 3))
    return actions, queries, ids, basal, observed, weights


def test_external_query_order_and_chunk_are_consistent(tmp_path):
    root, freeze = artifact(tmp_path)
    predictor = ADAPTER.ContextPriorPredictor(root, freeze_receipt=freeze)
    actions, queries, ids, basal, observed, weights = inputs()
    full = predictor.predict(actions, queries, ids, basal, observed, weights, chunk_size=7)
    chunk = predictor.predict(actions, queries, ids, basal, observed, weights, chunk_size=2)
    np.testing.assert_array_equal(full["mean_cp10k"], chunk["mean_cp10k"])
    order = np.asarray([4, 1, 6, 0, 2, 5, 3])
    changed = predictor.predict(
        actions, queries[order], ids[order], basal[:, order], observed[:, order], weights, chunk_size=3
    )
    np.testing.assert_allclose(changed["mean_cp10k"], full["mean_cp10k"][:, order], rtol=2e-6, atol=1e-6)
    np.testing.assert_array_equal(changed["query_ids"], ids[order])


def test_empty_action_is_exact_external_control_mixture(tmp_path):
    root, freeze = artifact(tmp_path)
    predictor = ADAPTER.ContextPriorPredictor(root, freeze_receipt=freeze)
    actions, queries, ids, basal, observed, weights = inputs()
    result = predictor.predict(
        actions, queries, ids, basal, observed, weights,
        action_mask=np.zeros((3, 1), np.bool_), chunk_size=2,
    )
    normalized = weights / weights.sum(1, keepdims=True)
    expected = normalized @ basal.astype(np.float64)
    np.testing.assert_array_equal(result["mean_cp10k"], expected)


def test_protocol_tamper_is_rejected(tmp_path):
    root, freeze = artifact(tmp_path)
    (root / "protocol.json").write_text(json.dumps({"modelConfig": {}}) + "\n")
    with pytest.raises(ValueError, match="protocol checksum"):
        ADAPTER.ContextPriorPredictor(root, freeze_receipt=freeze)


def test_metadata_gem_counts_uses_only_fitting_reconstruction_rows(monkeypatch):
    monkeypatch.setattr(RUNNER, "ACTION_COUNT", 2)
    monkeypatch.setattr(RUNNER, "CONTEXTS", 2)
    routing = {
        "action_ids": np.asarray(["g1", "g1", "g1", "g2", "g2", ""]),
        "intervention_role": np.asarray(["train", "train", "validation", "train", "train", "control"]),
        "reconstruction_role": np.asarray(["train", "held", "none", "train", "train", "train"]),
        "unresolved_action": np.zeros(6, np.bool_),
        "is_control": np.asarray([False, False, False, False, False, True]),
        "gem_group": np.asarray([1, 2, 1, 1, 2, 1]),
    }
    # Disable the production total check only for this tiny contract fixture.
    routing["action_ids"] = np.repeat(routing["action_ids"], [142599, 1, 1, 1, 1, 1])
    for key in ("intervention_role", "reconstruction_role", "unresolved_action", "is_control", "gem_group"):
        routing[key] = np.repeat(routing[key], [142599, 1, 1, 1, 1, 1])
    counts = RUNNER.metadata_gem_counts(routing, np.asarray(["g1", "g2"]), np.asarray([1, 2]))
    np.testing.assert_array_equal(counts, [[142599, 0], [1, 1]])


def test_chunk_gate_accepts_only_small_relative_and_log_differences():
    reference = np.asarray([[0.01, 10.0, 100.0]])
    close = reference + np.asarray([[5e-8, 5e-6, 5e-5]])
    assert RUNNER.chunk_consistency(close, reference)["queryChunkWithinTolerance"]
    far = reference.copy()
    far[0, 1] += 2e-5
    assert not RUNNER.chunk_consistency(far, reference)["queryChunkWithinTolerance"]
