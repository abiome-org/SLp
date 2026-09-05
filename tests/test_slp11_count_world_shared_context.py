from __future__ import annotations

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
RUNNER_PATH = ROOT / "scripts/run_slp11_count_world_shared_context.py"
INFERENCE_PATH = ROOT / "modules/slp-1-1-count-world-inference-v1/inference.py"
INFERENCE_V2_PATH = ROOT / "modules/slp-1-1-count-world-inference-v2/inference.py"
CORE_PATH = ROOT / "modules/slp-1-1-count-world-training-v1/count_latent_state.py"
FINALIZER_PATH = ROOT / "scripts/finalize_slp11_count_world_shared_context.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load(RUNNER_PATH, "test_count_world_shared_runner")
INFERENCE = load(INFERENCE_PATH, "test_count_world_shared_inference")
INFERENCE_V2 = load(INFERENCE_V2_PATH, "test_count_world_shared_inference_v2")
CORE = load(CORE_PATH, "test_count_world_shared_core")
FINALIZER = load(FINALIZER_PATH, "test_count_world_shared_finalizer")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def test_alternating_schedule_has_equal_exposure_and_k_only_is_fixed():
    assert [RUNNER.source_for_update("joint-alternating", i) for i in range(1, 7)] == [
        "k562",
        "rpe1",
        "k562",
        "rpe1",
        "k562",
        "rpe1",
    ]
    assert {RUNNER.source_for_update("k562-only", i) for i in range(1, 20)} == {
        "k562"
    }
    with pytest.raises(ValueError, match="one-based"):
        RUNNER.source_for_update("k562-only", 0)
    with pytest.raises(ValueError, match="unknown arm"):
        RUNNER.source_for_update("bad", 1)


def test_bounded_probe_slices_respect_small_and_nonmultiple_limits():
    assert FINALIZER.bounded_slices(100, 4, 16) == (slice(0, 4),)
    assert FINALIZER.bounded_slices(100, 18, 16) == (
        slice(0, 16),
        slice(16, 18),
    )
    assert FINALIZER.bounded_slices(7, None, 3) == (
        slice(0, 3),
        slice(3, 6),
        slice(6, 7),
    )
    assert FINALIZER.bounded_slices(0, 4, 16) == ()
    with pytest.raises(ValueError, match="row bound"):
        FINALIZER.bounded_slices(4, 2, 0)


def test_profile_metric_is_translation_stable_and_constant_is_undefined():
    rng = np.random.default_rng(731)
    truth = rng.normal(size=(9, 13))
    prediction = truth + rng.normal(scale=0.05, size=truth.shape)
    anchor = rng.normal(size=truth.shape)
    first, mse, correlation = RUNNER.stable_profile_metrics(
        prediction, truth, anchor
    )
    offset = np.linspace(-1000, 1000, truth.shape[1])[None]
    second, shifted_mse, shifted_correlation = RUNNER.stable_profile_metrics(
        prediction + offset, truth + offset, anchor + offset
    )
    assert first["definedGenes"] == second["definedGenes"]
    assert first["undefinedGenes"] == second["undefinedGenes"]
    assert first["independentlyQueryCenteredPearson"] == pytest.approx(
        second["independentlyQueryCenteredPearson"], abs=1e-12
    )
    assert first["geneProfileMse"] == pytest.approx(second["geneProfileMse"], abs=1e-12)
    np.testing.assert_allclose(mse, shifted_mse, rtol=0, atol=1e-12)
    np.testing.assert_allclose(correlation, shifted_correlation, rtol=0, atol=1e-12)
    constant = anchor + np.broadcast_to(np.linspace(-1, 1, anchor.shape[1]), anchor.shape)
    summary, _, values = RUNNER.stable_profile_metrics(constant, truth, anchor)
    assert summary["independentlyQueryCenteredPearson"] is None
    assert summary["definedGenes"] == 0
    assert np.isnan(values).all()


def test_normalizer_uses_persisted_float64_statistics_and_rejects_bad_scale():
    raw = np.asarray([[1, 2], [3, 4]], np.float32)
    mean = np.asarray([1.1, 2.2], np.float64)
    scale = np.asarray([0.3, 0.7], np.float64)
    actual = INFERENCE.normalize_actions(raw, mean, scale, 8)
    expected = ((raw.astype(np.float64) - mean) / scale).astype(np.float32)
    np.testing.assert_array_equal(actual[:, 0], expected)
    with pytest.raises(ValueError, match="invalid"):
        INFERENCE.normalize_actions(raw, mean, [1, 0], 8)


def synthetic_artifact(tmp_path: Path):
    artifact = tmp_path / "artifact"
    (artifact / "source").mkdir(parents=True)
    (artifact / "arms").mkdir()
    shutil.copyfile(CORE_PATH, artifact / "source/count_latent_state.py")
    torch.manual_seed(731)
    config = {"feature_dim": 3, "hidden_dim": 8, "state_dim": 2, "key_dim": 4, "dropout": 0.0}
    model = CORE.CountLatentState(CORE.Config(**config)).eval()
    save_file(model.state_dict(), str(artifact / "arms/joint-alternating.safetensors"))
    rng = np.random.default_rng(4)
    query = rng.normal(size=(5, 3)).astype(np.float32)
    basal = rng.uniform(0.1, 5, size=(2, 5)).astype(np.float32)
    np.savez_compressed(
        artifact / "reference-k562.npz",
        query_ids=np.asarray([f"q{i}" for i in range(5)]),
        context_ids=np.asarray(["c0", "c1"]),
        query_features=query,
        basal_rate=basal,
        feature_mean=np.zeros(3, np.float64),
        feature_scale=np.ones(3, np.float64),
        feature_clip=np.asarray(8, np.float32),
    )
    protocol = {"modelConfig": config}
    (artifact / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    names = (
        "source/count_latent_state.py",
        "arms/joint-alternating.safetensors",
        "reference-k562.npz",
    )
    manifest = {
        "protocolSha256": digest(artifact / "protocol.json"),
        "arms": {"joint-alternating": {"modelPath": "arms/joint-alternating.safetensors"}},
        "panels": {"k562": {"referencePath": "reference-k562.npz"}},
        "sha256": {name: digest(artifact / name) for name in names},
    }
    (artifact / "artifact-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return artifact, basal


def test_portable_predictor_empty_identity_query_subset_and_protocol_pin(tmp_path):
    artifact, basal = synthetic_artifact(tmp_path)
    predictor = INFERENCE.Predictor(
        artifact, "joint-alternating", "k562", device="cpu"
    )
    raw = np.ones((2, 3), np.float32)
    weights = np.asarray([[1, 0], [0.25, 0.75]], np.float64)
    empty = predictor.predict(
        raw,
        weights,
        action_mask=np.zeros((2, 1), np.bool_),
        query_indices=np.asarray([4, 1, 0]),
        chunk_size=2,
    )
    expected = (weights @ basal)[:, [4, 1, 0]]
    np.testing.assert_allclose(empty["mean_cp10k"], expected, rtol=2e-6, atol=2e-6)
    assert empty["query_ids"].tolist() == ["q4", "q1", "q0"]
    protocol = artifact / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="protocol checksum"):
        INFERENCE.Predictor(artifact, "joint-alternating", "k562")


def test_v2_predictor_does_not_mutate_float64_context_weights(tmp_path):
    artifact, _ = synthetic_artifact(tmp_path)
    predictor = INFERENCE_V2.Predictor(
        artifact, "joint-alternating", "k562", device="cpu"
    )
    raw = np.ones((2, 3), np.float32)
    weights = np.asarray([[2.0, 1.0], [1.0, 3.0]], np.float64)
    before = weights.copy()
    predictor.predict(raw, weights, query_indices=np.asarray([0]))
    np.testing.assert_array_equal(weights, before)
