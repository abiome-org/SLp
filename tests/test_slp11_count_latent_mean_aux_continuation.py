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
RUNNER_PATH = ROOT / "scripts/run_slp11_k562_count_latent_mean_aux_continuation.py"
CORE_PATH = ROOT / "modules/slp-1-1-count-latent-state-v1/count_latent_state.py"
INFERENCE_PATH = ROOT / "modules/slp-1-1-count-latent-continuation-inference-v1/inference.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load(RUNNER_PATH, "test_count_latent_mean_aux_runner")
CORE = load(CORE_PATH, "test_count_latent_mean_aux_core")
INFERENCE = load(INFERENCE_PATH, "test_count_latent_continuation_inference")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    torch.manual_seed(731)
    config = {
        "feature_dim": 4,
        "hidden_dim": 8,
        "state_dim": 3,
        "key_dim": 5,
        "dropout": 0.0,
    }
    root = tmp_path / "artifact"
    (root / "source").mkdir(parents=True)
    shutil.copy2(CORE_PATH, root / "source/count_latent_state.py")
    (root / "protocol.json").write_text(
        json.dumps({"modelConfig": config}), encoding="utf-8"
    )
    rng = np.random.default_rng(731)
    reference = {
        "query_ids": np.asarray(["q0", "q1", "q2", "q3", "q4"]),
        "gem_group_ids": np.asarray([11, 17, 23], dtype=np.int16),
        "query_features": rng.normal(size=(5, 4)).astype(np.float32),
        "feature_mean": rng.normal(size=4).astype(np.float32),
        "feature_scale": np.asarray([0.5, 2.0, 1.0, 3.0], dtype=np.float32),
        "feature_clip": np.asarray(8.0, dtype=np.float32),
        "basal_rate": rng.uniform(0.1, 30.0, size=(3, 5)).astype(np.float32),
        "basal_observed": np.ones((3, 5), dtype=np.bool_),
    }
    np.savez(root / "reference.npz", **reference)
    arms = {}
    for offset, arm in enumerate(("count-only", "mean-aux")):
        torch.manual_seed(731 + offset)
        model = CORE.CountLatentState(CORE.Config(**config)).eval()
        relative = f"arms/{arm}/model.safetensors"
        (root / "arms" / arm).mkdir(parents=True)
        save_file(model.state_dict(), root / relative)
        arms[arm] = {"modelPath": relative}
    files = ["reference.npz", "source/count_latent_state.py"] + [
        value["modelPath"] for value in arms.values()
    ]
    (root / "artifact-manifest.json").write_text(
        json.dumps(
            {
                "protocolSha256": sha(root / "protocol.json"),
                "arms": arms,
                "sha256": {name: sha(root / name) for name in files},
            }
        ),
        encoding="utf-8",
    )
    return root, reference


def test_mean_gene_draws_are_unique_deterministic_and_rng_independent():
    first = np.random.default_rng(1732)
    second = np.random.default_rng(1732)
    rows_a = RUNNER.draw_mean_rows(first, 1443)
    rows_b = RUNNER.draw_mean_rows(second, 1443)
    np.testing.assert_array_equal(rows_a, rows_b)
    assert len(rows_a) == len(set(rows_a.tolist())) == 16
    with pytest.raises(ValueError, match="at least 16"):
        RUNNER.draw_mean_rows(first, 15)


def test_anchored_mean_mse_matches_direct_full_fitting_definition():
    cp10k_sum = np.asarray([[4.0, 8.0], [18.0, 3.0]])
    cell_count = np.asarray([2, 3])
    gem_count = np.asarray([[1, 1], [2, 1]])
    basal = np.asarray([[1.0, 2.0], [3.0, 1.0]])
    target_mean = np.asarray([0.2, -0.1])
    weight = gem_count / gem_count.sum(1, keepdims=True)
    expected = np.mean(
        (
            np.log1p(weight @ basal)
            + target_mean
            - np.log1p(cp10k_sum / cell_count[:, None])
        )
        ** 2
    )
    assert RUNNER.anchored_mean_mse(
        cp10k_sum, cell_count, gem_count, basal, target_mean
    ) == pytest.approx(expected)


def test_per_gene_metrics_are_translation_stable_and_constant_is_undefined():
    truth = np.asarray([[1.0, 2.0, 4.0], [3.0, 2.0, 1.0], [2.0, 5.0, 3.0]])
    anchor = np.asarray([[0.5, 0.4, 0.3], [0.2, 0.1, 0.0], [0.4, 0.4, 0.4]])
    prediction = truth + np.asarray([2.0, -3.0, 1.0])[:, None]
    mse, correlation = RUNNER.per_gene_metrics(prediction, truth, anchor)
    shifted_mse, shifted_correlation = RUNNER.per_gene_metrics(
        prediction + 7.0, truth + 7.0, anchor + 7.0
    )
    np.testing.assert_allclose(mse, shifted_mse)
    np.testing.assert_allclose(correlation, shifted_correlation, equal_nan=True)
    _, constant = RUNNER.per_gene_metrics(anchor + 2.0, truth, anchor)
    assert np.isnan(constant).all()


def test_reconstruction_loss_uses_exact_nested_frozen_schema():
    report = {"strata": {"all": {"lossMean": 1.0236627872240598}}}
    assert RUNNER.reconstruction_loss(report) == 1.0236627872240598
    with pytest.raises(ValueError, match="report schema"):
        RUNNER.reconstruction_loss({"all": {"loss": 1.0}})
    with pytest.raises(ValueError, match="positive finite"):
        RUNNER.reconstruction_loss({"strata": {"all": {"lossMean": np.nan}}})


def test_portable_two_arm_inference_and_empty_mean(tmp_path):
    root, reference = artifact(tmp_path)
    actions = np.zeros((2, 4), dtype=np.float32)
    weights = np.asarray([[1.0, 0.0, 0.0], [0.25, 0.5, 0.25]])
    predictions = {}
    for arm in ("count-only", "mean-aux"):
        predictor = INFERENCE.Predictor(root, arm)
        result = predictor.predict(
            actions, weights, action_mask=np.zeros((2, 1), dtype=np.bool_)
        )
        np.testing.assert_allclose(
            result["mean_cp10k"], weights @ reference["basal_rate"], rtol=2e-6, atol=2e-6
        )
        predictions[arm] = predictor.predict(actions, weights)["mean_cp10k"]
    assert not np.array_equal(predictions["count-only"], predictions["mean-aux"])
    with pytest.raises(ValueError, match="unknown continuation arm"):
        INFERENCE.Predictor(root, "missing")


def test_portable_inference_enforces_protocol_and_selected_model_hashes(tmp_path):
    root, _ = artifact(tmp_path)
    INFERENCE.Predictor(root, "mean-aux")
    (root / "protocol.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="protocol checksum"):
        INFERENCE.Predictor(root, "mean-aux")

    root, _ = artifact(tmp_path / "second")
    with (root / "arms/mean-aux/model.safetensors").open("ab") as stream:
        stream.write(b"drift")
    with pytest.raises(ValueError, match="artifact checksum"):
        INFERENCE.Predictor(root, "mean-aux")
