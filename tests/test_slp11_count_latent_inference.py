from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "modules/slp-1-1-count-latent-state-v1/count_latent_state.py"
INFERENCE_PATH = ROOT / "modules/slp-1-1-count-latent-inference-v1/inference.py"
INFERENCE_V2_PATH = ROOT / "modules/slp-1-1-count-latent-inference-v2/inference.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load(CORE_PATH, "test_count_latent_core")
INFERENCE = load(INFERENCE_PATH, "test_count_latent_inference")
INFERENCE_V2 = load(INFERENCE_V2_PATH, "test_count_latent_inference_v2")


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
    model = CORE.CountLatentState(CORE.Config(**config)).eval()
    root = tmp_path / "artifact"
    (root / "source").mkdir(parents=True)
    shutil.copy2(CORE_PATH, root / "source/count_latent_state.py")
    save_file(model.state_dict(), root / "model.safetensors")
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
    (root / "protocol.json").write_text(
        json.dumps({"modelConfig": config}), encoding="utf-8"
    )
    hashes = {
        name: sha(root / name)
        for name in ("model.safetensors", "reference.npz", "source/count_latent_state.py")
    }
    (root / "artifact-manifest.json").write_text(
        json.dumps({"protocolSha256": sha(root / "protocol.json"), "sha256": hashes}),
        encoding="utf-8",
    )
    return root, reference


def test_empty_action_is_exact_weighted_control_and_exposure_absent(tmp_path):
    root, reference = artifact(tmp_path)
    predictor = INFERENCE.Predictor(root)
    actions = np.zeros((2, 4), dtype=np.float32)
    weights = np.asarray([[1.0, 0.0, 0.0], [0.25, 0.5, 0.25]])
    result = predictor.predict(actions, weights, action_mask=np.zeros((2, 1), bool))
    expected = weights @ reference["basal_rate"]
    np.testing.assert_allclose(result["mean_cp10k"], expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(result["mean_log1p_cp10k"], np.log1p(expected))
    assert "library" not in inspect.signature(predictor.predict).parameters
    assert "num_cells" not in inspect.signature(predictor.predict).parameters


def test_query_chunks_and_permutations_do_not_change_forecast(tmp_path):
    root, _ = artifact(tmp_path)
    predictor = INFERENCE.Predictor(root)
    actions = np.arange(8, dtype=np.float32).reshape(2, 4)
    weights = np.asarray([[1, 2, 3], [3, 2, 1]], dtype=np.float64)
    full = predictor.predict(actions, weights, chunk_size=2)
    order = np.asarray([4, 0, 3, 1, 2])
    changed = predictor.predict(actions, weights, query_indices=order, chunk_size=3)
    np.testing.assert_allclose(changed["mean_cp10k"], full["mean_cp10k"][:, order])
    np.testing.assert_array_equal(changed["query_ids"], full["query_ids"][order])


def test_batched_gem_mixture_matches_explicit_per_row_priors(tmp_path):
    root, reference = artifact(tmp_path)
    predictor = INFERENCE.Predictor(root)
    actions = np.arange(8, dtype=np.float32).reshape(2, 4)
    weights = np.asarray([[0.1, 0.2, 0.7], [0.7, 0.2, 0.1]], dtype=np.float64)
    result = predictor.predict(actions, weights)
    normalized = (actions - reference["feature_mean"]) / reference["feature_scale"]
    query = torch.as_tensor(reference["query_features"])
    basal = torch.as_tensor(reference["basal_rate"])
    context = predictor.model.encode_context(
        query, basal, torch.ones_like(basal, dtype=torch.bool)
    )
    expected = []
    for row in range(2):
        pieces = []
        for group in range(3):
            prior = predictor.model.prior_from_context(
                torch.as_tensor(normalized[row : row + 1, None], dtype=torch.float32),
                torch.ones(1, 1, dtype=torch.bool),
                context[group : group + 1],
            )
            pieces.append(
                predictor.model.population_mean(
                    prior, query, basal[group : group + 1]
                )[0].detach().numpy()
            )
        expected.append(weights[row] @ np.stack(pieces))
    np.testing.assert_allclose(result["mean_cp10k"], expected, rtol=2e-6, atol=2e-6)


def test_reference_and_caller_contracts_reject_ambiguous_inputs(tmp_path):
    root, _ = artifact(tmp_path)
    predictor = INFERENCE.Predictor(root)
    with pytest.raises(ValueError, match="gem_group_weights"):
        predictor.predict(np.ones((1, 4)), np.zeros((1, 3)))
    with pytest.raises(ValueError, match="action_mask"):
        predictor.predict(np.ones((1, 4)), np.ones((1, 3)), action_mask=np.ones((1, 1)))
    with pytest.raises(ValueError, match="query_indices"):
        predictor.predict(np.ones((1, 4)), np.ones((1, 3)), query_indices=[5])


def test_model_checksum_is_enforced(tmp_path):
    root, _ = artifact(tmp_path)
    with (root / "model.safetensors").open("ab") as stream:
        stream.write(b"drift")
    with pytest.raises(ValueError, match="checksum"):
        INFERENCE.Predictor(root)


def test_v2_enforces_protocol_checksum(tmp_path):
    root, _ = artifact(tmp_path)
    predictor = INFERENCE_V2.Predictor(root)
    assert len(predictor.query_ids) == 5
    (root / "protocol.json").write_text('{"modelConfig": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="protocol checksum"):
        INFERENCE_V2.Predictor(root)
