"""Artifact reload and joint sampling without biological corpus access."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1] / "modules/slp-1-1-world-transition-v1"
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("transition_inference_test", ROOT / "inference.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_checkpoint_reload_and_dynamic_queries(tmp_path):
    torch.manual_seed(731)
    config = MODULE.Config(4, hidden=8, state_dim=4, covariance_rank=2, dropout=0)
    model = MODULE.TransitionWorld(config).eval()
    from dataclasses import asdict
    (tmp_path / "model-config.json").write_text(json.dumps(asdict(config)))
    save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
    np.savez(tmp_path / "reference.npz", feature_mean=np.zeros(4), feature_std=np.ones(4))
    predictor = MODULE.Predictor(tmp_path)
    action = np.ones((2, 4), dtype=np.float32)
    query = np.arange(20, dtype=np.float32).reshape(5, 4) / 20
    reference, scale = np.zeros(5, np.float32), np.ones(5, np.float32)
    result = predictor.predict(action, query, reference, scale)
    with torch.no_grad():
        expected = model(torch.tensor(action), torch.tensor(query), torch.tensor(reference), torch.tensor(scale))
    np.testing.assert_allclose(result["mean"], expected["mean"].numpy(), rtol=1e-5, atol=1e-6)
    subset = predictor.predict(action, query[:2], reference[:2], scale[:2])
    np.testing.assert_allclose(subset["mean"], result["mean"][:, :2], rtol=1e-5, atol=1e-6)
    context_features = np.broadcast_to(query[:3], (2, 3, 4)).copy()
    context_values = np.arange(6, dtype=np.float32).reshape(2, 3)
    context_mask = np.ones((2, 3), dtype=bool)
    batched_reference = np.stack([reference, reference+1])
    batched_scale = np.stack([scale, scale*2])
    contextual = predictor.predict(action, query, batched_reference, batched_scale,
                                   context_features=context_features, context_values=context_values,
                                   context_mask=context_mask)
    with torch.no_grad():
        expected = model(torch.tensor(action), torch.tensor(query),
                         torch.tensor(batched_reference), torch.tensor(batched_scale),
                         context_features=torch.tensor(context_features),
                         context_values=torch.tensor(context_values), context_mask=torch.tensor(context_mask))
    np.testing.assert_allclose(contextual["mean"], expected["mean"].numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(contextual["factor"], expected["factor"].numpy(), rtol=1e-5, atol=1e-6)


def test_measurement_scale_changes_only_diagonal_uncertainty(tmp_path):
    torch.manual_seed(731)
    config = MODULE.Config(4, hidden=8, state_dim=4, covariance_rank=2, dropout=0)
    model = MODULE.TransitionWorld(config).eval()
    from dataclasses import asdict
    (tmp_path / "model-config.json").write_text(json.dumps(asdict(config)))
    save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
    np.savez(tmp_path / "reference.npz", feature_mean=np.zeros(4), feature_std=np.ones(4))
    predictor = MODULE.Predictor(tmp_path)
    action = np.ones((2, 4), dtype=np.float32)
    query = np.arange(20, dtype=np.float32).reshape(5, 4) / 20
    reference, scale = np.zeros(5, np.float32), np.ones(5, np.float32)

    low = predictor.predict(
        action, query, reference, scale, measurement_scale=np.full(5, 0.2)
    )
    high_scale = np.stack((np.full(5, 0.7), np.full(5, 1.3))).astype(np.float32)
    high = predictor.predict(
        action, query, reference, scale, measurement_scale=high_scale
    )

    for key in ("mean", "state", "factor"):
        np.testing.assert_array_equal(low[key], high[key])
    np.testing.assert_array_equal(high["scale"], high_scale)
    np.testing.assert_allclose(
        high["marginal_scale"],
        np.sqrt(high_scale**2 + np.square(high["factor"]).sum(-1)),
    )


def test_measurement_scale_validation(tmp_path):
    torch.manual_seed(731)
    config = MODULE.Config(4, hidden=8, state_dim=4, dropout=0)
    model = MODULE.TransitionWorld(config).eval()
    from dataclasses import asdict
    (tmp_path / "model-config.json").write_text(json.dumps(asdict(config)))
    save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
    np.savez(tmp_path / "reference.npz", feature_mean=np.zeros(4), feature_std=np.ones(4))
    predictor = MODULE.Predictor(tmp_path)
    action, query = np.ones((2, 4), np.float32), np.ones((3, 4), np.float32)

    for invalid in (np.ones((1, 3)), np.array([1.0, 0.0, 1.0]), np.array([1.0, np.nan, 1.0])):
        with np.testing.assert_raises_regex(ValueError, "measurement_scale"):
            predictor.predict(action, query, np.zeros(3), np.ones(3), measurement_scale=invalid)


def test_explicit_exposure_artifact_scale_loading(tmp_path):
    torch.manual_seed(731)
    config = MODULE.Config(4, hidden=8, state_dim=4, dropout=0)
    model = MODULE.TransitionWorld(config).eval()
    from dataclasses import asdict
    (tmp_path / "model-config.json").write_text(json.dumps(asdict(config)))
    save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
    np.savez(tmp_path / "reference.npz", feature_mean=np.zeros(4), feature_std=np.ones(4))
    biological = np.array([[0.04, 0.09, 0.16], [0.25, 0.36, 0.49]])
    sampling = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.savez(
        tmp_path / "exposure-uncertainty.npz",
        mean_biological_variance=biological,
        mean_sampling_variance=sampling,
        world_biological_variance=biological * 2,
        world_sampling_variance=sampling * 3,
        ridge_biological_variance=biological * 2,
    )
    predictor = MODULE.Predictor(tmp_path)

    scales = predictor.measurement_scales(
        np.array([10.0, 20.0]), np.array([0, 1]), np.array([2, 0])
    )
    expected_variance = np.array(
        [[0.32 + 9.0 / 10.0, 0.08 + 3.0 / 10.0],
         [0.98 + 18.0 / 20.0, 0.50 + 12.0 / 20.0]]
    )
    np.testing.assert_allclose(scales, np.sqrt(expected_variance), rtol=1e-6)


def test_fitted_linear_reference_matches_saved_context_ridge(tmp_path):
    torch.manual_seed(731)
    config = MODULE.Config(4, hidden=8, state_dim=4, dropout=0)
    model = MODULE.TransitionWorld(config).eval()
    from dataclasses import asdict
    (tmp_path / "model-config.json").write_text(json.dumps(asdict(config)))
    save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
    reference = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    reference_scale = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
    np.savez(
        tmp_path / "reference.npz",
        feature_mean=np.zeros(4),
        feature_std=np.ones(4),
        reference=reference,
        reference_scale=reference_scale,
    )
    feature_mean = np.array([[0.0, 1.0, 2.0, 3.0], [2.0, 3.0, 4.0, 5.0]])
    feature_std = np.array([[1.0, 2.0, 4.0, 8.0], [2.0, 4.0, 5.0, 10.0]])
    coefficient = np.arange(24, dtype=np.float64).reshape(2, 4, 3) / 10.0
    intercept = np.array([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]])
    np.savez(
        tmp_path / "linear-reference.npz",
        coefficient=coefficient,
        feature_mean=feature_mean,
        feature_std=feature_std,
        intercept=intercept,
    )
    predictor = MODULE.Predictor(tmp_path)
    actions = np.array([[3.0, 5.0, 9.0, 13.0], [4.0, 5.0, 6.0, 7.0]])
    contexts = np.array([1, 0])
    queries = np.array([2, 0, 2])

    fitted, scale = predictor.fitted_reference(actions, contexts, queries)
    standardized = (actions - feature_mean[contexts]) / feature_std[contexts]
    expected = intercept[contexts[:, None], queries[None, :]] + np.einsum(
        "bf,bfq->bq", standardized, coefficient[contexts][:, :, queries]
    )
    np.testing.assert_allclose(fitted, expected)
    np.testing.assert_array_equal(
        scale, reference_scale[contexts[:, None], queries[None, :]]
    )


def test_fitted_reference_falls_back_to_saved_context_mean(tmp_path):
    torch.manual_seed(731)
    config = MODULE.Config(4, hidden=8, state_dim=4, dropout=0)
    model = MODULE.TransitionWorld(config).eval()
    from dataclasses import asdict
    (tmp_path / "model-config.json").write_text(json.dumps(asdict(config)))
    save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
    reference = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    scale = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
    np.savez(
        tmp_path / "reference.npz", feature_mean=np.zeros(4), feature_std=np.ones(4),
        reference=reference, reference_scale=scale,
    )
    predictor = MODULE.Predictor(tmp_path)

    fitted, selected_scale = predictor.fitted_reference(
        np.ones((2, 4)), np.array([1, 0]), np.array([2, 0])
    )
    np.testing.assert_array_equal(fitted, np.array([[6.0, 4.0], [3.0, 1.0]]))
    np.testing.assert_array_equal(selected_scale, np.array([[0.7, 0.5], [0.4, 0.2]]))


def test_shared_state_sampling_recovers_covariance():
    prediction = {"mean": np.zeros((1, 2)), "scale": np.ones((1, 2)) * 0.2,
                  "factor": np.array([[[1.0], [2.0]]])}
    draws = MODULE.Predictor.sample(prediction, draws=100000, seed=731)[:, 0]
    expected = np.array([[1.04, 2.0], [2.0, 4.04]])
    np.testing.assert_allclose(np.cov(draws.T), expected, rtol=0.02, atol=0.02)
