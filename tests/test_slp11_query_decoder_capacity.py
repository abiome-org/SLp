import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load(
    ROOT / "modules/slp-1-1-query-decoder-capacity-v1/query_decoder.py",
    "slp11_query_decoder_capacity_test",
)
CORE = load(
    ROOT / "modules/slp-1-1-count-static-ridge-v1/count_static_ridge.py",
    "slp11_query_decoder_ridge_test",
)
AUDIT = load(
    ROOT / "scripts/audit_slp11_count_response_rank.py",
    "slp11_query_decoder_rank_audit_test",
)


def test_rank_factors_match_exact_teacher_prediction():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(71, 40))
    y = rng.normal(size=(71, 53))
    held = rng.normal(size=(13, 40))
    state = CORE.fit_state(x, y)
    factors = MODULE.rank32_factors(state)
    action = MODULE.action_state(CORE, state, held, factors["state_projection"])
    actual = MODULE.reconstruct(state["target_mean"], action, factors["query_loading"].T)
    expected = AUDIT.reduced_rank_prediction(CORE, state, held, 32)
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)


def test_decoder_zero_initialization_and_query_order_contract():
    torch.manual_seed(731)
    model = MODULE.QueryDecoder(feature_dim=7, hidden_dim=11, output_dim=3)
    features = torch.randn(19, 7)
    assert MODULE.parameter_count(model) == MODULE.expected_parameter_count(7, 11, 3)
    torch.testing.assert_close(model(features), torch.zeros(19, 3))
    order = torch.randperm(len(features))
    torch.testing.assert_close(model(features[order]), model(features)[order])


def test_rms_scaled_linear_descriptor_is_learnable_without_query_identity():
    torch.manual_seed(731)
    rng = np.random.default_rng(731)
    features = rng.normal(size=(256, 9)).astype(np.float32)
    target = features @ rng.normal(size=(9, 4)) + rng.normal(size=(1, 4))
    scale = MODULE.rms_scale(target)
    model = MODULE.QueryDecoder(feature_dim=9, hidden_dim=16, output_dim=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    x = torch.from_numpy(features)
    y = torch.from_numpy((target / scale).astype(np.float32))
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean(torch.square(model(x) - y))
        loss.backward()
        optimizer.step()
    assert float(torch.mean(torch.square(model(x) - y)).detach()) < 1e-4
    np.testing.assert_allclose(scale, np.sqrt(np.mean(target * target, axis=0)))
