"""Scientific numerical contracts for the new transition candidate."""
import importlib.util
import sys
from pathlib import Path

import torch

PATH = Path(__file__).resolve().parents[1] / "modules/slp-1-1-world-transition-v1/transition_model.py"
SPEC = importlib.util.spec_from_file_location("transition_architecture_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def model_inputs(rank=2):
    torch.manual_seed(731)
    model = MODULE.TransitionWorld(MODULE.Config(5, hidden=16, state_dim=8, covariance_rank=rank, dropout=0))
    return model, torch.randn(3, 2, 5), torch.randn(7, 5), torch.randn(7), torch.ones(7)


def test_query_panel_and_action_order_invariance():
    model, action, query, reference, scale = model_inputs()
    full = model(action, query, reference, scale)
    subset = torch.tensor([4, 0, 2])
    part = model(action.flip(1), query[subset], reference[subset], scale[subset])
    torch.testing.assert_close(full["mean"][:, subset], part["mean"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(full["factor"][:, subset], part["factor"], rtol=1e-5, atol=1e-6)


def test_joint_nll_matches_direct_marginal_distribution_with_missing_values():
    model, action, query, reference, scale = model_inputs()
    prediction = model(action, query, reference, scale)
    target = torch.randn(3, 7)
    mask = torch.tensor([[1,1,0,1,0,1,1],[1,0,1,1,1,0,1],[0,1,1,0,1,1,1]], dtype=torch.bool)
    expected = []
    for row in range(3):
        keep = mask[row]
        factor = prediction["factor"][row, keep]
        covariance = torch.diag(prediction["scale"][row, keep].square()) + factor @ factor.T
        distribution = torch.distributions.MultivariateNormal(prediction["mean"][row, keep], covariance_matrix=covariance)
        expected.append(-distribution.log_prob(target[row, keep]) / keep.sum())
    target[~mask] = float("nan")
    actual = MODULE.gaussian_loss(prediction, target, mask, joint=True)
    torch.testing.assert_close(actual, torch.stack(expected).mean())
    actual.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_masked_context_is_missing_not_nan():
    model, action, query, reference, scale = model_inputs()
    features = torch.randn(3, 2, 5)
    values = torch.randn(3, 2)
    mask = torch.tensor([[1,0],[1,0],[1,0]], dtype=torch.bool)
    first = model(action, query, reference, scale, context_features=features, context_values=values, context_mask=mask)
    features[:, 1] = float("nan")
    values[:, 1] = float("nan")
    second = model(action, query, reference, scale, context_features=features, context_values=values, context_mask=mask)
    torch.testing.assert_close(first["mean"], second["mean"])
