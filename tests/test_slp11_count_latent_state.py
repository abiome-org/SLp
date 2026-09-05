from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "count_latent_state", Path(__file__).resolve().parents[1]
    / "modules/slp-1-1-count-latent-state-v1/count_latent_state.py"
)
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)


def example():
    torch.manual_seed(731)
    model = CORE.CountLatentState(CORE.Config(5, hidden_dim=12, state_dim=4, key_dim=6, dropout=0.)).double().eval()
    queries = torch.randn(7, 5, dtype=torch.float64)
    actions = torch.randn(3, 2, 5, dtype=torch.float64)
    mask = torch.tensor([[True, False], [True, True], [False, False]])
    basal = torch.rand(3, 7, dtype=torch.float64) * 30 + 1
    basal_mask = torch.ones_like(basal, dtype=torch.bool)
    prior = model.prior(actions, mask, queries, basal, basal_mask)
    counts = torch.tensor([[0, 1, 2, 3, 0, 1, 4], [1, 0, 2, 1, 0, 0, 2], [2, 0, 0, 0, 0, 0, 0]], dtype=torch.float64)
    return model, queries, actions, mask, basal, basal_mask, prior, counts


def test_negative_binomial_mass_matches_torch_distribution():
    counts = torch.tensor([0., 1., 20., 2000.], dtype=torch.float64)
    log_mean = torch.tensor([-15., 0., 3., 8.], dtype=torch.float64)
    dispersion = torch.tensor([.02, 1., 3., 25.], dtype=torch.float64)
    expected = torch.distributions.NegativeBinomial(
        total_count=dispersion, logits=log_mean - dispersion.log()
    ).log_prob(counts)
    torch.testing.assert_close(CORE.negative_binomial_log_prob(counts, log_mean, dispersion), expected)


def test_empty_action_mean_and_query_chunking():
    model, q, _, _, basal, _, prior, _ = example()
    full = model.population_mean(prior, q, basal)
    torch.testing.assert_close(full[2], basal[2], rtol=0, atol=0)
    chunks = torch.cat([model.population_mean(prior, q[:3], basal[:, :3]),
                        model.population_mean(prior, q[3:], basal[:, 3:])], -1)
    torch.testing.assert_close(full, chunks, rtol=1e-12, atol=1e-12)
    order = torch.tensor([6, 1, 4, 3, 0, 2, 5])
    torch.testing.assert_close(full[:, order], model.population_mean(prior, q[order], basal[:, order]))


def test_prior_invariant_to_action_order_and_masked_values():
    model, q, a, mask, basal, basal_mask, prior, _ = example()
    changed = a.clone()
    changed[~mask] = torch.nan
    other = model.prior(changed.flip(1), mask.flip(1), q, basal, basal_mask)
    for key in prior:
        torch.testing.assert_close(prior[key], other[key])
    # Exposure is deliberately absent from the prior and population-mean APIs.
    assert "library" not in __import__("inspect").signature(model.prior).parameters


def test_population_mean_matches_gaussian_integral():
    model, q, _, _, basal, _, prior, _ = example()
    torch.manual_seed(1731)
    draws = 40000
    state = prior["mean"][0] + torch.randn(draws, 4, dtype=torch.float64) * (.5 * prior["logvar"][0]).exp()
    expanded = {name: values[0:1].expand(draws, -1) for name, values in prior.items()}
    with torch.no_grad():
        empirical = model.log_rate(state, expanded, q, basal[:1].expand(draws, -1))[0].exp().mean(0)
        expected = model.population_mean(prior, q, basal)[0]
    torch.testing.assert_close(empirical, expected, rtol=.015, atol=.02)


def test_shared_context_encoding_matches_repeated_context_and_gradients():
    model, q, actions, mask, basal, basal_mask, _, _ = example()
    rows = torch.tensor([0, 1, 0])
    direct = model.prior(actions, mask, q, basal[rows], basal_mask[rows])
    context = model.encode_context(q, basal[:2], basal_mask[:2])
    shared = model.prior_from_context(actions, mask, context[rows])
    torch.testing.assert_close(direct["mean"], shared["mean"])
    torch.testing.assert_close(direct["logvar"], shared["logvar"])
    parameter = model.context_encoder[0].weight
    left = torch.autograd.grad(direct["mean"].sum(), parameter)[0]
    right = torch.autograd.grad(shared["mean"].sum(), parameter)[0]
    torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)


def test_factored_context_matches_explicit_concatenation():
    model, q, _, _, basal, mask, _, _ = example()
    mask[1, 2] = False
    safe = torch.where(mask, basal, 0.)
    explicit = model.context_encoder(torch.cat((q[None].expand(3, -1, -1), safe.log1p()[..., None]), -1))
    explicit = (explicit * mask[..., None]).sum(1) / mask.sum(1)[:, None]
    factored = model.encode_context(q, basal, mask)
    torch.testing.assert_close(explicit, factored, rtol=1e-12, atol=1e-12)
    parameter = model.context_encoder[0].weight
    left = torch.autograd.grad(explicit.square().sum(), parameter)[0]
    right = torch.autograd.grad(factored.square().sum(), parameter)[0]
    torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)


def test_elbo_masking_and_finite_gradients():
    model, q, _, _, basal, _, prior, counts = example()
    observed = torch.ones_like(counts, dtype=torch.bool)
    observed[:, 2] = False
    library = torch.full((3,), 1000., dtype=torch.float64)
    epsilon = torch.zeros_like(prior["mean"])
    initial = model.elbo(counts, observed, library, q, basal, prior, epsilon=epsilon)
    counts[:, 2] = torch.nan
    changed = model.elbo(counts, observed, library, q, basal, prior, epsilon=epsilon)
    torch.testing.assert_close(initial["loss_per_cell"], changed["loss_per_cell"])
    changed["loss_per_cell"].mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.action_encoder[0].weight.grad.abs().sum() > 0
    assert model.query_dispersion[-1].weight.grad.abs().sum() > 0


def test_gaussian_kl_and_invalid_count_units():
    model, q, _, _, basal, _, prior, counts = example()
    kl = CORE.diagonal_gaussian_kl(prior["mean"], prior["logvar"], prior["mean"], prior["logvar"])
    torch.testing.assert_close(kl, torch.zeros_like(kl), rtol=0, atol=0)
    observed = torch.ones_like(counts, dtype=torch.bool)
    with pytest.raises(ValueError, match="library"):
        model.encode_cells(counts, observed, torch.ones(3), q, basal, prior)
    with pytest.raises(ValueError, match="integers"):
        model.encode_cells(counts + .1, observed, torch.full((3,), 1000.), q, basal, prior)
