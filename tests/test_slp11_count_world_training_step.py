import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "modules/slp-1-1-count-world-training-v1"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = load("count_world_training_core_test", "count_latent_state.py")
STEP = load("count_world_training_step_test", "training_step.py")


def fixture(queries=3):
    torch.manual_seed(117)
    model = CORE.CountLatentState(CORE.Config(4, 8, 2, 4, .2)).train()
    features = torch.randn(queries, 4)
    basal = torch.rand(2, queries) + 1
    mask = torch.ones_like(basal, dtype=torch.bool)
    cells = STEP.CellBatch(
        torch.randn(2, 1, 4), torch.ones(2, 1, dtype=torch.bool),
        torch.tensor([1, 0]), torch.ones(2, queries),
        torch.ones(2, queries, dtype=torch.bool), torch.full((2,), float(queries)),
    )
    populations = STEP.PopulationBatch(
        torch.randn(2, 1, 4), torch.ones(2, 1, dtype=torch.bool),
        torch.tensor([[1., 2.], [3., 0.]]), torch.rand(2, queries),
    )
    return model, features, basal, mask, cells, populations


def test_shared_context_graph_matches_separate_loss_and_gradients():
    model, q, b, mask, cells, populations = fixture()
    oracle = copy.deepcopy(model)
    noise = torch.tensor([[.3, -.2], [.1, .4]])
    torch.manual_seed(991)
    actual = STEP.training_losses(model, q, b, mask, cells, populations,
                                  mean_weight=.1, fitting_mean_scale=.4, epsilon=noise)
    actual["loss"].backward()
    torch.manual_seed(991)
    context = oracle.encode_context(q, b, mask)
    prior = oracle.prior_from_context(cells.actions, cells.action_mask, context[cells.context_index])
    count = oracle.elbo(cells.counts, cells.observed, cells.library, q,
                       b[cells.context_index], prior, epsilon=noise)["loss_per_cell"].mean()
    oracle.eval()
    context2 = oracle.encode_context(q, b, mask)
    independent_predictions = []
    for index in range(2):
        prior2 = oracle.prior_from_context(populations.actions[index:index+1].repeat(2, 1, 1),
                                           populations.action_mask[index:index+1].repeat(2, 1), context2)
        rates = oracle.population_mean(prior2, q, b)
        weights = populations.context_weights[index] / populations.context_weights[index].sum()
        independent_predictions.append((rates * weights[:, None]).sum(0).log1p())
    expected = count + .1 * (torch.stack(independent_predictions) - populations.target_log1p_mean).square().mean() / .4
    expected.backward()
    torch.testing.assert_close(actual["loss"], expected)
    for (name, p), (_, other) in zip(model.named_parameters(), oracle.named_parameters()):
        torch.testing.assert_close(p.grad, other.grad, atol=2e-5, rtol=2e-5, msg=name)
    assert model.training


def test_count_only_and_native_panel_sizes_keep_finite_gradients():
    for queries in (2, 5):
        model, q, b, mask, cells, _ = fixture(queries)
        output = STEP.training_losses(model, q, b, mask, cells)
        assert output["population_prediction"] is None
        assert output["mean_mse"] == 0
        output["loss"].backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_failed_population_validation_restores_training_mode():
    model, q, b, mask, cells, populations = fixture()
    model.action_encoder.eval()
    before_modes = [part.training for part in model.modules()]
    invalid = STEP.PopulationBatch(populations.actions, populations.action_mask,
                                   populations.context_weights, torch.full((2, 3), float("nan")))
    with pytest.raises(ValueError, match="finite"):
        STEP.training_losses(model, q, b, mask, cells, invalid,
                             mean_weight=.1, fitting_mean_scale=.4)
    assert model.training
    assert [part.training for part in model.modules()] == before_modes


def test_partial_control_panel_is_rejected():
    model, q, b, mask, cells, _ = fixture()
    mask[0, 0] = False
    with pytest.raises(ValueError, match="fully measured"):
        STEP.training_losses(model, q, b, mask, cells)


def test_unrelated_objective_module_cannot_override_local_copy(monkeypatch):
    monkeypatch.setitem(sys.modules, "molecular_mean_objective", object())
    isolated = load("isolated_count_step_collision_test", "training_step.py")
    assert isolated.normalized_profile_mse(torch.ones(1, 1), torch.zeros(1, 1), 1.) == 1
