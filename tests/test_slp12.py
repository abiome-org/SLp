"""Numerical contracts for the 1.2 shared model, not biological benchmarks."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'modules/slp-1-2'


def load(name):
    spec = importlib.util.spec_from_file_location('slp12_test_' + name, MODULE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


model_module = load('model')


def batch():
    # No module named `data` or `model` is installed globally by this test.
    b, o, a, q, f = 2, 5, 2, 4, 702
    torch.manual_seed(17)
    z = lambda *s: torch.zeros(s)
    i = lambda *s: torch.zeros(s, dtype=torch.long)
    m = lambda *s: torch.ones(s, dtype=torch.bool)
    return dict(observation_features=torch.randn(b, o, f), observation_values=torch.randn(b, o),
                observation_ids=torch.randint(1, 10, (b, o)), observation_mask=m(b, o), observation_modality=i(b, o),
                action_features=torch.randn(b, a, f), action_ids=torch.randint(1, 10, (b, a)),
                action_mask=m(b, a), action_mechanism=torch.tensor([[0, 1], [2, 0]]),
                action_values=z(b, a, 3), action_known=m(b, a, 3),
                query_features=torch.randn(b, q, f), query_ids=torch.randint(1, 10, (b, q)),
                query_modality=torch.tensor([[0, 1, 2, 0]] * b), query_mask=m(b, q),
                query_anchor=z(b, q), query_anchor_known=m(b, q),
                context=torch.randn(b, 128), context_known=m(b), assay=i(b), taxon=i(b),
                target=torch.randn(b, q))


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(12)
    return model_module.WorldModel(model_module.Config(width=32, heads=4, layers=2, gene_count=10, id_dropout=0.)).eval()


def test_permutation_symmetry_and_query_equivariance(model):
    original = batch()
    expected = model(original)['mean']
    changed = dict(original)
    for prefix, order in [('observation', [4, 2, 1, 0, 3]), ('action', [1, 0]), ('query', [3, 1, 0, 2])]:
        for key, value in original.items():
            if key.startswith(prefix + '_'):
                changed[key] = value[:, order]
    torch.testing.assert_close(model(changed)['mean'], expected[:, [3, 1, 0, 2]], atol=2e-6, rtol=2e-5)


def test_masked_actions_cannot_affect_predictions(model):
    original = batch()
    original['action_mask'][:, 1] = False
    expected = model(original)['mean']
    original['action_features'][:, 1] = float('nan')
    original['action_values'][:, 1] = float('nan')
    original['action_ids'][:, 1] = 9
    torch.testing.assert_close(model(original)['mean'], expected)


def test_mean_prediction_never_reads_targets(model):
    original = batch()
    expected = model(original)['mean']
    original['target'][:] = float('nan')
    torch.testing.assert_close(model(original)['mean'], expected)


def test_masked_query_nan_cannot_contaminate_valid_predictions(model):
    original = batch()
    original['query_mask'][:, -1] = False
    expected = model(original)['mean'][:, :-1]
    original['query_features'][:, -1] = float('nan')
    original['query_anchor'][:, -1] = float('nan')
    torch.testing.assert_close(model(original)['mean'][:, :-1], expected)


def test_fitness_loss_reaches_shared_backbone_and_observation_encoder(model):
    model.train()
    b = batch()
    loss = (model(b)['mean'][:, 2] - b['target'][:, 2]).square().mean()
    loss.backward()
    for parameter in [model.descriptor[0].weight, model.blocks[0].qkv.weight, model.context[0].weight, model.entity.weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_mechanism_and_context_change_fitness(model):
    b = batch()
    expected = model(b)['mean'][:, 2]
    b['action_mechanism'] = (b['action_mechanism'] + 1) % 3
    assert not torch.allclose(model(b)['mean'][:, 2], expected, atol=1e-8, rtol=0)
    b['context'] += 2
    assert not torch.allclose(model(b)['mean'][:, 2], expected, atol=1e-8, rtol=0)


def test_unknown_gene_descriptors_and_generation_roundtrip(model, tmp_path):
    from safetensors.torch import load_file, save_file
    b = batch()
    for key in ('observation_ids', 'action_ids', 'query_ids'):
        b[key].zero_()
    first = model.generate(b, steps=3, seed=111)
    torch.testing.assert_close(first, model.generate(b, steps=3, seed=111))
    assert not torch.allclose(first, model.generate(b, steps=3, seed=112))
    save_file(model.state_dict(), str(tmp_path / 'model.safetensors'))
    restored = model_module.WorldModel(model.config).eval()
    restored.load_state_dict(load_file(str(tmp_path / 'model.safetensors')))
    torch.testing.assert_close(first, restored.generate(b, steps=3, seed=111))


def test_flow_velocity_learns_from_observed_endpoint(model):
    model.train()
    b = batch()
    noise = torch.randn_like(b['target'])
    time = torch.tensor([.25, .75])
    target_residual = b['target'] - b['query_anchor']
    noisy = noise * (1 - time[:, None]) + target_residual * time[:, None]
    loss = (model(b, noisy, time)['velocity'] - (target_residual - noise)).square().mean()
    loss.backward()
    assert model.blocks[-1].qkv.weight.grad.abs().sum() > 0
    assert model.output.weight.grad[2].abs().sum() > 0


def test_config_rejects_invalid_attention_shape():
    with pytest.raises(ValueError):
        model_module.Config(width=31, heads=4)
