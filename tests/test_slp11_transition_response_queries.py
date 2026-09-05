"""Training-only response geometry and asymmetric action/query features."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"modules/slp-1-1-world-transition-v1"))
from response_queries import fit_query_response_descriptors
from transition_model import Config, TransitionWorld


def test_training_response_geometry_is_deterministic_and_respects_context_reference():
    rng = np.random.default_rng(7)
    context = np.arange(20) % 2
    residual = rng.normal(size=(20, 2)) @ rng.normal(size=(2, 10))
    reference = rng.normal(size=(2, 10))
    targets = residual + reference[context]
    descriptors, info = fit_query_response_descriptors(targets, context, reference, np.ones((2, 10)), rank=2)
    repeat, _ = fit_query_response_descriptors(targets, context, reference, np.ones((2, 10)), rank=2)
    shifted, _ = fit_query_response_descriptors(targets+4, context, reference+4, np.ones((2, 10)), rank=2)
    np.testing.assert_array_equal(descriptors, repeat)
    np.testing.assert_allclose(descriptors, shifted, atol=1e-5)
    assert info["standardized_training_variance_fraction"] > .99999


def test_query_response_descriptors_do_not_require_action_vocabulary():
    model = TransitionWorld(Config(4, hidden=8, state_dim=4, query_feature_dim=7, dropout=0)).eval()
    actions = torch.randn(2, 4)
    queries = torch.randn(5, 7)
    references, scales = torch.zeros(2, 5), torch.ones(2, 5)
    out = model(actions, queries, references, scales)
    subset = model(actions, queries[:2], references[:, :2], scales[:, :2])
    torch.testing.assert_close(out["mean"][:, :2], subset["mean"])
