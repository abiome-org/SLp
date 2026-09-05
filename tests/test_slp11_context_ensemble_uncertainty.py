import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "context_uncertainty", Path(__file__).parents[1] / "scripts/score_slp11_context_ensemble_uncertainty.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_masked_missing_queries_do_not_change_gene_loss():
    prediction = np.array([[2., np.nan], [3., 4.]])
    truth = np.array([[1., np.nan], [1., 2.]])
    observed = np.array([[True, False], [True, True]])
    np.testing.assert_array_equal(module.gene_mse(prediction, truth, observed), [1., 4.])
    with pytest.raises(ValueError):
        module.gene_mse(prediction, truth, np.zeros_like(observed))


def test_paired_ratio_bootstrap_preserves_exact_improvement_and_identity():
    baseline = np.array([1., 4., 9., 16.])
    result = module.paired_interval(.75 * baseline, baseline)
    assert result["mseImprovementPercent"] == 25.
    np.testing.assert_allclose(result["pairedGeneBootstrap95PercentileInterval"], [25., 25.])
    assert module.paired_interval(baseline, baseline)["pairedGeneBootstrap95PercentileInterval"] == [0., 0.]
    with pytest.raises(ValueError):
        module.paired_interval(baseline, baseline[:2])
