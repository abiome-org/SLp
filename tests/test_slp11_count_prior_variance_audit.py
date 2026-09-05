import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/audit_slp11_count_prior_variance.py"
SPEC = importlib.util.spec_from_file_location("count_prior_variance_audit_test", PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_identity_hash_selection_is_order_stable_by_gene():
    genes = np.asarray([f"ENSG{i:011d}" for i in range(180)])
    selected = genes[AUDIT.selected_rows(genes, 12)]
    reverse = genes[::-1]
    selected_reverse = reverse[AUDIT.selected_rows(reverse, 12)]
    np.testing.assert_array_equal(selected, selected_reverse)


def test_zero_variance_difference_makes_full_and_neutral_exactly_equal():
    rng = np.random.default_rng(731)
    delta_mean = rng.normal(size=(5, 3))
    loading = rng.normal(size=(7, 3))
    mean_term, variance_term = AUDIT.log_ratio_terms(
        delta_mean, np.zeros_like(delta_mean), loading
    )
    basal = rng.uniform(0.1, 20, size=mean_term.shape)
    full, neutral = AUDIT.population_from_terms(basal, mean_term, variance_term)
    np.testing.assert_array_equal(variance_term, np.zeros_like(variance_term))
    np.testing.assert_array_equal(full, neutral)


def test_log_ratio_decomposition_matches_scalar_definition():
    delta_mean = np.asarray([[2.0, -1.0], [0.5, 3.0]])
    delta_variance = np.asarray([[0.25, -0.5], [1.0, 0.125]])
    loading = np.asarray([[3.0, 2.0], [-1.0, 4.0], [0.5, -2.0]])
    mean_term, variance_term = AUDIT.log_ratio_terms(
        delta_mean, delta_variance, loading
    )
    expected_mean = np.empty_like(mean_term)
    expected_variance = np.empty_like(variance_term)
    for row in range(len(delta_mean)):
        for query in range(len(loading)):
            expected_mean[row, query] = sum(
                delta_mean[row, latent] * loading[query, latent]
                for latent in range(loading.shape[1])
            )
            expected_variance[row, query] = 0.5 * sum(
                delta_variance[row, latent] * loading[query, latent] ** 2
                for latent in range(loading.shape[1])
            )
    np.testing.assert_allclose(mean_term, expected_mean, rtol=0, atol=1e-15)
    np.testing.assert_allclose(variance_term, expected_variance, rtol=0, atol=1e-15)
