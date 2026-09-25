import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from slbench.metrics import average_precision, stratified_auc


def test_single_stratum_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = rng.normal(size=500) + y * 0.5
    s[::7] = np.round(s[::7], 1)  # ties
    auc, _ = stratified_auc(np.zeros(500, int), y, s)
    assert np.isclose(auc, roc_auc_score(y, s))


def test_stratified_is_pair_weighted_mean():
    rng = np.random.default_rng(1)
    k = rng.integers(0, 4, 1000)
    y = rng.integers(0, 2, 1000)
    s = rng.normal(size=1000) + y + k  # stratum offsets must not matter
    auc, den = stratified_auc(k, y, s)
    num = sum(roc_auc_score(y[k == j], s[k == j]) * (y[k == j] == 1).sum() * (y[k == j] == 0).sum() for j in range(4))
    assert np.isclose(auc, num / den)


def test_integer_weights_equal_duplication():
    rng = np.random.default_rng(2)
    k = rng.integers(0, 3, 300)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.normal(size=300) + y, 1)
    w = rng.integers(0, 3, 300)
    a, _ = stratified_auc(k, y, s, w.astype(float))
    rep = np.repeat(np.arange(300), w)
    b, _ = stratified_auc(k[rep], y[rep], s[rep])
    assert np.isclose(a, b)


def test_average_precision_perfect():
    assert average_precision(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2])) == 1.0


def test_average_precision_ties_are_order_independent():
    y = np.array([1, 0, 1, 0, 0, 1])
    s = np.array([0.8, 0.8, 0.4, 0.4, 0.4, 0.1])
    expected = average_precision_score(y, s)
    for order in (np.arange(6), np.arange(6)[::-1], np.array([4, 1, 5, 0, 2, 3])):
        assert np.isclose(average_precision(y[order], s[order]), expected)
    assert average_precision(y, np.zeros(6)) == y.mean()
