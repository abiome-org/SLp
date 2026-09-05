from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
MODULE = ROOT / "modules" / "slp-1-1-compositional-state-v1" / "data.py"
DATA = ROOT / "data" / "derived" / "slp11-norman-author-normalized-v2" / "norman-2019-author-normalized-development-v2.npz"
STATIC = ROOT / "data" / "derived" / "slp11-norman-static" / "ensembl116-goa2022-fixed-basis-v1" / "norman-extended-static-esm-go-features.npz"
pytestmark = pytest.mark.skipif(
    not DATA.is_file() or not STATIC.is_file(),
    reason="requires ignored pinned Norman development and static-feature artifacts",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("slp11_compositional_data", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMPOSITION = _load_module()


def test_fitting_only_canonical_contract_and_static_roster() -> None:
    data = COMPOSITION.load_compositional_data(DATA, STATIC)

    assert data.schema == COMPOSITION.SCHEMA
    assert data.y.shape == data.observed.shape == (130, 7226)
    assert data.gene_features.shape == (71, 577)
    assert len(data.single_rows) == 71
    assert len(data.combination_rows) == 59
    assert data.combination_single_rows.shape == (59, 2)
    assert data.combination_common_query_mask.shape == (59, 7226)
    assert np.all(data.combination_common_query_mask.sum(axis=1) == 7182)
    assert all(len(data.canonical_actions[row]) == 1 for row in data.single_rows)
    assert all(len(data.canonical_actions[row]) == 2 for row in data.combination_rows)


def test_fixed_pair_folds_hold_only_combinations_and_keep_both_singles() -> None:
    first = COMPOSITION.load_compositional_data(DATA, STATIC)
    second = COMPOSITION.load_compositional_data(DATA, STATIC)
    np.testing.assert_array_equal(first.combination_fold, second.combination_fold)

    seen: set[int] = set()
    for fold in range(COMPOSITION.PAIR_FOLDS):
        fit, held = first.fold_rows(fold)
        assert set(first.single_rows.tolist()).issubset(fit.tolist())
        assert set(held.tolist()).isdisjoint(fit.tolist())
        assert all(len(first.canonical_actions[row]) == 2 for row in held)
        for row in held:
            pair_position = int(np.flatnonzero(first.combination_rows == row)[0])
            assert set(first.combination_single_rows[pair_position]).issubset(fit.tolist())
        seen.update(held.tolist())
    assert seen == set(first.combination_rows.tolist())


def test_equal_construct_aggregation_and_test_only_path_rejection() -> None:
    data = COMPOSITION.load_compositional_data(DATA, STATIC)
    duplicate = next(i for i, rows in enumerate(data.source_record_indices) if len(rows) > 1)
    with np.load(DATA, allow_pickle=False) as archive:
        rows = np.asarray(data.source_record_indices[duplicate], dtype=np.int64)
        expected = archive["targets"][rows].mean(axis=0)
    np.testing.assert_allclose(data.y[duplicate], expected)

    forbidden = DATA.with_name("norman-2019-author-normalized-test-only-v2.npz")
    try:
        COMPOSITION.load_compositional_data(forbidden, STATIC, verify_pins=False)
    except COMPOSITION.CompositionalDataError as error:
        assert "test-only" in str(error)
    else:
        raise AssertionError("test-only artifact path must be rejected before opening")
