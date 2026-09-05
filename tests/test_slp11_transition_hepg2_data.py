from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

import hepg2_data
from control_normalization import fit_control_normalizer
from hepg2_data import (
    HepG2DataError,
    _route,
    aggregate_normalized_cells,
    build_population_table,
)


def _normalizer():
    controls = np.asarray(
        [
            [1, 1, 0],
            [3, 1, 2],
            [2, 0, 1],
            [4, 2, 1],
        ],
        dtype=np.float64,
    )
    return fit_control_normalizer(
        controls,
        np.asarray([10, 10, 10, 10]),
        np.asarray([1, 1, 2, 2]),
    )


def test_per_query_cell_counts_track_varying_gem_support() -> None:
    model = _normalizer()
    # Query 2 varies in GEM 1 but is constant (unsupported) in GEM 2.
    counts = np.asarray(
        [[5, 1, 4], [3, 4, 2], [6, 3, 1], [4, 2, 1]], dtype=np.float64
    )
    targets, observed, query_cells, record_cells = aggregate_normalized_cells(
        counts,
        np.asarray([10, 10, 10, 10]),
        np.asarray([1, 2, 1, 2]),
        np.asarray([0, 0, 1, 1]),
        model,
        population_count=2,
    )
    assert targets.shape == observed.shape == query_cells.shape == (2, 3)
    np.testing.assert_array_equal(record_cells, [2, 2])
    np.testing.assert_array_equal(query_cells[:, 2], [1, 1])
    assert observed[:, 2].all()
    assert np.all(query_cells <= record_cells[:, None])
    cell_values, cell_support = model.transform(
        counts,
        np.asarray([10, 10, 10, 10]),
        np.asarray([1, 2, 1, 2]),
    )
    for population in [0, 1]:
        rows = np.asarray([0, 0, 1, 1]) == population
        for query in range(3):
            expected = cell_values[rows, query][cell_support[rows, query]].mean()
            np.testing.assert_allclose(targets[population, query], expected)


def test_exact_construct_populations_remain_distinct_and_route_by_gene() -> None:
    table = build_population_table(
        np.asarray(["ENSG000001", "ENSG000001", "ENSG000001", "ENSG000002"]),
        np.asarray(["population-b", "population-a", "population-b", "population-c"]),
        np.asarray(["guide-b", "guide-a", "guide-b", "guide-c"]),
        np.asarray(["P1P2", "P1P2", "P1P2", "P1P2"]),
    )
    assert table.population_ids.tolist() == [
        "population-a",
        "population-b",
        "population-c",
    ]
    assert table.action_ids.tolist()[:2] == ["ENSG000001", "ENSG000001"]
    assert _route(table.action_ids[0]) == _route(table.action_ids[1])
    np.testing.assert_array_equal(np.bincount(table.cell_population_index), [1, 2, 1])


def test_population_identity_conflicts_are_rejected() -> None:
    with pytest.raises(HepG2DataError, match="multiple construct"):
        build_population_table(
            np.asarray(["ENSG000001", "ENSG000001"]),
            np.asarray(["same", "same"]),
            np.asarray(["guide-a", "guide-b"]),
            np.asarray(["P1P2", "P1P2"]),
        )


def test_aggregation_rejects_invalid_membership_without_dropping_cells() -> None:
    model = _normalizer()
    with pytest.raises(HepG2DataError, match="out of range"):
        aggregate_normalized_cells(
            np.asarray([[1, 2, 1]], dtype=np.float64),
            np.asarray([10]),
            np.asarray([1]),
            np.asarray([1]),
            model,
            population_count=1,
        )


def test_cli_defaults_to_metadata_plan_and_requires_explicit_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []

    def fake_plan(*args):
        calls.append("plan")
        return {"status": "planned-not-executed"}

    def forbidden_execute(*args, **kwargs):
        raise AssertionError("perturbed execution must not be the default")

    monkeypatch.setattr(hepg2_data, "plan", fake_plan)
    monkeypatch.setattr(hepg2_data, "execute", forbidden_execute)
    plan_path = tmp_path / "plan.json"
    assert hepg2_data.main(["--plan-output", str(plan_path)]) == 0
    assert calls == ["plan"]
    assert "planned-not-executed" in plan_path.read_text()
