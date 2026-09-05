from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/build_slp11_frangieh_static_features.py"
SPEC = importlib.util.spec_from_file_location("frangieh_static_builder_test", PATH)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILDER
SPEC.loader.exec_module(BUILDER)


def test_fixed_neighbor_features_do_not_depend_on_requested_output_subset() -> None:
    graph_ids = ("ENSG00000000001", "ENSG00000000002", "ENSG00000000003")
    graph = np.arange(3 * 577, dtype=np.float32).reshape(3, 577)
    full_ids = ("ENSG00000000001", "ENSG00000000004")
    full_base = np.stack((np.full(577, 7, np.float32), np.full(577, 9, np.float32)))
    edges = [
        (graph_ids[0], graph_ids[1], 0.8),
        (graph_ids[0], graph_ids[2], 0.7),
        (graph_ids[1], full_ids[1], 0.9),
    ]
    full, _ = BUILDER.fixed_neighborhood_features(
        full_ids, full_base, graph_ids, graph, edges
    )
    subset, _ = BUILDER.fixed_neighborhood_features(
        full_ids[:1], full_base[:1], graph_ids, graph, edges
    )
    assert np.array_equal(full[0], subset[0])


def test_feature_schema_and_missingness_flag_are_deterministic() -> None:
    ids = ("ENSG00000000001", "ENSG00000000002")
    values = np.zeros((2, 1156), dtype=np.float32)
    values[0, 320] = 1
    arrays = BUILDER.output_arrays(ids, values)
    assert tuple(arrays) == ("feature_values", "entity_taxon", "entity_id")
    assert arrays["feature_values"].dtype == np.dtype("<f4")
    assert arrays["entity_taxon"].tolist() == [9606, 9606]
    assert arrays["entity_id"].tolist() == list(ids)
    assert arrays["feature_values"][:, 320].tolist() == [1.0, 0.0]
