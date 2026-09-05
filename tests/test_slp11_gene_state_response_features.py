import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = (
    Path(__file__).parents[1]
    / "modules/slp-1-1-world-transition-v1/gene_state_response_features.py"
)
SPEC = importlib.util.spec_from_file_location("gene_state_response_features_test", PATH)
FEATURES = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FEATURES)


class Arrays(dict):
    @property
    def files(self):
        return list(self)


def graph():
    return Arrays(
        node_ids=np.array(["ENSG00000000001", "ENSG00000000002", "ENSG00000000003"]),
        node_taxon=np.full(3, 9606, dtype=np.int64),
        static_features=np.arange(3 * 577, dtype=np.float32).reshape(3, 577),
        static_feature_observed=np.ones(3, dtype=bool),
        feature_mean=np.zeros(577, dtype=np.float32),
        feature_scale=np.ones(577, dtype=np.float32),
        adjacency_indptr=np.array([0, 1, 1, 1], dtype=np.int64),
        adjacency_indices=np.array([1], dtype=np.int64),
        adjacency_weights=np.array([1.0], dtype=np.float32),
        action_node_index=np.array([0], dtype=np.int64),
        query_node_index=np.array([1], dtype=np.int64),
    )


def reference(query_id="ENSG00000000002"):
    values = np.zeros((1, 40), dtype=np.float32)
    values[0, -32:] = np.arange(32, dtype=np.float32) + 2
    mean = np.zeros(40, dtype=np.float32)
    mean[-32:] = 2
    std = np.ones(40, dtype=np.float32)
    std[-32:] = 2
    return Arrays(
        query_ids=np.array([query_id]),
        query_features=values,
        query_feature_mean=mean,
        query_feature_std=std,
    )


def test_augments_only_matched_query_nodes_and_preserves_static_block():
    source = graph()
    arrays, audit = FEATURES.augment_graph(source, reference())
    assert arrays["node_features"].shape == (3, 610)
    np.testing.assert_array_equal(
        arrays["node_features"][:, :577], source["static_features"]
    )
    np.testing.assert_array_equal(
        arrays["response_query_feature_observed"], [False, True, False]
    )
    np.testing.assert_array_equal(arrays["node_features"][[0, 2], 577:], 0)
    np.testing.assert_allclose(arrays["node_features"][1, 577:609], np.arange(32) / 2)
    assert arrays["node_features"][1, -1] == 1
    assert audit["response_descriptor_nodes"] == 1


def test_unmatched_or_ambiguous_reference_ids_fail_closed():
    with pytest.raises(FEATURES.GeneStateFeatureError, match="absent"):
        FEATURES.augment_graph(graph(), reference("ENSG00000000009"))
    duplicated = reference()
    duplicated["query_ids"] = np.repeat(duplicated["query_ids"], 2)
    duplicated["query_features"] = np.repeat(duplicated["query_features"], 2, axis=0)
    with pytest.raises(FEATURES.GeneStateFeatureError, match="unique"):
        FEATURES.augment_graph(graph(), duplicated)
