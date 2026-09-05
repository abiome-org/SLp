import gzip
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

PATH = Path(__file__).parents[1] / "scripts/build_slp11_gene_state_source3.py"
SPEC = importlib.util.spec_from_file_location("slp11_gene_state_adapter_test", PATH)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def gene(number):
    return f"ENSG{number:011d}"


def test_adapter_appends_missing_nodes_as_zero_isolates_and_normalizes_rows():
    base_ids = np.array([gene(1), gene(2), gene(3)])
    base = np.stack(
        (
            np.zeros(577),
            np.ones(577),
            np.full(577, 3.0),
        ),
    ).astype(np.float32)
    actions = np.array([gene(1), gene(4)])
    queries = np.array([gene(2), gene(5)])
    arrays, stats = MOD.graph_arrays(
        base_ids,
        base,
        actions,
        queries,
        np.array([gene(1), gene(4)]),
        [(gene(1), gene(2), 0.5), (gene(1), gene(3), 1.0), (gene(4), gene(2), 1.0)],
    )
    np.testing.assert_array_equal(arrays["node_ids"], [gene(i) for i in range(1, 6)])
    np.testing.assert_array_equal(arrays["static_feature_observed"], [True, True, True, False, False])
    np.testing.assert_array_equal(arrays["static_features"][[3, 4]], 0.0)
    np.testing.assert_array_equal(arrays["action_node_index"], [0, 3])
    np.testing.assert_array_equal(arrays["query_node_index"], [1, 4])
    assert stats["featureCoveredFittingGenes"] == 1
    assert stats["directedEdges"] == 4
    assert stats["nodesWithIncomingPhysicalEdges"] == 3

    row0 = slice(arrays["adjacency_indptr"][0], arrays["adjacency_indptr"][1])
    np.testing.assert_array_equal(arrays["adjacency_indices"][row0], [1, 2])
    np.testing.assert_allclose(arrays["adjacency_weights"][row0], [1 / 3, 2 / 3])
    for row in range(3):
        start, stop = arrays["adjacency_indptr"][row:row + 2]
        assert np.sum(arrays["adjacency_weights"][start:stop]) == np.float32(1.0)
    assert arrays["adjacency_indptr"][3] == arrays["adjacency_indptr"][5]


def test_string_parser_uses_exact_ensembl_aliases_threshold_and_pair_max(tmp_path):
    aliases = tmp_path / "9606.protein.aliases.v12.0.txt.gz"
    links = tmp_path / "9606.protein.physical.links.full.v12.0.txt.gz"
    with gzip.open(aliases, "wt", encoding="utf-8", newline="") as stream:
        stream.write("protein\talias\tsource\n")
        stream.write(f"p1\t{gene(1)}\tEnsembl_gene\n")
        stream.write(f"p2\t{gene(2)}\tEnsembl_gene\n")
        stream.write(f"amb\t{gene(3)}\tEnsembl_gene\n")
        stream.write(f"amb\t{gene(4)}\tEnsembl_gene\n")
    with gzip.open(links, "wt", encoding="utf-8", newline="") as stream:
        stream.write("protein1 protein2 experiments\n")
        stream.write("p1 p2 700\n")
        stream.write("p2 p1 900\n")
        stream.write("p1 amb 999\n")
        stream.write("p1 p2 699\n")
    MOD.ALIASES_SHA256 = hashlib.sha256(aliases.read_bytes()).hexdigest()
    MOD.LINKS_SHA256 = hashlib.sha256(links.read_bytes()).hexdigest()
    edges, stats = MOD.load_physical_edges(tmp_path)
    assert edges == [(gene(1), gene(2), 0.9)]
    assert stats == {
        "strongSourceRows": 3,
        "uniqueExactMappedGeneEdges": 1,
        "ambiguousAliasProteins": 1,
    }
