from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PATH = (
    Path(__file__).resolve().parents[1]
    / "modules/slp-1-1-guide-composition-v1/guide_composition.py"
)
SPEC = importlib.util.spec_from_file_location("test_guide_composition", PATH)
GUIDE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GUIDE
SPEC.loader.exec_module(GUIDE)


def test_canonical_3mer_counts_are_exact_and_reverse_complement_invariant():
    assert len(GUIDE.CANONICAL_3MERS) == 32
    sequence = "AAACCCGGGTTTACGTACGT"
    observed = GUIDE.canonical_3mer_frequencies(sequence)
    reversed_observed = GUIDE.canonical_3mer_frequencies(
        GUIDE.reverse_complement(sequence)
    )
    assert observed.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(observed, reversed_observed, atol=0, rtol=0)
    aaa = GUIDE.CANONICAL_3MERS.index("AAA")
    assert observed[aaa] == pytest.approx(2 / 18)


def test_pair_descriptor_is_pair_and_individual_orientation_invariant():
    a = "AAACCCGGGTTTACGTACGT"
    b = "GATTACAGATTACAGATTAC"
    expected = GUIDE.pair_descriptor(a, b)
    assert expected.shape == (71,)
    for left, right in (
        (b, a),
        (GUIDE.reverse_complement(a), b),
        (a, GUIDE.reverse_complement(b)),
        (GUIDE.reverse_complement(b), GUIDE.reverse_complement(a)),
    ):
        np.testing.assert_allclose(GUIDE.pair_descriptor(left, right), expected)


def test_exact_descriptor_tail_statistics():
    a = "AAAAAAAAAAAAAAAAAAAA"
    b = "CCCCCCCCCCCCCCCCCCCC"
    tail = GUIDE.pair_descriptor(a, b)[64:]
    np.testing.assert_allclose(
        tail,
        [0.5, 1.0, 20.0, 20.0, 0.0, 0.0, 1.0],
        rtol=0,
        atol=0,
    )


def test_gene_aggregation_uses_exact_cell_pair_proportions():
    genes = np.asarray(["g1", "g2"])
    actions = np.asarray(["g1", "g1", "g1", "g2"])
    pairs = np.asarray(["p1", "p1", "p2", "p2"])
    library = np.asarray(["p1", "p2"])
    sequence_a = np.asarray(["A" * 20, "ACGT" * 5])
    sequence_b = np.asarray(["C" * 20, "TGCA" * 5])
    values, cells, pair_counts = GUIDE.aggregate_gene_descriptors(
        genes, actions, pairs, library, sequence_a, sequence_b
    )
    first = GUIDE.pair_descriptor(sequence_a[0], sequence_b[0])
    second = GUIDE.pair_descriptor(sequence_a[1], sequence_b[1])
    np.testing.assert_allclose(values[0], (2 * first + second) / 3)
    np.testing.assert_allclose(values[1], second)
    np.testing.assert_array_equal(cells, [3, 1])
    np.testing.assert_array_equal(pair_counts, [2, 1])


def test_missing_pairs_genes_and_invalid_sequences_fail_closed():
    with pytest.raises(ValueError, match="exactly 20"):
        GUIDE.pair_descriptor("A" * 19, "C" * 20)
    with pytest.raises(ValueError, match="sequence join"):
        GUIDE.aggregate_gene_descriptors(
            np.asarray(["g1"]),
            np.asarray(["g1"]),
            np.asarray(["missing"]),
            np.asarray(["p1"]),
            np.asarray(["A" * 20]),
            np.asarray(["C" * 20]),
        )
    with pytest.raises(ValueError, match="gene roster"):
        GUIDE.aggregate_gene_descriptors(
            np.asarray(["g1"]),
            np.asarray(["g2"]),
            np.asarray(["p1"]),
            np.asarray(["p1"]),
            np.asarray(["A" * 20]),
            np.asarray(["C" * 20]),
        )
