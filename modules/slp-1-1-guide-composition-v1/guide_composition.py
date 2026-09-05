"""Pair- and orientation-invariant composition descriptors for dual guides."""
from __future__ import annotations

import itertools

import numpy as np

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of one strict DNA sequence."""
    value = str(sequence).upper()
    if not value or set(value) - set("ACGT"):
        raise ValueError("guide sequences must contain only A/C/G/T")
    return value.translate(_COMPLEMENT)[::-1]


def canonical_3mer_vocabulary() -> tuple[str, ...]:
    """Return the 32 lexicographically sorted reverse-complement classes."""
    kmers = ("".join(item) for item in itertools.product("ACGT", repeat=3))
    return tuple(sorted({min(kmer, reverse_complement(kmer)) for kmer in kmers}))


CANONICAL_3MERS = canonical_3mer_vocabulary()
_KMER_INDEX = {value: index for index, value in enumerate(CANONICAL_3MERS)}


def canonical_3mer_frequencies(sequence: str) -> np.ndarray:
    """Count 18 overlapping 3-mers from a strict 20-nt guide in 32 RC bins."""
    value = str(sequence).upper()
    if len(value) != 20 or set(value) - set("ACGT"):
        raise ValueError("each targeting guide must be exactly 20 A/C/G/T bases")
    result = np.zeros(32, dtype=np.float64)
    for left in range(18):
        kmer = value[left : left + 3]
        result[_KMER_INDEX[min(kmer, reverse_complement(kmer))]] += 1.0 / 18.0
    return result


def _gc_fraction(sequence: str) -> float:
    return (sequence.count("G") + sequence.count("C")) / len(sequence)


def _longest_homopolymer(sequence: str) -> float:
    longest = current = 1
    for left, right in itertools.pairwise(sequence):
        current = current + 1 if left == right else 1
        longest = max(longest, current)
    return float(longest)


def _base_entropy(sequence: str) -> float:
    frequencies = np.asarray([sequence.count(base) / len(sequence) for base in "ACGT"])
    positive = frequencies > 0
    return float(-np.sum(frequencies[positive] * np.log2(frequencies[positive])))


def pair_descriptor(sequence_a: str, sequence_b: str) -> np.ndarray:
    """Return the fixed 71-vector for an unordered, orientation-free guide pair."""
    a, b = str(sequence_a).upper(), str(sequence_b).upper()
    fa, fb = canonical_3mer_frequencies(a), canonical_3mer_frequencies(b)
    gc_a, gc_b = _gc_fraction(a), _gc_fraction(b)
    run_a, run_b = _longest_homopolymer(a), _longest_homopolymer(b)
    entropy_a, entropy_b = _base_entropy(a), _base_entropy(b)
    hamming = sum(left != right for left, right in zip(a, b, strict=True)) / 20.0
    rc_b = reverse_complement(b)
    hamming_rc = sum(left != right for left, right in zip(a, rc_b, strict=True)) / 20.0
    result = np.concatenate(
        (
            (fa + fb) / 2.0,
            np.abs(fa - fb),
            np.asarray(
                [
                    (gc_a + gc_b) / 2.0,
                    abs(gc_a - gc_b),
                    min(run_a, run_b),
                    max(run_a, run_b),
                    (entropy_a + entropy_b) / 2.0,
                    abs(entropy_a - entropy_b),
                    min(hamming, hamming_rc),
                ],
                dtype=np.float64,
            ),
        )
    )
    if result.shape != (71,) or not np.isfinite(result).all():
        raise AssertionError("guide descriptor contract drift")
    return result


def aggregate_gene_descriptors(
    gene_ids: np.ndarray,
    cell_action_ids: np.ndarray,
    cell_pair_ids: np.ndarray,
    library_pair_ids: np.ndarray,
    sequence_a: np.ndarray,
    sequence_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average pair descriptors by exact fitting-cell pair proportions.

    Returns descriptors, fitting-cell counts, and distinct-pair counts aligned
    to the caller's unique gene roster. Every selected cell must join exactly
    to one sequenced library pair; missing joins are rejected, never imputed.
    """
    genes = np.asarray(gene_ids).astype(str)
    actions = np.asarray(cell_action_ids).astype(str)
    pairs = np.asarray(cell_pair_ids).astype(str)
    library = np.asarray(library_pair_ids).astype(str)
    left = np.asarray(sequence_a).astype(str)
    right = np.asarray(sequence_b).astype(str)
    if (
        genes.ndim != 1
        or actions.ndim != 1
        or pairs.shape != actions.shape
        or library.ndim != 1
        or left.shape != library.shape
        or right.shape != library.shape
        or len(set(genes.tolist())) != len(genes)
        or len(set(library.tolist())) != len(library)
    ):
        raise ValueError("unique gene/library rosters and aligned cell arrays required")
    library_lookup = {value: row for row, value in enumerate(library)}
    missing = sorted(set(pairs.tolist()) - set(library_lookup))
    if missing:
        raise ValueError(f"fitting guide pair lacks a sequence join: {missing[0]}")
    gene_lookup = {value: row for row, value in enumerate(genes)}
    missing_genes = sorted(set(actions.tolist()) - set(gene_lookup))
    if missing_genes:
        raise ValueError(f"fitting action is absent from the gene roster: {missing_genes[0]}")
    used_pairs = sorted(set(pairs.tolist()))
    descriptors = {
        value: pair_descriptor(left[library_lookup[value]], right[library_lookup[value]])
        for value in used_pairs
    }
    output = np.zeros((len(genes), 71), dtype=np.float64)
    cell_counts = np.zeros(len(genes), dtype=np.int64)
    pair_counts = np.zeros(len(genes), dtype=np.int64)
    for gene in genes:
        selected = actions == gene
        if not selected.any():
            raise ValueError(f"gene has no fitting guide cells: {gene}")
        unique, counts = np.unique(pairs[selected], return_counts=True)
        weights = counts.astype(np.float64) / counts.sum()
        output[gene_lookup[gene]] = np.sum(
            np.stack([descriptors[value] for value in unique]) * weights[:, None], axis=0
        )
        cell_counts[gene_lookup[gene]] = int(counts.sum())
        pair_counts[gene_lookup[gene]] = len(unique)
    if not np.isfinite(output).all():
        raise AssertionError("aggregated guide descriptors are nonfinite")
    return output, cell_counts, pair_counts
