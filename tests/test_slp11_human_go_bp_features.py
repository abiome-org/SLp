from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/build_slp11_human_go_bp_features.py"
SPEC = importlib.util.spec_from_file_location("slp11_human_go_bp_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
BP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BP
SPEC.loader.exec_module(BP)


def gaf_row(xref: str, term: str, evidence: str, qualifier: str = "", aspect: str = "P") -> str:
    return (
        f"UniProtKB\t{xref}\tignored\t{qualifier}\t{term}\tPMID:1\t{evidence}\t\t"
        f"{aspect}\tignored\t\tprotein\ttaxon:9606\t20220919\tUniProt\t\t"
    )


def test_bp_filters_negation_and_perturbation_evidence() -> None:
    text = "\n".join(["!gaf-version: 2.2", gaf_row("P1", "GO:0000001", "IDA"), gaf_row("P1", "GO:0000002", "IMP"), gaf_row("P1", "GO:0000003", "IEA", "NOT"), gaf_row("P1", "GO:0000004", "IEA", aspect="F")]) + "\n"
    terms, stats = BP.parse_bp_gaf(gzip.compress(text.encode()), {"P1": frozenset({"ENSG00000000001"})}, ("ENSG00000000001",))
    assert terms["ENSG00000000001"] == frozenset({"GO:0000001"})
    assert stats["excludedPerturbationEvidenceCounts"] == {"IMP": 1}
    assert stats["discardedRows"]["negated-qualifier"] == 1


def test_vocabulary_is_derived_only_from_fitting_genes() -> None:
    terms = {"ENSG00000000001": frozenset({f"GO:{index:07d}" for index in range(1, 130)}), "ENSG00000000002": frozenset({"GO:9999999"})}
    fit, projected, vocabulary, coverage = BP.matrices_from_fit_terms(terms, ("ENSG00000000001",), ("ENSG00000000001", "ENSG00000000002"))
    assert fit.shape == (1, 129)
    assert projected[1].nnz == 0
    assert "GO:9999999" not in vocabulary
    assert coverage["projectionTermsOutsideFitVocabulary"] == 1


def test_deterministic_npz_is_pickle_free_and_byte_stable() -> None:
    arrays = {"feature_values": np.ones((2, 3), dtype="<f4"), "entity_id": np.asarray(["a", "b"])}
    first = BP.deterministic_npz(arrays)
    second = BP.deterministic_npz(arrays)
    assert first == second


def test_output_projection_equals_sparse_matrix_times_saved_components() -> None:
    matrix = BP.sparse.csr_matrix(np.asarray([[1, 0, 1], [0, 1, 0]], dtype=np.float32))
    components = np.asarray([[0.5, 0.25, -0.5], [1.0, 0.0, 1.0]], dtype=np.float32)
    expected = matrix @ components.T
    assert np.array_equal(expected, np.asarray([[0.0, 2.0], [0.25, 0.0]], dtype=np.float32))
