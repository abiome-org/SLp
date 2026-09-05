from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_slp11_human_go_features.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("build_slp11_human_go_features", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GO)


def _mapping(*rows: str) -> bytes:
    return gzip.compress((GO.MAPPING_HEADER + "\n" + "\n".join(rows) + "\n").encode())


def _map_row(gene: str, xref: str) -> str:
    return (
        f"{gene}\tENST00000000001\tENSP00000000001\t{xref}\t"
        "Uniprot/SWISSPROT\tDIRECT\t100\t100\t-"
    )


def _gaf_row(
    xref: str,
    term: str,
    *,
    qualifier: str = "enables",
    evidence: str = "IDA",
    aspect: str = "F",
    date: str = "20220901",
) -> str:
    return (
        f"UniProtKB\t{xref}\tGENE\t{qualifier}\t{term}\tPMID:1\t{evidence}\t\t"
        f"{aspect}\tname\t\tprotein\ttaxon:9606\t{date}\tUniProt\t\t"
    )


def _gaf(*rows: str) -> bytes:
    return gzip.compress(("!gaf-version: 2.2\n" + "\n".join(rows) + "\n").encode())


def test_exact_uniprot_to_stable_ensembl_mapping() -> None:
    genes = frozenset({"ENSG00000000001", "ENSG00000000002"})
    mapping, stats = GO.parse_mapping_bytes(
        _mapping(
            _map_row("ENSG00000000001", "P00001"),
            _map_row("ENSG00000000002", "P00001"),
        ),
        genes,
    )
    assert mapping == {"P00001": genes}
    assert stats["mappedUniverseGenes"] == 2


def test_filters_not_dates_bp_and_perturbation_evidence_before_mapping() -> None:
    genes = ("ENSG00000000001",)
    mapping = {"P00001": frozenset(genes)}
    rows = [
        _gaf_row("P00001", "GO:0000001"),
        _gaf_row("P00001", "GO:0000002", qualifier="NOT|enables"),
        _gaf_row("P00001", "GO:0000003", evidence="IMP"),
        _gaf_row("P00001", "GO:0000004", aspect="P"),
        _gaf_row("P00001", "GO:0000005", date="20230101"),
        _gaf_row("P99999", "GO:0000006", aspect="C"),
    ]
    terms, stats = GO.parse_gaf_bytes(_gaf(*rows), mapping, genes)
    assert terms == [frozenset({"GO:0000001"})]
    assert stats["discardedRows"] == {
        "after-date-cutoff": 1,
        "biological-process-aspect": 1,
        "negated-qualifier": 1,
        "no-exact-ensembl-universe-mapping": 1,
        "perturbation-derived-evidence": 1,
    }


def test_direct_matrix_keeps_zero_coverage_rows() -> None:
    matrix, terms, coverage = GO.direct_matrix(
        [frozenset({"GO:0000002"}), frozenset(), frozenset({"GO:0000001"})]
    )
    assert terms == ("GO:0000001", "GO:0000002")
    assert matrix.shape == (3, 2)
    assert matrix.getrow(1).nnz == 0
    assert coverage["zeroCoverageEntities"] == 1


def test_svd_is_deterministic_float32_and_cpu_only() -> None:
    matrix = sparse.csr_matrix(
        np.asarray([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]], dtype=np.float32)
    )
    first, _ = GO.fit_svd(matrix, components=2, seed=731)
    second, _ = GO.fit_svd(matrix, components=2, seed=731)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)
