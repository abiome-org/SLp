from __future__ import annotations

import gzip
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_slp11_go_features.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("build_slp11_go_features", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GO)


def _row(
    stable_id: str,
    term: str,
    *,
    qualifier: str = "enables",
    evidence: str = "IDA",
    aspect: str = "F",
    date: str = "20220801",
) -> str:
    return "\t".join(
        [
            "SGD", stable_id, "GENE", qualifier, term, "PMID:1", evidence, "",
            aspect, "name", "", "protein", "taxon:559292", date, "SGD", "", "",
        ]
    )


def _gaf(*rows: str) -> bytes:
    text = "!gaf-version: 2.2\n!date-generated: 2022-09-20T19:04\n"
    return gzip.compress((text + "\n".join(rows) + "\n").encode(), mtime=0)


def _entity(index: int, entity_id: str) -> dict[str, object]:
    return {"rowIndex": index, "ncbiTaxon": 4932, "entityId": entity_id}


def _provenance(index: int, entity_id: str, source_ids: list[str]) -> dict[str, object]:
    return {
        "rowIndex": index,
        "ncbiTaxon": 4932,
        "entityId": entity_id,
        "sourceSequenceIds": source_ids,
    }


def test_parser_keeps_only_direct_mf_cc_before_cutoff_without_perturbation_evidence() -> None:
    rows = [
        _row("S000000001", "GO:0000001", aspect="F", evidence="IDA"),
        _row("S000000001", "GO:0000002", aspect="C", evidence="IEA", date="20221231"),
        _row("S000000001", "GO:0000003", aspect="P"),
        _row("S000000001", "GO:0000004", qualifier="NOT|enables"),
        _row("S000000001", "GO:0000005", date="20230101"),
    ]
    rows.extend(
        _row("S000000002", f"GO:{index:07d}", evidence=evidence)
        for index, evidence in enumerate(sorted(GO.EXCLUDED_EVIDENCE), start=10)
    )

    annotations, statistics = GO.parse_gaf_bytes(_gaf(*rows))

    assert annotations == {
        "SGD:S000000001": frozenset({"GO:0000001", "GO:0000002"})
    }
    assert statistics["selectedAspectCounts"] == {
        "cellular_component": 1,
        "molecular_function": 1,
    }
    assert statistics["discardedRows"] == {
        "after-date-cutoff": 1,
        "biological-process-aspect": 1,
        "negated-qualifier": 1,
        "perturbation-derived-evidence": 6,
    }


def test_parser_requires_stable_sgd_identity_and_exact_gaf_shape() -> None:
    with pytest.raises(GO.GoFeatureError, match="stable SGD"):
        GO.parse_gaf_bytes(_gaf(_row("YAL001C", "GO:0000001")))
    malformed = _row("S000000001", "GO:0000001").rsplit("\t", 1)[0]
    with pytest.raises(GO.GoFeatureError, match="17 columns"):
        GO.parse_gaf_bytes(_gaf(malformed))


def test_exact_source_relations_project_terms_to_sgd_and_uniprot_with_zero_coverage() -> None:
    annotations = {
        "SGD:S000000001": frozenset({"GO:0000001"}),
        "SGD:S000000002": frozenset({"GO:0000002"}),
    }
    entities = [
        _entity(0, "SGD:S000000001"),
        _entity(1, "UniProtKB:P00001"),
        _entity(2, "UniProtKB:P00002"),
    ]
    provenance = [
        _provenance(0, "SGD:S000000001", ["SGD:S000000001"]),
        _provenance(
            1,
            "UniProtKB:P00001",
            ["SGD:S000000001", "SGD:S000000002"],
        ),
        _provenance(2, "UniProtKB:P00002", ["SGD:S000000003"]),
    ]

    matrix, terms, coverage = GO.project_direct_terms(annotations, entities, provenance)

    assert terms == ("GO:0000001", "GO:0000002")
    np.testing.assert_array_equal(
        matrix.toarray(), np.array([[1, 0], [1, 1], [0, 0]], dtype=np.float32)
    )
    assert coverage["entitiesWithDirectTerms"] == 2
    assert coverage["zeroCoverageEntities"] == 1
    assert coverage["coveredSgdSourceIds"] == 2


def test_svd_is_seeded_deterministic_and_preserves_explicit_zero_rows() -> None:
    matrix = GO.sparse.csr_matrix(
        np.array([[1, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 0]], dtype=np.float32)
    )

    first, first_model = GO.fit_truncated_svd(matrix, max_components=2, seed=731)
    second, second_model = GO.fit_truncated_svd(matrix, max_components=2, seed=731)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_model.components_, second_model.components_)
    np.testing.assert_array_equal(first[2], np.zeros(2, dtype=np.float32))
    assert first.shape == (4, 2)


def test_projection_rejects_relation_or_entity_order_drift() -> None:
    annotations = {"SGD:S000000001": frozenset({"GO:0000001"})}
    entities = [_entity(0, "UniProtKB:P00001")]
    provenance = [_provenance(0, "UniProtKB:P00001", ["UniProtKB:P00001"])]
    with pytest.raises(GO.GoFeatureError, match="sourceSequenceIds"):
        GO.project_direct_terms(annotations, entities, provenance)
