import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/build_slp11_human_essential_joint_registry.py"
SPEC = importlib.util.spec_from_file_location("joint_registry", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def roster(ids, roles, fitting=()):
    return {
        "action_ids": np.asarray(ids),
        "action_role": np.asarray(roles),
        "fitting_action_ids": np.asarray(fitting),
        "query_ids": np.asarray([], dtype="U1"),
    }


def static(ids, values):
    n = len(ids)
    return {
        "entity_id": np.asarray(ids),
        "entity_taxon": np.full(n, 9606),
        "feature_values": np.asarray(values, np.float32),
        "source_static_row_present": np.ones(n, bool),
        "esm_present": np.ones(n, bool),
        "go_direct_annotation_present": np.ones(n, bool),
        "go_exact_uniprot_mapping_present": np.ones(n, bool),
        "is_ensembl116_translated_gene": np.ones(n, bool),
    }


def test_role_conflict_fails_closed():
    with pytest.raises(ValueError, match="role conflict"):
        MOD.validate_role_contract({
            "k562": roster(["ENSG1"], ["train"]),
            "rpe1": roster(["ENSG1"], ["validation"]),
        })


def test_context_ids_include_source_context_and_reject_duplicate_gems():
    ids = MOD.make_context_ids("k562", "study-k562-day6", np.asarray([1, 2]))
    assert ids.tolist() == [
        "study-k562-day6::gem-group:001", "study-k562-day6::gem-group:002"
    ]
    with pytest.raises(ValueError, match="invalid GEM"):
        MOD.make_context_ids("rpe1", "study-rpe1-day7", np.asarray([1, 1]))


def test_source_index_preserves_native_query_order():
    source = roster(["ENSG2"], ["train"], ["ENSG2"])
    source["query_ids"] = np.asarray(["ENSG2", "ENSG1"])
    index = MOD.source_index(
        "k562", source, np.asarray(["ENSG1", "ENSG2"]),
        np.asarray(["ENSG2", "ENSG1"]), np.asarray([7]),
    )
    assert index["query_entity_index"].tolist() == [1, 0]
    with pytest.raises(ValueError, match="query order"):
        MOD.source_index(
            "k562", source, np.asarray(["ENSG1", "ENSG2"]),
            np.asarray(["ENSG1", "ENSG2"]), np.asarray([7]),
        )


def test_static_merge_bit_equality_normalizer_and_zero_coverage():
    values_a = np.zeros((2, 577), np.float32)
    values_a[1, 0] = 2.0
    values_b = np.zeros((2, 577), np.float32)
    values_b[0, 0] = 2.0
    values_b[1, 0] = 4.0
    arrays, audit = MOD.merge_static(
        {"k562": static(["ENSG1", "ENSG2"], values_a),
         "rpe1": static(["ENSG2", "ENSG3"], values_b)},
        np.asarray(["ENSG1", "ENSG2", "ENSG3"]), 8.0,
    )
    assert arrays["entity_id"].tolist() == ["ENSG1", "ENSG2", "ENSG3"]
    assert arrays["feature_mean"].dtype == np.float64
    assert arrays["feature_scale"][1] == 1.0
    assert audit["overlapRowsBitExact"] == 1
    assert audit["allZeroRows"] == 1
    broken = values_b.copy()
    broken[0, 0] = 3.0
    with pytest.raises(ValueError, match="overlapping static row differs"):
        MOD.merge_static(
            {"k562": static(["ENSG1", "ENSG2"], values_a),
             "rpe1": static(["ENSG2", "ENSG3"], broken)},
            np.asarray(["ENSG1", "ENSG2", "ENSG3"]), 8.0,
        )
