from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_test_build_slp11_k562_count_static_features",
    ROOT / "scripts" / "build_slp11_k562_count_static_features.py",
)
assert SPEC and SPEC.loader
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def test_normalization_includes_zero_rows_and_freezes_near_constant_scale() -> None:
    values = np.asarray(
        [[0.0, 2.0, 0.0], [2.0, 2.0, 4.0], [8.0, 9.0, 8.0]],
        dtype=np.float32,
    )
    normalized, mean, sd, scale = BUILD.normalization(values, np.asarray([0, 1]))
    np.testing.assert_array_equal(mean, [1.0, 2.0, 2.0])
    np.testing.assert_array_equal(sd, [1.0, 0.0, 2.0])
    np.testing.assert_array_equal(scale, [1.0, 1.0, 2.0])
    np.testing.assert_array_equal(normalized[:2].mean(0), 0.0)
    np.testing.assert_array_equal(normalized[:, 1], [0.0, 0.0, 7.0])


def test_duplicate_audit_distinguishes_roles_without_changing_rows() -> None:
    values = np.asarray([[0, 0], [1, 2], [0, 0], [1, 2], [3, 4]], np.float32)
    result = BUILD.duplicate_audit(
        values,
        {
            "query": np.asarray([1, 1, 1, 0, 0], bool),
            "action": np.asarray([0, 1, 0, 1, 1], bool),
        },
    )
    assert result == {
        "distinctFeatureRows": 3,
        "duplicateEquivalenceGroups": 2,
        "rowsBeyondFirstInDuplicateGroups": 2,
        "largestEquivalenceGroup": 2,
        "allZeroRows": 2,
        "allZeroByRole": {"query": 2, "action": 0},
    }


def test_missing_go_projection_uses_only_frozen_term_columns() -> None:
    components = np.zeros((256, 6876), np.float32)
    components[:, 0] = 1.0
    components[:, 1] = 3.0
    terms = np.asarray(
        ["GO:0000001", "GO:0000002"]
        + [f"GO:{index:07d}" for index in range(3, 6877)]
    )
    gaf = (
        "!gaf-version: 2.2\n"
        "UniProtKB\tP1\tx\t\tGO:0000001\tx\tIDA\tx\tF\tx\tx\tprotein\ttaxon:9606\t20220101\tx\t\t\n"
        "UniProtKB\tP1\tx\tNOT\tGO:0000002\tx\tIDA\tx\tF\tx\tx\tprotein\ttaxon:9606\t20220101\tx\t\t\n"
        "UniProtKB\tP1\tx\t\tGO:0000002\tx\tIMP\tx\tF\tx\tx\tprotein\ttaxon:9606\t20220101\tx\t\t\n"
    ).encode()
    import gzip

    projected, direct, audit = BUILD.project_missing_go(
        ["ENSG00000000001"],
        {"P1": frozenset({"ENSG00000000001"})},
        gzip.compress(gaf),
        {"components": components, "term_id": terms},
    )
    np.testing.assert_array_equal(projected["ENSG00000000001"], 1.0)
    assert direct == {"ENSG00000000001"}
    assert audit == {"rowsWithEligibleTerms": 1, "retainedAssociations": 1}


def test_frozen_artifact_preserves_every_source_overlap_bit_exact() -> None:
    output = (
        ROOT
        / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1"
        / BUILD.OUTPUT_NAME
    )
    source = (
        ROOT
        / "data/derived/slp11-human-shared-static/ensembl116-source3-esm8m-shared-go-complete-v2"
        / "human-static-esm8m-shared-go-mf-cc-features.npz"
    )
    if not output.is_file() or not source.is_file():
        pytest.skip("ignored frozen static artifacts are unavailable")
    with np.load(output, allow_pickle=False) as current, np.load(
        source, allow_pickle=False
    ) as previous:
        lookup = {str(entity): i for i, entity in enumerate(previous["entity_id"])}
        present = current["source_static_row_present"]
        source_rows = np.asarray(
            [lookup[str(entity)] for entity in current["entity_id"][present]], np.int64
        )
        np.testing.assert_array_equal(
            current["feature_values"][present], previous["feature_values"][source_rows]
        )
        assert np.all(current["entity_taxon"] == 9606)
        assert list(current["entity_id"]) == sorted(set(current["entity_id"].tolist()))
