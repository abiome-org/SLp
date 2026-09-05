from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_slp11_human_shared_static_features",
    ROOT / "scripts" / "build_slp11_human_shared_static_features.py",
)
assert SPEC and SPEC.loader
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def mapped(ids: list[str], values: np.ndarray) -> dict[tuple[int, str], np.ndarray]:
    return {(9606, entity_id): values[index] for index, entity_id in enumerate(ids)}


def test_assemble_preserves_sequence_and_shared_go_coordinates() -> None:
    translated_ids = ["ENSG00000000001", "ENSG00000000002"]
    translated_values = np.zeros((2, 577), dtype=np.float32)
    translated_values[:, :320] = np.arange(640, dtype=np.float32).reshape(2, 320)
    translated_values[:, 320] = 1.0
    source_ids = ["ENSG00000000001", "ENSG00000000003"]
    source_values = np.zeros((2, 321), dtype=np.float32)
    source_values[0] = translated_values[0, :321]
    go_values = np.vstack(
        [np.arange(512, dtype=np.float32).reshape(2, 256), np.zeros(256, dtype=np.float32)]
    )
    arrays, audit = BUILD.assemble(
        mapped(translated_ids, translated_values),
        mapped(source_ids, source_values),
        mapped(translated_ids + [source_ids[1]], go_values),
        {
            (9606, translated_ids[0]): True,
            (9606, translated_ids[1]): False,
            (9606, source_ids[1]): False,
        },
        {source_ids[0], source_ids[1]},
        {source_ids[0]},
        set(),
    )
    assert arrays["feature_values"].shape == (3, 577)
    np.testing.assert_array_equal(arrays["feature_values"][0, :321], source_values[0])
    np.testing.assert_array_equal(arrays["feature_values"][0, 321:], go_values[0])
    np.testing.assert_array_equal(arrays["feature_values"][2], 0.0)
    np.testing.assert_array_equal(arrays["esm_present"], [True, True, False])
    np.testing.assert_array_equal(
        arrays["go_direct_annotation_present"], [True, False, False]
    )
    assert audit["source3ProteinMissingRows"] == 1
    assert audit["sequenceConflicts"] == 0


def test_assemble_rejects_sequence_conflict() -> None:
    translated = np.zeros((1, 577), dtype=np.float32)
    translated[0, 320] = 1.0
    source = translated[:, :321].copy()
    source[0, 0] = 1.0
    go = np.zeros((1, 256), dtype=np.float32)
    entity_id = "ENSG00000000001"
    with pytest.raises(BUILD.HumanStaticError, match="conflict"):
        BUILD.assemble(
            mapped([entity_id], translated),
            mapped([entity_id], source),
            mapped([entity_id], go),
            {(9606, entity_id): False},
            {entity_id},
            set(),
            set(),
        )


def test_deterministic_npz_bytes() -> None:
    arrays = {"x": np.arange(4, dtype=np.float32)}
    assert BUILD.deterministic_npz(arrays) == BUILD.deterministic_npz(arrays)


def test_additional_go_projection_uses_frozen_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entity_id = "ENSG00000000003"
    shared = {
        "feature_values": np.zeros((1, 256), dtype=np.float32),
        "entity_taxon": np.asarray([4932], dtype=np.int64),
        "entity_id": np.asarray(["SGD:S000000001"]),
        "direct_annotation_present": np.asarray([False]),
    }
    components = np.zeros((256, 6876), dtype=np.float32)
    components[:, 0] = 1.0
    components[:, 1] = 2.0
    basis = {
        "components": components,
        "term_id": np.asarray(
            ["GO:0000001", "GO:0000002"]
            + [f"GO:{index:07d}" for index in range(3, 6877)]
        ),
    }
    monkeypatch.setattr(
        BUILD.GO_PARSER,
        "parse_mapping_bytes",
        lambda payload, ids: ({"P1": frozenset(ids)}, {}),
    )
    monkeypatch.setattr(
        BUILD.GO_PARSER,
        "parse_gaf_bytes",
        lambda payload, xrefs, ids: (
            [frozenset({"GO:0000001", "GO:0000002", "GO:9999999"})],
            {},
        ),
    )
    mapping = tmp_path / "mapping.gz"
    gaf = tmp_path / "go.gaf.gz"
    mapping.write_bytes(b"mapping")
    gaf.write_bytes(b"gaf")
    result, audit = BUILD.project_additional_go_rows(
        shared, basis, [entity_id], mapping, gaf
    )
    index = result["entity_id"].tolist().index(entity_id)
    np.testing.assert_array_equal(result["feature_values"][index], 3.0)
    assert result["direct_annotation_present"][index]
    assert audit["eligibleTermsOutsideFrozenSharedVocabulary"] == 1
