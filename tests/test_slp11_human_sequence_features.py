from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_slp11_human_sequence_features.py"
SPEC = importlib.util.spec_from_file_location(
    "build_slp11_human_sequence_features", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
HUMAN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HUMAN
SPEC.loader.exec_module(HUMAN)


def test_longest_translation_rule_and_stable_ties() -> None:
    longer = HUMAN.Translation(
        "ENSG00000000001", 2, "ENST00000000009", 1, "ENSP00000000009", 1, b"MAAA"
    )
    shorter = HUMAN.Translation(
        "ENSG00000000001", 2, "ENST00000000001", 1, "ENSP00000000001", 1, b"MAA"
    )
    tie_earlier_transcript = HUMAN.Translation(
        "ENSG00000000001", 2, "ENST00000000001", 9, "ENSP00000000008", 3, b"MCCC"
    )
    tie_later_protein = HUMAN.Translation(
        "ENSG00000000001", 2, "ENST00000000001", 2, "ENSP00000000009", 1, b"MDDD"
    )

    assert longer.selection_key < shorter.selection_key
    assert tie_earlier_transcript.selection_key < longer.selection_key
    assert tie_earlier_transcript.selection_key < tie_later_protein.selection_key


def test_stop_residue_normalization_is_explicit_and_length_preserving() -> None:
    source = b"MAU*XX"
    normalized = HUMAN.normalize_for_esm(source)
    assert normalized == b"MAUXXX"
    assert len(normalized) == len(source)


def test_windows_cover_every_residue_and_remove_overlap_bias() -> None:
    length = 3_500
    windows = HUMAN.chunk_windows(length, max_residues=1_022, overlap=128)
    coverage, weights = HUMAN.inverse_coverage_weights(length, windows)
    assert windows[0][0] == 0
    assert windows[-1][1] == length
    assert max(end - start for start, end in windows) == 1_022
    assert np.all(coverage >= 1)
    reconstructed = np.zeros(length, dtype=np.float64)
    for (start, end), weight in zip(windows, weights, strict=True):
        reconstructed[start:end] += weight
    np.testing.assert_array_equal(reconstructed, np.ones(length))


def test_npz_is_deterministic_compressed_and_pickle_free() -> None:
    arrays = {
        "feature_values": np.zeros((2, 321), dtype=np.float32),
        "entity_taxon": np.asarray([9606, 9606], dtype=np.int64),
        "entity_id": np.asarray(["ENSG00000000001", "ENSG00000000002"], dtype="<U15"),
    }
    arrays["feature_values"][0, -1] = 1.0
    first = HUMAN.deterministic_npz_bytes(arrays)
    second = HUMAN.deterministic_npz_bytes(arrays)
    assert first == second
    with np.load(io.BytesIO(first), allow_pickle=False) as loaded:
        assert loaded.files == ["feature_values", "entity_taxon", "entity_id"]
        assert loaded["feature_values"].dtype == np.float32
        assert loaded["feature_values"].shape == (2, 321)
        assert loaded["entity_taxon"].dtype == np.int64
        assert loaded["entity_id"].dtype.kind == "U"
        assert loaded["feature_values"][:, -1].tolist() == [1.0, 0.0]


def test_entity_list_rejects_versions_symbols_and_wrong_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("ENSG00000000001.1\nTP53\n", encoding="ascii", newline="\n")
    with pytest.raises(HUMAN.HumanSequenceFeatureError):
        HUMAN.load_entity_ids(path)
