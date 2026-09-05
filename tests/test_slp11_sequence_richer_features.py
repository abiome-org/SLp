from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_slp11_sequence_features.py"
SPEC = importlib.util.spec_from_file_location("build_slp11_sequence_features", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
FEATURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FEATURES)


def _entity(index: int, entity_id: str) -> dict[str, object]:
    return {
        "rowIndex": index,
        "ncbiTaxon": FEATURES.SPECIES_TAXON,
        "entityId": entity_id,
    }


def _provenance(
    index: int, entity_id: str, source_ids: list[str], peptide: bytes
) -> dict[str, object]:
    return {
        "rowIndex": index,
        "ncbiTaxon": FEATURES.SPECIES_TAXON,
        "entityId": entity_id,
        "sourceStrainTaxon": FEATURES.SOURCE_STRAIN_TAXON,
        "sourceSequenceIds": source_ids,
        "canonicalPeptideSha256": FEATURES.sha256_bytes(peptide),
        "canonicalPeptideLength": len(peptide),
    }


def test_dipeptides_distinguish_reversal_with_identical_composition() -> None:
    forward = FEATURES.peptide_features(b"MACDE")
    reverse = FEATURES.peptide_features(b"MEDCA")

    assert forward.dtype == np.dtype("<f4")
    np.testing.assert_array_equal(forward[:21], reverse[:21])
    assert not np.array_equal(forward[21:], reverse[21:])
    assert float(forward[21:].sum()) == pytest.approx(1.0)
    assert len(forward) == 421


def test_composite_keys_and_exact_multitarget_consensus() -> None:
    peptide = b"MACD"
    sequences = {
        "SGD:S000000001": peptide + b"*",
        "SGD:S000000002": peptide + b"*",
    }
    entities = [_entity(0, "UniProtKB:P00001")]
    provenance = [
        _provenance(
            0,
            "UniProtKB:P00001",
            ["SGD:S000000001", "SGD:S000000002"],
            peptide,
        )
    ]

    keys, peptides = FEATURES.resolve_entity_peptides(sequences, entities, provenance)
    assert keys == [(4932, "UniProtKB:P00001")]
    assert peptides == [peptide]

    disagreeing = dict(sequences)
    disagreeing["SGD:S000000002"] = b"MACE*"
    with pytest.raises(FEATURES.SequenceFeatureError, match="lacks exact consensus"):
        FEATURES.resolve_entity_peptides(disagreeing, entities, provenance)


def test_rejects_unsorted_or_duplicate_composite_keys() -> None:
    sequences = {
        "SGD:S000000001": b"MACD*",
        "SGD:S000000002": b"MEFG*",
    }
    entities = [
        _entity(0, "SGD:S000000002"),
        _entity(1, "SGD:S000000001"),
    ]
    provenance = [
        _provenance(0, "SGD:S000000002", ["SGD:S000000002"], b"MEFG"),
        _provenance(1, "SGD:S000000001", ["SGD:S000000001"], b"MACD"),
    ]

    with pytest.raises(FEATURES.SequenceFeatureError, match="uniquely sorted"):
        FEATURES.build_feature_arrays(sequences, entities, provenance)


def test_npz_is_deterministic_and_never_requires_pickle() -> None:
    arrays = {
        "feature_values": np.arange(842, dtype=np.float32).reshape(2, 421),
        "entity_taxon": np.asarray([4932, 4932], dtype=np.int64),
        "entity_id": np.asarray(["SGD:S000000001", "UniProtKB:P00001"], dtype="<U16"),
    }

    first = FEATURES.deterministic_npz_bytes(arrays)
    second = FEATURES.deterministic_npz_bytes(arrays)
    assert first == second

    with np.load(io.BytesIO(first), allow_pickle=False) as loaded:
        assert loaded.files == ["feature_values", "entity_taxon", "entity_id"]
        assert loaded["feature_values"].dtype == np.float32
        assert loaded["entity_taxon"].dtype == np.int64
        assert loaded["entity_id"].dtype.kind == "U"
        np.testing.assert_array_equal(loaded["entity_id"], arrays["entity_id"])


def test_esm_windows_cover_full_protein_and_overlap_weights_sum_to_one() -> None:
    length = 2_500
    windows = FEATURES.chunk_windows(length, max_residues=1_022, overlap=128)
    coverage, weights = FEATURES.window_inverse_coverage(length, windows)

    assert windows[0] == (0, 1_022)
    assert windows[-1] == (length - 1_022, length)
    assert max(end - start for start, end in windows) == 1_022
    assert np.all(coverage >= 1)
    reconstructed = np.zeros(length, dtype=np.float64)
    for (start, end), weight in zip(windows, weights, strict=True):
        reconstructed[start:end] += weight
    np.testing.assert_array_equal(reconstructed, np.ones(length))


def test_esm_cache_contract_is_exact_and_safetensors_only(tmp_path: Path) -> None:
    for name in FEATURES.ESM_FILE_SPECS:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "pytorch_model.bin").write_bytes(b"pickle is forbidden")

    with pytest.raises(FEATURES.SequenceFeatureError, match="file set mismatch"):
        FEATURES.verify_esm_model_dir(tmp_path)
    assert "model.safetensors" in FEATURES.ESM_FILE_SPECS
    assert "pytorch_model.bin" not in FEATURES.ESM_FILE_SPECS
