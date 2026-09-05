"""Focused contracts for ESM2-t33 extraction and feature construction."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/build_slp11_esm2_t33_features.py"
SPEC = importlib.util.spec_from_file_location("esm2_t33_feature_builder_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Translation:
    def __init__(self, peptide: bytes):
        self.peptide = peptide


class Sequence:
    @staticmethod
    def normalize_for_esm(peptide: bytes) -> bytes:
        return peptide.replace(b"*", b"X")


def test_unique_peptides_preserve_entity_rows_and_missingness() -> None:
    entities = ["ENSG1", "ENSG2", "ENSG3", "ENSG4"]
    translations = {
        "ENSG1": Translation(b"MA"),
        "ENSG2": Translation(b"MA"),
        "ENSG4": Translation(b"M*"),
    }
    unique, rows, present, missing = MODULE.unique_peptide_contract(
        entities, translations, Sequence
    )
    assert unique == [b"MA", b"MX"]
    assert rows == [0, 0, -1, 1]
    assert present == ["ENSG1", "ENSG2", "ENSG4"]
    assert missing == ["ENSG3"]


def test_shards_are_hash_verified_and_never_pickle_backed(tmp_path: Path) -> None:
    peptides = [b"MA", b"MKL"]
    indices = np.asarray([0, 1], dtype=np.int64)
    path, digest = MODULE.shard_paths(tmp_path, 0)
    np.savez_compressed(
        path,
        unique_index=indices,
        peptide_sha256=np.asarray(
            [MODULE.hashlib.sha256(peptide).hexdigest() for peptide in peptides]
        ),
        peptide_length=np.asarray([2, 3], dtype=np.int64),
        feature_values=np.zeros((2, 1280), dtype=np.float32),
    )
    digest.write_text(MODULE.sha256_file(path) + "\n", encoding="ascii")
    values = MODULE.load_verified_shard(tmp_path, 0, indices, peptides)
    assert values.shape == (2, 1280)
    digest.write_text("0" * 64 + "\n", encoding="ascii")
    try:
        MODULE.load_verified_shard(tmp_path, 0, indices, peptides)
    except ValueError as error:
        assert "hash drift" in str(error)
    else:
        raise AssertionError("tampered immutable shard was accepted")


def test_pca_and_primary_feature_contract_is_frozen_in_source() -> None:
    source = PATH.read_text(encoding="utf-8")
    assert 'n_components=320' in source
    assert 'svd_solver="randomized"' in source
    assert 'iterated_power=7' in source
    assert 'random_state=731' in source
    assert 'whiten=False' in source
    assert '"primaryArm": "esm650m_pca320_physical"' in source
    assert '"secondaryCannotRescuePrimaryFailure": True' in source
    assert '"action_ids", "split_train"' in source
