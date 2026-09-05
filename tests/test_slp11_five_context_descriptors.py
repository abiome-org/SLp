from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "modules/slp-1-1-world-transition-v1"))

import build_nadig_jurkat_controls as jurkat_controls
import build_slp11_five_context_descriptors as five_context
from build_slp11_common_context_descriptors import _copy_snapshot_with_context
from context_descriptor import pooled_control_fixed_panel_log2_cp10k


class _Dataset:
    def __init__(self, values: np.ndarray):
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.keys: list[object] = []

    def __getitem__(self, key: object) -> np.ndarray:
        self.keys.append(key)
        return self.values[key]


class _FakeFile(dict):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _fake_source() -> tuple[_FakeFile, _Dataset]:
    matrix = _Dataset(
        np.asarray([[1, 2], [100, 200], [3, 4], [300, 400]], dtype=np.float32)
    )
    source = _FakeFile(
        {
            "X": matrix,
            "obs/__categories/gene_id": _Dataset(
                np.asarray([b"non-targeting", b"ENSG000001"])
            ),
            "obs/gene_id": _Dataset(np.asarray([0, 1, 0, 1], dtype=np.int64)),
            "obs/UMI_count": _Dataset(np.asarray([10, 20, 10, 20], dtype=np.int64)),
            "obs/gem_group": _Dataset(np.asarray([1, 1, 2, 2], dtype=np.int64)),
            "var/gene_id": _Dataset(np.asarray([b"ENSG000010", b"ENSG000020"])),
        }
    )
    return source, matrix


def test_single_cell_and_equivalent_aggregate_pooling_match() -> None:
    cells = np.asarray(
        [[2, 4, 8], [4, 6, 10], [8, 2, 12], [10, 4, 14]], dtype=np.float64
    )
    panel = np.asarray([True, True, False])
    cell_descriptor, cell_observed = pooled_control_fixed_panel_log2_cp10k(
        cells, np.ones(4, dtype=np.int64), panel
    )
    aggregates = np.stack([cells[:2].mean(axis=0), cells[2:].mean(axis=0)])
    aggregate_descriptor, aggregate_observed = pooled_control_fixed_panel_log2_cp10k(
        aggregates, np.asarray([2, 2]), panel
    )
    np.testing.assert_allclose(aggregate_descriptor, cell_descriptor, rtol=1e-14, atol=0.0)
    np.testing.assert_array_equal(aggregate_observed, cell_observed)
    np.testing.assert_array_equal(cell_observed, panel)
    assert cell_descriptor[2] == 0.0


def test_five_context_reader_indexes_only_non_targeting_rows(monkeypatch) -> None:
    source, matrix = _fake_source()
    monkeypatch.setattr(five_context, "_sha256", lambda path: "pinned")
    monkeypatch.setattr(five_context.h5py, "File", lambda *args, **kwargs: source)
    query_ids = np.asarray(["ENSG000020", "ENSG000010", "ENSG999999"])
    aligned, weights, report = five_context._aligned_nadig_controls(
        ROOT / "fake.h5ad",
        "pinned",
        2,
        (4, 2),
        "test-context",
        query_ids,
    )
    rows, columns = matrix.keys[-1]
    np.testing.assert_array_equal(rows, [0, 2])
    assert columns == slice(None)
    np.testing.assert_array_equal(aligned[:, :2], [[2, 1], [4, 3]])
    np.testing.assert_array_equal(aligned[:, 2], 0.0)
    np.testing.assert_array_equal(weights, [1, 1])
    assert report["perturbedExpressionRowsRead"] == 0


def test_jurkat_normalizer_reader_indexes_only_non_targeting_rows(monkeypatch) -> None:
    source, matrix = _fake_source()
    monkeypatch.setattr(jurkat_controls, "_sha256", lambda path: "pinned")
    monkeypatch.setattr(jurkat_controls, "SOURCE_SHA256", "pinned")
    monkeypatch.setattr(jurkat_controls, "SOURCE_SHAPE", (4, 2))
    monkeypatch.setattr(jurkat_controls, "CONTROL_ROWS", 2)
    monkeypatch.setattr(jurkat_controls.h5py, "File", lambda *args, **kwargs: source)
    raw, depth, groups, queries = jurkat_controls._read_controls(ROOT / "fake.h5ad")
    rows, columns = matrix.keys[-1]
    np.testing.assert_array_equal(rows, [0, 2])
    assert columns == slice(None)
    np.testing.assert_array_equal(raw, [[1, 2], [3, 4]])
    np.testing.assert_array_equal(depth, [10, 10])
    np.testing.assert_array_equal(groups, [1, 2])
    np.testing.assert_array_equal(queries, ["ENSG000010", "ENSG000020"])


def test_snapshot_replaces_only_context_descriptor_payloads(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    destination = tmp_path / "destination.npz"
    np.savez_compressed(
        source,
        targets=np.arange(6, dtype=np.float32).reshape(2, 3),
        observed=np.ones((2, 3), dtype=np.bool_),
        split_train=np.asarray([0], dtype=np.int64),
        control_targets=np.ones((1, 3), dtype=np.float32),
        context_basal_expression=np.ones((1, 3), dtype=np.float32),
        context_value_space=np.asarray("old"),
    )
    _copy_snapshot_with_context(
        source,
        destination,
        np.zeros((1, 3), dtype=np.float32),
        np.asarray([[True, False, True]]),
        value_space="new",
    )
    replaced = {
        "context_basal_expression.npy",
        "context_basal_observed.npy",
        "context_value_space.npy",
    }
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(destination) as new:
        for name in old.namelist():
            if name not in replaced:
                assert hashlib.sha256(old.read(name)).digest() == hashlib.sha256(
                    new.read(name)
                ).digest()
