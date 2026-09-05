from __future__ import annotations

import gzip
import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/tools/rdata-1.1.0/site-packages"))
MODULE = ROOT / "modules" / "slp-1-1-yeast-seurat-stream-v1" / "seurat_stream.py"
SPEC = importlib.util.spec_from_file_location("slp11_yeast_seurat_stream", MODULE)
assert SPEC is not None and SPEC.loader is not None
stream = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stream
SPEC.loader.exec_module(stream)


def _info(
    kind: int, *, object_: bool = False, attrs: bool = False, tag: bool = False
) -> bytes:
    raw = (
        kind
        | (0x100 if object_ else 0)
        | (0x200 if attrs else 0)
        | (0x400 if tag else 0)
    )
    return struct.pack(">i", raw)


def _char(value: str) -> bytes:
    encoded = value.encode()
    return _info(9) + struct.pack(">i", len(encoded)) + encoded


def _symbol(value: str) -> bytes:
    return _info(1) + _char(value)


def _ref(index: int) -> bytes:
    return struct.pack(">I", (index << 8) | 255)


def _integer(values: list[int]) -> bytes:
    return (
        _info(13)
        + struct.pack(">i", len(values))
        + np.asarray(values, dtype=">i4").tobytes()
    )


def _real(values: list[float]) -> bytes:
    return (
        _info(14)
        + struct.pack(">i", len(values))
        + np.asarray(values, dtype=">f8").tobytes()
    )


def _strings(values: list[str]) -> bytes:
    return (
        _info(16) + struct.pack(">i", len(values)) + b"".join(_char(x) for x in values)
    )


def _vector(values: list[bytes]) -> bytes:
    return _info(19) + struct.pack(">i", len(values)) + b"".join(values)


def _pairlist(items: list[tuple[bytes, bytes]]) -> bytes:
    tail = _info(254)
    for tag, value in reversed(items):
        tail = _info(2, tag=True) + tag + value + tail
    return tail


def _s4(slots: list[tuple[str, bytes]], class_name: str) -> bytes:
    attrs = _pairlist(
        [
            *[(_symbol(name), value) for name, value in slots],
            (_symbol("class"), _strings([class_name])),
        ]
    )
    return _info(25, object_=True, attrs=True) + attrs


def _rdata(root: bytes) -> bytes:
    raw = b"RDX3\nX\n" + struct.pack(">iiii", 3, 262403, 197888, 5) + b"UTF-8" + root
    return gzip.compress(raw, mtime=0)


def test_reference_safe_selected_payload_and_truncated_prefix(tmp_path: Path) -> None:
    # The second pairlist tag refers to the first serialized symbol. The large
    # REAL vector is selected sparsely and is never materialized in full.
    root = _info(2, tag=True) + _symbol("values") + _real(list(range(20)))
    root += _info(2, tag=True) + _ref(1) + _info(254) + _info(254)
    path = tmp_path / "tiny.RData"
    path.write_bytes(_rdata(root))

    result = stream.inspect_rdata(
        path,
        materialize_limit=2,
        selected={("values",): [(2, 4), (18, 20)]},
    )
    assert result.complete
    payload = result.payloads[0]
    assert payload.length == 20 and payload.complete
    assert payload.selected_indices == (2, 3, 18, 19)
    assert payload.selected_values == (2.0, 3.0, 18.0, 19.0)
    second = result.root.value[1]  # first CDR
    assert second.tag.kind is stream.RType.REF
    assert second.tag.value is result.root.tag

    # Cutting in the numeric body must retain only a partial checksum and must
    # never masquerade as a complete object.
    raw = gzip.decompress(path.read_bytes())
    truncated = tmp_path / "prefix.bin"
    truncated.write_bytes(gzip.compress(raw[:90], mtime=0)[:-8])
    partial = stream.inspect_rdata(truncated, allow_truncated=True, materialize_limit=2)
    assert not partial.complete
    assert partial.error is not None
    if partial.payloads:
        assert not partial.payloads[-1].complete


def test_s4_dgc_slots_and_csc_range_plan(tmp_path: Path) -> None:
    dgc = _s4(
        [
            ("i", _integer([0, 2, 1, 2])),
            ("p", _integer([0, 2, 3, 4])),
            ("Dim", _integer([3, 3])),
            (
                "Dimnames",
                _vector([_strings(["g0", "g1", "g2"]), _strings(["c0", "c1", "c2"])]),
            ),
            ("x", _real([4.0, 1.0, 2.0, 3.0])),
        ],
        "dgCMatrix",
    )
    root = _info(2, tag=True) + _symbol("counts") + dgc + _info(254)
    path = tmp_path / "matrix.RData"
    path.write_bytes(_rdata(root))

    result = stream.inspect_rdata(path, materialize_limit=16)
    matrices = stream.find_dgc_matrices(result.root)
    assert matrices == [
        stream.DgcMatrixInventory(
            ("counts",), 3, 3, 4, ("i", "p", "Dim", "Dimnames", "x", "class")
        )
    ]
    assert stream.plan_csc_payload_ranges([0, 2, 3, 4], [(0, 1), (2, 3)]) == (
        (0, 2),
        (3, 4),
    )
    with pytest.raises(stream.SeuratStreamError, match="overlap"):
        stream.plan_csc_payload_ranges([0, 2, 3, 4], [(0, 2), (1, 3)])


def test_raw_rna_path_policy_and_invalid_sparse_metadata() -> None:
    assert stream.is_admissible_rna_counts_path(("seus", "assays", "RNA", "counts"))
    assert not stream.is_admissible_rna_counts_path(("seus", "assays", "SCT", "counts"))
    assert not stream.is_admissible_rna_counts_path(
        ("seus", "assays", "RNA", "scale.data")
    )

    slots = {
        "i": stream.Node(stream.RType.INT, np.array([0], dtype=np.int32)),
        "p": stream.Node(stream.RType.INT, np.array([0, 2], dtype=np.int32)),
        "Dim": stream.Node(stream.RType.INT, np.array([2, 1], dtype=np.int32)),
        "Dimnames": stream.Node(stream.RType.VEC, []),
        "x": stream.Node(stream.RType.REAL, np.array([1.0])),
    }
    with pytest.raises(stream.SeuratStreamError, match="nnz contract"):
        stream.validate_dgc_slots(slots)


def test_large_strings_skip_or_materialize_by_explicit_path(tmp_path: Path) -> None:
    root = _info(2, tag=True) + _symbol("metadata")
    root += _strings(["cell-a", "cell-b", "cell-c"]) + _info(254)
    path = tmp_path / "metadata.RData"
    path.write_bytes(_rdata(root))

    skipped = stream.inspect_rdata(path, materialize_limit=1)
    assert isinstance(skipped.root.value[0].value, stream.Payload)
    assert skipped.root.value[0].value.kind == "STR"

    selected = stream.inspect_rdata(
        path,
        materialize_limit=1,
        selected_string_paths={("metadata",)},
    )
    assert stream.string_values(selected.root.value[0]) == (
        "cell-a",
        "cell-b",
        "cell-c",
    )
