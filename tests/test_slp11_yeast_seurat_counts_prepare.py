from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_slp11_yeast_seurat_counts.py"
SPEC = importlib.util.spec_from_file_location("slp11_prepare_yeast_seurat", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


def _info(
    kind: int, *, object_: bool = False, attrs: bool = False, tag: bool = False
) -> bytes:
    return struct.pack(
        ">i",
        kind
        | (0x100 if object_ else 0)
        | (0x200 if attrs else 0)
        | (0x400 if tag else 0),
    )


def _char(value: str) -> bytes:
    encoded = value.encode()
    return _info(9) + struct.pack(">i", len(encoded)) + encoded


def _sym(value: str) -> bytes:
    return _info(1) + _char(value)


def _ints(values: list[int]) -> bytes:
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


def _nullable_strings(values: list[str | None]) -> bytes:
    encoded = [
        _info(9) + struct.pack(">i", -1) if value is None else _char(value)
        for value in values
    ]
    return _info(16) + struct.pack(">i", len(values)) + b"".join(encoded)


def _pairs(items: list[tuple[str, bytes]]) -> bytes:
    result = _info(254)
    for name, value in reversed(items):
        result = _info(2, tag=True) + _sym(name) + value + result
    return result


def _vec(values: list[bytes], attrs: list[tuple[str, bytes]] | None = None) -> bytes:
    attrs = [] if attrs is None else attrs
    return (
        _info(19, attrs=bool(attrs))
        + struct.pack(">i", len(values))
        + b"".join(values)
        + (_pairs(attrs) if attrs else b"")
    )


def _s4(slots: list[tuple[str, bytes]], class_name: str) -> bytes:
    return _info(25, object_=True, attrs=True) + _pairs(
        [*slots, ("class", _strings([class_name]))],
    )


def _fixture(path: Path) -> None:
    matrix = _s4(
        [
            ("i", _ints([0, 2, 1, 2])),
            ("p", _ints([0, 2, 3, 4])),
            ("Dim", _ints([3, 3])),
            (
                "Dimnames",
                _vec(
                    [
                        _strings(["URA3", "bc-YAL001C", "GFP"]),
                        _strings(["a", "b", "c"]),
                    ],
                ),
            ),
            ("x", _real([4.0, 1.0, 2.0, 3.0])),
        ],
        "dgCMatrix",
    )
    assay = _s4(
        [
            ("counts", matrix),
            ("data", _real([99.0] * 6)),
            ("scale.data", _real([88.0] * 6)),
        ],
        "Assay",
    )
    assays = _vec([assay], [("names", _strings(["RNA"]))])
    metadata = _vec(
        [
            _strings(["cell-c", "cell-a", "cell-b"]),
            _nullable_strings(["g2", None, "g1"]),
        ],
        [
            ("names", _strings(["barcode", "genotype_source"])),
            ("row.names", _strings(["r0", "r1", "r2"])),
            ("class", _strings(["data.frame"])),
        ],
    )
    seurat = _s4([("assays", assays), ("meta.data", metadata)], "Seurat")
    saved = _pairs([("seus", _vec([seurat], [("names", _strings(["ctx!A"]))]))])
    raw = b"RDX3\nX\n" + struct.pack(">iiii", 3, 262403, 197888, 5) + b"UTF-8" + saved
    path.write_bytes(gzip.compress(raw, mtime=0))


def test_discovers_exact_s4_paths_and_writes_selected_raw_counts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tiny.RData"
    _fixture(source)
    inventory = prepare.ss.inspect_rdata(
        source,
        materialize_limit=2,
        materialize_atomic_names={"p", "Dim"},
    )
    report, matrices, frames = prepare.discover_structure(inventory.root)
    assert report["s4Objects"]
    assert len(matrices) == 1
    matrix = matrices[0]
    assert matrix.semantic_path == ("seus", "ctx!A", "assays", "RNA", "counts")
    assert prepare.ss.is_admissible_rna_counts_path(matrix.semantic_path)
    assert (matrix.rows, matrix.columns, matrix.nnz) == (3, 3, 4)

    by_role: dict[str, str] = {}
    for index in range(1, 1000):
        candidate_id = f"SGD:S{index:09d}"
        by_role.setdefault(prepare.protected_role(candidate_id), candidate_id)
        if len(by_role) == 3:
            break
    selection = prepare.freeze_action_safe_columns(
        np.array(["", "not-mapped", by_role["pretrain"]]),
        np.array([True, False, False]),
    )
    assert selection.columns.tolist() == [0, 2]

    count_report = prepare.write_selected_csc(
        source,
        matrix,
        selection,
        tmp_path / "counts",
    )
    assert count_report["normalizationApplied"] is False
    np.testing.assert_array_equal(np.load(tmp_path / "counts/p.npy"), [0, 2, 3])
    np.testing.assert_array_equal(np.load(tmp_path / "counts/i.npy"), [0, 2, 2])
    np.testing.assert_allclose(np.load(tmp_path / "counts/x.npy"), [4.0, 1.0, 3.0])

    assert len(frames) == 1
    frame = frames[0]
    assert frame.semantic_path == ("seus", "ctx!A", "meta.data")
    meta_report = prepare.write_selected_metadata(
        source,
        frame,
        ("__row_names__", "barcode", "genotype_source"),
        tmp_path / "metadata",
    )
    assert meta_report["biologicalRolesAssigned"] is False
    assert np.load(tmp_path / "metadata/barcode.npy").tolist() == [
        "cell-c",
        "cell-a",
        "cell-b",
    ]
    assert np.load(tmp_path / "metadata/genotype_source.npy").tolist() == [
        "g2",
        "",
        "g1",
    ]
    assert np.load(tmp_path / "metadata/genotype_source_observed.npy").tolist() == [
        True,
        False,
        True,
    ]
    assert np.load(tmp_path / "metadata/__row_names__.npy").tolist() == [
        "r0",
        "r1",
        "r2",
    ]

    direct_inventory = prepare.ss.inspect_rdata(source, materialize_limit=4096)
    _, _, direct_frames = prepare.discover_structure(direct_inventory.root)
    prepare.write_selected_metadata(
        source,
        direct_frames[0],
        ("__row_names__",),
        tmp_path / "metadata-direct",
    )
    assert np.load(tmp_path / "metadata-direct/__row_names__.npy").tolist() == [
        "r0",
        "r1",
        "r2",
    ]


def test_complete_source_verification_and_no_partial_files(tmp_path: Path) -> None:
    source = tmp_path / "tiny.RData"
    _fixture(source)
    content = source.read_bytes()
    verified = prepare.verify_source(
        source,
        expected_bytes=len(content),
        expected_md5=hashlib.md5(content).hexdigest(),
        expected_sha256=hashlib.sha256(content).hexdigest(),
    )
    assert verified["bytes"] == len(content)
    partial = tmp_path / "tiny.RData.partial"
    partial.write_bytes(content)
    with pytest.raises(prepare.ss.SeuratStreamError, match="partial"):
        prepare.verify_source(
            partial,
            expected_bytes=len(content),
            expected_md5=verified["md5"],
            expected_sha256=verified["sha256"],
        )


def test_writer_rejects_normalized_or_duplicate_column_selection(
    tmp_path: Path,
) -> None:
    slots: dict[str, prepare.ss.Node] = {}
    bad = prepare.DgcCandidate(("seus", "assays", "SCT", "counts"), slots, 1, 1, 0)
    selection = prepare.FrozenColumnSelection(np.array([0]), "x", 1, 0, 0, 0)
    with pytest.raises(prepare.ss.SeuratStreamError, match="not an RNA/counts"):
        prepare.write_selected_csc(
            tmp_path / "unused", bad, selection, tmp_path / "out"
        )


def test_global_held_actions_are_excluded_but_queries_remain_unfiltered() -> None:
    by_role: dict[str, str] = {}
    for index in range(1, 1000):
        candidate_id = f"SGD:S{index:09d}"
        by_role.setdefault(prepare.protected_role(candidate_id), candidate_id)
        if len(by_role) == 3:
            break
    selection = prepare.freeze_action_safe_columns(
        np.array(
            [
                by_role["pretrain"],
                by_role["molecular-validation"],
                by_role["molecular-final"],
                "unmapped",
                "",
            ],
        ),
        np.array([False, False, False, False, True]),
    )
    assert selection.columns.tolist() == [0, 4]
    assert selection.excluded_held_count == 2
    assert selection.excluded_unmapped_count == 1


def test_raw_row_audit_flags_artificial_candidates_but_preserves_native_ura3(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tiny.RData"
    _fixture(source)
    inventory = prepare.ss.inspect_rdata(
        source,
        materialize_limit=2,
        materialize_atomic_names={"p", "Dim"},
        materialize_string_suffixes={
            ("Dimnames", "[0]"),
        },
    )
    _, matrices, _ = prepare.discover_structure(inventory.root)
    audit = prepare.audit_raw_rna_row_identifiers(matrices[0])
    assert audit["allIdentifiers"] == ["URA3", "bc-YAL001C", "GFP"]
    assert audit["nativeUra3ExactRows"] == ["URA3"]
    assert audit["lexicalFlags"]["barcodePrefix"] == ["bc-YAL001C"]
    assert audit["lexicalFlags"]["fluorescentReporter"] == ["GFP"]
    assert "no row removed" in audit["decision"]


def test_metadata_selection_excludes_global_held_and_development_test(
    tmp_path: Path,
) -> None:
    stable_by_rule: dict[str, str] = {}
    for index in range(1, 10000):
        stable = f"SGD:S{index:09d}"
        protected = prepare.protected_role(stable)
        development = prepare.development_role(stable)
        key = (
            "held"
            if protected != "pretrain"
            else "dev-test"
            if development == "test"
            else "fit"
            if development == "train"
            else "validation"
        )
        stable_by_rule.setdefault(key, stable)
        if {"held", "dev-test", "fit"}.issubset(stable_by_rule):
            break
    metadata = tmp_path / "metadata/frame-0"
    metadata.mkdir(parents=True)
    values = {
        "__row_names__": ["c0", "c1", "c2", "c3", "c4"],
        "condition": ["ctx", "ctx", "ctx", "ctx", "ctx"],
        "batch": ["b", "b", "b", "b", "b"],
        "clone": ["", "a", "b", "c", "d"],
        "assignment_consensus2": ["WT", "bc-A", "bc-B", "bc-C", "bc-D"],
        "kogene": ["WT", "A", "B", "C", "D"],
        "kosym": ["WT", "A", "B", "C", "D"],
    }
    for name, items in values.items():
        np.save(metadata / f"{name}.npy", np.asarray(items))
        np.save(metadata / f"{name}_observed.npy", np.ones(5, dtype=np.bool_))
    action_map = tmp_path / "current-orfs.jsonl"
    mapped = {
        "A": stable_by_rule["fit"],
        "B": stable_by_rule["dev-test"],
        "C": stable_by_rule["held"],
    }
    action_map.write_text(
        "\n".join(
            json.dumps(
                {
                    "schema": "slp.sgd-current-orf/v1",
                    "ncbiTaxon": 4932,
                    "canonicalSgdCurie": stable,
                    "systematicName": name,
                    "displayMetadata": {},
                },
            )
            for name, stable in mapped.items()
        )
        + "\n",
    )
    row_map = tmp_path / "row-map.jsonl"
    row_map.write_text(
        "\n".join(
            json.dumps(
                {
                    "schema": "slp.yeast-seurat-rna-row-identity/v1",
                    "rowIndex": index,
                    "sourceIdentifier": name,
                    "canonicalSgdCurie": stable,
                    "mappingClass": "current-orf-systematic",
                },
            )
            for index, (name, stable) in enumerate(mapped.items())
        )
        + "\n",
    )
    report = prepare.freeze_metadata_selections(
        tmp_path / "metadata",
        {"frames": [{"path": ["seus", "ctx", "meta.data"]}]},
        action_map,
        row_map,
        tmp_path / "selection",
    )
    selection = np.load(tmp_path / "selection/frame-0-selection.npz")
    assert selection["source_columns"].tolist() == [0, 1]
    assert report["frames"][0]["excludedDevelopmentTestCells"] == 1
    assert report["frames"][0]["excludedProtectedCells"] == 1
    assert report["frames"][0]["exactMapFailureCells"] == 1
