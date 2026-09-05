"""Bounded, reference-aware inspection of gzip RDX3/XDR Seurat objects.

This module intentionally does not convert a Seurat object.  It inventories the
serialized structure while large atomic vectors pass through a small buffer.
Offsets are positions in the *decompressed* RDX3 stream; extraction therefore
replays the gzip member from the beginning.  A truncated prefix can be useful
for profiling, but is always reported as incomplete.

The intended two-pass contract is:

1. inventory structure, stable identifiers and cell metadata; assign protected
   intervention roles without inspecting perturbation expression values;
2. replay the source and materialize only allow-listed CSC column payload
   ranges for fitting interventions and eligible controls.

Only the RNA ``counts`` dgCMatrix is an admissible molecular input.  Callers
must reject SCT, ``data`` and ``scale.data`` paths.  The parser never selects
rows from observed effects or expression values.
"""

from __future__ import annotations

import enum
import gzip
import hashlib
import json
import struct
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np

RDATA_VERSION = "1.1.0"


class SeuratStreamError(ValueError):
    """Raised when an R serialization violates the bounded parser contract."""


class TruncatedRData(SeuratStreamError):
    """Raised internally when a prefix ends before the declared object does."""


class RType(enum.IntEnum):
    NIL = 0
    SYM = 1
    LIST = 2
    CLO = 3
    ENV = 4
    PROM = 5
    LANG = 6
    SPECIAL = 7
    BUILTIN = 8
    CHAR = 9
    LGL = 10
    INT = 13
    REAL = 14
    CPLX = 15
    STR = 16
    DOT = 17
    VEC = 19
    EXPR = 20
    BCODE = 21
    EXTPTR = 22
    WEAKREF = 23
    RAW = 24
    S4 = 25
    ALTREP = 238
    BASEENV = 241
    EMPTYENV = 242
    NAMESPACE = 249
    MISSINGARG = 251
    GLOBALENV = 253
    NILVALUE = 254
    REF = 255


@dataclass(frozen=True)
class Payload:
    """One serialized atomic-vector payload (offsets exclude its length word)."""

    kind: str
    path: tuple[str, ...]
    length: int
    start: int
    end: int
    sha256: str
    complete: bool
    items_read: int
    selected_count: int = 0
    selected_indices: tuple[int, ...] = ()
    selected_values: tuple[int | float | str, ...] = ()


@dataclass
class Node:
    """A lightweight structural R node; large vectors contain a Payload only."""

    kind: RType
    value: object = None
    attributes: Node | None = None
    tag: Node | None = None
    reference_id: int | None = None
    object_flag: bool = False


@dataclass
class Inventory:
    complete: bool
    root: Node | None
    format_version: int | None
    writer_version: int | None
    minimum_version: int | None
    encoding: str | None
    decompressed_bytes: int
    payloads: list[Payload] = field(default_factory=list)
    error: str | None = None


class _Counted:
    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.offset = 0

    def read(self, size: int) -> bytes:
        data = self.stream.read(size)
        self.offset += len(data)
        return data


def _checked_runtime() -> None:
    import rdata
    from rdata.parser import RObjectType

    if rdata.__version__ != RDATA_VERSION:
        raise SeuratStreamError("rdata runtime version drift")
    for name, member in RType.__members__.items():
        if name in RObjectType.__members__ and RObjectType[name].value != int(member):
            raise SeuratStreamError(f"rdata serialization type drift: {name}")


class XdrStreamParser:
    """Reference-safe RDX3 parser that bounds large atomic-vector memory.

    ``selected`` maps a structural path tuple to zero-based element ranges.
    Selected values are retained in the corresponding :class:`Payload`; all
    other large data is checksummed and discarded.  Pairlist tags are included
    in paths, while generic vectors use ``[index]`` components.
    """

    def __init__(
        self,
        stream: BinaryIO,
        *,
        materialize_limit: int = 4096,
        selected: dict[tuple[str, ...], Sequence[tuple[int, int]]] | None = None,
        payload_sinks: dict[tuple[str, ...], Callable[[int, np.ndarray], None]]
        | None = None,
        selected_string_paths: Iterable[tuple[str, ...]] = (),
        materialize_string_suffixes: Iterable[tuple[str, ...]] = (),
        materialize_atomic_names: Iterable[str] = (),
        max_materialized_bytes: int = 256 << 20,
        max_rss_bytes: int | None = None,
        deadline_monotonic: float | None = None,
        chunk_bytes: int = 1 << 20,
    ):
        _checked_runtime()
        self.stream = _Counted(stream)
        self.materialize_limit = materialize_limit
        self.selected = selected or {}
        self.payload_sinks = payload_sinks or {}
        self.selected_string_paths = frozenset(selected_string_paths)
        self.materialize_string_suffixes = tuple(materialize_string_suffixes)
        self.materialize_atomic_names = frozenset(materialize_atomic_names)
        self.max_materialized_bytes = max_materialized_bytes
        self.max_rss_bytes = max_rss_bytes
        self.deadline_monotonic = deadline_monotonic
        self._process = None
        if max_rss_bytes is not None:
            import psutil

            self._process = psutil.Process()
        self.chunk_bytes = max(8, chunk_bytes)
        self.references: list[Node] = []
        self.payloads: list[Payload] = []
        self._active_payload: tuple[str, int, int, hashlib._Hash] | None = None

    def _check_bounds(self) -> None:
        if (
            self.deadline_monotonic is not None
            and time.monotonic() > self.deadline_monotonic
        ):
            raise SeuratStreamError(
                "RData inspection exceeded its fixed wall-time bound"
            )
        if (
            self.max_rss_bytes is not None
            and self._process is not None
            and self._process.memory_info().rss > self.max_rss_bytes
        ):
            raise SeuratStreamError("RData inspection exceeded its fixed RSS bound")

    def exact(self, length: int) -> bytes:
        data = self.stream.read(length)
        if len(data) != length:
            raise TruncatedRData(
                f"truncated R XDR stream at decompressed byte {self.stream.offset}",
            )
        return data

    def integer(self) -> int:
        return struct.unpack(">i", self.exact(4))[0]

    def _atomic(self, kind: RType, item_format: str, path: tuple[str, ...]) -> object:
        length = self.integer()
        if length < 0:
            raise SeuratStreamError("negative atomic-vector length")
        item_size = np.dtype(item_format).itemsize
        ranges = _normalize_ranges(self.selected.get(path, ()), length)
        named_materialization = bool(path) and path[-1] in self.materialize_atomic_names
        retain_all = (
            length <= self.materialize_limit or named_materialization
        ) and not ranges
        if retain_all and length * item_size > self.max_materialized_bytes:
            raise SeuratStreamError(
                f"materialized payload {path} exceeds byte bound",
            )
        selected_indices: list[int] = []
        selected_values: list[int | float] = []
        selected_count = 0
        materialized = bytearray() if retain_all else None
        if retain_all:
            ranges = ((0, length),)
        start = self.stream.offset
        digest = hashlib.sha256()
        self._active_payload = (kind.name, length, start, digest)
        cursor = 0
        range_index = 0
        complete = True
        read_error: BaseException | None = None
        try:
            while cursor < length:
                self._check_bounds()
                count = min(length - cursor, max(1, self.chunk_bytes // item_size))
                raw = self.stream.read(count * item_size)
                digest.update(raw)
                if materialized is not None:
                    materialized.extend(raw)
                complete_items = len(raw) // item_size
                if complete_items:
                    values = np.frombuffer(
                        raw[: complete_items * item_size],
                        dtype=np.dtype(item_format),
                    )
                    chunk_stop = cursor + complete_items
                    while (
                        range_index < len(ranges) and ranges[range_index][1] <= cursor
                    ):
                        range_index += 1
                    check = range_index
                    while check < len(ranges) and ranges[check][0] < chunk_stop:
                        lo, hi = ranges[check]
                        take_lo, take_hi = max(lo, cursor), min(hi, chunk_stop)
                        if take_lo < take_hi and not retain_all:
                            selected_slice = values[take_lo - cursor : take_hi - cursor]
                            selected_count += len(selected_slice)
                            sink = self.payload_sinks.get(path)
                            if sink is None and not retain_all:
                                selected_indices.extend(range(take_lo, take_hi))
                                selected_values.extend(selected_slice.tolist())
                            elif sink is not None:
                                sink(take_lo, selected_slice)
                        check += 1
                cursor += complete_items
                if len(raw) != count * item_size:
                    complete = False
                    break
        except (EOFError, OSError) as error:
            complete = False
            read_error = error
        finally:
            payload = Payload(
                kind=kind.name,
                path=path,
                length=length,
                start=start,
                end=self.stream.offset,
                sha256=digest.hexdigest(),
                complete=complete,
                items_read=cursor,
                selected_count=selected_count,
                selected_indices=tuple(selected_indices),
                selected_values=tuple(selected_values),
            )
            self.payloads.append(payload)
            self._active_payload = None
        if not complete:
            if read_error is not None and not isinstance(read_error, EOFError):
                raise read_error
            raise TruncatedRData(
                f"truncated {kind.name} payload declared length {length} at "
                f"decompressed byte {self.stream.offset}",
            )
        if retain_all:
            dtype = np.dtype(item_format).newbyteorder("=")
            assert materialized is not None
            return np.frombuffer(materialized, dtype=np.dtype(item_format)).astype(
                dtype,
                copy=True,
            )
        return payload

    def _skip_strings(self, length: int, path: tuple[str, ...]) -> Payload:
        start = self.stream.offset
        digest = hashlib.sha256()
        complete = True
        items_read = 0
        try:
            for _ in range(length):
                raw_info = self.exact(4)
                digest.update(raw_info)
                info = struct.unpack(">I", raw_info)[0]
                kind = RType(info & 0xFF)
                if kind is not RType.CHAR:
                    raise SeuratStreamError(
                        "unselected STR contains non-CHAR serialization; cannot skip safely",
                    )
                raw_length = self.exact(4)
                digest.update(raw_length)
                item_length = struct.unpack(">i", raw_length)[0]
                if item_length < -1:
                    raise SeuratStreamError("invalid CHARSXP length")
                if item_length >= 0:
                    remaining = item_length
                    while remaining:
                        data = self.exact(min(remaining, self.chunk_bytes))
                        digest.update(data)
                        remaining -= len(data)
                items_read += 1
        except (EOFError, OSError, TruncatedRData):
            complete = False
        payload = Payload(
            "STR",
            path,
            length,
            start,
            self.stream.offset,
            digest.hexdigest(),
            complete,
            items_read,
            0,
        )
        self.payloads.append(payload)
        if not complete:
            raise TruncatedRData(
                f"truncated STR payload declared length {length} at "
                f"decompressed byte {self.stream.offset}",
            )
        return payload

    def object(self, path: tuple[str, ...] = ()) -> Node:
        self._check_bounds()
        raw = self.integer()
        kind = RType(raw & 0xFF)
        if kind is RType.REF:
            ref = raw >> 8
            if ref < 1 or ref > len(self.references):
                raise SeuratStreamError(f"invalid R reference {ref}")
            return Node(kind, self.references[ref - 1], reference_id=ref)
        object_flag = bool(raw & 0x100)
        has_attributes = bool(raw & 0x200)
        has_tag = bool(raw & 0x400)
        attrs: Node | None = None
        tag: Node | None = None

        if kind in {RType.LIST, RType.LANG, RType.CLO, RType.PROM, RType.DOT}:
            if has_attributes:
                attrs = self.object(path + ("@attributes",))
            if has_tag:
                tag = self.object(path + ("@tag",))
            component = _symbol_text(tag) if tag is not None else "car"
            car = self.object(path + (component,))
            cdr = self.object(path)
            return Node(kind, (car, cdr), attrs, tag, object_flag=object_flag)

        if has_tag:
            raise SeuratStreamError(f"unexpected tag flag on {kind.name}")
        if kind in {
            RType.NIL,
            RType.NILVALUE,
            RType.BASEENV,
            RType.EMPTYENV,
            RType.MISSINGARG,
            RType.GLOBALENV,
            RType.S4,
        }:
            node = Node(kind, object_flag=object_flag)
        elif kind is RType.CHAR:
            length = self.integer()
            if length == -1:
                value = None
            elif length < -1:
                raise SeuratStreamError("invalid CHARSXP length")
            else:
                value = self.exact(length).decode("utf-8", errors="strict")
            node = Node(kind, value, object_flag=object_flag)
        elif kind is RType.SYM:
            node = Node(kind, self.object(path + ("@symbol",)), object_flag=object_flag)
            self.references.append(node)
            node.reference_id = len(self.references)
        elif kind in {RType.LGL, RType.INT}:
            node = Node(kind, self._atomic(kind, ">i4", path), object_flag=object_flag)
        elif kind is RType.REAL:
            node = Node(kind, self._atomic(kind, ">f8", path), object_flag=object_flag)
        elif kind is RType.CPLX:
            node = Node(kind, self._atomic(kind, ">c16", path), object_flag=object_flag)
        elif kind is RType.RAW:
            node = Node(kind, self._atomic(kind, ">u1", path), object_flag=object_flag)
        elif kind is RType.STR:
            length = self.integer()
            if length < 0:
                raise SeuratStreamError("negative R vector length")
            suffix_selected = any(
                len(path) >= len(suffix) and path[-len(suffix) :] == suffix
                for suffix in self.materialize_string_suffixes
            )
            if (
                length <= self.materialize_limit
                or path in self.selected_string_paths
                or suffix_selected
            ):
                values = [
                    self.object(path + (f"[{index}]",)) for index in range(length)
                ]
                node = Node(kind, values, object_flag=object_flag)
            else:
                node = Node(
                    kind, self._skip_strings(length, path), object_flag=object_flag
                )
        elif kind in {RType.VEC, RType.EXPR}:
            length = self.integer()
            if length < 0:
                raise SeuratStreamError("negative R vector length")
            values = [self.object(path + (f"[{index}]",)) for index in range(length)]
            node = Node(kind, values, object_flag=object_flag)
        elif kind is RType.ALTREP:
            node = Node(
                kind,
                tuple(self.object(path + (f"@altrep{index}",)) for index in range(3)),
            )
        elif kind in {RType.SPECIAL, RType.BUILTIN}:
            length = self.integer()
            node = Node(kind, self.exact(length), object_flag=object_flag)
        elif kind is RType.NAMESPACE:
            if self.integer() != 0:
                raise SeuratStreamError("namespace placeholder drift")
            length = self.integer()
            node = Node(kind, [self.object(path + (f"[{i}]",)) for i in range(length)])
            self.references.append(node)
            node.reference_id = len(self.references)
        elif kind is RType.ENV:
            # Environments enter the reference table before their recursive
            # contents, matching R's serializer and permitting self-reference.
            node = Node(kind, object_flag=True)
            self.references.append(node)
            node.reference_id = len(self.references)
            locked = bool(self.integer())
            node.value = (
                locked,
                self.object(path + ("@enclosure",)),
                self.object(path + ("@frame",)),
                self.object(path + ("@hash",)),
            )
            attrs = self.object(path + ("@attributes",))
            node.attributes = attrs
            has_attributes = False
        elif kind is RType.EXTPTR:
            node = Node(kind, object_flag=object_flag)
            self.references.append(node)
            node.reference_id = len(self.references)
            node.value = (
                self.object(path + ("@protected",)),
                self.object(path + ("@extptr_tag",)),
            )
        elif kind in {RType.WEAKREF, RType.BCODE}:
            raise SeuratStreamError(
                f"unsupported reference-bearing {kind.name}; refusing unsafe reference drift",
            )
        else:
            raise SeuratStreamError(f"unsupported R type {kind.name}")
        if has_attributes:
            attrs = self.object(path + ("@attributes",))
            node.attributes = attrs
        return node


def _normalize_ranges(
    ranges: Iterable[tuple[int, int]],
    length: int,
) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for start, stop in sorted(ranges):
        if start < 0 or stop < start or stop > length:
            raise SeuratStreamError("selected payload range outside vector")
        if start == stop:
            continue
        if result and start < result[-1][1]:
            raise SeuratStreamError("selected payload ranges overlap")
        result.append((start, stop))
    return tuple(result)


def _symbol_text(node: Node) -> str:
    if node.kind is RType.REF:
        node = node.value  # type: ignore[assignment]
    if node.kind is not RType.SYM:
        raise SeuratStreamError("pairlist tag is not a symbol")
    char = node.value
    if (
        not isinstance(char, Node)
        or char.kind is not RType.CHAR
        or not isinstance(char.value, str)
    ):
        raise SeuratStreamError("symbol does not contain a character")
    return char.value


def attributes(node: Node | None) -> dict[str, Node]:
    """Convert an R attribute/slot pairlist into a tag-to-node mapping."""
    result: dict[str, Node] = {}
    while node is not None and node.kind not in {RType.NIL, RType.NILVALUE}:
        if node.kind is not RType.LIST or node.tag is None:
            raise SeuratStreamError("malformed attribute pairlist")
        current = node
        car, node = current.value  # type: ignore[misc,assignment]
        name = _symbol_text(current.tag)
        if name in result:
            raise SeuratStreamError(f"duplicate R attribute/slot {name}")
        result[name] = car
    return result


def dereference(node: Node) -> Node:
    """Resolve a serialization reference without copying its target."""
    seen: set[int] = set()
    while node.kind is RType.REF:
        if node.reference_id in seen or not isinstance(node.value, Node):
            raise SeuratStreamError("cyclic or malformed R reference")
        seen.add(node.reference_id or -1)
        node = node.value
    return node


def string_values(node: Node) -> tuple[str, ...]:
    """Return a materialized STRSXP, resolving symbol-safe references."""
    node = dereference(node)
    if node.kind is not RType.STR or not isinstance(node.value, list):
        raise SeuratStreamError("expected materialized R character vector")
    result: list[str] = []
    for item in node.value:
        item = dereference(item)
        if item.kind is not RType.CHAR or not isinstance(item.value, str):
            raise SeuratStreamError("character vector contains a non-string value")
        result.append(item.value)
    return tuple(result)


def nullable_string_values(node: Node) -> tuple[str | None, ...]:
    """Return a materialized STRSXP while preserving serialized NA strings."""
    node = dereference(node)
    if node.kind is not RType.STR or not isinstance(node.value, list):
        raise SeuratStreamError("expected materialized R character vector")
    result: list[str | None] = []
    for item in node.value:
        item = dereference(item)
        if item.kind is not RType.CHAR or (
            item.value is not None and not isinstance(item.value, str)
        ):
            raise SeuratStreamError("character vector contains an invalid value")
        result.append(item.value)
    return tuple(result)


def class_names(node: Node) -> tuple[str, ...]:
    """Read the materialized S3/S4 class attribute, if present."""
    class_node = attributes(node.attributes).get("class")
    return () if class_node is None else string_values(class_node)


@dataclass(frozen=True)
class DgcMatrixInventory:
    path: tuple[str, ...]
    rows: int
    columns: int
    nnz: int
    slots: tuple[str, ...]


def find_dgc_matrices(root: Node) -> list[DgcMatrixInventory]:
    """Find completely inventoried dgCMatrix S4 nodes without copying payloads."""
    found: list[DgcMatrixInventory] = []

    def walk(node: Node, path: tuple[str, ...]) -> None:
        node = dereference(node)
        if node.kind is RType.S4:
            slots = attributes(node.attributes)
            if "dgCMatrix" in class_names(node):
                rows, columns, nnz = validate_dgc_slots(slots)
                found.append(DgcMatrixInventory(path, rows, columns, nnz, tuple(slots)))
            for name, child in slots.items():
                if name != "class":
                    walk(child, path + (name,))
            return
        if node.kind in {RType.LIST, RType.LANG, RType.CLO, RType.PROM, RType.DOT}:
            car, cdr = node.value  # type: ignore[misc]
            component = _symbol_text(node.tag) if node.tag is not None else "car"
            walk(car, path + (component,))
            if isinstance(cdr, Node) and cdr.kind not in {RType.NIL, RType.NILVALUE}:
                walk(cdr, path)
        elif node.kind in {RType.VEC, RType.EXPR} and isinstance(node.value, list):
            names_node = attributes(node.attributes).get("names")
            names = string_values(names_node) if names_node is not None else ()
            for index, child in enumerate(node.value):
                component = names[index] if index < len(names) else f"[{index}]"
                walk(child, path + (component,))

    walk(root, ())
    return found


def inspect_rdata(
    path: Path,
    *,
    allow_truncated: bool = False,
    materialize_limit: int = 4096,
    selected: dict[tuple[str, ...], Sequence[tuple[int, int]]] | None = None,
    payload_sinks: dict[tuple[str, ...], Callable[[int, np.ndarray], None]]
    | None = None,
    selected_string_paths: Iterable[tuple[str, ...]] = (),
    materialize_string_suffixes: Iterable[tuple[str, ...]] = (),
    materialize_atomic_names: Iterable[str] = (),
    max_materialized_bytes: int = 256 << 20,
    max_rss_bytes: int | None = None,
    max_seconds: float | None = None,
) -> Inventory:
    """Inspect a gzip RDX3 file or prefix without allocating large vectors."""
    parser: XdrStreamParser | None = None
    versions: tuple[int | None, int | None, int | None] = (None, None, None)
    encoding: str | None = None
    root: Node | None = None
    try:
        with gzip.open(path, "rb") as raw:
            parser = XdrStreamParser(
                raw,
                materialize_limit=materialize_limit,
                selected=selected,
                payload_sinks=payload_sinks,
                selected_string_paths=selected_string_paths,
                materialize_string_suffixes=materialize_string_suffixes,
                materialize_atomic_names=materialize_atomic_names,
                max_materialized_bytes=max_materialized_bytes,
                max_rss_bytes=max_rss_bytes,
                deadline_monotonic=(
                    None if max_seconds is None else time.monotonic() + max_seconds
                ),
            )
            if parser.exact(7) != b"RDX3\nX\n":
                raise SeuratStreamError("source must be gzip RDX3/XDR")
            versions = (parser.integer(), parser.integer(), parser.integer())
            if versions[0] != 3 or min(versions[1:]) <= 0:  # type: ignore[arg-type]
                raise SeuratStreamError("R serialization version drift")
            encoding_length = parser.integer()
            if encoding_length < 0 or encoding_length > 128:
                raise SeuratStreamError("invalid R serialization encoding length")
            encoding = parser.exact(encoding_length).decode("ascii")
            root = parser.object()
            if raw.read(1):
                raise SeuratStreamError("unexpected trailing decompressed data")
        return Inventory(
            True,
            root,
            *versions,
            encoding,
            parser.stream.offset,
            list(parser.payloads),
        )
    except (EOFError, gzip.BadGzipFile, TruncatedRData) as error:
        if not allow_truncated:
            raise SeuratStreamError(str(error)) from error
        offset = 0 if parser is None else parser.stream.offset
        payloads = [] if parser is None else list(parser.payloads)
        return Inventory(
            False,
            root,
            *versions,
            encoding,
            offset,
            payloads,
            str(error),
        )


def plan_csc_payload_ranges(
    column_pointers: Sequence[int],
    column_ranges: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Translate selected CSC column ranges into contiguous ``i``/``x`` ranges."""
    p = np.asarray(column_pointers, dtype=np.int64)
    if p.ndim != 1 or len(p) < 1 or p[0] != 0 or np.any(np.diff(p) < 0):
        raise SeuratStreamError("invalid CSC column pointer")
    columns = _normalize_ranges(column_ranges, len(p) - 1)
    payload = [(int(p[start]), int(p[stop])) for start, stop in columns]
    return _normalize_ranges(payload, int(p[-1]))


def is_admissible_rna_counts_path(path: Sequence[str]) -> bool:
    """Return whether a structural slot path denotes raw RNA counts only."""
    lowered = tuple(part.lower() for part in path)
    if any(part in {"sct", "data", "scale.data", "scaled.data"} for part in lowered):
        return False
    return "assays" in lowered and "counts" in lowered and "rna" in lowered


def validate_dgc_slots(slot_map: dict[str, Node]) -> tuple[int, int, int]:
    """Validate lightweight dgCMatrix structural metadata and return rows/cols/nnz."""
    required = {"i", "p", "Dim", "Dimnames", "x"}
    if not required.issubset(slot_map):
        raise SeuratStreamError(
            f"dgCMatrix missing slots {sorted(required - slot_map.keys())}"
        )
    dim = slot_map["Dim"].value
    p = slot_map["p"].value
    i_value, x_value = slot_map["i"].value, slot_map["x"].value
    if not isinstance(dim, np.ndarray) or dim.shape != (2,):
        raise SeuratStreamError(
            "dgCMatrix Dim must be a materialized length-two integer vector"
        )
    if not isinstance(p, np.ndarray) or p.shape != (int(dim[1]) + 1,):
        raise SeuratStreamError("dgCMatrix p length mismatch")
    i_len = len(i_value) if isinstance(i_value, np.ndarray) else i_value.length
    x_len = len(x_value) if isinstance(x_value, np.ndarray) else x_value.length
    if i_len != x_len or int(p[-1]) != i_len:
        raise SeuratStreamError("dgCMatrix nnz contract mismatch")
    return int(dim[0]), int(dim[1]), i_len


def inventory_summary(inventory: Inventory) -> dict[str, object]:
    """Return a JSON-safe summary that never serializes molecular values."""
    return {
        "schema": "slp.yeast-seurat-rdata-stream-inventory/v1",
        "complete": inventory.complete,
        "formatVersion": inventory.format_version,
        "writerVersion": inventory.writer_version,
        "minimumVersion": inventory.minimum_version,
        "encoding": inventory.encoding,
        "decompressedBytesConsumed": inventory.decompressed_bytes,
        "error": inventory.error,
        "payloads": [
            {
                "kind": payload.kind,
                "path": list(payload.path),
                "declaredLength": payload.length,
                "itemsRead": payload.items_read,
                "decompressedPayloadStart": payload.start,
                "decompressedPayloadEnd": payload.end,
                "consumedPayloadSha256": payload.sha256,
                "complete": payload.complete,
                "selectedItemCount": payload.selected_count,
            }
            for payload in inventory.payloads
        ],
    }


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--allow-truncated", action="store_true")
    parser.add_argument("--materialize-limit", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inventory_summary(
        inspect_rdata(
            args.path,
            allow_truncated=args.allow_truncated,
            materialize_limit=args.materialize_limit,
        ),
    )
    source_digest = hashlib.sha256()
    with args.path.open("rb") as source:
        while chunk := source.read(1 << 20):
            source_digest.update(chunk)
    result["source"] = {
        "path": str(args.path.resolve()),
        "bytes": args.path.stat().st_size,
        "sha256": source_digest.hexdigest(),
        "scope": "partial HTTP range only"
        if not result["complete"]
        else "complete file",
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise SeuratStreamError(f"refusing to overwrite {args.output}")
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    _main()
