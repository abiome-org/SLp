"""Bounded reader and adapter for the Nadal-Ribelles yeast RNA summaries."""

from __future__ import annotations

import enum
import gzip
import hashlib
import json
import re
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

NCBI_TAXON = 4932
RDATA_VERSION = "1.1.0"
HELD_DOMAIN = b"slp-1.1-yeast-global-held-v1\x00"
DEVELOPMENT_PREFIX = "slp11-development-v1|731|"
FILE_PATTERN = re.compile(r"^DEG_(Control|NaCl)_bc_(.+)\.csv$")


class RObjectType(enum.IntEnum):
    NIL = 0
    SYM = 1
    LIST = 2
    CLO = 3
    PROM = 5
    LANG = 6
    CHAR = 9
    LGL = 10
    INT = 13
    REAL = 14
    STR = 16
    DOT = 17
    VEC = 19
    EXPR = 20
    ALTREP = 238
    NILVALUE = 254
    REF = 255


class AtlasResponseError(ValueError):
    """Raised when a pinned summary violates the frozen molecular contract."""


@dataclass
class Object:
    kind: RObjectType
    value: object
    attributes: Object | None = None
    tag: Object | None = None


class BoundedXdrReader:
    """Small SAX-like reader for the simple R objects used by this source.

    rdata 1.1.0 supplies the pinned serialization type definitions. Unlike its
    general object converter, this reader discards repeated data-frame strings
    as they pass and therefore avoids multi-gigabyte Python object expansion.
    """

    def __init__(self, stream: BinaryIO):
        import rdata
        from rdata.parser import RObjectType as PinnedRObjectType

        if rdata.__version__ != RDATA_VERSION or any(
            PinnedRObjectType[name].value != int(value)
            for name, value in RObjectType.__members__.items()
        ):
            raise AtlasResponseError("rdata runtime version drift")
        self.stream = stream
        self.references: list[Object] = []

    def read_exact(self, length: int) -> bytes:
        value = self.stream.read(length)
        if len(value) != length:
            raise AtlasResponseError("truncated R XDR stream")
        return value

    def integer(self) -> int:
        return struct.unpack(">i", self.read_exact(4))[0]

    def info(self) -> tuple[RObjectType, bool, bool, bool, int]:
        raw = self.integer()
        kind = RObjectType(raw & 0xFF)
        if kind is RObjectType.REF:
            return kind, False, False, False, raw >> 8
        return kind, bool(raw & 0x100), bool(raw & 0x200), bool(raw & 0x400), 0

    def array(self, dtype: str) -> np.ndarray:
        length = self.integer()
        if length < 0:
            raise AtlasResponseError("negative R vector length")
        itemsize = np.dtype(dtype).itemsize
        return np.frombuffer(self.read_exact(length * itemsize), dtype=dtype).astype(
            np.dtype(dtype).newbyteorder("="), copy=True,
        )

    def object(self) -> Object:
        kind, _object, attributes, tagged, reference = self.info()
        if kind is RObjectType.REF:
            if reference < 1 or reference > len(self.references):
                raise AtlasResponseError("invalid R serialization reference")
            return self.references[reference - 1]
        attrs = self.object() if attributes and kind in {
            RObjectType.LIST, RObjectType.LANG, RObjectType.CLO, RObjectType.PROM,
            RObjectType.DOT,
        } else None
        tag = self.object() if tagged else None
        if kind in {RObjectType.NIL, RObjectType.NILVALUE}:
            result = Object(kind, None, attrs, tag)
        elif kind is RObjectType.CHAR:
            length = self.integer()
            value = None if length == -1 else self.read_exact(length).decode("utf-8")
            result = Object(kind, value, attrs, tag)
        elif kind is RObjectType.SYM:
            result = Object(kind, self.object(), attrs, tag)
            self.references.append(result)
        elif kind in {RObjectType.LIST, RObjectType.LANG, RObjectType.CLO,
                      RObjectType.PROM, RObjectType.DOT}:
            result = Object(kind, (self.object(), self.object()), attrs, tag)
        elif kind in {RObjectType.LGL, RObjectType.INT}:
            result = Object(kind, self.array(">i4"), attrs, tag)
        elif kind is RObjectType.REAL:
            result = Object(kind, self.array(">f8"), attrs, tag)
        elif kind in {RObjectType.STR, RObjectType.VEC, RObjectType.EXPR}:
            length = self.integer()
            if length < 0:
                raise AtlasResponseError("negative R vector length")
            values = [self.object() for _ in range(length)]
            if attributes:
                attrs = self.object()
            result = Object(kind, values, attrs, tag)
        elif kind is RObjectType.ALTREP:
            result = Object(kind, (self.object(), self.object(), self.object()), attrs, tag)
        else:
            raise AtlasResponseError(f"unsupported R object type {kind.name}")
        if attributes and attrs is None and kind not in {
            RObjectType.LIST, RObjectType.LANG, RObjectType.CLO, RObjectType.PROM,
            RObjectType.DOT, RObjectType.STR, RObjectType.VEC, RObjectType.EXPR,
        }:
            result.attributes = self.object()
        return result


def _text(value: Object) -> str:
    while value.kind is RObjectType.SYM:
        value = value.value  # type: ignore[assignment]
    if value.kind is not RObjectType.CHAR or not isinstance(value.value, str):
        raise AtlasResponseError("expected non-null R character")
    return value.value


def _strings(value: Object) -> list[str]:
    if value.kind is not RObjectType.STR or not isinstance(value.value, list):
        raise AtlasResponseError("expected R character vector")
    return [_text(item) for item in value.value]


def _attributes(value: Object | None) -> dict[str, Object]:
    result: dict[str, Object] = {}
    while value is not None and value.kind not in {RObjectType.NIL, RObjectType.NILVALUE}:
        if value.kind is not RObjectType.LIST or value.tag is None:
            raise AtlasResponseError("malformed R attribute pairlist")
        car, cdr = value.value  # type: ignore[misc]
        result[_text(value.tag)] = car
        value = cdr
    return result


def _read_string_vector(reader: BoundedXdrReader) -> tuple[str, ...]:
    kind, _object, attributes, tagged, _reference = reader.info()
    if kind is not RObjectType.STR or attributes or tagged:
        raise AtlasResponseError("fcs query names must be a plain character vector")
    length = reader.integer()
    return tuple(_text(reader.object()) for _ in range(length))


def _read_real_vector(reader: BoundedXdrReader) -> np.ndarray:
    kind, _object, attributes, tagged, _reference = reader.info()
    if kind is not RObjectType.REAL or attributes or tagged:
        raise AtlasResponseError("fcs logfoldchanges must be a plain numeric vector")
    return reader.array(">f8")


def _scan_fcs(
    path: Path,
    on_frame: Callable[[int, tuple[str, ...], np.ndarray], None],
) -> list[str]:
    with gzip.open(path, "rb") as stream:
        if stream.read(7) != b"RDX3\nX\n":
            raise AtlasResponseError("FC_genotype must be gzip RDX3/XDR")
        reader = BoundedXdrReader(stream)
        format_version, serialized_version, minimum_version = (
            reader.integer(), reader.integer(), reader.integer()
        )
        if format_version != 3 or min(serialized_version, minimum_version) <= 0:
            raise AtlasResponseError("R serialization version drift")
        encoding_length = reader.integer()
        if reader.read_exact(encoding_length) != b"UTF-8":
            raise AtlasResponseError("R serialization encoding drift")
        kind, _object, attributes, tagged, _reference = reader.info()
        if kind is not RObjectType.LIST or attributes or not tagged:
            raise AtlasResponseError("saved R root must be one tagged pairlist")
        if _text(reader.object()) != "fcs":
            raise AtlasResponseError("saved R object must be named fcs")
        kind, _object, attributes, tagged, _reference = reader.info()
        if kind is not RObjectType.VEC or not attributes or tagged:
            raise AtlasResponseError("fcs must be one named list")
        frame_count = reader.integer()
        if frame_count < 1 or frame_count > 10_000:
            raise AtlasResponseError("fcs frame count outside bound")
        for frame_index in range(frame_count):
            frame_kind, is_object, frame_attributes, frame_tagged, _ = reader.info()
            if (
                frame_kind is not RObjectType.VEC or not is_object
                or not frame_attributes or frame_tagged or reader.integer() != 2
            ):
                raise AtlasResponseError("fcs member must be a two-column data.frame")
            names = _read_string_vector(reader)
            values = _read_real_vector(reader)
            if len(names) != len(values) or not names:
                raise AtlasResponseError("fcs names/logfoldchanges length mismatch")
            if len(set(names)) != len(names):
                raise AtlasResponseError("fcs query names must be unique within a frame")
            on_frame(frame_index, names, values)
            attrs = _attributes(reader.object())
            if _strings(attrs.get("names")) != ["names", "logfoldchanges"]:  # type: ignore[arg-type]
                raise AtlasResponseError("fcs data.frame column contract drift")
            if _strings(attrs.get("class")) != ["data.frame"]:  # type: ignore[arg-type]
                raise AtlasResponseError("fcs member class drift")
        attrs = _attributes(reader.object())
        frame_names = _strings(attrs.get("names"))  # type: ignore[arg-type]
        if len(frame_names) != frame_count or len(set(frame_names)) != frame_count:
            raise AtlasResponseError("fcs list names must be complete and unique")
        tail = reader.object()
        if tail.kind not in {RObjectType.NIL, RObjectType.NILVALUE} or stream.read(1):
            raise AtlasResponseError("unexpected data after saved fcs object")
    return frame_names


def extract_fcs(path: Path, values_path: Path, observed_path: Path) -> dict[str, object]:
    """Two-pass stream of ragged fcs frames into bounded aligned memory maps."""
    query_union: set[str] = set()
    query_counts: list[int] = []
    finite_counts: list[int] = []

    def inventory(_index: int, names: tuple[str, ...], values: np.ndarray) -> None:
        query_union.update(names)
        query_counts.append(len(names))
        finite_counts.append(int(np.count_nonzero(np.isfinite(values))))

    frame_names = _scan_fcs(path, inventory)
    query_names = tuple(sorted(query_union))
    positions = {name: index for index, name in enumerate(query_names)}
    shape = (len(frame_names), len(query_names))
    matrix = np.lib.format.open_memmap(values_path, mode="w+", dtype=np.float32, shape=shape)
    matrix[:] = 0
    observed = np.lib.format.open_memmap(observed_path, mode="w+", dtype=np.bool_, shape=shape)
    observed[:] = False

    def write(frame: int, names: tuple[str, ...], values: np.ndarray) -> None:
        columns = np.fromiter(
            (positions[name] for name in names), dtype=np.int64, count=len(names),
        )
        finite = np.isfinite(values)
        matrix[frame, columns[finite]] = values[finite].astype(np.float32)
        observed[frame, columns[finite]] = True

    repeated_names = _scan_fcs(path, write)
    if repeated_names != frame_names:
        raise AtlasResponseError("fcs frame roster changed between bounded passes")
    matrix.flush()
    observed.flush()
    return {
        "frame_names": frame_names,
        "query_names": query_names,
        "frames": len(frame_names),
        "queries": len(query_names),
        "minimum_frame_queries": min(query_counts),
        "maximum_frame_queries": max(query_counts),
        "observed_values": sum(finite_counts),
        "missing_or_absent_values": int(np.prod(shape) - sum(finite_counts)),
        "query_roster_sha256": hashlib.sha256(
            "\n".join(query_names).encode("utf-8") + b"\n",
        ).hexdigest(),
    }


def protected_role(intervention_id: str) -> str:
    digest = hashlib.sha256(HELD_DOMAIN + intervention_id.encode("ascii")).hexdigest()
    bucket = int(digest[:16], 16) % 100
    return "molecular-final" if bucket < 10 else "molecular-validation" if bucket < 30 else "pretrain"


def development_role(intervention_id: str) -> str:
    digest = hashlib.sha256(
        f"{DEVELOPMENT_PREFIX}{NCBI_TAXON}|{intervention_id}".encode(),
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def exact_current_maps(path: Path) -> tuple[dict[str, str], set[str]]:
    """Return unambiguous exact systematic/standard-name to current SGD CURIE."""
    candidates: dict[str, set[str]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(raw)
        if item.get("schema") != "slp.sgd-current-orf/v1" or item.get("ncbiTaxon") != 4932:
            raise AtlasResponseError("current SGD mapping contract drift")
        curie = item["canonicalSgdCurie"]
        for name in (item.get("systematicName"), item.get("displayMetadata", {}).get("standardGeneName")):
            if isinstance(name, str) and name:
                candidates.setdefault(name, set()).add(curie)
    ambiguous = {name for name, ids in candidates.items() if len(ids) != 1}
    return {name: next(iter(ids)) for name, ids in candidates.items() if len(ids) == 1}, ambiguous
