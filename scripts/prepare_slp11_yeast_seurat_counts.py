"""Inventory Seurat RData metadata, then replay selected raw RNA CSC columns.

The first real-source invocation is inventory-only. It verifies the complete
source file, discovers S4 paths and data-frame schemas, retains only sparse
``p``/``Dim`` metadata, and skips/checksums large values. It never assigns
control, genotype, context, or protected-gene meaning.

The shard writer is a separate callable contract. Its selected columns must be
provided by an already-frozen metadata decision; it performs no selection from
counts and accepts only a discovered RNA/counts dgCMatrix.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RDATA_SITE = ROOT / "data" / "tools" / "rdata-1.1.0" / "site-packages"
MODULE = ROOT / "modules" / "slp-1-1-yeast-seurat-stream-v1" / "seurat_stream.py"
sys.path.insert(0, str(RDATA_SITE))
SPEC = importlib.util.spec_from_file_location("slp11_yeast_seurat_stream", MODULE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load yeast Seurat stream module")
ss = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ss
SPEC.loader.exec_module(ss)

SCHEMA = "slp.yeast-seurat-metadata-inventory/v1"
HELD_DOMAIN = b"slp-1.1-yeast-global-held-v1\x00"
SGD_CURIE = re.compile(r"^SGD:S[0-9]{9}$")
DEVELOPMENT_PREFIX = "slp11-development-v1|731|"


@dataclass(frozen=True)
class DgcCandidate:
    semantic_path: tuple[str, ...]
    slots: dict[str, ss.Node]
    rows: int
    columns: int
    nnz: int


@dataclass(frozen=True)
class DataFrameCandidate:
    semantic_path: tuple[str, ...]
    columns: dict[str, ss.Node]
    row_names: ss.Node | None


@dataclass(frozen=True)
class FrozenColumnSelection:
    columns: np.ndarray
    sha256: str
    control_count: int
    pretrain_action_count: int
    excluded_held_count: int
    excluded_unmapped_count: int


def protected_role(stable_sgd_id: str) -> str:
    """Recompute the global yeast intervention role from a stable SGD CURIE."""
    if SGD_CURIE.fullmatch(stable_sgd_id) is None:
        raise ss.SeuratStreamError(f"not a stable SGD CURIE: {stable_sgd_id!r}")
    digest = hashlib.sha256(HELD_DOMAIN + stable_sgd_id.encode("ascii")).hexdigest()
    bucket = int(digest[:16], 16) % 100
    if bucket < 10:
        return "molecular-final"
    return "molecular-validation" if bucket < 30 else "pretrain"


def development_role(stable_sgd_id: str) -> str:
    if SGD_CURIE.fullmatch(stable_sgd_id) is None:
        raise ss.SeuratStreamError(f"not a stable SGD CURIE: {stable_sgd_id!r}")
    digest = hashlib.sha256(
        f"{DEVELOPMENT_PREFIX}4932|{stable_sgd_id}".encode(),
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def load_exact_current_action_map(path: Path) -> dict[str, str]:
    candidates: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if (
            item.get("schema") != "slp.sgd-current-orf/v1"
            or item.get("ncbiTaxon") != 4932
        ):
            raise ss.SeuratStreamError("current SGD action map contract drift")
        stable = item["canonicalSgdCurie"]
        for name in (
            item.get("systematicName"),
            item.get("displayMetadata", {}).get("standardGeneName"),
        ):
            if isinstance(name, str) and name:
                candidates.setdefault(name, set()).add(stable)
    return {
        name: next(iter(stable_ids))
        for name, stable_ids in candidates.items()
        if len(stable_ids) == 1
    }


def build_strict_query_map(row_mapping_path: Path) -> dict[str, np.ndarray]:
    rows = [json.loads(line) for line in row_mapping_path.read_text().splitlines()]
    if any(
        item.get("schema") != "slp.yeast-seurat-rna-row-identity/v1" for item in rows
    ):
        raise ss.SeuratStreamError("RNA row mapping schema drift")
    rows.sort(key=lambda item: item["rowIndex"])
    if [item["rowIndex"] for item in rows] != list(range(len(rows))):
        raise ss.SeuratStreamError("RNA row mapping indices are not contiguous")
    strict = [
        item
        for item in rows
        if "alias-only" not in item["mappingClass"]
        and "candidate" not in item["mappingClass"]
    ]
    query_ids = sorted(item["canonicalSgdCurie"] for item in strict)
    if len(query_ids) != len(set(query_ids)):
        raise ss.SeuratStreamError("strict RNA query mapping is not one-to-one")
    positions = {stable: index for index, stable in enumerate(query_ids)}
    source_to_query = np.full(len(rows), -1, dtype=np.int64)
    for item in strict:
        source_to_query[item["rowIndex"]] = positions[item["canonicalSgdCurie"]]
    return {
        "source_row_ids": np.asarray([item["sourceIdentifier"] for item in rows]),
        "query_ids": np.asarray(query_ids),
        "source_to_query_index": source_to_query,
        "denominator_mask": np.ones(len(rows), dtype=np.bool_),
    }


def freeze_action_safe_columns(
    stable_action_ids: np.ndarray,
    explicit_control_mask: np.ndarray,
) -> FrozenColumnSelection:
    """Keep explicit controls and pretrain actions; exclude held/unmapped actions.

    ``explicit_control_mask`` must come from a separately reviewed source
    metadata rule. This function never guesses a control label.
    """
    actions = np.asarray(stable_action_ids)
    controls = np.asarray(explicit_control_mask, dtype=np.bool_)
    if actions.ndim != 1 or controls.shape != actions.shape:
        raise ss.SeuratStreamError("action IDs/control mask shape mismatch")
    selected: list[int] = []
    control_count = pretrain_count = held_count = unmapped_count = 0
    digest = hashlib.sha256()
    for index, (raw_action, control) in enumerate(zip(actions, controls, strict=True)):
        action = str(raw_action)
        if control:
            selected.append(index)
            control_count += 1
            digest.update(f"{index}\tCONTROL\n".encode())
        elif SGD_CURIE.fullmatch(action) is None:
            unmapped_count += 1
        elif protected_role(action) == "pretrain":
            selected.append(index)
            pretrain_count += 1
            digest.update(f"{index}\t{action}\n".encode())
        else:
            held_count += 1
    if not selected:
        raise ss.SeuratStreamError("metadata rule selected no safe columns")
    return FrozenColumnSelection(
        np.asarray(selected, dtype=np.int64),
        digest.hexdigest(),
        control_count,
        pretrain_count,
        held_count,
        unmapped_count,
    )


def _hashes(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            sha256.update(chunk)
            md5.update(chunk)
    return sha256.hexdigest(), md5.hexdigest()


def verify_source(
    path: Path,
    *,
    expected_bytes: int,
    expected_md5: str,
    expected_sha256: str,
) -> dict[str, object]:
    """Verify the immutable complete source before opening its gzip member."""
    if path.name.endswith(".partial"):
        raise ss.SeuratStreamError("refusing to inspect a partial acquisition")
    if path.stat().st_size != expected_bytes:
        raise ss.SeuratStreamError("source byte-size mismatch")
    sha256, md5 = _hashes(path)
    if md5.lower() != expected_md5.lower():
        raise ss.SeuratStreamError("source upstream MD5 mismatch")
    if sha256.lower() != expected_sha256.lower():
        raise ss.SeuratStreamError("source SHA-256 mismatch")
    return {
        "path": str(path.resolve()),
        "bytes": expected_bytes,
        "md5": md5,
        "sha256": sha256,
    }


def _attrs(node: ss.Node) -> dict[str, ss.Node]:
    return ss.attributes(ss.dereference(node).attributes)


def _payload(value: object) -> ss.Payload | None:
    return value if isinstance(value, ss.Payload) else None


def _node_descriptor(node: ss.Node) -> dict[str, object]:
    node = ss.dereference(node)
    payload = _payload(node.value)
    result: dict[str, object] = {"rType": node.kind.name}
    if payload is not None:
        result.update(
            {
                "length": payload.length,
                "payloadPath": list(payload.path),
                "payloadSha256": payload.sha256,
                "payloadComplete": payload.complete,
            },
        )
    elif isinstance(node.value, np.ndarray):
        result.update({"length": len(node.value), "dtype": str(node.value.dtype)})
    classes = ss.class_names(node)
    if classes:
        result["classes"] = list(classes)
    attrs = _attrs(node)
    if "levels" in attrs:
        try:
            result["levels"] = list(ss.string_values(attrs["levels"]))
        except ss.SeuratStreamError:
            result["levels"] = "unmaterialized"
    return result


def _matrix_row_identifiers(candidate: DgcCandidate) -> tuple[str, ...]:
    dimnames = ss.dereference(candidate.slots["Dimnames"])
    if dimnames.kind is not ss.RType.VEC or not isinstance(dimnames.value, list):
        raise ss.SeuratStreamError("dgCMatrix Dimnames must be a materialized list")
    if len(dimnames.value) != 2:
        raise ss.SeuratStreamError(
            "dgCMatrix Dimnames must have row and column members"
        )
    rows = ss.string_values(dimnames.value[0])
    if len(rows) != candidate.rows or len(set(rows)) != len(rows):
        raise ss.SeuratStreamError(
            "dgCMatrix row identifiers must be complete and unique"
        )
    return rows


def audit_raw_rna_row_identifiers(candidate: DgcCandidate) -> dict[str, object]:
    """Report exact raw rows and lexical flags without changing the query panel."""
    rows = _matrix_row_identifiers(candidate)
    patterns = {
        "barcodePrefix": re.compile(r"(?i)^bc[-_]"),
        "barcodeWord": re.compile(r"(?i)barcode"),
        "reporterWord": re.compile(r"(?i)reporter|transgene"),
        "fluorescentReporter": re.compile(r"(?i)^(?:e?gfp|mcherry|tdtomato)$"),
    }
    flagged = {
        name: [identifier for identifier in rows if pattern.search(identifier)]
        for name, pattern in patterns.items()
    }
    return {
        "allIdentifiers": list(rows),
        "count": len(rows),
        "orderedSha256": hashlib.sha256(
            ("\n".join(rows) + "\n").encode("utf-8"),
        ).hexdigest(),
        "lexicalFlags": flagged,
        "nativeUra3ExactRows": [
            identifier for identifier in rows if identifier == "URA3"
        ],
        "decision": "inspection only; no row removed or declared native/artificial",
    }


def discover_structure(
    root: ss.Node,
) -> tuple[dict[str, object], list[DgcCandidate], list[DataFrameCandidate]]:
    """Discover named S4/list/data-frame structure without molecular values."""
    s4_nodes: list[dict[str, object]] = []
    frames: list[dict[str, object]] = []
    candidates: list[DgcCandidate] = []
    frame_candidates: list[DataFrameCandidate] = []
    visited: set[int] = set()

    def walk(node: ss.Node, path: tuple[str, ...]) -> None:
        node = ss.dereference(node)
        if id(node) in visited:
            return
        visited.add(id(node))
        classes = ss.class_names(node)
        attrs = _attrs(node)
        if node.kind is ss.RType.S4:
            slot_names = tuple(name for name in attrs if name != "class")
            s4_nodes.append(
                {
                    "path": list(path),
                    "classes": list(classes),
                    "slots": list(slot_names),
                },
            )
            if "dgCMatrix" in classes:
                rows, columns, nnz = ss.validate_dgc_slots(attrs)
                candidates.append(DgcCandidate(path, attrs, rows, columns, nnz))
            for name in slot_names:
                walk(attrs[name], path + (name,))
            return
        if "data.frame" in classes and node.kind is ss.RType.VEC:
            names_node = attrs.get("names")
            names = ss.string_values(names_node) if names_node is not None else ()
            values = node.value
            if not isinstance(values, list) or len(values) != len(names):
                raise ss.SeuratStreamError("data.frame columns/names mismatch")
            frames.append(
                {
                    "path": list(path),
                    "columns": [
                        {"name": name, **_node_descriptor(value)}
                        for name, value in zip(names, values, strict=True)
                    ],
                    "rowNames": (
                        None
                        if "row.names" not in attrs
                        else _node_descriptor(attrs["row.names"])
                    ),
                },
            )
            frame_candidates.append(
                DataFrameCandidate(
                    path,
                    dict(zip(names, values, strict=True)),
                    attrs.get("row.names"),
                ),
            )
            for name, value in zip(names, values, strict=True):
                walk(value, path + (name,))
            return
        if node.kind in {
            ss.RType.LIST,
            ss.RType.LANG,
            ss.RType.CLO,
            ss.RType.PROM,
            ss.RType.DOT,
        }:
            car, cdr = node.value
            name = ss._symbol_text(node.tag) if node.tag is not None else "car"
            walk(car, path + (name,))
            if isinstance(cdr, ss.Node) and cdr.kind not in {
                ss.RType.NIL,
                ss.RType.NILVALUE,
            }:
                walk(cdr, path)
        elif node.kind in {ss.RType.VEC, ss.RType.EXPR} and isinstance(
            node.value, list
        ):
            names_node = attrs.get("names")
            names = ss.string_values(names_node) if names_node is not None else ()
            for index, value in enumerate(node.value):
                name = names[index] if index < len(names) else f"[{index}]"
                walk(value, path + (name,))

    walk(root, ())
    report = {"s4Objects": s4_nodes, "dataFrames": frames}
    return report, candidates, frame_candidates


def inventory_source(
    path: Path,
    *,
    expected_bytes: int,
    expected_md5: str,
    expected_sha256: str,
    max_seconds: float = 900,
    max_rss_bytes: int = 6 << 30,
) -> tuple[dict[str, object], list[DgcCandidate], ss.Inventory]:
    """Verify and inventory a complete source; never decode large count vectors."""
    started = time.monotonic()
    source = verify_source(
        path,
        expected_bytes=expected_bytes,
        expected_md5=expected_md5,
        expected_sha256=expected_sha256,
    )
    elapsed_hash = time.monotonic() - started
    remaining = max_seconds - elapsed_hash
    if remaining <= 0:
        raise ss.SeuratStreamError("source hashing exhausted inventory time bound")
    inventory = ss.inspect_rdata(
        path,
        materialize_limit=4096,
        materialize_atomic_names={"p", "Dim"},
        materialize_string_suffixes={("Dimnames", "[0]"), ("row.names",)},
        max_materialized_bytes=256 << 20,
        max_rss_bytes=max_rss_bytes,
        max_seconds=remaining,
    )
    if not inventory.complete or inventory.root is None:
        raise ss.SeuratStreamError("complete source produced an incomplete inventory")
    structure, candidates, _ = discover_structure(inventory.root)
    evidence_path = (
        ROOT
        / "data/sources/nadal-ribelles-2025-yeast-metadata-v1/biostudies-E-MTAB-14004.json"
    )
    report = {
        "schema": SCHEMA,
        "source": source,
        "runtimeSeconds": time.monotonic() - started,
        "limits": {"seconds": max_seconds, "rssBytes": max_rss_bytes},
        "rSerialization": {
            "formatVersion": inventory.format_version,
            "writerVersion": inventory.writer_version,
            "minimumVersion": inventory.minimum_version,
            "encoding": inventory.encoding,
            "decompressedBytes": inventory.decompressed_bytes,
        },
        **structure,
        "sparseCandidates": [
            {
                "path": list(candidate.semantic_path),
                "rows": candidate.rows,
                "columns": candidate.columns,
                "nnz": candidate.nnz,
                "admissibleRawRnaCountsPath": ss.is_admissible_rna_counts_path(
                    candidate.semantic_path,
                ),
                "i": _node_descriptor(candidate.slots["i"]),
                "p": _node_descriptor(candidate.slots["p"]),
                "x": _node_descriptor(candidate.slots["x"]),
                "rawRnaRowIdentifierAudit": (
                    audit_raw_rna_row_identifiers(candidate)
                    if ss.is_admissible_rna_counts_path(candidate.semantic_path)
                    else None
                ),
            }
            for candidate in candidates
        ],
        "interpretation": {
            "controlsAssigned": False,
            "contextsAssigned": False,
            "genotypesAssigned": False,
            "protectedRolesAssigned": False,
            "countValuesDecoded": False,
            "normalizedAssaysEligible": False,
        },
        "sourceCodeEvidence": {
            "path": str(evidence_path.resolve()),
            "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            "statement": (
                "BioStudies methods state that the sacCer3 reference was augmented with "
                "artificial bc-<systematic-name> chromosomes for targeted genotype "
                "barcodes, and that genotype barcodes were removed from the reference "
                "genome during standard Seurat normalization preparation. The actual "
                "raw RNA row roster is audited independently above."
            ),
            "nativeUra3Policy": (
                "An exact ordinary URA3 row is reported separately and is never removed "
                "merely because targeted amplification used the URA3 transcript."
            ),
        },
    }
    return report, candidates, inventory


class _MemmapSink:
    def __init__(self, array: np.memmap):
        self.array = array
        self.cursor = 0

    def __call__(self, _source_offset: int, values: np.ndarray) -> None:
        stop = self.cursor + len(values)
        self.array[self.cursor : stop] = values
        self.cursor = stop


def _payload_from_slot(slot: ss.Node, name: str) -> ss.Payload:
    value = ss.dereference(slot).value
    if not isinstance(value, ss.Payload) or not value.complete:
        raise ss.SeuratStreamError(f"{name} must be a complete skipped payload")
    return value


def write_selected_csc(
    source_path: Path,
    candidate: DgcCandidate,
    selection: FrozenColumnSelection,
    output_dir: Path,
    *,
    max_seconds: float = 900,
    max_rss_bytes: int = 6 << 30,
) -> dict[str, object]:
    """Replay one raw-RNA dgCMatrix into bounded CSC arrays for frozen columns."""
    if not ss.is_admissible_rna_counts_path(candidate.semantic_path):
        raise ss.SeuratStreamError("candidate is not an RNA/counts path")
    columns = np.asarray(selection.columns, dtype=np.int64)
    if columns.ndim != 1 or len(columns) == 0:
        raise ss.SeuratStreamError("selected columns must be a nonempty vector")
    if (
        np.any(np.diff(columns) <= 0)
        or columns[0] < 0
        or columns[-1] >= candidate.columns
    ):
        raise ss.SeuratStreamError(
            "selected columns must be sorted, unique and in range"
        )
    p = ss.dereference(candidate.slots["p"]).value
    if not isinstance(p, np.ndarray):
        raise ss.SeuratStreamError("candidate p must be materialized by metadata pass")
    column_ranges: list[tuple[int, int]] = []
    start = previous = int(columns[0])
    for column in columns[1:]:
        column = int(column)
        if column != previous + 1:
            column_ranges.append((start, previous + 1))
            start = column
        previous = column
    column_ranges.append((start, previous + 1))
    value_ranges = ss.plan_csc_payload_ranges(p, column_ranges)
    counts = np.diff(p)[columns]
    output_p = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    output_dir.mkdir(parents=True, exist_ok=False)
    output_i = np.lib.format.open_memmap(
        output_dir / "i.npy",
        mode="w+",
        dtype=np.int32,
        shape=(int(output_p[-1]),),
    )
    output_x = np.lib.format.open_memmap(
        output_dir / "x.npy",
        mode="w+",
        dtype=np.float64,
        shape=(int(output_p[-1]),),
    )
    i_sink, x_sink = _MemmapSink(output_i), _MemmapSink(output_x)
    i_payload = _payload_from_slot(candidate.slots["i"], "i")
    x_payload = _payload_from_slot(candidate.slots["x"], "x")
    replay = ss.inspect_rdata(
        source_path,
        selected={i_payload.path: value_ranges, x_payload.path: value_ranges},
        payload_sinks={i_payload.path: i_sink, x_payload.path: x_sink},
        materialize_atomic_names={"p", "Dim"},
        max_materialized_bytes=256 << 20,
        max_rss_bytes=max_rss_bytes,
        max_seconds=max_seconds,
    )
    if (
        not replay.complete
        or i_sink.cursor != output_p[-1]
        or x_sink.cursor != output_p[-1]
    ):
        raise ss.SeuratStreamError("selected CSC replay did not complete exactly")
    repeated = {payload.path: payload for payload in replay.payloads}
    if repeated[i_payload.path].sha256 != i_payload.sha256:
        raise ss.SeuratStreamError("i payload changed between inventory and extraction")
    if repeated[x_payload.path].sha256 != x_payload.sha256:
        raise ss.SeuratStreamError("x payload changed between inventory and extraction")
    output_i.flush()
    output_x.flush()
    np.save(output_dir / "p.npy", output_p)
    np.save(output_dir / "source_columns.npy", columns)
    report = {
        "schema": "slp.yeast-raw-rna-selected-csc/v1",
        "sourcePath": str(source_path.resolve()),
        "matrixPath": list(candidate.semantic_path),
        "shape": [candidate.rows, len(columns)],
        "nnz": int(output_p[-1]),
        "selectionProvenance": "caller-supplied metadata-only frozen column allowlist",
        "selectionSha256": selection.sha256,
        "selectionCounts": {
            "controls": selection.control_count,
            "pretrainActions": selection.pretrain_action_count,
            "excludedHeldActions": selection.excluded_held_count,
            "excludedUnmappedActions": selection.excluded_unmapped_count,
        },
        "globalHeldDomain": HELD_DOMAIN.rstrip(b"\x00").decode(),
        "normalizationApplied": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def write_selected_metadata(
    source_path: Path,
    frame: DataFrameCandidate,
    field_names: tuple[str, ...],
    output_dir: Path,
    *,
    max_seconds: float = 900,
    max_rss_bytes: int = 6 << 30,
) -> dict[str, object]:
    """Replay explicit metadata fields without assigning their biological meaning."""
    if not field_names or len(set(field_names)) != len(field_names):
        raise ss.SeuratStreamError("metadata field names must be nonempty and unique")
    requested_nodes = dict(frame.columns)
    if frame.row_names is not None:
        requested_nodes["__row_names__"] = frame.row_names
    missing = set(field_names) - requested_nodes.keys()
    if missing:
        raise ss.SeuratStreamError(
            f"metadata fields absent from frame: {sorted(missing)}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    selected: dict[tuple[str, ...], tuple[tuple[int, int], ...]] = {}
    sinks: dict[tuple[str, ...], _MemmapSink] = {}
    string_paths: set[tuple[str, ...]] = set()
    direct: dict[str, np.ndarray] = {}
    direct_observed: dict[str, np.ndarray] = {}
    for name in field_names:
        node = ss.dereference(requested_nodes[name])
        if isinstance(node.value, np.ndarray):
            direct[name] = node.value
            continue
        if node.kind is ss.RType.STR and isinstance(node.value, list):
            nullable = ss.nullable_string_values(node)
            direct_observed[name] = np.asarray(
                [value is not None for value in nullable],
                dtype=np.bool_,
            )
            direct[name] = np.asarray(
                ["" if value is None else value for value in nullable],
            )
            continue
        payload = _payload(node.value)
        if payload is None or not payload.complete:
            raise ss.SeuratStreamError(f"metadata field {name} is not extractable")
        if payload.kind == "STR":
            string_paths.add(payload.path)
            continue
        dtype = {"INT": np.int32, "LGL": np.int32, "REAL": np.float64}.get(
            payload.kind,
        )
        if dtype is None:
            raise ss.SeuratStreamError(
                f"unsupported metadata field type {payload.kind}"
            )
        array = np.lib.format.open_memmap(
            output_dir / f"{name}.npy",
            mode="w+",
            dtype=dtype,
            shape=(payload.length,),
        )
        selected[payload.path] = ((0, payload.length),)
        sinks[payload.path] = _MemmapSink(array)
    replay = ss.inspect_rdata(
        source_path,
        selected=selected,
        payload_sinks=sinks,
        selected_string_paths=string_paths,
        materialize_atomic_names={"p", "Dim"},
        max_materialized_bytes=256 << 20,
        max_rss_bytes=max_rss_bytes,
        max_seconds=max_seconds,
    )
    if not replay.complete or replay.root is None:
        raise ss.SeuratStreamError("metadata replay did not complete")
    _, _, repeated_frames = discover_structure(replay.root)
    repeated = next(
        (
            candidate
            for candidate in repeated_frames
            if candidate.semantic_path == frame.semantic_path
        ),
        None,
    )
    if repeated is None:
        raise ss.SeuratStreamError("metadata frame path changed during replay")
    lengths: dict[str, int] = {}
    for name in field_names:
        if name in direct:
            np.save(output_dir / f"{name}.npy", direct[name])
            if name in direct_observed:
                np.save(output_dir / f"{name}_observed.npy", direct_observed[name])
            lengths[name] = len(direct[name])
            continue
        original = ss.dereference(requested_nodes[name])
        payload = _payload(original.value)
        assert payload is not None
        if payload.kind == "STR":
            repeated_node = (
                repeated.row_names
                if name == "__row_names__"
                else repeated.columns[name]
            )
            if repeated_node is None:
                raise ss.SeuratStreamError(
                    "metadata row names disappeared during replay"
                )
            nullable = ss.nullable_string_values(repeated_node)
            observed = np.asarray(
                [value is not None for value in nullable],
                dtype=np.bool_,
            )
            values = np.asarray(["" if value is None else value for value in nullable])
            np.save(output_dir / f"{name}.npy", values, allow_pickle=False)
            np.save(output_dir / f"{name}_observed.npy", observed)
            lengths[name] = len(values)
        else:
            sink = sinks[payload.path]
            sink.array.flush()
            if sink.cursor != payload.length:
                raise ss.SeuratStreamError(
                    f"metadata field {name} replay length mismatch"
                )
            lengths[name] = sink.cursor
    if len(set(lengths.values())) != 1:
        raise ss.SeuratStreamError("selected metadata fields have unequal row counts")
    report = {
        "schema": "slp.yeast-seurat-selected-metadata/v1",
        "sourcePath": str(source_path.resolve()),
        "framePath": list(frame.semantic_path),
        "fields": list(field_names),
        "stringMissingness": "empty placeholder plus <field>_observed.npy mask",
        "rows": next(iter(lengths.values())),
        "biologicalRolesAssigned": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _vocabulary_summary(
    values: np.ndarray,
    observed: np.ndarray | None = None,
    *,
    full_limit: int = 5000,
) -> dict[str, object]:
    text = np.asarray(values).astype(str)
    if observed is None:
        observed = np.ones(len(text), dtype=np.bool_)
    observed = np.asarray(observed, dtype=np.bool_)
    if observed.shape != text.shape:
        raise ss.SeuratStreamError("metadata observed mask shape mismatch")
    text = text[observed]
    unique, counts = np.unique(text, return_counts=True)
    order = np.argsort(unique)
    unique, counts = unique[order], counts[order]
    digest = hashlib.sha256()
    for value, count in zip(unique, counts, strict=True):
        digest.update(f"{value}\t{int(count)}\n".encode())
    result: dict[str, object] = {
        "rows": len(observed),
        "observed": int(np.count_nonzero(observed)),
        "missing": int(np.count_nonzero(~observed)),
        "unique": len(unique),
        "vocabularyCountSha256": digest.hexdigest(),
        "first": unique[:20].tolist(),
        "last": unique[-20:].tolist(),
    }
    if len(unique) <= full_limit:
        result["allValuesAndCounts"] = [
            [value, int(count)]
            for value, count in zip(unique.tolist(), counts.tolist())
        ]
    return result


def replay_metadata_vocabulary(
    source_path: Path,
    output_dir: Path,
    *,
    field_names: tuple[str, ...],
    max_seconds_per_replay: float = 900,
    max_rss_bytes: int = 6 << 30,
) -> dict[str, object]:
    """Discover then persist metadata-only fields for every exact meta.data frame."""
    if output_dir.exists():
        raise ss.SeuratStreamError(f"refusing to overwrite {output_dir}")
    inventory = ss.inspect_rdata(
        source_path,
        materialize_limit=4096,
        materialize_atomic_names={"p", "Dim"},
        materialize_string_suffixes={("Dimnames", "[0]"), ("row.names",)},
        max_materialized_bytes=256 << 20,
        max_rss_bytes=max_rss_bytes,
        max_seconds=max_seconds_per_replay,
    )
    if not inventory.complete or inventory.root is None:
        raise ss.SeuratStreamError("metadata discovery replay did not complete")
    _, _, frames = discover_structure(inventory.root)
    frames = [frame for frame in frames if frame.semantic_path[-1:] == ("meta.data",)]
    if not frames:
        raise ss.SeuratStreamError("no exact meta.data frame discovered")
    output_dir.mkdir(parents=True)
    summaries: list[dict[str, object]] = []
    for index, frame in enumerate(frames):
        frame_dir = output_dir / f"frame-{index}"
        write_selected_metadata(
            source_path,
            frame,
            field_names,
            frame_dir,
            max_seconds=max_seconds_per_replay,
            max_rss_bytes=max_rss_bytes,
        )
        summaries.append(
            {
                "path": list(frame.semantic_path),
                "fields": {
                    name: _vocabulary_summary(
                        np.load(frame_dir / f"{name}.npy"),
                        (
                            np.load(frame_dir / f"{name}_observed.npy")
                            if (frame_dir / f"{name}_observed.npy").exists()
                            else None
                        ),
                    )
                    for name in field_names
                },
            },
        )
    report = {
        "schema": "slp.yeast-seurat-metadata-vocabulary/v1",
        "sourcePath": str(source_path.resolve()),
        "frames": summaries,
        "interpretation": (
            "Exact source metadata values and frequencies only; labels are not yet "
            "assigned as controls, interventions, contexts, or stable genes."
        ),
    }
    (output_dir / "vocabulary-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def freeze_metadata_selections(
    metadata_dir: Path,
    vocabulary_report: dict[str, object],
    current_action_map_path: Path,
    row_mapping_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Freeze metadata-only train/validation/control columns and query mapping."""
    if output_dir.exists():
        raise ss.SeuratStreamError(f"refusing to overwrite {output_dir}")
    action_map = load_exact_current_action_map(current_action_map_path)
    query_map = build_strict_query_map(row_mapping_path)
    output_dir.mkdir(parents=True)
    query_path = output_dir / "query-map.npz"
    np.savez_compressed(query_path, **query_map)
    frames_report: list[dict[str, object]] = []
    frames = vocabulary_report["frames"]
    for frame_index, frame_report in enumerate(frames):
        frame_dir = metadata_dir / f"frame-{frame_index}"
        assignment = np.load(frame_dir / "assignment_consensus2.npy")
        kogene = np.load(frame_dir / "kogene.npy")
        condition = np.load(frame_dir / "condition.npy")
        batch = np.load(frame_dir / "batch.npy")
        clone = np.load(frame_dir / "clone.npy")
        clone_observed = np.load(frame_dir / "clone_observed.npy")
        barcodes = np.load(frame_dir / "__row_names__.npy")
        n = len(assignment)
        if any(
            len(value) != n for value in (kogene, condition, batch, clone, barcodes)
        ):
            raise ss.SeuratStreamError("metadata selection fields differ in length")
        exact_context = frame_report["path"][1]
        if set(condition.tolist()) != {exact_context}:
            raise ss.SeuratStreamError(
                "condition values disagree with source list label"
            )
        control = assignment == "WT"
        if not np.array_equal(control, kogene == "WT"):
            raise ss.SeuratStreamError("assignment/kogene WT flags disagree")
        mutant = ~control
        assignment_prefixed = np.char.startswith(assignment, "bc-")
        invalid_assignment = mutant & ~assignment_prefixed
        stable = np.full(n, "", dtype="<U14")
        for source_name in np.unique(kogene[mutant]):
            mapped = action_map.get(str(source_name))
            if mapped is not None:
                stable[kogene == source_name] = mapped
        mapped = stable != ""
        suffix_disagreement = (
            mutant & assignment_prefixed & (np.char.lstrip(assignment, "bc-") != kogene)
        )
        protected = np.full(n, "control", dtype="<U20")
        development = np.full(n, "control", dtype="<U10")
        for stable_id in np.unique(stable[mapped]):
            mask = stable == stable_id
            protected[mask] = protected_role(str(stable_id))
            development[mask] = development_role(str(stable_id))
        eligible = control | (
            mutant
            & assignment_prefixed
            & mapped
            & (protected == "pretrain")
            & np.isin(development, ["train", "validation"])
        )
        selected_columns = np.flatnonzero(eligible)
        digest = hashlib.sha256()
        for column in selected_columns:
            label = "CONTROL" if control[column] else stable[column]
            digest.update(f"{int(column)}\t{label}\t{development[column]}\n".encode())
        selection = FrozenColumnSelection(
            selected_columns,
            digest.hexdigest(),
            int(np.count_nonzero(control)),
            int(np.count_nonzero(eligible & mutant)),
            int(np.count_nonzero(mutant & mapped & (protected != "pretrain"))),
            int(np.count_nonzero(mutant & ~mapped)),
        )
        output_path = output_dir / f"frame-{frame_index}-selection.npz"
        np.savez_compressed(
            output_path,
            source_columns=selected_columns,
            barcode=barcodes[eligible],
            condition=condition[eligible],
            batch=batch[eligible],
            clone=clone[eligible],
            clone_observed=clone_observed[eligible],
            assignment_consensus2=assignment[eligible],
            kogene=kogene[eligible],
            stable_action_id=stable[eligible],
            is_control=control[eligible],
            development_role=development[eligible],
        )
        gene_batch: list[list[object]] = []
        selected_mutants = eligible & mutant
        pairs, pair_counts = np.unique(
            np.stack([stable[selected_mutants], batch[selected_mutants]], axis=1),
            axis=0,
            return_counts=True,
        )
        for (stable_id, batch_id), pair_count in zip(
            pairs,
            pair_counts,
            strict=True,
        ):
            gene_batch.append(
                [
                    stable_id,
                    development_role(str(stable_id)),
                    batch_id,
                    int(pair_count),
                ],
            )
        unique_disagreements, disagreement_counts = (
            np.unique(
                np.stack(
                    [assignment[suffix_disagreement], kogene[suffix_disagreement]],
                    axis=1,
                ),
                axis=0,
                return_counts=True,
            )
            if np.any(suffix_disagreement)
            else (np.empty((0, 2), dtype=str), np.empty(0, dtype=int))
        )
        frames_report.append(
            {
                "frameIndex": frame_index,
                "context": exact_context,
                "sourceCells": n,
                "selectionSha256": selection.sha256,
                "selectedCells": len(selected_columns),
                "controls": selection.control_count,
                "selectedMutantCells": selection.pretrain_action_count,
                "selectedTrainCells": int(
                    np.count_nonzero(eligible & (development == "train"))
                ),
                "selectedValidationCells": int(
                    np.count_nonzero(eligible & (development == "validation")),
                ),
                "excludedDevelopmentTestCells": int(
                    np.count_nonzero(
                        mutant
                        & mapped
                        & (protected == "pretrain")
                        & (development == "test"),
                    ),
                ),
                "excludedProtectedCells": selection.excluded_held_count,
                "exactMapFailureCells": selection.excluded_unmapped_count,
                "invalidAssignmentPrefixCells": int(
                    np.count_nonzero(invalid_assignment)
                ),
                "assignmentSuffixDisagreementCells": int(
                    np.count_nonzero(suffix_disagreement),
                ),
                "assignmentSuffixDisagreements": [
                    [left, right, int(count)]
                    for (left, right), count in zip(
                        unique_disagreements.tolist(),
                        disagreement_counts.tolist(),
                        strict=True,
                    )
                ],
                "wtPerBatch": [
                    [batch_id, int(np.count_nonzero(control & (batch == batch_id)))]
                    for batch_id in np.unique(batch)
                ],
                "geneBatchCellCounts": gene_batch,
                "selectionArtifact": str(output_path.resolve()),
            },
        )
    report = {
        "schema": "slp.yeast-seurat-metadata-selection/v1",
        "rules": {
            "control": "assignment_consensus2 == WT and kogene == WT",
            "mutant": "assignment_consensus2 has bc- prefix and kogene has exact current SGD mapping",
            "protected": "global pretrain role only",
            "development": "train or validation; development test removed before count decoding",
            "cloneModelInput": False,
        },
        "currentActionMap": {
            "path": str(current_action_map_path.resolve()),
            "sha256": hashlib.sha256(current_action_map_path.read_bytes()).hexdigest(),
        },
        "rowMapping": {
            "path": str(row_mapping_path.resolve()),
            "sha256": hashlib.sha256(row_mapping_path.read_bytes()).hexdigest(),
        },
        "queryMap": {
            "path": str(query_path.resolve()),
            "sha256": hashlib.sha256(query_path.read_bytes()).hexdigest(),
            "strictQueries": len(query_map["query_ids"]),
            "denominatorRows": int(query_map["denominator_mask"].sum()),
        },
        "frames": frames_report,
    }
    (output_dir / "selection-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--expected-md5", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--max-rss-gib", type=float, default=6)
    args = parser.parse_args()
    if args.output.exists():
        raise ss.SeuratStreamError(f"refusing to overwrite {args.output}")
    report, _, _ = inventory_source(
        args.source,
        expected_bytes=args.expected_bytes,
        expected_md5=args.expected_md5,
        expected_sha256=args.expected_sha256,
        max_seconds=args.max_seconds,
        max_rss_bytes=int(args.max_rss_gib * (1 << 30)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
