"""Build author-normalized human Perturb-seq development artifacts, v2."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

TAXON = 9606
SEED = 731
MAX_SOURCE_BYTES = 256 * 1024 * 1024
MAX_ROWS = 10_000
MAX_GENES = 20_000
ENSG_RE = re.compile(r"^ENSG[0-9]+$")
ACTION_RE = re.compile(r"_(ENSG[0-9]+)$")
CONTROL_SUFFIX = "_non-targeting_non-targeting_non-targeting"
CONTEXT_IDS = (
    "replogle-2022-k562-essential-day-6",
    "replogle-2022-rpe1-essential-day-7",
)
VALUE_SPACE = "author-per-gemgroup-core-control-z-score-pseudobulk-mean-v1"
CONTEXT_VALUE_SPACE = "log2-1p-cp10k-shared-panel-raw-core-control-mean-v1"
NUM_CELLS_ROLE = "likelihood-only-measurement-precision-not-predictor"


class HumanDataV2Error(ValueError):
    """A normalized source or derived artifact violates the v2 contract."""


@dataclass(frozen=True)
class SourceSpec:
    context_id: str
    filename: str
    bytes: int
    sha256: str
    upstream_md5: str
    figshare_file_id: int


PRODUCTION_SOURCES = (
    SourceSpec(
        CONTEXT_IDS[0],
        "K562_essential_normalized_bulk_01.h5ad",
        79_766_954,
        "c1ca6456c9c9f1aa2b02c496eb64d1dc3e6a852edbd744d682b8d2c95fd36829",
        "30496767641cd2e660ee6ecb5baee132",
        35_780_870,
    ),
    SourceSpec(
        CONTEXT_IDS[1],
        "rpe1_normalized_bulk_01.h5ad",
        95_350_546,
        "a3c5bfd0f15d63938bc80c9b8874b9cd761e3a23caf5ffe7966bae4e887ec89d",
        "6f1e7d6a09e2f869759e3c4526b7f171",
        35_775_512,
    ),
)
RAW_CONTEXT_SOURCES = (
    SourceSpec(
        CONTEXT_IDS[0],
        "K562_essential_raw_bulk_01.h5ad",
        79_766_954,
        "80de95e54fcbca0e0537d569b43ec92fde6bd0482801505504baebe3118dcadf",
        "8321d5d3ffc99db2a5c71edca4189735",
        35_773_070,
    ),
    SourceSpec(
        CONTEXT_IDS[1],
        "rpe1_raw_bulk_01.h5ad",
        95_350_546,
        "603c655f1cfa41d649baf3ae63fca224cc11f297e40d4ed59d390b1e8d2e2db2",
        "74765fa87635467a869ea972356ae0e7",
        35_775_581,
    ),
)


def _hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, spec: SourceSpec) -> Path:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file() or resolved.name != spec.filename:
        raise HumanDataV2Error(f"invalid normalized source path: {spec.filename}")
    if not (0 < resolved.stat().st_size <= MAX_SOURCE_BYTES):
        raise HumanDataV2Error(f"normalized source exceeds byte bound: {spec.filename}")
    if resolved.stat().st_size != spec.bytes or _hash(resolved) != spec.sha256:
        raise HumanDataV2Error(f"normalized source size or SHA-256 drift: {spec.filename}")
    if _hash(resolved, "md5") != spec.upstream_md5:
        raise HumanDataV2Error(f"normalized source upstream MD5 drift: {spec.filename}")
    return resolved


def _strings(dataset: h5py.Dataset, label: str) -> tuple[str, ...]:
    if dataset.ndim != 1 or len(dataset) > MAX_GENES:
        raise HumanDataV2Error(f"{label} exceeds its bound")
    values = tuple(item.decode() if isinstance(item, bytes) else str(item) for item in dataset[:])
    if any(not item or item != item.strip() for item in values):
        raise HumanDataV2Error(f"{label} contains an invalid string")
    return values


def _split_name(action_id: str, seed: int = SEED) -> str:
    payload = f"slp11-development-v1|{seed}|{TAXON}|{action_id}".encode()
    bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def _read_source(path: Path, spec: SourceSpec) -> dict[str, object]:
    with h5py.File(_verify(path, spec), "r") as handle:
        if not {"X", "obs", "var"}.issubset(handle):
            raise HumanDataV2Error(f"unexpected AnnData layout: {spec.filename}")
        required_obs = {"gene_transcript", "num_cells_filtered", "core_control"}
        if not required_obs.issubset(handle["obs"]) or "gene_id" not in handle["var"]:
            raise HumanDataV2Error(f"normalized source metadata is incomplete: {spec.filename}")
        rows, genes = handle["X"].shape
        if not (0 < rows <= MAX_ROWS and 0 < genes <= MAX_GENES):
            raise HumanDataV2Error(f"normalized source dimensions exceed bounds: {spec.filename}")
        record_ids = _strings(handle["obs/gene_transcript"], "gene_transcript")
        gene_ids = _strings(handle["var/gene_id"], "gene_id")
        matrix = np.asarray(handle["X"][:], dtype=np.float32)
        num_cells = np.asarray(handle["obs/num_cells_filtered"][:], dtype=np.float32)
        core = np.asarray(handle["obs/core_control"][:], dtype=np.bool_)
    if len(record_ids) != rows or len(set(record_ids)) != rows:
        raise HumanDataV2Error(f"normalized source record IDs are not unique: {spec.filename}")
    if len(gene_ids) != genes or len(set(gene_ids)) != genes:
        raise HumanDataV2Error(f"normalized source gene IDs are not unique: {spec.filename}")
    if any(ENSG_RE.fullmatch(gene) is None for gene in gene_ids):
        raise HumanDataV2Error(f"normalized source gene IDs are not Ensembl IDs: {spec.filename}")
    if matrix.shape != (rows, genes) or num_cells.shape != (rows,) or core.shape != (rows,):
        raise HumanDataV2Error(f"normalized source array shape drift: {spec.filename}")
    controls = np.asarray([record.endswith(CONTROL_SUFFIX) for record in record_ids])
    core_controls = controls & core
    if not core_controls.any() or np.any(core & ~controls):
        raise HumanDataV2Error(f"core-control annotations are inconsistent: {spec.filename}")
    actions: list[str | None] = []
    unresolved: list[str] = []
    for record in record_ids:
        match = ACTION_RE.search(record)
        actions.append(match.group(1) if match is not None else None)
        if match is None and not record.endswith(CONTROL_SUFFIX):
            unresolved.append(record)
    return {
        "spec": spec,
        "record_ids": record_ids,
        "gene_ids": gene_ids,
        "matrix": matrix,
        "num_cells": num_cells,
        "actions": tuple(actions),
        "core_controls": core_controls,
        "unresolved": tuple(unresolved),
    }


def _align(source: dict[str, object], query_ids: tuple[str, ...], context: int) -> dict[str, np.ndarray]:
    genes = source["gene_ids"]
    assert isinstance(genes, tuple)
    lookup = {gene: column for column, gene in enumerate(genes)}
    columns = np.asarray([lookup[gene] for gene in query_ids], dtype=np.int64)
    matrix = np.asarray(source["matrix"])[:, columns]
    observed = np.isfinite(matrix)
    stored = np.where(observed, matrix, 0.0).astype(np.float32)
    records = source["record_ids"]
    actions = source["actions"]
    num_cells = np.asarray(source["num_cells"])
    core_controls = np.asarray(source["core_controls"])
    assert isinstance(records, tuple) and isinstance(actions, tuple)
    known = np.asarray([action is not None for action in actions])
    if np.any(~np.isfinite(num_cells[known | core_controls])) or np.any(
        num_cells[known | core_controls] <= 0
    ):
        raise HumanDataV2Error("num_cells_filtered must be finite and positive")
    if np.any(~observed[known].any(axis=1)) or np.any(~observed[core_controls].any(axis=1)):
        raise HumanDataV2Error("a retained normalized profile has no finite readout")
    control_values = stored[core_controls]
    control_mask = observed[core_controls]
    counts = control_mask.sum(axis=0)
    if np.any(counts == 0):
        raise HumanDataV2Error("core controls lack support for a shared query")
    basal = np.where(control_mask, control_values, 0.0).sum(axis=0) / counts
    selected = np.flatnonzero(known)
    return {
        "targets": stored[selected],
        "observed": observed[selected],
        "action_ids": np.asarray([actions[row] for row in selected]),
        "record_ids": np.asarray(
            [f"{CONTEXT_IDS[context]}|{records[row]}" for row in selected]
        ),
        "context_index": np.full(len(selected), context, dtype=np.int64),
        "num_cells_filtered": num_cells[selected].astype(np.float32),
        "basal": basal.astype(np.float32),
        "control_targets": control_values,
        "control_observed": control_mask,
        "control_context_index": np.full(int(core_controls.sum()), context, dtype=np.int64),
        "control_num_cells_filtered": num_cells[core_controls].astype(np.float32),
        "control_record_ids": np.asarray(
            [f"{CONTEXT_IDS[context]}|{records[row]}" for row in np.flatnonzero(core_controls)]
        ),
        "control_core": np.ones(int(core_controls.sum()), dtype=np.bool_),
    }


def _raw_context_basal(
    path: Path,
    spec: SourceSpec,
    query_ids: tuple[str, ...],
    expected_core_record_ids: np.ndarray,
) -> np.ndarray:
    """Read only raw core-control rows and form measured context expression."""

    with h5py.File(_verify(path, spec), "r") as handle:
        records = _strings(handle["obs/gene_transcript"], "raw gene_transcript")
        genes = _strings(handle["var/gene_id"], "raw gene_id")
        core = np.asarray(handle["obs/core_control"][:], dtype=np.bool_)
        controls = np.asarray([record.endswith(CONTROL_SUFFIX) for record in records])
        rows = np.flatnonzero(core & controls)
        raw_record_ids = np.asarray([f"{spec.context_id}|{records[row]}" for row in rows])
        if not np.array_equal(raw_record_ids, expected_core_record_ids):
            raise HumanDataV2Error("raw and normalized core-control identities disagree")
        lookup = {gene: column for column, gene in enumerate(genes)}
        columns = np.asarray([lookup[gene] for gene in query_ids], dtype=np.int64)
        raw = np.asarray(handle["X"][rows, :], dtype=np.float64)[:, columns]
    if not np.isfinite(raw).all() or np.any(raw < 0):
        raise HumanDataV2Error("raw core-control expression must be finite and nonnegative")
    denominator = raw.sum(axis=1)
    if np.any(denominator <= 0):
        raise HumanDataV2Error("raw core-control expression has an all-zero row")
    transformed = np.log2(1.0 + 10_000.0 * raw / denominator[:, None])
    return transformed.mean(axis=0).astype(np.float32)


def _string_array(values: Sequence[str]) -> np.ndarray:
    return np.asarray(values, dtype=f"<U{max(1, *(len(item) for item in values))}")


def _bundle(
    combined: dict[str, np.ndarray],
    selection: np.ndarray,
    split_names: np.ndarray,
    query_ids: tuple[str, ...],
    basal: np.ndarray,
    context_basal_expression: np.ndarray,
    controls: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    selected_splits = split_names[selection]
    return {
        "targets": combined["targets"][selection],
        "observed": combined["observed"][selection],
        "action_ids": _string_array(combined["action_ids"][selection].tolist()),
        "query_ids": _string_array(query_ids),
        "context_index": combined["context_index"][selection],
        "context_ids": _string_array(CONTEXT_IDS),
        "basal_control": basal,
        "context_basal_expression": context_basal_expression,
        "context_value_space": np.asarray(CONTEXT_VALUE_SPACE),
        "record_ids": _string_array(combined["record_ids"][selection].tolist()),
        "num_cells_filtered": combined["num_cells_filtered"][selection],
        "target_value_space": np.asarray(VALUE_SPACE),
        "num_cells_role": np.asarray(NUM_CELLS_ROLE),
        "control_targets": controls["control_targets"],
        "control_observed": controls["control_observed"],
        "control_context_index": controls["control_context_index"],
        "control_num_cells_filtered": controls["control_num_cells_filtered"],
        "control_record_ids": _string_array(controls["control_record_ids"].tolist()),
        "control_core": controls["control_core"],
        "split_train": np.flatnonzero(selected_splits == "train").astype(np.int64),
        "split_validation": np.flatnonzero(selected_splits == "validation").astype(np.int64),
        "split_test": np.flatnonzero(selected_splits == "test").astype(np.int64),
    }


def _npy(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    payload = array if array.ndim == 0 else np.ascontiguousarray(array)
    np.save(stream, payload, allow_pickle=False)
    return stream.getvalue()


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, _npy(arrays[name]))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(output.getvalue())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def build_human_development_v2(
    k562_path: str | Path,
    rpe1_path: str | Path,
    destination: str | Path,
    *,
    source_specs: tuple[SourceSpec, SourceSpec] = PRODUCTION_SOURCES,
    raw_context_paths: tuple[str | Path, str | Path] | None = None,
    raw_context_specs: tuple[SourceSpec, SourceSpec] = RAW_CONTEXT_SOURCES,
    expected_query_count: int | None = 7_226,
) -> dict[str, object]:
    """Build distinct author-normalized development and routed test bundles."""

    if tuple(spec.context_id for spec in source_specs) != CONTEXT_IDS:
        raise HumanDataV2Error("source contexts do not match the frozen order")
    sources = tuple(
        _read_source(Path(path), spec)
        for path, spec in zip((k562_path, rpe1_path), source_specs)
    )
    query_ids = tuple(sorted(set(sources[0]["gene_ids"]) & set(sources[1]["gene_ids"])))
    if not query_ids or (expected_query_count is not None and len(query_ids) != expected_query_count):
        raise HumanDataV2Error("shared readout panel count drift")
    aligned = tuple(_align(source, query_ids, context) for context, source in enumerate(sources))
    if raw_context_paths is None:
        raw_context_paths = tuple(
            Path(path).parent / spec.filename
            for path, spec in zip((k562_path, rpe1_path), raw_context_specs)
        )
    if tuple(spec.context_id for spec in raw_context_specs) != CONTEXT_IDS:
        raise HumanDataV2Error("raw context source contexts do not match the frozen order")
    context_basal_expression = np.stack(
        [
            _raw_context_basal(
                Path(path),
                spec,
                query_ids,
                aligned[context]["control_record_ids"],
            )
            for context, (path, spec) in enumerate(zip(raw_context_paths, raw_context_specs))
        ]
    )
    row_fields = (
        "targets",
        "observed",
        "action_ids",
        "record_ids",
        "context_index",
        "num_cells_filtered",
    )
    control_fields = (
        "control_targets",
        "control_observed",
        "control_context_index",
        "control_num_cells_filtered",
        "control_record_ids",
        "control_core",
    )
    combined = {field: np.concatenate([item[field] for item in aligned]) for field in row_fields}
    controls = {field: np.concatenate([item[field] for item in aligned]) for field in control_fields}
    basal = np.stack([item["basal"] for item in aligned]).astype(np.float32)
    split_names = np.asarray([_split_name(str(action)) for action in combined["action_ids"]])
    development = np.flatnonzero(split_names != "test").astype(np.int64)
    test = np.flatnonzero(split_names == "test").astype(np.int64)
    if set(combined["action_ids"][development]) & set(combined["action_ids"][test]):
        raise HumanDataV2Error("held-gene routing overlap")
    for context in range(len(CONTEXT_IDS)):
        train = (combined["context_index"] == context) & (split_names == "train")
        if np.any(~combined["observed"][train].any(axis=0)):
            raise HumanDataV2Error("a context training split lacks query support")

    destination_path = Path(destination)
    development_path = destination_path / "replogle-k562-rpe1-author-normalized-development-v2.npz"
    test_path = destination_path / "replogle-k562-rpe1-author-normalized-test-only-v2.npz"
    _write_npz(
        development_path,
        _bundle(
            combined,
            development,
            split_names,
            query_ids,
            basal,
            context_basal_expression,
            controls,
        ),
    )
    _write_npz(
        test_path,
        _bundle(
            combined,
            test,
            split_names,
            query_ids,
            basal,
            context_basal_expression,
            controls,
        ),
    )
    manifest = {
        "schema": "slp.human-perturbation-author-normalized/v2",
        "status": "derived-development-and-routed-test-not-omf-admitted",
        "ncbiTaxon": TAXON,
        "valueSpace": VALUE_SPACE,
        "normalization": (
            "author normalized: per-cell UMI scaling then per-gemgroup/per-gene "
            "core-control z-score; X is the perturbation-population mean"
        ),
        "queryPanel": "exact sorted K562/RPE1 Ensembl intersection",
        "queryCount": len(query_ids),
        "numCellsRole": NUM_CELLS_ROLE,
        "contextValueSpace": CONTEXT_VALUE_SPACE,
        "contextBasalExpression": (
            "mean log2(1+10000*x/sum(shared panel)) across raw source core-control populations"
        ),
        "split": "sha256(slp11-development-v1|731|9606|ENSG), first 8 bytes mod 100",
        "counts": {
            "records": len(combined["action_ids"]),
            "development": len(development),
            "testOnly": len(test),
            "train": int(np.count_nonzero(split_names == "train")),
            "validation": int(np.count_nonzero(split_names == "validation")),
            "coreControls": len(controls["control_core"]),
            "coreControlsByContext": [len(item["control_core"]) for item in aligned],
            "unresolvedActionsQuarantined": [len(source["unresolved"]) for source in sources],
        },
        "sources": [
            {
                "contextId": spec.context_id,
                "path": f"data/sources/human/{spec.filename}",
                "figshareFileId": spec.figshare_file_id,
                "bytes": spec.bytes,
                "upstreamMd5": spec.upstream_md5,
                "sha256": spec.sha256,
            }
            for spec in source_specs
        ],
        "rawContextSources": [
            {
                "contextId": spec.context_id,
                "path": f"data/sources/human/{spec.filename}",
                "figshareFileId": spec.figshare_file_id,
                "bytes": spec.bytes,
                "upstreamMd5": spec.upstream_md5,
                "sha256": spec.sha256,
                "rowsRead": "source-marked core controls only",
            }
            for spec in raw_context_specs
        ],
        "outputs": {
            "development": {
                "path": development_path.name,
                "bytes": development_path.stat().st_size,
                "sha256": _hash(development_path),
                "contains": ["train", "validation"],
            },
            "testOnly": {
                "path": test_path.name,
                "bytes": test_path.stat().st_size,
                "sha256": _hash(test_path),
                "contains": ["test"],
                "access": "sealed until candidate and rule lock",
            },
        },
        "testOutcomesAnalyzed": False,
        "benchmarkAccessed": False,
    }
    manifest_path = destination_path / "manifest-v2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"manifest": manifest, "manifestPath": str(manifest_path), "manifestSha256": _hash(manifest_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k562", required=True)
    parser.add_argument("--rpe1", required=True)
    parser.add_argument("--raw-k562", required=True)
    parser.add_argument("--raw-rpe1", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            build_human_development_v2(
                args.k562,
                args.rpe1,
                args.output,
                raw_context_paths=(args.raw_k562, args.raw_rpe1),
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
