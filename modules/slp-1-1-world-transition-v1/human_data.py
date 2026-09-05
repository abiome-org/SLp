"""Build bounded fitting-only human Perturb-seq development artifacts.

The source matrices are public Replogle et al. per-perturbation mean-population
profiles.  They are fractional cell means, so this adapter uses a frozen
Gaussian target transform and never treats them as negative-binomial counts.
"""

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
TRANSFORM_ID = "slp-transform:log2-1p-cp10k-shared-panel-v1"
AUTHOR_CODE_COMMIT = "3b25109aeb9c0c2026bd70abd50304a0ad4e5395"


class HumanDataError(ValueError):
    """A source or derived artifact violates the bounded human-data contract."""


@dataclass(frozen=True)
class SourceSpec:
    context_id: str
    filename: str
    bytes: int
    sha256: str
    upstream_md5: str


PRODUCTION_SOURCES = (
    SourceSpec(
        CONTEXT_IDS[0],
        "K562_essential_raw_bulk_01.h5ad",
        79_766_954,
        "80de95e54fcbca0e0537d569b43ec92fde6bd0482801505504baebe3118dcadf",
        "8321d5d3ffc99db2a5c71edca4189735",
    ),
    SourceSpec(
        CONTEXT_IDS[1],
        "rpe1_raw_bulk_01.h5ad",
        95_350_546,
        "603c655f1cfa41d649baf3ae63fca224cc11f297e40d4ed59d390b1e8d2e2db2",
        "74765fa87635467a869ea972356ae0e7",
    ),
)


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source(path: Path, spec: SourceSpec) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise HumanDataError(f"source is missing: {path}") from error
    if path.is_symlink() or not resolved.is_file():
        raise HumanDataError(f"source must be a regular file: {spec.filename}")
    if resolved.name != spec.filename:
        raise HumanDataError(f"source filename drift: {spec.filename}")
    size = resolved.stat().st_size
    if size <= 0 or size > MAX_SOURCE_BYTES:
        raise HumanDataError(f"source exceeds its byte bound: {spec.filename}")
    if size != spec.bytes or _hash_file(resolved, "sha256") != spec.sha256:
        raise HumanDataError(f"source SHA-256 or size drift: {spec.filename}")
    if spec.upstream_md5 and _hash_file(resolved, "md5") != spec.upstream_md5:
        raise HumanDataError(f"source upstream MD5 drift: {spec.filename}")
    return resolved


def _decode_strings(dataset: h5py.Dataset, label: str) -> tuple[str, ...]:
    if dataset.ndim != 1 or len(dataset) > MAX_GENES:
        raise HumanDataError(f"{label} shape exceeds its bound")
    values: list[str] = []
    for raw in dataset[:]:
        value = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        if not value or value != value.strip():
            raise HumanDataError(f"{label} contains an invalid string")
        values.append(value)
    return tuple(values)


@dataclass(frozen=True)
class SourceMetadata:
    path: Path
    spec: SourceSpec
    rows: int
    genes: int
    record_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    action_ids: tuple[str | None, ...]
    control_mask: np.ndarray
    unresolved_ids: tuple[str, ...]


def _inspect_metadata(path: Path, spec: SourceSpec) -> SourceMetadata:
    try:
        with h5py.File(path, "r") as handle:
            if set(handle) != {"X", "obs", "var"} or not isinstance(
                handle["X"], h5py.Dataset
            ):
                raise HumanDataError(f"unexpected AnnData layout: {spec.filename}")
            rows, genes = handle["X"].shape
            if not (0 < rows <= MAX_ROWS and 0 < genes <= MAX_GENES):
                raise HumanDataError(f"source dimensions exceed bounds: {spec.filename}")
            if handle["X"].dtype != np.dtype("float32"):
                raise HumanDataError(f"X must be float32: {spec.filename}")
            if "gene_transcript" not in handle["obs"] or "gene_id" not in handle["var"]:
                raise HumanDataError(f"stable identity metadata is absent: {spec.filename}")
            record_ids = _decode_strings(
                handle["obs/gene_transcript"], f"{spec.filename}/obs/gene_transcript"
            )
            gene_ids = _decode_strings(
                handle["var/gene_id"], f"{spec.filename}/var/gene_id"
            )
    except OSError as error:
        raise HumanDataError(f"could not read AnnData source: {spec.filename}") from error
    if len(record_ids) != rows or len(set(record_ids)) != rows:
        raise HumanDataError(f"observation IDs are duplicated: {spec.filename}")
    if (
        len(gene_ids) != genes
        or len(set(gene_ids)) != genes
        or any(ENSG_RE.fullmatch(item) is None for item in gene_ids)
    ):
        raise HumanDataError(f"readout Ensembl identities are invalid: {spec.filename}")

    action_ids: list[str | None] = []
    controls = np.zeros(rows, dtype=np.bool_)
    unresolved: list[str] = []
    for index, record_id in enumerate(record_ids):
        match = ACTION_RE.search(record_id)
        if match is not None:
            action_ids.append(match.group(1))
        elif record_id.endswith(CONTROL_SUFFIX):
            action_ids.append(None)
            controls[index] = True
        else:
            action_ids.append(None)
            unresolved.append(record_id)
    return SourceMetadata(
        path,
        spec,
        rows,
        genes,
        record_ids,
        gene_ids,
        tuple(action_ids),
        controls,
        tuple(unresolved),
    )


def _split_name(action_id: str, seed: int = SEED) -> str:
    payload = f"slp11-development-v1|{seed}|{TAXON}|{action_id}".encode()
    bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def _normalized_source(
    metadata: SourceMetadata,
    query_ids: tuple[str, ...],
    context_index: int,
) -> dict[str, object]:
    query_lookup = {gene: index for index, gene in enumerate(metadata.gene_ids)}
    source_indices = np.asarray([query_lookup[gene] for gene in query_ids], dtype=np.int64)
    with h5py.File(metadata.path, "r") as handle:
        raw = np.asarray(handle["X"][:], dtype=np.float64)[:, source_indices]
    if raw.shape != (metadata.rows, len(query_ids)):
        raise HumanDataError("aligned source matrix shape drift")
    if not np.isfinite(raw).all() or np.any(raw < 0):
        raise HumanDataError("raw mean abundance contains negative or non-finite values")
    # Fractional values distinguish these mean-population profiles from summed counts.
    fractional = int(np.count_nonzero(np.abs(raw - np.rint(raw)) > 1e-6))
    if fractional == 0:
        raise HumanDataError("source does not exhibit the declared fractional mean semantics")
    denominator = raw.sum(axis=1, dtype=np.float64)
    nonzero = denominator > 0
    transformed = np.empty_like(raw, dtype=np.float32)
    transformed[nonzero] = np.log2(
        1.0 + 10_000.0 * raw[nonzero] / denominator[nonzero, None]
    ).astype(np.float32)
    transformed[~nonzero] = 0.0
    if not np.isfinite(transformed[nonzero]).all():
        raise HumanDataError("Gaussian target transform produced non-finite values")

    controls = metadata.control_mask & nonzero
    if not controls.any():
        raise HumanDataError(f"context lacks a non-targeting basal control: {metadata.spec.context_id}")
    basal = transformed[controls].mean(axis=0, dtype=np.float64).astype(np.float32)
    known = np.asarray([item is not None for item in metadata.action_ids]) & nonzero
    action_ids = np.asarray([metadata.action_ids[i] for i in np.flatnonzero(known)])
    record_ids = np.asarray(
        [f"{metadata.spec.context_id}|{metadata.record_ids[i]}" for i in np.flatnonzero(known)]
    )
    return {
        "targets": transformed[known],
        "action_ids": action_ids,
        "record_ids": record_ids,
        "context_index": np.full(int(known.sum()), context_index, dtype=np.int64),
        "basal": basal,
        "fractional_values": fractional,
        "all_zero_rows": int((~nonzero).sum()),
        "control_rows": int(controls.sum()),
        "target_rows": int(known.sum()),
    }


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
    return stream.getvalue()


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, _npy_bytes(arrays[name]))
    _atomic_write(path, output.getvalue())


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _string_array(values: Sequence[str]) -> np.ndarray:
    width = max(1, *(len(item) for item in values))
    return np.asarray(values, dtype=f"<U{width}")


def _list_payload(values: Sequence[str]) -> bytes:
    return "".join(f"{item}\n" for item in values).encode("ascii")


def _bundle(
    targets: np.ndarray,
    action_ids: np.ndarray,
    record_ids: np.ndarray,
    context_index: np.ndarray,
    selection: np.ndarray,
    split_names: np.ndarray,
    query_ids: tuple[str, ...],
    basal_control: np.ndarray,
) -> dict[str, np.ndarray]:
    selected_splits = split_names[selection]
    return {
        "targets": targets[selection].astype(np.float32, copy=False),
        "observed": np.ones((len(selection), len(query_ids)), dtype=np.bool_),
        "action_ids": _string_array([str(item) for item in action_ids[selection]]),
        "query_ids": _string_array(query_ids),
        "context_index": context_index[selection].astype(np.int64, copy=False),
        "context_ids": _string_array(CONTEXT_IDS),
        "basal_control": basal_control.astype(np.float32, copy=False),
        "record_ids": _string_array([str(item) for item in record_ids[selection]]),
        "split_train": np.flatnonzero(selected_splits == "train").astype(np.int64),
        "split_validation": np.flatnonzero(selected_splits == "validation").astype(
            np.int64
        ),
        "split_test": np.flatnonzero(selected_splits == "test").astype(np.int64),
    }


def build_human_development(
    k562_path: str | Path,
    rpe1_path: str | Path,
    destination: str | Path,
    *,
    source_specs: tuple[SourceSpec, SourceSpec] = PRODUCTION_SOURCES,
    expected_query_count: int | None = 7_226,
) -> dict[str, object]:
    """Build deterministic development and sealed test-only NPZ artifacts."""

    if len(source_specs) != 2 or tuple(item.context_id for item in source_specs) != CONTEXT_IDS:
        raise HumanDataError("source contexts must match the frozen K562/RPE1 order")
    paths = (Path(k562_path), Path(rpe1_path))
    metadata = tuple(
        _inspect_metadata(_verify_source(path, spec), spec)
        for path, spec in zip(paths, source_specs)
    )
    query_ids = tuple(sorted(set(metadata[0].gene_ids) & set(metadata[1].gene_ids)))
    if not query_ids or (
        expected_query_count is not None and len(query_ids) != expected_query_count
    ):
        raise HumanDataError("shared readout panel count drift")

    normalized = tuple(
        _normalized_source(item, query_ids, index) for index, item in enumerate(metadata)
    )
    targets = np.concatenate([item["targets"] for item in normalized], axis=0)
    action_ids = np.concatenate([item["action_ids"] for item in normalized])
    record_ids = np.concatenate([item["record_ids"] for item in normalized])
    context_index = np.concatenate([item["context_index"] for item in normalized])
    basal_control = np.stack([item["basal"] for item in normalized]).astype(np.float32)
    if len({str(item) for item in record_ids}) != len(record_ids):
        raise HumanDataError("derived record IDs are not unique")
    split_names = np.asarray([_split_name(str(item)) for item in action_ids])
    development = np.flatnonzero(split_names != "test").astype(np.int64)
    test = np.flatnonzero(split_names == "test").astype(np.int64)
    if not len(development) or not len(test):
        raise HumanDataError("development or test split is empty")

    destination_path = Path(destination)
    development_path = destination_path / "replogle-k562-rpe1-development-v1.npz"
    test_path = destination_path / "replogle-k562-rpe1-test-only-v1.npz"
    _write_npz(
        development_path,
        _bundle(
            targets,
            action_ids,
            record_ids,
            context_index,
            development,
            split_names,
            query_ids,
            basal_control,
        ),
    )
    _write_npz(
        test_path,
        _bundle(
            targets,
            action_ids,
            record_ids,
            context_index,
            test,
            split_names,
            query_ids,
            basal_control,
        ),
    )

    action_universe = tuple(sorted({str(item) for item in action_ids}))
    entity_universe = tuple(sorted(set(query_ids) | set(action_universe)))
    list_specs: dict[str, dict[str, object]] = {}
    for filename, values in (
        ("action-ids.txt", action_universe),
        ("query-ids.txt", query_ids),
        ("entity-ids.txt", entity_universe),
    ):
        payload = _list_payload(values)
        output_path = destination_path / filename
        _atomic_write(output_path, payload)
        list_specs[filename] = {
            "path": filename,
            "count": len(values),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "order": "ascending ASCII Ensembl gene ID",
        }

    split_counts = {
        name: int(np.count_nonzero(split_names == name))
        for name in ("train", "validation", "test")
    }
    context_split_counts = {
        CONTEXT_IDS[context]: {
            name: int(np.count_nonzero((context_index == context) & (split_names == name)))
            for name in ("train", "validation", "test")
        }
        for context in range(2)
    }
    manifest: dict[str, object] = {
        "schema": "slp.human-perturbation-development/v1",
        "status": "derived-development-and-sealed-test-not-omf-admitted",
        "ncbiTaxon": TAXON,
        "sourceManifest": "sources/human-perturbation-development-v1.yaml",
        "rights": "rights/figshare-replogle-2022-processed-perturb-seq-cc-by-4.0.yaml",
        "sourceSemantics": {
            "X": "per-perturbation arithmetic mean of raw cell UMI abundance",
            "evidence": {
                "repository": "https://github.com/thomasmaxwellnorman/Perturbseq_GI",
                "commit": AUTHOR_CODE_COMMIT,
                "path": "perturbseq/cell_population.py",
                "method": "CellPopulation.average uses groupby_apply(..., {'mean': 'mean'})",
                "artifactCheck": "X is nonnegative, finite, and fractional",
            },
            "likelihood": "Gaussian; fractional cell means are not NB counts",
        },
        "transform": {
            "id": TRANSFORM_ID,
            "formula": "log2(1 + 10000 * x / sum(x_shared_7226))",
            "denominatorPanel": "exact sorted K562/RPE1 Ensembl intersection",
            "geneSelectionUsesOutcomes": False,
            "originalPanels": {
                CONTEXT_IDS[0]: metadata[0].genes,
                CONTEXT_IDS[1]: metadata[1].genes,
            },
        },
        "sources": [
            {
                "contextId": item.spec.context_id,
                "localPath": f"data/sources/human/{item.spec.filename}",
                "bytes": item.spec.bytes,
                "sha256": item.spec.sha256,
                "rows": item.rows,
                "readouts": item.genes,
                "targetRows": derived["target_rows"],
                "controlRows": derived["control_rows"],
                "allZeroRowsExcluded": derived["all_zero_rows"],
                "fractionalValuesObserved": derived["fractional_values"],
                "unresolvedActionsQuarantined": list(item.unresolved_ids),
            }
            for item, derived in zip(metadata, normalized)
        ],
        "counts": {
            "queries": len(query_ids),
            "actions": len(action_universe),
            "records": len(action_ids),
            "developmentRecords": len(development),
            "testOnlyRecords": len(test),
            "splitRecords": split_counts,
            "contextSplitRecords": context_split_counts,
            "controlsUsedForBasal": [item["control_rows"] for item in normalized],
        },
        "identityLists": list_specs,
        "outputs": {
            "development": {
                "path": development_path.name,
                "bytes": development_path.stat().st_size,
                "sha256": _hash_file(development_path, "sha256"),
                "contains": ["train", "validation"],
            },
            "testOnly": {
                "path": test_path.name,
                "bytes": test_path.stat().st_size,
                "sha256": _hash_file(test_path, "sha256"),
                "contains": ["test"],
                "access": "sealed until candidate and rule lock",
            },
        },
        "limitations": [
            "K562 and RPE1 original measured panels differ and were intersected before reading X.",
            "The contexts also differ in sampling day and screened action population.",
            "Pseudobulk rows are correlated summaries of contributing cells, not independent cells.",
            "No model, molecular evaluation, benchmark evaluation, or OMF admission occurred.",
        ],
    }
    manifest_payload = (
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    manifest_path = destination_path / "manifest.json"
    _atomic_write(manifest_path, manifest_payload)
    return {
        "manifest": manifest,
        "manifestPath": str(manifest_path),
        "manifestSha256": hashlib.sha256(manifest_payload).hexdigest(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k562", required=True)
    parser.add_argument("--rpe1", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build_human_development(args.k562, args.rpe1, args.output)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
