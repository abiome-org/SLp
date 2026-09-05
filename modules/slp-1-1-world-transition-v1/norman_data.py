"""Build leakage-routed Norman 2019 single/double CRISPRa molecular records."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import mmread

TAXON = 9606
SEED = 731
CONTEXT_ID = "norman-2019-k562-crispra-day5"
ENSG_RE = re.compile(r"^ENSG[0-9]+$")
QUERY_COUNT = 7_226
QUERY_BYTES = 115_616
QUERY_SHA256 = "645b8d563b440a4b7ab6a3bb42450594b408c4e7cb84e4fe2789a6620174f12c"
VALUE_SPACE = "log2-1p-cp10k-source-cell-mean-shared-query-panel-v1"
SOURCE_SPECS = {
    "matrix": (
        "GSE133344_filtered_matrix.mtx.gz",
        1_130_430_845,
        "7af10888c29be7e5be0466af100dfd6c5f1bc87f6d7cbfcb48270248d3639120",
    ),
    "genes": (
        "GSE133344_filtered_genes.tsv.gz",
        264_791,
        "0dcba3cf4f3095b3fc1fa31b402c562bca7eea8d7d9ffd753e7b446dc37b9e3d",
    ),
    "barcodes": (
        "GSE133344_filtered_barcodes.tsv.gz",
        429_928,
        "49529cff274aada5757e0b5ce320133437282398a22af14afc314e16eecade66",
    ),
    "identities": (
        "GSE133344_filtered_cell_identities.csv.gz",
        1_956_905,
        "daf30337e7f6f07096d57e0d81db784bef00c87bd1fc927f018792c2f7af81e4",
    ),
    "soft": (
        "GSE133344_family.soft.gz",
        3_780,
        "2f8620323a32518df603f361452f60a4379fdb9448151678e941c2e8f40ef9ab",
    ),
}
GTF_NAME = "cellranger-GRCh38-1.2.0_only_genes.gtf"
GTF_BYTES = 7_593_284
GTF_SHA256 = "796bb1f1d36c75462fea32e87cc54c66e2ee7b60a2e6eed3b6e0c02e8df7908b"
AUTHOR_REVISION = "3b25109aeb9c0c2026bd70abd50304a0ad4e5395"
NAME_REPLACER = {
    "C3orf72": "FOXL2NB",
    "C19orf26": "CBARP",
    "KIAA1804": "RP5-862P8.2",
    "RHOXF2": "RHOXF2B",
}


class NormanDataError(ValueError):
    """The Norman source or routing contract is invalid."""


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected_name: str, size: int, digest: str) -> Path:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or resolved.name != expected_name or not resolved.is_file():
        raise NormanDataError(f"invalid source path: {expected_name}")
    if resolved.stat().st_size != size or _hash(resolved) != digest:
        raise NormanDataError(f"source byte identity drift: {expected_name}")
    return resolved


def _read_lines(path: Path) -> tuple[str, ...]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return tuple(line.rstrip("\n\r") for line in stream)


def _load_query_ids(path: Path) -> tuple[str, ...]:
    payload = path.read_bytes()
    if (
        len(payload) != QUERY_BYTES
        or hashlib.sha256(payload).hexdigest() != QUERY_SHA256
        or not payload.endswith(b"\n")
        or b"\r" in payload
    ):
        raise NormanDataError("global query roster byte identity drift")
    values = tuple(payload.decode("ascii").splitlines())
    if len(values) != QUERY_COUNT or list(values) != sorted(set(values)):
        raise NormanDataError("global query roster ordering drift")
    return values


def _load_gtf_mapping(path: Path) -> dict[str, str]:
    _verify(path, GTF_NAME, GTF_BYTES, GTF_SHA256)
    mapping: dict[str, set[str]] = defaultdict(set)
    attribute_re = re.compile(r'(\w+) "([^"]+)"')
    with path.open("rt", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 9 or columns[2] != "gene":
                continue
            attributes = dict(attribute_re.findall(columns[8]))
            gene_id = attributes.get("gene_id", "").split(".")[0]
            gene_name = attributes.get("gene_name", "")
            if ENSG_RE.fullmatch(gene_id) and gene_name:
                mapping[gene_name].add(gene_id)
    return {name: next(iter(ids)) for name, ids in mapping.items() if len(ids) == 1}


def _split_name(action_id: str, seed: int = SEED) -> str:
    payload = f"slp11-development-v1|{seed}|{TAXON}|{action_id}".encode()
    bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def route_actions(actions: tuple[str, ...]) -> str:
    """Route a combination to its most restricted constituent split."""
    roles = {_split_name(action) for action in actions}
    return "test" if "test" in roles else "validation" if "validation" in roles else "train"


def _parse_target_symbols(identity: str) -> tuple[str, ...]:
    guide_target = identity.split("__", 1)[0]
    parts = tuple(NAME_REPLACER.get(item, item) for item in guide_target.split("_"))
    if len(parts) != 2 or any(not item for item in parts):
        raise NormanDataError("unexpected guide identity structure")
    return tuple(item for item in parts if not item.startswith("NegCtrl"))


def load_metadata(
    barcodes_path: Path, identities_path: Path, gtf_path: Path
) -> dict[str, object]:
    barcodes = _read_lines(barcodes_path)
    if len(barcodes) != len(set(barcodes)):
        raise NormanDataError("filtered barcodes are not unique")
    identities: dict[str, dict[str, str]] = {}
    with gzip.open(identities_path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            barcode = row.get("cell_barcode", "")
            if not barcode or barcode in identities:
                raise NormanDataError("cell identity barcode is empty or duplicated")
            identities[barcode] = row
    if set(identities) - set(barcodes):
        raise NormanDataError("cell identity table contains unknown barcodes")
    gene_mapping = _load_gtf_mapping(gtf_path)
    cell_condition: list[tuple[str, ...] | None] = []
    eligible = np.zeros(len(barcodes), dtype=np.bool_)
    unresolved_cells = 0
    for index, barcode in enumerate(barcodes):
        row = identities.get(barcode)
        if row is None:
            cell_condition.append(None)
            continue
        if not (
            row.get("good_coverage") == "True"
            and row.get("number_of_cells") == "1"
            and row.get("guide_identity") != "*"
        ):
            cell_condition.append(None)
            continue
        symbols = _parse_target_symbols(row["guide_identity"])
        if not symbols:
            actions: tuple[str, ...] = ()
        else:
            mapped = tuple(gene_mapping.get(symbol, "") for symbol in symbols)
            if any(not item for item in mapped) or len(set(mapped)) != len(mapped):
                unresolved_cells += 1
                cell_condition.append(None)
                continue
            actions = tuple(sorted(mapped))
        if len(actions) > 2:
            raise NormanDataError("more than two stable actions in one condition")
        eligible[index] = True
        cell_condition.append(actions)
    conditions = tuple(sorted({item for item in cell_condition if item is not None}))
    condition_index = {item: index for index, item in enumerate(conditions)}
    selected_columns = np.flatnonzero(eligible).astype(np.int64)
    selected_condition = np.asarray(
        [condition_index[cell_condition[index]] for index in selected_columns], dtype=np.int64
    )
    cell_counts = np.bincount(selected_condition, minlength=len(conditions)).astype(np.int64)
    return {
        "barcodes": barcodes,
        "conditions": conditions,
        "selected_columns": selected_columns,
        "selected_condition": selected_condition,
        "cell_counts": cell_counts,
        "unresolved_cells": unresolved_cells,
    }


def _string_array(values: Sequence[str]) -> np.ndarray:
    return np.asarray(values, dtype=f"<U{max(1, *(len(item) for item in values))}")


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


def _bundle(
    targets: np.ndarray,
    observed: np.ndarray,
    conditions: tuple[tuple[str, ...], ...],
    cell_counts: np.ndarray,
    query_ids: tuple[str, ...],
    selection: np.ndarray,
    split_names: np.ndarray,
    basal: np.ndarray,
) -> dict[str, np.ndarray]:
    selected_conditions = [conditions[index] for index in selection]
    offsets = np.zeros(len(selection) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(item) for item in selected_conditions])
    action_ids = [action for item in selected_conditions for action in item]
    selected_splits = split_names[selection]
    record_ids = [
        "norman-2019|" + "+".join(item) if item else "norman-2019|control"
        for item in selected_conditions
    ]
    return {
        "targets": targets[selection],
        "observed": observed[selection],
        "action_ids": _string_array(action_ids),
        "action_offsets": offsets,
        "query_ids": _string_array(query_ids),
        "context_ids": _string_array([CONTEXT_ID]),
        "context_index": np.zeros(len(selection), dtype=np.int64),
        "record_ids": _string_array(record_ids),
        "num_cells_filtered": cell_counts[selection],
        "basal_control": basal[None, :],
        "target_value_space": np.asarray(VALUE_SPACE),
        "split_train": np.flatnonzero(selected_splits == "train").astype(np.int64),
        "split_validation": np.flatnonzero(selected_splits == "validation").astype(np.int64),
        "split_test": np.flatnonzero(selected_splits == "test").astype(np.int64),
    }


def build_norman(
    source_dir: str | Path,
    gtf_path: str | Path,
    query_path: str | Path,
    destination: str | Path,
) -> dict[str, object]:
    source = Path(source_dir)
    verified = {
        label: _verify(source / name, name, size, digest)
        for label, (name, size, digest) in SOURCE_SPECS.items()
    }
    query_ids = _load_query_ids(Path(query_path))
    genes = _read_lines(verified["genes"])
    gene_ids = tuple(line.split("\t", 1)[0] for line in genes)
    if len(gene_ids) != 33_694 or len(set(gene_ids)) != len(gene_ids):
        raise NormanDataError("filtered gene roster drift")
    metadata = load_metadata(verified["barcodes"], verified["identities"], Path(gtf_path))
    conditions = metadata["conditions"]
    assert isinstance(conditions, tuple)
    selected_columns = np.asarray(metadata["selected_columns"])
    selected_condition = np.asarray(metadata["selected_condition"])
    cell_counts = np.asarray(metadata["cell_counts"])
    matrix = mmread(verified["matrix"]).tocsr()
    if matrix.shape != (len(gene_ids), len(metadata["barcodes"])):
        raise NormanDataError("filtered matrix dimensions drift")
    query_lookup = {gene: index for index, gene in enumerate(query_ids)}
    present_pairs = [(row, query_lookup[gene]) for row, gene in enumerate(gene_ids) if gene in query_lookup]
    source_rows = np.asarray([item[0] for item in present_pairs], dtype=np.int64)
    output_rows = np.asarray([item[1] for item in present_pairs], dtype=np.int64)
    membership = sparse.csr_matrix(
        (
            np.ones(len(selected_columns), dtype=np.float64),
            (np.arange(len(selected_columns)), selected_condition),
        ),
        shape=(len(selected_columns), len(conditions)),
    )
    sums = matrix[source_rows][:, selected_columns].astype(np.float64) @ membership
    means = sums.toarray().T / cell_counts[:, None]
    targets = np.zeros((len(conditions), len(query_ids)), dtype=np.float32)
    denominator = means.sum(axis=1)
    if np.any(denominator <= 0):
        raise NormanDataError("a pseudobulk has no counts on the shared query panel")
    targets[:, output_rows] = np.log2(
        1.0 + 10_000.0 * means / denominator[:, None]
    ).astype(np.float32)
    observed = np.zeros_like(targets, dtype=np.bool_)
    observed[:, output_rows] = True
    control_index = conditions.index(())
    basal = targets[control_index].copy()
    split_names = np.asarray(
        ["control" if not actions else route_actions(actions) for actions in conditions]
    )
    if any(
        role != ("control" if not actions else route_actions(actions))
        for actions, role in zip(conditions, split_names, strict=True)
    ):
        raise NormanDataError("constituent-priority record routing drift")
    development = np.flatnonzero(split_names != "test").astype(np.int64)
    test = np.flatnonzero(split_names == "test").astype(np.int64)
    development_path = Path(destination) / "norman-2019-development-v1.npz"
    test_path = Path(destination) / "norman-2019-test-only-v1.npz"
    _write_npz(
        development_path,
        _bundle(targets, observed, conditions, cell_counts, query_ids, development, split_names, basal),
    )
    _write_npz(
        test_path,
        _bundle(targets, observed, conditions, cell_counts, query_ids, test, split_names, basal),
    )
    action_union = sorted({action for actions in conditions for action in actions})
    pair_counts = {str(size): sum(len(item) == size for item in conditions) for size in (0, 1, 2)}
    manifest = {
        "schema": "slp.norman-2019-combination-response/v1",
        "status": "derived-development-and-routed-test-not-omf-admitted",
        "source": "GEO:GSE133344",
        "context": CONTEXT_ID,
        "ncbiTaxon": TAXON,
        "perturbation": "simultaneous-single-or-double-CRISPRa-activation",
        "valueSpace": VALUE_SPACE,
        "transform": "mean raw UMI count per source guide-target population, then log2(1+10000*x/sum(global-present-query-panel))",
        "split": "per constituent sha256(slp11-development-v1|731|9606|ENSG), test > validation > train",
        "counts": {
            "sourceGenes": len(gene_ids),
            "sourceCells": len(metadata["barcodes"]),
            "eligibleCells": len(selected_columns),
            "unresolvedCellsQuarantined": metadata["unresolved_cells"],
            "queryGenes": len(query_ids),
            "sourcePresentQueryGenes": len(output_rows),
            "uniqueActionGenes": len(action_union),
            "recordsByActionCount": pair_counts,
            "recordsBySplit": {
                role: int(np.count_nonzero(split_names == role))
                for role in ("control", "train", "validation", "test")
            },
            "cellsByActionCount": {
                str(size): int(sum(cell_counts[i] for i, item in enumerate(conditions) if len(item) == size))
                for size in (0, 1, 2)
            },
        },
        "identity": {
            "queryNamespace": "Ensembl-gene",
            "queryRosterSha256": QUERY_SHA256,
            "actionNamespace": "Ensembl-gene",
            "mapping": "author-pinned GRCh38-1.2.0 GTF exact gene_name to unique stable ENSG",
            "authorRepositoryRevision": AUTHOR_REVISION,
            "symbolIdentityStored": False,
        },
        "sourceFiles": {
            label: {"path": spec[0], "bytes": spec[1], "sha256": spec[2]}
            for label, spec in SOURCE_SPECS.items()
        },
        "outputs": {
            "development": {
                "path": development_path.name,
                "bytes": development_path.stat().st_size,
                "sha256": _hash(development_path),
                "contains": ["control", "train", "validation"],
            },
            "testOnly": {
                "path": test_path.name,
                "bytes": test_path.stat().st_size,
                "sha256": _hash(test_path),
                "contains": ["test"],
                "access": "sealed until candidate and rule lock",
            },
        },
        "testOutcomeMetricsComputed": False,
        "benchmarkDataConsumed": False,
    }
    manifest_path = Path(destination) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    action_path = Path(destination) / "action-ids.txt"
    action_path.write_text("".join(f"{item}\n" for item in action_union), encoding="ascii", newline="\n")
    return {
        "manifestPath": str(manifest_path),
        "manifestSha256": _hash(manifest_path),
        "manifest": manifest,
        "actionRosterSha256": _hash(action_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--query-ids", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_norman(args.source_dir, args.gtf, args.query_ids, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
