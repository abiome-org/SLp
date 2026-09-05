#!/usr/bin/env python3
"""Build deterministic order-sensitive SLp-1.1 yeast sequence features.

This exploratory builder consumes only the exact admitted SGD R64.5.1 protein
FASTA and the frozen outcome-blind sequence-statistics feature block.  The
feature block supplies the exact composite-key ordering and typed sequence
relations; the FASTA supplies canonical peptide bytes.  No outcome, split,
held-roster, reward, or benchmark artifact is accepted.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import itertools
import json
import os
import re
import sys
import tarfile
import time
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
DIPEPTIDE_ORDER = tuple(a + b for a in AA_ORDER for b in AA_ORDER)
FEATURE_NAMES = (
    "protein_length_div_4096",
    *(f"aa_fraction_{aa}" for aa in AA_ORDER),
    *(f"dipeptide_fraction_{pair}" for pair in DIPEPTIDE_ORDER),
)
SPECIES_TAXON = 4932
SOURCE_STRAIN_TAXON = 559292
EXPECTED_ROWS = 7_037
EXPECTED_FASTA_RECORDS = 6_722
EXPECTED_FASTA_GZIP_BYTES = 2_689_634
EXPECTED_FASTA_GZIP_SHA256 = (
    "17e8b47e1ae23178c6000fbc4ab548f102d1b250ef9dff5d811feb3f03dd2c5b"
)
EXPECTED_FASTA_BYTES = 5_511_467
EXPECTED_FASTA_SHA256 = (
    "e01f9e1ef7e5a01ff7cd0ee7a843e6d1c1da8c3777fdfac3a5293711d4c56518"
)
EXPECTED_FEATURE_BLOCK_BYTES = 4_392_960
EXPECTED_FEATURE_BLOCK_SHA256 = (
    "1b0aaec738b10ad3baa082d907d0c962c35c9b159b89fffca893fa1ecf5a7bed"
)
EXPECTED_ENTITY_ROWS_SHA256 = (
    "e487f428c6eb1eb58de0d3e8ca74f016841713c3014cc55370048eb3e8304572"
)
EXPECTED_SEQUENCE_PROVENANCE_SHA256 = (
    "5955ae6f8503b87370bf5116fdae8699ced9c4e3a0a378fd3843baaa7c2965fe"
)
EXPECTED_ENTITY_KEY_SET_SHA256 = (
    "82b8e2885939577fe6946e3b974a10cb947834118f2070e1bcbe4c2f2e6a5fd9"
)
SEQUENCE_RESOURCE = (
    "omf://abiome/slp/datasetsnapshot/"
    "slp-1-1-sgd-protein-sequences-r64-5-1@"
    "sha256:3b76017f5ac74d8d96efb1db52d14af91c9fb15995062110558ce4651cf3ba0c"
)
SEQUENCE_MANIFEST_DIGEST = (
    "sha256:8f88480196b5cd8f3c15d65dbdbc09f83305c371fb476c70a38825dad2be4283"
)
STATIC_UNIVERSE_RESOURCE = (
    "omf://abiome/slp/datasetsnapshot/slp-1-1-static-entity-universe-v1@"
    "sha256:de3efddf5a9e4f66496a1edda14b04de774e972bc7b9efd30964644de2a56cac"
)
FEATURE_BLOCK_SCHEMA = "slp.sequence-statistics-feature-block/v1"
OUTPUT_SCHEMA = "slp.sequence-dipeptide-feature-artifact/v1"
MANIFEST_NAME = "sequence-dipeptide-features.manifest.json"
NPZ_NAME = "sequence-dipeptide-features.npz"
ESM_NPZ_NAME = "sequence-esm2-features.npz"
ESM_MANIFEST_NAME = "sequence-esm2-features.manifest.json"
ESM_OUTPUT_SCHEMA = "slp.sequence-esm2-feature-artifact/v1"
ESM_REPOSITORY = "facebook/esm2_t6_8M_UR50D"
ESM_REVISION = "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"
ESM_LICENSE = "MIT"
ESM_HIDDEN_SIZE = 320
ESM_MAX_RESIDUES = 1_022
ESM_DEFAULT_OVERLAP = 128
ESM_FILE_SPECS = {
    "README.md": (
        1_705,
        "462a2f24724e19c6be0efab926315c294a863c9a9770e2c8b3d859b2d81a07de",
    ),
    "config.json": (
        775,
        "facb9355e8252149f6f99dff5fd1e32e890c3c655362b261d0c81cd6b5839c85",
    ),
    "model.safetensors": (
        31_384_292,
        "24c5fa474c48f3b754b86efe752d5f189d2bcd88190fa2270fc92b2ef3034189",
    ),
    "special_tokens_map.json": (
        125,
        "3aedcd4211c0d43aec4e607ff60a63255f3174ead795e997350f09a5f8cd9ee1",
    ),
    "tokenizer_config.json": (
        95,
        "7e9161ecdb548ec45a41cbc6b24aa4476fdd418461f491c4207baa99419a29ad",
    ),
    "vocab.txt": (
        93,
        "0b82cc0a7c7cf9e567b1e5892d793285b9fbae822c964ca48696f7db44598e03",
    ),
}
FASTA_ID_RE = re.compile(rb"(?:^| )SGDID:(S[0-9]{9}),")
VALID_PEPTIDE_RE = re.compile(rb"M[ACDEFGHIKLMNPQRSTVWY]*\*")
EXPECTED_TAR_MEMBERS = {
    "static-feature-block/entities.jsonl",
    "static-feature-block/excluded-non-current.jsonl",
    "static-feature-block/manifest.json",
    "static-feature-block/present.npy",
    "static-feature-block/sequence-provenance.jsonl",
    "static-feature-block/values.npy",
}


class SequenceFeatureError(ValueError):
    """Raised when a pinned input or sequence relation violates its contract."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _require_file(
    path: Path, expected_bytes: int, expected_sha256: str, label: str
) -> None:
    if not path.is_file():
        raise SequenceFeatureError(f"{label} is not a regular file: {path}")
    size = path.stat().st_size
    if size != expected_bytes:
        raise SequenceFeatureError(
            f"{label} byte count mismatch: expected {expected_bytes}, found {size}"
        )
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise SequenceFeatureError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, found {digest}"
        )


def parse_pinned_fasta(path: Path) -> dict[str, bytes]:
    """Parse and verify the exact admitted SGD gzip into SGD CURIE -> raw peptide."""

    _require_file(
        path,
        EXPECTED_FASTA_GZIP_BYTES,
        EXPECTED_FASTA_GZIP_SHA256,
        "SGD R64.5.1 FASTA gzip",
    )
    try:
        with gzip.open(path, "rb") as stream:
            payload = stream.read(EXPECTED_FASTA_BYTES + 1)
    except (OSError, EOFError) as exc:
        raise SequenceFeatureError("SGD FASTA gzip integrity check failed") from exc
    if (
        len(payload) != EXPECTED_FASTA_BYTES
        or sha256_bytes(payload) != EXPECTED_FASTA_SHA256
    ):
        raise SequenceFeatureError("decompressed SGD R64.5.1 FASTA identity mismatch")

    sequences: dict[str, bytes] = {}
    current_id: str | None = None
    chunks: list[bytes] = []

    def finish() -> None:
        nonlocal current_id, chunks
        if current_id is None:
            return
        sequence = b"".join(chunks)
        if not sequence:
            raise SequenceFeatureError(f"empty FASTA sequence for {current_id}")
        if current_id in sequences:
            raise SequenceFeatureError(f"duplicate FASTA SGD identifier {current_id}")
        sequences[current_id] = sequence

    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            raise SequenceFeatureError(f"blank FASTA line at {line_number}")
        if line.startswith(b">"):
            finish()
            match = FASTA_ID_RE.search(line[1:])
            if match is None:
                raise SequenceFeatureError(
                    f"missing exact SGDID token at FASTA line {line_number}"
                )
            current_id = "SGD:" + match.group(1).decode("ascii")
            chunks = []
        else:
            if current_id is None:
                raise SequenceFeatureError("FASTA sequence appears before first header")
            if any(byte != ord("*") and not 65 <= byte <= 90 for byte in line):
                raise SequenceFeatureError(
                    f"invalid FASTA sequence bytes at line {line_number}"
                )
            chunks.append(line)
    finish()
    if len(sequences) != EXPECTED_FASTA_RECORDS:
        raise SequenceFeatureError(
            f"FASTA record count mismatch: expected {EXPECTED_FASTA_RECORDS}, "
            f"found {len(sequences)}"
        )
    return sequences


def _read_jsonl(payload: bytes, label: str) -> list[dict[str, object]]:
    if not payload.endswith(b"\n") or b"\r" in payload:
        raise SequenceFeatureError(f"{label} must be LF-terminated JSONL")
    rows: list[dict[str, object]] = []
    for index, line in enumerate(payload.splitlines()):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SequenceFeatureError(f"invalid {label} row {index}") from exc
        if not isinstance(row, dict):
            raise SequenceFeatureError(f"{label} row {index} must be an object")
        rows.append(row)
    return rows


def load_pinned_feature_provenance(
    path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Read exact entity ordering and typed sequence relations from the frozen block."""

    _require_file(
        path,
        EXPECTED_FEATURE_BLOCK_BYTES,
        EXPECTED_FEATURE_BLOCK_SHA256,
        "sequence-statistics feature block",
    )
    try:
        with tarfile.open(path, mode="r:") as archive:
            members = archive.getmembers()
            names = {member.name for member in members}
            if names != EXPECTED_TAR_MEMBERS or any(
                not member.isfile() for member in members
            ):
                raise SequenceFeatureError(
                    "feature block has an unexpected tar member set"
                )
            blobs = {
                member.name: archive.extractfile(member).read()  # type: ignore[union-attr]
                for member in members
            }
    except (OSError, tarfile.TarError) as exc:
        raise SequenceFeatureError(
            "cannot read sequence-statistics feature block"
        ) from exc

    entity_payload = blobs["static-feature-block/entities.jsonl"]
    provenance_payload = blobs["static-feature-block/sequence-provenance.jsonl"]
    if sha256_bytes(entity_payload) != EXPECTED_ENTITY_ROWS_SHA256:
        raise SequenceFeatureError("frozen feature entity rows SHA-256 mismatch")
    if sha256_bytes(provenance_payload) != EXPECTED_SEQUENCE_PROVENANCE_SHA256:
        raise SequenceFeatureError("frozen sequence provenance SHA-256 mismatch")
    try:
        manifest = json.loads(blobs["static-feature-block/manifest.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SequenceFeatureError("invalid frozen feature block manifest") from exc
    if not isinstance(manifest, dict):
        raise SequenceFeatureError("frozen feature block manifest must be an object")
    if (
        manifest.get("schema") != FEATURE_BLOCK_SCHEMA
        or manifest.get("identityKey") != ["ncbiTaxon", "entityId"]
        or manifest.get("ordering") != "ascending-ncbiTaxon-then-codepoint-entityId"
        or manifest.get("semanticHashes", {}).get("entityKeySetSha256")
        != EXPECTED_ENTITY_KEY_SET_SHA256
        or manifest.get("inputs", {}).get("sgdProteinSequences", {}).get("resource")
        != SEQUENCE_RESOURCE
        or manifest.get("inputs", {}).get("staticEntityUniverse", {}).get("resource")
        != STATIC_UNIVERSE_RESOURCE
    ):
        raise SequenceFeatureError("frozen feature block provenance contract mismatch")
    entities = _read_jsonl(entity_payload, "feature entities")
    provenance = _read_jsonl(provenance_payload, "sequence provenance")
    if len(entities) != EXPECTED_ROWS or len(provenance) != EXPECTED_ROWS:
        raise SequenceFeatureError("frozen feature row count mismatch")
    return entities, provenance, manifest


def resolve_entity_peptides(
    sequences: Mapping[str, bytes],
    entities: Sequence[Mapping[str, object]],
    provenance: Sequence[Mapping[str, object]],
) -> tuple[list[tuple[int, str]], list[bytes]]:
    """Resolve entity sequences with exact composite identity and consensus semantics."""

    if len(entities) != len(provenance):
        raise SequenceFeatureError("entity/provenance row count mismatch")
    keys: list[tuple[int, str]] = []
    peptides: list[bytes] = []
    for index, (entity, source) in enumerate(zip(entities, provenance, strict=True)):
        taxon, entity_id = entity.get("ncbiTaxon"), entity.get("entityId")
        if (
            not isinstance(taxon, int)
            or isinstance(taxon, bool)
            or taxon != SPECIES_TAXON
        ):
            raise SequenceFeatureError(f"invalid entity taxon at row {index}")
        if not isinstance(entity_id, str) or not entity_id:
            raise SequenceFeatureError(f"invalid entity ID at row {index}")
        if entity.get("rowIndex") != index:
            raise SequenceFeatureError(f"entity row index mismatch at row {index}")
        if (
            source.get("rowIndex") != index
            or source.get("ncbiTaxon") != taxon
            or source.get("entityId") != entity_id
            or source.get("sourceStrainTaxon") != SOURCE_STRAIN_TAXON
        ):
            raise SequenceFeatureError(
                f"entity/provenance identity mismatch at row {index}"
            )
        source_ids = source.get("sourceSequenceIds")
        if (
            not isinstance(source_ids, list)
            or not source_ids
            or any(not isinstance(item, str) for item in source_ids)
            or source_ids != sorted(set(source_ids))
        ):
            raise SequenceFeatureError(f"invalid source sequence IDs at row {index}")
        try:
            raw_sequences = [sequences[item] for item in source_ids]
        except KeyError as exc:
            raise SequenceFeatureError(
                f"missing admitted source sequence {exc.args[0]} for {entity_id}"
            ) from exc
        if any(sequence != raw_sequences[0] for sequence in raw_sequences[1:]):
            raise SequenceFeatureError(
                f"typed one-to-many sequence relation lacks exact consensus for {entity_id}"
            )
        raw = raw_sequences[0]
        if VALID_PEPTIDE_RE.fullmatch(raw) is None:
            raise SequenceFeatureError(f"invalid current-ORF peptide for {entity_id}")
        peptide = raw[:-1]
        peptide_hash = source.get("canonicalPeptideSha256")
        if peptide_hash != sha256_bytes(peptide):
            raise SequenceFeatureError(
                f"canonical peptide hash mismatch for {entity_id}"
            )
        if source.get("canonicalPeptideLength") != len(peptide):
            raise SequenceFeatureError(
                f"canonical peptide length mismatch for {entity_id}"
            )
        keys.append((taxon, entity_id))
        peptides.append(peptide)
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise SequenceFeatureError("composite entity keys must be uniquely sorted")
    return keys, peptides


def peptide_features(peptide: bytes) -> np.ndarray:
    """Compute [scaled length, composition, ordered-dipeptide frequency]."""

    if not peptide or any(chr(byte) not in AA_ORDER for byte in peptide):
        raise SequenceFeatureError(
            "feature peptide must contain only canonical amino acids"
        )
    aa_index = {ord(aa): index for index, aa in enumerate(AA_ORDER)}
    counts = np.zeros(len(AA_ORDER), dtype=np.int64)
    pairs = np.zeros(len(DIPEPTIDE_ORDER), dtype=np.int64)
    for byte in peptide:
        counts[aa_index[byte]] += 1
    for left, right in itertools.pairwise(peptide):
        pairs[aa_index[left] * len(AA_ORDER) + aa_index[right]] += 1
    length = len(peptide)
    pair_denominator = max(length - 1, 1)
    output = np.empty(len(FEATURE_NAMES), dtype=np.dtype("<f4"))
    output[0] = np.float32(length / 4096.0)
    output[1:21] = counts.astype(np.float32) / np.float32(length)
    output[21:] = pairs.astype(np.float32) / np.float32(pair_denominator)
    return output


def build_feature_arrays(
    sequences: Mapping[str, bytes],
    entities: Sequence[Mapping[str, object]],
    provenance: Sequence[Mapping[str, object]],
) -> dict[str, np.ndarray]:
    keys, peptides = resolve_entity_peptides(sequences, entities, provenance)
    values = np.stack([peptide_features(peptide) for peptide in peptides]).astype(
        np.dtype("<f4"), copy=False
    )
    max_id_chars = max(len(entity_id) for _, entity_id in keys)
    identifiers = np.asarray(
        [entity_id for _, entity_id in keys], dtype=f"<U{max_id_chars}"
    )
    taxa = np.asarray([taxon for taxon, _ in keys], dtype=np.dtype("<i8"))
    if not np.isfinite(values).all():
        raise SequenceFeatureError("non-finite feature value generated")
    return {
        "feature_values": values,
        "entity_taxon": taxa,
        "entity_id": identifiers,
    }


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, allow_pickle=False)
    return stream.getvalue()


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Create compressed NPZ bytes with fixed member order and ZIP metadata."""

    expected = ("feature_values", "entity_taxon", "entity_id")
    if tuple(arrays) != expected:
        raise SequenceFeatureError(f"NPZ arrays must be ordered as {expected}")
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in expected:
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, _npy_bytes(arrays[name]), compresslevel=9)
    return output.getvalue()


def verify_esm_model_dir(model_dir: Path) -> dict[str, dict[str, object]]:
    """Verify every model file against the exact pinned Hugging Face revision."""

    if not model_dir.is_dir():
        raise SequenceFeatureError(f"ESM model directory does not exist: {model_dir}")
    actual_files = {path.name for path in model_dir.iterdir() if path.is_file()}
    expected_files = set(ESM_FILE_SPECS)
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise SequenceFeatureError(
            f"ESM model file set mismatch; missing={missing}, unexpected={unexpected}"
        )
    verified: dict[str, dict[str, object]] = {}
    for name in sorted(ESM_FILE_SPECS):
        expected_bytes, expected_sha256 = ESM_FILE_SPECS[name]
        _require_file(model_dir / name, expected_bytes, expected_sha256, f"ESM {name}")
        verified[name] = {"bytes": expected_bytes, "sha256": expected_sha256}
    try:
        config = json.loads((model_dir / "config.json").read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SequenceFeatureError("invalid pinned ESM config.json") from exc
    if (
        config.get("model_type") != "esm"
        or config.get("hidden_size") != ESM_HIDDEN_SIZE
        or config.get("num_hidden_layers") != 6
        or config.get("max_position_embeddings") != 1_026
        or config.get("hidden_dropout_prob") != 0.0
        or config.get("attention_probs_dropout_prob") != 0.0
    ):
        raise SequenceFeatureError("pinned ESM architecture contract mismatch")
    return verified


def chunk_windows(
    length: int,
    max_residues: int = ESM_MAX_RESIDUES,
    overlap: int = ESM_DEFAULT_OVERLAP,
) -> list[tuple[int, int]]:
    """Tile every residue without truncation, using explicit overlapping windows."""

    if length < 1:
        raise SequenceFeatureError("cannot chunk an empty peptide")
    if max_residues < 2 or overlap < 0 or overlap >= max_residues:
        raise SequenceFeatureError("invalid ESM chunk length/overlap")
    if length <= max_residues:
        return [(0, length)]
    final_start = length - max_residues
    starts = list(range(0, final_start + 1, max_residues - overlap))
    if starts[-1] != final_start:
        starts.append(final_start)
    windows = [(start, min(start + max_residues, length)) for start in starts]
    if windows[0][0] != 0 or windows[-1][1] != length:
        raise SequenceFeatureError("internal ESM chunk coverage error")
    return windows


def window_inverse_coverage(
    length: int, windows: Sequence[tuple[int, int]]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return per-residue coverage and weights that sum to one across windows."""

    coverage = np.zeros(length, dtype=np.int32)
    for start, end in windows:
        if start < 0 or end <= start or end > length:
            raise SequenceFeatureError("invalid ESM window boundary")
        coverage[start:end] += 1
    if np.any(coverage < 1):
        raise SequenceFeatureError("ESM windows silently omit one or more residues")
    weights = [
        np.reciprocal(coverage[start:end].astype(np.float64)) for start, end in windows
    ]
    reconstructed = np.zeros(length, dtype=np.float64)
    for (start, end), weight in zip(windows, weights, strict=True):
        reconstructed[start:end] += weight
    if not np.array_equal(reconstructed, np.ones(length, dtype=np.float64)):
        raise SequenceFeatureError("ESM overlap weights do not sum to one per residue")
    return coverage, weights


def _esm_feature_arrays(
    keys: Sequence[tuple[int, str]],
    peptides: Sequence[bytes],
    model_dir: Path,
    *,
    device_name: str,
    batch_size: int,
    max_residues: int,
    overlap: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Extract full-protein ESM features with overlap-corrected residue means."""

    if batch_size < 1:
        raise SequenceFeatureError("ESM batch size must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:
        raise SequenceFeatureError(
            "torch and transformers are required for ESM extraction"
        ) from exc

    if device_name == "cuda" and not torch.cuda.is_available():
        raise SequenceFeatureError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_dir,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float32,
    )
    model.eval().to(device)
    if int(model.config.hidden_size) != ESM_HIDDEN_SIZE:
        raise SequenceFeatureError("loaded ESM hidden size mismatch")

    tokenizer_probe = tokenizer(
        "MA", add_special_tokens=True, return_special_tokens_mask=True
    )
    if len(tokenizer_probe["input_ids"]) != 4 or tokenizer_probe[
        "special_tokens_mask"
    ] != [1, 0, 0, 1]:
        raise SequenceFeatureError(
            "loaded ESM tokenizer special-token contract mismatch"
        )

    unique_peptides: list[bytes] = []
    unique_by_peptide: dict[bytes, int] = {}
    entity_unique_indices: list[int] = []
    for peptide in peptides:
        unique_index = unique_by_peptide.get(peptide)
        if unique_index is None:
            unique_index = len(unique_peptides)
            unique_by_peptide[peptide] = unique_index
            unique_peptides.append(peptide)
        entity_unique_indices.append(unique_index)

    all_windows: list[list[tuple[int, int]]] = []
    all_weights: list[list[np.ndarray]] = []
    tasks: list[tuple[int, int, int, int]] = []
    long_proteins = 0
    max_length = 0
    max_coverage = 0
    for unique_index, peptide in enumerate(unique_peptides):
        length = len(peptide)
        max_length = max(max_length, length)
        windows = chunk_windows(length, max_residues, overlap)
        coverage, weights = window_inverse_coverage(length, windows)
        max_coverage = max(max_coverage, int(coverage.max()))
        long_proteins += int(length > max_residues)
        all_windows.append(windows)
        all_weights.append(weights)
        for window_index, (start, end) in enumerate(windows):
            tasks.append((unique_index, window_index, start, end))
    tasks.sort(key=lambda item: (-(item[3] - item[2]), item[0], item[2]))

    embedding_sums = np.zeros((len(unique_peptides), ESM_HIDDEN_SIZE), dtype=np.float64)
    started = time.monotonic()
    batch_count = (len(tasks) + batch_size - 1) // batch_size
    with torch.inference_mode():
        for batch_index, offset in enumerate(range(0, len(tasks), batch_size), start=1):
            batch_tasks = tasks[offset : offset + batch_size]
            batch_sequences = [
                unique_peptides[unique_index][start:end].decode("ascii")
                for unique_index, _, start, end in batch_tasks
            ]
            encoded = tokenizer(
                batch_sequences,
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_attention_mask=True,
                return_tensors="pt",
            )
            expected_tokens = [len(sequence) + 2 for sequence in batch_sequences]
            actual_tokens = encoded["attention_mask"].sum(dim=1).tolist()
            if (
                actual_tokens != expected_tokens
                or max(actual_tokens) > max_residues + 2
            ):
                raise SequenceFeatureError(
                    "ESM tokenizer truncated or expanded a peptide window"
                )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            hidden = (
                model.esm(**encoded).last_hidden_state.detach().float().cpu().numpy()
            )
            for row, (unique_index, window_index, start, end) in enumerate(batch_tasks):
                residue_count = end - start
                token_values = hidden[row, 1 : residue_count + 1]
                if token_values.shape != (residue_count, ESM_HIDDEN_SIZE):
                    raise SequenceFeatureError("ESM output/token alignment mismatch")
                weights = all_weights[unique_index][window_index]
                embedding_sums[unique_index] += np.sum(
                    token_values.astype(np.float64) * weights[:, None], axis=0
                )
            if batch_index == 1 or batch_index % 25 == 0 or batch_index == batch_count:
                elapsed = time.monotonic() - started
                print(
                    f"ESM batches {batch_index}/{batch_count}; elapsed={elapsed:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )

    unique_embeddings = np.empty(
        (len(unique_peptides), ESM_HIDDEN_SIZE), dtype=np.dtype("<f4")
    )
    for index, peptide in enumerate(unique_peptides):
        unique_embeddings[index] = (
            embedding_sums[index] / np.float64(len(peptide))
        ).astype(np.float32)
    values = unique_embeddings[np.asarray(entity_unique_indices, dtype=np.int64)]
    if not np.isfinite(values).all():
        raise SequenceFeatureError("non-finite ESM embedding generated")
    max_id_chars = max(len(entity_id) for _, entity_id in keys)
    arrays = {
        "feature_values": values.astype(np.dtype("<f4"), copy=False),
        "entity_taxon": np.asarray([taxon for taxon, _ in keys], dtype=np.dtype("<i8")),
        "entity_id": np.asarray(
            [entity_id for _, entity_id in keys], dtype=f"<U{max_id_chars}"
        ),
    }
    statistics: dict[str, object] = {
        "entities": len(keys),
        "uniqueCanonicalPeptides": len(unique_peptides),
        "windowCount": len(tasks),
        "uniquePeptidesRequiringMultipleWindows": long_proteins,
        "maximumPeptideLength": max_length,
        "maximumWindowCoveragePerResidue": max_coverage,
        "deviceType": device.type,
        "deviceName": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "torchVersion": torch.__version__,
    }
    return arrays, statistics


def build_esm_artifact(
    fasta_path: Path,
    feature_block_path: Path,
    model_dir: Path,
    output_dir: Path,
    *,
    device_name: str,
    batch_size: int,
    max_residues: int,
    overlap: int,
) -> dict[str, object]:
    model_files = verify_esm_model_dir(model_dir)
    sequences = parse_pinned_fasta(fasta_path)
    entities, provenance, _ = load_pinned_feature_provenance(feature_block_path)
    keys, peptides = resolve_entity_peptides(sequences, entities, provenance)
    arrays, statistics = _esm_feature_arrays(
        keys,
        peptides,
        model_dir,
        device_name=device_name,
        batch_size=batch_size,
        max_residues=max_residues,
        overlap=overlap,
    )
    npz_payload = deterministic_npz_bytes(arrays)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / ESM_NPZ_NAME
    manifest_path = output_dir / ESM_MANIFEST_NAME
    if npz_path.exists() or manifest_path.exists():
        raise SequenceFeatureError(
            f"refusing to overwrite existing output in {output_dir}"
        )
    npz_path.write_bytes(npz_payload)
    manifest: dict[str, object] = {
        "schema": ESM_OUTPUT_SCHEMA,
        "identity": {
            "key": ["ncbiTaxon", "entityId"],
            "ordering": "ascending-ncbiTaxon-then-codepoint-entityId",
            "entityKeySetSha256": EXPECTED_ENTITY_KEY_SET_SHA256,
            "rows": len(keys),
            "speciesTaxon": SPECIES_TAXON,
            "sourceStrainTaxon": SOURCE_STRAIN_TAXON,
        },
        "arrays": {
            "feature_values": {
                "dtype": "little-endian-float32",
                "shape": list(arrays["feature_values"].shape),
            },
            "entity_taxon": {
                "dtype": "little-endian-int64",
                "shape": list(arrays["entity_taxon"].shape),
            },
            "entity_id": {
                "dtype": str(arrays["entity_id"].dtype),
                "shape": list(arrays["entity_id"].shape),
                "pickleRequired": False,
            },
        },
        "featureDefinition": {
            "dimension": ESM_HIDDEN_SIZE,
            "formula": (
                "For each residue, average final-layer ESM representations over all "
                "overlapping windows containing that residue; then take the uniform "
                "mean over every residue in the full canonical peptide."
            ),
            "maxResiduesPerWindow": max_residues,
            "requestedOverlapResidues": overlap,
            "truncation": False,
            "specialTokensExcluded": True,
            "fittedOnSlpOutcomes": False,
            "identifierFeatures": False,
        },
        "model": {
            "repository": ESM_REPOSITORY,
            "revision": ESM_REVISION,
            "license": ESM_LICENSE,
            "licenseEvidence": [
                "https://huggingface.co/facebook/esm2_t6_8M_UR50D/tree/" + ESM_REVISION,
                "https://github.com/facebookresearch/esm/blob/main/LICENSE",
            ],
            "files": model_files,
            "architecture": "ESM-2 t6 8M UR50D",
            "finalLayer": 6,
            "hiddenSize": ESM_HIDDEN_SIZE,
            "parametersApproximate": 8_000_000,
        },
        "inputs": {
            "sgdProteinSequences": {
                "resource": SEQUENCE_RESOURCE,
                "manifestDigest": SEQUENCE_MANIFEST_DIGEST,
                "gzipSha256": EXPECTED_FASTA_GZIP_SHA256,
                "decompressedSha256": EXPECTED_FASTA_SHA256,
                "license": "CC-BY-4.0",
            },
            "sequenceStatisticsFeatureBlock": {
                "archiveSha256": EXPECTED_FEATURE_BLOCK_SHA256,
                "entityRowsSha256": EXPECTED_ENTITY_ROWS_SHA256,
                "sequenceProvenanceSha256": EXPECTED_SEQUENCE_PROVENANCE_SHA256,
                "staticEntityUniverseResource": STATIC_UNIVERSE_RESOURCE,
            },
        },
        "coverage": statistics,
        "runtime": {
            "batchSize": batch_size,
            "deterministicAlgorithms": True,
            "float32ModelInference": True,
            "float64OverlapAccumulation": True,
            "tf32": False,
        },
        "accessBoundary": {
            "staticSequencesConsumed": True,
            "heldRosterConsumed": False,
            "quantitativeOutcomesConsumed": False,
            "trainingPartitionAssignmentsConsumed": False,
            "benchmarkDataConsumed": False,
        },
        "artifact": {
            "path": ESM_NPZ_NAME,
            "bytes": len(npz_payload),
            "sha256": sha256_bytes(npz_payload),
            "compression": "zip-deflate-level-9-fixed-metadata",
        },
        "limitations": [
            "Chunked representations approximate a single pass for proteins longer than the model context; overlapping contexts are averaged per residue.",
            "ESM-2 was pretrained externally on UR50D; this artifact does not establish downstream benefit or training-set independence for any specific sequence.",
            "Static sequences include held entities but no held quantitative intervention outcome.",
            "This exploratory artifact has not been admitted as an OMF DatasetSnapshot.",
        ],
    }
    manifest_path.write_bytes(_canonical_json(manifest))
    return manifest


def build_artifact(
    fasta_path: Path, feature_block_path: Path, output_dir: Path
) -> dict[str, object]:
    sequences = parse_pinned_fasta(fasta_path)
    entities, provenance, source_manifest = load_pinned_feature_provenance(
        feature_block_path
    )
    arrays = build_feature_arrays(sequences, entities, provenance)
    npz_payload = deterministic_npz_bytes(arrays)

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / NPZ_NAME
    manifest_path = output_dir / MANIFEST_NAME
    if npz_path.exists() or manifest_path.exists():
        raise SequenceFeatureError(
            f"refusing to overwrite existing output in {output_dir}"
        )
    npz_path.write_bytes(npz_payload)
    values = arrays["feature_values"]
    manifest: dict[str, object] = {
        "schema": OUTPUT_SCHEMA,
        "identity": {
            "key": ["ncbiTaxon", "entityId"],
            "ordering": "ascending-ncbiTaxon-then-codepoint-entityId",
            "entityKeySetSha256": EXPECTED_ENTITY_KEY_SET_SHA256,
            "rows": int(values.shape[0]),
            "speciesTaxon": SPECIES_TAXON,
            "sourceStrainTaxon": SOURCE_STRAIN_TAXON,
        },
        "arrays": {
            "feature_values": {
                "dtype": "little-endian-float32",
                "shape": list(values.shape),
            },
            "entity_taxon": {
                "dtype": "little-endian-int64",
                "shape": list(arrays["entity_taxon"].shape),
            },
            "entity_id": {
                "dtype": str(arrays["entity_id"].dtype),
                "shape": list(arrays["entity_id"].shape),
                "pickleRequired": False,
            },
        },
        "featureDefinition": {
            "dimension": len(FEATURE_NAMES),
            "names": list(FEATURE_NAMES),
            "aminoAcidOrder": AA_ORDER,
            "dipeptideOrder": list(DIPEPTIDE_ORDER),
            "formula": (
                "[len(peptide)/4096] + "
                "[count(aa)/len(peptide) for aa in aminoAcidOrder] + "
                "[count(pair)/max(len(peptide)-1,1) for pair in dipeptideOrder]"
            ),
            "stopPolicy": "strip exactly one validated terminal stop",
            "fitted": False,
            "trainableParameters": 0,
            "identifierFeatures": False,
            "learnedModel": None,
        },
        "inputs": {
            "sgdProteinSequences": {
                "resource": SEQUENCE_RESOURCE,
                "manifestDigest": SEQUENCE_MANIFEST_DIGEST,
                "gzipBytes": EXPECTED_FASTA_GZIP_BYTES,
                "gzipSha256": EXPECTED_FASTA_GZIP_SHA256,
                "decompressedBytes": EXPECTED_FASTA_BYTES,
                "decompressedSha256": EXPECTED_FASTA_SHA256,
                "license": "CC-BY-4.0",
                "rightsDeclaration": "rights/sgd-protein-sequences-r64-5-1-cc-by-4.0.yaml",
            },
            "sequenceStatisticsFeatureBlock": {
                "archiveBytes": EXPECTED_FEATURE_BLOCK_BYTES,
                "archiveSha256": EXPECTED_FEATURE_BLOCK_SHA256,
                "schema": source_manifest["schema"],
                "staticEntityUniverseResource": STATIC_UNIVERSE_RESOURCE,
                "entityRowsSha256": EXPECTED_ENTITY_ROWS_SHA256,
                "sequenceProvenanceSha256": EXPECTED_SEQUENCE_PROVENANCE_SHA256,
            },
        },
        "accessBoundary": {
            "staticSequencesConsumed": True,
            "heldRosterConsumed": False,
            "quantitativeOutcomesConsumed": False,
            "trainingPartitionAssignmentsConsumed": False,
            "benchmarkDataConsumed": False,
            "freeTextDescriptionsUsedAsFeatures": False,
        },
        "artifact": {
            "path": NPZ_NAME,
            "bytes": len(npz_payload),
            "sha256": sha256_bytes(npz_payload),
            "compression": "zip-deflate-level-9-fixed-metadata",
        },
        "limitations": [
            "Dipeptide frequencies retain local residue order only; they do not encode long-range structure or function.",
            "Static sequences include held entities but no held quantitative intervention outcome.",
            "This exploratory artifact has not been admitted as an OMF DatasetSnapshot.",
        ],
    }
    manifest_path.write_bytes(_canonical_json(manifest))
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dipeptide", "esm"), default="dipeptide")
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--feature-block", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--esm-model-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-residues", type=int, default=ESM_MAX_RESIDUES)
    parser.add_argument("--overlap", type=int, default=ESM_DEFAULT_OVERLAP)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "dipeptide":
            manifest = build_artifact(args.fasta, args.feature_block, args.output_dir)
        else:
            if args.esm_model_dir is None:
                raise SequenceFeatureError("--esm-model-dir is required in ESM mode")
            manifest = build_esm_artifact(
                args.fasta,
                args.feature_block,
                args.esm_model_dir,
                args.output_dir,
                device_name=args.device,
                batch_size=args.batch_size,
                max_residues=args.max_residues,
                overlap=args.overlap,
            )
    except (OSError, SequenceFeatureError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
