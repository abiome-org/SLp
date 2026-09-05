#!/usr/bin/env python3
"""Build composite-keyed human ESM-2 features from static metadata and sequences."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import sys
import time
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPECIES_TAXON = 9606
ENTITY_COUNT = 7_542
ENTITY_LIST_BYTES = 120_672
ENTITY_LIST_SHA256 = "c6836645dcfc24788f2c06110ddc08ee4949d97f710dd117db12db1949d9b33e"
QUERY_COUNT = 7_226
QUERY_LIST_SHA256 = "645b8d563b440a4b7ab6a3bb42450594b408c4e7cb84e4fe2789a6620174f12c"
ACTION_COUNT = 2_392
ACTION_LIST_SHA256 = "2884efd414949bfc3c7dc5f376aa69f0470080afdcab255b4a88f67cc53ac9ed"
ENTITY_RE = re.compile(r"^ENSG[0-9]+$")

ENSEMBL_RELEASE = 116
ENSEMBL_ASSEMBLY = "GRCh38.p14"
ENSEMBL_FASTA_NAME = "Homo_sapiens.GRCh38.pep.all.fa.gz"
ENSEMBL_FASTA_BYTES = 23_319_936
ENSEMBL_FASTA_SHA256 = (
    "9b43da92651b35814597af6a8b18f500b768679a49fa4678224f384917ce7668"
)
ENSEMBL_FASTA_BSD_SUM = 21_612
ENSEMBL_FASTA_BLOCKS = 22_774
ENSEMBL_DECOMPRESSED_BYTES = 266_769_619
ENSEMBL_DECOMPRESSED_SHA256 = (
    "3f1ef9848ae79d3810ef5c7bff3482d7fb0554618adf7f3655828e918f50a7c5"
)
ENSEMBL_FASTA_RECORDS = 382_428
ENSEMBL_STABLE_GENES = 23_879
CHECKSUMS_SPEC = (
    112,
    "2b5910595a13a02e17a94271a5169a1b329d81018e10f72a0f7cc9c46ae9b7f2",
)
README_SPEC = (
    2_432,
    "e3b45d40a0ad5fd0240e6f92db4b46ab5518abe77c2fa3d35d770a5f435fd9a3",
)
FASTA_HEADER_RE = re.compile(
    rb"^>(ENSP[0-9]+)\.([0-9]+) pep [^ ]+ "
    rb"gene:(ENSG[0-9]+)\.([0-9]+) "
    rb"transcript:(ENST[0-9]+)\.([0-9]+)(?: |$)"
)
SOURCE_ALPHABET = frozenset(b"ACDEFGHIKLMNPQRSTVWYXBU*")
ESM_NORMALIZED_ALPHABET = frozenset(b"ACDEFGHIKLMNPQRSTVWYXBU")

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

NPZ_NAME = "human-sequence-esm2-features.npz"
MANIFEST_NAME = "human-sequence-esm2-features.manifest.json"
PROVENANCE_NAME = "selected-protein-provenance.jsonl"
OUTPUT_SCHEMA = "slp.human-sequence-esm2-feature-artifact/v1"


class HumanSequenceFeatureError(ValueError):
    """Raised when a pinned source or deterministic feature contract fails."""


@dataclass(frozen=True)
class Translation:
    gene_id: str
    gene_version: int
    transcript_id: str
    transcript_version: int
    protein_id: str
    protein_version: int
    peptide: bytes

    @property
    def selection_key(self) -> tuple[int, str, str]:
        return -len(self.peptide), self.transcript_id, self.protein_id


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def require_file(
    path: Path, expected_bytes: int, expected_sha256: str, label: str
) -> None:
    if not path.is_file():
        raise HumanSequenceFeatureError(f"{label} is not a regular file: {path}")
    size = path.stat().st_size
    if size != expected_bytes:
        raise HumanSequenceFeatureError(
            f"{label} byte count mismatch: expected {expected_bytes}, found {size}"
        )
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise HumanSequenceFeatureError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, found {digest}"
        )


def bsd_sum(path: Path) -> tuple[int, int]:
    checksum = 0
    byte_count = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            byte_count += len(chunk)
            for byte in chunk:
                checksum = (checksum >> 1) | ((checksum & 1) << 15)
                checksum = (checksum + byte) & 0xFFFF
    return checksum, (byte_count + 1023) // 1024


def load_entity_ids(path: Path) -> list[str]:
    require_file(path, ENTITY_LIST_BYTES, ENTITY_LIST_SHA256, "human entity ID list")
    payload = path.read_bytes()
    if not payload.endswith(b"\n") or b"\r" in payload:
        raise HumanSequenceFeatureError("human entity list must be LF-terminated ASCII")
    try:
        identifiers = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise HumanSequenceFeatureError("human entity list must be ASCII") from exc
    if (
        len(identifiers) != ENTITY_COUNT
        or identifiers != sorted(set(identifiers))
        or any(ENTITY_RE.fullmatch(item) is None for item in identifiers)
    ):
        raise HumanSequenceFeatureError("human entity list identity contract mismatch")
    return identifiers


def verify_source_dir(source_dir: Path) -> Path:
    fasta = source_dir / ENSEMBL_FASTA_NAME
    require_file(fasta, ENSEMBL_FASTA_BYTES, ENSEMBL_FASTA_SHA256, "Ensembl FASTA")
    require_file(source_dir / "CHECKSUMS", *CHECKSUMS_SPEC, "Ensembl CHECKSUMS")
    require_file(source_dir / "README", *README_SPEC, "Ensembl README")
    if bsd_sum(fasta) != (ENSEMBL_FASTA_BSD_SUM, ENSEMBL_FASTA_BLOCKS):
        raise HumanSequenceFeatureError("Ensembl upstream BSD checksum mismatch")
    checksum_lines = (source_dir / "CHECKSUMS").read_text(encoding="ascii").splitlines()
    expected_line = (
        f"{ENSEMBL_FASTA_BSD_SUM} {ENSEMBL_FASTA_BLOCKS} {ENSEMBL_FASTA_NAME}"
    )
    if expected_line not in checksum_lines:
        raise HumanSequenceFeatureError("Ensembl FASTA is absent from pinned CHECKSUMS")
    return fasta


def parse_longest_translations(
    fasta_path: Path,
) -> tuple[dict[str, Translation], dict[str, int]]:
    selected: dict[str, Translation] = {}
    seen_proteins: set[str] = set()
    digest = hashlib.sha256()
    decompressed_bytes = 0
    records = 0
    raw_stops = 0
    raw_x = 0
    raw_u = 0
    current_fields: tuple[bytes, bytes, bytes, bytes, bytes, bytes] | None = None
    chunks: list[bytes] = []

    def finish() -> None:
        nonlocal records, raw_stops, raw_x, raw_u, current_fields, chunks
        if current_fields is None:
            return
        protein, protein_version, gene, gene_version, transcript, transcript_version = (
            current_fields
        )
        peptide = b"".join(chunks)
        if not peptide or any(byte not in SOURCE_ALPHABET for byte in peptide):
            raise HumanSequenceFeatureError(
                f"invalid Ensembl peptide alphabet for {protein.decode('ascii')}"
            )
        protein_id = protein.decode("ascii")
        if protein_id in seen_proteins:
            raise HumanSequenceFeatureError(
                f"duplicate stable Ensembl protein {protein_id}"
            )
        seen_proteins.add(protein_id)
        candidate = Translation(
            gene_id=gene.decode("ascii"),
            gene_version=int(gene_version),
            transcript_id=transcript.decode("ascii"),
            transcript_version=int(transcript_version),
            protein_id=protein_id,
            protein_version=int(protein_version),
            peptide=peptide,
        )
        previous = selected.get(candidate.gene_id)
        if previous is None or candidate.selection_key < previous.selection_key:
            selected[candidate.gene_id] = candidate
        raw_stops += peptide.count(b"*")
        raw_x += peptide.count(b"X")
        raw_u += peptide.count(b"U")
        records += 1

    try:
        with gzip.open(fasta_path, "rb") as stream:
            for line_number, line in enumerate(stream, start=1):
                digest.update(line)
                decompressed_bytes += len(line)
                stripped = line.strip()
                if not stripped:
                    raise HumanSequenceFeatureError(
                        f"blank Ensembl FASTA line at {line_number}"
                    )
                if stripped.startswith(b">"):
                    finish()
                    match = FASTA_HEADER_RE.match(stripped)
                    if match is None:
                        raise HumanSequenceFeatureError(
                            f"invalid Ensembl header identity at line {line_number}"
                        )
                    current_fields = match.groups()
                    chunks = []
                else:
                    if current_fields is None:
                        raise HumanSequenceFeatureError(
                            "Ensembl sequence occurs before its header"
                        )
                    chunks.append(stripped)
            finish()
    except (OSError, EOFError) as exc:
        raise HumanSequenceFeatureError(
            "Ensembl FASTA gzip integrity check failed"
        ) from exc
    if (
        decompressed_bytes != ENSEMBL_DECOMPRESSED_BYTES
        or digest.hexdigest() != ENSEMBL_DECOMPRESSED_SHA256
        or records != ENSEMBL_FASTA_RECORDS
        or len(selected) != ENSEMBL_STABLE_GENES
    ):
        raise HumanSequenceFeatureError(
            "Ensembl decompressed content contract mismatch"
        )
    counts = {
        "fastaRecords": records,
        "stableGenes": len(selected),
        "sourceStopResidues": raw_stops,
        "sourceUnknownResidues": raw_x,
        "sourceSelenocysteineResidues": raw_u,
    }
    return selected, counts


def normalize_for_esm(peptide: bytes) -> bytes:
    normalized = peptide.replace(b"*", b"X")
    if not normalized or any(
        byte not in ESM_NORMALIZED_ALPHABET for byte in normalized
    ):
        raise HumanSequenceFeatureError("invalid normalized ESM peptide")
    return normalized


def verify_esm_model_dir(model_dir: Path) -> dict[str, dict[str, object]]:
    if not model_dir.is_dir():
        raise HumanSequenceFeatureError(
            f"ESM model directory does not exist: {model_dir}"
        )
    actual_files = {path.name for path in model_dir.iterdir() if path.is_file()}
    if actual_files != set(ESM_FILE_SPECS):
        raise HumanSequenceFeatureError("ESM model file set mismatch")
    verified: dict[str, dict[str, object]] = {}
    for name in sorted(ESM_FILE_SPECS):
        expected_bytes, expected_sha256 = ESM_FILE_SPECS[name]
        require_file(model_dir / name, expected_bytes, expected_sha256, f"ESM {name}")
        verified[name] = {"bytes": expected_bytes, "sha256": expected_sha256}
    config = json.loads((model_dir / "config.json").read_bytes())
    if (
        config.get("model_type") != "esm"
        or config.get("hidden_size") != ESM_HIDDEN_SIZE
        or config.get("num_hidden_layers") != 6
        or config.get("max_position_embeddings") != 1_026
    ):
        raise HumanSequenceFeatureError("pinned ESM architecture contract mismatch")
    return verified


def chunk_windows(
    length: int,
    max_residues: int = ESM_MAX_RESIDUES,
    overlap: int = ESM_DEFAULT_OVERLAP,
) -> list[tuple[int, int]]:
    if length < 1 or max_residues < 2 or overlap < 0 or overlap >= max_residues:
        raise HumanSequenceFeatureError("invalid ESM peptide/chunk contract")
    if length <= max_residues:
        return [(0, length)]
    final_start = length - max_residues
    starts = list(range(0, final_start + 1, max_residues - overlap))
    if starts[-1] != final_start:
        starts.append(final_start)
    return [(start, min(start + max_residues, length)) for start in starts]


def inverse_coverage_weights(
    length: int, windows: Sequence[tuple[int, int]]
) -> tuple[np.ndarray, list[np.ndarray]]:
    coverage = np.zeros(length, dtype=np.int32)
    for start, end in windows:
        if start < 0 or end <= start or end > length:
            raise HumanSequenceFeatureError("invalid ESM window boundary")
        coverage[start:end] += 1
    if np.any(coverage < 1):
        raise HumanSequenceFeatureError("ESM windows silently omit residues")
    weights = [
        np.reciprocal(coverage[start:end].astype(np.float64)) for start, end in windows
    ]
    check = np.zeros(length, dtype=np.float64)
    for (start, end), weight in zip(windows, weights, strict=True):
        check[start:end] += weight
    if not np.array_equal(check, np.ones(length, dtype=np.float64)):
        raise HumanSequenceFeatureError("ESM overlap weights do not sum to one")
    return coverage, weights


def extract_esm(
    peptides: Sequence[bytes],
    model_dir: Path,
    *,
    device_name: str,
    batch_size: int,
    max_residues: int,
    overlap: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if batch_size < 1:
        raise HumanSequenceFeatureError("ESM batch size must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:
        raise HumanSequenceFeatureError("torch and transformers are required") from exc
    if device_name == "cuda" and not torch.cuda.is_available():
        raise HumanSequenceFeatureError("CUDA requested but unavailable")
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
    probe = tokenizer("MA", add_special_tokens=True, return_special_tokens_mask=True)
    if len(probe["input_ids"]) != 4 or probe["special_tokens_mask"] != [1, 0, 0, 1]:
        raise HumanSequenceFeatureError("ESM tokenizer special-token contract mismatch")

    unique_peptides: list[bytes] = []
    unique_index: dict[bytes, int] = {}
    row_indices: list[int] = []
    for peptide in peptides:
        index = unique_index.get(peptide)
        if index is None:
            index = len(unique_peptides)
            unique_index[peptide] = index
            unique_peptides.append(peptide)
        row_indices.append(index)

    tasks: list[tuple[int, int, int, int]] = []
    all_weights: list[list[np.ndarray]] = []
    long_peptides = 0
    maximum_length = 0
    maximum_coverage = 0
    for unique, peptide in enumerate(unique_peptides):
        windows = chunk_windows(len(peptide), max_residues, overlap)
        coverage, weights = inverse_coverage_weights(len(peptide), windows)
        all_weights.append(weights)
        maximum_length = max(maximum_length, len(peptide))
        maximum_coverage = max(maximum_coverage, int(coverage.max()))
        long_peptides += int(len(peptide) > max_residues)
        for window_index, (start, end) in enumerate(windows):
            tasks.append((unique, window_index, start, end))
    tasks.sort(key=lambda item: (-(item[3] - item[2]), item[0], item[2]))
    sums = np.zeros((len(unique_peptides), ESM_HIDDEN_SIZE), dtype=np.float64)
    started = time.monotonic()
    batches = (len(tasks) + batch_size - 1) // batch_size
    with torch.inference_mode():
        for batch_index, offset in enumerate(range(0, len(tasks), batch_size), start=1):
            batch_tasks = tasks[offset : offset + batch_size]
            sequences = [
                unique_peptides[unique][start:end].decode("ascii")
                for unique, _, start, end in batch_tasks
            ]
            encoded = tokenizer(
                sequences,
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_attention_mask=True,
                return_tensors="pt",
            )
            expected_tokens = [len(sequence) + 2 for sequence in sequences]
            actual_tokens = encoded["attention_mask"].sum(dim=1).tolist()
            if (
                actual_tokens != expected_tokens
                or max(actual_tokens) > max_residues + 2
            ):
                raise HumanSequenceFeatureError(
                    "ESM tokenizer truncated a peptide window"
                )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            hidden = (
                model.esm(**encoded).last_hidden_state.detach().float().cpu().numpy()
            )
            for row, (unique, window_index, start, end) in enumerate(batch_tasks):
                residue_count = end - start
                token_values = hidden[row, 1 : residue_count + 1]
                if token_values.shape != (residue_count, ESM_HIDDEN_SIZE):
                    raise HumanSequenceFeatureError(
                        "ESM token/output alignment mismatch"
                    )
                sums[unique] += np.sum(
                    token_values.astype(np.float64)
                    * all_weights[unique][window_index][:, None],
                    axis=0,
                )
            if batch_index == 1 or batch_index % 25 == 0 or batch_index == batches:
                print(
                    f"human ESM batches {batch_index}/{batches}; "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
    embeddings = np.empty(
        (len(unique_peptides), ESM_HIDDEN_SIZE), dtype=np.dtype("<f4")
    )
    for index, peptide in enumerate(unique_peptides):
        embeddings[index] = (sums[index] / np.float64(len(peptide))).astype(np.float32)
    values = embeddings[np.asarray(row_indices, dtype=np.int64)]
    if not np.isfinite(values).all():
        raise HumanSequenceFeatureError("non-finite ESM feature generated")
    stats: dict[str, object] = {
        "presentEntityPeptides": len(peptides),
        "uniqueNormalizedPeptides": len(unique_peptides),
        "windowCount": len(tasks),
        "uniquePeptidesRequiringMultipleWindows": long_peptides,
        "maximumPeptideLength": maximum_length,
        "maximumWindowCoveragePerResidue": maximum_coverage,
        "deviceType": device.type,
        "deviceName": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU",
        "torchVersion": torch.__version__,
    }
    return values, stats


def npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, allow_pickle=False)
    return stream.getvalue()


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    expected = ("feature_values", "entity_taxon", "entity_id")
    if tuple(arrays) != expected:
        raise HumanSequenceFeatureError(f"NPZ arrays must be ordered as {expected}")
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in expected:
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, npy_bytes(arrays[name]), compresslevel=9)
    return output.getvalue()


def build_artifact(
    entity_list: Path,
    source_dir: Path,
    model_dir: Path,
    output_dir: Path,
    *,
    device_name: str,
    batch_size: int,
    max_residues: int,
    overlap: int,
) -> dict[str, object]:
    identifiers = load_entity_ids(entity_list)
    fasta_path = verify_source_dir(source_dir)
    model_files = verify_esm_model_dir(model_dir)
    translations, source_counts = parse_longest_translations(fasta_path)

    present_ids = [item for item in identifiers if item in translations]
    missing_ids = [item for item in identifiers if item not in translations]
    normalized = [normalize_for_esm(translations[item].peptide) for item in present_ids]
    embeddings, coverage = extract_esm(
        normalized,
        model_dir,
        device_name=device_name,
        batch_size=batch_size,
        max_residues=max_residues,
        overlap=overlap,
    )

    values = np.zeros((len(identifiers), ESM_HIDDEN_SIZE + 1), dtype=np.dtype("<f4"))
    row_by_id = {item: index for index, item in enumerate(identifiers)}
    for embedding, entity_id in zip(embeddings, present_ids, strict=True):
        row = row_by_id[entity_id]
        values[row, :ESM_HIDDEN_SIZE] = embedding
        values[row, ESM_HIDDEN_SIZE] = np.float32(1.0)
    taxa = np.full(len(identifiers), SPECIES_TAXON, dtype=np.dtype("<i8"))
    entity_ids = np.asarray(identifiers, dtype="<U15")
    arrays = {
        "feature_values": values,
        "entity_taxon": taxa,
        "entity_id": entity_ids,
    }
    npz_payload = deterministic_npz_bytes(arrays)

    provenance_rows: list[dict[str, object]] = []
    selected_stop_residues = 0
    for entity_id in present_ids:
        selected = translations[entity_id]
        normalized_peptide = normalize_for_esm(selected.peptide)
        stop_count = selected.peptide.count(b"*")
        selected_stop_residues += stop_count
        provenance_rows.append(
            {
                "entityId": entity_id,
                "ncbiTaxon": SPECIES_TAXON,
                "sourceGeneId": f"{selected.gene_id}.{selected.gene_version}",
                "selectedTranscriptId": (
                    f"{selected.transcript_id}.{selected.transcript_version}"
                ),
                "selectedProteinId": f"{selected.protein_id}.{selected.protein_version}",
                "selectionRule": "longest-then-stable-transcript-then-stable-protein",
                "peptideLength": len(selected.peptide),
                "sourcePeptideSha256": sha256_bytes(selected.peptide),
                "esmPeptideSha256": sha256_bytes(normalized_peptide),
                "stopResiduesMappedToX": stop_count,
            }
        )
    provenance_payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
        for row in provenance_rows
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / NPZ_NAME,
        output_dir / MANIFEST_NAME,
        output_dir / PROVENANCE_NAME,
    ]
    if any(path.exists() for path in paths):
        raise HumanSequenceFeatureError(f"refusing to overwrite output in {output_dir}")
    paths[0].write_bytes(npz_payload)
    paths[2].write_bytes(provenance_payload)
    manifest: dict[str, object] = {
        "schema": OUTPUT_SCHEMA,
        "identity": {
            "key": ["ncbiTaxon", "entityId"],
            "ordering": "ascending-ncbiTaxon-then-codepoint-entityId",
            "rows": len(identifiers),
            "ncbiTaxon": SPECIES_TAXON,
            "entityNamespace": "Ensembl-gene",
            "entityIdList": {
                "bytes": ENTITY_LIST_BYTES,
                "sha256": ENTITY_LIST_SHA256,
                "count": ENTITY_COUNT,
            },
            "queryIdList": {"sha256": QUERY_LIST_SHA256, "count": QUERY_COUNT},
            "actionIdList": {"sha256": ACTION_LIST_SHA256, "count": ACTION_COUNT},
        },
        "arrays": {
            "feature_values": {
                "dtype": "little-endian-float32",
                "shape": list(values.shape),
            },
            "entity_taxon": {
                "dtype": "little-endian-int64",
                "shape": list(taxa.shape),
            },
            "entity_id": {
                "dtype": str(entity_ids.dtype),
                "shape": list(entity_ids.shape),
                "pickleRequired": False,
            },
        },
        "featureDefinition": {
            "dimension": ESM_HIDDEN_SIZE + 1,
            "embeddingColumns": [0, ESM_HIDDEN_SIZE - 1],
            "proteinPresentColumn": ESM_HIDDEN_SIZE,
            "missingProteinRule": "zero-320-vector-plus-zero-presence-flag",
            "presentProteinRule": "ESM-320-vector-plus-one-presence-flag",
            "poolingFormula": (
                "Average each residue's final-layer ESM representation across all "
                "overlapping windows containing it, then uniformly mean every "
                "residue in the full selected peptide."
            ),
            "maxResiduesPerWindow": max_residues,
            "requestedOverlapResidues": overlap,
            "truncation": False,
            "specialTokensExcluded": True,
            "sourceStopNormalization": "map '*' to ESM residue X",
            "fittedOnSlpOutcomes": False,
            "identifierFeatures": False,
        },
        "coverage": {
            **coverage,
            "presentEntities": len(present_ids),
            "missingProteinEntities": len(missing_ids),
            "missingProteinEntityIds": missing_ids,
            "selectedSourceStopResiduesMappedToX": selected_stop_residues,
        },
        "selection": {
            "rule": (
                "greatest peptide length, then ascending stable ENST transcript ID, "
                "then ascending stable ENSP protein ID"
            ),
            "displaySymbolsUsed": False,
            "sourceVersionSuffixRemovedOnlyForStableIdJoin": True,
            "provenance": {
                "path": PROVENANCE_NAME,
                "bytes": len(provenance_payload),
                "records": len(provenance_rows),
                "sha256": sha256_bytes(provenance_payload),
            },
        },
        "source": {
            "manifest": "sources/ensembl-human-protein-sequences-release-116.yaml",
            "rights": "rights/ensembl-human-protein-sequences-release-116.yaml",
            "release": ENSEMBL_RELEASE,
            "assembly": ENSEMBL_ASSEMBLY,
            "fasta": {
                "name": ENSEMBL_FASTA_NAME,
                "bytes": ENSEMBL_FASTA_BYTES,
                "sha256": ENSEMBL_FASTA_SHA256,
                "upstreamBsdSum": ENSEMBL_FASTA_BSD_SUM,
                "upstreamBlocks1024": ENSEMBL_FASTA_BLOCKS,
                "decompressedBytes": ENSEMBL_DECOMPRESSED_BYTES,
                "decompressedSha256": ENSEMBL_DECOMPRESSED_SHA256,
            },
            "counts": source_counts,
        },
        "model": {
            "repository": ESM_REPOSITORY,
            "revision": ESM_REVISION,
            "license": ESM_LICENSE,
            "files": model_files,
            "architecture": "ESM-2 t6 8M UR50D",
            "finalLayer": 6,
            "hiddenSize": ESM_HIDDEN_SIZE,
        },
        "runtime": {
            "batchSize": batch_size,
            "deterministicAlgorithms": True,
            "float32ModelInference": True,
            "float64OverlapAccumulation": True,
            "tf32": False,
        },
        "accessBoundary": {
            "h5adMetadataIdentityListsConsumed": True,
            "staticProteinSequencesConsumed": True,
            "quantitativeMatricesConsumed": False,
            "molecularOutcomesConsumed": False,
            "trainingPartitionAssignmentsConsumed": False,
            "benchmarkDataConsumed": False,
        },
        "artifact": {
            "path": NPZ_NAME,
            "bytes": len(npz_payload),
            "sha256": sha256_bytes(npz_payload),
            "compression": "zip-deflate-level-9-fixed-metadata",
        },
        "limitations": [
            "Longest translated peptide is a deterministic gene representation and may not match the expressed isoform in either cell line.",
            "Chunked representations approximate a single pass for proteins exceeding the ESM context; overlapping contexts are averaged per residue.",
            "ESM-2 was pretrained externally on UR50D; per-sequence pretraining membership was not audited.",
            "Missing translated proteins are retained explicitly and provide no sequence embedding signal.",
            "This exploratory artifact is not an admitted OMF DatasetSnapshot.",
        ],
    }
    paths[1].write_bytes(canonical_json(manifest))
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--entity-list", type=Path, required=True)
    result.add_argument("--source-dir", type=Path, required=True)
    result.add_argument("--esm-model-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--max-residues", type=int, default=ESM_MAX_RESIDUES)
    result.add_argument("--overlap", type=int, default=ESM_DEFAULT_OVERLAP)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        manifest = build_artifact(
            args.entity_list,
            args.source_dir,
            args.esm_model_dir,
            args.output_dir,
            device_name=args.device,
            batch_size=args.batch_size,
            max_residues=args.max_residues,
            overlap=args.overlap,
        )
    except (OSError, HumanSequenceFeatureError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
