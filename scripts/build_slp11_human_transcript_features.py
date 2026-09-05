"""Build static Ensembl 116 transcript 4-mer features by stable ENSG identity."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/sources/ensembl116-human-transcripts-v1"
DEFAULT_OUTPUT = ROOT / "data/derived/slp11-human-transcript-sequence/ensembl116-kmer4-v1"
FRANGIEH = ROOT / "data/derived/slp11-frangieh/paired-development-v1"
REFERENCE = ROOT / "results/slp11-transition/frangieh-paired-state-physical1156-seed731-v2/reference.npz"
FRANGIEH_METADATA_AUDIT = ROOT / "data/sources/frangieh-2021-scp1064-v1/h5ad-metadata-audit.json"
RIGHTS_RECORD = ROOT / "rights/ensembl-human-transcripts-release-116.yaml"
SOURCE_RECORD = ROOT / "sources/ensembl-human-transcripts-release-116.yaml"
FEATURE_NAME = "human-transcript-kmer4-features.npz"
GENE_RE = re.compile(r"^gene:(ENSG[0-9]{11})(?:\.([0-9]+))?$")
TRANSCRIPT_RE = re.compile(r"^(ENST[0-9]{11})(?:\.([0-9]+))?$")
BASES = "ACGT"
KMER_ORDER = tuple(a + b + c + d for a in BASES for b in BASES for c in BASES for d in BASES)


@dataclass(frozen=True)
class Transcript:
    gene_id: str
    gene_versioned_id: str
    transcript_id: str
    transcript_stable_id: str
    strand: str
    source_kind: str
    sequence: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bsd_sum(path: Path) -> tuple[int, int]:
    checksum = 0
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            size += len(block)
            for value in block:
                checksum = ((checksum >> 1) | ((checksum & 1) << 15))
                checksum = (checksum + value) & 0xFFFF
    return checksum, (size + 1023) // 1024


def parse_bsd_checksums(path: Path) -> dict[str, tuple[int, int]]:
    result = {}
    for line in path.read_text(encoding="ascii").splitlines():
        parts = line.split()
        if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError(f"malformed BSD checksum line: {line!r}")
        result[parts[2]] = (int(parts[0]), int(parts[1]))
    return result


def verify_sources(source_dir: Path) -> dict:
    manifest_path = source_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified = {}
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"source pin mismatch: {relative}")
        verified[str(relative).replace("\\", "/")] = {
            "bytes": entry["bytes"], "sha256": entry["sha256"]
        }
    upstream = {}
    for kind, fasta_name in (
        ("cdna", "Homo_sapiens.GRCh38.cdna.all.fa.gz"),
        ("ncrna", "Homo_sapiens.GRCh38.ncrna.fa.gz"),
    ):
        checksums = parse_bsd_checksums(source_dir / kind / "CHECKSUMS")
        for name in (fasta_name, "README"):
            path = source_dir / kind / name
            expected = checksums[name]
            actual = bsd_sum(path)
            if actual != expected:
                raise ValueError(f"upstream BSD checksum mismatch: {path}")
            upstream[f"{kind}/{name}"] = {"bsd_checksum": actual[0], "blocks_1024": actual[1]}
    return {
        "acquisition_manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "acquisition_manifest_sha256": sha256(manifest_path),
        "rights_record": str(RIGHTS_RECORD.relative_to(ROOT)).replace("\\", "/"),
        "rights_record_sha256": sha256(RIGHTS_RECORD),
        "source_record": str(SOURCE_RECORD.relative_to(ROOT)).replace("\\", "/"),
        "source_record_sha256": sha256(SOURCE_RECORD),
        "files": verified,
        "upstream_bsd_checksums": upstream,
    }


def parse_header(header: str, source_kind: str, sequence: str) -> Transcript:
    parts = header.split()
    if len(parts) < 4:
        raise ValueError("short Ensembl FASTA header")
    transcript_match = TRANSCRIPT_RE.fullmatch(parts[0])
    gene_token = next((part for part in parts if part.startswith("gene:")), "")
    gene_match = GENE_RE.fullmatch(gene_token)
    if transcript_match is None or gene_match is None:
        raise ValueError("header lacks stable versioned ENST/ENSG identity")
    location = parts[2]
    strand_match = re.search(r":(-?1)$", location)
    strand = strand_match.group(1) if strand_match else "unknown"
    return Transcript(
        gene_id=gene_match.group(1),
        gene_versioned_id=gene_token.removeprefix("gene:"),
        transcript_id=parts[0],
        transcript_stable_id=transcript_match.group(1),
        strand=strand,
        source_kind=source_kind,
        sequence=sequence,
    )


def iter_fasta(path: Path, source_kind: str):
    header: str | None = None
    pieces: list[str] = []
    with gzip.open(path, "rt", encoding="ascii", newline=None) as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\r\n")
            if line.startswith(">"):
                if header is not None:
                    yield parse_header(header, source_kind, "".join(pieces).upper())
                header = line[1:]
                pieces = []
            elif header is None:
                raise ValueError("sequence precedes first FASTA header")
            else:
                pieces.append(line)
    if header is not None:
        yield parse_header(header, source_kind, "".join(pieces).upper())


def choose_longest(current: Transcript | None, candidate: Transcript) -> Transcript:
    if current is None or len(candidate.sequence) > len(current.sequence):
        return candidate
    if len(candidate.sequence) < len(current.sequence):
        return current
    candidate_key = (candidate.transcript_stable_id, candidate.transcript_id)
    current_key = (current.transcript_stable_id, current.transcript_id)
    return candidate if candidate_key < current_key else current


def select_transcripts(source_dir: Path) -> tuple[dict[str, Transcript], dict]:
    selected: dict[str, Transcript] = {}
    statistics = {"cdna": 0, "ncrna": 0, "candidate_bases": 0, "replacements": 0}
    for source_kind, name in (
        ("cdna", "Homo_sapiens.GRCh38.cdna.all.fa.gz"),
        ("ncrna", "Homo_sapiens.GRCh38.ncrna.fa.gz"),
    ):
        for candidate in iter_fasta(source_dir / source_kind / name, source_kind):
            statistics[source_kind] += 1
            statistics["candidate_bases"] += len(candidate.sequence)
            current = selected.get(candidate.gene_id)
            chosen = choose_longest(current, candidate)
            if current is not None and chosen is candidate:
                statistics["replacements"] += 1
            selected[candidate.gene_id] = chosen
    return selected, statistics


def transcript_features(sequence: str) -> np.ndarray:
    raw = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    lookup = np.full(256, -1, dtype=np.int16)
    lookup[np.frombuffer(b"ACGT", dtype=np.uint8)] = np.arange(4, dtype=np.int16)
    codes = lookup[raw]
    valid_bases = codes >= 0
    counts = np.zeros(256, dtype=np.float64)
    if len(codes) >= 4:
        valid = valid_bases[:-3] & valid_bases[1:-2] & valid_bases[2:-1] & valid_bases[3:]
        if np.any(valid):
            index = codes[:-3] * 64 + codes[1:-2] * 16 + codes[2:-1] * 4 + codes[3:]
            counts = np.bincount(index[valid], minlength=256).astype(np.float64)
            counts /= counts.sum()
    result = np.empty(259, dtype=np.float32)
    result[:256] = counts.astype(np.float32)
    result[256] = np.float32(np.log1p(len(sequence)))
    result[257] = np.float32(1.0 - valid_bases.mean()) if len(sequence) else np.float32(1.0)
    result[258] = 1.0
    return result


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    return info


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", allowZip64=True, compresslevel=9) as archive:
        for name in sorted(arrays):
            array = np.asarray(arrays[name])
            if array.dtype.hasobject:
                raise ValueError("object arrays are forbidden")
            with archive.open(_zip_info(name), "w", force_zip64=True) as member:
                np.lib.format.write_array(member, array, allow_pickle=False)


def exact_row_classes(values: np.ndarray) -> tuple[int, int, int, np.ndarray]:
    values = np.ascontiguousarray(values)
    packed = values.view(np.dtype((np.void, values.dtype.itemsize * values.shape[1]))).ravel()
    _, inverse, counts = np.unique(packed, return_inverse=True, return_counts=True)
    return len(counts), int(len(values) - len(counts)), int(counts.max(initial=0)), inverse


def build(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    protocol = {
        "schema": "slp.human-transcript-kmer4-protocol/v1",
        "status": "frozen-before-fasta-sequence-access",
        "hypothesis": "strand-preserving transcript composition distinguishes static RNA queries lacking translated-protein features",
        "feature_formula": {
            "dimensions": 259,
            "columns_0_255": "lexicographic A,C,G,T 4-mer proportions over windows containing only A/C/G/T; ambiguous windows skipped",
            "column_256": "log1p transcript length",
            "column_257": "fraction of sequence characters outside A/C/G/T",
            "column_258": "transcript present flag",
            "strand": "use FASTA transcript sequence verbatim; no reverse complement",
        },
        "selection": "longest transcript per unversioned stable ENSG across cDNA and ncRNA; ties by stable ENST then versioned ENST, lexicographically ascending",
        "access_boundary": "static Ensembl sequence only; no molecular or benchmark outcomes; no symbols/descriptions as features",
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    started = time.perf_counter()
    sources = verify_sources(args.source_dir)
    selected, scan_statistics = select_transcripts(args.source_dir)

    query_ids = np.asarray((FRANGIEH / "rna-query-ensembl-ids.txt").read_text(encoding="ascii").splitlines())
    frangieh_metadata = json.loads(FRANGIEH_METADATA_AUDIT.read_text(encoding="utf-8"))
    action_ids = np.asarray(sorted(set(frangieh_metadata["interventions"]["stable_action_ensembl_ids"])))
    if len(action_ids) != 237 or any(re.fullmatch(r"ENSG[0-9]{11}", item) is None for item in action_ids):
        raise ValueError("unexpected metadata-only Frangieh action roster")
    output_ids = np.asarray(sorted(set(selected) | set(query_ids) | set(action_ids)), dtype=str)
    values = np.zeros((len(output_ids), 259), dtype=np.float32)
    provenance = []
    sequence_hashes = {}
    selected_bases = 0
    source_counts = {"cdna": 0, "ncrna": 0}
    for index, gene_id in enumerate(output_ids):
        transcript = selected.get(gene_id)
        if transcript is None:
            continue
        values[index] = transcript_features(transcript.sequence)
        selected_bases += len(transcript.sequence)
        source_counts[transcript.source_kind] += 1
        sequence_digest = hashlib.sha256(transcript.sequence.encode("ascii")).hexdigest()
        sequence_hashes[gene_id] = sequence_digest
        provenance.append((gene_id, transcript, sequence_digest))
    multiplicity: dict[str, int] = {}
    for sequence_digest in sequence_hashes.values():
        multiplicity[sequence_digest] = multiplicity.get(sequence_digest, 0) + 1
    provenance_path = args.output_dir / "selected-transcript-provenance.jsonl"
    with provenance_path.open("w", encoding="utf-8", newline="\n") as stream:
        for gene_id, transcript, sequence_digest in provenance:
            record = {
                "entity_id": gene_id,
                "gene_versioned_id": transcript.gene_versioned_id,
                "transcript_stable_id": transcript.transcript_stable_id,
                "transcript_versioned_id": transcript.transcript_id,
                "source_kind": transcript.source_kind,
                "strand": transcript.strand,
                "length": len(transcript.sequence),
                "ambiguous_fraction": float(values[np.searchsorted(output_ids, gene_id), 257]),
                "raw_sequence_sha256": sequence_digest,
                "selected_sequence_multiplicity": multiplicity[sequence_digest],
            }
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    feature_path = args.output_dir / FEATURE_NAME
    write_deterministic_npz(feature_path, {
        "feature_values": values,
        "entity_taxon": np.full(len(output_ids), 9606, dtype=np.int64),
        "entity_id": output_ids,
    })

    feature_lookup = {gene: index for index, gene in enumerate(output_ids)}
    query_rows = values[np.fromiter((feature_lookup[x] for x in query_ids), dtype=np.int64, count=len(query_ids))]
    action_rows = values[np.fromiter((feature_lookup[x] for x in action_ids), dtype=np.int64, count=len(action_ids))]
    query_present = query_rows[:, 258] == 1
    action_present = action_rows[:, 258] == 1

    with np.load(REFERENCE, allow_pickle=False) as reference:
        if not np.array_equal(reference["rna_query_ids"].astype(str), query_ids):
            raise ValueError("frozen reference RNA query axis differs")
        old = reference["rna_query_features"].astype(np.float32)
    old_unique, old_excess, old_largest, old_inverse = exact_row_classes(old)
    _, old_counts = np.unique(old_inverse, return_counts=True)
    largest_class = old_inverse == int(np.argmax(old_counts))
    largest_new = query_rows[largest_class]
    largest_unique, largest_excess, largest_after, _ = exact_row_classes(largest_new)
    combined = np.concatenate((old, query_rows), axis=1)
    combined_unique, combined_excess, combined_largest, _ = exact_row_classes(combined)

    present_sequences = list(sequence_hashes.values())
    duplicate_sizes = np.asarray(list(multiplicity.values()), dtype=np.int64)
    manifest = {
        "schema": "slp.human-transcript-kmer4-feature-artifact/v1",
        "status": "exploratory-static-feature-artifact-not-omf-admitted",
        "source": sources,
        "artifact": {"path": FEATURE_NAME, "sha256": sha256(feature_path), "bytes": feature_path.stat().st_size},
        "arrays": {
            "feature_values": {"shape": list(values.shape), "dtype": "float32", "dimension": 259},
            "entity_taxon": {"shape": [len(output_ids)], "dtype": "int64", "value": 9606},
            "entity_id": {"shape": [len(output_ids)], "dtype": str(output_ids.dtype), "ordering": "ascending codepoint stable ENSG"},
        },
        "selection": protocol["selection"],
        "feature_formula": protocol["feature_formula"],
        "scan": {
            **scan_statistics,
            "selected_genes": len(selected),
            "selected_bases": selected_bases,
            "selected_source_counts": source_counts,
            "unique_selected_sequences": len(set(present_sequences)),
            "duplicate_sequence_groups": int(np.sum(duplicate_sizes > 1)),
            "genes_in_duplicate_sequence_groups": int(duplicate_sizes[duplicate_sizes > 1].sum()),
            "maximum_selected_sequence_multiplicity": int(duplicate_sizes.max(initial=0)),
        },
        "coverage": {
            "frangieh_rna_queries": len(query_ids),
            "frangieh_rna_queries_present": int(query_present.sum()),
            "frangieh_rna_queries_missing": int((~query_present).sum()),
            "frangieh_actions": len(action_ids),
            "frangieh_actions_present": int(action_present.sum()),
            "frangieh_actions_missing": int((~action_present).sum()),
            "frangieh_action_roster_path": str(FRANGIEH_METADATA_AUDIT.relative_to(ROOT)).replace("\\", "/"),
            "frangieh_action_roster_sha256": sha256(FRANGIEH_METADATA_AUDIT),
        },
        "frozen_reference_distinctness": {
            "reference_path": str(REFERENCE.relative_to(ROOT)).replace("\\", "/"),
            "reference_sha256": sha256(REFERENCE),
            "old_unique_rows": old_unique,
            "old_duplicate_excess": old_excess,
            "old_largest_equivalence_group": old_largest,
            "previously_unsupported_group_size": int(largest_class.sum()),
            "previously_unsupported_group_transcript_present": int(query_present[largest_class].sum()),
            "previously_unsupported_group_unique_transcript_rows": largest_unique,
            "previously_unsupported_group_duplicate_excess_after_transcript": largest_excess,
            "previously_unsupported_group_largest_equivalence_after_transcript": largest_after,
            "combined_unique_rows": combined_unique,
            "combined_duplicate_excess": combined_excess,
            "combined_largest_equivalence_group": combined_largest,
        },
        "provenance": {"path": provenance_path.name, "sha256": sha256(provenance_path), "rows": len(provenance)},
        "runtime_seconds": time.perf_counter() - started,
        "limitations": [
            "4-mer composition is strand-sensitive but discards positions beyond local windows",
            "zero features with presence=0 mean no stable gene transcript in the two pinned Ensembl exports",
            "duplicate feature rows remain observationally indistinguishable to a model without learned gene identity",
            "no molecular response, held-out outcome, gene symbol, or free-text description entered selection or features",
        ],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.source_dir = args.source_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


if __name__ == "__main__":
    result = build(parse_args())
    print(json.dumps({"artifact": result["artifact"], "coverage": result["coverage"], "runtime_seconds": result["runtime_seconds"]}, indent=2))
