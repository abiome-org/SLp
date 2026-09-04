"""Deterministic, outcome-blind amino-acid statistics feature block.

The implementation intentionally uses only Python's standard library.  It
accepts exactly a relation-closed entity universe, the release-pinned SGD
protein FASTA, and the two immutable SGD mapping artifacts needed to verify
the join.  Identifiers are retained solely as provenance and row keys; model
values contain only length/4096 and amino-acid fractions.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import re
import struct
import tarfile
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

FEATURE_BLOCK_SCHEMA = "slp.sequence-statistics-feature-block/v1"
FEATURE_ENTITY_SCHEMA = "slp.static-feature-entity/v1"
SEQUENCE_PROVENANCE_SCHEMA = "slp.sequence-feature-provenance/v1"
EXCLUDED_SEQUENCE_SCHEMA = "slp.excluded-non-current-sequence/v1"
AUDIT_SCHEMA = "slp.sequence-statistics-feature-block-audit/v1"
UNIVERSE_SCHEMA = "slp.static-entity-universe/v1"
ENTITY_SCHEMA = "slp.static-entity/v1"
RELATION_SCHEMA = "slp.proteome-protein-relation/v1"
CURRENT_ORF_SCHEMA = "slp.sgd-current-orf/v1"
MAPPING_MANIFEST_SCHEMA = "slp.sgd-stable-id-mapping/v1"

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
FEATURE_NAMES = ("protein_length_div_4096",) + tuple(
    f"aa_fraction_{residue}" for residue in AA_ORDER
)
FEATURE_DIM = 21
SPECIES_TAXON = 4932
STRAIN_TAXON = 559292
SOURCE_RELEASE = "R64.5.1"
HEADER_RELEASE = "64-5-1"
MAPPING_ID = "slp-sgd-map:2026-08-28-object-set-v1"
MAPPING_SHA256 = "6fd789df6099b78a8842baa8f1d20ab0a3fe77f27ce512ee783444eb2627ef2a"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RESOURCE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
INPUT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
SGD_CURIE_RE = re.compile(r"^SGD:S[0-9]{9}$")
UNIPROT_CURIE_RE = re.compile(r"^UniProtKB:[A-Z0-9][A-Z0-9-]{0,31}$")
FASTA_HEADER_RE = re.compile(
    r"^>(?P<systematic>\S+) (?P<display>\S+) SGDID:(?P<sgdid>S[0-9]{9}), (?P<description>.+)$"
)
SOURCE_PEPTIDE_RE = re.compile(rf"^[{AA_ORDER}*]+$")
CURRENT_PEPTIDE_RE = re.compile(rf"^M[{AA_ORDER}]*\*$")

FORBIDDEN_KEY_PARTS = {
    "abundance", "benchmark", "embedding", "expression", "fitness", "fold",
    "label", "measurement", "outcome", "phenotype", "reward", "score", "split",
    "targetvalue", "trajectory",
}

SOURCE_CLASS_TOKENS = {
    "verified-orf": "Verified ORF",
    "uncharacterized-orf": "Uncharacterized ORF",
    "dubious-orf": "Dubious ORF",
    "transposable-element-gene": "transposable_element_gene",
    "pseudogene": "pseudogene",
    "blocked-reading-frame": "blocked_reading_frame",
}

LIMITATIONS = [
    "twenty-one hand-designed sequence statistics are a weak deterministic baseline, not a learned protein representation",
    "static sequences cover held entities but no held quantitative intervention outcome is consumed",
    "all five ambiguous protein mappings require identical related peptides; no target is selected",
]

ACCESS_BOUNDARY = {
    "inputNames": ["staticEntityUniverse", "sgdProteinSequences", "sgdCurrentOrfs", "sgdMappingManifest"],
    "heldRosterConsumed": False,
    "quantitativeOutcomesConsumed": False,
    "trainingPartitionAssignmentsConsumed": False,
    "benchmarkDataConsumed": False,
    "freeTextDescriptionUsedAsFeature": False,
    "identifiersUsedAsNumericFeatures": False,
    "identifiersMergedAcrossTaxa": False,
}


class SequenceFeatureBlockError(ValueError):
    """Raised when an input or output violates the frozen feature contract."""


def _bootstrap_nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SequenceFeatureBlockError(f"{label} must be a non-empty trimmed string")
    return value


def _bootstrap_bare_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise SequenceFeatureBlockError(f"{label} must be a lowercase SHA-256")
    return value


def _bootstrap_prefixed_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise SequenceFeatureBlockError(f"{label} must use the sha256: prefix")
    _bootstrap_bare_digest(value.removeprefix("sha256:"), label)
    return value


def _bootstrap_canonical_relative(value: object, label: str) -> str:
    relative = _bootstrap_nonempty(value, label)
    posix = PurePosixPath(relative)
    if relative != posix.as_posix() or posix.is_absolute() or "\\" in relative or ":" in relative or any(part in {"", ".", ".."} for part in posix.parts):
        raise SequenceFeatureBlockError(f"{label} is not a canonical relative path")
    return relative


def _bootstrap_dataset_resource(value: object, label: str) -> tuple[str, str]:
    resource = _bootstrap_nonempty(value, label)
    if not resource.startswith("omf://"):
        raise SequenceFeatureBlockError(f"{label} must be an OMF DatasetSnapshot URI")
    identity, separator, revision = resource.removeprefix("omf://").rpartition("@")
    if not separator:
        raise SequenceFeatureBlockError(f"{label} must carry an exact revision")
    _bootstrap_prefixed_digest(revision, f"{label} revision")
    parts = identity.split("/")
    if len(parts) < 3 or parts[-2] != "datasetsnapshot" or RESOURCE_NAME_RE.fullmatch(parts[-1]) is None:
        raise SequenceFeatureBlockError(f"{label} must identify a DatasetSnapshot")
    return parts[-1], revision


@dataclass(frozen=True)
class FileSpec:
    name: str
    bytes: int
    sha256: str
    mode: int = 0o644

    def __post_init__(self) -> None:
        _bootstrap_canonical_relative(self.name, "file name")
        if type(self.bytes) is not int or self.bytes < 0:
            raise SequenceFeatureBlockError("file bytes must be a non-negative integer")
        _bootstrap_bare_digest(self.sha256, "file SHA-256")
        if type(self.mode) is not int or not 0 <= self.mode <= 0o7777:
            raise SequenceFeatureBlockError("file mode must be a valid permission integer")


@dataclass(frozen=True)
class ExpectedDataset:
    resource: str
    manifest_digest: str
    tree_digest: str
    files: tuple[FileSpec, ...]

    def __post_init__(self) -> None:
        _bootstrap_dataset_resource(self.resource, "expected dataset resource")
        _bootstrap_prefixed_digest(self.manifest_digest, "expected dataset manifest")
        _bootstrap_prefixed_digest(self.tree_digest, "expected dataset tree")
        if not self.files or len({item.name for item in self.files}) != len(self.files):
            raise SequenceFeatureBlockError("expected dataset file set is empty or duplicated")


@dataclass(frozen=True)
class ExpectedArtifact:
    manifest_digest: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _bootstrap_prefixed_digest(self.manifest_digest, "expected artifact manifest")
        if type(self.bytes) is not int or self.bytes < 0:
            raise SequenceFeatureBlockError("artifact bytes must be a non-negative integer")
        _bootstrap_bare_digest(self.sha256, "artifact payload SHA-256")


@dataclass(frozen=True)
class ExpectedContract:
    universe: ExpectedDataset
    sequences: ExpectedDataset
    current_orfs: ExpectedArtifact
    mapping_manifest: ExpectedArtifact
    mapping_id: str
    mapping_sha256: str
    fasta_decompressed_bytes: int
    fasta_decompressed_sha256: str
    fasta_records: int
    current_orf_records: int
    non_current_records: int
    universe_genes: int
    universe_proteins: int
    universe_rows: int
    current_outside_universe: int
    present_values: int
    universe_entity_key_sha256: str
    universe_entity_jsonl_sha256: str
    universe_manifest_sha256: str
    universe_relation_jsonl_sha256: str
    source_class_counts: tuple[tuple[str, int], ...]
    stop_absent_non_current: tuple[tuple[str, str], ...]
    internal_stop_non_current: tuple[str, ...]
    non_m_start_non_current: tuple[str, ...]
    multi_target_peptide_sha256: tuple[tuple[str, str], ...]
    output_sequence_provenance_sha256: str | None = None
    output_excluded_non_current_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.mapping_id or self.mapping_id != self.mapping_id.strip():
            raise SequenceFeatureBlockError("mapping ID must be non-empty and trimmed")
        _bootstrap_bare_digest(self.mapping_sha256, "mapping SHA-256")
        _bootstrap_bare_digest(self.fasta_decompressed_sha256, "decompressed FASTA SHA-256")
        _bootstrap_bare_digest(self.universe_entity_key_sha256, "universe key SHA-256")
        _bootstrap_bare_digest(self.universe_entity_jsonl_sha256, "universe entity JSONL SHA-256")
        _bootstrap_bare_digest(self.universe_manifest_sha256, "universe manifest SHA-256")
        _bootstrap_bare_digest(self.universe_relation_jsonl_sha256, "universe relation JSONL SHA-256")
        for name, value in (
            ("fastaDecompressedBytes", self.fasta_decompressed_bytes),
            ("fastaRecords", self.fasta_records),
            ("currentOrfRecords", self.current_orf_records),
            ("nonCurrentRecords", self.non_current_records),
            ("universeGenes", self.universe_genes),
            ("universeProteins", self.universe_proteins),
            ("universeRows", self.universe_rows),
            ("currentOutsideUniverse", self.current_outside_universe),
            ("presentValues", self.present_values),
        ):
            if type(value) is not int or value < 0:
                raise SequenceFeatureBlockError(f"{name} must be a non-negative integer")
        if dict(self.source_class_counts).keys() != SOURCE_CLASS_TOKENS.keys():
            raise SequenceFeatureBlockError("source-class contract is incomplete")
        for _name, digest in self.multi_target_peptide_sha256:
            _bootstrap_bare_digest(digest, "multi-target peptide SHA-256")
        if self.output_sequence_provenance_sha256 is not None:
            _bootstrap_bare_digest(
                self.output_sequence_provenance_sha256,
                "pinned output sequence-provenance SHA-256",
            )
        if self.output_excluded_non_current_sha256 is not None:
            _bootstrap_bare_digest(
                self.output_excluded_non_current_sha256,
                "pinned output excluded-sequence SHA-256",
            )


@dataclass(frozen=True)
class PinnedDataset:
    input_name: str
    path: Path
    resource: str
    revision: str
    manifest_digest: str


@dataclass(frozen=True)
class LiteralArtifact:
    input_name: str
    path: Path
    manifest_digest: str


PRODUCTION_CONTRACT = ExpectedContract(
    universe=ExpectedDataset(
        resource=(
            "omf://abiome/slp/datasetsnapshot/slp-1-1-static-entity-universe-v1@"
            "sha256:de3efddf5a9e4f66496a1edda14b04de774e972bc7b9efd30964644de2a56cac"
        ),
        manifest_digest="sha256:a65f94081c0b60a8b486ed968b58fc4d021ba3ea7f5f11425d3a1635cbb10684",
        tree_digest="sha256:7ec354f427cfd8a2fcc3de1004c7e4ac77402a78b5cc0a0b5ef89ba24656fd3f",
        files=(
            FileSpec("entity-universe-audit.json", 4_880, "339412ea008cf383db2258d0788d71c2cf357183b331d49f4168aa7f113f1a0f"),
            FileSpec("entity-universe.tar", 1_525_760, "d947bf618b854dd33a7157ac0f0380c544e9a4377bddb00806c9ca07f689a544"),
        ),
    ),
    sequences=ExpectedDataset(
        resource=(
            "omf://abiome/slp/datasetsnapshot/slp-1-1-sgd-protein-sequences-r64-5-1@"
            "sha256:3b76017f5ac74d8d96efb1db52d14af91c9fb15995062110558ce4651cf3ba0c"
        ),
        manifest_digest="sha256:8f88480196b5cd8f3c15d65dbdbc09f83305c371fb476c70a38825dad2be4283",
        tree_digest="sha256:823a18ed8039ee44ee44b860551fea749b9012c941e6b9cd5163938da19b168a",
        files=(
            FileSpec("dates_of_genome_releases.tab", 2_050, "cc5d40722442a605d1d6dcf9a36442d87076829a04f776f87b3de0020f92f9e7"),
            FileSpec("orf_protein.README", 930, "b53064bef6424f0e9b5c5a6af88602bb15949e68bedb397b6731b094ebca5be9"),
            FileSpec("orf_trans_all_R64-5-1_20240529.fasta.gz", 2_689_634, "17e8b47e1ae23178c6000fbc4ab548f102d1b250ef9dff5d811feb3f03dd2c5b"),
        ),
    ),
    current_orfs=ExpectedArtifact(
        "sha256:e67f0e8773feae108ecdb687139885e01ca972ff4aec95cd1358b33db1ea1192",
        2_135_394,
        "df7b717cad88dc3672f72f8148f6a9132d12abe6ba020b220b091a8da8f7004d",
    ),
    mapping_manifest=ExpectedArtifact(
        "sha256:c74ea81ce604357b998e5f09130dff85bf8a7a26504b9b2426f8038608c52d9c",
        3_818,
        "570557ab1201913a18de9790f8adc5ee2e3cb56c6bb0e8d588fe43660c0214e1",
    ),
    mapping_id=MAPPING_ID,
    mapping_sha256=MAPPING_SHA256,
    fasta_decompressed_bytes=5_511_467,
    fasta_decompressed_sha256="e01f9e1ef7e5a01ff7cd0ee7a843e6d1c1da8c3777fdfac3a5293711d4c56518",
    fasta_records=6_722,
    current_orf_records=6_613,
    non_current_records=109,
    universe_genes=5_187,
    universe_proteins=1_850,
    universe_rows=7_037,
    current_outside_universe=1_426,
    present_values=147_777,
    universe_entity_key_sha256="82b8e2885939577fe6946e3b974a10cb947834118f2070e1bcbe4c2f2e6a5fd9",
    universe_entity_jsonl_sha256="03d0ea520668cedf48bae7055d998b8f906d0a6638cb9daa102c8a5963e30b33",
    universe_manifest_sha256="19e309855c9ecd16500244890a24a5d0954724e250120cb3e1bb7cba4adfd149",
    universe_relation_jsonl_sha256="c72996b4ddc6870a3ab722060eef2fa2747fa9dd121d3e70514dd196c5283b8d",
    source_class_counts=(
        ("verified-orf", 5_271),
        ("uncharacterized-orf", 659),
        ("dubious-orf", 683),
        ("transposable-element-gene", 91),
        ("pseudogene", 12),
        ("blocked-reading-frame", 6),
    ),
    stop_absent_non_current=(
        ("SGD:S000000087", "YAR061W"),
        ("SGD:S000001838", "YFL056C"),
    ),
    internal_stop_non_current=(
        "SGD:S000000087",
        "SGD:S000000911",
        "SGD:S000001429",
        "SGD:S000001482",
        "SGD:S000002541",
        "SGD:S000005513",
        "SGD:S000005557",
        "SGD:S000001838",
    ),
    non_m_start_non_current=("SGD:S000001437",),
    multi_target_peptide_sha256=(
        ("UniProtKB:P02309", "a2b741dd2f24cf766d7b0526d2b81778c13477e3f95659a736eb8c60dac74212"),
        ("UniProtKB:P02994", "a6836830f169bb95a0b0a54ebcdc9d919ef6d454b34a7c5fbc38950c5e1b33b9"),
        ("UniProtKB:P10081", "8299c66c9dc688af630ffdee0674a5c699c0e384c694c7223e87ae9b403d5083"),
        ("UniProtKB:P32324", "0444c3f2b1cdcf680f2786991645fb0fd96eb95520bb3df1a09201a35e001e51"),
        ("UniProtKB:P61830", "95590309e518c62ddd5611164d5628f84fd56f4c7c1dd2483ea145b6c57a8535"),
    ),
    output_sequence_provenance_sha256=(
        "5955ae6f8503b87370bf5116fdae8699ced9c4e3a0a378fd3843baaa7c2965fe"
    ),
    output_excluded_non_current_sha256=(
        "aeb6be983ff828c517aaed9def31d9401a111b312e344e8349678eae75e7972f"
    ),
)


@dataclass(frozen=True)
class Bounds:
    max_manifest_bytes: int = 1_048_576
    max_line_bytes: int = 16_384
    max_fasta_bytes: int = 16 * 1024 * 1024
    max_sequence_length: int = 100_000
    max_records: int = 20_000
    max_archive_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        for name, value, minimum, maximum in (
            ("maxManifestBytes", self.max_manifest_bytes, 256, 16 * 1024 * 1024),
            ("maxLineBytes", self.max_line_bytes, 128, 1024 * 1024),
            ("maxFastaBytes", self.max_fasta_bytes, 1024, 1024**3),
            ("maxSequenceLength", self.max_sequence_length, 1, 10_000_000),
            ("maxRecords", self.max_records, 1, 20_000_000),
            ("maxArchiveBytes", self.max_archive_bytes, 1024, 1024**3),
        ):
            if type(value) is not int or not minimum <= value <= maximum:
                raise SequenceFeatureBlockError(f"{name} must be an integer in [{minimum}, {maximum}]")


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _canonical_json_bytes(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _pretty_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _omf_tree_digest(files: Iterable[FileSpec]) -> str:
    """Reproduce OMF 1.0's RFC-8785 tree digest for this integer/string shape."""
    entries = [
        {
            "path": item.name,
            "mode": item.mode,
            "size": item.bytes,
            "digest": f"sha256:{item.sha256}",
        }
        for item in sorted(files, key=lambda item: item.name)
    ]
    return f"sha256:{_sha256_bytes(canonical_json(entries).encode('utf-8'))}"


def _jsonl_bytes(records: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json_bytes(record) for record in records)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise SequenceFeatureBlockError(f"could not hash {path.name}") from error
    return digest.hexdigest()


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SequenceFeatureBlockError(f"{label} must be a non-empty trimmed string")
    return value


def _bare_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise SequenceFeatureBlockError(f"{label} must be a lowercase SHA-256")
    return value


def _prefixed_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise SequenceFeatureBlockError(f"{label} must use the sha256: prefix")
    _bare_digest(value.removeprefix("sha256:"), label)
    return value


def _canonical_relative(value: object, label: str) -> str:
    relative = _nonempty(value, label)
    posix = PurePosixPath(relative)
    if (
        relative != posix.as_posix()
        or posix.is_absolute()
        or "\\" in relative
        or ":" in relative
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise SequenceFeatureBlockError(f"{label} is not a canonical relative path")
    return relative


def _dataset_resource(value: object, label: str) -> tuple[str, str]:
    resource = _nonempty(value, label)
    if not resource.startswith("omf://"):
        raise SequenceFeatureBlockError(f"{label} must be an OMF DatasetSnapshot URI")
    identity, separator, revision = resource.removeprefix("omf://").rpartition("@")
    if not separator:
        raise SequenceFeatureBlockError(f"{label} must carry an exact revision")
    _prefixed_digest(revision, f"{label} revision")
    parts = identity.split("/")
    if (
        len(parts) < 3
        or parts[-2] != "datasetsnapshot"
        or RESOURCE_NAME_RE.fullmatch(parts[-1]) is None
        or any(not part or part in {".", ".."} or any(char.isspace() for char in part) for part in parts)
    ):
        raise SequenceFeatureBlockError(f"{label} must identify a DatasetSnapshot")
    return parts[-1], revision


def _resolved_path(value: object, label: str, *, directory: bool) -> Path:
    path = Path(_nonempty(str(value), label))
    cursor = path.absolute()
    while True:
        if cursor.is_symlink():
            raise SequenceFeatureBlockError(f"{label} must not contain a symlink")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SequenceFeatureBlockError(f"{label} does not exist") from error
    if directory != resolved.is_dir() or (not directory and not resolved.is_file()):
        expected = "directory" if directory else "regular file"
        raise SequenceFeatureBlockError(f"{label} must be a {expected}")
    return resolved


def resolve_pinned_dataset(value: object, input_name: str) -> PinnedDataset:
    if INPUT_NAME_RE.fullmatch(input_name) is None or not isinstance(value, dict):
        raise SequenceFeatureBlockError(f"{input_name} must be a materialized DatasetSnapshot")
    if set(value) != {"resource", "mode", "path", "manifestDigest"}:
        raise SequenceFeatureBlockError(f"{input_name} has a spoofed DatasetSnapshot shape")
    resource_name, revision = _dataset_resource(value["resource"], f"{input_name}.resource")
    if value["mode"] != "copy":
        raise SequenceFeatureBlockError(f"{input_name} must be copied, not mutable")
    manifest = _prefixed_digest(value["manifestDigest"], f"{input_name}.manifestDigest")
    root = _resolved_path(value["path"], f"{input_name}.path", directory=True)
    if root.name != resource_name or root.parent.name != input_name or root.parent.parent.name != "inputs":
        raise SequenceFeatureBlockError(f"{input_name}.path is inconsistent with OMF materialization")
    return PinnedDataset(input_name, root, str(value["resource"]), revision, manifest)


def resolve_literal_artifact(value: object, input_name: str) -> LiteralArtifact:
    if INPUT_NAME_RE.fullmatch(input_name) is None or not isinstance(value, dict):
        raise SequenceFeatureBlockError(f"{input_name} must be a literal OMF artifact")
    if set(value) != {"resource", "kind", "artifacts", "paths", "path"}:
        raise SequenceFeatureBlockError(f"{input_name} has a spoofed artifact shape")
    if value["kind"] != "artifact":
        raise SequenceFeatureBlockError(f"{input_name}.kind must be artifact")
    artifacts, paths = value["artifacts"], value["paths"]
    if not isinstance(artifacts, dict) or set(artifacts) != {"payload"}:
        raise SequenceFeatureBlockError(f"{input_name}.artifacts must contain only payload")
    if not isinstance(paths, dict) or set(paths) != {"payload"}:
        raise SequenceFeatureBlockError(f"{input_name}.paths must contain only payload")
    digest = _prefixed_digest(artifacts["payload"], f"{input_name}.artifacts.payload")
    if value["resource"] != f"artifact:{digest}":
        raise SequenceFeatureBlockError(f"{input_name}.resource does not match its payload")
    if paths["payload"] != value["path"]:
        raise SequenceFeatureBlockError(f"{input_name}.path differs from paths.payload")
    path = _resolved_path(value["path"], f"{input_name}.path", directory=False)
    if (
        path.name != "payload"
        or path.parent.name != "payload"
        or path.parent.parent.name != input_name
        or path.parent.parent.parent.name != "inputs"
    ):
        raise SequenceFeatureBlockError(f"{input_name}.path is inconsistent with OMF materialization")
    return LiteralArtifact(input_name, path, digest)


def _regular_child(root: Path, name: str) -> Path:
    relative = _canonical_relative(name, "snapshot file")
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor /= part
        if cursor.is_symlink():
            raise SequenceFeatureBlockError(f"snapshot file must not be a symlink: {name}")
    try:
        path = cursor.resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as error:
        raise SequenceFeatureBlockError(f"snapshot file is missing or escapes its root: {name}") from error
    if not path.is_file():
        raise SequenceFeatureBlockError(f"snapshot member is not a regular file: {name}")
    return path


def _verify_dataset(dataset: PinnedDataset, expected: ExpectedDataset) -> dict[str, Path]:
    if dataset.resource != expected.resource or dataset.manifest_digest != expected.manifest_digest:
        raise SequenceFeatureBlockError(f"{dataset.input_name} immutable identity drift")
    if _omf_tree_digest(expected.files) != expected.tree_digest:
        raise SequenceFeatureBlockError(f"{dataset.input_name} OMF tree digest does not bind its files")
    actual_names = {item.name for item in dataset.path.iterdir()}
    expected_by_name = {item.name: item for item in expected.files}
    if actual_names != set(expected_by_name):
        raise SequenceFeatureBlockError(f"{dataset.input_name} file set drift")
    paths: dict[str, Path] = {}
    for name, spec in sorted(expected_by_name.items()):
        path = _regular_child(dataset.path, name)
        if path.stat().st_size != spec.bytes:
            raise SequenceFeatureBlockError(f"{dataset.input_name}/{name} byte drift")
        if _sha256_file(path) != spec.sha256:
            raise SequenceFeatureBlockError(f"{dataset.input_name}/{name} digest drift")
        paths[name] = path
    return paths


def _verify_artifact(artifact: LiteralArtifact, expected: ExpectedArtifact) -> None:
    if artifact.manifest_digest != expected.manifest_digest:
        raise SequenceFeatureBlockError(f"{artifact.input_name} artifact-manifest drift")
    if artifact.path.stat().st_size != expected.bytes:
        raise SequenceFeatureBlockError(f"{artifact.input_name} payload byte drift")
    if _sha256_file(artifact.path) != expected.sha256:
        raise SequenceFeatureBlockError(f"{artifact.input_name} payload digest drift")


def _strict_fields(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise SequenceFeatureBlockError(f"{label} fields drift")
    return value


def _reject_forbidden_keys(value: object, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise SequenceFeatureBlockError(f"{label} has a non-string key")
            folded = key.casefold().replace("_", "").replace("-", "")
            if any(part in folded for part in FORBIDDEN_KEY_PARTS):
                raise SequenceFeatureBlockError(f"{label} contains forbidden key {key!r}")
            _reject_forbidden_keys(child, label)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_keys(child, label)


def _read_canonical_json(path: Path, maximum_bytes: int, label: str) -> dict[str, Any]:
    if path.stat().st_size > maximum_bytes:
        raise SequenceFeatureBlockError(f"{label} exceeds its byte bound")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SequenceFeatureBlockError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict) or raw not in {_canonical_json_bytes(value), _pretty_json_bytes(value)}:
        raise SequenceFeatureBlockError(f"{label} is not canonical JSON")
    return value


def _jsonl_blob(payload: bytes, bounds: Bounds, maximum_records: int, label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if payload and not payload.endswith(b"\n"):
        raise SequenceFeatureBlockError(f"{label} lacks a terminal LF")
    for line_number, raw in enumerate(payload.splitlines(keepends=True), start=1):
        if len(raw) > bounds.max_line_bytes:
            raise SequenceFeatureBlockError(f"{label}:{line_number} exceeds maxLineBytes")
        if not raw.endswith(b"\n") or raw in {b"\n", b"\r\n"}:
            raise SequenceFeatureBlockError(f"{label}:{line_number} is not canonical JSONL")
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SequenceFeatureBlockError(f"{label}:{line_number} is invalid JSON") from error
        if not isinstance(value, dict) or raw != _canonical_json_bytes(value):
            raise SequenceFeatureBlockError(f"{label}:{line_number} is not canonical JSON")
        _reject_forbidden_keys(value, f"{label}:{line_number}")
        records.append(value)
        if len(records) > maximum_records:
            raise SequenceFeatureBlockError(f"{label} exceeds its record bound")
    return records


def _read_canonical_jsonl(path: Path, bounds: Bounds, maximum_records: int, label: str) -> list[dict[str, Any]]:
    if path.stat().st_size > bounds.max_fasta_bytes:
        raise SequenceFeatureBlockError(f"{label} exceeds its byte bound")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise SequenceFeatureBlockError(f"could not read {label}") from error
    return _jsonl_blob(payload, bounds, maximum_records, label)


def framed_key_sha256(keys: Iterable[tuple[int, str]]) -> str:
    items = {f"{taxon}\t{entity}" for taxon, entity in keys}
    try:
        payload = b"".join(item.encode("ascii") + b"\n" for item in sorted(items))
    except UnicodeEncodeError as error:
        raise SequenceFeatureBlockError("semantic identity keys must be ASCII") from error
    return _sha256_bytes(payload)


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.type = tarfile.REGTYPE
    return info


def _tar_bytes(members: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(members):
            payload = members[name]
            archive.addfile(_tar_info(name, len(payload)), io.BytesIO(payload))
    return output.getvalue()


def _write_tar(path: Path, members: Mapping[str, bytes]) -> None:
    if path.exists() or path.is_symlink():
        raise SequenceFeatureBlockError("refusing to overwrite feature-block archive")
    path.write_bytes(_tar_bytes(members))


def _read_exact_tar(path: Path, expected_names: Sequence[str], bounds: Bounds, label: str) -> dict[str, bytes]:
    if path.is_symlink() or not path.is_file():
        raise SequenceFeatureBlockError(f"{label} must be a regular file")
    if path.stat().st_size > bounds.max_archive_bytes:
        raise SequenceFeatureBlockError(f"{label} exceeds maxArchiveBytes")
    blobs: dict[str, bytes] = {}
    try:
        with tarfile.open(path, mode="r:") as archive:
            members = archive.getmembers()
            if [item.name for item in members] != list(expected_names):
                raise SequenceFeatureBlockError(f"{label} member set or order drift")
            for member in members:
                if (
                    not member.isfile()
                    or member.mode != 0o644
                    or member.mtime != 0
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.pax_headers
                ):
                    raise SequenceFeatureBlockError(f"{label} member metadata drift")
                stream = archive.extractfile(member)
                if stream is None:
                    raise SequenceFeatureBlockError(f"{label} member is unreadable")
                blobs[member.name] = stream.read()
    except (OSError, tarfile.TarError) as error:
        raise SequenceFeatureBlockError(f"{label} is invalid") from error
    if _tar_bytes(blobs) != path.read_bytes():
        raise SequenceFeatureBlockError(f"{label} is not canonical deterministic USTAR")
    return blobs


def _file_ref(path: str, payload: bytes, *, records: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {"path": path, "sha256": _sha256_bytes(payload), "bytes": len(payload)}
    if records is not None:
        result["records"] = records
    return result


def _validate_file_ref(value: object, path: str, payload: bytes, label: str, *, records: int | None = None) -> None:
    fields = {"path", "sha256", "bytes"} | ({"records"} if records is not None else set())
    ref = _strict_fields(value, fields, label)
    if ref != _file_ref(path, payload, records=records):
        raise SequenceFeatureBlockError(f"{label} does not match payload")


def _validate_universe(paths: Mapping[str, Path], bounds: Bounds, expected: ExpectedContract) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, object]]:
    names = (
        "static-entity-universe/entities.jsonl",
        "static-entity-universe/manifest.json",
        "static-entity-universe/relations.jsonl",
    )
    archive_path = paths["entity-universe.tar"]
    blobs = _read_exact_tar(archive_path, names, bounds, "entity-universe archive")
    manifest_payload = blobs[names[1]]
    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SequenceFeatureBlockError("entity-universe manifest is invalid") from error
    if not isinstance(manifest, dict) or manifest_payload != _pretty_json_bytes(manifest):
        raise SequenceFeatureBlockError("entity-universe manifest is not canonical JSON")
    if _sha256_bytes(manifest_payload) != expected.universe_manifest_sha256:
        raise SequenceFeatureBlockError("entity-universe inner manifest digest drift")
    manifest = _strict_fields(
        manifest,
        {"schema", "version", "identityKey", "ordering", "source", "identityMapping", "semanticSetHashes", "inputs", "entities", "relations", "contentPolicy"},
        "entity-universe manifest",
    )
    if manifest["schema"] != UNIVERSE_SCHEMA or manifest["version"] != 1:
        raise SequenceFeatureBlockError("entity-universe schema drift")
    if manifest["identityKey"] != ["ncbiTaxon", "entityId"] or manifest["ordering"] != "ascending-ncbiTaxon-then-codepoint-entityId":
        raise SequenceFeatureBlockError("entity-universe identity contract drift")
    source = _strict_fields(manifest["source"], {"id", "release", "ncbiTaxon"}, "entity-universe source")
    if source["ncbiTaxon"] != SPECIES_TAXON:
        raise SequenceFeatureBlockError("entity-universe species taxon drift")
    mapping = _strict_fields(manifest["identityMapping"], {"id", "sha256"}, "entity-universe mapping")
    if mapping != {"id": expected.mapping_id, "sha256": expected.mapping_sha256}:
        raise SequenceFeatureBlockError("entity-universe mapping drift")

    entity_payload, relation_payload = blobs[names[0]], blobs[names[2]]
    if _sha256_bytes(entity_payload) != expected.universe_entity_jsonl_sha256:
        raise SequenceFeatureBlockError("entity-universe entity-set digest drift")
    if _sha256_bytes(relation_payload) != expected.universe_relation_jsonl_sha256:
        raise SequenceFeatureBlockError("entity-universe relation-set digest drift")
    entities = _jsonl_blob(entity_payload, bounds, bounds.max_records, "entity-universe entities")
    relations = _jsonl_blob(relation_payload, bounds, bounds.max_records, "entity-universe relations")
    keys: list[tuple[int, str]] = []
    gene_keys: set[tuple[int, str]] = set()
    protein_keys: set[tuple[int, str]] = set()
    for index, record in enumerate(entities, start=1):
        row = _strict_fields(record, {"schema", "ncbiTaxon", "entityId", "entityClass", "usages"}, f"entity row {index}")
        entity_id, entity_class = row["entityId"], row["entityClass"]
        if row["schema"] != ENTITY_SCHEMA or row["ncbiTaxon"] != SPECIES_TAXON:
            raise SequenceFeatureBlockError("entity row schema or taxon drift")
        if entity_class == "gene" and (not isinstance(entity_id, str) or SGD_CURIE_RE.fullmatch(entity_id) is None):
            raise SequenceFeatureBlockError("invalid gene identity")
        if entity_class == "protein" and (not isinstance(entity_id, str) or UNIPROT_CURIE_RE.fullmatch(entity_id) is None):
            raise SequenceFeatureBlockError("invalid protein identity")
        if entity_class not in {"gene", "protein"}:
            raise SequenceFeatureBlockError("unknown entity class")
        usages = row["usages"]
        if not isinstance(usages, list) or not usages or usages != sorted(set(usages)) or any(item not in {"action", "readout-query", "relation-support"} for item in usages):
            raise SequenceFeatureBlockError("invalid entity usage set")
        key = (SPECIES_TAXON, entity_id)
        keys.append(key)
        (gene_keys if entity_class == "gene" else protein_keys).add(key)
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise SequenceFeatureBlockError("entity rows are duplicated or out of order")
    if len(gene_keys) != expected.universe_genes or len(protein_keys) != expected.universe_proteins or len(keys) != expected.universe_rows:
        raise SequenceFeatureBlockError("entity-universe count drift")
    if framed_key_sha256(keys) != expected.universe_entity_key_sha256:
        raise SequenceFeatureBlockError("entity-universe composite-key digest drift")

    relation_map: dict[str, tuple[str, ...]] = {}
    edge_keys: set[tuple[int, str]] = set()
    for index, record in enumerate(relations, start=1):
        row = _strict_fields(
            record,
            {"schema", "proteinId", "sourceAccession", "sourceAccessionType", "ncbiTaxon", "currentOrfRelations", "currentOrfRelationCount", "chooseFirstAllowed"},
            f"relation row {index}",
        )
        protein = row["proteinId"]
        targets = row["currentOrfRelations"]
        if row["schema"] != RELATION_SCHEMA or row["ncbiTaxon"] != SPECIES_TAXON or row["chooseFirstAllowed"] is not False:
            raise SequenceFeatureBlockError("relation schema, taxon, or ambiguity policy drift")
        if not isinstance(protein, str) or UNIPROT_CURIE_RE.fullmatch(protein) is None or (SPECIES_TAXON, protein) not in protein_keys:
            raise SequenceFeatureBlockError("relation protein is absent from the universe")
        if not isinstance(targets, list) or not targets or targets != sorted(set(targets)) or row["currentOrfRelationCount"] != len(targets):
            raise SequenceFeatureBlockError("relation targets are missing, duplicated, or unordered")
        if any(not isinstance(item, str) or SGD_CURIE_RE.fullmatch(item) is None or (SPECIES_TAXON, item) not in gene_keys for item in targets):
            raise SequenceFeatureBlockError("relation target is absent from the gene universe")
        accession_type = _strict_fields(row["sourceAccessionType"], {"source", "type", "namespaceInferred", "caseNormalization"}, "relation accession type")
        if accession_type != {"source": "UniProtKB", "type": "UniProtKB ID", "namespaceInferred": False, "caseNormalization": "none"}:
            raise SequenceFeatureBlockError("relation accession typing drift")
        if row["sourceAccession"] != protein.removeprefix("UniProtKB:") or protein in relation_map:
            raise SequenceFeatureBlockError("relation accession mismatch or duplicate")
        relation_map[protein] = tuple(targets)
        edge_keys.update((SPECIES_TAXON, item) for item in targets)
    if set(relation_map) != {key[1] for key in protein_keys} or not edge_keys <= gene_keys:
        raise SequenceFeatureBlockError("entity universe is not relation closed")

    entity_section = _strict_fields(manifest["entities"], {"format", "file", "counts"}, "manifest entities")
    if entity_section["format"] != ENTITY_SCHEMA:
        raise SequenceFeatureBlockError("entity format drift")
    _validate_file_ref(entity_section["file"], "entities.jsonl", entity_payload, "entity file reference", records=len(entities))
    relation_section = _strict_fields(manifest["relations"], {"format", "file", "relationSetSha256", "edges", "oneToManyRecords", "chooseFirstAllowed", "targetGenes", "targetsInUniverse"}, "manifest relations")
    if relation_section["format"] != RELATION_SCHEMA or relation_section["chooseFirstAllowed"] is not False:
        raise SequenceFeatureBlockError("relation manifest contract drift")
    _validate_file_ref(relation_section["file"], "relations.jsonl", relation_payload, "relation file reference", records=len(relations))
    if relation_section["relationSetSha256"] != _sha256_bytes(relation_payload):
        raise SequenceFeatureBlockError("relation set hash drift")
    hashes = manifest["semanticSetHashes"]
    if not isinstance(hashes, dict) or hashes.get("fullEntityKeySet") != {"basis": "ncbiTaxon-TAB-entityId", "sha256": expected.universe_entity_key_sha256}:
        raise SequenceFeatureBlockError("entity-universe semantic key hash drift")
    policy = _strict_fields(manifest["contentPolicy"], {"containsDisplaySymbols", "containsNumericFeatures", "containsOutcomesOrLabels", "containsTrainingPartitionAssignments", "crossTaxonIdentityMerge"}, "entity-universe content policy")
    if any(value is not False for value in policy.values()):
        raise SequenceFeatureBlockError("entity-universe content policy drift")

    audit = _read_canonical_json(paths["entity-universe-audit.json"], bounds.max_manifest_bytes, "entity-universe audit")
    if audit.get("schema") != "slp.static-entity-universe-audit/v1":
        raise SequenceFeatureBlockError("entity-universe audit schema drift")
    outputs = audit.get("outputs")
    if not isinstance(outputs, dict) or outputs.get("archiveSha256") != _sha256_file(archive_path) or outputs.get("manifestSha256") != _sha256_bytes(manifest_payload):
        raise SequenceFeatureBlockError("entity-universe audit does not bind its archive")
    if audit.get("inputs") != manifest["inputs"] or audit.get("semanticSetHashes") != manifest["semanticSetHashes"] or audit.get("identityMapping") != mapping or audit.get("source") != source:
        raise SequenceFeatureBlockError("entity-universe audit provenance drift")
    provenance = {
        "resource": expected.universe.resource,
        "revision": expected.universe.resource.rpartition("@")[2],
        "manifestDigest": expected.universe.manifest_digest,
        "treeDigest": expected.universe.tree_digest,
        "archive": _file_ref("entity-universe.tar", archive_path.read_bytes()),
        "audit": _file_ref("entity-universe-audit.json", paths["entity-universe-audit.json"].read_bytes()),
        "innerManifestSha256": _sha256_bytes(manifest_payload),
        "entityKeySetSha256": expected.universe_entity_key_sha256,
    }
    return entities, relations, manifest, provenance


def _mapping_output_spec(manifest: Mapping[str, Any], name: str) -> dict[str, Any]:
    basis = manifest.get("digestBasis")
    outputs = basis.get("outputFiles") if isinstance(basis, dict) else None
    matches = [item for item in outputs or [] if isinstance(item, dict) and item.get("name") == name]
    if len(matches) != 1 or set(matches[0]) != {"name", "records", "bytes", "sha256"}:
        raise SequenceFeatureBlockError(f"mapping manifest output contract drift for {name}")
    return matches[0]


def _load_mapping(current: LiteralArtifact, mapping_artifact: LiteralArtifact, bounds: Bounds, expected: ExpectedContract) -> tuple[dict[str, str], dict[str, object]]:
    _verify_artifact(current, expected.current_orfs)
    _verify_artifact(mapping_artifact, expected.mapping_manifest)
    manifest = _read_canonical_json(mapping_artifact.path, bounds.max_manifest_bytes, "SGD mapping manifest")
    if manifest.get("schema") != MAPPING_MANIFEST_SCHEMA or manifest.get("identityMappingId") != expected.mapping_id or manifest.get("identityMappingSha256") != expected.mapping_sha256 or manifest.get("ncbiTaxon") != SPECIES_TAXON:
        raise SequenceFeatureBlockError("SGD mapping manifest identity drift")
    basis = manifest.get("digestBasis")
    if not isinstance(basis, dict) or _sha256_bytes(_canonical_json_bytes(basis)) != expected.mapping_sha256:
        raise SequenceFeatureBlockError("SGD mapping digest basis does not reproduce its identity")
    spec = _mapping_output_spec(manifest, "current-orfs.jsonl")
    if spec != {"name": "current-orfs.jsonl", "records": expected.current_orf_records, "bytes": expected.current_orfs.bytes, "sha256": expected.current_orfs.sha256}:
        raise SequenceFeatureBlockError("current-ORF specification differs from its mapping manifest")
    records = _read_canonical_jsonl(current.path, bounds, bounds.max_records, "current ORFs")
    by_id: dict[str, str] = {}
    systematic_ids: set[str] = set()
    expected_fields = {"schema", "canonicalSgdCurie", "systematicName", "featureQualifier", "ncbiTaxon", "displayMetadata", "secondaryIdentifiers", "secondaryIdentifiersResolve"}
    for index, record in enumerate(records, start=1):
        row = _strict_fields(record, expected_fields, f"current ORF row {index}")
        curie, systematic = row["canonicalSgdCurie"], row["systematicName"]
        if row["schema"] != CURRENT_ORF_SCHEMA or row["ncbiTaxon"] != SPECIES_TAXON:
            raise SequenceFeatureBlockError("current ORF schema or taxon drift")
        if not isinstance(curie, str) or SGD_CURIE_RE.fullmatch(curie) is None or curie in by_id:
            raise SequenceFeatureBlockError("current ORF identity is invalid or duplicated")
        if not isinstance(systematic, str) or not systematic or systematic != systematic.strip() or systematic in systematic_ids:
            raise SequenceFeatureBlockError("current ORF systematic name is invalid or ambiguous")
        if not isinstance(row["featureQualifier"], str) or row["featureQualifier"] != row["featureQualifier"].strip():
            raise SequenceFeatureBlockError("current ORF feature qualifier is invalid")
        display = _strict_fields(row["displayMetadata"], {"aliases", "resolvesIdentity", "standardGeneName"}, "current ORF display metadata")
        if display["resolvesIdentity"] is not False or row["secondaryIdentifiersResolve"] is not False:
            raise SequenceFeatureBlockError("non-primary identifiers must not resolve identity")
        if not isinstance(display["aliases"], list) or any(not isinstance(item, str) for item in display["aliases"]):
            raise SequenceFeatureBlockError("current ORF aliases are malformed")
        if not isinstance(display["standardGeneName"], (str, type(None))) or not isinstance(row["secondaryIdentifiers"], list):
            raise SequenceFeatureBlockError("current ORF display or secondary metadata is malformed")
        by_id[curie] = systematic
        systematic_ids.add(systematic)
    if len(by_id) != expected.current_orf_records:
        raise SequenceFeatureBlockError("current ORF record count drift")
    provenance = {
        "resource": f"artifact:{expected.current_orfs.manifest_digest}",
        "artifactManifestDigest": expected.current_orfs.manifest_digest,
        "payload": _file_ref("payload", current.path.read_bytes(), records=len(records)),
    }
    mapping_provenance = {
        "resource": f"artifact:{expected.mapping_manifest.manifest_digest}",
        "artifactManifestDigest": expected.mapping_manifest.manifest_digest,
        "payload": _file_ref("payload", mapping_artifact.path.read_bytes()),
        "identityMappingId": expected.mapping_id,
        "identityMappingSha256": expected.mapping_sha256,
    }
    return by_id, {"sgdCurrentOrfs": provenance, "sgdMappingManifest": mapping_provenance}


def _source_class(description: str) -> str:
    """Parse the structural class field without consulting curated free text."""
    release_separator = f"Genome Release {HEADER_RELEASE}, "
    if description.count(release_separator) != 1:
        raise SequenceFeatureBlockError("FASTA header release field is missing or ambiguous")
    coordinate_prefix, remainder = description.split(release_separator, 1)
    if not coordinate_prefix.startswith("Chr ") or " from " not in coordinate_prefix:
        raise SequenceFeatureBlockError("FASTA header coordinate field is malformed")
    if remainder.startswith("reverse complement, "):
        remainder = remainder.removeprefix("reverse complement, ")
    source_token, separator, free_text = remainder.partition(", ")
    if not separator or not free_text:
        raise SequenceFeatureBlockError("FASTA header structural class field is malformed")
    matches = [name for name, token in SOURCE_CLASS_TOKENS.items() if source_token == token]
    if len(matches) != 1:
        raise SequenceFeatureBlockError("FASTA header structural class is unsupported")
    return matches[0]


def _decompress_exact(path: Path, bounds: Bounds, expected: ExpectedContract) -> bytes:
    try:
        with gzip.open(path, "rb") as stream:
            payload = stream.read(bounds.max_fasta_bytes + 1)
            if len(payload) > bounds.max_fasta_bytes or stream.read(1):
                raise SequenceFeatureBlockError("decompressed FASTA exceeds maxFastaBytes")
    except (OSError, EOFError, gzip.BadGzipFile) as error:
        raise SequenceFeatureBlockError("protein FASTA is not a valid gzip stream") from error
    if len(payload) != expected.fasta_decompressed_bytes or _sha256_bytes(payload) != expected.fasta_decompressed_sha256:
        raise SequenceFeatureBlockError("decompressed FASTA content drift")
    return payload


def _parse_fasta(payload: bytes, current_by_id: Mapping[str, str], bounds: Bounds, expected: ExpectedContract) -> tuple[dict[str, bytes], list[dict[str, object]], dict[str, int]]:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise SequenceFeatureBlockError("protein FASTA must be ASCII") from error
    if "\r" in text or not text.endswith("\n"):
        raise SequenceFeatureBlockError("protein FASTA must use LF endings with a terminal LF")
    raw_sequences: dict[str, str] = {}
    metadata: dict[str, tuple[str, str, str]] = {}
    current_header: tuple[str, str, str] | None = None
    fragments: list[str] = []

    def finish() -> None:
        nonlocal current_header, fragments
        if current_header is None:
            return
        curie, systematic, source_class = current_header
        raw = "".join(fragments)
        if not raw or len(raw) > bounds.max_sequence_length or SOURCE_PEPTIDE_RE.fullmatch(raw) is None:
            raise SequenceFeatureBlockError(f"invalid peptide alphabet or length for {curie}")
        if curie in current_by_id and CURRENT_PEPTIDE_RE.fullmatch(raw) is None:
            raise SequenceFeatureBlockError(
                f"current ORF {curie} must start M, use the 20-AA alphabet, and have exactly one terminal stop"
            )
        raw_sequences[curie] = raw
        metadata[curie] = (systematic, source_class, raw)
        current_header, fragments = None, []

    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.startswith(">"):
            finish()
            match = FASTA_HEADER_RE.fullmatch(line)
            if match is None:
                raise SequenceFeatureBlockError(f"FASTA header contract drift at line {line_number}")
            curie = f"SGD:{match.group('sgdid')}"
            if curie in raw_sequences:
                raise SequenceFeatureBlockError(f"duplicate FASTA SGD identity: {curie}")
            current_header = (curie, match.group("systematic"), _source_class(match.group("description")))
        else:
            if current_header is None or not line:
                raise SequenceFeatureBlockError(f"FASTA sequence line is orphaned or empty at line {line_number}")
            if len(line) > bounds.max_line_bytes:
                raise SequenceFeatureBlockError(f"FASTA line {line_number} exceeds maxLineBytes")
            fragments.append(line)
            if sum(map(len, fragments)) > bounds.max_sequence_length + 1:
                raise SequenceFeatureBlockError("FASTA sequence exceeds maxSequenceLength")
    finish()
    if len(raw_sequences) != expected.fasta_records:
        raise SequenceFeatureBlockError("FASTA record count drift")
    if not set(current_by_id) <= set(raw_sequences):
        raise SequenceFeatureBlockError("FASTA does not cover every current ORF")
    for curie, systematic in current_by_id.items():
        if metadata[curie][0] != systematic:
            raise SequenceFeatureBlockError(f"FASTA systematic-name mismatch for {curie}")
    non_current = sorted(set(raw_sequences) - set(current_by_id))
    if len(non_current) != expected.non_current_records:
        raise SequenceFeatureBlockError("non-current FASTA record count drift")
    stop_absent = tuple(sorted((curie, metadata[curie][0]) for curie in non_current if not metadata[curie][2].endswith("*")))
    if stop_absent != tuple(sorted(expected.stop_absent_non_current)):
        raise SequenceFeatureBlockError("stop-absent non-current sequence set drift")
    internal_stop = tuple(sorted(
        curie for curie in non_current
        if "*" in (metadata[curie][2].removesuffix("*"))
    ))
    if internal_stop != tuple(sorted(expected.internal_stop_non_current)):
        raise SequenceFeatureBlockError("internal-stop non-current sequence set drift")
    non_m_start = tuple(sorted(curie for curie in non_current if not metadata[curie][2].startswith("M")))
    if non_m_start != tuple(sorted(expected.non_m_start_non_current)):
        raise SequenceFeatureBlockError("non-methionine-start non-current sequence set drift")
    source_counts = Counter(item[1] for item in metadata.values())
    if source_counts != Counter(dict(expected.source_class_counts)):
        raise SequenceFeatureBlockError("FASTA source-class count drift")
    excluded = [
        {
            "schema": EXCLUDED_SEQUENCE_SCHEMA,
            "ncbiTaxon": SPECIES_TAXON,
            "sourceStrainTaxon": STRAIN_TAXON,
            "sequenceId": curie,
            "systematicName": metadata[curie][0],
            "sourceClass": metadata[curie][1],
            "exclusionReason": "not-current-orf",
            "terminalStopPresent": metadata[curie][2].endswith("*"),
            "internalStopCount": (metadata[curie][2].removesuffix("*")).count("*"),
            "startsWithMethionine": metadata[curie][2].startswith("M"),
            "rawSequenceSha256": _sha256_bytes(metadata[curie][2].encode("ascii")),
        }
        for curie in non_current
    ]
    return {
        curie: raw_sequences[curie][:-1].encode("ascii")
        for curie in current_by_id
    }, excluded, dict(source_counts)


def _load_sequences(paths: Mapping[str, Path], current_by_id: Mapping[str, str], bounds: Bounds, expected: ExpectedContract) -> tuple[dict[str, bytes], list[dict[str, object]], dict[str, object]]:
    fasta_path = paths["orf_trans_all_R64-5-1_20240529.fasta.gz"]
    payload = _decompress_exact(fasta_path, bounds, expected)
    sequences, excluded, source_counts = _parse_fasta(payload, current_by_id, bounds, expected)
    provenance = {
        "resource": expected.sequences.resource,
        "revision": expected.sequences.resource.rpartition("@")[2],
        "manifestDigest": expected.sequences.manifest_digest,
        "treeDigest": expected.sequences.tree_digest,
        "files": [_file_ref(spec.name, paths[spec.name].read_bytes()) for spec in sorted(expected.sequences.files, key=lambda item: item.name)],
        "decompressedFasta": {"bytes": len(payload), "sha256": _sha256_bytes(payload), "records": expected.fasta_records},
        "source": {"speciesTaxon": SPECIES_TAXON, "strainTaxon": STRAIN_TAXON, "release": SOURCE_RELEASE, "headerRelease": HEADER_RELEASE},
        "sourceClassCounts": {key: source_counts.get(key, 0) for key in sorted(SOURCE_CLASS_TOKENS)},
    }
    return sequences, excluded, provenance


def _npy_bytes(descr: str, shape: tuple[int, ...], payload: bytes) -> bytes:
    if descr not in {"<f4", "|b1"} or not shape or any(type(item) is not int or item < 0 for item in shape):
        raise SequenceFeatureBlockError("unsupported NPY dtype or shape")
    shape_text = repr(shape)
    header = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_text}, }}".encode("latin1")
    preamble = b"\x93NUMPY\x01\x00"
    padding = (-((len(preamble) + 2 + len(header) + 1) % 64)) % 64
    header += b" " * padding + b"\n"
    if len(header) > 65_535:
        raise SequenceFeatureBlockError("NPY v1.0 header exceeds uint16")
    return preamble + struct.pack("<H", len(header)) + header + payload


def _parse_npy(payload: bytes, expected_descr: str, expected_shape: tuple[int, ...], label: str) -> bytes:
    if len(payload) < 10 or payload[:8] != b"\x93NUMPY\x01\x00":
        raise SequenceFeatureBlockError(f"{label} is not NPY v1.0")
    header_length = struct.unpack("<H", payload[8:10])[0]
    expected_empty = _npy_bytes(expected_descr, expected_shape, b"")
    expected_header_length = struct.unpack("<H", expected_empty[8:10])[0]
    if header_length != expected_header_length or payload[:10 + header_length] != expected_empty:
        raise SequenceFeatureBlockError(f"{label} NPY header is not canonical")
    data = payload[10 + header_length:]
    elements = math.prod(expected_shape)
    expected_bytes = elements * (4 if expected_descr == "<f4" else 1)
    if len(data) != expected_bytes:
        raise SequenceFeatureBlockError(f"{label} NPY payload size drift")
    return data


def _feature_vector(peptide: bytes) -> tuple[float, ...]:
    if not peptide or any(chr(value) not in AA_ORDER for value in peptide):
        raise SequenceFeatureBlockError("feature peptide is empty or noncanonical")
    length = len(peptide)
    counts = Counter(peptide)
    return (length / 4096.0,) + tuple(counts[ord(residue)] / length for residue in AA_ORDER)


def _float32_rows(rows: Iterable[Sequence[float]]) -> bytes:
    output = bytearray()
    for row in rows:
        if len(row) != FEATURE_DIM or any(not math.isfinite(value) for value in row):
            raise SequenceFeatureBlockError("feature row is malformed or non-finite")
        output.extend(struct.pack("<21f", *row))
    return bytes(output)


def _validate_manifest_inputs(inputs_value: object, expected: ExpectedContract) -> dict[str, Any]:
    inputs = _strict_fields(
        inputs_value,
        {"staticEntityUniverse", "sgdProteinSequences", "sgdCurrentOrfs", "sgdMappingManifest"},
        "feature inputs",
    )
    universe = _strict_fields(
        inputs["staticEntityUniverse"],
        {"resource", "revision", "manifestDigest", "treeDigest", "archive", "audit", "innerManifestSha256", "entityKeySetSha256"},
        "feature inputs.staticEntityUniverse",
    )
    if (
        universe["resource"] != expected.universe.resource
        or universe["revision"] != expected.universe.resource.rpartition("@")[2]
        or universe["manifestDigest"] != expected.universe.manifest_digest
        or universe["treeDigest"] != expected.universe.tree_digest
        or universe["innerManifestSha256"] != expected.universe_manifest_sha256
        or universe["entityKeySetSha256"] != expected.universe_entity_key_sha256
    ):
        raise SequenceFeatureBlockError("static entity-universe input provenance drift")
    universe_specs = {item.name: item for item in expected.universe.files}
    for key, name in (("archive", "entity-universe.tar"), ("audit", "entity-universe-audit.json")):
        spec = universe_specs[name]
        if universe[key] != {"path": name, "sha256": spec.sha256, "bytes": spec.bytes}:
            raise SequenceFeatureBlockError("static entity-universe payload provenance drift")

    sequences = _strict_fields(
        inputs["sgdProteinSequences"],
        {"resource", "revision", "manifestDigest", "treeDigest", "files", "decompressedFasta", "source", "sourceClassCounts"},
        "feature inputs.sgdProteinSequences",
    )
    expected_sequence_source = {
        "speciesTaxon": SPECIES_TAXON,
        "strainTaxon": STRAIN_TAXON,
        "release": SOURCE_RELEASE,
        "headerRelease": HEADER_RELEASE,
    }
    if (
        sequences["resource"] != expected.sequences.resource
        or sequences["revision"] != expected.sequences.resource.rpartition("@")[2]
        or sequences["manifestDigest"] != expected.sequences.manifest_digest
        or sequences["treeDigest"] != expected.sequences.tree_digest
        or sequences["decompressedFasta"] != {
            "bytes": expected.fasta_decompressed_bytes,
            "sha256": expected.fasta_decompressed_sha256,
            "records": expected.fasta_records,
        }
        or sequences["source"] != expected_sequence_source
        or sequences["sourceClassCounts"] != {key: value for key, value in sorted(expected.source_class_counts)}
    ):
        raise SequenceFeatureBlockError("SGD protein-sequence input provenance drift")
    expected_sequence_files = [
        {"path": spec.name, "sha256": spec.sha256, "bytes": spec.bytes}
        for spec in sorted(expected.sequences.files, key=lambda item: item.name)
    ]
    if sequences["files"] != expected_sequence_files:
        raise SequenceFeatureBlockError("SGD protein-sequence file provenance drift")

    for name, artifact_expected, records in (
        ("sgdCurrentOrfs", expected.current_orfs, expected.current_orf_records),
        ("sgdMappingManifest", expected.mapping_manifest, None),
    ):
        extra = {"identityMappingId", "identityMappingSha256"} if name == "sgdMappingManifest" else set()
        artifact = _strict_fields(
            inputs[name], {"resource", "artifactManifestDigest", "payload"} | extra,
            f"feature inputs.{name}",
        )
        if artifact["resource"] != f"artifact:{artifact_expected.manifest_digest}" or artifact["artifactManifestDigest"] != artifact_expected.manifest_digest:
            raise SequenceFeatureBlockError(f"{name} artifact identity drift")
        payload_expected: dict[str, object] = {
            "path": "payload", "sha256": artifact_expected.sha256, "bytes": artifact_expected.bytes,
        }
        if records is not None:
            payload_expected["records"] = records
        if artifact["payload"] != payload_expected:
            raise SequenceFeatureBlockError(f"{name} payload provenance drift")
        if name == "sgdMappingManifest" and (
            artifact["identityMappingId"] != expected.mapping_id
            or artifact["identityMappingSha256"] != expected.mapping_sha256
        ):
            raise SequenceFeatureBlockError("mapping identity provenance drift")
    return inputs


def validate_archive(
    path: str | Path,
    bounds: Bounds,
    *,
    expected: ExpectedContract = PRODUCTION_CONTRACT,
) -> dict[str, Any]:
    archive_path = Path(path)
    names = (
        "static-feature-block/entities.jsonl",
        "static-feature-block/excluded-non-current.jsonl",
        "static-feature-block/manifest.json",
        "static-feature-block/present.npy",
        "static-feature-block/sequence-provenance.jsonl",
        "static-feature-block/values.npy",
    )
    blobs = _read_exact_tar(archive_path, names, bounds, "sequence feature-block archive")
    manifest_payload = blobs[names[2]]
    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SequenceFeatureBlockError("feature-block manifest is invalid") from error
    if not isinstance(manifest, dict) or manifest_payload != _pretty_json_bytes(manifest):
        raise SequenceFeatureBlockError("feature-block manifest is not canonical JSON")
    manifest = _strict_fields(
        manifest,
        {"schema", "version", "identityKey", "ordering", "featureDefinition", "source", "identityMapping", "inputs", "files", "counts", "semanticHashes", "contentPolicy"},
        "feature-block manifest",
    )
    if (
        manifest["schema"] != FEATURE_BLOCK_SCHEMA
        or type(manifest["version"]) is not int
        or manifest["version"] != 1
        or manifest["identityKey"] != ["ncbiTaxon", "entityId"]
        or manifest["ordering"] != "ascending-ncbiTaxon-then-codepoint-entityId"
    ):
        raise SequenceFeatureBlockError("feature-block schema or identity contract drift")
    definition = _strict_fields(
        manifest["featureDefinition"],
        {"names", "dimension", "dtype", "layout", "floatSemantics", "formula", "lengthScale", "aminoAcidOrder", "stopPolicy", "normalization", "clipping", "fitted", "parameterCount", "identifierFeatures", "missingness"},
        "feature definition",
    )
    if (
        any(type(definition[key]) is not bool for key in ("clipping", "fitted", "identifierFeatures"))
        or type(definition["parameterCount"]) is not int
        or definition != {
        "names": list(FEATURE_NAMES),
        "dimension": FEATURE_DIM,
        "dtype": "little-endian-float32",
        "layout": "C-row-major",
        "floatSemantics": "IEEE-754-binary32-round-to-nearest-ties-to-even",
        "formula": "[len(peptide)/4096] + [count(aa)/len(peptide) for aa in aminoAcidOrder]",
        "lengthScale": 4096,
        "aminoAcidOrder": AA_ORDER,
        "stopPolicy": "feature-bearing-current-ORFs-require-and-strip-exactly-one-terminal-stop",
        "normalization": "none-beyond-declared-ratios",
        "clipping": False,
        "fitted": False,
        "parameterCount": 0,
        "identifierFeatures": False,
        "missingness": "all-values-present",
        }
    ):
        raise SequenceFeatureBlockError("immutable feature definition drift")
    if manifest["source"] != {"speciesTaxon": SPECIES_TAXON, "strainTaxon": STRAIN_TAXON, "release": SOURCE_RELEASE}:
        raise SequenceFeatureBlockError("feature source provenance drift")
    if manifest["identityMapping"] != {"id": expected.mapping_id, "sha256": expected.mapping_sha256}:
        raise SequenceFeatureBlockError("feature identity mapping drift")
    _validate_manifest_inputs(manifest["inputs"], expected)
    counts = _strict_fields(
        manifest["counts"],
        {"rows", "genes", "proteins", "featuresPerRow", "presentValues", "excludedNonCurrentSequences", "currentOrfsOutsideUniverse", "multiTargetProteinConsensus", "missingEntities", "ambiguousEntities", "trainableParameters"},
        "feature counts",
    )
    expected_counts = {
        "rows": expected.universe_rows,
        "genes": expected.universe_genes,
        "proteins": expected.universe_proteins,
        "featuresPerRow": FEATURE_DIM,
        "presentValues": expected.present_values,
        "excludedNonCurrentSequences": expected.non_current_records,
        "currentOrfsOutsideUniverse": expected.current_outside_universe,
        "multiTargetProteinConsensus": len(expected.multi_target_peptide_sha256),
        "missingEntities": 0,
        "ambiguousEntities": 0,
        "trainableParameters": 0,
    }
    if any(type(value) is not int for value in counts.values()) or counts != expected_counts:
        raise SequenceFeatureBlockError("feature production counts drift")
    rows = counts["rows"]
    entities = _jsonl_blob(blobs[names[0]], bounds, bounds.max_records, "feature entities")
    excluded = _jsonl_blob(blobs[names[1]], bounds, bounds.max_records, "excluded non-current sequences")
    provenance = _jsonl_blob(blobs[names[4]], bounds, bounds.max_records, "sequence provenance")
    if len(excluded) != counts["excludedNonCurrentSequences"]:
        raise SequenceFeatureBlockError("excluded sequence count does not match the manifest")
    if (
        expected.output_sequence_provenance_sha256 is not None
        and _sha256_bytes(blobs[names[4]]) != expected.output_sequence_provenance_sha256
    ):
        raise SequenceFeatureBlockError(
            "sequence provenance differs from the production-pinned source/relation map"
        )
    if (
        expected.output_excluded_non_current_sha256 is not None
        and _sha256_bytes(blobs[names[1]]) != expected.output_excluded_non_current_sha256
    ):
        raise SequenceFeatureBlockError(
            "excluded-sequence provenance differs from the production-pinned quarantine"
        )
    if len(entities) != rows or len(provenance) != rows:
        raise SequenceFeatureBlockError("feature row/provenance count mismatch")
    keys: list[tuple[int, str]] = []
    for index, (entity, provenance_row) in enumerate(zip(entities, provenance, strict=True)):
        row = _strict_fields(entity, {"schema", "rowIndex", "ncbiTaxon", "entityId"}, f"feature entity {index}")
        prov = _strict_fields(
            provenance_row,
            {"schema", "rowIndex", "ncbiTaxon", "entityId", "sourceStrainTaxon", "sourceSequenceIds", "derivation", "canonicalPeptideSha256", "canonicalPeptideLength", "aminoAcidCounts"},
            f"sequence provenance {index}",
        )
        if (
            row["schema"] != FEATURE_ENTITY_SCHEMA
            or type(row["rowIndex"]) is not int
            or row["rowIndex"] != index
            or row["ncbiTaxon"] != SPECIES_TAXON
        ):
            raise SequenceFeatureBlockError("feature entity ordering or taxon drift")
        if (
            prov["schema"] != SEQUENCE_PROVENANCE_SCHEMA
            or type(prov["rowIndex"]) is not int
            or prov["rowIndex"] != index
            or (prov["ncbiTaxon"], prov["entityId"])
            != (row["ncbiTaxon"], row["entityId"])
            or prov["sourceStrainTaxon"] != STRAIN_TAXON
        ):
            raise SequenceFeatureBlockError("sequence provenance does not align with entity row")
        ids = prov["sourceSequenceIds"]
        if not isinstance(ids, list) or not ids or ids != sorted(set(ids)) or any(not isinstance(item, str) or SGD_CURIE_RE.fullmatch(item) is None for item in ids):
            raise SequenceFeatureBlockError("sequence provenance IDs are invalid")
        _bare_digest(prov["canonicalPeptideSha256"], "canonical peptide hash")
        length, aa_counts = prov["canonicalPeptideLength"], prov["aminoAcidCounts"]
        if type(length) is not int or length <= 0 or not isinstance(aa_counts, list) or len(aa_counts) != len(AA_ORDER) or any(type(value) is not int or value < 0 for value in aa_counts) or sum(aa_counts) != length:
            raise SequenceFeatureBlockError("sequence sufficient statistics are invalid")
        if row["entityId"].startswith("SGD:"):
            if prov["derivation"] != "direct-current-orf" or ids != [row["entityId"]]:
                raise SequenceFeatureBlockError("gene sequence provenance is not direct")
        elif row["entityId"].startswith("UniProtKB:"):
            if prov["derivation"] != "exact-related-peptide-consensus":
                raise SequenceFeatureBlockError("protein sequence provenance is not relation-derived")
        else:
            raise SequenceFeatureBlockError("feature entity namespace is unsupported")
        keys.append((row["ncbiTaxon"], row["entityId"]))
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise SequenceFeatureBlockError("feature entity keys are duplicated or out of order")
    if framed_key_sha256(keys) != expected.universe_entity_key_sha256:
        raise SequenceFeatureBlockError("feature rows differ from the pinned composite universe")
    gene_count = sum(entity["entityId"].startswith("SGD:") for entity in entities)
    if gene_count != expected.universe_genes or len(entities) - gene_count != expected.universe_proteins:
        raise SequenceFeatureBlockError("feature namespace counts drift")
    multi_target = {
        row["entityId"]: row["canonicalPeptideSha256"]
        for row in provenance if len(row["sourceSequenceIds"]) > 1
    }
    if multi_target != dict(expected.multi_target_peptide_sha256):
        raise SequenceFeatureBlockError("multi-target peptide consensus provenance drift")

    excluded_fields = {"schema", "ncbiTaxon", "sourceStrainTaxon", "sequenceId", "systematicName", "sourceClass", "exclusionReason", "terminalStopPresent", "internalStopCount", "startsWithMethionine", "rawSequenceSha256"}
    excluded_ids: list[str] = []
    absent_stop: list[tuple[str, str]] = []
    internal_stop: list[str] = []
    non_m_start: list[str] = []
    for index, excluded_row in enumerate(excluded):
        row = _strict_fields(excluded_row, excluded_fields, f"excluded sequence {index}")
        if row["schema"] != EXCLUDED_SEQUENCE_SCHEMA or row["ncbiTaxon"] != SPECIES_TAXON or row["sourceStrainTaxon"] != STRAIN_TAXON or row["exclusionReason"] != "not-current-orf":
            raise SequenceFeatureBlockError("excluded sequence schema or provenance drift")
        if not isinstance(row["sequenceId"], str) or SGD_CURIE_RE.fullmatch(row["sequenceId"]) is None or not isinstance(row["systematicName"], str) or not row["systematicName"]:
            raise SequenceFeatureBlockError("excluded sequence identity is invalid")
        if row["sourceClass"] not in SOURCE_CLASS_TOKENS or type(row["terminalStopPresent"]) is not bool or type(row["startsWithMethionine"]) is not bool or type(row["internalStopCount"]) is not int or row["internalStopCount"] < 0:
            raise SequenceFeatureBlockError("excluded sequence anomaly metadata is invalid")
        _bare_digest(row["rawSequenceSha256"], "excluded raw-sequence SHA-256")
        excluded_ids.append(row["sequenceId"])
        if not row["terminalStopPresent"]:
            absent_stop.append((row["sequenceId"], row["systematicName"]))
        if row["internalStopCount"]:
            internal_stop.append(row["sequenceId"])
        if not row["startsWithMethionine"]:
            non_m_start.append(row["sequenceId"])
    if excluded_ids != sorted(set(excluded_ids)) or tuple(absent_stop) != tuple(sorted(expected.stop_absent_non_current)) or tuple(internal_stop) != tuple(sorted(expected.internal_stop_non_current)) or tuple(non_m_start) != tuple(sorted(expected.non_m_start_non_current)):
        raise SequenceFeatureBlockError("excluded sequence anomaly set drift")
    present_data = _parse_npy(blobs[names[3]], "|b1", (rows, FEATURE_DIM), "present.npy")
    if any(value != 1 for value in present_data) or counts["presentValues"] != rows * FEATURE_DIM:
        raise SequenceFeatureBlockError("feature mask is not entirely present")
    values_data = _parse_npy(blobs[names[5]], "<f4", (rows, FEATURE_DIM), "values.npy")
    expected_value_data = _float32_rows(
        (row["canonicalPeptideLength"] / 4096.0,) + tuple(
            count / row["canonicalPeptideLength"] for count in row["aminoAcidCounts"]
        )
        for row in provenance
    )
    if values_data != expected_value_data:
        raise SequenceFeatureBlockError("feature values do not equal the declared sufficient-statistics transform")
    file_fields = _strict_fields(manifest["files"], {"entities", "excludedNonCurrent", "present", "sequenceProvenance", "values"}, "feature files")
    for key, member, record_count in (
        ("entities", names[0], len(entities)),
        ("excludedNonCurrent", names[1], len(excluded)),
        ("present", names[3], None),
        ("sequenceProvenance", names[4], len(provenance)),
        ("values", names[5], None),
    ):
        _validate_file_ref(file_fields[key], member.removeprefix("static-feature-block/"), blobs[member], f"files.{key}", records=record_count)
    hashes = _strict_fields(manifest["semanticHashes"], {"entityKeySetSha256", "featureDefinitionSha256", "sequenceProvenanceSha256"}, "semantic hashes")
    if hashes["entityKeySetSha256"] != framed_key_sha256(keys) or hashes["featureDefinitionSha256"] != _sha256_bytes(_canonical_json_bytes(definition)) or hashes["sequenceProvenanceSha256"] != _sha256_bytes(blobs[names[4]]):
        raise SequenceFeatureBlockError("feature semantic hash drift")
    policy = _strict_fields(manifest["contentPolicy"], {"containsIdentifiersAsValues", "containsOutcomesOrLabels", "containsTrainingPartitionAssignments", "containsBenchmarkData", "containsFreeTextDescriptions", "crossTaxonIdentityMerge"}, "feature content policy")
    if any(value is not False for value in policy.values()):
        raise SequenceFeatureBlockError("feature content policy drift")
    return manifest


def validate_audit(
    audit_path: str | Path,
    archive_path: str | Path,
    bounds: Bounds,
    *,
    expected: ExpectedContract = PRODUCTION_CONTRACT,
) -> dict[str, Any]:
    audit_file = Path(audit_path)
    archive_file = Path(archive_path)
    audit = _read_canonical_json(audit_file, bounds.max_manifest_bytes, "feature-block audit")
    audit = _strict_fields(
        audit,
        {"schema", "inputs", "source", "identityMapping", "featureDefinition", "outputs", "counts", "multiTargetPeptideConsensus", "accessBoundary", "limitations"},
        "feature-block audit",
    )
    if audit["schema"] != AUDIT_SCHEMA:
        raise SequenceFeatureBlockError("feature-block audit schema drift")
    manifest = validate_archive(archive_file, bounds, expected=expected)
    if (
        audit["inputs"] != manifest["inputs"]
        or audit["source"] != manifest["source"]
        or audit["identityMapping"] != manifest["identityMapping"]
        or audit["featureDefinition"] != manifest["featureDefinition"]
        or audit["counts"] != manifest["counts"]
    ):
        raise SequenceFeatureBlockError("feature-block audit does not bind its manifest")
    names = (
        "static-feature-block/entities.jsonl",
        "static-feature-block/excluded-non-current.jsonl",
        "static-feature-block/manifest.json",
        "static-feature-block/present.npy",
        "static-feature-block/sequence-provenance.jsonl",
        "static-feature-block/values.npy",
    )
    blobs = _read_exact_tar(archive_file, names, bounds, "sequence feature-block archive")
    expected_outputs = {
        "archive": _file_ref("sequence-feature-block.tar", archive_file.read_bytes()),
        "manifestSha256": _sha256_bytes(blobs[names[2]]),
        "entityRowsSha256": _sha256_bytes(blobs[names[0]]),
        "excludedNonCurrentSha256": _sha256_bytes(blobs[names[1]]),
        "presentNpySha256": _sha256_bytes(blobs[names[3]]),
        "sequenceProvenanceSha256": _sha256_bytes(blobs[names[4]]),
        "valuesNpySha256": _sha256_bytes(blobs[names[5]]),
    }
    if audit["outputs"] != expected_outputs:
        raise SequenceFeatureBlockError("feature-block audit output hashes drift")
    expected_consensus = [
        {"proteinId": protein, "canonicalPeptideSha256": digest}
        for protein, digest in sorted(expected.multi_target_peptide_sha256)
    ]
    if audit["multiTargetPeptideConsensus"] != expected_consensus:
        raise SequenceFeatureBlockError("feature-block audit consensus evidence drift")
    boundary = audit["accessBoundary"]
    if (
        not isinstance(boundary, dict)
        or set(boundary) != set(ACCESS_BOUNDARY)
        or any(type(boundary[key]) is not bool for key in ACCESS_BOUNDARY if key != "inputNames")
        or boundary != ACCESS_BOUNDARY
        or audit["limitations"] != LIMITATIONS
    ):
        raise SequenceFeatureBlockError("feature-block audit boundary or limitations drift")
    return audit


def build_sequence_feature_block(
    universe_dataset: PinnedDataset,
    sequence_dataset: PinnedDataset,
    current_artifact: LiteralArtifact,
    mapping_artifact: LiteralArtifact,
    destination: str | Path,
    bounds: Bounds,
    *,
    expected: ExpectedContract = PRODUCTION_CONTRACT,
) -> dict[str, object]:
    universe_paths = _verify_dataset(universe_dataset, expected.universe)
    sequence_paths = _verify_dataset(sequence_dataset, expected.sequences)
    entities, relations, _universe_manifest, universe_provenance = _validate_universe(universe_paths, bounds, expected)
    current_by_id, mapping_provenance = _load_mapping(current_artifact, mapping_artifact, bounds, expected)
    current_sequences, excluded, sequence_provenance_input = _load_sequences(sequence_paths, current_by_id, bounds, expected)
    gene_ids = {item["entityId"] for item in entities if item["entityClass"] == "gene"}
    if not gene_ids <= set(current_by_id) or not gene_ids <= set(current_sequences):
        raise SequenceFeatureBlockError("universe gene lacks a current SGD peptide")
    if len(set(current_by_id) - gene_ids) != expected.current_outside_universe:
        raise SequenceFeatureBlockError("current-ORF outside-universe count drift")
    relation_map = {item["proteinId"]: tuple(item["currentOrfRelations"]) for item in relations}

    feature_entities: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    feature_rows: list[tuple[float, ...]] = []
    multi_target_hashes: dict[str, str] = {}
    for row_index, entity in enumerate(entities):
        entity_id = entity["entityId"]
        if entity["entityClass"] == "gene":
            source_ids = (entity_id,)
            derivation = "direct-current-orf"
        else:
            source_ids = relation_map[entity_id]
            derivation = "exact-related-peptide-consensus"
        peptides = [current_sequences[item] for item in source_ids]
        if not peptides or any(peptide != peptides[0] for peptide in peptides[1:]):
            raise SequenceFeatureBlockError(f"related peptides do not form an exact consensus for {entity_id}")
        peptide = peptides[0]
        peptide_sha = _sha256_bytes(peptide)
        residue_counts = Counter(peptide)
        if len(source_ids) > 1:
            multi_target_hashes[entity_id] = peptide_sha
        feature_entities.append({"schema": FEATURE_ENTITY_SCHEMA, "rowIndex": row_index, "ncbiTaxon": SPECIES_TAXON, "entityId": entity_id})
        provenance_rows.append({
            "schema": SEQUENCE_PROVENANCE_SCHEMA,
            "rowIndex": row_index,
            "ncbiTaxon": SPECIES_TAXON,
            "entityId": entity_id,
            "sourceStrainTaxon": STRAIN_TAXON,
            "sourceSequenceIds": list(source_ids),
            "derivation": derivation,
            "canonicalPeptideSha256": peptide_sha,
            "canonicalPeptideLength": len(peptide),
            "aminoAcidCounts": [residue_counts[ord(residue)] for residue in AA_ORDER],
        })
        feature_rows.append(_feature_vector(peptide))
    if multi_target_hashes != dict(expected.multi_target_peptide_sha256):
        raise SequenceFeatureBlockError("multi-target peptide consensus set drift")
    if len(feature_rows) != expected.universe_rows or len(feature_rows) * FEATURE_DIM != expected.present_values:
        raise SequenceFeatureBlockError("feature matrix cardinality drift")

    entity_bytes = _jsonl_bytes(feature_entities)
    excluded_bytes = _jsonl_bytes(excluded)
    provenance_bytes = _jsonl_bytes(provenance_rows)
    values_data = _float32_rows(feature_rows)
    values_bytes = _npy_bytes("<f4", (len(feature_rows), FEATURE_DIM), values_data)
    present_bytes = _npy_bytes("|b1", (len(feature_rows), FEATURE_DIM), b"\x01" * expected.present_values)
    definition = {
        "names": list(FEATURE_NAMES),
        "dimension": FEATURE_DIM,
        "dtype": "little-endian-float32",
        "layout": "C-row-major",
        "floatSemantics": "IEEE-754-binary32-round-to-nearest-ties-to-even",
        "formula": "[len(peptide)/4096] + [count(aa)/len(peptide) for aa in aminoAcidOrder]",
        "lengthScale": 4096,
        "aminoAcidOrder": AA_ORDER,
        "stopPolicy": "feature-bearing-current-ORFs-require-and-strip-exactly-one-terminal-stop",
        "normalization": "none-beyond-declared-ratios",
        "clipping": False,
        "fitted": False,
        "parameterCount": 0,
        "identifierFeatures": False,
        "missingness": "all-values-present",
    }
    inputs = {
        "staticEntityUniverse": universe_provenance,
        "sgdProteinSequences": sequence_provenance_input,
        **mapping_provenance,
    }
    genes = sum(item["entityClass"] == "gene" for item in entities)
    proteins = len(entities) - genes
    manifest = {
        "schema": FEATURE_BLOCK_SCHEMA,
        "version": 1,
        "identityKey": ["ncbiTaxon", "entityId"],
        "ordering": "ascending-ncbiTaxon-then-codepoint-entityId",
        "featureDefinition": definition,
        "source": {"speciesTaxon": SPECIES_TAXON, "strainTaxon": STRAIN_TAXON, "release": SOURCE_RELEASE},
        "identityMapping": {"id": expected.mapping_id, "sha256": expected.mapping_sha256},
        "inputs": inputs,
        "files": {
            "entities": _file_ref("entities.jsonl", entity_bytes, records=len(feature_entities)),
            "excludedNonCurrent": _file_ref("excluded-non-current.jsonl", excluded_bytes, records=len(excluded)),
            "present": _file_ref("present.npy", present_bytes),
            "sequenceProvenance": _file_ref("sequence-provenance.jsonl", provenance_bytes, records=len(provenance_rows)),
            "values": _file_ref("values.npy", values_bytes),
        },
        "counts": {
            "rows": len(feature_rows), "genes": genes, "proteins": proteins,
            "featuresPerRow": FEATURE_DIM, "presentValues": len(feature_rows) * FEATURE_DIM,
            "excludedNonCurrentSequences": len(excluded),
            "currentOrfsOutsideUniverse": len(set(current_by_id) - gene_ids),
            "multiTargetProteinConsensus": len(multi_target_hashes),
            "missingEntities": 0,
            "ambiguousEntities": 0,
            "trainableParameters": 0,
        },
        "semanticHashes": {
            "entityKeySetSha256": framed_key_sha256((item["ncbiTaxon"], item["entityId"]) for item in feature_entities),
            "featureDefinitionSha256": _sha256_bytes(_canonical_json_bytes(definition)),
            "sequenceProvenanceSha256": _sha256_bytes(provenance_bytes),
        },
        "contentPolicy": {
            "containsIdentifiersAsValues": False,
            "containsOutcomesOrLabels": False,
            "containsTrainingPartitionAssignments": False,
            "containsBenchmarkData": False,
            "containsFreeTextDescriptions": False,
            "crossTaxonIdentityMerge": False,
        },
    }
    manifest_bytes = _pretty_json_bytes(manifest)
    members = {
        "static-feature-block/entities.jsonl": entity_bytes,
        "static-feature-block/excluded-non-current.jsonl": excluded_bytes,
        "static-feature-block/manifest.json": manifest_bytes,
        "static-feature-block/present.npy": present_bytes,
        "static-feature-block/sequence-provenance.jsonl": provenance_bytes,
        "static-feature-block/values.npy": values_bytes,
    }
    destination_path = Path(destination).resolve()
    if destination_path.exists() or destination_path.is_symlink():
        raise SequenceFeatureBlockError("destination must not already exist")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination_path.name}-", dir=destination_path.parent) as temporary:
        staging = Path(temporary) / destination_path.name
        staging.mkdir()
        archive_path = staging / "sequence-feature-block.tar"
        _write_tar(archive_path, members)
        validated = validate_archive(archive_path, bounds, expected=expected)
        if validated != manifest:
            raise SequenceFeatureBlockError("post-write validation changed the manifest")
        archive_sha = _sha256_file(archive_path)
        audit = {
            "schema": AUDIT_SCHEMA,
            "inputs": inputs,
            "source": manifest["source"],
            "identityMapping": manifest["identityMapping"],
            "featureDefinition": definition,
            "outputs": {
                "archive": _file_ref("sequence-feature-block.tar", archive_path.read_bytes()),
                "manifestSha256": _sha256_bytes(manifest_bytes),
                "entityRowsSha256": _sha256_bytes(entity_bytes),
                "excludedNonCurrentSha256": _sha256_bytes(excluded_bytes),
                "presentNpySha256": _sha256_bytes(present_bytes),
                "sequenceProvenanceSha256": _sha256_bytes(provenance_bytes),
                "valuesNpySha256": _sha256_bytes(values_bytes),
            },
            "counts": manifest["counts"],
            "multiTargetPeptideConsensus": [
                {"proteinId": protein, "canonicalPeptideSha256": digest}
                for protein, digest in sorted(multi_target_hashes.items())
            ],
            "accessBoundary": ACCESS_BOUNDARY,
            "limitations": LIMITATIONS,
        }
        audit_path = staging / "sequence-feature-block-audit.json"
        audit_path.write_bytes(_pretty_json_bytes(audit))
        validated_audit = validate_audit(audit_path, archive_path, bounds, expected=expected)
        if validated_audit != audit:
            raise SequenceFeatureBlockError("post-write validation changed the audit")
        audit_sha = _sha256_file(audit_path)
        staging.replace(destination_path)
    return {
        "archiveSha256": archive_sha,
        "auditSha256": audit_sha,
        "manifestSha256": _sha256_bytes(manifest_bytes),
        "entityRowsSha256": _sha256_bytes(entity_bytes),
        "valuesNpySha256": _sha256_bytes(values_bytes),
        "presentNpySha256": _sha256_bytes(present_bytes),
        "sequenceProvenanceSha256": _sha256_bytes(provenance_bytes),
        "excludedNonCurrentSha256": _sha256_bytes(excluded_bytes),
        "featureDefinitionSha256": manifest["semanticHashes"]["featureDefinitionSha256"],
        "entityKeySetSha256": manifest["semanticHashes"]["entityKeySetSha256"],
        "rows": len(feature_rows),
        "featureDimension": FEATURE_DIM,
        "presentValues": len(feature_rows) * FEATURE_DIM,
        "excludedNonCurrentSequences": len(excluded),
        "currentOrfsOutsideUniverse": len(set(current_by_id) - gene_ids),
        "multiTargetProteinConsensus": len(multi_target_hashes),
        "audit": audit,
    }
