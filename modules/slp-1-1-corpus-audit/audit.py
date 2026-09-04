"""Content and leakage checks for separately governed SLp corpus snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "slp.corpus/v1"
ROLES = {
    "pretrain": "pretrain",
    "molecularValidation": "molecular-validation",
    "molecularReward": "molecular-reward",
}


class CorpusAuditError(ValueError):
    pass


@dataclass(frozen=True)
class Corpus:
    root: Path
    manifest: dict[str, Any]
    trajectory_genes: frozenset[str]
    records: int


def _relative_file(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise CorpusAuditError("corpus paths must be non-empty strings")
    portable = PurePosixPath(value)
    if portable.is_absolute() or ".." in portable.parts:
        raise CorpusAuditError(f"unsafe corpus path: {value!r}")
    path = root.joinpath(*portable.parts)
    if not path.is_file():
        raise CorpusAuditError(f"missing corpus file: {value}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_corpus(root: str | Path, expected_role: str) -> Corpus:
    root = Path(root).resolve()
    manifest_path = root / "corpus.json" if root.is_dir() else root
    if not manifest_path.is_file():
        raise CorpusAuditError("snapshot must contain corpus.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "datasetId",
        "version",
        "role",
        "labelClass",
        "benchmarkLabelsPresent",
        "speciesTaxa",
        "modalities",
        "trajectoryGenes",
        "shards",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise CorpusAuditError(f"corpus manifest missing fields: {', '.join(missing)}")
    if manifest["schema"] != SCHEMA:
        raise CorpusAuditError(f"unsupported corpus schema: {manifest['schema']!r}")
    if manifest["role"] != expected_role:
        raise CorpusAuditError(
            f"expected {expected_role!r} corpus, received {manifest['role']!r}"
        )
    if manifest["labelClass"] != "molecular":
        raise CorpusAuditError("training corpora may contain only molecular labels")
    if not isinstance(manifest["benchmarkLabelsPresent"], bool):
        raise CorpusAuditError("benchmarkLabelsPresent must be boolean")
    species = manifest["speciesTaxa"]
    if (
        not isinstance(species, list)
        or not species
        or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in species)
        or len(species) != len(set(species))
    ):
        raise CorpusAuditError("speciesTaxa must contain unique positive NCBI taxonomy IDs")
    modalities = manifest["modalities"]
    if (
        not isinstance(modalities, list)
        or not modalities
        or any(not isinstance(item, str) or not item for item in modalities)
    ):
        raise CorpusAuditError("modalities must be a non-empty string list")

    gene_path = _relative_file(root, manifest["trajectoryGenes"])
    genes = frozenset(
        line.strip()
        for line in gene_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not genes or any(":" not in gene for gene in genes):
        raise CorpusAuditError("trajectory genes must be stable CURIE identifiers")

    shards = manifest["shards"]
    if not isinstance(shards, list) or not shards:
        raise CorpusAuditError("corpus must contain at least one shard")
    records = 0
    seen_paths: set[str] = set()
    for shard in shards:
        if not isinstance(shard, dict) or set(shard) != {"path", "sha256", "records"}:
            raise CorpusAuditError("each shard requires only path, sha256, and records")
        path_value = shard["path"]
        if not isinstance(path_value, str):
            raise CorpusAuditError("shard paths must be strings")
        if path_value in seen_paths:
            raise CorpusAuditError(f"duplicate shard path: {path_value}")
        seen_paths.add(path_value)
        path = _relative_file(root, path_value)
        expected = shard["sha256"]
        if not isinstance(expected, str) or len(expected) != 64 or _sha256(path) != expected:
            raise CorpusAuditError(f"shard digest mismatch: {path_value}")
        count = shard["records"]
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise CorpusAuditError("shard records must be positive integers")
        records += count
    return Corpus(root=root, manifest=manifest, trajectory_genes=genes, records=records)


def audit_corpora(paths: dict[str, str | Path], strict: bool = True) -> dict[str, Any]:
    corpora = {
        name: load_corpus(paths[name], expected_role)
        for name, expected_role in ROLES.items()
    }
    fitting_genes = corpora["pretrain"].trajectory_genes | corpora["molecularReward"].trajectory_genes
    validation_genes = corpora["molecularValidation"].trajectory_genes
    leaked_genes = sorted(fitting_genes & validation_genes)
    benchmark_records = sum(
        corpus.records
        for corpus in corpora.values()
        if corpus.manifest["benchmarkLabelsPresent"]
    )
    species = sorted(
        {
            taxon
            for corpus in corpora.values()
            for taxon in corpus.manifest["speciesTaxa"]
        }
    )
    passed = benchmark_records == 0 and (not strict or not leaked_genes)
    return {
        "schema": "slp.corpus-audit/v1",
        "strictInterventionIsolation": strict,
        "auditPassed": passed,
        "leakageViolations": len(leaked_genes),
        "leakedTrajectoryGenes": leaked_genes,
        "benchmarkLabelRecords": benchmark_records,
        "records": {name: corpus.records for name, corpus in corpora.items()},
        "speciesTaxa": species,
        "datasets": {
            name: {
                "datasetId": corpus.manifest["datasetId"],
                "version": corpus.manifest["version"],
                "role": corpus.manifest["role"],
                "modalities": corpus.manifest["modalities"],
            }
            for name, corpus in corpora.items()
        },
    }
