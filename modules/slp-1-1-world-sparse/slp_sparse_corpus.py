"""Strict typed dictionaries, sparse CSR targets, and deterministic sampling.

The loader intentionally keeps provenance dictionaries outside the model. It
materializes only the selected records and their bounded query panels; it never
constructs an array whose axes are the full record and query dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re
from typing import Any, Sequence

import numpy as np
import torch

from slp_sparse_architecture import (
    LIKELIHOODS,
    NEGATIVE_BINOMIAL,
    WorldBatch,
    WorldConfig,
)


CORPUS_SCHEMA = "slp.corpus/v1.1"
SAMPLING_SCHEME = "slp.source-intervention-replicate-record/v1"
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
ROLES = {
    "pretrain",
    "molecular-validation",
    "molecular-reward",
    "molecular-final",
}
ACCESS_ROLES = {"world", "likelihood", "audit"}
CURIE = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*:[^\s]+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RESOURCE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class FileReference:
    path: str
    sha256: str
    count: int


@dataclass(frozen=True)
class ShardReference:
    path: str
    sha256: str
    records: int
    target_values: int


@dataclass(frozen=True)
class CovariateDefinition:
    id: str
    unit: str
    access: str


@dataclass(frozen=True)
class ReadoutDefinition:
    id: str
    likelihood: str
    unit: str
    implicit_zero: bool

    @property
    def likelihood_index(self) -> int:
        return LIKELIHOODS.index(self.likelihood)


@dataclass(frozen=True)
class SparseShard:
    reference: ShardReference
    arrays: dict[str, np.ndarray]

    @property
    def records(self) -> int:
        return self.reference.records


@dataclass(frozen=True)
class BatchProvenance:
    record_id: tuple[str, ...]
    observation_unit_id: tuple[str, ...]
    source_index: torch.Tensor
    replicate_id: tuple[str, ...]
    perturbation_id: tuple[str, ...]
    species_taxon: torch.Tensor


@dataclass(frozen=True)
class LikelihoodCovariates:
    record_value: torch.Tensor
    record_present: torch.Tensor
    context_value: torch.Tensor
    context_present: torch.Tensor
    action_value: torch.Tensor
    action_present: torch.Tensor
    observation_value: torch.Tensor
    observation_present: torch.Tensor


@dataclass(frozen=True)
class MaterializedBatch:
    world: WorldBatch
    target_value: torch.Tensor
    target_observed: torch.Tensor
    likelihood_covariates: LikelihoodCovariates
    provenance: BatchProvenance


@dataclass(frozen=True)
class RecordLocation:
    shard_index: int
    row_index: int


@dataclass(frozen=True)
class CorpusIndex:
    root: Path
    dataset_id: str
    version: str
    role: str
    feature_pack_revision: str
    feature_pack_sha256: str
    normalization_id: str
    value_space: str
    content_digest: str
    sources: tuple[str, ...]
    source_weights: tuple[float, ...]
    species_taxa: tuple[int, ...]
    species_feature_value: dict[int, tuple[float, ...]]
    species_feature_present: dict[int, tuple[bool, ...]]
    entity_types: tuple[str, ...]
    context_types: tuple[str, ...]
    action_types: tuple[str, ...]
    covariates: dict[str, tuple[CovariateDefinition, ...]]
    readouts: tuple[ReadoutDefinition, ...]
    entity_feature_dim: int
    species_feature_dim: int
    entity_id: np.ndarray
    entity_type: np.ndarray
    entity_species_taxon: np.ndarray
    entity_feature_value: np.ndarray
    entity_feature_present: np.ndarray
    query_id: np.ndarray
    query_entity_index: np.ndarray
    query_readout_index: np.ndarray
    panel_id: np.ndarray
    panel_indptr: np.ndarray
    panel_query_index: np.ndarray
    trajectory_genes: frozenset[str]
    bounds: dict[str, int]
    shards: tuple[ShardReference, ...]

    @classmethod
    def load(cls, root: str | Path) -> "CorpusIndex":
        requested_root = Path(root).absolute()
        _reject_symlink_components(requested_root)
        root = requested_root.resolve()
        manifest_path = root / "corpus.json"
        if not manifest_path.is_file():
            raise ValueError("snapshot must contain corpus.json")
        _reject_symlink_components(manifest_path)
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ValueError("corpus.json exceeds the 4 MiB manifest bound")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {
            "schema",
            "datasetId",
            "version",
            "role",
            "labelClass",
            "benchmarkLabelsPresent",
            "rights",
            "modalities",
            "sources",
            "sampling",
            "species",
            "featurePack",
            "entityTypes",
            "contextTypes",
            "actionTypes",
            "covariates",
            "readoutTypes",
            "entityDictionary",
            "queryDictionary",
            "queryPanels",
            "trajectoryGenes",
            "normalization",
            "bounds",
            "shards",
        }
        _expect_keys(manifest, required, "corpus manifest")
        if manifest["schema"] != CORPUS_SCHEMA:
            raise ValueError(f"expected {CORPUS_SCHEMA}")
        dataset_id = _require_curie(manifest["datasetId"], "datasetId")
        version = _require_nonempty(manifest["version"], "version")
        role = manifest["role"]
        if role not in ROLES:
            raise ValueError("invalid corpus role")
        if manifest["labelClass"] != "molecular":
            raise ValueError("world corpora must contain molecular labels only")
        if manifest["benchmarkLabelsPresent"] is not False:
            raise ValueError("benchmark labels are forbidden")
        _parse_rights(manifest["rights"])
        _require_unique_curies(manifest["modalities"], "modalities")
        sources = _parse_sources(manifest["sources"])
        source_weights = _parse_sampling(manifest["sampling"], len(sources))
        feature_pack = _parse_feature_pack(manifest["featurePack"])
        entity_feature_dim = feature_pack[0]
        species_feature_dim = feature_pack[1]
        feature_pack_revision = feature_pack[2]
        feature_pack_sha256 = feature_pack[3]
        species_taxa, species_value, species_present = _parse_species(
            manifest["species"], species_feature_dim
        )
        entity_types = _require_unique_curies(manifest["entityTypes"], "entityTypes")
        context_types = _require_unique_curies(manifest["contextTypes"], "contextTypes")
        action_types = _require_unique_curies(manifest["actionTypes"], "actionTypes")
        covariates = _parse_covariates(manifest["covariates"])
        readouts = _parse_readouts(manifest["readoutTypes"])
        normalization_id, value_space = _parse_normalization(manifest["normalization"])
        bounds = _parse_bounds(manifest["bounds"])

        entity_ref = _parse_file_ref(manifest["entityDictionary"], "entityDictionary")
        query_ref = _parse_file_ref(manifest["queryDictionary"], "queryDictionary")
        panel_ref = _parse_file_ref(manifest["queryPanels"], "queryPanels")
        genes_ref = _parse_file_ref(
            manifest["trajectoryGenes"], "trajectoryGenes", allow_empty=True
        )
        entity_arrays = _load_npz(
            root,
            entity_ref,
            {
                "entity_id",
                "entity_type",
                "entity_species_taxon",
                "entity_feature_value",
                "entity_feature_present",
            },
        )
        (
            entity_id,
            entity_type,
            entity_species_taxon,
            entity_feature_value,
            entity_feature_present,
        ) = _validate_entity_dictionary(
            entity_arrays,
            entity_ref.count,
            entity_feature_dim,
            len(entity_types),
            set(species_taxa),
        )
        query_arrays = _load_npz(
            root,
            query_ref,
            {"query_id", "query_entity_index", "query_readout_index"},
        )
        query_id, query_entity_index, query_readout_index = _validate_query_dictionary(
            query_arrays,
            query_ref.count,
            entity_ref.count,
            len(readouts),
        )
        panel_arrays = _load_npz(
            root,
            panel_ref,
            {"panel_id", "panel_indptr", "panel_query_index"},
        )
        panel_id, panel_indptr, panel_query_index = _validate_query_panels(
            panel_arrays,
            panel_ref.count,
            query_ref.count,
            bounds["maxPanelQueries"],
        )
        gene_path = _resolve_file(root, genes_ref.path)
        _verify_digest(gene_path, genes_ref.sha256)
        gene_lines = [
            line.strip()
            for line in gene_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if len(gene_lines) != genes_ref.count or gene_lines != sorted(set(gene_lines)):
            raise ValueError("trajectoryGenes must be a sorted unique counted CURIE list")
        trajectory_genes = frozenset(
            _require_curie(item, "trajectory gene") for item in gene_lines
        )
        if not trajectory_genes.issubset(set(str(item) for item in entity_id)):
            raise ValueError("trajectoryGenes must resolve through the entity dictionary")
        shards = _parse_shards(manifest["shards"], root, bounds)
        identity_document = {
            "manifestSha256": _sha256_path(manifest_path),
            "files": [
                {"path": reference.path, "sha256": reference.sha256}
                for reference in (entity_ref, query_ref, panel_ref, genes_ref, *shards)
            ],
        }
        content_digest = hashlib.sha256(
            json.dumps(
                identity_document, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

        result = cls(
            root=root,
            dataset_id=dataset_id,
            version=version,
            role=role,
            feature_pack_revision=feature_pack_revision,
            feature_pack_sha256=feature_pack_sha256,
            normalization_id=normalization_id,
            value_space=value_space,
            content_digest=content_digest,
            sources=sources,
            source_weights=source_weights,
            species_taxa=species_taxa,
            species_feature_value=species_value,
            species_feature_present=species_present,
            entity_types=entity_types,
            context_types=context_types,
            action_types=action_types,
            covariates=covariates,
            readouts=readouts,
            entity_feature_dim=entity_feature_dim,
            species_feature_dim=species_feature_dim,
            entity_id=entity_id,
            entity_type=entity_type,
            entity_species_taxon=entity_species_taxon,
            entity_feature_value=entity_feature_value,
            entity_feature_present=entity_feature_present,
            query_id=query_id,
            query_entity_index=query_entity_index,
            query_readout_index=query_readout_index,
            panel_id=panel_id,
            panel_indptr=panel_indptr,
            panel_query_index=panel_query_index,
            trajectory_genes=trajectory_genes,
            bounds=bounds,
            shards=shards,
        )
        seen_records: set[str] = set()
        seen_sources: set[int] = set()
        active_species_actions: set[str] = set()
        for shard_index in range(len(shards)):
            shard = result.load_shard(shard_index)
            record_ids = {str(item) for item in shard.arrays["record_id"]}
            if seen_records & record_ids:
                raise ValueError("record_id values must be unique across shards")
            seen_records.update(record_ids)
            seen_sources.update(int(item) for item in shard.arrays["source_index"])
            action_references = shard.arrays["action_entity_index"][
                shard.arrays["action_mask"]
            ]
            for entity_index in action_references:
                if int(result.entity_species_taxon[int(entity_index)]) != 0:
                    active_species_actions.add(str(result.entity_id[int(entity_index)]))
        if seen_sources != set(range(len(sources))):
            raise ValueError("every declared source must have at least one record")
        if active_species_actions != set(trajectory_genes):
            missing = sorted(active_species_actions - set(trajectory_genes))
            extra = sorted(set(trajectory_genes) - active_species_actions)
            raise ValueError(
                "trajectoryGenes must exactly match active species-specific actions; "
                f"missing={missing}, extra={extra}"
            )
        return result

    def load_shard(self, shard_index: int) -> SparseShard:
        try:
            reference = self.shards[shard_index]
        except IndexError as error:
            raise ValueError("shard index is out of range") from error
        path = _resolve_file(self.root, reference.path)
        _verify_digest(path, reference.sha256)
        with np.load(path, allow_pickle=False) as source:
            required = _required_shard_arrays()
            if set(source.files) != required:
                missing = sorted(required - set(source.files))
                extra = sorted(set(source.files) - required)
                raise ValueError(f"shard arrays mismatch; missing={missing}, extra={extra}")
            arrays = {name: source[name] for name in required}
        self._validate_shard_arrays(arrays, reference)
        return SparseShard(reference=reference, arrays=arrays)

    def world_config(self, **overrides: Any) -> WorldConfig:
        config: dict[str, Any] = {
            "entity_feature_dim": self.entity_feature_dim,
            "species_feature_dim": self.species_feature_dim,
            "entity_types": len(self.entity_types),
            "context_types": len(self.context_types),
            "action_types": len(self.action_types),
            "readout_types": len(self.readouts),
            "record_covariate_dim": len(self._covariate_indices("record", "world")),
            "context_covariate_dim": len(self._covariate_indices("context", "world")),
            "action_covariate_dim": len(self._covariate_indices("action", "world")),
            "observation_covariate_dim": len(
                self._covariate_indices("observation", "world")
            ),
        }
        config.update(overrides)
        return WorldConfig(**config)

    def materialize_batch(
        self, shard: SparseShard, rows: Sequence[int]
    ) -> MaterializedBatch:
        row_index = np.asarray(rows, dtype=np.int64)
        if row_index.ndim != 1 or not row_index.size:
            raise ValueError("rows must be a non-empty one-dimensional selection")
        if row_index.min() < 0 or row_index.max() >= shard.records:
            raise ValueError("selected row is outside the shard")
        arrays = shard.arrays
        batch_size = len(row_index)
        context_ref = arrays["context_entity_index"][row_index]
        action_ref = arrays["action_entity_index"][row_index]
        context_mask = arrays["context_mask"][row_index]
        action_mask = arrays["action_mask"][row_index]
        context_value, context_present, context_entity_type = self._entity_features(
            context_ref, context_mask
        )
        action_value, action_present, action_entity_type = self._entity_features(
            action_ref, action_mask
        )

        selected_panels = arrays["query_panel_index"][row_index]
        panel_queries = [self._panel_queries(int(panel)) for panel in selected_panels]
        query_count = max(len(items) for items in panel_queries)
        query_value = np.zeros(
            (batch_size, query_count, self.entity_feature_dim), dtype=np.float32
        )
        query_present = np.zeros_like(query_value, dtype=np.bool_)
        query_entity_type = np.full((batch_size, query_count), -1, dtype=np.int64)
        readout_type = np.full((batch_size, query_count), -1, dtype=np.int64)
        likelihood_type = np.full((batch_size, query_count), -1, dtype=np.int64)
        query_mask = np.zeros((batch_size, query_count), dtype=np.bool_)
        target_value = np.zeros((batch_size, query_count), dtype=np.float32)
        target_observed = np.zeros((batch_size, query_count), dtype=np.bool_)
        target_indptr = arrays["target_indptr"]
        target_query_index = arrays["target_query_index"]
        sparse_target_value = arrays["target_value"]
        for batch_row, (source_row, queries) in enumerate(zip(row_index, panel_queries)):
            width = len(queries)
            query_mask[batch_row, :width] = True
            query_entities = self.query_entity_index[queries]
            query_value[batch_row, :width] = self.entity_feature_value[query_entities]
            query_present[batch_row, :width] = self.entity_feature_present[query_entities]
            query_entity_type[batch_row, :width] = self.entity_type[query_entities]
            query_readouts = self.query_readout_index[queries]
            readout_type[batch_row, :width] = query_readouts
            likelihood_type[batch_row, :width] = np.asarray(
                [self.readouts[int(item)].likelihood_index for item in query_readouts],
                dtype=np.int64,
            )
            target_observed[batch_row, :width] = np.asarray(
                [self.readouts[int(item)].implicit_zero for item in query_readouts],
                dtype=np.bool_,
            )
            local_index = {int(query): position for position, query in enumerate(queries)}
            start = int(target_indptr[source_row])
            stop = int(target_indptr[source_row + 1])
            for offset in range(start, stop):
                position = local_index[int(target_query_index[offset])]
                target_value[batch_row, position] = sparse_target_value[offset]
                target_observed[batch_row, position] = True

        world = WorldBatch(
            context_features=_float_tensor(context_value),
            context_feature_present=_bool_tensor(context_present),
            context_entity_type=_long_tensor(context_entity_type),
            context_type=_long_tensor(arrays["context_type"][row_index]),
            context_covariates=_float_tensor(
                self._select_covariates(
                    arrays["context_covariate_value"][row_index], "context", "world"
                )
            ),
            context_covariate_present=_bool_tensor(
                self._select_covariates(
                    arrays["context_covariate_present"][row_index], "context", "world"
                )
            ),
            context_mask=_bool_tensor(context_mask),
            action_features=_float_tensor(action_value),
            action_feature_present=_bool_tensor(action_present),
            action_entity_type=_long_tensor(action_entity_type),
            action_type=_long_tensor(arrays["action_type"][row_index]),
            action_covariates=_float_tensor(
                self._select_covariates(
                    arrays["action_covariate_value"][row_index], "action", "world"
                )
            ),
            action_covariate_present=_bool_tensor(
                self._select_covariates(
                    arrays["action_covariate_present"][row_index], "action", "world"
                )
            ),
            action_mask=_bool_tensor(action_mask),
            query_features=_float_tensor(query_value),
            query_feature_present=_bool_tensor(query_present),
            query_entity_type=_long_tensor(query_entity_type),
            readout_type=_long_tensor(readout_type),
            likelihood_type=_long_tensor(likelihood_type),
            query_mask=_bool_tensor(query_mask),
            species_features=_float_tensor(arrays["species_feature_value"][row_index]),
            species_feature_present=_bool_tensor(
                arrays["species_feature_present"][row_index]
            ),
            record_covariates=_float_tensor(
                self._select_covariates(
                    arrays["record_covariate_value"][row_index], "record", "world"
                )
            ),
            record_covariate_present=_bool_tensor(
                self._select_covariates(
                    arrays["record_covariate_present"][row_index], "record", "world"
                )
            ),
            observation_covariates=_float_tensor(
                self._select_covariates(
                    arrays["observation_covariate_value"][row_index],
                    "observation",
                    "world",
                )
            ),
            observation_covariate_present=_bool_tensor(
                self._select_covariates(
                    arrays["observation_covariate_present"][row_index],
                    "observation",
                    "world",
                )
            ),
        )
        likelihood_covariates = LikelihoodCovariates(
            record_value=_float_tensor(
                self._select_covariates(
                    arrays["record_covariate_value"][row_index], "record", "likelihood"
                )
            ),
            record_present=_bool_tensor(
                self._select_covariates(
                    arrays["record_covariate_present"][row_index],
                    "record",
                    "likelihood",
                )
            ),
            context_value=_float_tensor(
                self._select_covariates(
                    arrays["context_covariate_value"][row_index],
                    "context",
                    "likelihood",
                )
            ),
            context_present=_bool_tensor(
                self._select_covariates(
                    arrays["context_covariate_present"][row_index],
                    "context",
                    "likelihood",
                )
            ),
            action_value=_float_tensor(
                self._select_covariates(
                    arrays["action_covariate_value"][row_index], "action", "likelihood"
                )
            ),
            action_present=_bool_tensor(
                self._select_covariates(
                    arrays["action_covariate_present"][row_index],
                    "action",
                    "likelihood",
                )
            ),
            observation_value=_float_tensor(
                self._select_covariates(
                    arrays["observation_covariate_value"][row_index],
                    "observation",
                    "likelihood",
                )
            ),
            observation_present=_bool_tensor(
                self._select_covariates(
                    arrays["observation_covariate_present"][row_index],
                    "observation",
                    "likelihood",
                )
            ),
        )
        provenance = BatchProvenance(
            record_id=tuple(str(item) for item in arrays["record_id"][row_index]),
            observation_unit_id=tuple(
                str(item) for item in arrays["observation_unit_id"][row_index]
            ),
            source_index=_long_tensor(arrays["source_index"][row_index]),
            replicate_id=tuple(str(item) for item in arrays["replicate_id"][row_index]),
            perturbation_id=tuple(
                str(item) for item in arrays["perturbation_id"][row_index]
            ),
            species_taxon=_long_tensor(arrays["species_taxon"][row_index]),
        )
        return MaterializedBatch(
            world=world,
            target_value=_float_tensor(target_value),
            target_observed=_bool_tensor(target_observed),
            likelihood_covariates=likelihood_covariates,
            provenance=provenance,
        )

    def record_sampler(self, seed: int) -> "DeterministicHierarchicalSampler":
        locations: list[RecordLocation] = []
        source_index: list[int] = []
        perturbation_id: list[str] = []
        replicate_id: list[str] = []
        for shard_index in range(len(self.shards)):
            arrays = self.load_shard(shard_index).arrays
            for row_index in range(len(arrays["record_id"])):
                locations.append(RecordLocation(shard_index, row_index))
                source_index.append(int(arrays["source_index"][row_index]))
                perturbation_id.append(str(arrays["perturbation_id"][row_index]))
                replicate_id.append(str(arrays["replicate_id"][row_index]))
        return DeterministicHierarchicalSampler(
            source_index=source_index,
            perturbation_id=perturbation_id,
            replicate_id=replicate_id,
            source_weights=self.source_weights,
            seed=seed,
            locations=locations,
        )

    def _panel_queries(self, panel_index: int) -> np.ndarray:
        start = int(self.panel_indptr[panel_index])
        stop = int(self.panel_indptr[panel_index + 1])
        return self.panel_query_index[start:stop]

    def _entity_features(
        self, reference: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        safe_reference = np.where(mask, reference, 0)
        value = self.entity_feature_value[safe_reference].copy()
        present = self.entity_feature_present[safe_reference].copy()
        entity_type = self.entity_type[safe_reference].copy()
        value[~mask] = 0
        present[~mask] = False
        entity_type[~mask] = -1
        return value, present, entity_type

    def _covariate_indices(self, axis: str, access: str) -> tuple[int, ...]:
        return tuple(
            index
            for index, definition in enumerate(self.covariates[axis])
            if definition.access == access
        )

    def _select_covariates(
        self, values: np.ndarray, axis: str, access: str
    ) -> np.ndarray:
        return values[..., self._covariate_indices(axis, access)]

    def _validate_shard_arrays(
        self, arrays: dict[str, np.ndarray], reference: ShardReference
    ) -> None:
        records = reference.records
        for name in (
            "record_id",
            "observation_unit_id",
            "source_index",
            "replicate_id",
            "perturbation_id",
            "species_taxon",
            "species_feature_value",
            "species_feature_present",
            "context_entity_index",
            "context_type",
            "context_mask",
            "context_covariate_value",
            "context_covariate_present",
            "record_covariate_value",
            "record_covariate_present",
            "action_entity_index",
            "action_type",
            "action_mask",
            "action_covariate_value",
            "action_covariate_present",
            "observation_covariate_value",
            "observation_covariate_present",
            "query_panel_index",
        ):
            if arrays[name].shape[0] != records:
                raise ValueError(f"{name} record count does not match manifest")
        _validate_curie_array(arrays["record_id"], records, "record_id", unique=True)
        _validate_curie_array(
            arrays["observation_unit_id"], records, "observation_unit_id"
        )
        _validate_curie_array(arrays["replicate_id"], records, "replicate_id")
        _validate_curie_array(arrays["perturbation_id"], records, "perturbation_id")
        _require_int64(arrays["source_index"], (records,), "source_index")
        _validate_index(arrays["source_index"], len(self.sources), "source_index")
        _require_int64(arrays["species_taxon"], (records,), "species_taxon")
        if not set(int(item) for item in arrays["species_taxon"]).issubset(
            set(self.species_taxa)
        ):
            raise ValueError("species_taxon contains an undeclared taxon")
        _require_float32(
            arrays["species_feature_value"],
            (records, self.species_feature_dim),
            "species_feature_value",
        )
        _require_bool(
            arrays["species_feature_present"],
            (records, self.species_feature_dim),
            "species_feature_present",
        )
        _validate_masked_storage(
            arrays["species_feature_value"],
            arrays["species_feature_present"],
            "species features",
        )
        for row, taxon in enumerate(arrays["species_taxon"]):
            expected_value = np.asarray(self.species_feature_value[int(taxon)], np.float32)
            expected_present = np.asarray(self.species_feature_present[int(taxon)], np.bool_)
            if not np.array_equal(arrays["species_feature_value"][row], expected_value):
                raise ValueError("species feature value does not match declared taxon")
            if not np.array_equal(
                arrays["species_feature_present"][row], expected_present
            ):
                raise ValueError("species feature presence does not match declared taxon")

        context_shape = arrays["context_entity_index"].shape
        action_shape = arrays["action_entity_index"].shape
        if len(context_shape) != 2 or context_shape[1] > self.bounds["maxContextTokens"]:
            raise ValueError("context token bound exceeded")
        if len(action_shape) != 2 or action_shape[1] > self.bounds["maxActionTokens"]:
            raise ValueError("action token bound exceeded")
        _validate_token_references(
            arrays,
            "context",
            context_shape,
            len(self.entity_id),
            len(self.context_types),
            len(self.covariates["context"]),
        )
        _validate_token_references(
            arrays,
            "action",
            action_shape,
            len(self.entity_id),
            len(self.action_types),
            len(self.covariates["action"]),
        )
        if not (
            arrays["context_mask"].any(axis=1) | arrays["action_mask"].any(axis=1)
        ).all():
            raise ValueError("every record requires context or action memory")
        for axis in ("record", "observation"):
            value_name = f"{axis}_covariate_value"
            present_name = f"{axis}_covariate_present"
            shape = (records, len(self.covariates[axis]))
            _require_float32(arrays[value_name], shape, value_name)
            _require_bool(arrays[present_name], shape, present_name)
            _validate_masked_storage(arrays[value_name], arrays[present_name], axis)

        _require_int64(arrays["query_panel_index"], (records,), "query_panel_index")
        _validate_index(arrays["query_panel_index"], len(self.panel_id), "query_panel_index")
        _require_int64(arrays["target_indptr"], (records + 1,), "target_indptr")
        _require_int64(
            arrays["target_query_index"],
            (reference.target_values,),
            "target_query_index",
        )
        _require_float32(
            arrays["target_value"], (reference.target_values,), "target_value"
        )
        _validate_csr(arrays["target_indptr"], reference.target_values, "target")
        _validate_index(
            arrays["target_query_index"], len(self.query_id), "target_query_index"
        )
        if not np.isfinite(arrays["target_value"]).all():
            raise ValueError("target values must be finite")
        for row in range(records):
            panel_queries = self._panel_queries(int(arrays["query_panel_index"][row]))
            if not len(panel_queries):
                raise ValueError("every record requires a non-empty query panel")
            taxon = int(arrays["species_taxon"][row])
            for axis in ("context", "action"):
                active_references = arrays[f"{axis}_entity_index"][row][
                    arrays[f"{axis}_mask"][row]
                ]
                referenced_taxa = self.entity_species_taxon[active_references]
                if any(int(item) not in (0, taxon) for item in referenced_taxa):
                    raise ValueError(f"{axis} entity taxon does not match the record")
            query_entities = self.query_entity_index[panel_queries]
            query_taxa = self.entity_species_taxon[query_entities]
            if any(int(item) not in (0, taxon) for item in query_taxa):
                raise ValueError("query entity taxon does not match the record")
            start = int(arrays["target_indptr"][row])
            stop = int(arrays["target_indptr"][row + 1])
            sparse_queries = arrays["target_query_index"][start:stop]
            if len(sparse_queries) != len(set(int(item) for item in sparse_queries)):
                raise ValueError("a target query can occur only once per record")
            if not set(int(item) for item in sparse_queries).issubset(
                set(int(item) for item in panel_queries)
            ):
                raise ValueError("target query must belong to the record query panel")
            observed = set(int(item) for item in sparse_queries)
            observed.update(
                int(query)
                for query in panel_queries
                if self.readouts[int(self.query_readout_index[int(query)])].implicit_zero
            )
            if not observed:
                raise ValueError("every record requires at least one observed target")
            for offset in range(start, stop):
                query = int(arrays["target_query_index"][offset])
                definition = self.readouts[int(self.query_readout_index[query])]
                value = float(arrays["target_value"][offset])
                if definition.likelihood == "negative-binomial" and (
                    value < 0 or value != float(round(value))
                ):
                    raise ValueError("negative-binomial targets must be non-negative counts")


class DeterministicHierarchicalSampler:
    """Explicit source weights, then uniform perturbation/replicate/record draws."""

    def __init__(
        self,
        source_index: Sequence[int],
        perturbation_id: Sequence[str],
        replicate_id: Sequence[str],
        source_weights: Sequence[float],
        seed: int,
        locations: Sequence[RecordLocation] | None = None,
    ) -> None:
        count = len(source_index)
        if not count or len(perturbation_id) != count or len(replicate_id) != count:
            raise ValueError("sampler provenance arrays must be non-empty and aligned")
        if not source_weights or any(
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not np.isfinite(weight)
            or weight <= 0
            for weight in source_weights
        ):
            raise ValueError("every source requires an explicit positive finite weight")
        source_count = len(source_weights)
        if set(int(item) for item in source_index) != set(range(source_count)):
            raise ValueError("sampler requires records from every weighted source")
        self.source_index = tuple(int(item) for item in source_index)
        self.perturbation_id = tuple(str(item) for item in perturbation_id)
        self.replicate_id = tuple(str(item) for item in replicate_id)
        self.source_weights = tuple(float(item) for item in source_weights)
        self.seed = int(seed)
        self.locations = tuple(locations) if locations is not None else tuple(range(count))
        if len(self.locations) != count:
            raise ValueError("sampler locations must align with provenance")

    def schedule(self, draws: int) -> tuple[Any, ...]:
        if not isinstance(draws, int) or isinstance(draws, bool) or draws <= 0:
            raise ValueError("draws must be a positive integer")
        random_source = random.Random(self.seed)
        quotas = _weighted_quotas(self.source_weights, draws)
        source_slots: list[int] = []
        for source, quota in enumerate(quotas):
            source_slots.extend([source] * quota)
        random_source.shuffle(source_slots)

        groups: dict[int, dict[str, dict[str, list[int]]]] = {}
        for position, source in enumerate(self.source_index):
            groups.setdefault(source, {}).setdefault(
                self.perturbation_id[position], {}
            ).setdefault(self.replicate_id[position], []).append(position)
        states: dict[int, _HierarchyState] = {
            source: _HierarchyState(groups[source], random.Random(self.seed + 104729 * (source + 1)))
            for source in groups
        }
        return tuple(self.locations[states[source].next()] for source in source_slots)


class _HierarchyState:
    def __init__(self, groups: dict[str, dict[str, list[int]]], rng: random.Random):
        self.perturbations = sorted(groups)
        rng.shuffle(self.perturbations)
        self.replicates: dict[str, list[str]] = {}
        self.records: dict[tuple[str, str], list[int]] = {}
        for perturbation in self.perturbations:
            replicates = sorted(groups[perturbation])
            rng.shuffle(replicates)
            self.replicates[perturbation] = replicates
            for replicate in replicates:
                records = sorted(groups[perturbation][replicate])
                rng.shuffle(records)
                self.records[(perturbation, replicate)] = records
        self.perturbation_cursor = 0
        self.replicate_cursor = {item: 0 for item in self.perturbations}
        self.record_cursor = {item: 0 for item in self.records}

    def next(self) -> int:
        perturbation = self.perturbations[
            self.perturbation_cursor % len(self.perturbations)
        ]
        self.perturbation_cursor += 1
        replicates = self.replicates[perturbation]
        replicate_cursor = self.replicate_cursor[perturbation]
        replicate = replicates[replicate_cursor % len(replicates)]
        self.replicate_cursor[perturbation] += 1
        key = (perturbation, replicate)
        records = self.records[key]
        record_cursor = self.record_cursor[key]
        self.record_cursor[key] += 1
        return records[record_cursor % len(records)]


def pinned_dataset_path(value: object, input_name: str = "corpus") -> str:
    """Resolve OMF's exact copied, revision-pinned DatasetSnapshot input shape."""

    required = {"manifestDigest", "mode", "path", "resource"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(
            f"{input_name} must be an exact materialized OMF DatasetSnapshot input"
        )
    if value["mode"] != "copy":
        raise ValueError(f"{input_name} must be an immutable copied DatasetSnapshot")
    manifest_digest = value["manifestDigest"]
    if (
        not isinstance(manifest_digest, str)
        or not manifest_digest.startswith("sha256:")
        or SHA256.fullmatch(manifest_digest.removeprefix("sha256:")) is None
    ):
        raise ValueError(f"{input_name}.manifestDigest must be admission-pinned")
    resource = value["resource"]
    if not isinstance(resource, str) or not resource.startswith("omf://"):
        raise ValueError(f"{input_name}.resource must be an OMF DatasetSnapshot URI")
    identity, separator, revision = resource.removeprefix("omf://").rpartition("@")
    if (
        not separator
        or not revision.startswith("sha256:")
        or SHA256.fullmatch(revision.removeprefix("sha256:")) is None
    ):
        raise ValueError(f"{input_name}.resource must have an admission-pinned revision")
    parts = identity.split("/")
    if (
        len(parts) < 3
        or parts[-2] != "datasetsnapshot"
        or RESOURCE_NAME.fullmatch(parts[-1]) is None
        or any(not part or part in {".", ".."} or any(char.isspace() for char in part) for part in parts)
    ):
        raise ValueError(f"{input_name}.resource kind must be DatasetSnapshot")
    resource_name = parts[-1]
    path_value = value["path"]
    if not isinstance(path_value, str) or not path_value or path_value != path_value.strip():
        raise ValueError(f"{input_name}.path must be a non-empty trimmed string")
    requested = Path(path_value).absolute()
    _reject_symlink_components(requested)
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{input_name}.path does not exist") from error
    if not resolved.is_dir():
        raise ValueError(f"{input_name}.path must materialize a corpus directory")
    if (
        resolved.name != resource_name
        or resolved.parent.name != input_name
        or resolved.parent.parent.name != "inputs"
    ):
        raise ValueError(
            f"{input_name}.path is inconsistent with its input name and DatasetSnapshot resource"
        )
    return str(resolved)


def _weighted_quotas(weights: Sequence[float], draws: int) -> tuple[int, ...]:
    total = float(sum(weights))
    raw = [draws * float(weight) / total for weight in weights]
    quotas = [int(np.floor(item)) for item in raw]
    remaining = draws - sum(quotas)
    order = sorted(range(len(weights)), key=lambda item: (-(raw[item] - quotas[item]), item))
    for index in order[:remaining]:
        quotas[index] += 1
    return tuple(quotas)


def _required_shard_arrays() -> set[str]:
    return {
        "record_id",
        "observation_unit_id",
        "source_index",
        "replicate_id",
        "perturbation_id",
        "species_taxon",
        "species_feature_value",
        "species_feature_present",
        "context_entity_index",
        "context_type",
        "context_mask",
        "context_covariate_value",
        "context_covariate_present",
        "record_covariate_value",
        "record_covariate_present",
        "action_entity_index",
        "action_type",
        "action_mask",
        "action_covariate_value",
        "action_covariate_present",
        "observation_covariate_value",
        "observation_covariate_present",
        "query_panel_index",
        "target_indptr",
        "target_query_index",
        "target_value",
    }


def _expect_keys(value: Any, expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{name} fields mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_curie(value: Any, name: str) -> str:
    value = _require_nonempty(value, name)
    if not CURIE.fullmatch(value):
        raise ValueError(f"{name} must be a namespace-bearing CURIE")
    return value


def _require_unique_curies(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty CURIE list")
    items = tuple(_require_curie(item, name) for item in value)
    if len(items) != len(set(items)):
        raise ValueError(f"{name} must be unique")
    return items


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _parse_rights(value: Any) -> None:
    _expect_keys(
        value,
        {"revision", "trainingAllowed", "redistributionAllowed"},
        "rights",
    )
    _require_curie(value["revision"], "rights revision")
    if value["trainingAllowed"] is not True:
        raise ValueError("trainingAllowed must be true")
    if not isinstance(value["redistributionAllowed"], bool):
        raise ValueError("redistributionAllowed must be boolean")


def _parse_sources(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("sources must be a non-empty list")
    sources: list[str] = []
    for item in value:
        _expect_keys(item, {"id"}, "source")
        sources.append(_require_curie(item["id"], "source id"))
    if len(sources) != len(set(sources)):
        raise ValueError("source ids must be unique")
    return tuple(sources)


def _parse_sampling(value: Any, source_count: int) -> tuple[float, ...]:
    _expect_keys(value, {"scheme", "sourceWeights"}, "sampling")
    if value["scheme"] != SAMPLING_SCHEME:
        raise ValueError(f"sampling scheme must be {SAMPLING_SCHEME}")
    weights = value["sourceWeights"]
    if not isinstance(weights, list) or len(weights) != source_count:
        raise ValueError("sourceWeights must align exactly with sources")
    parsed: list[float] = []
    for weight in weights:
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not np.isfinite(weight)
            or weight <= 0
        ):
            raise ValueError("source weights must be positive finite numbers")
        parsed.append(float(weight))
    return tuple(parsed)


def _parse_feature_pack(value: Any) -> tuple[int, int, str, str]:
    _expect_keys(
        value,
        {"revision", "sha256", "entityFeatureDim", "speciesFeatureDim"},
        "featurePack",
    )
    revision = _require_curie(value["revision"], "feature pack revision")
    digest = _require_sha(value["sha256"], "feature pack sha256")
    dims = (value["entityFeatureDim"], value["speciesFeatureDim"])
    if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in dims):
        raise ValueError("feature dimensions must be positive integers")
    return dims[0], dims[1], revision, digest


def _parse_species(
    value: Any, feature_dim: int
) -> tuple[
    tuple[int, ...], dict[int, tuple[float, ...]], dict[int, tuple[bool, ...]]
]:
    if not isinstance(value, list) or not value:
        raise ValueError("species must be a non-empty list")
    taxa: list[int] = []
    values: dict[int, tuple[float, ...]] = {}
    present: dict[int, tuple[bool, ...]] = {}
    for item in value:
        _expect_keys(item, {"taxon", "featureValue", "featurePresent"}, "species")
        taxon = item["taxon"]
        if not isinstance(taxon, int) or isinstance(taxon, bool) or taxon <= 0:
            raise ValueError("species taxon must be a positive NCBI taxonomy id")
        feature_value = np.asarray(item["featureValue"])
        feature_present = np.asarray(item["featurePresent"])
        if feature_value.shape != (feature_dim,) or feature_present.shape != (feature_dim,):
            raise ValueError("species features must match speciesFeatureDim")
        if feature_present.dtype != np.bool_:
            raise ValueError("species feature presence must be boolean")
        if not np.issubdtype(feature_value.dtype, np.number):
            raise ValueError("species feature values must be numeric")
        feature_value = feature_value.astype(np.float32)
        _validate_masked_storage(feature_value, feature_present, "species declaration")
        taxa.append(taxon)
        values[taxon] = tuple(float(entry) for entry in feature_value)
        present[taxon] = tuple(bool(entry) for entry in feature_present)
    if len(taxa) != len(set(taxa)):
        raise ValueError("species taxa must be unique")
    return tuple(taxa), values, present


def _parse_covariates(value: Any) -> dict[str, tuple[CovariateDefinition, ...]]:
    _expect_keys(value, {"record", "context", "action", "observation"}, "covariates")
    result: dict[str, tuple[CovariateDefinition, ...]] = {}
    all_ids: list[str] = []
    for axis in ("record", "context", "action", "observation"):
        definitions = value[axis]
        if not isinstance(definitions, list):
            raise ValueError(f"{axis} covariates must be a list")
        parsed: list[CovariateDefinition] = []
        for item in definitions:
            _expect_keys(item, {"id", "unit", "access"}, f"{axis} covariate")
            identifier = _require_curie(item["id"], "covariate id")
            unit = _require_curie(item["unit"], "covariate unit")
            access = item["access"]
            if access not in ACCESS_ROLES:
                raise ValueError("covariate access must be world, likelihood, or audit")
            parsed.append(CovariateDefinition(identifier, unit, access))
            all_ids.append(identifier)
        result[axis] = tuple(parsed)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("covariate ids must be globally unique")
    return result


def _parse_readouts(value: Any) -> tuple[ReadoutDefinition, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("readoutTypes must be a non-empty list")
    result: list[ReadoutDefinition] = []
    for item in value:
        _expect_keys(item, {"id", "likelihood", "unit", "implicitZero"}, "readout")
        identifier = _require_curie(item["id"], "readout id")
        unit = _require_curie(item["unit"], "readout unit")
        likelihood = item["likelihood"]
        if likelihood not in LIKELIHOODS:
            raise ValueError("readout likelihood must be gaussian or negative-binomial")
        implicit_zero = item["implicitZero"]
        if not isinstance(implicit_zero, bool):
            raise ValueError("implicitZero must be boolean")
        if implicit_zero and likelihood != "negative-binomial":
            raise ValueError("implicitZero is permitted only for count readouts")
        result.append(ReadoutDefinition(identifier, likelihood, unit, implicit_zero))
    ids = [item.id for item in result]
    if len(ids) != len(set(ids)):
        raise ValueError("readout ids must be unique")
    return tuple(result)


def _parse_normalization(value: Any) -> tuple[str, str]:
    _expect_keys(value, {"id", "valueSpace"}, "normalization")
    return (
        _require_curie(value["id"], "normalization id"),
        _require_curie(value["valueSpace"], "normalization valueSpace"),
    )


def _parse_bounds(value: Any) -> dict[str, int]:
    names = {
        "maxRecordsPerShard",
        "maxContextTokens",
        "maxActionTokens",
        "maxPanelQueries",
        "maxTargetsPerRecord",
    }
    _expect_keys(value, names, "bounds")
    for name in names:
        if not isinstance(value[name], int) or isinstance(value[name], bool) or value[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return {name: int(value[name]) for name in names}


def _parse_file_ref(
    value: Any, name: str, *, allow_empty: bool = False
) -> FileReference:
    _expect_keys(value, {"path", "sha256", "count"}, name)
    path = _require_nonempty(value["path"], f"{name} path")
    digest = _require_sha(value["sha256"], f"{name} sha256")
    count = value["count"]
    minimum = 0 if allow_empty else 1
    if not isinstance(count, int) or isinstance(count, bool) or count < minimum:
        qualifier = "non-negative" if allow_empty else "positive"
        raise ValueError(f"{name} count must be {qualifier}")
    return FileReference(path, digest, count)


def _parse_shards(
    value: Any, root: Path, bounds: dict[str, int]
) -> tuple[ShardReference, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("shards must be a non-empty list")
    result: list[ShardReference] = []
    paths: set[str] = set()
    for item in value:
        _expect_keys(item, {"path", "sha256", "records", "targetValues"}, "shard")
        path = _require_nonempty(item["path"], "shard path")
        if path in paths:
            raise ValueError("shard paths must be unique")
        paths.add(path)
        digest = _require_sha(item["sha256"], "shard sha256")
        records = item["records"]
        targets = item["targetValues"]
        if (
            not isinstance(records, int)
            or isinstance(records, bool)
            or records <= 0
            or records > bounds["maxRecordsPerShard"]
        ):
            raise ValueError("shard records violate maxRecordsPerShard")
        if not isinstance(targets, int) or isinstance(targets, bool) or targets < 0:
            raise ValueError("targetValues must be a non-negative integer")
        if targets > records * bounds["maxTargetsPerRecord"]:
            raise ValueError("targetValues violate maxTargetsPerRecord")
        _verify_digest(_resolve_file(root, path), digest)
        result.append(ShardReference(path, digest, records, targets))
    return tuple(result)


def _load_npz(root: Path, reference: FileReference, required: set[str]) -> dict[str, np.ndarray]:
    path = _resolve_file(root, reference.path)
    _verify_digest(path, reference.sha256)
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != required:
            raise ValueError(f"{reference.path} contains unexpected or missing arrays")
        return {name: source[name] for name in required}


def _validate_entity_dictionary(
    arrays: dict[str, np.ndarray],
    count: int,
    feature_dim: int,
    type_count: int,
    species_taxa: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    entity_id = arrays["entity_id"]
    entity_type = arrays["entity_type"]
    entity_species_taxon = arrays["entity_species_taxon"]
    value = arrays["entity_feature_value"]
    present = arrays["entity_feature_present"]
    _validate_curie_array(entity_id, count, "entity_id", unique=True)
    _require_int64(entity_type, (count,), "entity_type")
    _validate_index(entity_type, type_count, "entity_type")
    _require_int64(entity_species_taxon, (count,), "entity_species_taxon")
    if any(int(item) != 0 and int(item) not in species_taxa for item in entity_species_taxon):
        raise ValueError("entity species taxon must be declared or zero for species-neutral entities")
    _require_float32(value, (count, feature_dim), "entity_feature_value")
    _require_bool(present, (count, feature_dim), "entity_feature_present")
    _validate_masked_storage(value, present, "entity features")
    return entity_id, entity_type, entity_species_taxon, value, present


def _validate_query_dictionary(
    arrays: dict[str, np.ndarray], count: int, entity_count: int, readout_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query_id = arrays["query_id"]
    entity_index = arrays["query_entity_index"]
    readout_index = arrays["query_readout_index"]
    _validate_curie_array(query_id, count, "query_id", unique=True)
    _require_int64(entity_index, (count,), "query_entity_index")
    _validate_index(entity_index, entity_count, "query_entity_index")
    _require_int64(readout_index, (count,), "query_readout_index")
    _validate_index(readout_index, readout_count, "query_readout_index")
    return query_id, entity_index, readout_index


def _validate_query_panels(
    arrays: dict[str, np.ndarray], count: int, query_count: int, max_queries: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    panel_id = arrays["panel_id"]
    indptr = arrays["panel_indptr"]
    query_index = arrays["panel_query_index"]
    _validate_curie_array(panel_id, count, "panel_id", unique=True)
    _require_int64(indptr, (count + 1,), "panel_indptr")
    if query_index.ndim != 1 or query_index.dtype != np.int64:
        raise ValueError("panel_query_index must be one-dimensional int64")
    _validate_csr(indptr, len(query_index), "panel")
    _validate_index(query_index, query_count, "panel_query_index")
    used: set[int] = set()
    for panel in range(count):
        values = query_index[indptr[panel] : indptr[panel + 1]]
        if not len(values) or len(values) > max_queries:
            raise ValueError("query panel must be non-empty and within maxPanelQueries")
        if len(values) != len(set(int(item) for item in values)):
            raise ValueError("query panel cannot contain duplicates")
        used.update(int(item) for item in values)
    if used != set(range(query_count)):
        raise ValueError("every query dictionary entry must occur in a panel")
    return panel_id, indptr, query_index


def _validate_token_references(
    arrays: dict[str, np.ndarray],
    axis: str,
    shape: tuple[int, ...],
    entity_count: int,
    type_count: int,
    covariate_count: int,
) -> None:
    reference = arrays[f"{axis}_entity_index"]
    token_type = arrays[f"{axis}_type"]
    mask = arrays[f"{axis}_mask"]
    value = arrays[f"{axis}_covariate_value"]
    present = arrays[f"{axis}_covariate_present"]
    _require_int64(reference, shape, f"{axis}_entity_index")
    _require_int64(token_type, shape, f"{axis}_type")
    _require_bool(mask, shape, f"{axis}_mask")
    if (reference[~mask] != -1).any() or (token_type[~mask] != -1).any():
        raise ValueError(f"padded {axis} indices must use -1 sentinels")
    _validate_index(reference[mask], entity_count, f"{axis}_entity_index")
    _validate_index(token_type[mask], type_count, f"{axis}_type")
    covariate_shape = (*shape, covariate_count)
    _require_float32(value, covariate_shape, f"{axis}_covariate_value")
    _require_bool(present, covariate_shape, f"{axis}_covariate_present")
    _validate_masked_storage(value, present, f"{axis} covariates")
    if present[~mask].any():
        raise ValueError(f"padded {axis} tokens cannot contain covariates")


def _validate_curie_array(
    value: np.ndarray, count: int, name: str, unique: bool = False
) -> None:
    if value.ndim != 1 or value.shape != (count,) or value.dtype.kind not in "US":
        raise ValueError(f"{name} must be a fixed-width string vector")
    items = [str(item) for item in value]
    for item in items:
        _require_curie(item, name)
    if unique and len(items) != len(set(items)):
        raise ValueError(f"{name} must be unique")


def _require_int64(value: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if value.dtype != np.int64 or value.shape != shape:
        raise ValueError(f"{name} must have int64 dtype and shape {shape}")


def _require_float32(value: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if value.dtype != np.float32 or value.shape != shape:
        raise ValueError(f"{name} must have float32 dtype and shape {shape}")


def _require_bool(value: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if value.dtype != np.bool_ or value.shape != shape:
        raise ValueError(f"{name} must have bool dtype and shape {shape}")


def _validate_index(value: np.ndarray, upper: int, name: str) -> None:
    if value.size and (value.min() < 0 or value.max() >= upper):
        raise ValueError(f"{name} is out of range")


def _validate_csr(indptr: np.ndarray, values: int, name: str) -> None:
    if indptr[0] != 0 or indptr[-1] != values or (np.diff(indptr) < 0).any():
        raise ValueError(f"{name} CSR indptr is invalid")


def _validate_masked_storage(value: np.ndarray, present: np.ndarray, name: str) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must store finite values")
    if (value[~present] != 0).any():
        raise ValueError(f"missing {name} must be stored as numeric zero")


def _resolve_file(root: Path, value: str) -> Path:
    if "\\" in value:
        raise ValueError(f"unsafe corpus path: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"unsafe corpus path: {value!r}")
    root = root.resolve()
    unresolved_path = root.joinpath(*relative.parts)
    _reject_symlink_components(unresolved_path)
    path = unresolved_path.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"corpus path escapes the snapshot: {value!r}") from error
    if not path.is_file():
        raise ValueError(f"missing corpus file: {value}")
    return path


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    for component in reversed((absolute, *absolute.parents)):
        if component.exists() and component.is_symlink():
            raise ValueError(f"corpus paths cannot contain symlinks: {component}")


def _verify_digest(path: Path, expected: str) -> None:
    if _sha256_path(path) != expected:
        raise ValueError(f"file digest mismatch: {path.name}")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _float_tensor(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))


def _long_tensor(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value, dtype=np.int64))


def _bool_tensor(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value, dtype=np.bool_))
