"""Self-contained NumPy adapter for native count panels.

The adapter loads fitting/control count data only.  It preserves each source's
query axis and library denominator and exposes deterministic sampling contracts;
it contains no model, training, split selection, or benchmark behavior.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

REGISTRY_SCHEMA = "slp.human-essential-joint-training-registry/v1"
EXPECTED_MEAN_SCALE = {
    "k562": 0.004324449194506417,
    "rpe1": 0.01041484917,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_npz(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        missing = set(required) - set(source.files)
        if missing:
            raise ValueError(f"missing arrays in {path.name}: {sorted(missing)}")
        arrays = {name: source[name] for name in required}
    if any(value.dtype.kind == "O" for value in arrays.values()):
        raise ValueError(f"object arrays are forbidden: {path.name}")
    return arrays


def _as_readonly_float32(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"invalid {name}; expected finite {shape}")
    array = array.copy()
    array.flags.writeable = False
    return array


def _readonly_identity(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).astype(str).copy()
    if result.ndim != 1 or len(set(result.tolist())) != len(result):
        raise ValueError("identity axis must contain unique one-dimensional values")
    result.flags.writeable = False
    return result


def _basal_rate(control: dict[str, np.ndarray], query_count: int) -> np.ndarray:
    raw = np.asarray(control["raw_count_sum"])
    library = np.asarray(control["library_count_sum"])
    cells = np.asarray(control["num_cells"])
    if (
        raw.ndim != 2
        or raw.shape[1] != query_count
        or library.shape != (len(raw),)
        or cells.shape != (len(raw),)
        or np.any(raw < 0)
        or np.any(library <= 0)
        or np.any(cells <= 0)
    ):
        raise ValueError("invalid reconstruction-training control moments")
    rate = 10_000.0 * (raw.astype(np.float64) + 0.5) / (
        library.astype(np.float64)[:, None] + 0.5 * query_count
    )
    result = rate.astype(np.float32)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("invalid smoothed basal rates")
    result.flags.writeable = False
    return result


def _population_contract(
    moments: dict[str, np.ndarray], basal_rate: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    total = np.asarray(moments["cp10k_sum"], dtype=np.float64)
    cells = np.asarray(moments["cell_count"], dtype=np.float64)
    gem_cells = np.asarray(moments["gem_cell_count"], dtype=np.float64)
    if (
        total.ndim != 2
        or cells.shape != (len(total),)
        or gem_cells.shape != (len(total), len(basal_rate))
        or total.shape[1] != basal_rate.shape[1]
        or not np.isfinite(total).all()
        or np.any(total < 0)
        or np.any(cells <= 0)
        or np.any(gem_cells < 0)
        or not np.array_equal(gem_cells.sum(axis=1), cells)
    ):
        raise ValueError("invalid fitting population moments")
    targets64 = np.log1p(total / cells[:, None])
    weights64 = gem_cells / cells[:, None]
    anchor = np.log1p(weights64 @ basal_rate.astype(np.float64))
    residual = targets64 - anchor
    mean_scale = float(np.mean(np.square(residual - residual.mean(axis=0, dtype=np.float64))))
    targets = targets64.astype(np.float32)
    weights = weights64.astype(np.float32)
    targets.flags.writeable = False
    weights.flags.writeable = False
    return targets, weights, mean_scale


def _validate_cell_metadata(
    metadata: dict[str, np.ndarray], gene_ids: np.ndarray, gem_group: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], tuple[tuple[np.ndarray, ...], ...]]:
    required = (
        "action_ids", "population_ids", "gem_group", "is_control", "library_size",
    )
    if not all(name in metadata for name in required):
        raise ValueError("incomplete cell metadata")
    n = len(metadata["action_ids"])
    if any(np.asarray(metadata[name]).shape != (n,) for name in required):
        raise ValueError("cell metadata axes differ")
    action = np.asarray(metadata["action_ids"]).astype(str)
    population = np.asarray(metadata["population_ids"]).astype(str)
    gem = np.asarray(metadata["gem_group"])
    control = np.asarray(metadata["is_control"])
    if control.dtype != np.bool_:
        raise ValueError("control flags must be Boolean")
    library = np.asarray(metadata["library_size"])
    genes = np.asarray(gene_ids).astype(str)
    gene_lookup = {gene: row for row, gene in enumerate(genes)}
    gem_lookup = {int(value): row for row, value in enumerate(np.asarray(gem_group))}
    if len(gene_lookup) != len(genes) or len(gem_lookup) != len(gem_group):
        raise ValueError("duplicate gene or GEM identity")
    if (not np.isfinite(library).all() or np.any(library <= 0)
            or not np.array_equal(library, np.round(library))
            or set(action[~control]) != set(genes) or np.any(action[control] != "")):
        raise ValueError("training row roles do not match fitting/control contract")
    try:
        action_index = np.asarray([-1 if is_control else gene_lookup[gene] for gene, is_control in zip(action, control)], np.int64)
        context_index = np.asarray([gem_lookup[int(value)] for value in gem], np.int64)
    except KeyError as error:
        raise ValueError(f"row references unknown action or GEM: {error}") from error
    control_rows = tuple(np.flatnonzero(control & (context_index == index)) for index in range(len(gem_group)))
    if any(len(rows) == 0 for rows in control_rows):
        raise ValueError("every GEM context requires fitting-control cells")
    populations: list[tuple[np.ndarray, ...]] = []
    for gene_index in range(len(genes)):
        rows = np.flatnonzero(action_index == gene_index)
        labels = sorted(set(population[rows]))
        if not labels or "" in labels:
            raise ValueError("target cells require exact population identity")
        groups = tuple(rows[population[rows] == label] for label in labels)
        if any(len(group) == 0 for group in groups):
            raise AssertionError("empty population group")
        populations.append(groups)
    action_index.flags.writeable = False
    context_index.flags.writeable = False
    return action_index, context_index, control_rows, tuple(populations)


@dataclass(frozen=True)
class PanelData:
    """One source-native fitting/control count panel."""

    source_id: str
    query_features: np.ndarray
    gene_action_features: np.ndarray
    basal_rate: np.ndarray
    gene_ids: np.ndarray
    query_ids: np.ndarray
    context_ids: np.ndarray
    population_targets: np.ndarray
    population_context_weights: np.ndarray
    fitting_mean_scale: float
    cell_metadata: MappingProxyType
    counts: np.memmap
    _cell_action_index: np.ndarray
    _cell_context_index: np.ndarray
    _control_rows_by_context: tuple[np.ndarray, ...]
    _target_rows_by_gene_population: tuple[tuple[np.ndarray, ...], ...]

    def sample_cells(
        self, rng: np.random.Generator, n_controls: int = 64, n_targets: int = 64
    ) -> dict[str, np.ndarray]:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator")
        if n_controls < 0 or n_targets < 0 or n_controls + n_targets <= 0:
            raise ValueError("sample sizes must be nonnegative with positive total")
        control_context = rng.integers(len(self.context_ids), size=n_controls)
        control_rows = np.asarray([
            rows[rng.integers(len(rows))]
            for rows in (self._control_rows_by_context[index] for index in control_context)
        ], dtype=np.int64)
        target_genes = rng.integers(len(self.gene_ids), size=n_targets)
        target_rows = []
        for gene in target_genes:
            populations = self._target_rows_by_gene_population[int(gene)]
            selected = populations[int(rng.integers(len(populations)))]
            target_rows.append(selected[int(rng.integers(len(selected)))])
        target_rows_array = np.asarray(target_rows, dtype=np.int64)
        rows = np.concatenate((control_rows, target_rows_array))
        action_index = self._cell_action_index[rows]
        batch = len(rows)
        actions = np.zeros((batch, 1, self.gene_action_features.shape[1]), np.float32)
        target_mask = action_index >= 0
        actions[target_mask, 0] = self.gene_action_features[action_index[target_mask]]
        counts_uint = np.asarray(self.counts[rows])
        library_int = np.asarray(self.cell_metadata["library_size"])[rows]
        if not np.array_equal(counts_uint.sum(axis=1, dtype=np.uint64), library_int.astype(np.uint64)):
            raise ValueError("sampled count rows do not match full native-panel library sizes")
        return {
            "actions": actions,
            "action_mask": target_mask[:, None],
            "context_index": self._cell_context_index[rows].copy(),
            "counts": counts_uint.astype(np.float32),
            "observed": np.ones((batch, len(self.query_ids)), dtype=bool),
            "library": library_int.astype(np.float32),
            "row_index": rows,
        }

    def sample_populations(self, rng: np.random.Generator, n: int = 16) -> dict[str, np.ndarray]:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator")
        if n <= 0 or n > len(self.gene_ids):
            raise ValueError("n must select unique fitting genes without replacement")
        genes = np.asarray(rng.choice(len(self.gene_ids), size=n, replace=False), np.int64)
        return {
            "actions": self.gene_action_features[genes, None].copy(),
            "action_mask": np.ones((n, 1), dtype=bool),
            "context_weights": self.population_context_weights[genes].copy(),
            "target_log1p_mean": self.population_targets[genes].copy(),
            "gene_index": genes,
        }

    def replace_features(
        self, query_features: np.ndarray, gene_action_features: np.ndarray
    ) -> PanelData:
        """Return a validated copy with explicitly supplied aligned features."""
        query_value = np.asarray(query_features)
        action_value = np.asarray(gene_action_features)
        if query_value.ndim != 2 or action_value.ndim != 2:
            raise ValueError("replacement features must be two-dimensional")
        if query_value.shape[1] != action_value.shape[1]:
            raise ValueError("query_features and gene_action_features widths differ")
        query = _as_readonly_float32(
            query_value, (len(self.query_ids), query_value.shape[1]),
            "query_features",
        )
        action = _as_readonly_float32(
            action_value,
            (len(self.gene_ids), query.shape[1]),
            "gene_action_features",
        )
        return replace(self, query_features=query, gene_action_features=action)


def _resolve(workspace_root: Path, entry: dict[str, Any]) -> Path:
    path = (workspace_root / entry["path"]).resolve()
    try:
        path.relative_to(workspace_root.resolve())
    except ValueError as error:
        raise ValueError("registry path escapes workspace root") from error
    if not path.is_file() or _sha256(path) != entry["sha256"]:
        raise ValueError(f"registry member checksum mismatch: {entry['path']}")
    return path


def load_panels(registry_path: str | Path, workspace_root: str | Path) -> OrderedDict[str, PanelData]:
    """Load K562/RPE1 fitting panels from a pinned outcome-free registry."""
    root = Path(workspace_root).resolve()
    registry_file = Path(registry_path)
    if not registry_file.is_absolute():
        registry_file = root / registry_file
    registry_file = registry_file.resolve()
    try:
        registry_file.relative_to(root)
    except ValueError as error:
        raise ValueError("registry must be inside workspace_root") from error
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    if (
        registry.get("schema") != REGISTRY_SCHEMA
        or registry.get("noJointTargetMatrix") is not True
        or registry.get("developmentOrTestCountsAccessed") is not False
    ):
        raise ValueError("unsupported or unsafe registry")
    members = {name: _resolve(root, entry) for name, entry in registry["artifacts"].items()}
    registry_dir = registry_file.parent
    static_entry = registry["static"]
    static_path = registry_dir / static_entry["path"]
    if _sha256(static_path) != static_entry["sha256"]:
        raise ValueError("shared static checksum mismatch")
    static = _load_npz(
        static_path,
        ("entity_id", "entity_taxon", "normalized_feature_values", "feature_values"),
    )
    if not np.all(static["entity_taxon"] == 9606):
        raise ValueError("shared static taxonomy drift")

    panels: OrderedDict[str, PanelData] = OrderedDict()
    for source in ("k562", "rpe1"):
        index_entry = registry["indices"][source]
        index_path = registry_dir / index_entry["path"]
        if _sha256(index_path) != index_entry["sha256"]:
            raise ValueError(f"source index checksum mismatch: {source}")
        index = _load_npz(
            index_path,
            ("source_id", "query_ids", "query_entity_index", "action_ids", "action_entity_index",
             "action_role", "fitting_action_ids", "fitting_action_entity_index", "gem_group",
             "context_ids", "full_native_library_query_count"),
        )
        query_ids = index["query_ids"].astype(str)
        genes = index["fitting_action_ids"].astype(str)
        if str(index["source_id"]) != source:
            raise ValueError(f"source identity drift: {source}")
        if int(index["full_native_library_query_count"]) != len(query_ids):
            raise ValueError(f"native query count drift: {source}")
        if (not np.array_equal(static["entity_id"][index["query_entity_index"]].astype(str), query_ids)
                or not np.array_equal(static["entity_id"][index["fitting_action_entity_index"]].astype(str), genes)):
            raise ValueError(f"static identity index drift: {source}")
        query_features = _as_readonly_float32(
            static["normalized_feature_values"][index["query_entity_index"]],
            (len(query_ids), 577), "query_features",
        )
        action_features = _as_readonly_float32(
            static["normalized_feature_values"][index["fitting_action_entity_index"]],
            (len(genes), 577), "gene_action_features",
        )
        controls = _load_npz(
            members[f"{source}_control_moments"],
            ("query_ids", "query_taxon", "gem_group", "raw_count_sum", "library_count_sum", "num_cells"),
        )
        moments = _load_npz(
            members[f"{source}_fit_moments"],
            ("query_ids", "query_taxon", "gem_group", "action_ids", "cp10k_sum", "cell_count", "gem_cell_count"),
        )
        if (
            not np.array_equal(query_ids, controls["query_ids"].astype(str))
            or not np.array_equal(query_ids, moments["query_ids"].astype(str))
            or not np.array_equal(index["gem_group"], controls["gem_group"])
            or not np.array_equal(index["gem_group"], moments["gem_group"])
            or not np.array_equal(genes, moments["action_ids"].astype(str))
            or not np.all(controls["query_taxon"] == 9606)
            or not np.all(moments["query_taxon"] == 9606)
        ):
            raise ValueError(f"panel identity axis drift: {source}")
        basal = _basal_rate(controls, len(query_ids))
        targets, weights, mean_scale = _population_contract(moments, basal)
        if not np.isclose(mean_scale, EXPECTED_MEAN_SCALE[source], rtol=0.0, atol=5e-12):
            raise ValueError(f"fitting anchored-mean scale drift: {source}")
        row_names = (
            "query_ids", "source_row_index", "cell_ids", "action_ids", "guide_pair_ids",
            "population_ids", "gem_group", "is_control", "library_size",
        )
        rows = _load_npz(members[f"{source}_rows"], row_names)
        if not np.array_equal(query_ids, rows.pop("query_ids").astype(str)):
            raise ValueError(f"training-row query order drift: {source}")
        action_index, context_index, control_groups, target_groups = _validate_cell_metadata(
            rows, genes, index["gem_group"]
        )
        expected_shape = (
            int(registry["sources"][source]["trainingRows"]["fit"])
            + int(registry["sources"][source]["trainingRows"]["control"]),
            len(query_ids),
        )
        if len(rows["action_ids"]) != expected_shape[0]:
            raise ValueError(f"training row count drift: {source}")
        count_path = members[f"{source}_counts"]
        if count_path.stat().st_size != int(np.prod(expected_shape)) * np.dtype("<u2").itemsize:
            raise ValueError(f"count mmap byte size drift: {source}")
        count_mmap = np.memmap(count_path, dtype="<u2", mode="r", shape=expected_shape, order="C")
        if count_mmap.flags.writeable:
            raise ValueError("count mmap must be read-only")
        immutable_rows = {}
        for name, value in rows.items():
            copied = np.asarray(value).copy()
            copied.flags.writeable = False
            immutable_rows[name] = copied
        panels[source] = PanelData(
            source_id=source,
            query_features=query_features,
            gene_action_features=action_features,
            basal_rate=basal,
            gene_ids=_readonly_identity(genes),
            query_ids=_readonly_identity(query_ids),
            context_ids=_readonly_identity(index["context_ids"].astype(str)),
            population_targets=targets,
            population_context_weights=weights,
            fitting_mean_scale=mean_scale,
            cell_metadata=MappingProxyType(immutable_rows),
            counts=count_mmap,
            _cell_action_index=action_index,
            _cell_context_index=context_index,
            _control_rows_by_context=control_groups,
            _target_rows_by_gene_population=target_groups,
        )
    if set(panels["k562"].context_ids) & set(panels["rpe1"].context_ids):
        raise ValueError("cross-panel context collision")
    return panels


__all__ = ["PanelData", "load_panels"]
