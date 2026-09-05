"""Fitting-only Norman 2019 data boundary for compositional state experiments.

This loader deliberately accepts the author-normalized development artifact but
uses only its original ``split_train`` rows.  The original validation routes and
the separately sealed test-only artifact are not outcomes for this experiment.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = "slp.norman-fitting-compositional-state/v1"
TAXON = 9606
FEATURE_DIM = 577
PAIR_FOLDS = 3
PAIR_FOLD_NAMESPACE = "slp11-norman-fitting-leave-combination-out-v1"
GLOBAL_SPLIT_NAMESPACE = "slp11-development-v1"
GLOBAL_SPLIT_SEED = 731
DATA_SHA256 = "ab81e7ed07d7f111b3dfc964cece28a2db7de0dcf5975f6ff1a3bc2db0be683e"
FEATURE_SHA256 = "7b3d78af66f013e2d1df3a3f98924707ed111bc795757753e82a5e8f495408b5"


class CompositionalDataError(ValueError):
    """Raised when an input violates the fitting-only composition contract."""


@dataclass(frozen=True)
class CompositionalData:
    """Canonical fitting records and a fixed leave-combination-out protocol."""

    schema: str
    query_ids: np.ndarray
    action_ids: np.ndarray
    gene_features: np.ndarray
    canonical_actions: tuple[tuple[str, ...], ...]
    source_record_indices: tuple[tuple[int, ...], ...]
    action_feature_index: np.ndarray
    action_mask: np.ndarray
    y: np.ndarray
    observed: np.ndarray
    single_rows: np.ndarray
    combination_rows: np.ndarray
    combination_single_rows: np.ndarray
    combination_common_query_mask: np.ndarray
    combination_fold: np.ndarray
    target_value_space: str
    aggregation: str = "equal-construct mean within canonical action set"

    def fold_rows(self, fold: int) -> tuple[np.ndarray, np.ndarray]:
        """Return canonical fit rows and held combination rows for one fixed fold.

        All singles are retained in fitting.  Only combination records are held,
        so this tests composition from observed single endpoints without changing
        the repository's global held-gene routes.
        """

        if fold not in range(PAIR_FOLDS):
            raise CompositionalDataError(f"fold must be in [0, {PAIR_FOLDS})")
        held = self.combination_rows[self.combination_fold == fold]
        fit_combinations = self.combination_rows[self.combination_fold != fold]
        return np.concatenate((self.single_rows, fit_combinations)), held


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unpack(action_ids: np.ndarray, offsets: np.ndarray) -> tuple[tuple[str, ...], ...]:
    if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0:
        raise CompositionalDataError("invalid action offsets")
    if offsets[-1] != len(action_ids) or np.any(np.diff(offsets) < 1):
        raise CompositionalDataError("every record must contain an action")
    records = tuple(
        tuple(str(value) for value in action_ids[offsets[i] : offsets[i + 1]])
        for i in range(len(offsets) - 1)
    )
    if any(len(x) not in (1, 2) or x != tuple(sorted(set(x))) for x in records):
        raise CompositionalDataError("actions must be sorted unique singletons or pairs")
    return records


def _pair_fold(actions: tuple[str, ...]) -> int:
    payload = f"{PAIR_FOLD_NAMESPACE}|{'+'.join(actions)}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % PAIR_FOLDS


def _global_constituent_split(action_id: str) -> str:
    payload = f"{GLOBAL_SPLIT_NAMESPACE}|{GLOBAL_SPLIT_SEED}|{TAXON}|{action_id}".encode()
    bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def load_compositional_data(
    development_path: str | Path,
    static_feature_path: str | Path,
    *,
    verify_pins: bool = True,
) -> CompositionalData:
    """Load only original fitting rows and construct a fixed composition test."""

    development_path = Path(development_path)
    static_feature_path = Path(static_feature_path)
    if "test-only" in development_path.name.lower():
        raise CompositionalDataError("test-only artifacts are forbidden")
    if verify_pins and _sha256(development_path) != DATA_SHA256:
        raise CompositionalDataError("Norman development artifact SHA-256 mismatch")
    if verify_pins and _sha256(static_feature_path) != FEATURE_SHA256:
        raise CompositionalDataError("Norman static feature SHA-256 mismatch")

    with np.load(development_path, allow_pickle=False) as archive:
        required = {
            "action_ids", "action_offsets", "query_ids", "record_ids", "targets",
            "observed", "split_train", "split_validation", "split_test",
            "target_value_space",
        }
        if not required.issubset(archive.files):
            raise CompositionalDataError("Norman development fields are incomplete")
        action_ids = archive["action_ids"]
        action_offsets = archive["action_offsets"]
        query_ids = archive["query_ids"].astype(str)
        split_train = archive["split_train"].astype(np.int64)
        split_validation = archive["split_validation"].astype(np.int64)
        split_test = archive["split_test"].astype(np.int64)
        target_value_space = str(archive["target_value_space"].item())
        target_shape = archive["targets"].shape
        observed_shape = archive["observed"].shape
        record_count = len(archive["record_ids"])
        # Retain quantitative outcomes for original fitting rows only.
        fitting_targets = archive["targets"][split_train].astype(np.float32)
        fitting_observed = archive["observed"][split_train].astype(np.bool_)
    records = _unpack(action_ids, action_offsets)
    n_records, n_queries = target_shape
    if len(records) != n_records or observed_shape != (n_records, n_queries) or record_count != n_records:
        raise CompositionalDataError("record, target, and mask axes disagree")
    if len(split_test):
        raise CompositionalDataError("development artifact unexpectedly contains test rows")
    train = split_train
    validation = split_validation
    if not len(train) or set(train.tolist()) & set(validation.tolist()):
        raise CompositionalDataError("original train/validation routes are invalid")
    if np.any(train < 0) or np.any(train >= n_records):
        raise CompositionalDataError("fitting row index is out of range")

    # Canonicalization happens after selecting split_train: no validation outcome
    # can influence a target, mask, roster, fold, or normalization choice.
    grouped: dict[tuple[str, ...], list[int]] = {}
    for row in train:
        grouped.setdefault(records[int(row)], []).append(int(row))
    if any(_global_constituent_split(gene) != "train" for action in grouped for gene in action):
        raise CompositionalDataError("a fitting action violates the global held-gene route")
    canonical_actions = tuple(sorted(grouped, key=lambda x: (len(x), x)))
    source_indices = tuple(tuple(grouped[x]) for x in canonical_actions)
    fitting_position = {int(row): position for position, row in enumerate(train)}
    y = np.zeros((len(canonical_actions), n_queries), dtype=np.float32)
    observed = np.zeros_like(y, dtype=np.bool_)
    for output_row, rows in enumerate(source_indices):
        index = np.asarray([fitting_position[row] for row in rows], dtype=np.int64)
        counts = fitting_observed[index].sum(axis=0)
        observed[output_row] = counts > 0
        y[output_row] = np.divide(
            np.where(fitting_observed[index], fitting_targets[index], 0.0).sum(axis=0),
            counts,
            out=np.zeros(n_queries, dtype=np.float32),
            where=counts > 0,
        )
    if not np.isfinite(y[observed]).all():
        raise CompositionalDataError("observed fitting outcomes must be finite")

    action_roster = np.asarray(sorted({g for action in canonical_actions for g in action}))
    with np.load(static_feature_path, allow_pickle=False) as archive:
        ids = archive["entity_id"].astype(str)
        taxa = archive["entity_taxon"]
        values = archive["feature_values"]
    if len(ids) != len(set(ids.tolist())) or values.shape != (len(ids), FEATURE_DIM):
        raise CompositionalDataError("static feature axes are invalid")
    if not np.array_equal(taxa, np.full(len(ids), TAXON, dtype=np.int64)):
        raise CompositionalDataError("static pack is not wholly human")
    lookup = {gene: row for row, gene in enumerate(ids)}
    if any(gene not in lookup for gene in action_roster):
        raise CompositionalDataError("a fitting action lacks static features")
    gene_features = np.asarray([values[lookup[g]] for g in action_roster], dtype=np.float32)
    if not np.isfinite(gene_features).all():
        raise CompositionalDataError("fitting action features contain non-finite values")
    roster_lookup = {gene: row for row, gene in enumerate(action_roster)}
    action_feature_index = np.full((len(canonical_actions), 2), -1, dtype=np.int64)
    action_mask = np.zeros((len(canonical_actions), 2), dtype=np.bool_)
    for row, actions in enumerate(canonical_actions):
        for column, gene in enumerate(actions):
            action_feature_index[row, column] = roster_lookup[gene]
            action_mask[row, column] = True

    single_rows = np.asarray(
        [i for i, actions in enumerate(canonical_actions) if len(actions) == 1],
        dtype=np.int64,
    )
    combination_rows = np.asarray(
        [i for i, actions in enumerate(canonical_actions) if len(actions) == 2],
        dtype=np.int64,
    )
    single_lookup = {canonical_actions[row][0]: row for row in single_rows}
    if any(gene not in single_lookup for row in combination_rows for gene in canonical_actions[row]):
        raise CompositionalDataError("every combination requires both fitting single endpoints")
    combination_single_rows = np.asarray(
        [[single_lookup[g] for g in canonical_actions[row]] for row in combination_rows],
        dtype=np.int64,
    )
    combination_common_query_mask = np.stack(
        [
            observed[row] & observed[left] & observed[right]
            for row, (left, right) in zip(combination_rows, combination_single_rows, strict=True)
        ]
    )
    combination_fold = np.asarray(
        [_pair_fold(canonical_actions[row]) for row in combination_rows], dtype=np.int64
    )
    if set(combination_fold.tolist()) != set(range(PAIR_FOLDS)):
        raise CompositionalDataError("fixed combination folds contain an empty fold")

    return CompositionalData(
        schema=SCHEMA,
        query_ids=query_ids,
        action_ids=action_roster,
        gene_features=gene_features,
        canonical_actions=canonical_actions,
        source_record_indices=source_indices,
        action_feature_index=action_feature_index,
        action_mask=action_mask,
        y=y,
        observed=observed,
        single_rows=single_rows,
        combination_rows=combination_rows,
        combination_single_rows=combination_single_rows,
        combination_common_query_mask=combination_common_query_mask,
        combination_fold=combination_fold,
        target_value_space=target_value_space,
    )
