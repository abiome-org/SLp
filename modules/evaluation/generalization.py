"""Leakage-audited molecular generalization splits and simple baselines.

The protocols in this module operate on molecular perturbation outcomes, not
synthetic-lethality labels.  Missing provenance makes a protocol ineligible;
one metadata axis is never silently substituted for another.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping, Sequence

import numpy as np


PROTOCOLS = (
    "pair_cold",
    "composition_gene_cold",
    "intervention_gene_cold",
    "context_cold",
    "source_cold",
    "condition_cold",
    "source_gene_cold",
)


def _metadata(values: np.ndarray | Sequence[str] | None, rows: int, name: str):
    if values is None:
        return None
    array = np.asarray(values).astype(str)
    if array.ndim == 0:
        array = np.repeat(array.reshape(1), rows)
    if array.shape != (rows,):
        raise ValueError(f"{name} must have shape ({rows},), got {array.shape}")
    if np.any(np.char.strip(array) == ""):
        raise ValueError(f"{name} contains an empty identifier")
    return array


@dataclass(frozen=True)
class GeneralizationTable:
    """Canonical perturbation-outcome table used by every hard split."""

    actions: np.ndarray
    target: np.ndarray | None = None
    action_modes: np.ndarray | Sequence[Sequence[str]] | None = None
    action_doses: np.ndarray | Sequence[Sequence[float]] | None = None
    context: np.ndarray | Sequence[str] | None = None
    source: np.ndarray | Sequence[str] | None = None
    condition: np.ndarray | Sequence[str] | None = None
    target_semantics: str | None = None

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions)
        if actions.ndim != 2 or not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("actions must be an integer array with shape [rows, slots]")
        if actions.shape[0] == 0 or actions.shape[1] == 0:
            raise ValueError("actions must contain at least one row and one slot")
        actions = actions.astype("int64", copy=False)
        if np.any(actions < -1):
            raise ValueError("action identifiers must be non-negative; -1 is padding")
        modes = None if self.action_modes is None else np.asarray(self.action_modes).astype(str)
        if modes is not None and modes.shape != actions.shape:
            raise ValueError(f"action_modes must have shape {actions.shape}")
        if modes is not None:
            if np.any((actions >= 0) & (np.char.strip(modes) == "")):
                raise ValueError("every valid action must have an action mode")
            if np.any((actions < 0) & (np.char.strip(modes) != "")):
                raise ValueError("padded actions cannot have an action mode")
        doses = None if self.action_doses is None else np.asarray(self.action_doses, dtype="float64")
        if doses is not None and doses.shape != actions.shape:
            raise ValueError(f"action_doses must have shape {actions.shape}")
        if doses is not None:
            if not np.all(np.isfinite(doses)):
                raise ValueError("action_doses must be finite")
            if np.any((actions >= 0) & (doses <= 0)):
                raise ValueError("every valid action must have a positive dose")
            if np.any((actions < 0) & (doses != 0)):
                raise ValueError("padded actions must have zero dose")
        for index, row in enumerate(actions):
            valid = row[row >= 0]
            if not len(valid):
                raise ValueError("every row must contain at least one action")
            row_modes = np.repeat("unspecified", len(valid)) if modes is None else modes[index][row >= 0]
            row_doses = np.ones(len(valid)) if doses is None else doses[index][row >= 0]
            identities = np.asarray(
                [f"{gene}:{mode}@{dose:.9g}" for gene, mode, dose in zip(valid, row_modes, row_doses)]
            )
            if len(np.unique(identities)) != len(identities):
                raise ValueError("an action set cannot contain duplicate interventions")
        rows = len(actions)
        target = None if self.target is None else np.asarray(self.target)
        if target is not None:
            if target.ndim == 1:
                target = target[:, None]
            if target.ndim != 2 or target.shape[0] != rows:
                raise ValueError(f"target must have shape ({rows}, dimensions)")
            if not np.issubdtype(target.dtype, np.number) or not np.all(np.isfinite(target)):
                raise ValueError("target must contain only finite numeric values")
            target = target.astype("float64", copy=False)
        semantics = None if self.target_semantics is None else str(self.target_semantics)
        if semantics is not None and semantics not in {"perturbation_delta", "absolute_state"}:
            raise ValueError("target_semantics must be perturbation_delta or absolute_state")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "action_modes", modes)
        object.__setattr__(self, "action_doses", doses)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "context", _metadata(self.context, rows, "context"))
        object.__setattr__(self, "source", _metadata(self.source, rows, "source"))
        object.__setattr__(self, "condition", _metadata(self.condition, rows, "condition"))
        object.__setattr__(self, "target_semantics", semantics)

    @property
    def cardinality(self) -> np.ndarray:
        return (self.actions >= 0).sum(axis=1)

    def action_sets(self) -> np.ndarray:
        keys = []
        for index, row in enumerate(self.actions):
            valid = row >= 0
            members = [str(gene) for gene in row[valid]]
            if self.action_modes is not None or self.action_doses is not None:
                modes = (
                    np.repeat("unspecified", valid.sum())
                    if self.action_modes is None
                    else self.action_modes[index][valid]
                )
                doses = (
                    np.ones(valid.sum())
                    if self.action_doses is None
                    else self.action_doses[index][valid]
                )
                members = [
                    f"{gene}:{mode}@{dose:.9g}"
                    for gene, mode, dose in zip(row[valid], modes, doses)
                ]
            keys.append("+".join(sorted(members)))
        return np.asarray(keys)


@dataclass(frozen=True)
class EvidenceRequirements:
    min_train_rows: int = 128
    min_test_rows: int = 32
    min_test_action_sets: int = 16
    min_test_genes: int = 8

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class GeneralizationSplit:
    protocol: str
    fold: int
    train: np.ndarray
    test: np.ndarray
    excluded: np.ndarray
    eligible: bool
    reasons: tuple[str, ...]
    audit: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "fold": self.fold,
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "audit": dict(self.audit),
        }


def _fold(value: object, folds: int, seed: int, namespace: str) -> int:
    payload = f"{seed}\0{namespace}\0{value}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % folds


def _held_genes(table: GeneralizationTable, folds: int, fold: int, seed: int) -> set[int]:
    genes = np.unique(table.actions[table.actions >= 0])
    return {int(gene) for gene in genes if _fold(int(gene), folds, seed, "gene") == fold}


def _rows_contain(actions: np.ndarray, genes: set[int]) -> np.ndarray:
    if not genes:
        return np.zeros(len(actions), dtype=bool)
    return np.isin(actions, np.fromiter(genes, dtype="int64")).any(axis=1)


def _rows_all_in(actions: np.ndarray, genes: set[int]) -> np.ndarray:
    if not genes:
        return np.zeros(len(actions), dtype=bool)
    valid = actions >= 0
    return (np.isin(actions, np.fromiter(genes, dtype="int64")) | ~valid).all(axis=1)


def _group_holdout(values: np.ndarray, folds: int, fold: int, seed: int, namespace: str):
    groups = sorted(
        np.unique(values).tolist(),
        key=lambda value: (
            _fold(value, 2**31 - 1, seed, namespace),
            str(value),
        ),
    )
    held = {value for rank, value in enumerate(groups) if rank % folds == fold}
    return np.isin(values, list(held)), held


def _counts(table: GeneralizationTable, rows: np.ndarray) -> dict[str, int]:
    indices = np.flatnonzero(rows)
    actions = table.actions[indices]
    valid = actions[actions >= 0]
    multi = table.cardinality[indices] >= 2
    return {
        "rows": int(len(indices)),
        "multi_action_rows": int(multi.sum()),
        "unique_action_sets": int(len(np.unique(table.action_sets()[indices]))),
        "unique_genes": int(len(np.unique(valid))),
        "action_modes": 0 if table.action_modes is None else int(
            len(np.unique(table.action_modes[indices][table.actions[indices] >= 0]))
        ),
        "contexts": 0 if table.context is None else int(len(np.unique(table.context[indices]))),
        "sources": 0 if table.source is None else int(len(np.unique(table.source[indices]))),
        "conditions": 0 if table.condition is None else int(len(np.unique(table.condition[indices]))),
    }


def _finish(
    table: GeneralizationTable,
    protocol: str,
    fold: int,
    train: np.ndarray,
    test: np.ndarray,
    excluded: np.ndarray,
    requirements: EvidenceRequirements,
    reasons: Iterable[str] = (),
) -> GeneralizationSplit:
    train_counts = _counts(table, train)
    test_counts = _counts(table, test)
    failures = list(reasons)
    thresholds = {
        "train rows": (train_counts["rows"], requirements.min_train_rows),
        "test rows": (test_counts["rows"], requirements.min_test_rows),
        "test action sets": (
            test_counts["unique_action_sets"],
            requirements.min_test_action_sets,
        ),
        "test genes": (test_counts["unique_genes"], requirements.min_test_genes),
    }
    failures.extend(
        f"insufficient {name}: {actual} < {minimum}"
        for name, (actual, minimum) in thresholds.items()
        if actual < minimum
    )
    audit = {
        "train": train_counts,
        "test": test_counts,
        "excluded_rows": int(excluded.sum()),
        "thresholds": {
            "min_train_rows": requirements.min_train_rows,
            "min_test_rows": requirements.min_test_rows,
            "min_test_action_sets": requirements.min_test_action_sets,
            "min_test_genes": requirements.min_test_genes,
        },
    }
    return GeneralizationSplit(
        protocol,
        fold,
        np.flatnonzero(train),
        np.flatnonzero(test),
        np.flatnonzero(excluded),
        not failures,
        tuple(failures),
        audit,
    )


def make_split(
    table: GeneralizationTable,
    protocol: str,
    *,
    folds: int = 5,
    fold: int = 0,
    seed: int = 731,
    requirements: EvidenceRequirements | None = None,
) -> GeneralizationSplit:
    """Construct one fold and assert its defining leakage boundary."""

    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    if folds < 2 or not 0 <= fold < folds:
        raise ValueError("folds must be at least two and fold must be in range")
    requirements = requirements or EvidenceRequirements()
    rows = len(table.actions)
    multi = table.cardinality >= 2
    train = np.zeros(rows, dtype=bool)
    test = np.zeros(rows, dtype=bool)
    unavailable: list[str] = []

    if protocol == "pair_cold":
        keys = table.action_sets()
        held, _ = _group_holdout(keys, folds, fold, seed, "action_set")
        test = held & multi
        train = ~test
        overlap = set(keys[train]) & set(keys[test])
        if overlap:
            raise AssertionError("pair-cold split leaked an action set")

    elif protocol in {"composition_gene_cold", "intervention_gene_cold"}:
        held_genes = _held_genes(table, folds, fold, seed)
        contains = _rows_contain(table.actions, held_genes)
        all_held = _rows_all_in(table.actions, held_genes)
        test = multi & all_held
        if protocol == "composition_gene_cold":
            train = (table.cardinality == 1) | (multi & ~contains)
            train_multi_genes = set(table.actions[train & multi].ravel()) - {-1}
            if train_multi_genes & held_genes:
                raise AssertionError("composition-cold split leaked a held gene into a composition")
        else:
            train = ~contains
            train_genes = set(table.actions[train].ravel()) - {-1}
            if train_genes & held_genes:
                raise AssertionError("intervention-cold split leaked a held intervention")

    elif protocol in {"context_cold", "source_cold", "condition_cold"}:
        field_name = protocol.removesuffix("_cold")
        values = getattr(table, field_name)
        if values is None:
            unavailable.append(f"missing required {field_name} metadata")
        else:
            held, _ = _group_holdout(values, folds, fold, seed, field_name)
            test = held & multi
            train = ~held
            if set(values[train]) & set(values[test]):
                raise AssertionError(f"{protocol} leaked a {field_name}")

    else:  # source_gene_cold
        if table.source is None:
            unavailable.append("missing required source metadata")
        else:
            held_sources, _ = _group_holdout(table.source, folds, fold, seed, "source")
            held_genes = _held_genes(table, folds, fold, seed)
            contains = _rows_contain(table.actions, held_genes)
            all_held = _rows_all_in(table.actions, held_genes)
            test = held_sources & multi & all_held
            train = ~held_sources & ~contains
            if set(table.source[train]) & set(table.source[test]):
                raise AssertionError("source-gene-cold split leaked a source")
            train_genes = set(table.actions[train].ravel()) - {-1}
            if train_genes & held_genes:
                raise AssertionError("source-gene-cold split leaked a held intervention")

    excluded = ~(train | test)
    if np.any(train & test):
        raise AssertionError("split has rows assigned to both train and test")
    return _finish(table, protocol, fold, train, test, excluded, requirements, unavailable)


def make_suite(
    table: GeneralizationTable,
    *,
    protocols: Sequence[str] = PROTOCOLS,
    folds: int = 5,
    seed: int = 731,
    requirements: EvidenceRequirements | None = None,
) -> list[GeneralizationSplit]:
    return [
        make_split(
            table,
            protocol,
            folds=folds,
            fold=fold,
            seed=seed,
            requirements=requirements,
        )
        for protocol in protocols
        for fold in range(folds)
    ]


def cardinality_mean_baseline(
    table: GeneralizationTable, split: GeneralizationSplit
) -> np.ndarray:
    """Predict the training mean for the same action cardinality."""

    if table.target is None:
        raise ValueError("a target is required for baseline prediction")
    if not len(split.train):
        raise ValueError("the split has no training rows")
    global_mean = table.target[split.train].mean(axis=0)
    train_cardinality = table.cardinality[split.train]
    predictions = []
    for row in split.test:
        matched = split.train[train_cardinality == table.cardinality[row]]
        predictions.append(table.target[matched].mean(axis=0) if len(matched) else global_mean)
    return np.asarray(predictions)


def additive_single_baseline(
    table: GeneralizationTable,
    split: GeneralizationSplit,
    *,
    match_fields: Sequence[str] = ("source", "context", "condition"),
) -> tuple[np.ndarray, dict[str, object]]:
    """Sum source/context/condition-matched singleton deltas with audited fallback."""

    if table.target is None:
        raise ValueError("a target is required for baseline prediction")
    if table.target_semantics != "perturbation_delta":
        raise ValueError("the additive baseline requires target_semantics=perturbation_delta")
    fields = tuple(field for field in match_fields if getattr(table, field) is not None)
    singleton = split.train[table.cardinality[split.train] == 1]
    exact: dict[tuple[object, ...], list[np.ndarray]] = {}
    action_only: dict[tuple[object, ...], list[np.ndarray]] = {}
    gene_only: dict[int, list[np.ndarray]] = {}
    for row in singleton:
        gene = int(table.actions[row][table.actions[row] >= 0][0])
        mode = "unspecified" if table.action_modes is None else table.action_modes[row][table.actions[row] >= 0][0]
        dose = 1.0 if table.action_doses is None else float(table.action_doses[row][table.actions[row] >= 0][0])
        metadata = tuple(getattr(table, field)[row] for field in fields)
        exact.setdefault((gene, mode, dose, *metadata), []).append(table.target[row])
        action_only.setdefault((gene, mode, dose), []).append(table.target[row])
        gene_only.setdefault(gene, []).append(table.target[row])
    exact_mean = {key: np.mean(values, axis=0) for key, values in exact.items()}
    action_mean = {key: np.mean(values, axis=0) for key, values in action_only.items()}
    gene_mean = {key: np.mean(values, axis=0) for key, values in gene_only.items()}
    zero = np.zeros(table.target.shape[1], dtype="float64")
    exact_actions = action_fallback_actions = gene_fallback_actions = missing_actions = 0
    predictions = []
    for row in split.test:
        metadata = tuple(getattr(table, field)[row] for field in fields)
        prediction = zero.copy()
        valid = table.actions[row] >= 0
        modes = np.repeat("unspecified", valid.sum()) if table.action_modes is None else table.action_modes[row][valid]
        doses = np.ones(valid.sum()) if table.action_doses is None else table.action_doses[row][valid]
        for gene, mode, dose in zip(table.actions[row][valid], modes, doses):
            key = (int(gene), mode, float(dose), *metadata)
            if key in exact_mean:
                prediction += exact_mean[key]
                exact_actions += 1
            elif (int(gene), mode, float(dose)) in action_mean:
                prediction += action_mean[(int(gene), mode, float(dose))]
                action_fallback_actions += 1
            elif int(gene) in gene_mean:
                prediction += gene_mean[int(gene)]
                gene_fallback_actions += 1
            else:
                missing_actions += 1
        predictions.append(prediction)
    total = exact_actions + action_fallback_actions + gene_fallback_actions + missing_actions
    audit = {
        "matched_fields": list(fields),
        "training_singletons": int(len(singleton)),
        "action_coverage": 0.0 if not total else (exact_actions + action_fallback_actions + gene_fallback_actions) / total,
        "exact_actions": exact_actions,
        "action_fallback_actions": action_fallback_actions,
        "gene_fallback_actions": gene_fallback_actions,
        "missing_actions": missing_actions,
    }
    return np.asarray(predictions), audit


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype="float64")
    prediction = np.asarray(prediction, dtype="float64")
    if target.shape != prediction.shape or target.ndim != 2 or not len(target):
        raise ValueError("target and prediction must be non-empty arrays with identical shapes")
    error = prediction - target
    absolute = np.abs(error)
    huber = np.where(absolute <= 1.0, 0.5 * error**2, absolute - 0.5).mean()
    norms = np.linalg.norm(target, axis=1) * np.linalg.norm(prediction, axis=1)
    valid = norms > 0
    cosine = np.sum(target[valid] * prediction[valid], axis=1) / norms[valid]
    flat_target = target.ravel()
    flat_prediction = prediction.ravel()
    pearson = np.nan
    if flat_target.std() > 0 and flat_prediction.std() > 0:
        pearson = np.corrcoef(flat_target, flat_prediction)[0, 1]
    return {
        "huber": float(huber),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mean_cosine": float(cosine.mean()) if len(cosine) else float("nan"),
        "pearson": float(pearson),
    }
