"""Target-based exposure rules apply before any outcome-fitted operation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

HUMAN = 9606
KINDS = {"rna", "protein", "fitness", "interaction", "sl"}


@dataclass(frozen=True)
class Action:
    gene: str
    mechanism: str
    method: str
    allele: str = "unknown"
    dose_molar: float | None = None
    time_hours: float | None = None
    temperature_celsius: float | None = None

    def __post_init__(self):
        if not self.gene or not self.mechanism or not self.method or not self.allele:
            raise ValueError("Action identity and mechanism must be explicit")
        for value in (self.dose_molar, self.time_hours, self.temperature_celsius):
            if value is not None and not math.isfinite(value):
                raise ValueError("Nonfinite action metadata")
        if any(v is not None and v < 0 for v in (self.dose_molar, self.time_hours)):
            raise ValueError("Negative dose or duration")

    @property
    def numeric(self):
        values = (self.dose_molar, self.time_hours, self.temperature_celsius)
        return tuple(
            0.0 if v is None else math.log1p(v) if i < 2 else v / 100.0
            for i, v in enumerate(values)
        ), tuple(v is not None for v in values)


def canonical_pair(a, b):
    if not a or not b or a == b:
        raise ValueError("SL pairs require two distinct resolved genes")
    return tuple(sorted((a, b)))


@dataclass(frozen=True)
class Record:
    record_id: str
    source_id: str
    study_id: str
    experiment_id: str
    replicate_id: str
    taxon: int
    targets: tuple[str, ...]
    context: str
    assay: str
    kind: str
    units: str
    value: float
    query_gene: str | None
    mechanism: str
    method: str
    label_scope: str
    role: str
    license: str
    raw_object: str
    raw_locator: str
    lineage: tuple[str, ...] = ()
    actions: tuple[Action, ...] = ()

    def __post_init__(self):
        if self.kind not in KINDS or not math.isfinite(self.value):
            raise ValueError("Invalid measurement")
        if self.role not in {"quantitative", "sl_label", "quarantine"}:
            raise ValueError("Unknown admission role")
        if self.kind == "sl" and (
            self.taxon != HUMAN or len(self.targets) != 2 or self.value not in (0, 1)
        ):
            raise ValueError("SL requires a binary human pair label")
        if self.role != "quarantine" and (
            (self.kind == "sl") != (self.role == "sl_label")
        ):
            raise ValueError(
                "Quantitative measurements and curated labels must stay distinct"
            )
        if tuple(sorted(set(self.targets))) != self.targets:
            raise ValueError("Targets must be a canonical set")
        if self.actions and tuple(a.gene for a in self.actions) != self.targets:
            raise ValueError("Action metadata must match sorted intervention targets")
        genes = self.targets + ((self.query_gene,) if self.query_gene else ())
        if any(not gene.startswith(str(self.taxon) + ":") for gene in genes):
            raise ValueError("Genes must retain their native species")
        required = (
            self.record_id,
            self.source_id,
            self.study_id,
            self.experiment_id,
            self.context,
            self.assay,
            self.units,
            self.mechanism,
            self.method,
            self.label_scope,
            self.license,
            self.raw_object,
            self.raw_locator,
        )
        if not all(required):
            raise ValueError(
                "Missing provenance or semantics; use explicit unknown metadata"
            )

    @property
    def condition(self):
        return (
            self.study_id,
            self.taxon,
            self.context,
            self.assay,
            self.targets,
            self.mechanism,
            self.method,
            tuple(tuple(asdict(a).values()) for a in self.actions),
        )

    @property
    def unit(self):
        return (self.study_id, self.experiment_id, self.replicate_id)

    @property
    def measurement_key(self):
        return (self.unit, self.condition, self.kind, self.units, self.query_gene)

    @classmethod
    def from_dict(cls, value):
        return cls(
            **{
                **value,
                "targets": tuple(value["targets"]),
                "lineage": tuple(value.get("lineage", ())),
                "actions": tuple(
                    Action(**a) if isinstance(a, dict) else a
                    for a in value.get("actions", ())
                ),
            }
        )


@dataclass(frozen=True)
class Fold:
    name: str
    outer_held: frozenset[str]
    inner_held: frozenset[str] = frozenset()

    def __post_init__(self):
        if not self.outer_held or self.outer_held & self.inner_held:
            raise ValueError(
                "Outer holdout must be nonempty and disjoint from inner holdout"
            )
        if any(not x.startswith("9606:") for x in self.forbidden):
            raise ValueError("Strict human CV3 uses resolved human gene identities")

    @property
    def forbidden(self):
        return self.outer_held | self.inner_held

    def permits(self, row, stage):
        if stage not in {"pretrain", "adapt"}:
            raise ValueError("Unknown fitting stage")
        if row.role == "quarantine":
            return False
        if row.taxon == HUMAN and self.forbidden.intersection(row.targets):
            return False
        return row.role == "quantitative" if stage == "pretrain" else row.taxon == HUMAN

    def evaluation_pairs(self, rows, *, inner):
        held = self.inner_held if inner else self.outer_held
        if not held:
            raise ValueError("Requested holdout is empty")
        return [
            r
            for r in rows
            if r.kind == "sl" and r.role == "sl_label" and set(r.targets) <= held
        ]


def fitting_view(rows, fold, stage):
    rows = list(rows)
    selected = [r for r in rows if fold.permits(r, stage)]
    ids = {r.record_id for r in rows}
    if len(ids) != len(rows):
        raise ValueError("Duplicate record IDs")
    kept = {r.record_id for r in selected}
    # A derived measurement cannot outlive one of its source measurements.
    if any(set(r.lineage) - kept for r in selected):
        raise ValueError("Derived measurements have lineage outside the fitting view")
    receipt = {
        "fold": fold.name,
        "stage": stage,
        "forbidden_genes": sorted(fold.forbidden),
        "input_rows": len(rows),
        "fitting_rows": len(selected),
        "human_intervention_genes": sorted(
            {g for r in selected if r.taxon == HUMAN for g in r.targets}
        ),
        "record_ids_sha256": hashlib.sha256(
            "\n".join(sorted(kept)).encode()
        ).hexdigest(),
    }
    if fold.forbidden.intersection(receipt["human_intervention_genes"]):
        raise AssertionError("Forbidden human fitting exposure")
    return selected, receipt


def resolve_benchmark(rows, alias_map):
    """Resolve every pair exactly; the caller receives no silently reduced suite."""
    resolved, missing = [], []
    for i, row in enumerate(rows):
        try:
            pair = canonical_pair(alias_map[row["a"]], alias_map[row["b"]])
        except (KeyError, ValueError):
            missing.append({"row": i, "a": row["a"], "b": row["b"]})
        else:
            resolved.append({**row, "a": pair[0], "b": pair[1]})
    if missing:
        raise ValueError("Unresolved benchmark pairs: " + json.dumps(missing[:100]))
    return resolved


def deduplicate(rows):
    """Merge exact repeat publications, reject disagreeing measurements explicitly."""
    kept, aliases = {}, {}
    for row in rows:
        key = row.measurement_key
        if key in kept:
            previous = kept[key]
            if previous.value != row.value:
                raise ValueError("Conflicting duplicate measurement: " + row.record_id)
            aliases[row.record_id] = previous.record_id
        else:
            kept[key] = row
    return list(kept.values()), aliases


def write_jsonl(path, rows):
    with open(path, "w") as file:
        for row in rows:
            file.write(json.dumps(asdict(row), sort_keys=True, allow_nan=False) + "\n")
