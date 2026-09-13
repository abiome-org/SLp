"""Versioned static features, fitting-only scaling and resumable condition draws."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from records import fitting_view

KINDS = {
    name: i for i, name in enumerate(("rna", "protein", "fitness", "interaction", "sl"))
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


class Features:
    def __init__(self, directory, *, manifest_name="manifest.json"):
        root = Path(directory)
        if Path(manifest_name).name != manifest_name:
            raise ValueError("Invalid feature manifest filename")
        self.manifest = json.loads((root / manifest_name).read_text())
        if self.manifest.get("schema") != "slp.static-features/v2":
            raise ValueError("Require a provenance-bearing r2 static feature manifest")
        for name, expected in self.manifest["files"].items():
            if Path(name).name != name or digest(root / name) != expected:
                raise ValueError("Feature file hash or path mismatch")
        for name in ("genes.json", "sequence.npy", "annotation.npy", "known.npy"):
            if name not in self.manifest["files"]:
                raise ValueError("Unpinned feature file")
        if self.manifest.get("fitted_human_intervention_genes") != []:
            raise ValueError("Static features may not use human perturbation fitting")
        genes = json.loads((root / "genes.json").read_text())
        if len(set(genes)) != len(genes):
            raise ValueError("Duplicate canonical feature identity")
        self.index = {gene: i for i, gene in enumerate(genes)}
        self.sequence = np.load(
            root / "sequence.npy", mmap_mode="r", allow_pickle=False
        )
        self.annotation = np.load(
            root / "annotation.npy", mmap_mode="r", allow_pickle=False
        )
        self.known = np.load(root / "known.npy", mmap_mode="r", allow_pickle=False)
        if any(
            len(x) != len(genes) for x in (self.sequence, self.annotation, self.known)
        ) or self.known.shape != (len(genes), 2):
            raise ValueError("Feature roster/shape mismatch")

    def get(self, gene):
        if gene not in self.index:
            # Missing sequence is represented by an explicit manifest row, not
            # by accidentally treating an unresolved identity as an unknown gene.
            raise ValueError("Gene absent from admitted feature manifest: " + gene)
        i = self.index[gene]
        return self.sequence[i], self.annotation[i], self.known[i]


class Scales:
    def __init__(self, rows):
        stats = {}
        for row in rows:
            if row.kind == "sl":
                continue
            key = (row.assay, row.kind, row.units)
            n, mean, m2 = stats.get(key, (0, 0.0, 0.0))
            n += 1
            delta = row.value - mean
            mean += delta / n
            stats[key] = (n, mean, m2 + delta * (row.value - mean))
        self.values = {
            key: (mean, max(math.sqrt(m2 / max(n - 1, 1)), 1e-6))
            for key, (n, mean, m2) in stats.items()
        }

    def transform(self, row):
        if row.kind == "sl":
            return row.value
        key = (row.assay, row.kind, row.units)
        if key not in self.values:
            raise ValueError("No fitting-only scale for assay/kind/units")
        mean, scale = self.values[key]
        return (row.value - mean) / scale


class ConditionSampler:
    """Conditions shuffle without replacement; replicates shuffle inside them.

    A family is source × measurement kind. Explicit family weights are required;
    counts determine relative source weights within a kind, never implicitly the
    allocation between measurement kinds. Budgets count condition draws.
    """

    def __init__(
        self, rows, kind_weights, *, temperature=0.7, seed=731, max_cycles=None
    ):
        self.rows = rows
        grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for i, row in enumerate(rows):
            family = (row.source_id, row.kind)
            grouped[family][row.condition][row.unit].append(i)
        self.families = sorted(grouped)
        if not self.families:
            raise ValueError("Empty fitting view")
        if set(kind_weights) != {k for _, k in self.families} or any(
            v <= 0 for v in kind_weights.values()
        ):
            raise ValueError("Explicit positive weight required for each admitted kind")
        self.conditions = [
            [list(units.values()) for units in grouped[f].values()]
            for f in self.families
        ]
        counts = [len(c) ** temperature for c in self.conditions]
        totals = {
            kind: sum(counts[i] for i, (_, k) in enumerate(self.families) if k == kind)
            for kind in kind_weights
        }
        self.weights = [
            kind_weights[k] * counts[i] / totals[k]
            for i, (_, k) in enumerate(self.families)
        ]
        self.max_cycles = max_cycles
        self.random = random.Random(seed)
        self.queues = [[] for _ in self.families]
        self.draws = [0 for _ in self.families]
        self.replica_queues = {}
        self.signature = hashlib.sha256(
            json.dumps(
                {
                    "records": [r.record_id for r in rows],
                    "weights": self.weights,
                    "max_cycles": max_cycles,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def draw(self, max_queries=128):
        weights = [
            w
            if self.max_cycles is None
            or self.draws[i] < self.max_cycles * len(self.conditions[i])
            else 0.0
            for i, w in enumerate(self.weights)
        ]
        if not any(weights):
            raise StopIteration("Condition repeat budget exhausted")
        family = self.random.choices(range(len(weights)), weights=weights, k=1)[0]
        if not self.queues[family]:
            self.queues[family] = list(range(len(self.conditions[family])))
            self.random.shuffle(self.queues[family])
        condition = self.queues[family].pop()
        key = (family, condition)
        if not self.replica_queues.get(key):
            self.replica_queues[key] = list(
                range(len(self.conditions[family][condition]))
            )
            self.random.shuffle(self.replica_queues[key])
        self.draws[family] += 1
        unit = self.conditions[family][condition][self.replica_queues[key].pop()]
        selected = self.random.sample(unit, min(max_queries, len(unit)))
        return [self.rows[i] for i in selected]

    def state_dict(self):
        return {
            "signature": self.signature,
            "random": self.random.getstate(),
            "queues": self.queues,
            "draws": self.draws,
            "replica_queues": self.replica_queues,
        }

    def load_state_dict(self, state):
        if state["signature"] != self.signature:
            raise ValueError("Sampler corpus/config changed on resume")
        self.random.setstate(state["random"])
        self.queues, self.draws, self.replica_queues = (
            state["queues"],
            state["draws"],
            state["replica_queues"],
        )


class Corpus:
    def __init__(self, rows, fold, stage):
        self.rows, self.exposure = fitting_view(rows, fold, stage)
        if not self.rows:
            raise ValueError("Fold has no fitting data")
        self.scales = Scales(self.rows)


class BatchBuilder:
    """One observed measurement per sampled condition/replicate.

    Basal observations must be supplied separately with an explicit control-only
    provenance manifest. Absence is encoded, never imputed from outcome rows.
    """

    def __init__(self, features, scales, vocabulary, config, basal=None):
        self.features, self.scales, self.vocab, self.config = (
            features,
            scales,
            vocabulary,
            config,
        )
        self.basal = basal or {}
        for context, item in self.basal.items():
            if item.get("role") not in {
                "observational",
                "unperturbed_control",
            } or not item.get("provenance"):
                raise ValueError(
                    "Basal input lacks permitted control/observational provenance: "
                    + context
                )

    def __call__(self, units):
        if not units or any(not unit for unit in units):
            raise ValueError("Nonempty experimental units required")
        for unit in units:
            if any(
                r.unit != unit[0].unit or r.condition != unit[0].condition for r in unit
            ):
                raise ValueError(
                    "An example must belong to one experimental unit and condition"
                )
        rows = [unit[0] for unit in units]
        c, n = self.config, len(rows)
        nq = max(len(unit) for unit in units)
        na = max(1, max(len(r.targets) for r in rows))
        no = max(
            1,
            max(
                len(self.basal.get(r.context, {}).get("observations", [])) for r in rows
            ),
        )
        b = {
            "context": torch.zeros(n, c.context_dim),
            "context_known": torch.zeros(n, dtype=torch.bool),
        }
        for key in ("assay", "taxon", "scope"):
            b[key] = torch.zeros(n, dtype=torch.long)
        for prefix, count in (("observation", no), ("action", na), ("query", nq)):
            b[prefix + "_sequence"] = torch.zeros(n, count, c.sequence_dim)
            b[prefix + "_annotation"] = torch.zeros(n, count, c.annotation_dim)
            b[prefix + "_known"] = torch.zeros(n, count, 2, dtype=torch.bool)
            b[prefix + "_mask"] = torch.zeros(n, count, dtype=torch.bool)
        b.update(
            {
                "observation_values": torch.zeros(n, no),
                "observation_kind": torch.zeros(n, no, dtype=torch.long),
                "action_values": torch.zeros(n, na, 3),
                "action_numeric_known": torch.zeros(n, na, 3, dtype=torch.bool),
                "action_mechanism": torch.zeros(n, na, dtype=torch.long),
                "action_method": torch.zeros(n, na, dtype=torch.long),
                "query_kind": torch.zeros(n, nq, dtype=torch.long),
                "target": torch.zeros(n, nq),
            }
        )
        for i, row in enumerate(rows):
            for key, value in (
                ("assay", row.assay),
                ("taxon", str(row.taxon)),
                ("scope", row.label_scope),
            ):
                b[key][i] = self.vocab[key][value]
            basal = self.basal.get(row.context, {})
            if "context" in basal:
                b["context"][i] = torch.tensor(basal["context"])
                b["context_known"][i] = True
            for j, observation in enumerate(basal.get("observations", [])):
                self._gene(b, "observation", i, j, observation["gene"])
                b["observation_values"][i, j] = observation["value"]
                b["observation_kind"][i, j] = KINDS[observation["kind"]]
            for j, gene in enumerate(row.targets):
                self._gene(b, "action", i, j, gene)
                action = row.actions[j] if row.actions else None
                b["action_mechanism"][i, j] = self.vocab["mechanism"][
                    action.mechanism if action else row.mechanism
                ]
                b["action_method"][i, j] = self.vocab["method"][
                    action.method if action else row.method
                ]
                if action:
                    values, known = action.numeric
                    b["action_values"][i, j] = torch.tensor(values)
                    b["action_numeric_known"][i, j] = torch.tensor(known)
            for j, measurement in enumerate(units[i]):
                if measurement.query_gene:
                    self._gene(b, "query", i, j, measurement.query_gene)
                b["query_mask"][i, j] = True
                b["query_kind"][i, j] = KINDS[measurement.kind]
                b["target"][i, j] = self.scales.transform(measurement)
        return b

    def _gene(self, batch, prefix, i, j, gene):
        seq, ann, known = self.features.get(gene)
        batch[prefix + "_sequence"][i, j] = torch.tensor(np.array(seq))
        batch[prefix + "_annotation"][i, j] = torch.tensor(np.array(ann))
        batch[prefix + "_known"][i, j] = torch.tensor(np.array(known))
        batch[prefix + "_mask"][i, j] = True
