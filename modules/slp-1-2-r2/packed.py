"""Memory-mapped real corpus, exact target exclusions and grouped sampling.

Cloudflare performs source preparation. This module verifies and materializes
those immutable arrays on the GPU cache disk, then constructs fold indices.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from data import digest, KINDS


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def materialize(manifests, directory, genes):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    expected = hashlib.sha256(encode(genes).encode()).hexdigest()
    documents = [json.loads(Path(p).read_text()) for p in manifests]
    identity = {str(p): digest(p) for p in manifests}
    signature = hashlib.sha256(encode(identity).encode()).hexdigest()
    receipt_path = root / "cache.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["source_signature"] != signature:
            raise ValueError("Corpus cache identity mismatch")
        if digest(root / "measurements.npy") != receipt["array_sha256"]:
            raise ValueError("Corpus cache checksum mismatch")
        return receipt_path
    total = sum(m["rows"] for m in documents)
    at = 0
    target = None
    templates = []
    for source_path, manifest in zip(manifests, documents):
        if (
            manifest["schema"] != "slp.packed-measurements/v1"
            or manifest["genes_sha256"] != expected
        ):
            raise ValueError("Packed corpus schema/static identity mismatch")
        if manifest["outcome_fitted_transforms"] != []:
            raise ValueError("Fitted transforms must follow fold filtering")
        offset = len(templates)
        templates.extend(manifest["templates"])
        for shard in manifest["shards"]:
            file = Path(source_path).parent / shard["name"]
            if file.name != shard["name"] or digest(file) != shard["sha256"]:
                raise ValueError("Packed shard checksum mismatch")
            raw = gzip.decompress(file.read_bytes())
            if hashlib.sha256(raw).hexdigest() != shard["npy_sha256"]:
                raise ValueError("Unpacked array checksum mismatch")
            rows = np.load(io.BytesIO(raw), allow_pickle=False)
            required = {
                "template",
                "targets",
                "mechanism",
                "method",
                "numeric",
                "known",
                "query",
                "value",
                "condition",
                "unit",
                "native_row",
            }
            if (
                set(rows.dtype.names or ()) != required
                or rows["targets"].shape != (len(rows), 4)
                or rows["numeric"].shape != (len(rows), 4, 3)
            ):
                raise ValueError("Unexpected packed field contract")
            if not np.isfinite(rows["value"]).all() or len(rows) != shard["rows"]:
                raise ValueError("Invalid packed measurement count/value")
            if rows["template"].max(initial=0) >= len(manifest["templates"]):
                raise ValueError("Invalid template index")
            if (
                rows["targets"].max(initial=-1) >= len(genes)
                or rows["targets"].min(initial=-1) < -1
            ):
                raise ValueError("Invalid action identity")
            if (
                rows["query"].max(initial=-1) >= len(genes)
                or rows["query"].min(initial=-1) < -1
            ):
                raise ValueError("Invalid query identity")
            if target is None:
                target = np.lib.format.open_memmap(
                    root / "measurements.tmp.npy",
                    mode="w+",
                    dtype=rows.dtype,
                    shape=(total,),
                )
            if rows.dtype != target.dtype:
                raise ValueError("Packed field types differ")
            rows["template"] += offset
            target[at : at + len(rows)] = rows
            at += len(rows)
    if at != total or target is None:
        raise ValueError("Incomplete/empty corpus materialization")
    target.flush()
    del target
    (root / "measurements.tmp.npy").replace(root / "measurements.npy")
    receipt = {
        "schema": "slp.packed-cache/v1",
        "source_signature": signature,
        "source_manifests": identity,
        "array_sha256": digest(root / "measurements.npy"),
        "templates": templates,
        "rows": total,
        "genes_sha256": expected,
    }
    receipt_path.write_text(encode(receipt))
    return receipt_path


class View:
    def __init__(self, cache, genes, fold, stage, *, verify=True, taxa=None):
        if stage not in ("pretrain", "adapt"):
            raise ValueError("Unknown fitting stage")
        self.root = Path(cache).parent
        self.manifest = json.loads(Path(cache).read_text())
        self.rows = np.load(
            self.root / "measurements.npy", allow_pickle=False, mmap_mode="r"
        )
        if (
            verify
            and digest(self.root / "measurements.npy") != self.manifest["array_sha256"]
        ):
            raise ValueError("Changed corpus cache")
        if (
            hashlib.sha256(encode(genes).encode()).hexdigest()
            != self.manifest["genes_sha256"]
        ):
            raise ValueError("Changed static gene identity")
        self.templates = self.manifest["templates"]
        human = np.array([t["taxon"] == 9606 for t in self.templates])
        admitted = np.array([t.get("training_allowed") is True for t in self.templates])
        if taxa is not None:
            admitted &= np.array([t["taxon"] in taxa for t in self.templates])
        stages = np.array(
            [
                t["kind"] != "sl" if stage == "pretrain" else t["taxon"] == 9606
                for t in self.templates
            ]
        )
        forbidden_gene = np.array([g in fold.forbidden for g in genes] + [False])
        selected = []
        exposures = 0
        by_template = np.zeros(len(self.templates), np.int64)
        for lo in range(0, len(self.rows), 262144):
            part = self.rows[lo : lo + 262144]
            tid = part["template"]
            forbidden = forbidden_gene[part["targets"]].any(1) & human[tid]
            keep = admitted[tid] & stages[tid] & ~forbidden
            exposures += int((forbidden & keep).sum())
            selected.append(np.flatnonzero(keep).astype(np.int64) + lo)
            by_template += np.bincount(tid[keep], minlength=len(by_template))
        eligible = np.concatenate(selected)
        del selected
        if exposures:
            raise AssertionError("Forbidden human intervention exposure")
        families = sorted(
            {
                (t["source"], t["kind"])
                for i, t in enumerate(self.templates)
                if by_template[i] > 0
            }
        )
        if not families:
            raise ValueError("No admitted fitting families")
        self.families = []
        template_family = np.array(
            [
                families.index((t["source"], t["kind"]))
                if (t["source"], t["kind"]) in families
                else -1
                for t in self.templates
            ]
        )
        family_for_rows = template_family[self.rows["template"][eligible]]
        for family, (source, kind) in enumerate(families):
            indices = eligible[family_for_rows == family]
            # Two keys keep all coordinates of each experimental unit together.
            order = np.lexsort(
                (self.rows["unit"][indices], self.rows["condition"][indices])
            )
            indices = indices[order]
            conditions = self.rows["condition"][indices]
            boundaries = np.r_[
                0, np.flatnonzero(conditions[1:] != conditions[:-1]) + 1, len(indices)
            ]
            self.families.append(
                {
                    "source": source,
                    "kind": kind,
                    "indices": indices,
                    "boundaries": boundaries,
                    "conditions": len(boundaries) - 1,
                }
            )
        self.receipt = {
            "fold": fold.name,
            "stage": stage,
            "forbidden_genes": sorted(fold.forbidden),
            "forbidden_human_exposures": exposures,
            "input_rows": len(self.rows),
            "fitting_rows": len(eligible),
            "quarantined_templates": [
                i for i, allowed in enumerate(admitted) if not allowed
            ],
            "by_template": by_template.tolist(),
            "cache_sha256": digest(cache),
            "families": [
                {k: v for k, v in f.items() if k not in ("indices", "boundaries")}
                for f in self.families
            ],
        }
        self.signature = hashlib.sha256(encode(self.receipt).encode()).hexdigest()
        self.scales = self._scales(eligible)

    def _scales(self, eligible):
        n = np.zeros(len(self.templates), np.int64)
        mean = np.zeros(len(n))
        m2 = np.zeros(len(n))
        for lo in range(0, len(eligible), 262144):
            part = self.rows[eligible[lo : lo + 262144]]
            for tid in np.unique(part["template"]):
                if self.templates[int(tid)]["kind"] == "sl":
                    continue
                values = part["value"][part["template"] == tid].astype(np.float64)
                count = len(values)
                avg = values.mean()
                square = np.square(values - avg).sum()
                delta = avg - mean[tid]
                total = n[tid] + count
                m2[tid] += square + delta * delta * n[tid] * count / total
                mean[tid] += delta * count / total
                n[tid] = total
        # Pool by assay/kind/units, not by storage shard or individual context.
        groups = {}
        for i, t in enumerate(self.templates):
            if not n[i]:
                continue
            key = (t["assay"], t["kind"], t["units"])
            old_n, old_mean, old_m2 = groups.get(key, (0, 0.0, 0.0))
            total = old_n + n[i]
            delta = mean[i] - old_mean
            groups[key] = (
                total,
                old_mean + delta * n[i] / total,
                old_m2 + m2[i] + delta * delta * old_n * n[i] / total,
            )
        result = []
        for t in self.templates:
            key = (t["assay"], t["kind"], t["units"])
            if t["kind"] == "sl":
                result.append((0.0, 1.0))
                continue
            if key not in groups:
                result.append((float("nan"), float("nan")))
                continue
            count, avg, square = groups[key]
            result.append(
                (
                    float(avg),
                    max(float(np.sqrt(max(square, 0) / max(count - 1, 1))), 1e-6),
                )
            )
        return np.asarray(result, np.float32)


class Sampler:
    def __init__(
        self, view, kind_weights, *, seed=731, temperature=0.7, max_cycles=None
    ):
        self.view = view
        self.random = random.Random(seed)
        self.max_cycles = max_cycles
        if set(kind_weights) != {f["kind"] for f in view.families} or any(
            v <= 0 for v in kind_weights.values()
        ):
            raise ValueError(
                "Every admitted kind requires a positive explicit allocation"
            )
        self.counts = [f["conditions"] for f in view.families]
        totals = {
            k: sum(
                f["conditions"] ** temperature for f in view.families if f["kind"] == k
            )
            for k in kind_weights
        }
        self.weights = [
            kind_weights[f["kind"]] * f["conditions"] ** temperature / totals[f["kind"]]
            for f in view.families
        ]
        self.positions = [0] * len(self.counts)
        self.cycles = [0] * len(self.counts)
        self.permutations = [self._permutation(n) for n in self.counts]
        self.signature = hashlib.sha256(
            encode(
                {
                    "view": view.signature,
                    "weights": self.weights,
                    "max_cycles": max_cycles,
                }
            ).encode()
        ).hexdigest()

    def _permutation(self, n):
        a = self.random.randrange(1, n) if n > 1 else 1
        while math.gcd(a, n) != 1:
            a = self.random.randrange(1, n)
        return a, self.random.randrange(n)

    def draw(self, max_queries=128):
        weights = [
            w if self.max_cycles is None or self.cycles[i] < self.max_cycles else 0
            for i, w in enumerate(self.weights)
        ]
        if not any(weights):
            raise StopIteration("Condition repeat budget exhausted")
        family = self.random.choices(range(len(weights)), weights=weights, k=1)[0]
        return self.draw_family(family, max_queries)

    def draw_family(self, family, max_queries=128):
        if self.max_cycles is not None and self.cycles[family] >= self.max_cycles:
            raise StopIteration("Condition repeat budget exhausted")
        a, b = self.permutations[family]
        ordinal = (a * self.positions[family] + b) % self.counts[family]
        self.positions[family] += 1
        if self.positions[family] == self.counts[family]:
            self.cycles[family] += 1
            self.positions[family] = 0
            self.permutations[family] = self._permutation(self.counts[family])
        f = self.view.families[family]
        lo, hi = f["boundaries"][ordinal : ordinal + 2]
        indices = f["indices"][lo:hi]
        units = self.view.rows["unit"][indices]
        boundaries = np.r_[0, np.flatnonzero(units[1:] != units[:-1]) + 1, len(indices)]
        unit = self.random.randrange(len(boundaries) - 1)
        u0, u1 = boundaries[unit : unit + 2]
        selected = indices[
            [
                u0 + j
                for j in self.random.sample(range(u1 - u0), min(max_queries, u1 - u0))
            ]
        ]
        return (
            self.view.measurements(selected, self.random, max_queries)
            if hasattr(self.view, "measurements")
            else self.view.rows[selected].copy()
        )

    def state_dict(self):
        return {
            "signature": self.signature,
            "random": self.random.getstate(),
            "positions": self.positions,
            "cycles": self.cycles,
            "permutations": self.permutations,
        }

    def load_state_dict(self, state):
        if state["signature"] != self.signature:
            raise ValueError("Changed corpus/fold/sampling recipe")
        self.random.setstate(state["random"])
        self.positions = list(state["positions"])
        self.cycles = list(state["cycles"])
        self.permutations = list(state["permutations"])


class Builder:
    def __init__(self, features, templates, scales, vocabulary, config, basal=None):
        self.features, self.templates, self.scales, self.vocab, self.config = (
            features,
            templates,
            scales,
            vocabulary,
            config,
        )
        self.basal = basal or {}
        for b in self.basal.values():
            if b.get("role") not in (
                "observational",
                "unperturbed_control",
            ) or not b.get("provenance"):
                raise ValueError("Basal input has no permitted provenance")

    def __call__(self, units):
        if not units or any(not len(u) for u in units):
            raise ValueError("Empty experimental unit")
        for unit in units:
            if (unit["unit"] != unit["unit"][0]).any() or (
                unit["condition"] != unit["condition"][0]
            ).any():
                raise ValueError("Mixed experimental unit or conditioning")
        first = np.array([u[0] for u in units])
        n = len(units)
        c = self.config
        templates = [self.templates[int(t)] for t in first["template"]]
        na = max(1, int((first["targets"] >= 0).sum(1).max()))
        nq = max(len(u) for u in units)
        no = max(
            1,
            max(
                len(self.basal.get(t["context"], {}).get("observations", []))
                for t in templates
            ),
        )
        batch = {
            "context": np.zeros((n, c.context_dim), np.float32),
            "context_known": np.zeros(n, bool),
        }
        ids = {
            "action": first["targets"][:, :na].copy(),
            "query": np.full((n, nq), -1, np.int32),
            "observation": np.full((n, no), -1, np.int32),
        }
        batch.update(
            {
                "action_values": first["numeric"][:, :na].copy(),
                "action_numeric_known": first["known"][:, :na].copy(),
                "action_mechanism": first["mechanism"][:, :na].astype(np.int64),
                "action_method": first["method"][:, :na].astype(np.int64),
                "query_kind": np.zeros((n, nq), np.int64),
                "query_mask": np.zeros((n, nq), bool),
                "target": np.zeros((n, nq), np.float32),
                "observation_values": np.zeros((n, no), np.float32),
                "observation_kind": np.zeros((n, no), np.int64),
            }
        )
        for key in ("assay", "taxon", "scope"):
            batch[key] = np.array(
                [self.vocab[key][str(t[key])] for t in templates], np.int64
            )
        for i, (unit, t) in enumerate(zip(units, templates)):
            q = len(unit)
            ids["query"][i, :q] = unit["query"]
            batch["query_mask"][i, :q] = True
            batch["query_kind"][i, :q] = [
                KINDS[self.templates[int(j)]["kind"]] for j in unit["template"]
            ]
            scale = self.scales[unit["template"]]
            if not np.isfinite(scale).all():
                raise ValueError("No fitting-only target scale")
            batch["target"][i, :q] = (unit["value"] - scale[:, 0]) / scale[:, 1]
            basal = self.basal.get(t["context"], {})
            if "context" in basal:
                batch["context"][i] = basal["context"]
                batch["context_known"][i] = True
            for j, obs in enumerate(basal.get("observations", [])):
                ids["observation"][i, j] = self.features.index[obs["gene"]]
                batch["observation_values"][i, j] = obs["value"]
                batch["observation_kind"][i, j] = KINDS[obs["kind"]]
        for prefix, index in ids.items():
            mask = index >= 0
            safe = np.maximum(index, 0)
            batch[prefix + "_sequence"] = np.asarray(
                self.features.sequence[safe]
            ).copy()
            batch[prefix + "_annotation"] = np.asarray(
                self.features.annotation[safe]
            ).copy()
            batch[prefix + "_known"] = (
                np.asarray(self.features.known[safe]).copy() & mask[..., None]
            )
            if prefix != "query":
                batch[prefix + "_mask"] = mask
        return {k: torch.from_numpy(v) for k, v in batch.items()}


def vocabulary(templates):
    return {
        key: {
            value: i for i, value in enumerate(sorted({str(t[key]) for t in templates}))
        }
        for key in ("assay", "taxon", "scope")
    }
