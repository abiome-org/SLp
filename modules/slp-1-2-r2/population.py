"""Dense molecular endpoints with one metadata record per experimental unit."""

import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from cloud_io import get_json, request
from data import digest
from packed import encode, Sampler


def fetch(job, directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    manifest = get_json(job, "population-manifest.json")
    for name, spec in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("Invalid population file path")
        path = root / name
        if path.exists() and digest(path) == spec["sha256"]:
            continue
        sha = hashlib.sha256()
        size = 0
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as file:
            for part in spec["parts"]:
                with request(job, part["name"]) as response:
                    body = response.read(32 * 1024 * 1024 + 1)
                if (
                    len(body) != part["bytes"]
                    or hashlib.sha256(body).hexdigest() != part["sha256"]
                ):
                    raise ValueError("Population part checksum mismatch")
                chunk = gzip.decompress(body)
                sha.update(chunk)
                size += len(chunk)
                file.write(chunk)
        if size != spec["bytes"] or sha.hexdigest() != spec["sha256"]:
            raise ValueError("Population file checksum mismatch")
        tmp.replace(path)
    (root / "population-manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    with request(job, manifest["basal"]) as response:
        body = response.read(32 * 1024 * 1024 + 1)
    if hashlib.sha256(body).hexdigest() != manifest["basal_sha256"]:
        raise ValueError("Basal observation checksum mismatch")
    (root / "basal.json").write_bytes(body)
    return root / "population-manifest.json"


class View:
    def __init__(self, path, genes, fold, stage, *, verify=True, taxa=None):
        if stage not in ("pretrain", "adapt"):
            raise ValueError("Unknown fitting stage")
        self.root = Path(path).parent
        self.manifest = json.loads(Path(path).read_text())
        m = self.manifest
        if (
            m["schema"] != "slp.population-pack/v1"
            or m["outcome_fitted_transforms"] != []
        ):
            raise ValueError("Unadmitted molecular transforms/schema")
        if m["genes_sha256"] != hashlib.sha256(encode(genes).encode()).hexdigest():
            raise ValueError("Population/static gene identity mismatch")
        if verify:
            for name, spec in m["files"].items():
                if (
                    Path(name).name != name
                    or digest(self.root / name) != spec["sha256"]
                ):
                    raise ValueError("Population artifact checksum mismatch")
        self.rows = np.load(self.root / "units.npy", mmap_mode="r", allow_pickle=False)
        self.targets = np.load(
            self.root / "targets.npy", mmap_mode="r", allow_pickle=False
        )
        self.observed = np.load(
            self.root / "observed.npy", mmap_mode="r", allow_pickle=False
        )
        self.query = np.load(self.root / "query-indices.npy", allow_pickle=False)
        if (
            self.targets.shape != (m["units"], m["queries"])
            or self.observed.shape != self.targets.shape
            or self.query.shape != (m["queries"],)
            or len(self.rows) != m["units"]
        ):
            raise ValueError("Population axes disagree")
        if self.query.min(initial=0) < 0 or self.query.max(initial=0) >= len(genes):
            raise ValueError("Unresolved molecular output identity")
        if self.rows["targets"].min(initial=-1) < -1 or self.rows["targets"].max(
            initial=-1
        ) >= len(genes):
            raise ValueError("Unresolved molecular intervention identity")
        self.templates = m["templates"]
        tid = self.rows["template"]
        if tid.max(initial=0) >= len(self.templates):
            raise ValueError("Unknown molecular template")
        human = np.array([t["taxon"] == 9606 for t in self.templates])
        allowed = np.array(
            [
                t.get("training_allowed") is True and t["kind"] == "rna"
                for t in self.templates
            ]
        )
        held = np.array([g in fold.forbidden for g in genes] + [False])
        forbidden = held[self.rows["targets"]].any(1) & human[tid]
        keep = allowed[tid] & ~forbidden
        if taxa is not None:
            keep &= np.array([t["taxon"] in taxa for t in self.templates])[tid]
        if stage == "adapt":
            keep &= human[tid]
        valid = np.zeros(len(self.rows), bool)
        for lo in range(0, len(valid), 512):
            valid[lo : lo + 512] = self.observed[lo : lo + 512].any(1)
        keep &= valid
        eligible = np.flatnonzero(keep)
        if not len(eligible):
            raise ValueError("No admitted molecular fitting populations")
        families = sorted(
            {
                (self.templates[int(i)]["source"], "rna")
                for i in np.unique(tid[eligible])
            }
        )
        self.families = []
        for source, kind in families:
            indices = eligible[
                np.array(
                    [self.templates[int(i)]["source"] == source for i in tid[eligible]]
                )
            ]
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
            "manifest_sha256": digest(path),
            "forbidden_genes": sorted(fold.forbidden),
            "forbidden_human_exposures": int((forbidden & keep).sum()),
            "input_units": len(keep),
            "fitting_units": len(eligible),
            "empty_units": int((~valid).sum()),
            "families": [
                {k: v for k, v in f.items() if k not in ("indices", "boundaries")}
                for f in self.families
            ],
        }
        self.signature = hashlib.sha256(encode(self.receipt).encode()).hexdigest()
        self.scales = self._scales(eligible)

    def _scales(self, eligible):
        groups = {}
        for lo in range(0, len(eligible), 256):
            selected = eligible[lo : lo + 256]
            tid = self.rows["template"][selected]
            values = np.asarray(self.targets[selected], np.float64)
            observed = self.observed[selected]
            for t in np.unique(tid):
                spec = self.templates[int(t)]
                key = (spec["assay"], spec["kind"], spec["units"])
                x = values[tid == t][observed[tid == t]]
                if not np.isfinite(x).all():
                    raise ValueError("Nonfinite observed molecular value")
                n, avg, m2 = groups.get(key, (0, 0.0, 0.0))
                count = len(x)
                if not count:
                    continue
                mean = float(x.mean())
                delta = mean - avg
                total = n + count
                groups[key] = (
                    total,
                    avg + delta * count / total,
                    m2
                    + float(np.square(x - mean).sum())
                    + delta * delta * n * count / total,
                )
        result = []
        for t in self.templates:
            key = (t["assay"], t["kind"], t["units"])
            if key not in groups:
                result.append((float("nan"), float("nan")))
                continue
            n, mean, m2 = groups[key]
            result.append((mean, max(float(np.sqrt(m2 / max(n - 1, 1))), 1e-6)))
        return np.asarray(result, np.float32)

    def measurements(self, indices, rng, max_queries):
        if len(indices) != 1:
            raise ValueError("Dense molecular unit must have one metadata record")
        i = int(indices[0])
        available = np.flatnonzero(self.observed[i])
        selected = available[
            rng.sample(range(len(available)), min(max_queries, len(available)))
        ]
        rows = np.repeat(self.rows[[i]], len(selected))
        rows["query"] = self.query[selected]
        rows["value"] = self.targets[i, selected]
        return rows


class Mixture:
    """Temperature sampling across every admitted source/kind, independent of size in cells or coordinates."""

    def __init__(
        self, views, kind_weights, *, seed=731, temperature=0.7, max_cycles=None
    ):
        import random

        self.random = random.Random(seed)
        self.children = []
        self.offsets = []
        self.candidates = []
        self.templates = []
        scales = []
        for i, view in enumerate(views):
            kinds = {f["kind"] for f in view.families}
            self.children.append(
                Sampler(
                    view,
                    {k: 1.0 for k in kinds},
                    seed=seed + i,
                    temperature=temperature,
                    max_cycles=max_cycles,
                )
            )
            self.offsets.append(len(self.templates))
            self.templates.extend(view.templates)
            scales.append(view.scales)
            self.candidates.extend((i, j) for j in range(len(view.families)))
        self.scales = np.concatenate(scales)
        families = [views[i].families[j] for i, j in self.candidates]
        if set(kind_weights) != {f["kind"] for f in families} or any(
            w <= 0 for w in kind_weights.values()
        ):
            raise ValueError("Explicit allocation required for all admitted kinds")
        totals = {
            k: sum(f["conditions"] ** temperature for f in families if f["kind"] == k)
            for k in kind_weights
        }
        self.weights = [
            kind_weights[f["kind"]] * f["conditions"] ** temperature / totals[f["kind"]]
            for f in families
        ]
        self.signature = hashlib.sha256(
            encode(
                {
                    "views": [v.signature for v in views],
                    "weights": self.weights,
                    "max_cycles": max_cycles,
                }
            ).encode()
        ).hexdigest()

    def draw(self, max_queries=128):
        weights = [
            w
            if self.children[i].max_cycles is None
            or self.children[i].cycles[j] < self.children[i].max_cycles
            else 0
            for w, (i, j) in zip(self.weights, self.candidates)
        ]
        if not any(weights):
            raise StopIteration("All condition repeat budgets exhausted")
        at = self.random.choices(range(len(weights)), weights=weights, k=1)[0]
        i, j = self.candidates[at]
        rows = self.children[i].draw_family(j, max_queries)
        rows["template"] += self.offsets[i]
        return rows

    def state_dict(self):
        return {
            "signature": self.signature,
            "random": self.random.getstate(),
            "children": [c.state_dict() for c in self.children],
        }

    def load_state_dict(self, state):
        if state["signature"] != self.signature:
            raise ValueError("Changed molecular/corpus mixture")
        self.random.setstate(state["random"])
        if len(state["children"]) != len(self.children):
            raise ValueError("Changed mixture children")
        for child, value in zip(self.children, state["children"]):
            child.load_state_dict(value)
