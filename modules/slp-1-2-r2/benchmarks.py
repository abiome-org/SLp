"""Published row-preserving SL folds, with explicit inner/outer fitting access."""

import gzip
import json
from pathlib import Path
from records import Fold, Record
from data import digest


class Benchmark:
    def __init__(self, path, features):
        self.path = Path(path)
        self.spec = json.loads(self.path.read_text())
        self.features = features
        self.identity = digest(self.path)
        self.outer = frozenset(self.spec["outer_held"])
        self.inner = frozenset(self.spec["inner_held"])
        if self.outer & self.inner:
            raise ValueError("Inner and outer held genes overlap")
        if any(g not in features.index for g in self.outer | self.inner):
            raise ValueError("Unresolved benchmark feature identity")

    def fold(self, scope):
        if scope not in ("inner", "outer"):
            raise ValueError("Unknown fold scope")
        return Fold(
            self.spec["name"] + "-" + scope,
            self.outer,
            self.inner if scope == "inner" else frozenset(),
        )

    def partition(self, name):
        if name not in self.spec["partitions"]:
            raise ValueError("No such official partition")
        path = self.path.parent / self.spec["partitions"][name]
        if path.parent != self.path.parent:
            raise ValueError("Invalid benchmark manifest path")
        manifest = json.loads(path.read_text())
        rows = []
        for shard in manifest["shards"]:
            file = path.parent / shard["name"]
            if file.name != shard["name"] or digest(file) != shard["sha256"]:
                raise ValueError("Benchmark shard identity mismatch")
            body = file.read_bytes()
            if len(body) != shard["bytes"]:
                raise ValueError("Benchmark shard size mismatch")
            for raw in gzip.decompress(body).splitlines():
                row = json.loads(raw)
                if row["row"] != len(rows):
                    raise ValueError("Official benchmark row order changed")
                if any(g not in self.features.index for g in row["targets"]):
                    raise ValueError("Unresolved official benchmark pair")
                rows.append(row)
        if len(rows) != manifest["rows"]:
            raise ValueError("Official benchmark row count changed")
        return rows

    def labels(self, scope, partition):
        """Only both-inner genes are scored for a constructed inner CV3 split."""
        if scope not in ("inner", "outer") or partition not in ("fit", "evaluate"):
            raise ValueError("Unknown label access")
        if scope == "outer" and partition == "evaluate":
            return self.partition("test")
        training = self.partition("train")
        if partition == "fit":
            if scope == "outer" and "valid" in self.spec["partitions"]:
                training += self.partition("valid")
            forbidden = self.fold(scope).forbidden
            return [r for r in training if not (set(r["targets"]) & forbidden)]
        validation = (
            self.partition("valid") if "valid" in self.spec["partitions"] else training
        )
        return [
            r
            for r in validation
            if set(r["targets"]) <= self.inner and not (set(r["targets"]) & self.outer)
        ]

    def records(self, scope, partition):
        result = []
        for i, r in enumerate(self.labels(scope, partition)):
            context = r["context"]
            source = self.spec["benchmark"]
            result.append(
                Record(
                    record_id=f"{self.spec['name']}/{scope}/{partition}/{i}",
                    source_id=source,
                    study_id=str(self.spec["protocol"].get("study", source)),
                    experiment_id=f"{self.spec['name']}/{partition}/{i}",
                    replicate_id="official-row",
                    taxon=9606,
                    targets=tuple(sorted(r["targets"])),
                    context=context,
                    assay="human-SL",
                    kind="sl",
                    units="binary-SL-call",
                    value=float(r["label"]),
                    query_gene=None,
                    mechanism="unknown",
                    method="unknown",
                    label_scope="pan-cancer"
                    if context == "pan-cancer"
                    else "cell-line",
                    role="sl_label",
                    license="source-benchmark-terms",
                    raw_object="sha256:" + self.identity,
                    raw_locator=f"{scope}/{partition}/official-row:{r['row']}",
                )
            )
        return result


def verify_training_labels(rows, fold):
    if any(
        r.taxon != 9606 or r.kind != "sl" or set(r.targets) & fold.forbidden
        for r in rows
    ):
        raise ValueError("SL fitting labels violate the human fold exposure rule")
