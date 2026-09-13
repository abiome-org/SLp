"""Disk-backed experimental units and exact fold views; no full record list.

SQLite files are built on cloud CPU disks and cached next to the training GPU.
The reference Record contract remains the boundary for source adapters.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import zlib

from records import Fold, Record

DDL = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE family (id INTEGER PRIMARY KEY, source TEXT, kind TEXT, UNIQUE(source, kind));
CREATE TABLE condition (id INTEGER PRIMARY KEY, family_id INTEGER NOT NULL, identity TEXT UNIQUE NOT NULL, taxon INTEGER NOT NULL);
CREATE TABLE target (condition_id INTEGER NOT NULL, gene TEXT NOT NULL, PRIMARY KEY(condition_id,gene)) WITHOUT ROWID;
CREATE TABLE unit (id INTEGER PRIMARY KEY, condition_id INTEGER NOT NULL, identity TEXT UNIQUE NOT NULL);
CREATE TABLE measurement (id INTEGER PRIMARY KEY, unit_id INTEGER NOT NULL, record_id TEXT UNIQUE NOT NULL, value REAL NOT NULL, assay TEXT NOT NULL, kind TEXT NOT NULL, units TEXT NOT NULL, body BLOB NOT NULL);
CREATE INDEX unit_condition ON unit(condition_id);
CREATE INDEX measurement_unit ON measurement(unit_id);
CREATE INDEX target_gene ON target(gene,condition_id);
"""


def encode(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)


def build(path, rows, provenance):
    """Only source-admitted records enter this cache; quarantine stays in R2.

    Fit-dependent derived records are constructed in a separate admitted fold
    view. This builder rejects them to prevent fold-independent normalization.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA cache_size=-131072")
    db.executescript(DDL)
    families = {}
    count = 0
    try:
        for row in rows:
            if row.role == "quarantine":
                continue
            if row.lineage:
                raise ValueError("Build derived measurements only after the fold mask")
            family = (row.source_id, row.kind)
            if family not in families:
                cursor = db.execute(
                    "INSERT INTO family(source,kind) VALUES (?,?)", family
                )
                families[family] = cursor.lastrowid
            condition_key = encode((family, row.condition))
            db.execute(
                "INSERT OR IGNORE INTO condition(family_id,identity,taxon) VALUES (?,?,?)",
                (families[family], condition_key, row.taxon),
            )
            condition_id = db.execute(
                "SELECT id FROM condition WHERE identity=?", (condition_key,)
            ).fetchone()[0]
            db.executemany(
                "INSERT OR IGNORE INTO target VALUES (?,?)",
                ((condition_id, gene) for gene in row.targets),
            )
            unit_key = encode((condition_id, row.unit))
            db.execute(
                "INSERT OR IGNORE INTO unit(condition_id,identity) VALUES (?,?)",
                (condition_id, unit_key),
            )
            unit_id = db.execute(
                "SELECT id FROM unit WHERE identity=?", (unit_key,)
            ).fetchone()[0]
            db.execute(
                "INSERT INTO measurement(unit_id,record_id,value,assay,kind,units,body) VALUES (?,?,?,?,?,?,?)",
                (
                    unit_id,
                    row.record_id,
                    row.value,
                    row.assay,
                    row.kind,
                    row.units,
                    zlib.compress(encode(asdict(row)).encode(), level=1),
                ),
            )
            count += 1
            if count % 10000 == 0:
                db.commit()
        db.execute(
            "INSERT INTO metadata VALUES ('provenance',?)", (encode(provenance),)
        )
        db.execute("INSERT INTO metadata VALUES ('records',?)", (str(count),))
        db.execute("INSERT INTO metadata VALUES ('complete','true')")
        db.commit()
    finally:
        db.close()
    return {"records": count, "bytes": path.stat().st_size}


class View:
    def __init__(self, path, fold: Fold, stage):
        if stage not in {"pretrain", "adapt"}:
            raise ValueError("Unknown fitting stage")
        self.path = Path(path)
        digest = hashlib.sha256()
        with self.path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
        self.db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        self.db.execute("PRAGMA cache_size=-131072")
        if self.db.execute(
            "SELECT value FROM metadata WHERE key='complete'"
        ).fetchone() != ("true",):
            raise ValueError("Incomplete corpus cache")
        self.db.execute("CREATE TEMP TABLE forbidden (gene TEXT PRIMARY KEY)")
        self.db.executemany(
            "INSERT INTO forbidden VALUES (?)", ((g,) for g in fold.forbidden)
        )
        rule = "f.kind != 'sl'" if stage == "pretrain" else "c.taxon = 9606"
        self.db.execute(
            "CREATE TEMP TABLE eligible AS SELECT c.id AS condition_id, c.family_id FROM condition c JOIN family f ON f.id=c.family_id WHERE "
            + rule
            + " AND (c.taxon != 9606 OR NOT EXISTS (SELECT 1 FROM target t JOIN forbidden x ON x.gene=t.gene WHERE t.condition_id=c.id))"
        )
        self.db.execute("CREATE UNIQUE INDEX eligible_id ON eligible(condition_id)")
        forbidden_count = self.db.execute(
            "SELECT count(*) FROM eligible e JOIN condition c ON c.id=e.condition_id JOIN target t ON t.condition_id=c.id JOIN forbidden x ON x.gene=t.gene WHERE c.taxon=9606"
        ).fetchone()[0]
        if forbidden_count:
            raise AssertionError("Forbidden human intervention exposure")
        self.families = self.db.execute(
            "SELECT f.id,f.source,f.kind,count(*) FROM eligible e JOIN family f ON f.id=e.family_id GROUP BY f.id ORDER BY f.id"
        ).fetchall()
        if not self.families:
            raise ValueError("No fitting conditions")
        self.receipt = {
            "fold": fold.name,
            "stage": stage,
            "forbidden_genes": sorted(fold.forbidden),
            "database_sha256": digest.hexdigest(),
            "forbidden_human_exposures": forbidden_count,
            "families": self.families,
            "source_provenance": json.loads(
                self.db.execute(
                    "SELECT value FROM metadata WHERE key='provenance'"
                ).fetchone()[0]
            ),
        }
        self.signature = hashlib.sha256(encode(self.receipt).encode()).hexdigest()

    def scales(self):
        # Welford streaming accumulation avoids cancellation at large offsets.
        stats = {}
        query = "SELECT m.assay,m.kind,m.units,m.value FROM measurement m JOIN unit u ON m.unit_id=u.id JOIN eligible e ON e.condition_id=u.condition_id WHERE m.kind != 'sl'"
        for assay, kind, units, value in self.db.execute(query):
            key = (assay, kind, units)
            n, mean, m2 = stats.get(key, (0, 0.0, 0.0))
            n += 1
            delta = value - mean
            mean += delta / n
            stats[key] = (n, mean, m2 + delta * (value - mean))
        return {
            key: (mean, max((max(0.0, m2) / max(n - 1, 1)) ** 0.5, 1e-6))
            for key, (n, mean, m2) in stats.items()
        }

    def close(self):
        self.db.close()


class Sampler:
    """One shuffled condition cycle on disk, bounded Python memory.

    A coprime affine permutation visits each dense ordinal once per cycle.
    Each new cycle chooses a new multiplier/offset. This is not a uniformly
    random permutation of all N! orderings; source mixing and unit/coordinate
    draws supply additional randomization. State size is independent of rows.
    """

    def __init__(
        self, view, kind_weights, *, seed=731, temperature=0.7, max_cycles=None
    ):
        import math

        self.view, self.random, self.max_cycles = view, random.Random(seed), max_cycles
        if set(kind_weights) != {f[2] for f in view.families} or any(
            v <= 0 for v in kind_weights.values()
        ):
            raise ValueError("Positive weights required for every admitted kind")
        self.families = view.families
        self.counts = [f[3] for f in self.families]
        totals = {
            kind: sum(f[3] ** temperature for f in self.families if f[2] == kind)
            for kind in kind_weights
        }
        self.weights = [
            kind_weights[f[2]] * f[3] ** temperature / totals[f[2]]
            for f in self.families
        ]
        self.positions, self.cycles, self.permutations = (
            [0] * len(self.families),
            [0] * len(self.families),
            [],
        )
        db = view.db
        db.execute(
            "CREATE TEMP TABLE ordinal (family_id INTEGER, ordinal INTEGER, condition_id INTEGER, PRIMARY KEY(family_id,ordinal)) WITHOUT ROWID"
        )
        db.execute(
            "INSERT INTO ordinal SELECT family_id, row_number() OVER (PARTITION BY family_id ORDER BY condition_id)-1, condition_id FROM eligible"
        )
        self.math = math
        for count in self.counts:
            self.permutations.append(self._permutation(count))
        self.signature = hashlib.sha256(
            encode(
                {
                    "view": view.signature,
                    "weights": self.weights,
                    "max_cycles": max_cycles,
                }
            ).encode()
        ).hexdigest()

    def _permutation(self, count):
        if count == 1:
            return (1, 0)
        multiplier = self.random.randrange(1, count)
        while self.math.gcd(multiplier, count) != 1:
            multiplier = self.random.randrange(1, count)
        return multiplier, self.random.randrange(count)

    def draw(self, max_queries=128):
        weights = [
            w if self.max_cycles is None or self.cycles[i] < self.max_cycles else 0.0
            for i, w in enumerate(self.weights)
        ]
        if not any(weights):
            raise StopIteration("Condition repeat budget exhausted")
        family = self.random.choices(range(len(weights)), weights=weights, k=1)[0]
        count = self.counts[family]
        multiplier, offset = self.permutations[family]
        ordinal = (self.positions[family] * multiplier + offset) % count
        condition = self.view.db.execute(
            "SELECT condition_id FROM ordinal WHERE family_id=? AND ordinal=?",
            (self.families[family][0], ordinal),
        ).fetchone()[0]
        self.positions[family] += 1
        if self.positions[family] == count:
            self.positions[family] = 0
            self.cycles[family] += 1
            self.permutations[family] = self._permutation(count)
        units = self.view.db.execute(
            "SELECT id FROM unit WHERE condition_id=?", (condition,)
        ).fetchall()
        unit = self.random.choice(units)[0]
        ids = self.view.db.execute(
            "SELECT id FROM measurement WHERE unit_id=?", (unit,)
        ).fetchall()
        selected = self.random.sample(ids, min(max_queries, len(ids)))
        return [
            Record.from_dict(
                json.loads(
                    zlib.decompress(
                        self.view.db.execute(
                            "SELECT body FROM measurement WHERE id=?", (i,)
                        ).fetchone()[0]
                    )
                )
            )
            for (i,) in selected
        ]

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
            raise ValueError("Disk sampler view or recipe changed on resume")
        self.random.setstate(state["random"])
        self.positions, self.cycles, self.permutations = (
            state["positions"],
            state["cycles"],
            state["permutations"],
        )
