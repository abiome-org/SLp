"""Source-native staging: preserve outcomes and lineage without fitting anything.

Canonical alias resolution, fold views, likelihood admission and learned
transforms follow this stage. These shards are explicitly not fitting-ready.
"""

from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tarfile
import tempfile
import zipfile

import openpyxl

from inspect_corpus import download, request, save, text_metadata


class Shards:
    def __init__(self, spec, family):
        self.spec, self.family = spec, family
        self.buffer, self.files = [], []
        self.rows = 0
        self.buffer_bytes = 0
        self.genes, self.contexts, self.arities = set(), Counter(), Counter()

    def add(self, value):
        self.buffer.append(
            json.dumps(value, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        )
        self.rows += 1
        self.buffer_bytes += len(self.buffer[-1])
        self.genes.update(value.get("targets", []))
        self.contexts[value.get("context", "unspecified")] += 1
        self.arities[len(value.get("targets", []))] += 1
        if len(self.buffer) >= 32768 or self.buffer_bytes >= 16 * 1024 * 1024:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        payload = gzip.compress(b"".join(self.buffer), compresslevel=3, mtime=0)
        name = f"{self.family}-{len(self.files):05d}.jsonl.gz"
        with request(
            f"/outputs/{os.environ['SLP_JOB']}/{name}", data=payload, prep=True
        ) as response:
            response.read(4096)
        self.files.append(
            {
                "name": name,
                "bytes": len(payload),
                "rows": len(self.buffer),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        self.buffer.clear()
        self.buffer_bytes = 0

    def finish(self, **metadata):
        self.flush()
        report = {
            "schema": "slp.source-native-staging/v1",
            "source": self.spec,
            "family": self.family,
            "rows": self.rows,
            "genes": sorted(self.genes),
            "contexts": dict(self.contexts),
            "intervention_arity": dict(self.arities),
            "shards": self.files,
            "fitting_ready": False,
            **metadata,
        }
        save(self.family + "-manifest", report)
        return {
            "family": self.family,
            "rows": self.rows,
            "shards": len(self.files),
            "genes": len(self.genes),
        }


def finite(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def costanzo(path, spec):
    out = Shards(spec, "costanzo")
    excluded = Counter()
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not Path(name).name.startswith("SGA_") or not name.endswith(".txt"):
                continue
            with archive.open(name) as raw:
                for line, r in enumerate(
                    csv.DictReader(
                        io.TextIOWrapper(raw, encoding="utf-8-sig"), delimiter="\t"
                    ),
                    2,
                ):
                    a, b = r["Query Strain ID"], r["Array Strain ID"]
                    ga, gb = a.split("_")[0], b.split("_")[0]
                    if not re.fullmatch(
                        r"Y[A-P][LR]\d{3}[CW](?:-[AB])?", ga
                    ) or not re.fullmatch(r"Y[A-P][LR]\d{3}[CW](?:-[AB])?", gb):
                        excluded["unrecognized_orf"] += 1
                        continue
                    values = {
                        key: finite(r[key])
                        for key in (
                            "Genetic interaction score (ε)",
                            "P-value",
                            "Query single mutant fitness (SMF)",
                            "Array SMF",
                            "Double mutant fitness",
                            "Double mutant fitness standard deviation",
                        )
                    }
                    if all(
                        values[k] is None
                        for k in (
                            "Genetic interaction score (ε)",
                            "Double mutant fitness",
                        )
                    ):
                        excluded["no_finite_outcome"] += 1
                        continue
                    out.add(
                        {
                            "locator": f"{Path(name).name}:{line}",
                            "taxon": 559292,
                            "targets": sorted(set((ga, gb))),
                            "context": r["Arraytype/Temp"],
                            "query_strain": a,
                            "array_strain": b,
                            "query_allele": r["Query allele name"],
                            "array_allele": r["Array allele name"],
                            "measurements": values,
                        }
                    )
    return out.finish(
        exclusions=dict(excluded),
        target_namespace="SGD systematic ORF",
        semantics="Native epsilon and untransformed relative fitness; alleles and temperature retained",
    )


def in4mer(path, spec):
    out = Shards(spec, "in4mer-original")
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        for sheet in workbook.worksheets:
            if not sheet.title.startswith("Fig3D-"):
                continue
            iterator = sheet.iter_rows(values_only=True)
            header = next(iterator)
            for line, row in enumerate(iterator, 2):
                item = dict(zip(header, row))
                score = finite(row[-1])
                if score is None:
                    continue
                target = str(item["Gene"])
                targets = target.split("_")
                out.add(
                    {
                        "locator": f"{sheet.title}:{line}",
                        "taxon": 9606,
                        "targets": targets,
                        "context": sheet.title.removeprefix("Fig3D-"),
                        "construct": item["CloneID"],
                        "array": item["Array"],
                        "class": item["Class"],
                        "raw_target": target,
                        "measurement": score,
                        "measurement_column": header[-1],
                    }
                )
    finally:
        workbook.close()
    return out.finish(
        target_namespace="publisher symbol set; canonicalization pending",
        semantics="Original construct-level processed depletion; controls retained and not relabeled as gene perturbations",
    )


def spidr(path, spec):
    out = Shards(spec, "spidr")
    with path.open() as file:
        for line, row in enumerate(csv.DictReader(file), 2):
            score = finite(row["sens.score"])
            if score is None:
                continue
            out.add(
                {
                    "locator": f"scores.csv:{line}",
                    "taxon": 9606,
                    "targets": sorted(row["gene_combination"].split(";")),
                    "context": "RPE1",
                    "measurement": score,
                    "measurement_kind": "GEMINI sensitive lethality score",
                }
            )
    return out.finish(
        target_namespace="publisher gene symbols",
        semantics="Native sensitive-lethality score; not raw viability or a binary SL label",
    )


def sql_values(raw):
    """Parse SQLite INSERT literals only; never execute publisher SQL."""
    text = raw.decode("utf-8").strip()
    match = re.fullmatch(r"INSERT INTO ([a-z_]+) VALUES\((.*)\);", text)
    if not match:
        raise ValueError("Unsupported INSERT syntax")
    values = next(
        csv.reader(
            [match[2]], delimiter=",", quotechar="'", doublequote=True, strict=True
        )
    )
    return match[1], values


def slkb(path, spec):
    labels = Shards(spec, "slkb-original")
    counts = Shards(spec, "slkb-counts")
    guides = Shards(spec, "slkb-guides")
    states, origins = Counter(), Counter()
    with (
        zipfile.ZipFile(path) as archive,
        archive.open("SQL_Dumps/SLKB-sqlite3_dump.sql") as file,
    ):
        for line, raw in enumerate(file, 1):
            if not raw.startswith(
                (
                    b"INSERT INTO cdko_original_sl_results VALUES",
                    b"INSERT INTO cdko_sgrna_counts VALUES",
                    b"INSERT INTO cdko_experiment_design VALUES",
                )
            ):
                continue
            table, v = sql_values(raw)
            if table == "cdko_original_sl_results":
                if len(v) != 12:
                    raise ValueError("SLKB label schema changed")
                states[v[7]] += 1
                origins[v[3]] += 1
                labels.add(
                    {
                        "locator": f"sqlite:{line}",
                        "taxon": 9606,
                        "study": v[3],
                        "context": v[4],
                        "targets": sorted(set(v[5:7])),
                        "pair_id": v[1],
                        "label_text": v[7],
                        "score": finite(v[8]),
                        "statistic": finite(v[9]),
                        "score_cutoff": finite(v[10]),
                        "statistic_cutoff": finite(v[11]),
                    }
                )
            elif table == "cdko_sgrna_counts":
                if len(v) != 12:
                    raise ValueError("SLKB count schema changed")
                counts.add(
                    {
                        "locator": f"sqlite:{line}",
                        "guide_pair_id": v[0],
                        "guide_ids": v[1:3],
                        "pair_id": v[3],
                        "orientation": v[4],
                        "t0_counts": v[5],
                        "t0_replicates": v[6],
                        "tend_counts": v[7],
                        "tend_replicates": v[8],
                        "target_type": v[9],
                        "study": v[10],
                        "context": v[11],
                    }
                )
            else:
                if len(v) != 5:
                    raise ValueError("SLKB guide schema changed")
                guides.add(
                    {
                        "guide_id": v[0],
                        "guide_name": v[1],
                        "sequence": v[2],
                        "target": v[3],
                        "study": v[4],
                    }
                )
    a = labels.finish(
        label_states=dict(states),
        studies=dict(origins),
        semantics="Original author labels and scores; binary label admission requires evidence/negative-definition audit",
    )
    b = counts.finish(
        semantics="Raw guide counts; no library normalization or contrast fitted before fold masking"
    )
    c = guides.finish(
        semantics="Raw guide design needed to resolve actual intervention targets"
    )
    return [a, b, c]


def harle_nested(path, spec):
    members = []
    with tarfile.open(path, "r|gz") as outer:
        for member in outer:
            if not member.isfile() or not member.name.endswith(".tar.gz"):
                continue
            with (
                outer.extractfile(member) as source,
                tarfile.open(fileobj=source, mode="r|gz") as inner,
            ):
                for sub in inner:
                    if not sub.isfile():
                        continue
                    item = {"archive": member.name, "name": sub.name, "bytes": sub.size}
                    if sub.name.endswith((".tsv", ".txt", ".csv")):
                        with inner.extractfile(sub) as stream:
                            item.update(text_metadata(stream, sub.name))
                    members.append(item)
    save("harle-nested", {"source": spec, "members": members, "fitting_ready": False})
    return {"family": "harle-nested", "members": len(members)}


def main():
    manifest = json.load(request("/manifest"))
    operations = {
        "slkb-2023-41055392": slkb,
        "in4mer-2024-44435672": in4mer,
        "spidr-2025-scores": spidr,
        "harle-2025-46763995": harle_nested,
        "costanzo-2016-pairwise": costanzo,
    }
    summary = []
    for spec in manifest["objects"]:
        if spec["id"] not in operations:
            continue
        with tempfile.TemporaryDirectory() as directory:
            path = download(spec, directory)
            result = operations[spec["id"]](path, spec)
            summary.extend(result if isinstance(result, list) else [result])
    save(
        "complete",
        {
            "job": os.environ["SLP_JOB"],
            "state": "complete",
            "summary": summary,
            "fitting_ready": False,
            "pending": [
                "Canonical gene resolution",
                "Source-specific likelihood and negative-label admission",
                "Strict CV3 fold views",
                "Static sequence features",
                "Harle measurement adapter",
            ],
            "excluded_duplicate_exports": [
                "In4mer Supp_table1 reanalyses of earlier studies",
                "In4mer plot-level copies of original measurements",
            ],
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        save("failed", {"state": "failed", "error_type": type(exc).__name__})
        raise SystemExit(1) from None
