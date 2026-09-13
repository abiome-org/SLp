"""Canonical, compact measurement shards built entirely in Cloudflare.

Gene indices join the complete static manifest. Raw/source-native rows remain
durable lineage; packed data contain no fitted transforms and no fold labels.
"""

from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import tempfile
import traceback
from pathlib import Path

import numpy as np

from inspect_corpus import download, request, save

IDENTITY = "identity-r2-20260912-v5"
NATIVE = "native-r2-20260912-v1"
PROTOCOL = "protocol-r2-20260912-v2"
KINDS = {k: i for i, k in enumerate(("rna", "protein", "fitness", "interaction", "sl"))}
MECHANISMS = {
    k: i
    for i, k in enumerate(
        (
            "unknown",
            "knockout",
            "crispri",
            "crispra",
            "deletion",
            "temperature_sensitive",
            "damP",
            "ligand_addition",
        )
    )
}
METHODS = {
    k: i
    for i, k in enumerate(
        ("unknown", "Cas9", "Cas12a", "dCas9-KRAB", "SGA", "protein-addition")
    )
}
DTYPE = np.dtype(
    [
        ("template", "<u4"),
        ("targets", "<i4", (4,)),
        ("mechanism", "u1", (4,)),
        ("method", "u1", (4,)),
        ("numeric", "<f4", (4, 3)),
        ("known", "?", (4, 3)),
        ("query", "<i4"),
        ("value", "<f4"),
        ("condition", "S32"),
        ("unit", "S32"),
        ("native_row", "<u4"),
    ]
)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def metadata(job, name):
    return json.load(request(f"/outputs/{job}/{name}.json", prep=True))


def native_rows(job, family):
    manifest = metadata(job, family + "-manifest")
    config = json.loads(Path("prep-jobs.json").read_text()).get(
        os.environ["SLP_JOB"], {}
    )
    start = config.get("shard_start", 0) if family == "costanzo" else 0
    end = (
        config.get("shard_end", len(manifest["shards"]))
        if family == "costanzo"
        else len(manifest["shards"])
    )
    for shard in manifest["shards"][start:end]:
        with request(f"/outputs/{job}/{shard['name']}", prep=True) as response:
            body = response.read(32 * 1024 * 1024 + 1)
        if (
            len(body) != shard["bytes"]
            or hashlib.sha256(body).hexdigest() != shard["sha256"]
        ):
            raise ValueError("Source-native shard checksum mismatch")
        for line, raw in enumerate(gzip.decompress(body).splitlines()):
            yield (
                json.loads(raw),
                {
                    "job": job,
                    "manifest": family + "-manifest.json",
                    "shard": shard["name"],
                    "sha256": shard["sha256"],
                },
                line,
            )


class Writer:
    def __init__(self, genes, name):
        self.genes, self.index = genes, {g: i for i, g in enumerate(genes)}
        self.name, self.templates, self.template_index = name, [], {}
        self.buffer = np.zeros(65536, DTYPE)
        self.buffer["targets"] = -1
        self.buffer["query"] = -1
        self.length = self.total = 0
        self.shards, self.coverage, self.quarantine = [], Counter(), Counter()
        self.target_genes = set()

    def add(
        self,
        *,
        template,
        genes,
        value,
        native_row,
        unit,
        query=None,
        mechanisms=None,
        method="unknown",
        hours=None,
        temperature=None,
        alleles=None,
    ):
        if len(genes) > 4 or len(set(genes)) != len(genes):
            raise ValueError("Unsupported or duplicated canonical action set")
        if not math.isfinite(value) or (
            template["kind"] == "sl" and value not in (0, 1)
        ):
            raise ValueError("Invalid admitted outcome")
        mechanisms = mechanisms or ["unknown"] * len(genes)
        methods = [method] * len(genes) if isinstance(method, str) else list(method)
        times = (
            list(hours) if isinstance(hours, (list, tuple)) else [hours] * len(genes)
        )
        if (
            len(methods) != len(genes)
            or len(mechanisms) != len(genes)
            or len(times) != len(genes)
        ):
            raise ValueError("Each action requires its own mechanism and method")
        order = sorted(range(len(genes)), key=lambda i: genes[i])
        genes = [genes[i] for i in order]
        mechanisms = [mechanisms[i] for i in order]
        methods = [methods[i] for i in order]
        times = [times[i] for i in order]
        alleles = [alleles[i] for i in order] if alleles else []
        if any(
            g not in self.index or not g.startswith(str(template["taxon"]) + ":")
            for g in genes + ([query] if query else [])
        ):
            raise ValueError(
                "Admitted identity absent from static roster or wrong species"
            )
        key = encode(template)
        if key not in self.template_index:
            self.template_index[key] = len(self.templates)
            self.templates.append(template)
        tid = self.template_index[key]
        row = self.buffer[self.length]
        row["template"] = tid
        row["targets"][: len(genes)] = [self.index[g] for g in genes]
        row["mechanism"][: len(genes)] = [MECHANISMS[m] for m in mechanisms]
        row["method"][: len(genes)] = [METHODS[m] for m in methods]
        for i, time in enumerate(times):
            if time is not None:
                if time < 0:
                    raise ValueError("Negative intervention duration")
                row["numeric"][i, 1] = math.log1p(time)
                row["known"][i, 1] = True
        if temperature is not None:
            row["numeric"][: len(genes), 2] = temperature / 100.0
            row["known"][: len(genes), 2] = True
        row["query"] = self.index[query] if query else -1
        row["value"], row["native_row"] = value, native_row
        # Storage shard and locator are provenance, never different conditions.
        conditioning = {k: v for k, v in template.items() if k not in ("lineage",)}
        row["condition"] = hashlib.sha256(
            encode(
                (conditioning, genes, mechanisms, methods, times, temperature, alleles)
            ).encode()
        ).digest()
        row["unit"] = hashlib.sha256(encode(unit).encode()).digest()
        self.length += 1
        self.total += 1
        self.target_genes.update(genes)
        self.coverage[(template["source"], template["kind"], str(len(genes)))] += 1
        if self.length == len(self.buffer):
            self.flush()

    def flush(self):
        if not self.length:
            return
        stream = io.BytesIO()
        np.save(stream, self.buffer[: self.length], allow_pickle=False)
        raw = stream.getvalue()
        payload = gzip.compress(raw, compresslevel=1, mtime=0)
        name = f"{self.name}-{len(self.shards):05d}.bin.gz"
        with request(
            f"/outputs/{os.environ['SLP_JOB']}/{name}", data=payload, prep=True
        ) as response:
            response.read(4096)
        self.shards.append(
            {
                "name": name,
                "rows": self.length,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "npy_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        self.buffer.fill(0)
        self.buffer["targets"] = -1
        self.buffer["query"] = -1
        self.length = 0

    def finish(self):
        self.flush()
        result = {
            "schema": "slp.packed-measurements/v1",
            "name": self.name,
            "job": os.environ["SLP_JOB"],
            "rows": self.total,
            "dtype": DTYPE.descr,
            "shards": self.shards,
            "templates": self.templates,
            "gene_manifest_job": IDENTITY,
            "genes_sha256": hashlib.sha256(encode(self.genes).encode()).hexdigest(),
            "mechanisms": MECHANISMS,
            "methods": METHODS,
            "target_genes": sorted(self.target_genes),
            "coverage": [
                {"source": s, "kind": k, "arity": int(a), "rows": n}
                for (s, k, a), n in self.coverage.items()
            ],
            "quarantine": dict(self.quarantine),
            "outcome_fitted_transforms": [],
            "admission": "canonical measurements; fold masks still required before fitting",
            "source_processing": "publisher-processed observations retained; no SLp outcome fitting in this artifact",
        }
        save(self.name + "-manifest", result)
        return {
            "name": self.name,
            "rows": self.total,
            "shards": len(self.shards),
            "genes": len(self.target_genes),
            "quarantine": dict(self.quarantine),
        }


def resolve(targets, taxon, aliases, writer, family):
    genes = []
    for g in targets:
        canonical = aliases.get(g) if taxon == 9606 else f"{taxon}:" + g
        if canonical not in writer.index:
            writer.quarantine[family + ":unresolved-or-absent-feature:" + str(g)] += 1
            return None
        genes.append(canonical)
    if len(set(genes)) != len(genes):
        writer.quarantine[family + ":aliases-collapse-targets"] += 1
        return None
    return genes


def yeast(writer, aliases):
    for row, lineage, line in native_rows(NATIVE, "costanzo"):
        genes = resolve(row["targets"], 559292, aliases, writer, "costanzo")
        if genes is None:
            continue
        # Alleles stay attached to their original query/array ORFs before sorting.
        native_genes = [
            row["query_strain"].split("_")[0],
            row["array_strain"].split("_")[0],
        ]
        if len(set(native_genes)) != 2:
            writer.quarantine["costanzo:self-pair"] += 1
            continue
        genes = ["559292:" + g for g in native_genes]
        alleles = [row["query_allele"], row["array_allele"]]
        mechanisms = []
        for strain, allele in zip((row["query_strain"], row["array_strain"]), alleles):
            mechanisms.append(
                "temperature_sensitive"
                if "ts" in strain.lower()
                else "damP"
                if "damp" in strain.lower() or "damp" in allele.lower()
                else "deletion"
                if re.search(r"_(?:sn|dma)\d+$", strain)
                else "unknown"
            )
        temp = re.search(r"(\d{2})(?:$|\D)", row["context"])
        for field, kind, units in (
            ("Genetic interaction score (ε)", "interaction", "epsilon"),
            ("Double mutant fitness", "fitness", "relative-fitness"),
        ):
            value = row["measurements"][field]
            if value is None:
                continue
            template = {
                "source": "costanzo-2016",
                "study": "PMID:27708008",
                "taxon": 559292,
                "context": row["context"],
                "assay": "SGA-" + units,
                "kind": kind,
                "units": units,
                "scope": "strain-condition",
                "license": "unverified",
                "training_allowed": False,
                "rights_record": "rights/costanzo-2016-sga.yaml",
                "lineage": lineage,
            }
            writer.add(
                template=template,
                genes=genes,
                value=value,
                native_row=line,
                unit=(
                    row["query_strain"],
                    row["array_strain"],
                    row["context"],
                    row["locator"],
                ),
                mechanisms=mechanisms,
                method="SGA",
                temperature=float(temp[1]) if temp else None,
                alleles=alleles,
            )
    return writer.finish()


def human_combinations(writer, aliases):
    sources = [
        (
            NATIVE,
            "in4mer-original",
            "fitness",
            "log2-fold-change",
            "knockout",
            "Cas12a",
        ),
        (
            NATIVE,
            "spidr",
            "interaction",
            "GEMINI-sensitive-score",
            "crispri",
            "dCas9-KRAB",
        ),
        (
            PROTOCOL,
            "harle-gi",
            "interaction",
            "native-mean-normalized-GI",
            "knockout",
            "Cas9",
        ),
        (
            PROTOCOL,
            "harle-fitness",
            "fitness",
            "publisher-log2-fold-change",
            "knockout",
            "Cas9",
        ),
    ]
    for job, family, kind, units, mechanism, method in sources:
        for row, lineage, line in native_rows(job, family):
            if family == "in4mer-original" and len(row["targets"]) == 3:
                writer.quarantine[
                    family + ":unresolved-fourth-guide-in-prototype-triple"
                ] += 1
                continue
            genes = resolve(row["targets"], 9606, aliases, writer, family)
            if genes is None:
                continue
            context = row.get("depmap_id") or row["context"]
            mechanisms = [mechanism] * len(genes)
            methods = [method] * len(genes)
            if family == "spidr":
                background = aliases["TP53"]
                if background in genes:
                    writer.quarantine[
                        "spidr:target-overlaps-engineered-TP53-background"
                    ] += 1
                    continue
                genes.append(background)
                mechanisms.append("knockout")
                methods.append("unknown")
                context = "RPE1-TP53-KO"
            template = {
                "source": family,
                "study": family.split("-")[0],
                "taxon": 9606,
                "context": context,
                "assay": family,
                "kind": kind,
                "units": units,
                "scope": "cell-line",
                "license": "MIT" if family.startswith("harle") else "CC-BY-4.0",
                "training_allowed": True,
                "lineage": lineage,
            }
            hours = [336.0] * (len(genes) - 1) + [None] if family == "spidr" else None
            if family == "in4mer-original":
                match = re.search(r"_T(\d+)", row["measurement_column"])
                hours = float(match[1]) * 24 if match else None
            writer.add(
                template=template,
                genes=genes,
                value=row["measurement"],
                native_row=line,
                unit=(
                    family,
                    row.get("construct", row["locator"]),
                    row.get("replicate", "publisher-aggregate"),
                    context,
                ),
                mechanisms=mechanisms,
                method=methods,
                hours=hours,
            )
        save("progress-" + family, {"state": "packed", "rows_so_far": writer.total})
    return writer.finish()


def depmap(writer, aliases):
    sources = json.load(request("/manifest"))["objects"]
    spec = next(s for s in sources if s["id"] == "depmap-24q2-46489063")
    with tempfile.TemporaryDirectory() as directory:
        path = download(spec, directory)
        with path.open() as file:
            reader = csv.reader(file)
            header = next(reader)
            resolved = []
            for column in header[1:]:
                match = re.fullmatch(r"(.+) \((\d+)\)", column)
                if not match:
                    raise ValueError("DepMap gene column schema changed")
                candidates = {
                    aliases[v] for v in (match[1], "ENTREZ:" + match[2]) if v in aliases
                }
                resolved.append(
                    next(iter(candidates)) if len(candidates) == 1 else None
                )
            for line, row in enumerate(reader, 2):
                context = row[0]
                template = {
                    "source": "depmap-24q2",
                    "study": "DepMap-24Q2",
                    "taxon": 9606,
                    "context": context,
                    "assay": "DepMap-Chronos",
                    "kind": "fitness",
                    "units": "Chronos-gene-effect",
                    "scope": "cell-line",
                    "license": spec["license"],
                    "training_allowed": True,
                    "lineage": {
                        "raw_object": spec["key"],
                        "checksum": spec["checksum"],
                        "row_is_csv_line": True,
                        "query_column": "resolved intervention gene",
                    },
                }
                for gene, value in zip(resolved, row[1:]):
                    if gene is None or gene not in writer.index:
                        writer.quarantine[
                            "depmap:unresolved-or-conflicting-identifier"
                        ] += 1
                        continue
                    try:
                        value = float(value)
                    except ValueError:
                        continue
                    if not math.isfinite(value):
                        continue
                    writer.add(
                        template=template,
                        genes=[gene],
                        value=value,
                        native_row=line,
                        unit=(context, gene),
                        mechanisms=["knockout"],
                        method="Cas9",
                    )
    return writer.finish()


if __name__ == "__main__":
    try:
        aliases = metadata(IDENTITY, "aliases")["human"]
        genes = metadata(IDENTITY, "sequence-inputs-manifest")["genes"]
        job = os.environ["SLP_JOB"]
        kind = (
            "yeast"
            if job.startswith("pack-yeast")
            else "depmap"
            if job.startswith("pack-depmap")
            else "human-combinations"
        )
        writer = Writer(genes, kind)
        function = {
            "yeast": yeast,
            "depmap": depmap,
            "human-combinations": human_combinations,
        }[kind]
        result = function(writer, aliases)
        save(
            "complete",
            {
                "state": "complete",
                "summary": result,
                "research_training_launched": False,
            },
        )
    except Exception as exc:
        save(
            "failed",
            {
                "type": type(exc).__name__,
                "http_status": getattr(exc, "code", None),
                "detail": str(exc)[:400]
                if isinstance(exc, (ValueError, KeyError))
                else None,
                "frames": [
                    {"file": Path(f.filename).name, "line": f.lineno}
                    for f in traceback.extract_tb(exc.__traceback__)
                ],
            },
        )
        raise SystemExit(1) from None
