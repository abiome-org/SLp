"""Cloud-only capture of canonical gene aliases and static protein sequences.

No outcome data or benchmark labels are used to compute these features.
Raw source bytes and exact checksums are preserved alongside normalized rows.
"""

from collections import defaultdict
import csv
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
import time
import urllib.error
import urllib.request

from inspect_corpus import download, request, save
from normalize_corpus import Shards

SOURCES = {
    "hgnc": "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
    "human-peptides": "https://ftp.ensembl.org/pub/release-116/fasta/homo_sapiens/pep/Homo_sapiens.GRCh38.pep.all.fa.gz",
    "yeast-proteins": "https://rest.uniprot.org/uniprotkb/stream?query=proteome%3AUP000002311&format=tsv&fields=accession,gene_primary,gene_synonym,gene_oln,sequence,reviewed",
    "musl-roster": "https://raw.githubusercontent.com/JieZheng-ShanghaiTech/MuSL/f8021cfc618fafae8c330b694d0fa7c46db5f1a5/processed_data/meta_table_7684.csv",
}
PHASE = "starting"


def protein_capture(name, url):
    """Use bounded pages; bulk stream requests can time out at the publisher."""
    current = url.replace("/stream?", "/search?size=500&")
    chunks, pages, release = [], [], None
    total_bytes = 0
    while current:
        for attempt in range(4):
            try:
                req = urllib.request.Request(
                    current, headers={"User-Agent": "SLp-Corpus-Preparation/1.0"}
                )
                with urllib.request.urlopen(req, timeout=90) as response:
                    body = response.read(8 * 1024 * 1024 + 1)
                    if len(body) > 8 * 1024 * 1024:
                        raise ValueError("Protein page exceeded capture limit")
                    this_release = response.headers.get("X-UniProt-Release")
                    link = response.headers.get("Link", "")
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise
                time.sleep(2**attempt)
        if release is not None and this_release != release:
            raise ValueError("UniProt release changed during paginated capture")
        release = this_release
        pages.append(
            {
                "url": current,
                "sha256": hashlib.sha256(body).hexdigest(),
                "bytes": len(body),
            }
        )
        chunks.append(body if len(pages) == 1 else body.partition(b"\n")[2])
        total_bytes += len(chunks[-1])
        if total_bytes > 128 * 1024 * 1024 or len(pages) > 500:
            raise ValueError("Protein source exceeds bounded capture")
        match = re.search(r'<(https://rest\.uniprot\.org/[^>]+)>;\s*rel="next"', link)
        current = match[1] if match else None
    body = b"".join(chunks)
    metadata = {
        "url": url,
        "pages": pages,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "uniprot_release": release,
        "capture": "paginated TSV; one common header",
    }
    payload = gzip.compress(body, compresslevel=3, mtime=0)
    with request(
        f"/outputs/{os.environ['SLP_JOB']}/{name}-raw.bin.gz", data=payload, prep=True
    ) as response:
        response.read(4096)
    save(name + "-source", metadata)
    return body, metadata


def capture(name, url):
    req = urllib.request.Request(
        url, headers={"User-Agent": "SLp-Corpus-Preparation/1.0"}
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        body = response.read(128 * 1024 * 1024 + 1)
        if len(body) > 128 * 1024 * 1024:
            raise ValueError("Static source exceeds bounded capture size")
        metadata = {
            "url": url,
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "uniprot_release": response.headers.get("X-UniProt-Release"),
        }
    if name == "human-peptides":
        if (
            metadata["sha256"]
            != "9b43da92651b35814597af6a8b18f500b768679a49fa4678224f384917ce7668"
            or len(body) != 23319936
        ):
            raise ValueError("Ensembl 116 peptide checksum mismatch")
        payload = body
    else:
        payload = gzip.compress(body, compresslevel=3, mtime=0)
    if (
        name == "musl-roster"
        and hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
        != "d0ce4cd075d1d63d1994b716e4319620b3696bbc"
    ):
        raise ValueError("Pinned MuSL roster Git object mismatch")
    with request(
        f"/outputs/{os.environ['SLP_JOB']}/{name}-raw.bin.gz", data=payload, prep=True
    ) as response:
        response.read(4096)
    save(name + "-source", metadata)
    return body, metadata


def aliases(hgnc):
    approved, secondary, rows = {}, defaultdict(set), {}
    stable_ids = defaultdict(set)
    for row in hgnc:
        gene = "9606:" + row["hgnc_id"]
        if row["symbol"] in approved and approved[row["symbol"]] != gene:
            raise ValueError("Conflicting approved HGNC symbol")
        identifiers = [row["symbol"], row["hgnc_id"]]
        if row.get("ensembl_gene_id"):
            stable_ids[row["ensembl_gene_id"]].add(gene)
        if row.get("entrez_id"):
            stable_ids["ENTREZ:" + row["entrez_id"]].add(gene)
        for identifier in filter(None, identifiers):
            if identifier in approved and approved[identifier] != gene:
                raise ValueError("Conflicting approved HGNC identifier")
            approved[identifier] = gene
        for field in ("alias_symbol", "prev_symbol"):
            for alias in row.get(field, "").split("|"):
                if alias:
                    secondary[alias].add(gene)
        rows[gene] = row
    secondary.update(stable_ids)
    ambiguous = {
        k: sorted(v) for k, v in secondary.items() if len(v) > 1 and k not in approved
    }
    result = {
        k: next(iter(v))
        for k, v in secondary.items()
        if len(v) == 1 and k not in approved
    }
    result.update(approved)
    return result, ambiguous, rows


def harle_measurements():
    manifest = json.load(request("/manifest"))
    spec = next(x for x in manifest["objects"] if x["id"] == "harle-2025-46763995")
    gi, fitness = Shards(spec, "harle-gi"), Shards(spec, "harle-fitness")
    with tempfile.TemporaryDirectory() as directory:
        path = download(spec, directory)
        with tarfile.open(path, "r|gz") as outer:
            for member in outer:
                if member.name not in (
                    "DATA/postprocessing.tar.gz",
                    "DATA/preprocessing.tar.gz",
                ):
                    continue
                with (
                    outer.extractfile(member) as stream,
                    tarfile.open(fileobj=stream, mode="r|gz") as inner,
                ):
                    for sub in inner:
                        if sub.name == "postprocessing/combined_gene_level_results.tsv":
                            with inner.extractfile(sub) as file:
                                for line, row in enumerate(
                                    csv.DictReader(
                                        (line.decode("utf-8") for line in file),
                                        delimiter="\t",
                                    ),
                                    2,
                                ):
                                    value = row["mean_norm_gi"]
                                    if value in ("NA", "", "NaN"):
                                        continue
                                    gi.add(
                                        {
                                            "locator": f"{sub.name}:{line}",
                                            "taxon": 9606,
                                            "targets": sorted(
                                                set((row["targetA"], row["targetB"]))
                                            ),
                                            "context": row["cell_line_label"],
                                            "depmap_id": row["depMapID"],
                                            "measurement": float(value),
                                            "kind": "native_mean_normalized_GI",
                                            "replicates": row["n_replicates"],
                                            "guide_pairs": row["n_guide_pairs"],
                                        }
                                    )
                        elif (
                            sub.name
                            == "preprocessing/lfc_matrix.unscaled.samples_removed.tsv"
                        ):
                            with inner.extractfile(sub) as file:
                                for line, row in enumerate(
                                    csv.DictReader(
                                        (line.decode("utf-8") for line in file),
                                        delimiter="\t",
                                    ),
                                    2,
                                ):
                                    guide_types = row["guide_type"].split("|")
                                    guide_genes = [row["sgrnaA"], row["sgrnaB"]]
                                    if len(guide_types) != 2:
                                        raise ValueError("Harle guide types changed")
                                    actual = sorted(
                                        {
                                            g
                                            for g, t in zip(guide_genes, guide_types)
                                            if t == "gene"
                                        }
                                    )
                                    if row["singles_target_gene"] not in (
                                        "NA",
                                        "",
                                    ) and actual != [row["singles_target_gene"]]:
                                        raise ValueError(
                                            "Harle actual single target disagreement"
                                        )
                                    for column, value in row.items():
                                        sample = re.fullmatch(r"(.+) (R\d+)", column)
                                        if not sample or value in ("NA", "", "NaN"):
                                            continue
                                        fitness.add(
                                            {
                                                "locator": f"{sub.name}:{line}:{column}",
                                                "taxon": 9606,
                                                "targets": actual,
                                                "context": sample[1],
                                                "replicate": sample[2],
                                                "construct": row["id"],
                                                "measurement": float(value),
                                                "kind": "publisher_unscaled_LFC",
                                                "guide_type": row["guide_type"],
                                                "library_pair": row["sorted_gene_pair"],
                                            }
                                        )
    return [
        gi.finish(semantics="Native measured normalized GI, not binary hit calls"),
        fitness.finish(
            semantics="Original construct log fold changes, actual gene-targeting guides only"
        ),
    ]


def main():
    global PHASE
    captured = {}
    for name, url in SOURCES.items():
        PHASE = "capture-" + name
        # Retry normalization against already captured immutable source bytes.
        # These four v3 captures completed before its explicit ambiguity failure.
        previous = "identity-r2-20260912-v3"
        metadata = json.load(
            request(f"/outputs/{previous}/{name}-source.json", prep=True)
        )
        with request(f"/outputs/{previous}/{name}-raw.bin.gz", prep=True) as response:
            raw = response.read(32 * 1024 * 1024 + 1)
        body = raw if name == "human-peptides" else gzip.decompress(raw)
        if (
            len(body) != metadata["bytes"]
            or hashlib.sha256(body).hexdigest() != metadata["sha256"]
        ):
            raise ValueError("Prior captured source checksum mismatch")
        metadata = {**metadata, "raw_job": previous}
        captured[name] = body, metadata
        save(name + "-source", metadata)
        save("progress-" + name, {"state": "captured", "source": name})
    PHASE = "canonical-identity"
    hgnc = list(
        csv.DictReader(
            io.StringIO(captured["hgnc"][0].decode("utf-8-sig")), delimiter="\t"
        )
    )
    mapping, ambiguous, gene_info = aliases(hgnc)
    gene_sequences = defaultdict(dict)
    unresolved = CounterLike()

    def peptides():
        header, parts = None, []
        with gzip.GzipFile(fileobj=io.BytesIO(captured["human-peptides"][0])) as raw:
            for line in io.TextIOWrapper(raw):
                if line.startswith(">"):
                    if header:
                        yield header, "".join(parts)
                    header, parts = line.strip(), []
                else:
                    parts.append(line.strip())
        if header:
            yield header, "".join(parts)

    for header, sequence in peptides():
        match = re.search(r" gene:(ENSG\d+)(?:\.\d+)?(?:\s|$)", header)
        if not match:
            raise ValueError("Missing Ensembl peptide gene")
        gene = mapping.get(match[1])
        symbol = re.search(r" gene_symbol:(\S+)", header)
        if gene is None and symbol:
            candidate = mapping.get(symbol[1])
            if candidate and candidate in ambiguous.get(match[1], []):
                gene = candidate
        accession = header.split()[0][1:]
        if not gene:
            unresolved.add("human-peptides", accession)
            continue
        sequence = sequence.rstrip("*")
        if not sequence:
            unresolved.add("empty-human-peptide", accession)
            continue
        if not re.fullmatch("[A-Z]+", sequence):
            unresolved.add(
                "non-amino-acid-symbols-replaced-with-X",
                {
                    "accession": accession,
                    "symbols": sorted(set(re.findall("[^A-Z]", sequence))),
                },
            )
            sequence = re.sub("[^A-Z]", "X", sequence)
        gene_sequences[gene][accession] = {
            "sequence": sequence,
            "reviewed": "Ensembl116",
        }
    for source in ("yeast-proteins",):
        text = captured[source][0].decode("utf-8-sig")
        for row in csv.DictReader(io.StringIO(text), delimiter="\t"):
            if source == "human-proteins":
                gene = mapping.get(row["Gene Names (primary)"])
            else:
                candidates = re.findall(
                    r"\bY[A-P][LR]\d{3}[CW](?:-[A-Z])?\b",
                    row["Gene Names (ordered locus)"],
                )
                gene = "559292:" + candidates[0] if len(set(candidates)) == 1 else None
            if not gene:
                unresolved.add(source, row["Entry"])
                continue
            sequence = row["Sequence"]
            if not sequence or not re.fullmatch("[A-Z]+", sequence):
                raise ValueError("Invalid protein sequence")
            gene_sequences[gene][row["Entry"]] = {
                "sequence": sequence,
                "reviewed": row["Reviewed"],
            }
    out = Shards(
        {"sources": {k: v[1] for k, v in captured.items()}, "role": "static-only"},
        "sequence-inputs",
    )
    types = sorted({r["locus_group"] for r in gene_info.values()})
    if len(types) > 60:
        raise ValueError("Annotation schema capacity exceeded")
    native = json.load(
        request("/outputs/native-r2-20260912-v1/costanzo-manifest.json", prep=True)
    )
    yeast_roster = {"559292:" + g for g in native["genes"]}
    for gene in sorted(set(gene_info) | set(gene_sequences) | yeast_roster):
        entries = gene_sequences[gene]
        # Prefer reviewed canonical entries, then longest sequence; pin the exact
        # selection and retain every candidate accession and sequence hash.
        ranked = sorted(
            entries,
            key=lambda a: (
                entries[a]["reviewed"] != "reviewed",
                -len(entries[a]["sequence"]),
                a,
            ),
        )
        accession = ranked[0] if ranked else None
        sequence = entries[accession]["sequence"] if accession else None
        annotation = [0.0] * 64
        annotation[0] = float(gene.startswith("9606:"))
        annotation[1] = float(gene.startswith("559292:"))
        if gene in gene_info:
            annotation[2 + types.index(gene_info[gene]["locus_group"])] = 1.0
        out.add(
            {
                "gene": gene,
                "targets": [gene],
                "sequence": sequence,
                "accession": accession,
                "annotation": annotation,
                "known": [sequence is not None, True],
                "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest()
                if sequence
                else None,
                "candidate_sequences": {
                    a: hashlib.sha256(v["sequence"].encode()).hexdigest()
                    for a, v in entries.items()
                },
            }
        )
    summary = out.finish(
        annotation_locus_groups=types,
        sequence_selection="reviewed-first, longest, accession order",
        missing_sequences=sum(not gene_sequences[g] for g in gene_info),
        fitted_human_intervention_genes=[],
    )
    roster = list(
        csv.DictReader(io.StringIO(captured["musl-roster"][0].decode("utf-8-sig")))
    )
    resolved, missing = [], []
    for row in roster:
        stable = row["ensembl_gene_id"].split(".")[0]
        stable_gene, symbol_gene = mapping.get(stable), mapping.get(row["symbol"])
        if stable_gene and symbol_gene and stable_gene != symbol_gene:
            missing.append(
                {
                    **row,
                    "reason": "stable-symbol-disagreement",
                    "candidates": [stable_gene, symbol_gene],
                }
            )
            continue
        gene = stable_gene or symbol_gene
        if gene:
            resolved.append({**row, "canonical": gene})
        else:
            missing.append(row)
    save(
        "aliases",
        {
            "human": mapping,
            "ambiguous": ambiguous,
            "source_sha256": captured["hgnc"][1]["sha256"],
        },
    )
    save(
        "musl-roster",
        {
            "rows": resolved,
            "unresolved": missing,
            "original_rows": len(roster),
            "source": captured["musl-roster"][1],
        },
    )
    PHASE = "harle-measurements"
    harle = harle_measurements()
    save(
        "complete",
        {
            "job": os.environ["SLP_JOB"],
            "state": "complete",
            "summary": summary,
            "harle": harle,
            "unresolved_proteins": unresolved.value,
            "unresolved_benchmark_genes": len(missing),
            "fitting_ready": False,
            "pending": [
                "Frozen ESM extraction",
                "Fold-specific quantitative and SL views",
            ],
        },
    )


class CounterLike:
    def __init__(self):
        self.value = defaultdict(list)

    def add(self, source, value):
        self.value[source].append(value)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        save(
            "failed",
            {
                "state": "failed",
                "error_type": type(exc).__name__,
                "http_status": getattr(exc, "code", None),
                "phase": PHASE,
                "detail": str(exc)[:300] if isinstance(exc, ValueError) else None,
            },
        )
        raise SystemExit(1) from None
