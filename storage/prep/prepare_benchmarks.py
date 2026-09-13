"""Capture official CV3 splits in Cloudflare; never execute publisher pickles.

Feng's 25 GB uncompressed archive is streamed and only the exact split/roster
members are retained. Labels remain evaluation data until a fold admits them.
"""

import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import urllib.request

from inspect_corpus import download, request, save
from normalize_corpus import Shards
from safe_arrays import load, describe

IDENTITY = "identity-r2-20260912-v5"
MUSL = "https://raw.githubusercontent.com/JieZheng-ShanghaiTech/MuSL/f8021cfc618fafae8c330b694d0fa7c46db5f1a5/processed_data/data/CV3_bins_32/fold_data/"
MUSL_HASHES = {
    "test_labels_seed42.pkl": "7f269973330c2e64ca85f13bfe027a93fc5408a0",
    "test_labels_seed432.pkl": "d266ec6b0f5c975cfc66122018a9084a9ccaa8fa",
    "test_pairs_seed42.pkl": "4c800c64942737346d16ac16cf5dde12b0a1801d",
    "test_pairs_seed432.pkl": "89c97eaf0a4105f99c38e8b5a4ba17e6cfa55266",
    "train_labels_seed42.pkl": "1fd55ef219a32b1c271d9a7f831dd7c64ad5aa48",
    "train_labels_seed432.pkl": "37d3a0a342071429315bb08f6bddf89bb7908739",
    "train_pairs_seed42.pkl": "98d1f669e2ee3af5e8845874f65a582adf423b9b",
    "train_pairs_seed432.pkl": "899871bedd57f60461486f8c4800d87a831790d1",
}
PHASE = "starting"


def put_bytes(name, body):
    payload = gzip.compress(body, compresslevel=3, mtime=0)
    if len(payload) > 32 * 1024 * 1024:
        raise ValueError("Captured member exceeds output cap: " + name)
    with request(
        f"/outputs/{os.environ['SLP_JOB']}/{name}.bin.gz", data=payload, prep=True
    ) as response:
        response.read(4096)
    return {
        "name": name + ".bin.gz",
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }


def fetch(url, limit=16 * 1024 * 1024):
    req = urllib.request.Request(
        url, headers={"User-Agent": "SLp-Corpus-Preparation/1.0"}
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        body = response.read(limit + 1)
    if len(body) > limit:
        raise ValueError("Publisher split exceeds pinned capture limit")
    return body


def resolve_roster(rows, aliases):
    resolved, missing = {}, []
    for row in rows:
        stable = row.get("ensembl_gene_id", "").split(".")[0]
        symbols = [row.get("symbol", "")]
        matches = {aliases[v] for v in [stable, *symbols] if v in aliases}
        if len(matches) != 1:
            missing.append({**row, "candidates": sorted(matches)})
        else:
            resolved[int(row["unified_id"])] = next(iter(matches))
    return resolved, missing


def emit_fold(name, partitions, *, benchmark, protocol):
    sets, outputs, counts = {}, {}, {}
    for partition, rows in partitions.items():
        # Preserve the official row order and multiplicity for exact evaluation.
        out = Shards(
            {"benchmark": benchmark, "protocol": protocol}, name + "-" + partition
        )
        genes, seen, conflicts = set(), {}, 0
        for i, row in enumerate(rows):
            a, b, y = row[:3]
            if a == b or y not in (0, 1):
                raise ValueError("Invalid official SL pair")
            genes.update((a, b))
            pair = tuple(sorted((a, b)))
            if pair in seen and seen[pair] != y:
                conflicts += 1
            seen[pair] = y
            out.add(
                {
                    "row": i,
                    "targets": [a, b],
                    "label": int(y),
                    "context": protocol.get("context", "pan-cancer"),
                    **({"score": float(row[3])} if len(row) > 3 else {}),
                }
            )
        summary = out.finish(
            partition=partition,
            unique_pairs=len(seen),
            contradictory_duplicate_rows=conflicts,
            role="benchmark-labels",
            duplicate_policy="preserve official order, multiplicity and labels; no outcome-based cleaning",
        )
        outputs[partition] = out.family + "-manifest.json"
        counts[partition] = summary
        sets[partition] = genes
    overlap = sets["train"] & sets["test"]
    train_genes = sets["train"] - sets["test"] - sets.get("valid", set())
    # Inner development split derives only from outer-training gene identities.
    inner = (sets.get("valid", set()) - sets["test"]) or {
        g
        for g in train_genes
        if int.from_bytes(
            hashlib.sha256((name + ":731:" + g).encode()).digest()[:8], "big"
        )
        % 5
        == 0
    }
    result = {
        "name": name,
        "benchmark": benchmark,
        "protocol": protocol,
        "partitions": outputs,
        "counts": counts,
        "outer_held": sorted(sets["test"]),
        "inner_held": sorted(inner),
        "outer_train_genes": sorted(sets["train"] - sets["test"]),
        "official_canonical_train_test_gene_overlap": sorted(overlap),
        "strict_human_exposure_rule": "exclude every target in outer_held or inner_held before fitting, including any official rows whose aliases overlap",
    }
    save(name + "-fold", result)
    return {
        "name": name,
        "manifest": name + "-fold.json",
        "benchmark": benchmark,
        "counts": counts,
    }


def musl(aliases):
    roster = json.load(request(f"/outputs/{IDENTITY}/musl-roster.json", prep=True))
    if roster["unresolved"]:
        save("musl-unresolved", roster["unresolved"])
        return [], {"unresolved_genes": len(roster["unresolved"])}
    # MuSL's tensors index the ordered 7,684-row table. Its unified_id column
    # retains sparse identifiers from a larger source universe and is not the
    # tensor index. Preserve every row, including the original order.
    mapping = {i: r["canonical"] for i, r in enumerate(roster["rows"])}
    values, receipts = {}, {}
    for name, expected in MUSL_HASHES.items():
        body = fetch(MUSL + name)
        actual = hashlib.sha1(
            b"blob " + str(len(body)).encode() + b"\0" + body
        ).hexdigest()
        if actual != expected:
            raise ValueError("MuSL Git object mismatch")
        receipts[name] = put_bytes("musl-" + name.replace(".", "-"), body)
        values[name] = load(body)
    result = []
    for seed in (42, 432):
        for fold in range(5):
            partitions = {}
            for partition in ("train", "test"):
                pairs = values[f"{partition}_pairs_seed{seed}.pkl"][fold]
                labels = values[f"{partition}_labels_seed{seed}.pkl"][fold]
                if len(pairs) != len(labels):
                    raise ValueError("MuSL pairs/labels mismatch")
                partitions[partition] = [
                    (mapping[int(a)], mapping[int(b)], int(y))
                    for (a, b), y in zip(pairs, labels)
                ]
            result.append(
                emit_fold(
                    f"musl-s{seed}-f{fold}",
                    partitions,
                    benchmark="MuSL-CV3",
                    protocol={
                        "seed": seed,
                        "fold": fold,
                        "negatives": "official CV3_bins_32",
                        "repository_revision": "f8021cfc618fafae8c330b694d0fa7c46db5f1a5",
                    },
                )
            )
    return result, receipts


def feng():
    manifest = json.load(request("/manifest"))
    specs = sorted(
        (r for r in manifest["objects"] if r["id"].startswith("feng-2024-")),
        key=lambda x: x["id"],
    )
    if len(specs) != 5:
        raise ValueError("Five pinned Feng archive parts required")
    captured, listing = {}, []
    with tempfile.TemporaryDirectory() as directory:
        archive_path = Path(directory) / "feng.tar.gz"
        digest = hashlib.md5()
        with archive_path.open("wb") as dst:
            for spec in specs:
                part = download(spec, directory)
                with part.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                        dst.write(chunk)
                part.unlink()
                save(
                    "progress-" + spec["id"],
                    {"state": "assembled", "source": spec["id"]},
                )
        if digest.hexdigest() != "5a36306fac1e31352c6d4645c8a57a6a":
            raise ValueError("Concatenated Feng archive MD5 mismatch")
        with tarfile.open(archive_path, "r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                if "CV3" in member.name or member.name.endswith("meta_table_9845.csv"):
                    listing.append({"name": member.name, "bytes": member.size})
                wanted = member.name.endswith("meta_table_9845.csv") or (
                    "data_split" in member.name
                    and Path(member.name).name in ("CV3_1.npy", "CV3_1.pkl")
                )
                if not wanted:
                    continue
                if member.size > 256 * 1024 * 1024:
                    raise ValueError("Selected Feng member too large")
                with archive.extractfile(member) as stream:
                    body = stream.read()
                key = "feng-" + member.name.replace("/", "-").replace(".", "-")
                receipt = put_bytes(key, body)
                captured[member.name] = receipt
                if member.name.endswith(".csv"):
                    captured[member.name]["columns"] = next(
                        csv.reader(io.StringIO(body.decode()))
                    )
                else:
                    captured[member.name]["structure"] = describe(
                        load(body, numpy_file=member.name.endswith(".npy"))
                    )
        save(
            "feng-inventory",
            {
                "members": listing,
                "captured": captured,
                "archive_md5": digest.hexdigest(),
            },
        )
    return captured


def slamr(aliases):
    base = "https://raw.githubusercontent.com/Rrrrachellll/SLAMR/90eb8a2236ca2ce222e00e3d6edb9b5f58643713/data_slb_filtered/"
    outputs, missing = [], set()
    for study, context in (
        ("28319113", "A549"),
        ("30033366", "JURKAT"),
        ("30033366", "K562"),
    ):
        url = base + f"{study}/{context}_scenario3_fold5_seed88.pkl"
        body = fetch(url)
        receipt = put_bytes("slamr-" + context + "-official", body)
        folds = load(body)
        if len(folds) != 5:
            raise ValueError("SLAMR fold count changed")
        for i, (train, valid, test) in enumerate(folds):
            native = {
                "train": train,
                "valid": [
                    (a, b, s, y)
                    for a, partners in valid.items()
                    for b, s, y in partners
                ],
                "test": [
                    (a, b, s, y) for a, partners in test.items() for b, s, y in partners
                ],
            }
            unresolved = {
                g
                for rows in native.values()
                for row in rows
                for g in row[:2]
                if g not in aliases
            }
            missing.update(unresolved)
            if unresolved:
                continue
            partitions = {
                k: [
                    (aliases[a], aliases[b], int(y == "SL"), float(s))
                    for a, b, s, y in rows
                ]
                for k, rows in native.items()
            }
            outputs.append(
                emit_fold(
                    f"slamr-{context}-f{i}",
                    partitions,
                    benchmark="SLAMR-scenario3-" + context,
                    protocol={
                        "seed": 88,
                        "fold": i,
                        "context": context,
                        "study": study,
                        "raw": receipt,
                        "source_url": url,
                        "metric": "partner ranking, reported separately",
                    },
                )
            )
    save("slamr-unresolved", sorted(missing))
    return outputs


def main():
    global PHASE
    aliases = json.load(request(f"/outputs/{IDENTITY}/aliases.json", prep=True))[
        "human"
    ]
    PHASE = "musl"
    folds, receipts = musl(aliases)
    save("musl-complete", {"folds": folds, "sources": receipts})
    PHASE = "slamr"
    folds += slamr(aliases)
    save("small-benchmarks-complete", {"folds": folds})
    PHASE = "feng"
    inventory = feng()
    save(
        "complete",
        {
            "state": "captured",
            "folds": folds,
            "feng": inventory,
            "fitting_ready": False,
            "pending": [
                "Feng inspected-array normalization",
                "complete canonical identity audit",
            ],
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        save(
            "failed",
            {
                "state": "failed",
                "phase": PHASE,
                "type": type(exc).__name__,
                "http_status": getattr(exc, "code", None),
                "detail": str(exc)[:400]
                if isinstance(exc, (ValueError, KeyError))
                else None,
            },
        )
        raise SystemExit(1) from None
