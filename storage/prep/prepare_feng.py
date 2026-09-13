"""Canonical official Feng balanced CV3 folds, preserving every published row."""

import csv
import hashlib
import io
import json
import numpy as np
from inspect_corpus import request, save
from inspect_feng import captured
from prepare_benchmarks import emit_fold
from safe_arrays import load


def main():
    split_bytes = captured("feng---data-data_split-CV3_1-npy.bin.gz")
    roster_bytes = captured("feng---data-preprocessed_data-meta_table_9845-csv.bin.gz")
    splits = load(split_bytes, numpy_file=True)
    if splits.shape != (2, 4, 5):
        raise ValueError("Official Feng balanced CV3 shape changed")
    aliases = json.load(
        request("/outputs/identity-r2-20260912-v5/aliases.json", prep=True)
    )["human"]
    roster = list(csv.DictReader(io.StringIO(roster_bytes.decode())))
    mapping = {}
    unresolved = []
    disagreements = []
    for row in roster:
        candidates = {
            aliases[v]
            for v in (
                row["hgnc_id"],
                row["ensembl_gene_id"].split(".")[0],
                row["symbol"],
            )
            if v in aliases
        }
        if row["hgnc_id"] in aliases:
            canonical = aliases[row["hgnc_id"]]
            if candidates != {canonical}:
                disagreements.append(
                    {
                        "unified_id": row["unified_id"],
                        "canonical": canonical,
                        "other_candidates": sorted(candidates - {canonical}),
                    }
                )
        elif len(candidates) == 1:
            canonical = next(iter(candidates))
        else:
            unresolved.append(row)
            continue
        mapping[int(row["unified_id"])] = canonical
    used = {
        int(g)
        for label in range(2)
        for partition in (2, 3)
        for fold in range(5)
        for g in np.asarray(splits[label, partition, fold]).ravel()
    }
    missing = sorted(used - set(mapping))
    save(
        "identity-audit",
        {
            "roster_genes": len(roster),
            "resolved_genes": len(mapping),
            "used_genes": len(used),
            "unresolved_used_indices": missing,
            "unresolved_roster": unresolved,
            "stable_id_disagreements": disagreements,
        },
    )
    if missing:
        raise ValueError(
            "Unresolved genes in official Feng pairs; no row may be omitted"
        )
    folds = []
    for fold in range(5):
        partitions = {}
        for partition, index in (("train", 2), ("test", 3)):
            rows = []
            for label, p in ((1, 0), (0, 1)):
                pairs = np.asarray(splits[p, index, fold])
                if (
                    pairs.ndim != 2
                    or pairs.shape[1] != 2
                    or pairs.dtype.kind not in "iu"
                ):
                    raise ValueError("Feng pair matrix schema changed")
                rows.extend((mapping[int(a)], mapping[int(b)], label) for a, b in pairs)
            partitions[partition] = rows
        folds.append(
            emit_fold(
                f"feng-cv3-random1-f{fold}",
                partitions,
                benchmark="Feng-CV3-random1",
                protocol={
                    "fold": fold,
                    "negatives": "official CV3_1.npy, random negatives, positive:negative 1:1",
                    "code_revision": "04274e801b820a81f8dd92eb362d154086b80d9a",
                    "source_archive_md5": "5a36306fac1e31352c6d4645c8a57a6a",
                    "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
                    "roster_sha256": hashlib.sha256(roster_bytes).hexdigest(),
                    "array_contract": "[positive,negative] x [train-adj,test-adj,train-pairs,test-pairs] x five folds",
                },
            )
        )
    save(
        "complete",
        {"state": "complete", "folds": folds, "research_training_launched": False},
    )


if __name__ == "__main__":
    main()
