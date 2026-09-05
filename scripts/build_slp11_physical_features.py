"""Append direct human physical-neighborhood information to frozen static features."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"modules/slp-1-1-world-transition-v1"))
from physical_features import neighborhood_features

BASE = Path("data/derived/slp11-human-gwps-static/ensembl116-goa2022-fixed-basis-v1/gwps-extended-static-esm-go-features.npz")
SOURCE = Path("data/sources/string-human-physical-v12.0")
DESTINATION = Path("data/derived/slp11-human-physical/direct-experiments700-v1")
HASHES = {
    "9606.protein.aliases.v12.0.txt.gz":"b65f730b993ed0c1bd72edf4565d3d425db42861101b29699704810e8f125680",
    "9606.protein.physical.links.full.v12.0.txt.gz":"b28f494f58e1ace634ef1fe41734ada5be37f151e3168bb9658bc6ca1dd1a954",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    if sha(BASE) != "a2f3153478c00c191e5a9e218badb3327a180a56948a4c9c6a6926cc506ff02b":
        raise ValueError("static base drift")
    for name, expected in HASHES.items():
        if sha(SOURCE/name) != expected:
            raise ValueError("physical source drift")
    with np.load(BASE, allow_pickle=False) as archive:
        data = {name:archive[name] for name in archive.files}
    if not np.all(data["entity_taxon"] == 9606):
        raise ValueError("human physical graph requires human features")
    lookup = {str(gene):index for index,gene in enumerate(data["entity_id"])}
    mapping = defaultdict(set)
    with gzip.open(SOURCE/"9606.protein.aliases.v12.0.txt.gz", "rt", encoding="utf-8") as stream:
        next(stream)
        for line in stream:
            protein, gene, source = line.rstrip("\n").split("\t", 2)
            if protein.startswith("9606.") and source == "Ensembl_gene" and re.fullmatch(r"ENSG\d{11}", gene):
                mapping[protein].add(gene)
    exact = {protein:next(iter(genes)) for protein,genes in mapping.items() if len(genes) == 1}
    edges = []
    strong_rows = 0
    with gzip.open(SOURCE/"9606.protein.physical.links.full.v12.0.txt.gz", "rt", encoding="utf-8") as stream:
        columns = next(stream).split()
        experiment_column = columns.index("experiments")
        for line in stream:
            fields = line.split()
            confidence = int(fields[experiment_column])
            if confidence < 700:
                continue
            strong_rows += 1
            left, right = exact.get(fields[0]), exact.get(fields[1])
            if left in lookup and right in lookup:
                edges.append((lookup[left], lookup[right], confidence/1000))
    original = data["feature_values"]
    data["feature_values"], summary = neighborhood_features(original, edges)
    if not np.array_equal(data["feature_values"][:, :original.shape[1]], original):
        raise ValueError("base feature rows changed")
    DESTINATION.mkdir(parents=True, exist_ok=False)
    output = DESTINATION/"human-esm-go-physical-features.npz"
    np.savez_compressed(output, **data)
    manifest = {
        "inputs":{"base":sha(BASE), **HASHES}, "output_sha256":sha(output),
        "taxon":9606, "entities":len(lookup), **summary,
        "protein_mapping":"unique exact Ensembl_gene aliases; ambiguous mappings excluded",
        "ambiguous_proteins":sum(len(genes) != 1 for genes in mapping.values()),
        "direct_experiment_rows_above_threshold":strong_rows,
        "edge_rule":"STRING12.0 physical subnetwork, direct human experiments>=700/1000 only; collapse duplicate gene pairs by maximum; no self edge",
        "excluded_channels":["textmining", "database", "experiments_transferred", "database_transferred", "homology", "combined_score"],
        "feature_rule":"retain original577; append confidence-weighted known-neighbor mean577, log1p induced degree and neighbor-presence flag",
        "limitation":"induced graph on cached genes; physical associations are not direction, causality or quantitative perturbation outcomes; static2023 graph used for retrospective modeling",
        "molecular_outcomes_accessed":False, "source_hashes":{p.name:sha(p) for p in (Path(__file__), Path(__file__).resolve().parents[1]/"modules/slp-1-1-world-transition-v1/physical_features.py")},
    }
    (DESTINATION/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
