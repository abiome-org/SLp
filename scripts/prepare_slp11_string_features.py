#!/usr/bin/env python3
"""Pair SLIM's published STRING vectors with SLp human gene rosters."""
from __future__ import annotations

import argparse, hashlib, json, re
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EMBEDDING = ROOT / "data/tooling/slim-5a7e9ade/data/gene_string_embeddings.v0.3.h5"
GTF = ROOT / "data/sources/replogle-perturbseq-gi-code/data_sharing/cellranger-GRCh38-1.2.0_only_genes.gtf"
OUT = ROOT / "data/derived/slp11-string-embedding-v03"
SOURCES = {
    "k562": ROOT / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1/k562-essential-count-static577.npz",
    "rpe1": ROOT / "data/derived/slp11-human-rpe1-essential-count-static/ensembl116-esm8m-shared-go-v1/rpe1-essential-count-static577.npz",
    "norman": ROOT / "data/derived/slp11-norman-static/ensembl116-goa2022-fixed-basis-v1/norman-extended-static-esm-go-features.npz",
}

def digest(p):
    with open(p, "rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()

def mapping(path):
    answer = {}
    pattern = re.compile(r'gene_id "([^"]+)".*gene_name "([^"]+)"')
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"): continue
            match = pattern.search(line)
            if match: answer.setdefault(match.group(1).split(".")[0], match.group(2))
    return answer

def main(out):
    out.mkdir(parents=True, exist_ok=False)
    names = mapping(GTF)
    report = {"schema":"slp.string-embedding-feature-pack-report/v1", "embeddingSha256":digest(EMBEDDING),
              "embeddingRevision":"SLIM git 5a7e9ade5d0a6b6331e6dbc81181450605047bcc",
              "license":"MIT in the pinned SLIM repository; cite Hu et al., npj Systems Biology and Applications (2026), DOI 10.1038/s41540-026-00746-8",
              "identifierMapping":"Ensembl stable gene ID -> gene_name from the Replogle source Cell Ranger GRCh38 GTF; exact symbol lookup in the SLIM HDF5", "sources":{}}
    with h5py.File(EMBEDDING, "r") as h5:
        for label, source in SOURCES.items():
            with np.load(source, allow_pickle=False) as z: ids=z["entity_id"].astype(str); tax=z["entity_taxon"].astype(np.int64)
            symbols=np.asarray([names.get(x, "") for x in ids]); present=np.asarray([s in h5 for s in symbols])
            values=np.zeros((len(ids),64),np.float32)
            for i in np.flatnonzero(present): values[i]=h5[symbols[i]][:]
            np.savez_compressed(out/f"{label}-string64.npz", schema=np.asarray("slp.string64-feature-pack/v1"), entity_id=ids,
                entity_taxon=tax, gene_symbol=symbols, feature_values=values, feature_present=present,
                feature_source_sha256=np.asarray(report["embeddingSha256"]))
            report["sources"][label]={"entities":len(ids),"mappedSymbols":int(np.count_nonzero(symbols!="")),"covered":int(present.sum()),"coverage":float(present.mean()),"sourceSha256":digest(source)}
    (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    rights={"schema":"slp.data-rights/v1","trainingAllowed":True,"redistributionAllowed":None,
            "basis":"The exact HDF5 is tracked in the SLIM repository under its root MIT LICENSE with no stated carveout. This does not assert a separate upstream license for the Hu et al. vectors.",
            "requiredAttribution":["RasmussenLab/SLIM commit 5a7e9ade5d0a6b6331e6dbc81181450605047bcc","Hu et al., Molecular maps of diseases from omics data and network embeddings, DOI 10.1038/s41540-026-00746-8"]}
    (out/"rights.json").write_text(json.dumps(rights,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report["sources"],sort_keys=True))

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,default=OUT); main(p.parse_args().output)
