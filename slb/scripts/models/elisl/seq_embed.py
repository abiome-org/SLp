"""SeqVec per-protein embeddings (ELISL's sequence source) for all human SLB genes of the given splits.

One protein per gene: UniProt reviewed canonical sequence, chosen via HGNC uniprot_ids (fallback: UniProt primary
gene name). Embeddings are computed with the original `seqvec` CLI (seqvec==0.4.1, allennlp 0.9, --protein =
mean over residues of the summed 3 ELMo layers) inside docker image slb/elisl-seqvec, and cached per gene in
external/models/elisl/_slb/cache/seqvec.parquet (index gene, 1024 float32 columns). Only missing genes are embedded.

usage: python seq_embed.py <split> [<split> ...]      (env ELISL_GPU=1 to use the GPU; claim it on the board first)
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

import common as C


def gene2uniprot(genes):
    seqs = {}
    acc = None
    for line in open(C.RAW / "uniprot_human/uniprot_reviewed_9606.fasta"):
        if line.startswith(">"):
            acc = line.split("|")[1]
            seqs[acc] = []
        else:
            seqs[acc].append(line.strip())
    seqs = {k: "".join(v) for k, v in seqs.items()}
    h = pd.read_csv(C.RAW / "ids/hgnc_complete_set.txt", sep="\t", low_memory=False, usecols=["symbol", "uniprot_ids"])
    hg = {s: [a for a in str(u).split("|") if a in seqs] for s, u in zip(h.symbol, h.uniprot_ids) if isinstance(u, str)}
    up = pd.read_csv(C.RAW / "uniprot_human/uniprot_reviewed_9606_genes.tsv", sep="\t")
    prim = {}
    for a, g in zip(up["Entry"], up["Gene Names (primary)"]):
        if isinstance(g, str):
            for x in g.split(";"):
                prim.setdefault(x.strip(), a)
    out = {}
    for g in genes:
        a = (hg.get(g) or [None])[0] or prim.get(g)
        if a is not None:
            out[g] = (a, seqs[a])
    return out


def main(splits):
    genes = C.genes_of(*splits)
    p = C.CACHE / "seqvec.parquet"
    have = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    todo = [g for g in genes if g not in have.index]
    m = gene2uniprot(todo)
    C.log(f"seqvec: {len(genes)} genes, cached {len(genes) - len(todo)}, to embed {len(m)} "
          f"(no UniProt reviewed protein: {len(todo) - len(m)})")
    if not m:
        return
    wd = C.WORK / "seqvec_tmp"
    wd.mkdir(parents=True, exist_ok=True)
    with open(wd / "in.fasta", "w") as f:
        for g, (a, s) in m.items():
            f.write(f">{g}\n{s}\n")
    model = C.RAW / "seqvec"
    cmd = ["docker", "run", "--rm", "-v", f"{wd}:/w", "-v", f"{model}:/model:ro"]
    gpu = os.environ.get("ELISL_GPU") == "1"
    if gpu:
        cmd += ["--gpus", "all"]
    else:
        cmd += ["--cpus", os.environ.get("ELISL_CPUS", "8"), "-e", "OMP_NUM_THREADS=" + os.environ.get("ELISL_CPUS", "8")]
    cmd += ["slb/elisl-seqvec", "seqvec", "-i", "/w/in.fasta", "-o", "/w/out.npz", "--model", "/model", "--protein",
            "--split-char", " ", "--id", "0", "--silent"]
    if not gpu:
        cmd += ["--cpu"]
    C.log(" ".join(cmd))
    subprocess.run(cmd, check=True)
    e = dict(np.load(wd / "out.npz"))
    new = pd.DataFrame.from_dict(e, orient="index").astype("float32")
    new.columns = [str(i) for i in range(new.shape[1])]
    allv = pd.concat([have, new]) if len(have) else new
    allv = allv[~allv.index.duplicated()]
    allv.to_parquet(p)
    pd.DataFrame([(g, a) for g, (a, _) in m.items()], columns=["gene", "uniprot"]).to_csv(
        C.CACHE / "seqvec_gene2uniprot.tsv", sep="\t", index=False, mode="a", header=not (C.CACHE / "seqvec_gene2uniprot.tsv").exists())
    C.log(f"seqvec cache now {len(allv)} genes")


if __name__ == "__main__":
    main(sys.argv[1:] or ["train", "dev"])
