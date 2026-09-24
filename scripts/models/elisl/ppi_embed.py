"""node2vec embeddings of the human PPI graph (ELISL's PPI source), cached per gene.

ELISL: STRING v11.0 edges with experiments>0 OR database>0 (unweighted, undirected), node2vec (node2vec==0.3.3)
dimensions=64, walk_length=30, num_walks=200, p=q=1, then gensim Word2Vec(window=10, min_count=1, sg=1).
SLB deviation (leakage contract): the STRING *experiments* channel imports interaction-database records that
include genetic-interaction evidence, so it is replaced by BioGRID physical-only interactions (shared bundle,
data/interim/bundle/human/ppi.parquet column biogrid_phys). STRING v11.0 'database' channel is kept as in ELISL.
With p=q=1 node2vec's biased walk is a uniform random walk, so walks are generated directly (vectorised numpy)
instead of via the node2vec package's O(E*deg) transition-probability precomputation; Word2Vec settings are
the package defaults ELISL used (gensim 3.8.3, sg=1, window 10, min_count 1, 5 epochs), except batch_words
(ELISL: 4, a throughput-only setting) left at gensim's default. Output: _slb/cache/node2vec.parquet.
"""
import gzip
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

import common as C

DIM, WL, NW, WINDOW = 64, 30, 200, 10
SEED = 124


def edges():
    info = pd.read_csv(C.RAW / "string_v11/9606.protein.info.v11.0.txt.gz", sep="\t", usecols=[0, 1])
    info.columns = ["protein", "name"]
    res = C.resolver()
    p2g = {p: res.get(n) for p, n in zip(info.protein, info.name)}
    s = pd.read_csv(C.RAW / "string_v11/9606.protein.links.full.v11.0.txt.gz", sep=" ",
                    usecols=["protein1", "protein2", "database"])
    s = s[s.database > 0]
    e1 = pd.DataFrame({"a": s.protein1.map(p2g), "b": s.protein2.map(p2g)}).dropna()
    b = pd.read_parquet(ROOT_BUNDLE / "ppi.parquet", columns=["gene_a", "gene_b", "biogrid_phys"])
    b = b[b.biogrid_phys > 0]
    e2 = pd.DataFrame({"a": b.gene_a.values, "b": b.gene_b.values})
    e = pd.concat([e1, e2])
    e = e[e.a != e.b]
    lo = np.where(e.a < e.b, e.a, e.b)
    hi = np.where(e.a < e.b, e.b, e.a)
    e = pd.DataFrame({"a": lo, "b": hi}).drop_duplicates()
    C.log(f"ppi edges: STRING v11 database>0 {len(e1):,} (directed rows), BioGRID physical {len(e2):,}; "
          f"union undirected {len(e):,}")
    return e


ROOT_BUNDLE = C.ROOT / "data/interim/bundle/human"


def main():
    out = C.CACHE / "node2vec.parquet"
    if out.exists():
        C.log("node2vec cached")
        return
    e = edges()
    nodes = np.array(sorted(set(e.a) | set(e.b)))
    idx = {g: i for i, g in enumerate(nodes)}
    ia = e.a.map(idx).values
    ib = e.b.map(idx).values
    n = len(nodes)
    A = sp.coo_matrix((np.ones(2 * len(ia)), (np.r_[ia, ib], np.r_[ib, ia])), shape=(n, n)).tocsr()
    indptr, indices = A.indptr, A.indices
    deg = np.diff(indptr)
    C.log(f"graph: {n:,} nodes, {len(e):,} edges")
    rng = np.random.default_rng(SEED)
    corpus = C.WORK / "node2vec_walks.txt"
    done = corpus.with_suffix(".done")
    with (open(os.devnull, "w") if done.exists() else open(corpus, "w")) as f:
        for w in range(0 if done.exists() else NW):
            order = rng.permutation(n)
            walk = np.empty((n, WL), dtype=np.int64)
            walk[:, 0] = order
            for t in range(1, WL):
                cur = walk[:, t - 1]
                walk[:, t] = indices[indptr[cur] + (rng.random(n) * deg[cur]).astype(np.int64)]
            f.write("\n".join(" ".join(nodes[r]) for r in walk))
            f.write("\n")
    done.touch()
    from gensim.models import Word2Vec
    workers = int(os.environ.get("ELISL_CPUS", "8"))
    model = Word2Vec(corpus_file=str(corpus), size=DIM, window=WINDOW, min_count=1, sg=1, workers=workers, seed=SEED)
    emb = pd.DataFrame(np.vstack([model.wv[g] for g in nodes]), index=nodes).astype("float32")
    emb.columns = [str(i) for i in range(DIM)]
    emb.to_parquet(out)
    corpus.unlink()
    done.unlink()
    C.log(f"node2vec: {emb.shape}")


if __name__ == "__main__":
    main()
