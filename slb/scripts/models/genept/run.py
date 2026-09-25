"""GenePT pair classifier (Chen & Zou 2024, "GenePT: a simple but hard-to-beat foundation model for genes and
cells built from ChatGPT", bioRxiv 10.1101/2023.10.16.562533; embeddings Zenodo 10833191, CC BY 4.0).

Gene embedding = OpenAI text-embedding-ada-002 of the NCBI gene summary (1536-d). The 33,703 summaries contain no
'synthetic lethal' mentions (checked), so the inputs carry no SL labels. Pair features: PCA(128) of the embeddings
(fit on all genes), then [a*b, |a-b|, cosine]; LightGBM (n_jobs 8) fitted on SLB human train rows only.
Variant genept__slbtrain. Human only (NCBI human gene summaries).
"""
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import slb  # noqa: E402

EMB = slb.ROOT / "external/models/genept/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle"


def pair_x(d, Z, gi, cos):
    ia = d.gene_a.map(gi)
    ib = d.gene_b.map(gi)
    ok = (ia.notna() & ib.notna()).to_numpy()
    X = np.full((len(d), 2 * Z.shape[1] + 1), np.nan, np.float32)
    a, b = Z[ia[ok].astype(int)], Z[ib[ok].astype(int)]
    X[ok] = np.hstack([a * b, np.abs(a - b), (cos[ia[ok].astype(int)] * cos[ib[ok].astype(int)]).sum(1, keepdims=True)])
    return X, ok


def main(split):
    e = pickle.load(open(EMB, "rb"))
    res = slb.symbol_resolver()
    genes, vecs = [], []
    for g, v in e.items():
        s = res(g) or g
        genes.append(s)
        vecs.append(np.asarray(v, np.float32))
    E = np.vstack(vecs)
    df = pd.DataFrame(E, index=genes)
    df = df[~df.index.duplicated()]
    E = df.to_numpy()
    cos = E / np.linalg.norm(E, axis=1, keepdims=True)
    Z = PCA(128, random_state=0).fit_transform(E).astype(np.float32)
    gi = {g: i for i, g in enumerate(df.index)}
    tr = slb.load("train")
    tr = tr[tr.species == "human"]
    ev = slb.load(split)
    h = ev[ev.species == "human"]
    Xt, okt = pair_x(tr, Z, gi, cos)
    Xe, oke = pair_x(h, Z, gi, cos)
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.5, n_jobs=8, verbose=-1, random_state=0)
    m.fit(Xt[okt], tr.label.to_numpy()[okt])
    out = ev[["example_id"]].copy()
    out["score"] = np.nan
    s = np.full(len(h), np.nan)
    s[oke] = m.predict_proba(Xe[oke])[:, 1]
    out.loc[h.index, "score"] = s
    print(f"train rows {okt.sum():,}; {split} human rows with embeddings {oke.sum():,}/{len(h):,}", file=sys.stderr)
    slb.write(out, "genept__slbtrain", split)
    slb.evaluate("genept__slbtrain", split)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else slb.SPLIT)
