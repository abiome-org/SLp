"""Write ESM4SL-format train/val/test CSVs for one species from SLB.

usage: python prep.py <species> <split> <workdir> <mode: whole|mean> [val_gene_frac=0.05]

* labels: SLB train.parquet only.  Validation (checkpoint selection / early stopping, as the original's
  sl_val_*.csv) = train rows touching a random `val_gene_frac` of that species' train genes (gene-held-out,
  seed 0); the remaining train rows are used for fitting.  The eval split is never used for fitting/selection.
* genes -> int indices (the original keys embeddings by Entrez int); columns 0,1,2(label),ctx,example_id.
* whole: writes <workdir>/emb/index.tsv (idx, gene, path) pointing at per-residue files from embed.py
  mean : writes <workdir>/emb.parquet (idx, gene, e0..e1279) from data/interim/esm2_650m/<species>.parquet
Rows whose genes have no embedding are dropped from train and left unscored in test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import slb  # noqa: E402

PERRES = slb.ROOT / "external/models/esm4sl/_slb/perres"
MEAN = slb.ROOT / "data/interim/esm2_650m"


def main(sp, split, wd, mode, frac=0.05):
    wd = Path(wd)
    wd.mkdir(parents=True, exist_ok=True)
    if mode == "whole":
        have = {p.stem: p for p in (PERRES / sp).glob("*.pt")}
    else:
        m = pd.read_parquet(MEAN / f"{sp}.parquet")
        have = dict.fromkeys(m.gene.astype(str))
    tr = slb.load("train")
    te = slb.load(split)
    tr, te = tr[tr.species == sp], te[te.species == sp]
    genes = sorted((set(tr.gene_a) | set(tr.gene_b) | set(te.gene_a) | set(te.gene_b)) & set(have))
    idx = {g: i for i, g in enumerate(genes)}
    if mode == "whole":
        pd.DataFrame({"idx": range(len(genes)), "gene": genes, "path": [str(have[g]) for g in genes]}) \
            .to_csv(wd / "index.tsv", sep="\t", index=False)
    else:
        m = m[m.gene.isin(idx)].copy()
        m.insert(0, "idx", m.gene.map(idx))
        m.to_parquet(wd / "emb.parquet", index=False)

    def fmt(d, labelled=True):
        d = d[d.gene_a.isin(idx) & d.gene_b.isin(idx)]
        return pd.DataFrame({"0": d.gene_a.map(idx).values, "1": d.gene_b.map(idx).values,
                             "2": (d.label.fillna(0).astype(int).values if labelled and "label" in d else 0),
                             "ctx": d.context_id.values, "example_id": d.example_id.values})

    tr = tr[tr.label.notna()]
    tg = sorted(set(tr.gene_a) | set(tr.gene_b))
    rng = np.random.default_rng(0)
    vg = set(rng.choice(tg, size=max(1, int(frac * len(tg))), replace=False))
    isv = tr.gene_a.isin(vg) | tr.gene_b.isin(vg)
    fmt(tr[~isv]).to_csv(wd / "train.csv", index=False)
    fmt(tr[isv]).to_csv(wd / "val.csv", index=False)
    t = fmt(te, labelled=False)
    # dummy alternating labels: the harness computes AUROC on the test set at the end and crashes on one class;
    # eval-split labels are never given to the model (its in-harness test metrics are meaningless by design)
    t["2"] = [i % 2 for i in range(len(t))]
    t.to_csv(wd / "test.csv", index=False)
    print(f"{sp}: train {(~isv).sum()} (pos {int(tr[~isv].label.sum())}), val {isv.sum()} "
          f"(pos {int(tr[isv].label.sum())}), {split} scorable {len(t)}/{len(te)}", file=sys.stderr)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], a[3], *(float(x) for x in a[4:]))
