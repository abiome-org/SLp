"""Cell-line feature matrices for ESM4SL's cell branch (CellCNN expects [n_genes=4079, 6 channels]).

usage: python cellfeat.py <out.npz>

The original ESM4SL used SynergyX's pre-built 985-cell-line tensor (4079 genes x {exp, mut, cn, eff, dep, met},
z-normalised; file not distributed, only clname2embed.npy for 170 lines, most SLB lines missing).  We rebuild an
analogous tensor for EVERY human SLB context from DepMap 24Q4 so all contexts are treated identically:
  genes    : the 4079 genes with the highest variance of CRISPR gene effect across DepMap lines, among genes present
             in the expression, copy-number and gene-effect matrices (sorted by symbol)
  channels : 0 expression log2(TPM+1) z-scored across DepMap lines, 1 damaging somatic mutation (0/1),
             2 copy number z-scored, 3 CRISPR gene effect z-scored, 4 expression raw / 10, 5 gene effect raw
             (SynergyX's 'dep' (dependency probability) and 'met' (methylation) are not available in our DepMap
             download; channels 4-5 replace them)
Missing data (line absent from a matrix) -> 0 in that channel.  Contexts with no DepMap model get an all-zero tensor.
Context -> DepMap mapping: contexts.parquet depmap_id; if missing, PROXIES below (documented approximations).
Single-gene DepMap data only (no pair labels) -> allowed by the SLB leakage contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import slb  # noqa: E402

D = slb.RAW / "depmap"
# contexts without depmap_id in contexts.parquet -> DepMap StrippedCellLineName prefix used as a proxy
PROXIES = {"hTERT-RPE1": "RPE1SS"}   # RPE1-ss* single-cell-derived hTERT-RPE1 clones (Ben-David lab), averaged
NGENES = 4079


def read(name):
    m = pd.read_csv(D / name, index_col=0)
    m.columns = [c.split(" (")[0] for c in m.columns]
    return m.loc[:, ~m.columns.duplicated()]


def context_models() -> dict[str, list[str]]:
    c = slb.contexts()
    c = c[c.species == "human"]
    models = pd.read_csv(D / "Model_24Q4.csv", usecols=["ModelID", "StrippedCellLineName"])
    out = {}
    for ctx, dm in zip(c.context_id, c.depmap_id):
        name = ctx.split(":", 1)[1]
        if isinstance(dm, str):
            out[ctx] = [dm]
        elif name in PROXIES:
            out[ctx] = models.ModelID[models.StrippedCellLineName.str.startswith(PROXIES[name])].tolist()
        else:
            out[ctx] = []
    return out


def main(out):
    ex, cn, ge = read("OmicsExpressionProteinCodingGenesTPMLogp1_24Q4.csv"), read("OmicsCNGene_24Q4.csv"), \
        read("CRISPRGeneEffect_24Q4.csv")
    mu = read("OmicsSomaticMutationsMatrixDamaging_24Q4.csv")
    common = ex.columns.intersection(cn.columns).intersection(ge.columns)
    genes = sorted(ge[common].var().sort_values(ascending=False).index[:NGENES])
    z = lambda m: ((m[genes] - m[genes].mean()) / m[genes].std().replace(0, 1)).fillna(0)
    exz, cnz, gez = z(ex), z(cn), z(ge)
    mug = mu.reindex(columns=genes).fillna(0).clip(0, 1)
    feats, info = {}, []
    for ctx, ids in context_models().items():
        x = np.zeros((NGENES, 6), dtype=np.float32)
        def avg(m, ids=ids):
            i = [k for k in ids if k in m.index]
            return (m.loc[i].mean().values, len(i)) if i else (None, 0)
        for ch, m in [(0, exz), (1, mug), (2, cnz), (3, gez), (4, ex[genes] / 10.0), (5, ge[genes].fillna(0))]:
            v, n = avg(m)
            if v is not None:
                x[:, ch] = np.nan_to_num(v)
        feats[ctx] = x
        info.append((ctx, ",".join(ids), avg(ex)[1], avg(ge)[1]))
    np.savez_compressed(out, genes=np.array(genes), **{k: v for k, v in feats.items()})
    info = pd.DataFrame(info, columns=["context_id", "depmap_ids", "n_expr", "n_effect"])
    info.to_csv(str(out).replace(".npz", "_contexts.tsv"), sep="\t", index=False)
    print(info.to_string(), file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
