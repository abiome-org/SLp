"""Per-context single-gene omics features for cell-specific graph SL models (MVGCN-iSL, MSGT-SL, MLEC-iSL, ...).

All inputs are single-gene data permitted by the SLB leakage rules (DepMap 24Q4):
  exp  OmicsExpressionProteinCodingGenesTPMLogp1 (default model profile per line)
  ess  CRISPRGeneEffect (Chronos)
  cnv  OmicsCNGene (log2 relative CN)
  mut  OmicsSomaticMutationsMatrixDamaging (0/1/2 damaging alleles)
Output: SLB_WORK/depmap_context_feats.parquet  (context_id, gene, exp, ess, cnv, mut)
        SLB_WORK/depmap_pan_feats.parquet      (gene, exp_mean, exp_sd, ess_mean, ess_sd, cnv_mean, mut_freq)
        SLB_WORK/depmap_matrices.npz           (genes x lines matrices of exp and ess, for co-expression / co-essentiality graphs)
Contexts without a DepMap line (hTERT-RPE1, C092) get the pan-line means (flagged has_line=0).
"""
import os, re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(os.environ.get("SLB_BENCH", ROOT / "data/slb"))
if not BENCH.is_absolute():
    BENCH = ROOT / BENCH
OUT = Path(os.environ.get("SLB_WORK", ROOT / "external/models/_slb_work" / BENCH.name))
DM = ROOT / "data/raw/depmap"


def _sym(cols):
    return [re.sub(r"\s*\(.*\)$", "", c) for c in cols]


def load_matrix(fn, index_col=0):
    df = pd.read_csv(fn, index_col=index_col, low_memory=False)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ess = pd.read_csv(DM / "CRISPRGeneEffect_24Q4.csv", index_col=0)
    ess.columns = _sym(ess.columns)
    exp = pd.read_csv(DM / "OmicsExpressionProteinCodingGenesTPMLogp1_24Q4.csv", index_col=0)
    # 24Q4 expression file is indexed by model id already in some releases, by profile in others
    if "ModelID" in exp.columns:
        exp = exp.drop(columns=[c for c in exp.columns if not re.search(r"\(\d+\)$", c) and c != "ModelID"])
        exp = exp[exp.get("IsDefaultEntryForModel", pd.Series("Yes", index=exp.index)) == "Yes"] if "IsDefaultEntryForModel" in exp.columns else exp
        exp = exp.set_index("ModelID")
    exp.columns = _sym(exp.columns)
    exp = exp.select_dtypes("number")
    cnv = pd.read_csv(DM / "OmicsCNGene_24Q4.csv", index_col=0)
    cnv.columns = _sym(cnv.columns)
    mut = pd.read_csv(DM / "OmicsSomaticMutationsMatrixDamaging_24Q4.csv", index_col=0)
    if "ModelID" in mut.columns:
        mut = mut.set_index("ModelID")
    mut.columns = _sym(mut.columns)
    mut = mut.select_dtypes("number")
    for name, m in [("ess", ess), ("exp", exp), ("cnv", cnv), ("mut", mut)]:
        print(name, m.shape, m.index[:2].tolist())
    for m in (ess, exp, cnv, mut):
        m.drop(columns=m.columns[m.columns.duplicated()], inplace=True)
    genes = sorted(set(ess.columns) | set(exp.columns))
    pan = pd.DataFrame(index=genes)
    pan["exp_mean"], pan["exp_sd"] = exp.mean(), exp.std()
    pan["ess_mean"], pan["ess_sd"] = ess.mean(), ess.std()
    pan["cnv_mean"] = cnv.mean()
    pan["mut_freq"] = (mut > 0).mean()
    pan.index.name = "gene"
    pan.reset_index().to_parquet(OUT / "depmap_pan_feats.parquet", index=False)
    ctx = pd.read_parquet(BENCH / "contexts.parquet")
    ctx = ctx[ctx.species == "human"]
    rows = []
    for _, c in ctx.iterrows():
        d = pd.DataFrame(index=genes)
        dm = c.depmap_id if isinstance(c.depmap_id, str) else None
        for name, m, panc in [("exp", exp, "exp_mean"), ("ess", ess, "ess_mean"), ("cnv", cnv, "cnv_mean"), ("mut", mut, None)]:
            if dm is not None and dm in m.index:
                d[name] = m.loc[dm].reindex(genes).values
                d[name + "_has"] = 1
            else:
                d[name] = pan[panc].values if panc else 0.0
                d[name + "_has"] = 0
        d["context_id"] = c.context_id
        d.index.name = "gene"
        rows.append(d.reset_index())
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(OUT / "depmap_context_feats.parquet", index=False)
    print(out.groupby("context_id")[["exp_has", "ess_has", "cnv_has", "mut_has"]].first().to_string())
    common = [g for g in genes if g in ess.columns and g in exp.columns]
    lines = sorted(set(ess.index) & set(exp.index))
    np.savez_compressed(OUT / "depmap_matrices.npz", genes=np.array(common), lines=np.array(lines),
                        ess=ess.loc[lines, common].values.astype(np.float32), exp=exp.loc[lines, common].values.astype(np.float32))


if __name__ == "__main__":
    main()
