"""Shared data layer for the cell-specific multi-graph SL models of the Fan/Kunjie lab
(MVGCN-iSL, MSGT-SL, MLEC-iSL): SLB human rows -> gene index, cell-independent gene graphs, per-context
omics node features, fit/valid/predict edge sets. Labels are read from train.parquet only.

Graphs (cell-independent, as in the original repos):
  PPI-physical  BIOGRID-9606.csv shipped with MVGCN-iSL, 'physical' rows only
  co-exp        coexpression_exp_0.5.csv shipped with MVGCN-iSL (CCLE expression correlation)
  co-ess        coexpression_ess_0.2.csv shipped with MVGCN-iSL (DepMap co-essentiality)
  SL            SLB fit-split positives, pooled over contexts (the original used the training SL pairs of the cell line)
  The original's 'PPI-genetic' graph (BioGRID genetic interactions) is DROPPED: it is a GI/SL-edge source (leaky).
Node features ('raw_omics'): per context, [exp, mut, cnv, ess] of that cell line from DepMap 24Q4, z-scored over
genes (sklearn.preprocessing.scale, as the original). Contexts without a DepMap line use pan-line means.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(os.environ.get("SLB_BENCH", ROOT / "data/bench/slb1.2"))
if not BENCH.is_absolute():
    BENCH = ROOT / BENCH
WORK = Path(os.environ.get("SLB_WORK", ROOT / "external/models/_slb_work" / BENCH.name))
SPLIT = os.environ.get("SLB_SPLIT", "dev")
MV = ROOT / "external/models/MVGCNiSL/data"


def ensure_prep():
    import subprocess
    if not (WORK / "human_train_rows.parquet").exists():
        subprocess.run(["uv", "run", "python", str(ROOT / "scripts/models/_common/slb_pairs.py")], check=True, cwd=ROOT)
    if not (WORK / "depmap_context_feats.parquet").exists():
        py = ROOT / "external/models/SL_benchmark/.venv-prep/bin/python"
        subprocess.run([str(py), str(ROOT / "scripts/models/_common/depmap_feats.py")], check=True, cwd=ROOT)


def predict_rows():
    f = BENCH / f"{SPLIT}.parquet"
    if not f.exists():
        f = BENCH / f"{SPLIT}_inputs.parquet"
    d = pd.read_parquet(f, columns=["example_id", "species", "context_id", "gene_a", "gene_b"])
    return d[d.species == "human"].reset_index(drop=True)


def load_graphs(graph_types):
    out = {}
    if "PPI-physical" in graph_types:
        b = pd.read_csv(MV / "BIOGRID-9606.csv", index_col=0)
        b = b[b["Experimental System Type"] == "physical"]
        out["PPI-physical"] = b[["Official Symbol Interactor A", "Official Symbol Interactor B"]].set_axis(["gene1", "gene2"], axis=1)
    if "co-exp" in graph_types:
        out["co-exp"] = pd.read_csv(MV / "coexpression_exp_0.5.csv")[["gene1", "gene2"]]
    if "co-ess" in graph_types:
        out["co-ess"] = pd.read_csv(MV / "coexpression_ess_0.2.csv")[["gene1", "gene2"]]
    return out


def build(graph_types=("SL", "PPI-physical", "co-exp", "co-ess")):
    ensure_prep()
    tr = pd.read_parquet(WORK / "human_train_rows.parquet")
    tr = tr[tr.split.isin(["fit", "valid"])]
    pr = predict_rows()
    gr = load_graphs(graph_types)
    genes = set(tr.gene_a) | set(tr.gene_b) | set(pr.gene_a) | set(pr.gene_b)
    for g in gr.values():
        genes |= set(g.gene1) | set(g.gene2)
    genes = sorted(str(x) for x in genes)
    gm = {g: i for i, g in enumerate(genes)}
    edges = {}
    for k, g in gr.items():
        e = np.array([g.gene1.astype(str).map(gm).values, g.gene2.astype(str).map(gm).values])
        e = np.concatenate([e, e[::-1]], axis=1)
        e = np.unique(e[:, e[0] != e[1]], axis=1)
        edges[k] = e
    if "SL" in graph_types:
        p = tr[(tr.split == "fit") & (tr.label == 1)][["gene_a", "gene_b"]].drop_duplicates()
        e = np.array([p.gene_a.map(gm).values, p.gene_b.map(gm).values])
        edges["SL"] = np.unique(np.concatenate([e, e[::-1]], axis=1), axis=1)
    order = [k for k in graph_types if k in edges]
    # per-context node features
    cf = pd.read_parquet(WORK / "depmap_context_feats.parquet")
    ctxs = sorted(set(tr.context_id) | set(pr.context_id))
    from sklearn.preprocessing import scale
    X = {}
    for c in ctxs:
        d = cf[cf.context_id == c].set_index("gene")[["exp", "mut", "cnv", "ess"]]
        x = d.reindex(genes).fillna(0.0).values.astype(np.float64)
        X[c] = scale(x).astype(np.float32)
    for d in (tr, pr):
        d["i"], d["j"] = d.gene_a.map(gm), d.gene_b.map(gm)
    print(f"genes {len(genes)}; graphs " + ", ".join(f"{k}:{edges[k].shape[1] // 2}" for k in order) +
          f"; contexts {len(ctxs)}; train rows fit {int((tr.split == 'fit').sum())} valid {int((tr.split == 'valid').sum())}; predict rows {len(pr)}")
    return dict(genes=genes, gm=gm, edges=[edges[k] for k in order], graph_names=order, X=X, train=tr, pred=pr)


def outdir():
    """results/models for SLB-1.2 (historical default), results/models/<bench name>/ for any other version."""
    return ROOT / "results/models" if BENCH.name == "slb1.2" else ROOT / "results/models" / BENCH.name


def write_preds(pr, scores, name):
    out = outdir() / f"{name}_{SPLIT}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"example_id": pr.example_id.values, "score": np.asarray(scores, dtype=np.float64)}).to_parquet(out, index=False)
    print("wrote", out, len(pr))
