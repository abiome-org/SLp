"""MuSL `allspecies` variant: the GNN branch alone (the repo's own `--use_gnn` ablation: GraphSAGE over a PPI
graph, nodes initialised with ESM2 embeddings, projector to 1024-d, pair MLP head) trained per species on SLB
train rows, using the shared bundle data/interim/bundle/<species>/{esm2,ppi}.parquet.
The CNN + statistical branches need a bulk expression compendium (TCGA; human only), so they are not part of it.

Graph: bundle PPI edges with >=1 BioGRID physical row or any STRING non-experimental channel >= 400 (the bundle
has no genetic-interaction, STRING experimental or textmining evidence). Nodes: genes with an ESM2 embedding.
Species: SLB train species with bundle esm2 + ppi. Training rows capped at MUSL_MAX_TRAIN (label-stratified
subsample) per species; early stopping on a gene-held-out 10% of train (val AUPR, as the original).
usage: external/models/musl/.venv/bin/python scripts/models/musl/train_bundle.py <split> <out.csv>
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader
from torch_geometric.utils import coalesce, to_torch_csr_tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import slb  # noqa: E402

REPO = slb.ROOT / "external/models/musl"
sys.path.insert(0, str(REPO))
from src import utils as U  # noqa: E402
from src.module import MuSL, EdgeDataset, train_step  # noqa: E402

B = slb.ROOT / "data/interim/bundle"
W = REPO / "_slb"
W.mkdir(exist_ok=True)
torch.set_num_threads(int(os.environ.get("SLB_THREADS", 8)))
MAX_TRAIN = int(os.environ.get("MUSL_MAX_TRAIN", 300_000))
# batch 1024 / patience 5 / <= 30 epochs (original: 128 / 10 / 100) so that it runs on CPU in reasonable time:
# every batch runs the GNN over the whole species graph.
CFG = dict(hidden_dim=128, unified_dim=256, target_node_dim=1024, lr=1e-4, wd=1e-3,
           batch_size=int(os.environ.get("MUSL_BUNDLE_BATCH", 1024)), seed=432,
           patience=int(os.environ.get("MUSL_PATIENCE", 5)), epochs=int(os.environ.get("MUSL_EPOCHS", 30)), lr_factor=0.5, lr_patience=3, min_lr=1e-6)


def run_species(sp, tr, ev, dev):
    emb = pd.read_parquet(B / sp / "esm2.parquet")
    genes = emb.gene.astype(str).tolist()
    gid = {g: i for i, g in enumerate(genes)}
    X = torch.tensor(emb.drop(columns=["gene"]).to_numpy(np.float32))
    ppi = pd.read_parquet(B / sp / "ppi.parquet")
    m = ppi.biogrid_phys > 0
    for c in [c for c in ppi.columns if c.startswith("string_")]:
        m |= ppi[c] >= 400
    a, b = ppi.loc[m, "gene_a"].map(gid), ppi.loc[m, "gene_b"].map(gid)
    ok = a.notna() & b.notna()
    p = np.stack([a[ok].astype(int), b[ok].astype(int)], 1)
    ei = torch.tensor(np.concatenate([p, p[:, ::-1]]).T.copy(), dtype=torch.long)
    ei = to_torch_csr_tensor(coalesce(ei).flip(0), size=(len(genes), len(genes)))  # sparse mean aggregation
    tr = tr.assign(i=tr.gene_a.map(gid), j=tr.gene_b.map(gid)).dropna(subset=["i", "j"])
    if len(tr) > MAX_TRAIN:
        tr = tr.groupby("label").sample(frac=MAX_TRAIN / len(tr), random_state=2025)
    rng = np.random.default_rng(2025)
    tg = np.array(sorted(set(tr.gene_a) | set(tr.gene_b)))
    vg = set(rng.choice(tg, size=max(1, len(tg) // 10), replace=False))
    isv = (tr.gene_a.isin(vg) | tr.gene_b.isin(vg)).values
    print(f"{sp}: {len(genes):,} ESM2 genes, {len(p):,} PPI edges; train rows {len(tr):,} (val {isv.sum():,}, "
          f"SL {int(tr.label.sum()):,})", file=sys.stderr, flush=True)
    dummy = torch.zeros(len(genes), 2)  # CNN/stat branches disabled; EdgeDataset still renders (empty) images
    base = EdgeDataset(torch.zeros(2, 1, dtype=torch.long), torch.zeros(1, dtype=torch.long), dummy,
                       state=torch.zeros(1, 35), bins=32, cal_state=True)
    import copy

    def mk(pr, y):
        d = copy.copy(base)
        d.edges = torch.tensor(np.asarray(pr, dtype=np.int64).T.copy())
        d.labels = torch.tensor(np.asarray(y, dtype=np.int64))
        d.state = torch.zeros(len(y), 35)
        return d

    P = tr[["i", "j"]].astype(int).values
    y = tr.label.astype(int).values
    kw = dict(batch_size=CFG["batch_size"], num_workers=2)
    lfit = DataLoader(mk(P[~isv], y[~isv]), shuffle=True, **kw)
    dval = mk(P[isv], y[isv])
    lval = DataLoader(dval, shuffle=False, **kw)
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    model = MuSL(num_nodes=len(genes), node_embedding_dim=X.shape[1], hidden_dim=CFG["hidden_dim"],
                 target_node_dim=CFG["target_node_dim"], node_embeddings=X.to(dev), unified_dim=CFG["unified_dim"],
                 enable_cross_attention=False, enable_adaptive_fusion=False, enable_contrastive_learning=False,
                 use_cnn=False, use_gnn=True, use_stat=False).to(dev)
    opt = U.customAdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=CFG["lr_factor"],
                                                       patience=CFG["lr_patience"], min_lr=CFG["min_lr"])
    crit = nn.CrossEntropyLoss()
    ckpt = W / f"gnn_{slb.BENCH.name}_{sp}_b{CFG['batch_size']}.pth"

    def predict(loader, head="gnn_pred"):
        model.eval()
        out = []
        with torch.no_grad():
            for bt in loader:
                r = train_step(model, bt, ei, crit, dev)
                out.append(F.softmax(r["predictions"][head], 1)[:, 1].cpu())
        return torch.cat(out).numpy()

    if not (ckpt.exists() and not os.environ.get("MUSL_RETRAIN")):
        best, bad = -1.0, 0
        for ep in range(CFG["epochs"]):
            t0 = time.time()
            model.train()
            for bt in lfit:
                opt.zero_grad()
                r = train_step(model, bt, ei, crit, dev)
                r["loss_gnn"].backward()  # original single-module training: loss of the enabled branch only
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
            pv = predict(lval)
            pr_, rc_, _ = precision_recall_curve(dval.labels.numpy(), pv)
            aupr = auc(rc_, pr_)
            sched.step(aupr)
            print(f"{sp} epoch {ep + 1}: val AUPR {aupr:.4f} AUROC {roc_auc_score(dval.labels.numpy(), pv):.4f} "
                  f"({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
            if aupr > best:
                best, bad = aupr, 0
                torch.save(model.state_dict(), ckpt)
            else:
                bad += 1
                if bad >= CFG["patience"]:
                    break
    model.load_state_dict(torch.load(ckpt))
    ev = ev.assign(i=ev.gene_a.map(gid), j=ev.gene_b.map(gid))
    pe = ev.dropna(subset=["i", "j"])[["i", "j"]].astype(int).drop_duplicates()
    if len(pe):
        f = predict(DataLoader(mk(pe.values, np.zeros(len(pe))), shuffle=False, **kw))
        r = predict(DataLoader(mk(pe.values[:, ::-1], np.zeros(len(pe))), shuffle=False, **kw))
        pe["score"] = 0.5 * (f + r)
        ev = ev.merge(pe, on=["i", "j"], how="left")
    else:
        ev["score"] = np.nan
    print(f"{sp}: scored {ev.score.notna().sum():,}/{len(ev):,}", file=sys.stderr)
    return ev[["example_id", "score"]]


def main(split, out_csv):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_all, ev_all = slb.load("train"), slb.load(split)
    outs = []
    only = os.environ.get("MUSL_SPECIES", "").split()  # optional restriction (debugging / staged runs)
    for sp in sorted(tr_all.species.unique()):
        if only and sp not in only:
            continue
        if not ((B / sp / "esm2.parquet").exists() and (B / sp / "ppi.parquet").exists()):
            print(f"{sp}: bundle esm2/ppi missing -> skipped", file=sys.stderr)
            continue
        outs.append(run_species(sp, tr_all[(tr_all.species == sp) & tr_all.label.notna()],
                                ev_all[ev_all.species == sp], dev))
    pd.concat(outs).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
