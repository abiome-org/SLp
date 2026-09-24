"""MuSL (Fang et al., IEEE JBHI 2026) retrained on SLB train labels (human), scored on an SLB split.

Uses the repo's own model / dataset / feature code (external/models/musl/src/module.py, utils.py) with the
README "full MuSL" configuration (CNN + GNN + Stat branches, ESM2 node init, cross-attention, adaptive fusion,
contrastive loss, custom AdamW, lr 1e-4, wd 1e-3, batch 128, seed 432, patience 10, ReduceLROnPlateau).
Changes vs. the original (see notes/models/musl.md):
  * labels: SLB human train rows (one row per context x pair; context-agnostic model) instead of SynLethDB
    positives + random negatives; single model instead of 5-fold CV; early stopping (test AUPR, as in the
    original) on a gene-held-out 10% of SLB train, never on the evaluation split.
  * graph nodes: every gene with an ESM2 embedding (authors' protein_embeddings.pt, 19.8k genes) instead of the
    7,684 SynLethDB SL genes (that node list is itself derived from SL labels).
  * PPI: BioGRID 5.0.261 human *physical* interactions (the bundled merged_ppi_gene_names.csv has undocumented
    provenance: 710 of its edges are BioGRID genetic-only interactions and 260k are in neither BioGRID set).
  * expression: authors' tcga_all.h5ad (TCGA bulk RNA-seq, Zenodo 17098066); genes absent -> zeros (as the
    original get_cell_expression does). Stat features computed in chunks (torch.quantile size limit) with
    16 MI worker processes instead of 48.
usage (from repo root): external/models/musl/.venv/bin/python scripts/models/musl/train_slb.py <split> <out.csv>
"""
from __future__ import annotations

import functools
import os
import sys
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
from src.module import MuSL, EdgeDataset, train_step, min_max_normalize  # noqa: E402

W = REPO / "_slb"
W.mkdir(exist_ok=True)
RAW = slb.RAW / "musl"
THREADS = int(os.environ.get("SLB_THREADS", 8))
torch.set_num_threads(THREADS)
U.get_mutual_info = functools.partial(U.get_mutual_info, max_workers=THREADS)  # used inside calculate_features
_mi_row = U.compute_mi_row


def _mi_row_1thread(args):  # MI worker processes: one BLAS/OpenMP thread each (machine-wide thread cap)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        return _mi_row(args)


U.compute_mi_row = _mi_row_1thread
res = slb.symbol_resolver()
MAX_STEPS = int(os.environ.get("MUSL_MAX_STEPS", 10**9))  # debugging only

CFG = dict(hidden_dim=128, unified_dim=256, target_node_dim=1024, lr=1e-4, wd=1e-3, batch_size=128, seed=432,
           patience=int(os.environ.get("MUSL_PATIENCE", 10)), epochs=int(os.environ.get("MUSL_EPOCHS", 100)), bins=32, contrastive_weight=0.1,
           lr_factor=0.5, lr_patience=3, min_lr=1e-6, num_workers=4)


def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_inputs():
    """genes (ESM2 universe, current HGNC), ESM2 matrix, expression matrix (genes x samples), PPI edge_index."""
    cache = W / "inputs.pt"
    if cache.exists():
        return torch.load(cache)
    emb = torch.load(RAW / "protein_embeddings.pt", map_location="cpu")
    e2 = {}
    for k, v in emb.items():
        s = res(k) or k
        if s not in e2 or s == k:
            e2[s] = v.float()
    genes = sorted(e2)
    gid = {g: i for i, g in enumerate(genes)}
    X = torch.stack([e2[g] for g in genes])
    import scanpy as sc
    ad = sc.read_h5ad(RAW / "tcga_all.h5ad")
    names = ad.var["gene_name"].astype(str).values if "gene_name" in ad.var else ad.var_names.astype(str).values
    M = ad.X.toarray() if hasattr(ad.X, "toarray") else np.asarray(ad.X)
    E = torch.zeros(len(genes), M.shape[0])
    seen = set()
    hit = 0
    for j, n in enumerate(names):
        s = res(n) or n
        if s in gid and s not in seen:
            seen.add(s)
            E[gid[s]] = torch.from_numpy(M[:, j].astype(np.float32))
            hit += 1
    print(f"ESM2 genes {len(genes):,}; TCGA samples {M.shape[0]:,}; genes with expression {hit:,}", file=sys.stderr)
    b = pd.read_csv(slb.RAW / "biogrid/BIOGRID-ORGANISM-Homo_sapiens-5.0.261.tab3.txt", sep="\t", dtype=str,
                    usecols=["Entrez Gene Interactor A", "Entrez Gene Interactor B", "Experimental System Type",
                             "Organism ID Interactor A", "Organism ID Interactor B"])
    b = b[(b["Experimental System Type"] == "physical") & (b["Organism ID Interactor A"] == "9606")
          & (b["Organism ID Interactor B"] == "9606")]
    a = b["Entrez Gene Interactor A"].map(res).map(gid)
    c = b["Entrez Gene Interactor B"].map(res).map(gid)
    ok = a.notna() & c.notna() & (a != c)
    pairs = np.unique(np.sort(np.stack([a[ok].astype(int), c[ok].astype(int)], 1), axis=1), axis=0)
    ei = torch.tensor(np.concatenate([pairs, pairs[:, ::-1]]).T.copy(), dtype=torch.long)  # undirected
    print(f"BioGRID physical PPI: {len(pairs):,} undirected edges among ESM2 genes", file=sys.stderr)
    out = {"genes": genes, "X": X, "E": E, "edge_index": ei}
    torch.save(out, cache)
    return out


def _pair_cache(name, P, compute):
    """Row cache keyed by gene-index pair (i, j) (order matters), shared across splits / benchmark versions
    (the gene universe is fixed by protein_embeddings.pt): only pairs not seen before are computed."""
    f = W / f"{name}_cache.pt"
    C = torch.load(f) if f.exists() else {"P": np.zeros((0, 2), np.int64), "V": None}
    key = {tuple(p): k for k, p in enumerate(map(tuple, C["P"]))}
    miss = np.array([p for p in map(tuple, P) if p not in key], dtype=np.int64).reshape(-1, 2)
    if len(miss):
        miss = np.unique(miss, axis=0)
        V = compute(miss)
        C = {"P": np.concatenate([C["P"], miss]), "V": V if C["V"] is None else torch.cat([C["V"], V])}
        torch.save(C, f)
        key = {tuple(p): k for k, p in enumerate(map(tuple, C["P"]))}
    return C["V"][torch.tensor([key[tuple(p)] for p in map(tuple, P)], dtype=torch.long)]


def raw_stats(E, P: np.ndarray, tag: str) -> torch.Tensor:
    """35 MuSL statistical features (utils.calculate_features) for gene-index pairs P (n x 2); chunked because
    torch.quantile refuses > 16M elements."""
    def compute(Q):
        logE = torch.log1p(E)
        mu, sd = logE.mean(1, keepdim=True), logE.std(1, keepdim=True)
        sd = torch.where(sd < 1e-10, torch.ones_like(sd), sd)
        normE = (logE - mu) / sd
        step = max(1, 16_000_000 // E.shape[1])
        out = []
        for s in range(0, len(Q), step):
            gp = torch.tensor(Q[s:s + step].T, dtype=torch.long)
            out.append(torch.nan_to_num(U.calculate_features(E, logE, normE, gp), nan=0.0).float())
            print(f"stats {tag}: {min(s + step, len(Q)):,}/{len(Q):,}", file=sys.stderr, flush=True)
        return torch.cat(out)
    return _pair_cache("stat", P, compute)


def main(split, out_csv):
    set_seed(CFG["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inp = load_inputs()
    genes, X, E, ei = inp["genes"], inp["X"], inp["E"], inp["edge_index"]
    # GraphSAGE mean aggregation via a sparse adjacency (identical result for coalesced edges, checked; the
    # edge-list path needs E x 1024 message tensors = 7.9 GB on the 19.8k-gene BioGRID graph)
    ei = to_torch_csr_tensor(coalesce(ei).flip(0), size=(len(genes), len(genes)))
    gid = {g: i for i, g in enumerate(genes)}

    tr = slb.load("train")
    tr = tr[(tr.species == "human") & tr.label.notna()].copy()
    tr["i"], tr["j"] = tr.gene_a.map(gid), tr.gene_b.map(gid)
    n0 = len(tr)
    tr = tr.dropna(subset=["i", "j"])
    print(f"train rows with both genes embedded {len(tr):,}/{n0:,} ({int(tr.label.sum())} SL)", file=sys.stderr)
    rng = np.random.default_rng(2025)
    tg = np.array(sorted(set(tr.gene_a) | set(tr.gene_b)))
    vg = set(rng.choice(tg, size=len(tg) // 10, replace=False))
    tr["val"] = tr.gene_a.isin(vg) | tr.gene_b.isin(vg)

    ev = slb.load(split)
    ev = ev[ev.species == "human"].copy()
    ev["i"], ev["j"] = ev.gene_a.map(gid), ev.gene_b.map(gid)
    evok = ev.dropna(subset=["i", "j"])

    # stat features on unique pairs, normalised with parameters fitted on the fit (non-val) train pairs
    up = lambda d: np.unique(d[["i", "j"]].astype(int).values, axis=0)  # noqa: E731
    Ptr, Pev = up(tr), up(evok)
    Str, Sev = raw_stats(E, Ptr, "train"), raw_stats(E, Pev, split)
    fitmask = pd.Series(map(tuple, Ptr)).isin(set(map(tuple, tr.loc[~tr.val, ["i", "j"]].astype(int).values))).values
    npar = U.get_normalize_params(Str[fitmask].numpy().copy())
    Ntr = U.normalize_with_params(Str.numpy().copy(), npar).float()
    Nev = U.normalize_with_params(Sev.numpy().copy(), npar).float()
    ktr = {tuple(p): k for k, p in enumerate(map(tuple, Ptr))}
    kev = {tuple(p): k for k, p in enumerate(map(tuple, Pev))}

    import copy
    import hashlib
    base = EdgeDataset(torch.zeros(2, 1, dtype=torch.long), torch.zeros(1, dtype=torch.long), E,
                       state=torch.zeros(1, 35), bins=CFG["bins"], cal_state=True)  # log/z-scored expression once

    def mk(edges, labels, state):
        d = copy.copy(base)
        d.edges, d.labels, d.state = edges, labels, state
        return d

    def images(P, tag):
        """EdgeDataset's 3x32x32 joint-expression histograms, rendered once with the same code and cached
        (counts <= n_samples = 10,090 are stored exactly as int16)."""
        def compute(Q):
            d = mk(torch.tensor(Q.T.copy(), dtype=torch.long), torch.zeros(len(Q), dtype=torch.long),
                   torch.zeros(len(Q), 35))
            im = torch.cat([b["image"].short() for b in DataLoader(d, batch_size=512, num_workers=THREADS)])
            print(f"rendered {len(im):,} {tag} images", file=sys.stderr, flush=True)
            return im
        return _pair_cache("img", P, compute)

    class Cached(torch.utils.data.Dataset):
        """Same items as EdgeDataset (image, state, edge, label) with pre-rendered images."""

        def __init__(self, edges, labels, state, img, rows):
            self.edges, self.labels, self.state, self.img, self.rows = edges, labels, state, img, rows

        def __len__(self):
            return self.edges.size(1)

        def __getitem__(self, k):
            return {"image": self.img[self.rows[k]].float(), "state": self.state[k], "edge": self.edges[:, k],
                    "label": self.labels[k]}

    Itr = images(Ptr, "train")
    if os.environ.get("MUSL_PREP_ONLY"):  # CPU pre-computation of all cached features, then stop
        Prev = Pev[:, ::-1].copy()
        images(Pev, split), raw_stats(E, Prev, split + "_rev"), images(Prev, split + "_rev")
        print("prep only: caches complete", file=sys.stderr)
        return

    def ds(d, S, key, img):
        pr = d[["i", "j"]].astype(int).values
        idx = torch.tensor([key[tuple(p)] for p in pr])
        return Cached(torch.tensor(pr.T, dtype=torch.long), torch.tensor(d.label.fillna(0).astype(int).values),
                      S[idx], img, idx)

    dfit, dval = ds(tr[~tr.val], Ntr, ktr, Itr), ds(tr[tr.val], Ntr, ktr, Itr)
    kw = dict(batch_size=CFG["batch_size"], num_workers=CFG["num_workers"])
    lfit = DataLoader(dfit, shuffle=True, **kw)
    lval = DataLoader(dval, shuffle=False, **kw)

    model = MuSL(num_nodes=len(genes), node_embedding_dim=X.shape[1], hidden_dim=CFG["hidden_dim"],
                 target_node_dim=CFG["target_node_dim"], node_embeddings=X.to(dev), unified_dim=CFG["unified_dim"],
                 enable_cross_attention=True, enable_adaptive_fusion=True, enable_contrastive_learning=True,
                 use_cnn=True, use_gnn=True, use_stat=True).to(dev)
    opt = U.customAdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=CFG["lr_factor"],
                                                       patience=CFG["lr_patience"], min_lr=CFG["min_lr"])
    crit = nn.CrossEntropyLoss()
    ckpt = W / f"best_model_{slb.BENCH.name}.pth"  # cached per benchmark dir; MUSL_RETRAIN=1 to refit
    best, bad = -1.0, 0
    if ckpt.exists() and not os.environ.get("MUSL_RETRAIN"):
        CFG["epochs"] = 0
        print(f"using cached checkpoint {ckpt}", file=sys.stderr)

    def predict(loader):
        model.eval()
        ps = []
        with torch.no_grad():
            for b in loader:
                r = train_step(model, b, ei, crit, dev, alpha=CFG["contrastive_weight"])
                ps.append(F.softmax(r["predictions"]["final_pred"], 1)[:, 1].cpu())
        return torch.cat(ps).numpy()

    import time
    for ep in range(CFG["epochs"]):
        t0 = time.time()
        model.train()
        tot = 0.0
        for step, b in enumerate(lfit):
            if step >= MAX_STEPS:
                break
            opt.zero_grad()
            r = train_step(model, b, ei, crit, dev, alpha=CFG["contrastive_weight"])
            r["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tot += r["total_loss"].item()
        pv = predict(lval)
        yv = dval.labels.numpy()
        pr, rc, _ = precision_recall_curve(yv, pv)
        aupr, auroc = auc(rc, pr), roc_auc_score(yv, pv)
        sched.step(aupr)
        print(f"epoch {ep + 1}: loss {tot / len(lfit):.4f} val AUPR {aupr:.4f} AUROC {auroc:.4f} "
              f"({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
        if aupr > best:
            best, bad = aupr, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= CFG["patience"]:
                print("early stop", file=sys.stderr)
                break

    model.load_state_dict(torch.load(ckpt))
    # score each unique eval pair in both gene orders (fusion is order-dependent) and average
    pe = pd.DataFrame(Pev, columns=["i", "j"])
    pe["label"] = 0
    Prev = Pev[:, ::-1].copy()
    ar = torch.arange(len(Pev))
    z = torch.zeros(len(Pev), dtype=torch.long)
    fwd = Cached(torch.tensor(Pev.T.copy(), dtype=torch.long), z, Nev, images(Pev, split), ar)
    Srev = raw_stats(E, Prev, split + "_rev")
    Nrev = U.normalize_with_params(Srev.numpy().copy(), npar).float()
    rev = Cached(torch.tensor(Prev.T.copy(), dtype=torch.long), z, Nrev, images(Prev, split + "_rev"), ar)
    # NB: MuSL min-max normalises probabilities per evaluation set; a monotone transform, so ranking is unchanged
    pe["score"] = 0.5 * (predict(DataLoader(fwd, shuffle=False, **kw)) + predict(DataLoader(rev, shuffle=False, **kw)))
    ev = ev.merge(pe[["i", "j", "score"]], on=["i", "j"], how="left")
    ev[["example_id", "score"]].to_csv(out_csv, index=False)
    print(f"best val AUPR {best:.4f}; scored {ev.score.notna().sum():,}/{len(ev):,} {split} human rows",
          file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
