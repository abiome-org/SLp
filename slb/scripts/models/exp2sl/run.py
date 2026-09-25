"""EXP2SL (Wan et al. 2020, Front Pharmacol 11:112; github FangpingWan/EXP2SL) retrained on SLB.

Model (ported from model.py to device-agnostic torch, CPU by default): per-gene 978-d L1000 shRNA consensus
signature -> MLP encoder (dnn_layers x hidden d, ReLU) -> linear head on the concatenated pair (symmetrised by
averaging both orders). Loss (Rank_Loss2): MSE to +1 / -1 for labelled SL / non-SL pairs + semi_weight x BPR terms
ranking unknown pairs above sampled negatives and positives above unknowns. Adam lr 1e-3, weight decay l2, grad clip
5, 1000 full-batch epochs, as in the original.
Hyper-parameters are not reported per cell line in the paper (grid: BPR weight {16..128}, l2 {1e-4..0.1}, layers
{0..4}, d {32..256}); fixed a priori here, NOT tuned on dev: semi_weight 32, l2 0.01, 1 layer, d 128.
Features: LINCS 2020 level-5 trt_sh.cgs (consensus gene signatures) at 96 h, landmark genes (the original used
consensus signatures of the same kind, 96 h; the original feature pickle is no longer downloadable).
Variants:
  exp2sl__cellline  one model per SLB context that is an L1000 cell line with shRNA CGS (e.g. A-375,
                    A-549, HT-29), trained on that context's SLB train rows (as in the paper: cell-line-specific)
  exp2sl__pancell   extension: per-gene signature averaged over all L1000 lines, one model on all human SLB train
                    rows, applied to every human context
Human only. SL labels: SLB train only (the repo's GEMINI labels from Shen/Zhao/Big-Papi screens are not used).
"""
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import slb  # noqa: E402

torch.set_num_threads(8)
L1000 = slb.RAW / "lincs2020"
CACHE = slb.ROOT / "external/models/_statsl_cache"
SW, L2, LAYERS, DIM, EPOCHS = 32.0, 0.01, 1, 128, 1000


def norm(s):
    return "".join(ch for ch in str(s).upper() if ch.isalnum())


def cgs_features():
    p = CACHE / "lincs2020_trt_sh_cgs_landmark.parquet"
    if p.exists():
        return pd.read_parquet(p)
    si = pd.read_csv(L1000 / "siginfo_beta.txt", sep="\t", usecols=["sig_id", "pert_type", "cell_iname", "cmap_name",
                                                                   "pert_itime"], low_memory=False)
    si = si[si.pert_type == "trt_sh.cgs"]
    si["t96"] = si.pert_itime == "96 h"
    si = si.sort_values("t96", ascending=False).drop_duplicates(["cell_iname", "cmap_name"])
    gi = pd.read_csv(L1000 / "geneinfo_beta.txt", sep="\t")
    lm = gi[gi.feature_space == "landmark"]
    with h5py.File(L1000 / "level5_beta_trt_sh_n238351x12328.gctx", "r") as f:
        cols = [x.decode() for x in f["0/META/COL/id"][:]]
        rows = [x.decode() for x in f["0/META/ROW/id"][:]]
        ci = {c: i for i, c in enumerate(cols)}
        ri = {r: i for i, r in enumerate(rows)}
        gidx = np.array(sorted(ri[str(g)] for g in lm.gene_id if str(g) in ri))
        sidx = np.array(sorted(ci[s] for s in si.sig_id if s in ci))
        M = f["0/DATA/0/matrix"]
        out = np.empty((len(sidx), len(gidx)), np.float32)
        for k in range(0, len(sidx), 2000):
            blk = M[sidx[k:k + 2000], :]
            out[k:k + 2000] = blk[:, gidx]
    sig = [cols[i] for i in sidx]
    df = pd.DataFrame(out, columns=[rows[i] for i in gidx])
    meta = si.set_index("sig_id").loc[sig, ["cell_iname", "cmap_name"]].reset_index(drop=True)
    res = slb.symbol_resolver()
    meta["gene"] = [res(g) or g for g in meta.cmap_name]
    df = pd.concat([meta[["cell_iname", "gene"]], df], axis=1)
    df.to_parquet(p)
    return df


class Net(nn.Module):
    def __init__(self, feats):
        super().__init__()
        self.X = torch.tensor(feats, dtype=torch.float32)
        self.enc = nn.Sequential(nn.Linear(feats.shape[1], DIM), nn.ReLU(),
                                 *[m for _ in range(LAYERS) for m in (nn.Linear(DIM, DIM), nn.ReLU())])
        self.head = nn.Linear(2 * DIM, 1)

    def score(self, H, i, j):
        return 0.5 * (self.head(torch.cat([H[i], H[j]], 1)) + self.head(torch.cat([H[j], H[i]], 1))).squeeze(1)


def fit_predict(feats, pos, neg, test, seed=0):
    """feats: genes x 978; pos/neg/test: (2, n) index arrays. Returns scores for test pairs."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = feats.shape[0]
    model = Net(feats)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=L2)
    known = set(map(tuple, np.hstack([pos, neg, test]).T)) | set(map(tuple, np.hstack([pos, neg, test])[::-1].T))
    bce = nn.BCELoss(reduction="sum")
    for ep in range(EPOCHS):
        npos = pos.shape[1]
        uk = rng.integers(0, n, size=(2, npos * 2))
        uk = uk[:, [(a != b) and ((a, b) not in known) for a, b in uk.T]][:, :npos]
        ns = neg[:, rng.integers(0, neg.shape[1], npos)]
        H = model.enc(model.X)
        op, on, ons, ou = (model.score(H, *pos), model.score(H, *neg), model.score(H, *ns), model.score(H, *uk))
        loss = ((op - 1) ** 2).sum() + ((on + 1) ** 2).sum()
        k = min(len(ou), len(ons), len(op))
        loss = loss + SW * (bce(torch.sigmoid(ou[:k] - ons[:k]), torch.ones(k)) +
                            bce(torch.sigmoid(op[:k] - ou[:k]), torch.ones(k)))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
    with torch.no_grad():
        H = model.enc(model.X)
        return model.score(H, *test).numpy()


def run_block(feat: pd.DataFrame, tr: pd.DataFrame, ev: pd.DataFrame):
    genes = list(feat.index)
    gi = {g: i for i, g in enumerate(genes)}
    tr = tr[tr.gene_a.isin(gi) & tr.gene_b.isin(gi)]
    ev_ok = ev.gene_a.isin(gi) & ev.gene_b.isin(gi)
    if tr.label.sum() < 5 or ev_ok.sum() == 0:
        return pd.Series(np.nan, index=ev.index), len(tr)
    # one label per pair (max over contexts / duplicates)
    t = tr.assign(i=tr.gene_a.map(gi), j=tr.gene_b.map(gi)).groupby(["i", "j"]).label.max().reset_index()
    pos = t[t.label == 1][["i", "j"]].to_numpy().T
    neg = t[t.label == 0][["i", "j"]].to_numpy().T
    e = ev[ev_ok]
    test = np.vstack([e.gene_a.map(gi).to_numpy(), e.gene_b.map(gi).to_numpy()])
    X = feat.to_numpy(np.float32)
    s = fit_predict(X, pos, neg, test)
    out = pd.Series(np.nan, index=ev.index)
    out[e.index] = s
    return out, len(t)


def main(split):
    F = cgs_features()
    lm = [c for c in F.columns if c not in ("cell_iname", "gene")]
    tr_all = slb.load("train")
    ev_all = slb.load(split)
    tr_h = tr_all[tr_all.species == "human"]
    ev_h = ev_all[ev_all.species == "human"]
    # cell-line variant
    lines = {norm(c): c for c in F.cell_iname.unique()}
    s_cl = pd.Series(np.nan, index=ev_all.index)
    for ctx in sorted(ev_h.context_id.unique()):
        cl = lines.get(norm(ctx.split(":", 1)[1]))
        if cl is None:
            continue
        feat = F[F.cell_iname == cl].drop_duplicates("gene").set_index("gene")[lm]
        s, ntr = run_block(feat, tr_h[tr_h.context_id == ctx], ev_h[ev_h.context_id == ctx])
        s_cl.loc[s.index] = s
        print(f"exp2sl cellline {ctx} -> {cl}: {len(feat)} genes, {ntr} train pairs, scored {int(s.notna().sum())}",
              file=sys.stderr)
    # pan-cell variant
    feat = F.groupby("gene")[lm].mean()
    s_pc, ntr = run_block(feat, tr_h, ev_h)
    print(f"exp2sl pancell: {len(feat)} genes, {ntr} train pairs, scored {int(s_pc.notna().sum())}", file=sys.stderr)
    for name, s in [("exp2sl__cellline", s_cl), ("exp2sl__pancell", s_pc)]:
        out = ev_all[["example_id"]].copy()
        out["score"] = s.reindex(ev_all.index)
        slb.write(out, name, split)
        slb.evaluate(name, split)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else slb.SPLIT)
