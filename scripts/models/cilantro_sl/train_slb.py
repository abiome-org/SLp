"""Cilantro-SL stages 1-2 on SLB: viability (FiLM) pretraining -> pair classifier -> Mondrian conformal calibration.

usage: python train_slb.py <deltas.parquet> <split> <workdir>

Uses the ORIGINAL network modules (nn_helpers/net_layers.FiLMNet, SLNet) and the original hyper-parameters
(notebooks 5_pretraining / 6_classifier + training_framework defaults); the drivers are re-written because
via_film / pair_classifier hard-code /work/magroup paths and pandas-1.3 idioms.
  1. FiLMNet(protein_size=128): input 512-d Geneformer delta embedding, FiLM-conditioned on the 128-d Gene2vec
     embedding (Cilantro's gene2vec_embs.pt); target DepMap CRISPR gene effect; MSE, Adam lr 1e-3, batch 512,
     100 epochs, all rows with a known gene effect (original: test_size 0.0).  Viability embedding = l3 output
     (32-d, pre-ReLU) = 'extract_nn_second_last'.
  2. Pair features = [via(c, gene_a) | via(c, gene_b)] (64-d) in the SLB context c of the example (original:
     every DepMap line where both genes have deltas, with cell-agnostic SynLethDB labels).
     SLNet, CrossEntropy with normalised inverse-frequency class weights, Adam lr 1e-4, batch 128, 100 epochs,
     5-fold CV over SLB-train pairs (original: framework.run_cv(folds=5)); fold nets are ensembled.
  3. Mondrian conformal (single class, as in the notebook's mondrian_class_dict={}): each fold net is
     calibrated on its held-out train fold; p_SL = (#{calib scores >= 1 - P(SL)} + 1) / (n + 1).
Output: <workdir>/<split>_scores.parquet (example_id, score = mean fold P(SL) averaged over both gene orders,
p_sl = mean conformal p-value).  SLB labels used: train.parquet only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import slb  # noqa: E402

REPO = slb.ROOT / "external/models/cilantro_sl"
sys.path.insert(0, str(REPO / "nn_helpers"))
from net_layers import FiLMNet, SLNet  # noqa: E402

torch.set_num_threads(8)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)


def batches(n, bs, shuffle, rng):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for s in range(0, n, bs):
        yield idx[s:s + bs]


def viability_embeddings(df: pd.DataFrame) -> np.ndarray:
    g2v = torch.load(slb.RAW / "cilantro_sl/gene2vec_embs.pt", weights_only=False)
    keep = df.ensembl.isin(g2v.keys()).to_numpy()
    print(f"deltas {len(df)}, with gene2vec {keep.sum()}", file=sys.stderr)
    X = torch.tensor(df.filter(regex=r"^d\d+$").to_numpy(np.float32))
    G = torch.stack([g2v[e].float() if e in g2v else torch.zeros(128) for e in df.ensembl])
    y = torch.tensor(df.viability.to_numpy(np.float32))
    tr = np.where(keep & ~np.isnan(df.viability.to_numpy()))[0]
    net = FiLMNet(protein_size=128).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    rng = np.random.default_rng(0)
    for ep in range(100):
        tot = 0.0
        for b in batches(len(tr), 512, True, rng):
            i = tr[b]
            out, _ = net(X[i].to(DEV), G[i].to(DEV))
            loss = mse(out, y[i].to(DEV).unsqueeze(1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(i)
        if ep % 20 == 0 or ep == 99:
            print(f"viability epoch {ep}: mse {tot / len(tr):.4f} (n={len(tr)})", file=sys.stderr)
    net.eval()
    with torch.no_grad():
        pred, emb = [], []
        for b in batches(len(df), 4096, False, rng):
            o, e = net(X[b].to(DEV), G[b].to(DEV))
            pred.append(o.cpu())
            emb.append(e.cpu())
    pred = torch.cat(pred).squeeze(1).numpy()
    ok = ~np.isnan(df.viability.to_numpy())
    print(f"viability fit: pearson r (train rows) = {np.corrcoef(pred[ok & keep], df.viability.to_numpy()[ok & keep])[0, 1]:.3f}",
          file=sys.stderr)
    emb = torch.cat(emb).numpy()
    emb[~keep] = np.nan   # original drops genes without a gene2vec vector
    return emb


def pair_matrix(d: pd.DataFrame, lut: dict) -> tuple[np.ndarray, np.ndarray]:
    ia = [lut.get((c, a), -1) for c, a in zip(d.context_id, d.gene_a)]
    ib = [lut.get((c, b), -1) for c, b in zip(d.context_id, d.gene_b)]
    return np.array(ia), np.array(ib)


def main(deltas, split, wd):
    wd = Path(wd)
    wd.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(deltas)
    V = viability_embeddings(df)
    ok = ~np.isnan(V).any(1)
    lut = {(c, g): i for i, (c, g, o) in enumerate(zip(df.context_id, df.gene, ok)) if o}

    tr = slb.load("train")
    tr = tr[(tr.species == "human") & tr.label.notna()]
    ia, ib = pair_matrix(tr, lut)
    m = (ia >= 0) & (ib >= 0)
    tr, ia, ib = tr[m], ia[m], ib[m]
    Xtr = np.concatenate([V[ia], V[ib]], 1).astype(np.float32)
    ytr = tr.label.astype(int).to_numpy()
    print(f"train pairs with features: {len(tr)} (pos {ytr.sum()})", file=sys.stderr)

    te = slb.load(split)
    te = te[te.species == "human"]
    ja, jb = pair_matrix(te, lut)
    mt = (ja >= 0) & (jb >= 0)
    Xab = np.concatenate([V[ja[mt]], V[jb[mt]]], 1).astype(np.float32)
    Xba = np.concatenate([V[jb[mt]], V[ja[mt]]], 1).astype(np.float32)
    print(f"{split} human rows with features: {mt.sum()}/{len(te)}", file=sys.stderr)

    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(len(Xtr)), 5)
    P, PV = [], []
    for k in range(5):
        cal = folds[k]
        fit = np.concatenate([folds[j] for j in range(5) if j != k])
        counts = np.bincount(ytr[fit], minlength=2).astype(np.float32)
        w = 1.0 / np.maximum(counts, 1)
        w = torch.tensor(w / w.sum()).to(DEV)
        net = SLNet().to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=1e-4)
        ce = nn.CrossEntropyLoss(weight=w)
        Xf, yf = torch.tensor(Xtr[fit]), torch.tensor(ytr[fit])
        for ep in range(100):
            net.train()
            for b in batches(len(fit), 128, True, rng):
                out = net(Xf[b].to(DEV))
                loss = ce(out, yf[b].to(DEV))
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            prob = lambda X: torch.softmax(net(torch.tensor(X).to(DEV)), 1)[:, 1].cpu().numpy()
            pc = prob(Xtr[cal])
            p = (prob(Xab) + prob(Xba)) / 2
        s_cal = np.sort(np.where(ytr[cal] == 1, 1 - pc, pc))      # 1 - P(true class)
        n = len(s_cal)
        pv = (n - np.searchsorted(s_cal, 1 - p, side="left") + 1) / (n + 1)
        from sklearn.metrics import roc_auc_score
        if len(set(ytr[cal])) == 2:
            print(f"fold {k}: calib AUROC {roc_auc_score(ytr[cal], pc):.3f} (n={n})", file=sys.stderr)
        P.append(p)
        PV.append(pv)
    out = pd.DataFrame({"example_id": te.example_id.to_numpy()[mt], "score": np.mean(P, 0), "p_sl": np.mean(PV, 0)})
    out.to_parquet(wd / f"{split}_scores.parquet", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
