"""Struct2SL (Hu et al., CSBJ 2025) on SLB, human.
Per-gene features from the authors' figshare release (10.6084/m9.figshare.33439123, Apache-2.0; data/raw/struct2sl):
AlphaFold2-contact-map node2vec (128), SeqVec sequence embedding (1024), STRING-v12-physical node2vec (128) for
17,180 human genes; pair feature = [f(a), f(b)] z-scored with training statistics; the released MLP class
(external/models/Struct2SL/Struct2SL.py: 256-128-64, dropout 0.5, sigmoid output) and loss (BCEWithLogits on the
sigmoid output + 0.001 * L2 norm of all weights, Adam lr 0.004 wd 0.001, grad clip 1, ReduceLROnPlateau, early stop
15 on validation loss, batch 64) are used unchanged.
Variants (SLB_VARIANT):
  slbtrain (default): retrained on SLB human train `fit` pairs with measured negatives (the original only samples
                      random negatives when measured ones are fewer than positives: never the case here); early stop on
                      SLB `valid` (held-aside train families). -> results/models/struct2sl_<split>.parquet
  released:           the authors' bestmodel.pt (trained on SynLethDB 2.0: LEAKY). Its z-score statistics are not
                      released; they are re-estimated from the authors' training pairs (Human_SL_ff + Human_nonSL).
                      -> results/models/struct2sl__released_<split>.parquet
Pairs with a gene lacking features are left out (the evaluator fills them with the median). Score is order-averaged."""
import os, sys, pickle, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/models/_common"))
import cellgraph
import importlib.util
spec = importlib.util.spec_from_file_location("s2s", ROOT / "external/models/Struct2SL/Struct2SL.py"); s2s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s2s)
VAR = os.environ.get("SLB_VARIANT", "slbtrain")
SEED = int(os.environ.get("SLB_SEED", "42")); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(int(os.environ.get("SLB_THREADS", "8"))); torch.set_num_interop_threads(1)
D = ROOT / "data/raw/struct2sl"
if not (D / "struct/struct_features").exists():
    import zipfile; zipfile.ZipFile(D / "struct_features.zip").extractall(D / "struct")
P = pickle.load(open(D / "ppi_features.npz", "rb")); S = pickle.load(open(D / "sequence_features.npz", "rb"))
if VAR == "slbtrain":
    # leak-free PPI block: node2vec on BioGRID physical (ppi_node2vec.py) instead of the released STRING-physical node2vec
    _pp = cellgraph.WORK / "struct2sl_ppi_node2vec.pkl"
    if not _pp.exists():
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts/models/struct2sl/ppi_node2vec.py")], check=True)
    P = pickle.load(open(_pp, "rb"))
T = {}
for f in os.listdir(D / "struct/struct_features"):
    with open(D / "struct/struct_features" / f) as fh:
        T[f[:-4]] = np.array(fh.readline().split(), dtype=np.float32)
G = set(P) & set(S) & set(T)
F = {g: np.concatenate([T[g], np.asarray(S[g], dtype=np.float32).ravel(), np.asarray(P[g], dtype=np.float32)]) for g in G}


def X(a, b):
    return np.stack([np.concatenate([F[x], F[y]]) for x, y in zip(a, b)])


cellgraph.ensure_prep()
pr = cellgraph.predict_rows()
ok = pr.gene_a.isin(G) & pr.gene_b.isin(G)
prk = pr[ok].reset_index(drop=True)
t0 = time.time()
if VAR == "released":
    sl = pd.read_csv(D / "Human_SL_ff.csv", sep="\t"); ns = pd.read_csv(D / "Human_nonSL.csv")
    a = list(sl.iloc[:, 0]) + list(ns.iloc[:, 0]); b = list(sl.iloc[:, 1]) + list(ns.iloc[:, 1])
    k = [i for i in range(len(a)) if a[i] in G and b[i] in G]
    Xt = X([a[i] for i in k], [b[i] for i in k]); mu, sd = Xt.mean(0), Xt.std(0)
    model = s2s.MLP(Xt.shape[1], [256, 128, 64], 1, dropout=0.5)
    model.load_state_dict(torch.load(D / "bestmodel.pt", map_location="cpu"))
else:
    tr = pd.read_parquet(cellgraph.WORK / "human_train_pairs.parquet")
    tr = tr[tr.gene_a.isin(G) & tr.gene_b.isin(G)]
    fit, val = tr[tr.split == "fit"], tr[tr.split == "valid"]
    Xf, Xv = X(fit.gene_a, fit.gene_b), X(val.gene_a, val.gene_b)
    mu, sd = Xf.mean(0), Xf.std(0); sd[sd == 0] = 1
    Xf, Xv = (Xf - mu) / sd, (Xv - mu) / sd
    yf, yv = fit.label.values.astype(np.float32), val.label.values.astype(np.float32)
    model = s2s.MLP(Xf.shape[1], [256, 128, 64], 1, dropout=0.5)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))
    opt = torch.optim.Adam(model.parameters(), lr=0.004, weight_decay=0.001)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    tl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(Xf, dtype=torch.float32), torch.tensor(yf)), batch_size=64, shuffle=True)
    Xv_t, yv_t = torch.tensor(Xv, dtype=torch.float32), torch.tensor(yv)
    best, state, wait = 1e9, None, 0
    from sklearn.metrics import roc_auc_score
    for ep in range(10000):
        model.train()
        for xb, yb in tl:
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb.unsqueeze(1)) + 0.001 * sum(torch.norm(p, 2) for p in model.parameters())
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            ov = model(Xv_t); vl = crit(ov, yv_t.unsqueeze(1)).item()
        sch.step(vl)
        print(f"epoch {ep} val_loss {vl:.4f} val_auc {roc_auc_score(yv, ov.squeeze(1).numpy()):.4f} {time.time() - t0:.0f}s", flush=True)
        if vl < best:
            best, state, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 15:
                break
    model.load_state_dict(state)
sd = np.where(sd == 0, 1, sd)
model.eval()
with torch.no_grad():
    s1 = model(torch.tensor((X(prk.gene_a, prk.gene_b) - mu) / sd, dtype=torch.float32)).squeeze(1).numpy()
    s2 = model(torch.tensor((X(prk.gene_b, prk.gene_a) - mu) / sd, dtype=torch.float32)).squeeze(1).numpy()
name = "struct2sl" if VAR == "slbtrain" else "struct2sl__released"
cellgraph.write_preds(prk, (s1 + s2) / 2, name)
print(f"covered {ok.sum()} of {len(pr)} human rows; runtime_sec {int(time.time() - t0)}")
