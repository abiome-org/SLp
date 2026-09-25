"""MVGCN-iSL (Fan et al. 2023, Front. Genet.) on SLB, human, retrained on SLB train only.
Uses the original GCN_pool model class (external/models/MVGCNiSL/code/model.py) and the original training recipe
(AdamW lr 1e-4, balanced negatives resampled every epoch, batch 512, BCE, early stopping on validation loss,
patience 150, max 500 epochs, max-pooling over graph views). Changes for SLB:
contexts are pooled with shared weights (each context's pairs are decoded from embeddings computed with that
context's omics node features); the leaky PPI-genetic view is dropped; validation = held-aside train families."""
import os, sys, time, random
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/models/_common"))
sys.path.insert(0, str(ROOT / "external/models/MVGCNiSL/code"))
import cellgraph
from model import GCN_pool

SEED = int(os.environ.get("SLB_SEED", "5959"))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(int(os.environ.get("SLB_THREADS", "8"))); torch.set_num_interop_threads(1)
dev = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("SLB_GPU", "0") == "1" else "cpu")
EPOCHS, PATIENCE, LR, BS, OUT = int(os.environ.get("MV_EPOCHS", 500)), int(os.environ.get("MV_PATIENCE", 150)), 1e-4, 512, 64

D = cellgraph.build()
E = [torch.tensor(e, dtype=torch.long, device=dev) for e in D["edges"]]
X = {c: torch.tensor(x, device=dev) for c, x in D["X"].items()}
tr = D["train"]
fit, val = tr[tr.split == "fit"], tr[tr.split == "valid"]
model = GCN_pool(4, OUT, len(E)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR)


def embed(c):
    zs = [model.encode(X[c], e) for e in E]
    z = torch.cat(zs, 1).unsqueeze(1).reshape(zs[0].shape[0], len(E), -1).transpose(1, 2)
    return F.max_pool2d(z, (1, len(E))).squeeze(2)


def epoch_train():
    model.train(); tot = 0.0
    for c, g in fit.groupby("context_id"):
        pos = g[g.label == 1]; neg = g[g.label == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        neg = neg.sample(n=min(len(pos), len(neg)))
        ij = np.concatenate([np.stack([pos.i, pos.j]), np.stack([neg.i, neg.j])], 1)
        ij = np.concatenate([ij, ij[::-1]], 1)  # both orientations (original duplicate=True)
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))] * 2)
        p = np.random.permutation(ij.shape[1]); ij, y = ij[:, p], y[p]
        for s in range(0, ij.shape[1], BS):
            opt.zero_grad()
            z = embed(c)
            logit = model.decode(z, torch.tensor(ij[:, s:s + BS], device=dev))
            loss = F.binary_cross_entropy_with_logits(logit.view(-1), torch.tensor(y[s:s + BS], dtype=torch.float, device=dev))
            loss.backward(); opt.step(); tot += loss.item()
    return tot


@torch.no_grad()
def score(rows):
    model.eval(); out = np.zeros(len(rows))
    for c, g in rows.groupby("context_id"):
        z = embed(c)
        a = torch.tensor(np.stack([g.i.values, g.j.values]), device=dev)
        s = (torch.sigmoid(model.decode(z, a).view(-1)) + torch.sigmoid(model.decode(z, a.flip(0)).view(-1))) / 2
        out[rows.index.get_indexer(g.index)] = s.cpu().numpy()
    return out


@torch.no_grad()
def val_loss():
    # balanced validation set, as the original (generate_torch_edges(balanced=True) for val)
    rng = np.random.default_rng(0)
    vp = val[val.label == 1]; vn = val[val.label == 0].sample(n=min(len(vp), (val.label == 0).sum()), random_state=0)
    import pandas as pd; v = pd.concat([vp, vn])
    v = v.reset_index(drop=True)
    s = np.clip(score(v), 1e-7, 1 - 1e-7)
    from sklearn.metrics import roc_auc_score
    return float(-np.mean(v.label * np.log(s) + (1 - v.label) * np.log(1 - s))), roc_auc_score(v.label, s)


best, best_state, wait, t0 = 1e9, None, 0, time.time()
for ep in range(1, EPOCHS + 1):
    l = epoch_train()
    vl, va = val_loss()
    if vl < best:
        best, best_state, wait = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
    else:
        wait += 1
    print(f"epoch {ep} train_loss {l:.3f} val_loss {vl:.4f} val_auc {va:.4f} best {best:.4f} wait {wait} {time.time() - t0:.0f}s", flush=True)
    if wait >= PATIENCE:
        print("early stop"); break
    if time.time() - t0 > float(os.environ.get('MV_MAXTIME', '1800')):  # SLB: wall-clock cap (shared GPU); best-val checkpoint is kept
        print('time cap reached'); break
model.load_state_dict(best_state)
pr = D["pred"]
cellgraph.write_preds(pr, score(pr), os.environ.get("SLB_NAME", "mvgcn_isl"))
print("runtime_sec", int(time.time() - t0))
