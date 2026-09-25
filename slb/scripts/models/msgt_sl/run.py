"""MSGT-SL (Multi-omics Sampling-based Graph Transformer, arXiv 2310.11082) on SLB, human, retrained on SLB train.
Uses the released GCN_transformer_pool + transformer classes (external/models/MSGT-SL/MSGT-SL/code) and the
released training loop of main_version02.py: 2-layer GCN on the FIRST input graph only (the released code keeps
edge_index_list[0], which is the training SL graph), a 2-layer/4-head transformer (d_model 512) over the nodes of
each mini-batch of 50 pairs, MLP decoder on the concatenated pair embedding, AdamW lr 1e-4, balanced negatives
resampled every epoch, early stopping on validation loss (patience 5). The random-walk "sampling" of neighbour
nodes is computed but then overwritten with an empty list in the released code, so it is not used here either.
SLB changes: contexts pooled with shared weights (each context's pairs use that context's omics node features);
validation = held-aside train gene families; max 100 epochs (released default: 1)."""
import os, sys, time, random, types
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/models/_common"))
sys.path.insert(0, str(ROOT / "external/models/MSGT-SL/MSGT-SL/code"))
import cellgraph
from model import GCN_transformer_pool

SEED = int(os.environ.get("SLB_SEED", "5959"))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(int(os.environ.get("SLB_THREADS", "8"))); torch.set_num_interop_threads(1)
dev = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("SLB_GPU", "0") == "1" else "cpu")
args = types.SimpleNamespace(src_len=512, d_model=512, d_ff=2048, d_k=64, d_v=64, n_layers=2, n_heads=4, device=str(dev), tgt_len=512)
EPOCHS, PATIENCE, BS = int(os.environ.get("MSGT_EPOCHS", 100)), int(os.environ.get("MSGT_PATIENCE", 5)), 50

D = cellgraph.build()
E0 = torch.tensor(D["edges"][0], dtype=torch.long, device=dev)  # released code: first graph only (= SL)
print("graph used:", D["graph_names"][0])
X = {c: torch.tensor(x, device=dev) for c, x in D["X"].items()}
tr = D["train"]; fit, val = tr[tr.split == "fit"], tr[tr.split == "valid"]
model = GCN_transformer_pool(4, 64, 1, args).to(dev)


def modified_transformer(inp):  # released version pads with a CPU tensor; same logic, device-safe
    L = len(inp)
    mask = torch.ones(args.src_len, args.src_len, dtype=torch.bool, device=inp.device)
    mask[:L, :L] = 0
    mask = mask.unsqueeze(0).unsqueeze(0).repeat(1, args.n_heads, 1, 1)
    inp = torch.cat([inp, torch.zeros((args.src_len - L, 64), device=inp.device)], 0)
    return model.transformer(inp, mask)[:L]


opt = torch.optim.AdamW(model.parameters(), lr=1e-4)


def run_batch(z, ij):
    nodes = torch.unique(torch.cat([ij[0], ij[1]]))
    pos = {int(n): k for k, n in enumerate(nodes)}
    out = modified_transformer(z[nodes])
    a = out[[pos[int(i)] for i in ij[0]]]; b = out[[pos[int(i)] for i in ij[1]]]
    return model.decode(torch.cat([a, b], 1)).view(-1)


def epoch_train():
    model.train(); tot = 0.0
    for c, g in fit.groupby("context_id"):
        p, n = g[g.label == 1], g[g.label == 0]
        if len(p) == 0 or len(n) == 0:
            continue
        n = n.sample(n=min(len(p), len(n)))
        ij = np.concatenate([np.stack([p.i, p.j]), np.stack([n.i, n.j])], 1)
        y = np.concatenate([np.ones(len(p)), np.zeros(len(n))])
        k = np.random.permutation(len(y)); ij, y = ij[:, k], y[k]
        for s in range(0, max(1, len(y) - BS + 1), BS):  # released loop drops the last partial batch
            opt.zero_grad()
            z = model.encode(X[c], E0)
            lo = run_batch(z, torch.tensor(ij[:, s:s + BS], device=dev))
            loss = F.binary_cross_entropy_with_logits(lo, torch.tensor(y[s:s + BS], dtype=torch.float, device=dev))
            loss.backward(); opt.step(); tot += loss.item()
    return tot


@torch.no_grad()
def score(rows):
    model.eval(); out = np.zeros(len(rows))
    for c, g in rows.groupby("context_id"):
        z = model.encode(X[c], E0)
        idx = rows.index.get_indexer(g.index)
        ij = np.stack([g.i.values, g.j.values])
        for s in range(0, len(g), BS):
            t = torch.tensor(ij[:, s:s + BS], device=dev)
            out[idx[s:s + BS]] = ((torch.sigmoid(run_batch(z, t)) + torch.sigmoid(run_batch(z, t.flip(0)))) / 2).cpu().numpy()
    return out


vp = val[val.label == 1]; vn = val[val.label == 0].sample(n=min(len(vp), int((val.label == 0).sum())), random_state=0)
V = pd.concat([vp, vn]).reset_index(drop=True)
best, state, wait, t0 = 1e9, None, 0, time.time()
from sklearn.metrics import roc_auc_score
for ep in range(1, EPOCHS + 1):
    l = epoch_train()
    s = np.clip(score(V), 1e-7, 1 - 1e-7)
    vl = float(-np.mean(V.label * np.log(s) + (1 - V.label) * np.log(1 - s)))
    if vl < best:
        best, state, wait = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
    else:
        wait += 1
    print(f"epoch {ep} train_loss {l:.3f} val_loss {vl:.4f} val_auc {roc_auc_score(V.label, s):.4f} wait {wait} {time.time() - t0:.0f}s", flush=True)
    if wait >= PATIENCE:
        break
    if time.time() - t0 > float(os.environ.get('MSGT_MAXTIME', '1800')):  # SLB: wall-clock cap (shared GPU); best-val checkpoint is kept
        print('time cap reached'); break
model.load_state_dict(state)
pr = D["pred"]
cellgraph.write_preds(pr, score(pr), os.environ.get("SLB_NAME", "msgt_sl"))
print("runtime_sec", int(time.time() - t0))
