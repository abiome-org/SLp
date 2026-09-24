"""Re-train TransE_l2 KG embeddings (DGL-KE style: logsigmoid loss, gamma margin, chunked uniform negatives,
dim 400) on the SynLethKG copy WITHOUT SL / SR / non-SL relations (fin_kg_wo_sl_9845.csv of the SLB universe).
Replaces the embeddings shipped with NSF4SL / Feng et al., whose training KG may include SL edges.
Writes kg_TransE_l2_entity.npy (row = unified id) and nsf4sl_data/entities.tsv (identity map)."""
import os, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(os.environ.get("SLB_BENCH", ROOT / "data/bench/slb1.2"))
if not BENCH.is_absolute():
    BENCH = ROOT / BENCH
WORK = Path(os.environ.get("SLB_WORK", ROOT / "external/models/_slb_work" / BENCH.name))
PP = WORK / "feng/data/preprocessed_data"
torch.set_num_threads(int(os.environ.get("SLB_THREADS", "16")))
dev = "cuda" if torch.cuda.is_available() and os.environ.get("SLB_GPU") == "1" else "cpu"
kg = pd.read_csv(PP / "fin_kg_wo_sl_9845.csv").values.astype(np.int64)  # A, r, B
h, r, t = kg[:, 0], kg[:, 1], kg[:, 2]
n_ent, n_rel = int(max(h.max(), t.max())) + 1, int(r.max()) + 1
dim, gamma, lr, bs, neg, epochs = 400, 19.9, 0.25, 1000, 200, int(os.environ.get("TRANSE_EPOCHS", "8"))
g = torch.Generator().manual_seed(0)
E = torch.nn.Embedding(n_ent, dim, sparse=True); R = torch.nn.Embedding(n_rel, dim, sparse=True)
rng = (gamma + 2.0) / dim
torch.nn.init.uniform_(E.weight, -rng, rng); torch.nn.init.uniform_(R.weight, -rng, rng)
E, R = E.to(dev), R.to(dev)
opt = torch.optim.SparseAdam(list(E.parameters()) + list(R.parameters()), lr=0.01) if os.environ.get("TRANSE_ADAM") else torch.optim.Adagrad(list(E.parameters()) + list(R.parameters()), lr=lr)
H, Rr, T = torch.from_numpy(h), torch.from_numpy(r), torch.from_numpy(t)
for ep in range(epochs):
    perm = torch.randperm(len(H), generator=g); tot = 0.0; t0 = time.time()
    for i in range(0, len(perm), bs):
        idx = perm[i:i + bs]
        hh, rr, tt = E(H[idx].to(dev)), R(Rr[idx].to(dev)), E(T[idx].to(dev))
        pos = gamma - torch.norm(hh + rr - tt, p=2, dim=-1)
        negs = E(torch.randint(0, n_ent, (neg,), generator=g).to(dev))
        if np.random.rand() < 0.5:  # corrupt tail
            d = (hh + rr)[:, None, :] - negs[None, :, :]
        else:  # corrupt head
            d = negs[None, :, :] + (rr - tt)[:, None, :]
        negsc = gamma - torch.norm(d, p=2, dim=-1)
        loss = (-torch.nn.functional.logsigmoid(pos).mean() - torch.nn.functional.logsigmoid(-negsc).mean()) / 2
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    print(f"epoch {ep} loss {tot / (len(perm) / bs):.4f} {time.time() - t0:.0f}s", flush=True)
np.save(PP / "kg_TransE_l2_entity.npy", E.weight.detach().cpu().numpy().astype(np.float32))
(PP / "nsf4sl_data").mkdir(exist_ok=True)
pd.DataFrame({"row": np.arange(n_ent), "uid": np.arange(n_ent)}).to_csv(PP / "nsf4sl_data/entities.tsv", sep="\t", header=False, index=False)
print("saved", n_ent, dim)
