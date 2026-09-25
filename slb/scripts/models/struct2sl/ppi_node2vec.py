"""Leak-free replacement for Struct2SL's PPI feature: node2vec (the repo's node2vec-master defaults: 128 dims, walk
length 80, 10 walks per node, window 10, p = q = 1) on the human BioGRID-physical edges of the shared bundle
(data/interim/bundle/human/ppi.parquet, biogrid_phys > 0). The released feature was node2vec on STRING v12 physical
links, whose combined score includes textmining / experiment channels (see lead's STRING warning). Output:
SLB_WORK/struct2sl_ppi_node2vec.pkl  {gene symbol: 128-d float32}"""
import os, pickle, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch_geometric.nn import Node2Vec
ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(os.environ.get("SLB_BENCH", ROOT / "data/slb"))
WORK = Path(os.environ.get("SLB_WORK", ROOT / "external/models/_slb_work" / BENCH.name))
torch.set_num_threads(int(os.environ.get("SLB_THREADS", "8"))); torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() and os.environ.get("SLB_GPU") == "1" else "cpu"
p = pd.read_parquet(ROOT / "data/interim/bundle/human/ppi.parquet")
p = p[p.biogrid_phys > 0]
genes = sorted(set(p.gene_a) | set(p.gene_b)); gm = {g: i for i, g in enumerate(genes)}
e = torch.tensor(np.array([p.gene_a.map(gm).values, p.gene_b.map(gm).values]), dtype=torch.long)
e = torch.cat([e, e.flip(0)], 1)
m = Node2Vec(e, embedding_dim=128, walk_length=80, context_size=10, walks_per_node=10, p=1.0, q=1.0, num_negative_samples=1, sparse=True).to(dev)
ld = m.loader(batch_size=256, shuffle=True, num_workers=0)
opt = torch.optim.SparseAdam(list(m.parameters()), lr=0.01)
t0 = time.time()
for ep in range(int(os.environ.get("N2V_EPOCHS", 5))):
    tot = 0
    for pw, nw in ld:
        opt.zero_grad(); l = m.loss(pw.to(dev), nw.to(dev)); l.backward(); opt.step(); tot += l.item()
    print(f"epoch {ep} loss {tot / len(ld):.4f} {time.time() - t0:.0f}s", flush=True)
emb = m.embedding.weight.detach().cpu().numpy().astype(np.float32)
out = WORK / "struct2sl_ppi_node2vec.pkl"; out.parent.mkdir(parents=True, exist_ok=True)
pickle.dump({g: emb[i] for g, i in gm.items()}, open(out, "wb"))
print("wrote", out, len(genes), "genes", e.shape[1] // 2, "edges")
