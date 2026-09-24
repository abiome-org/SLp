"""LukePi (Tao et al., bioRxiv 2025; JieZheng-ShanghaiTech/LukePi) on SLB, human.
Self-supervised HGT pre-trained on PrimeKG (no SL labels; authors' checkpoint Primekg_HGT_0.2_0.001 from their Google
Drive, data/raw/lukepi) + an MLP pair head fine-tuned on SLB train pairs, following src/finetune_LukePi.py and
test_LukePi.sh: frozen encoder (freeze=1), node inputs = seeded 16-d node-type embeddings, HGTLoader with 1024 nodes
per type x 4 layers and one batch holding all training genes, head 256-128-64-32-2 with cross-entropy, Adam lr1 0.003,
50 epochs, last epoch kept (the released script has no model selection). Pairs: SLB human train `fit` pairs with
their measured labels (contexts pooled: SL in >= 1 context = 1); validation = held-aside train families (logged only).
Genes are mapped to PrimeKG gene/protein nodes by symbol (PrimeKG node table shipped with MiT4SL data); pairs with an
unmapped gene are left out (median-filled by the evaluator). Score = softmax P(SL), averaged over both orders."""
import os, sys, json, pickle, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/models/_common")); sys.path.insert(0, str(ROOT / "external/models/LukePi/src"))
import cellgraph
# the released src/model.py does not import (IndentationError at ` class GIN`, line 68); take the HGT class only
_src = open(ROOT / "external/models/LukePi/src/model.py").read(); _ns = {}
exec(_src[:_src.index(" class GIN")], _ns); HGT = _ns["HGT"]
from torch_geometric.loader import HGTLoader
from sklearn.metrics import roc_auc_score
torch.set_num_threads(int(os.environ.get("SLB_THREADS", "8")))
dev = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("SLB_GPU", "0") == "1" else "cpu")
L = ROOT / "data/raw/lukepi/Configuration_LukePi"
cfg = json.load(open(L / "Primekg_HGT_0.2_0.001/config.json"))
torch.manual_seed(0); np.random.seed(0)
kg = pickle.load(open(L / "kgdata.pkl", "rb"))
ni = json.load(open(ROOT / "external/models/LukePi/data/BKG/node_index_dic.json"))
NT = "gene/protein"
emb = torch.nn.Embedding(len(kg.node_types), 16); torch.nn.init.xavier_uniform_(emb.weight.data)
for i, t in enumerate(kg.node_types):
    kg[t].x = emb(torch.tensor(i)).repeat([kg[t].x.shape[0], 1]).detach()
nodes = pd.read_csv(ROOT / "data/raw/mit4sl/data/data/MultiOmics_feature/kg_data/Primenode.csv", dtype=str)
g = nodes[nodes.node_type == NT]
sym2idx = {}
for nidx, name in zip(g.node_index, g.node_name):
    if nidx in ni[NT]:
        sym2idx.setdefault(name, ni[NT][nidx])
cellgraph.ensure_prep()
tr = pd.read_parquet(cellgraph.WORK / "human_train_pairs.parquet")
tr["a"], tr["b"] = tr.gene_a.map(sym2idx), tr.gene_b.map(sym2idx)
tr = tr.dropna(subset=["a", "b"]); tr[["a", "b"]] = tr[["a", "b"]].astype(int)
fit, val = tr[tr.split == "fit"].reset_index(drop=True), tr[tr.split == "valid"].reset_index(drop=True)
pr = cellgraph.predict_rows()
pr["a"], pr["b"] = pr.gene_a.map(sym2idx), pr.gene_b.map(sym2idx)
ok = pr.a.notna() & pr.b.notna(); prk = pr[ok].reset_index(drop=True); prk[["a", "b"]] = prk[["a", "b"]].astype(int)
print(f"mapped: fit {len(fit)} pairs ({fit.label.sum()} SL), valid {len(val)}, predict {ok.sum()}/{len(pr)} rows", flush=True)
model = HGT(kg, 2 * cfg["emb_dim"], cfg["emb_dim"], cfg["num_heads"], cfg["num_layer"]).to(dev)
ck = torch.load(L / "Primekg_HGT_0.2_0.001/checkpoint", map_location="cpu")
with torch.no_grad():  # materialise lazy Linear(-1) layers before loading
    model({t: kg[t].x[:2].to(dev) for t in kg.node_types}, {e: torch.zeros((2, 0), dtype=torch.long, device=dev) for e in kg.edge_types})
model.load_state_dict(ck["model_state_dict"])
for p in model.parameters():
    p.requires_grad = False
E = cfg["emb_dim"]
head = nn.Sequential(nn.Linear(2 * E, E), nn.ReLU(), nn.Linear(E, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2)).to(dev)
opt = torch.optim.Adam(head.parameters(), lr=0.003, weight_decay=0)


def loader(pairs):
    nodes = sorted(set(pairs.a) | set(pairs.b))
    mask = torch.zeros(kg[NT].x.shape[0], dtype=torch.bool); mask[nodes] = True
    return HGTLoader(kg, num_samples={k: [cfg["sample_nodes"]] * cfg["sample_layers"] for k in kg.node_types}, shuffle=False,
                     batch_size=len(nodes), input_nodes=(NT, mask), num_workers=0), len(nodes)


def logits(batch, n, pairs, flip=False):
    rep = model(batch.x_dict, batch.edge_index_dict)[NT]
    if hasattr(batch[NT], "n_id"):
        ids = batch[NT].n_id[:n].cpu().numpy()
    else:  # PyG 2.2 HGTLoader: the first `batch_size` nodes are the (mask-ordered, i.e. sorted) input nodes
        ids = np.array(sorted(set(pairs.a) | set(pairs.b)))
    m = {int(v): k for k, v in enumerate(ids)}
    a, b = pairs.a.map(m).values, pairs.b.map(m).values
    if flip:
        a, b = b, a
    return head(torch.cat([rep[a], rep[b]], 1))


ld_fit, n_fit = loader(fit); ld_val, n_val = loader(val)
t0 = time.time(); crit = nn.CrossEntropyLoss()
for ep in range(1, 51):
    model.eval(); head.train()
    for batch in ld_fit:
        batch = batch.to(dev)
        out = logits(batch, n_fit, fit)
        loss = crit(out, torch.tensor(fit.label.values, device=dev))
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        for batch in ld_val:
            s = torch.softmax(logits(batch.to(dev), n_val, val), 1)[:, 1].cpu().numpy()
    print(f"epoch {ep} loss {loss.item():.4f} valid AUROC {roc_auc_score(val.label, s):.4f} {time.time() - t0:.0f}s", flush=True)
ld_pr, n_pr = loader(prk)
with torch.no_grad():
    for batch in ld_pr:
        batch = batch.to(dev)
        s = (torch.softmax(logits(batch, n_pr, prk), 1)[:, 1] + torch.softmax(logits(batch, n_pr, prk, flip=True), 1)[:, 1]).cpu().numpy() / 2
cellgraph.write_preds(prk, s, "lukepi")
print("runtime_sec", int(time.time() - t0))
