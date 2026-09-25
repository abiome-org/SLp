"""MLEC-iSL (Fan et al., Brief Bioinform 2024, bbae425) on SLB, human, retrained on SLB train only.
Original recipe (external/models/MLEC-iSL/code, task='connectivity'): a multi-layer encoder (population CCLE
expression/essentiality PCA-32 features + cell-specific [exp, mut, cnv, ess] encoder -> per-network 2-layer GCN ->
full graph transformer over all genes) regresses each gene's *SL connectivity* (number of SL partners) with MSE;
a logistic regression on (connectivity_a, connectivity_b) then scores pairs (fit on true connectivity of training
genes, applied to predicted connectivity of test genes). AdamW lr 5e-5, embed 32, hidden 64, 4 heads, 1 layer.
SLB changes:
 - connectivity targets are computed from SLB *fit* rows only, per context (the audited repo computed them before
   the gene split, leaking held-out edges into training targets);
 - contexts pooled with shared weights: one optimiser step per context per epoch using that context's omics;
 - graphs: Opticon pathway + BioGRID physical; the 'PPI-genetic' (BioGRID genetic interactions) view is dropped;
 - node universe: SLB genes + graph genes with mean DepMap log2(TPM+1) > 1 (the original kept genes expressed in its one cell line), so the full
   attention fits a 24 GB GPU; max 300 epochs, patience 50 (original 1000 / 200)."""
import os, sys, time, random
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/models/_common"))
sys.path.insert(0, str(ROOT / "external/models/MLEC-iSL/code"))
import cellgraph
from model import MLEC_iSL

SEED = int(os.environ.get("SLB_SEED", "5959"))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(int(os.environ.get("SLB_THREADS", "8"))); torch.set_num_interop_threads(1)
dev = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("SLB_GPU", "0") == "1" else "cpu")
EPOCHS, PATIENCE = int(os.environ.get("MLEC_EPOCHS", 300)), int(os.environ.get("MLEC_PATIENCE", 50))

cellgraph.ensure_prep()
W = cellgraph.WORK
tr = pd.read_parquet(W / "human_train_rows.parquet"); tr = tr[tr.split.isin(["fit", "valid"])]
pr = cellgraph.predict_rows()
pan = pd.read_parquet(W / "depmap_pan_feats.parquet").set_index("gene")
expressed = set(pan.index[pan.exp_mean > float(os.environ.get("MLEC_EXP_MIN", "1.0"))])
opt_ = pd.read_csv(ROOT / "data/raw/mlec_isl/Opticon_networks.csv", encoding="utf-8-sig").set_axis(["gene1", "gene2"], axis=1)
b = pd.read_csv(cellgraph.MV / "BIOGRID-9606.csv", index_col=0)
b = b[b["Experimental System Type"] == "physical"][["Official Symbol Interactor A", "Official Symbol Interactor B"]].set_axis(["gene1", "gene2"], axis=1)
graphs = {"pathway": opt_, "PPI-physical": b}
slb_genes = set(tr.gene_a) | set(tr.gene_b) | set(pr.gene_a) | set(pr.gene_b)
genes = set(slb_genes)
for g in graphs.values():
    g = g[g.gene1.isin(expressed) & g.gene2.isin(expressed)]
    genes |= set(g.gene1) | set(g.gene2)
genes = sorted(str(x) for x in genes); gm = {g: i for i, g in enumerate(genes)}; N = len(genes)
E = []
for k, g in graphs.items():
    g = g[g.gene1.astype(str).isin(gm) & g.gene2.astype(str).isin(gm)]
    e = np.array([g.gene1.astype(str).map(gm).values, g.gene2.astype(str).map(gm).values])
    if k != "pathway":  # BioGRID made undirected as in the original loader; Opticon kept as directed regulator->target
        e = np.concatenate([e, e[::-1]], 1)
    E.append(torch.tensor(np.unique(e, axis=1), dtype=torch.long, device=dev))
print(f"N={N} genes; edges " + ", ".join(f"{k}:{e.shape[1]}" for k, e in zip(graphs, E)), flush=True)
if os.environ.get("MLEC_DRY"):
    sys.exit(0)
# population (CCLE/DepMap) features: PCA-32 over cell lines of expression and of essentiality
Z = np.load(W / "depmap_matrices.npz", allow_pickle=True)
glob = []
for key in ["exp", "ess"]:
    M = np.nan_to_num(Z[key]).T  # genes x lines
    emb = PCA(n_components=32, random_state=0).fit_transform(M)
    x = np.zeros((N, 32)); idx = {g: i for i, g in enumerate(Z["genes"])}
    for g, i in gm.items():
        if g in idx:
            x[i] = emb[idx[g]]
    glob.append(torch.tensor(scale(x), dtype=torch.float, device=dev))
cf = pd.read_parquet(W / "depmap_context_feats.parquet")
ctxs = sorted(set(tr.context_id) | set(pr.context_id))
XC = {}
for c in ctxs:
    d = cf[cf.context_id == c].set_index("gene")[["exp", "mut", "cnv", "ess"]].reindex(genes).fillna(0.0).values
    XC[c] = torch.tensor(scale(d), dtype=torch.float, device=dev)
for d in (tr, pr):
    d["i"], d["j"] = d.gene_a.map(gm), d.gene_b.map(gm)


def connectivity(rows):
    r2 = pd.concat([rows[["context_id", "i", "label"]], rows[["context_id", "j", "label"]].rename(columns={"j": "i"})])
    return r2.groupby(["context_id", "i"]).label.sum().astype(float)


conn_fit = connectivity(tr[tr.split == "fit"]); conn_val = connectivity(tr[tr.split == "valid"])
model = MLEC_iSL(64, 32, 4, 1, num_CCLE=2, num_network=len(E), cell_specific_flag=True, GT_flag=True, task="connectivity").to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)


def fwd(c, idx):
    return model(E, glob, [XC[c]], torch.tensor(idx, dtype=torch.long, device=dev))


best, state, wait, t0 = 1e18, None, 0, time.time()
for ep in range(1, EPOCHS + 1):
    model.train(); tl = 0.0
    for c in conn_fit.index.get_level_values(0).unique():
        s = conn_fit.loc[c]
        opt.zero_grad()
        loss = F.mse_loss(fwd(c, s.index.values).view(-1), torch.tensor(s.values, dtype=torch.float, device=dev))
        loss.backward(); opt.step(); tl += loss.item()
    model.eval(); vl, n = 0.0, 0
    with torch.no_grad():
        for c in conn_val.index.get_level_values(0).unique():
            s = conn_val.loc[c]
            vl += F.mse_loss(fwd(c, s.index.values).view(-1), torch.tensor(s.values, dtype=torch.float, device=dev)).item() * len(s); n += len(s)
    vl /= n
    if vl < best:
        best, state, wait = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
    else:
        wait += 1
    print(f"epoch {ep} train_mse {tl:.3f} val_mse {vl:.4f} wait {wait} {time.time() - t0:.0f}s", flush=True)
    if wait >= PATIENCE:
        break
    if time.time() - t0 > float(os.environ.get('MLEC_MAXTIME', '1800')):  # SLB: wall-clock cap (shared GPU); best-val checkpoint is kept
        print('time cap reached'); break
model.load_state_dict(state); model.eval()
# logistic regression on (conn_a, conn_b): train on true fit connectivity, balanced 1:1 as the original
fit = tr[tr.split == "fit"].copy()
fit["ca"] = [conn_fit.get((c, i), 0.0) for c, i in zip(fit.context_id, fit.i)]
fit["cb"] = [conn_fit.get((c, j), 0.0) for c, j in zip(fit.context_id, fit.j)]
pos = fit[fit.label == 1]; neg = fit[fit.label == 0].sample(n=len(pos), random_state=SEED)
tb = pd.concat([pos, neg])
clf = LogisticRegression(class_weight="balanced").fit(tb[["ca", "cb"]].values, tb.label.values)
pred = {}
with torch.no_grad():
    for c, g in pr.groupby("context_id"):
        u = np.unique(np.concatenate([g.i.values, g.j.values]))
        p = fwd(c, u).view(-1).cpu().numpy()
        for k, v in zip(u, p):
            pred[(c, k)] = v
Xd = np.array([[pred[(c, i)], pred[(c, j)]] for c, i, j in zip(pr.context_id, pr.i, pr.j)])
s = (clf.predict_proba(Xd)[:, 1] + clf.predict_proba(Xd[:, ::-1])[:, 1]) / 2
cellgraph.write_preds(pr, s, os.environ.get("SLB_NAME", "mlec_isl"))
print("runtime_sec", int(time.time() - t0))
