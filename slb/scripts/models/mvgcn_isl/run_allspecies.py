"""MVGCN-iSL `allspecies` variant: the same GCN_pool architecture and training recipe as run.py, applied per species
with species-agnostic inputs from the shared bundle (data/interim/bundle/<sp>/):
  views       SL (SLB fit positives of that species), PPI (bundle BioGRID physical or STRING non-exp/non-textmining
              channel >= 400), co-expression (bundle STRING coexpression channel >= 400; stands in for MVGCN's CCLE
              co-expression view), [co-essentiality has no cross-species analogue and is dropped]
  node feats  single-gene fitness effect (bundle fitness.parquet) + 16 PCA components of ESM-2 650M mean embeddings
              (data/interim/esm2_650m/<sp>.parquet; zeros where missing), z-scored (MVGCN's 4 cell-line omics features
              exist only for human cancer lines)
Contexts of a species share node features (no per-strain omics). Output: results/models/[<bench>/]mvgcn_isl__allspecies_<split>.parquet"""
import os, sys, time, random
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/models/_common")); sys.path.insert(0, str(ROOT / "external/models/MVGCNiSL/code"))
import cellgraph
from model import GCN_pool
torch.set_num_threads(int(os.environ.get("SLB_THREADS", "8"))); torch.set_num_interop_threads(1)
dev = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("SLB_GPU", "0") == "1" else "cpu")
EPOCHS, PATIENCE, LR, BS, OUT = int(os.environ.get("MV_EPOCHS", 500)), int(os.environ.get("MV_PATIENCE", 150)), 1e-4, 512, 64
BENCH, WORK, SPLIT = cellgraph.BENCH, cellgraph.WORK, cellgraph.SPLIT
f = BENCH / f"{SPLIT}.parquet"
if not f.exists():
    f = BENCH / f"{SPLIT}_inputs.parquet"
PRED = pd.read_parquet(f, columns=["example_id", "species", "context_id", "gene_a", "gene_b"])
species = sorted(pd.read_parquet(BENCH / "train.parquet", columns=["species"]).species.unique())
if os.environ.get("SLB_ONLY"):
    species = [x for x in species if x in os.environ["SLB_ONLY"].split(",")]
parts = []
for sp in species:
    t0 = time.time(); random.seed(5959); np.random.seed(5959); torch.manual_seed(5959)
    if not (WORK / f"{sp}_train_rows.parquet").exists():
        import subprocess
        subprocess.run(["uv", "run", "python", str(ROOT / "scripts/models/_common/slb_pairs.py")], env={**os.environ, "SLB_SPECIES": sp}, check=True, cwd=ROOT)
    tr = pd.read_parquet(WORK / f"{sp}_train_rows.parquet"); tr = tr[tr.split.isin(["fit", "valid"])]
    pr = PRED[PRED.species == sp].reset_index(drop=True)
    B = ROOT / "data/interim/bundle" / sp
    if len(pr) == 0 or not (B / "ppi.parquet").exists() or tr.label.sum() == 0:
        print(f"{sp}: skipped (no prediction rows / bundle / positives)"); continue
    ppi = pd.read_parquet(B / "ppi.parquet"); sc = [c for c in ppi.columns if c.startswith("string_")]
    views = {"PPI": ppi[(ppi.biogrid_phys > 0) | (ppi[sc].max(axis=1) >= 400)],
             "co-exp": ppi[ppi.get("string_coexpression", pd.Series(0, index=ppi.index)) >= 400]}
    genes = set(tr.gene_a) | set(tr.gene_b) | set(pr.gene_a) | set(pr.gene_b)
    for v in views.values():
        genes |= set(v.gene_a) | set(v.gene_b)
    genes = sorted(genes); gm = {g: i for i, g in enumerate(genes)}; N = len(genes)
    E = []
    p = tr[(tr.split == "fit") & (tr.label == 1)]
    e = np.array([p.gene_a.map(gm).values, p.gene_b.map(gm).values]); E.append(np.concatenate([e, e[::-1]], 1))
    for v in views.values():
        e = np.array([v.gene_a.map(gm).values, v.gene_b.map(gm).values]); E.append(np.concatenate([e, e[::-1]], 1))
    E = [torch.tensor(np.unique(x, axis=1), dtype=torch.long, device=dev) for x in E]
    fit_ = pd.read_parquet(B / "fitness.parquet").drop_duplicates("gene").set_index("gene").effect
    X = np.zeros((N, 17))
    X[:, 0] = fit_.reindex(genes).fillna(0.0).values
    ef = ROOT / "data/interim/esm2_650m" / f"{sp}.parquet"
    if ef.exists():
        es = pd.read_parquet(ef).drop_duplicates("gene").set_index("gene")
        es = es[[c for c in es.columns if c.startswith("e")]]
        pc = PCA(n_components=16, random_state=0).fit_transform(es.values)
        pcd = pd.DataFrame(pc, index=es.index)
        X[:, 1:] = pcd.reindex(genes).fillna(0.0).values
    Xt = torch.tensor(scale(X), dtype=torch.float, device=dev)
    for d in (tr, pr):
        d["i"], d["j"] = d.gene_a.map(gm), d.gene_b.map(gm)
    fit, val = tr[tr.split == "fit"], tr[tr.split == "valid"]
    model = GCN_pool(Xt.shape[1], OUT, len(E)).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=LR)

    def embed():
        zs = [model.encode(Xt, e) for e in E]
        z = torch.cat(zs, 1).unsqueeze(1).reshape(N, len(E), -1).transpose(1, 2)
        return F.max_pool2d(z, (1, len(E))).squeeze(2)

    def score(rows):
        model.eval()
        with torch.no_grad():
            z = embed(); a = torch.tensor(np.stack([rows.i.values, rows.j.values]), device=dev)
            return ((torch.sigmoid(model.decode(z, a).view(-1)) + torch.sigmoid(model.decode(z, a.flip(0)).view(-1))) / 2).cpu().numpy()

    vp = val[val.label == 1]; vn = val[val.label == 0].sample(n=min(len(vp), int((val.label == 0).sum())), random_state=0)
    V = pd.concat([vp, vn]).reset_index(drop=True)
    best, state, wait = 1e9, None, 0
    posf, negf = fit[fit.label == 1], fit[fit.label == 0]
    for ep in range(1, EPOCHS + 1):
        model.train()
        n = negf.sample(n=min(len(posf), len(negf)))
        ij = np.concatenate([np.stack([posf.i, posf.j]), np.stack([n.i, n.j])], 1); ij = np.concatenate([ij, ij[::-1]], 1)
        y = np.concatenate([np.ones(len(posf)), np.zeros(len(n))] * 2); k = np.random.permutation(len(y)); ij, y = ij[:, k], y[k]
        for s in range(0, len(y), BS):
            opt.zero_grad(); z = embed()
            loss = F.binary_cross_entropy_with_logits(model.decode(z, torch.tensor(ij[:, s:s + BS], device=dev)).view(-1), torch.tensor(y[s:s + BS], dtype=torch.float, device=dev))
            loss.backward(); opt.step()
        if len(V):
            s_ = np.clip(score(V), 1e-7, 1 - 1e-7); vl = float(-np.mean(V.label * np.log(s_) + (1 - V.label) * np.log(1 - s_)))
        else:
            vl = 0.0
        if vl < best:
            best, state, wait = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
        if ep % 10 == 0:
            print(f"{sp} epoch {ep} val_loss {vl:.4f} best {best:.4f} {time.time() - t0:.0f}s", flush=True)
        if wait >= PATIENCE:
            break
        if time.time() - t0 > float(os.environ.get('MV_MAXTIME_SPECIES', '240')):  # SLB: per-species wall-clock cap
            print('time cap reached'); break
    model.load_state_dict(state)
    parts.append(pd.DataFrame({"example_id": pr.example_id.values, "score": score(pr)}))
    print(f"{sp}: N={N}, views {[int(e.shape[1] // 2) for e in E]}, fit {len(fit)} ({int(fit.label.sum())} SL), scored {len(pr)}, {time.time() - t0:.0f}s", flush=True)
out = cellgraph.outdir() / f"mvgcn_isl__allspecies_{SPLIT}.parquet"; out.parent.mkdir(parents=True, exist_ok=True)
pd.concat(parts).to_parquet(out, index=False); print("wrote", out)
