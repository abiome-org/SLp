"""Train KR4SL (inductive RED-GNN variant, external/models/KR4SL/inductive) on SLB inputs from build.py and score
every human pair of SLB_SPLIT. Hyper-parameters as the released train.py (lr 2.1e-4, lamb 0.0011, decay 0.9937,
hidden 48, attn 1, 3 layers, batch 50, 15 epochs, validation every 3 epochs, early stop 3, model selected on
validation NDCG@50). The released inductive evaluate() references an undefined `filters` variable (NameError), so
validation NDCG@50 is computed here with the released cal_ndcg on the same score matrix. Dev scoring: for a pair
(a, b) the model is queried with (a, SL_GsG, ?) in inductive mode and the score of b is read (and vice versa);
entities not reached within 3 hops get 0. Requires CUDA (the released model hard-codes .cuda())."""
import os, sys, time, random, shutil
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(os.environ.get("SLB_BENCH", ROOT / "data/bench/slb1.2"))
if not BENCH.is_absolute():
    BENCH = ROOT / BENCH
WORK = Path(os.environ.get("SLB_WORK", ROOT / "external/models/_slb_work" / BENCH.name))
SPLIT = os.environ.get("SLB_SPLIT", "dev")
D = WORK / "kr4sl"
torch.set_num_threads(int(os.environ.get("SLB_THREADS", "8"))); torch.set_num_interop_threads(1)
# lay out KR4SL's expected relative paths: <run>/inductive (code), <run>/data/all_entities*.{txt,npy}, <run>/SLB{,_ind}
R = D / "run"
if R.exists():
    shutil.rmtree(R)
shutil.copytree(ROOT / "external/models/KR4SL/inductive", R / "inductive")
(R / "data").mkdir(parents=True)
for f in ["all_entities.txt", "all_entities_pretrain_emb.npy"]:
    os.symlink(D / f, R / "data" / f)
for d in ["SLB", "SLB_ind"]:
    shutil.copytree(D / d, R / d)
# torch >= 2 refuses to index a CPU tensor with CUDA indices (the released code relied on torch 1.11 allowing it):
# keep the pretrained entity embeddings on the GPU instead (identical values)
_m = (R / "inductive/models.py").read_text()
_m = _m.replace("entity_pretrain_emb = torch.tensor(entity_pretrain_emb, dtype=torch.float32)", "entity_pretrain_emb = torch.tensor(entity_pretrain_emb, dtype=torch.float32).cuda()")
(R / "inductive/models.py").write_text(_m)
# numpy >= 1.24 refuses ragged np.array(list of answer arrays) without dtype=object
_l = (R / "inductive/load_data.py").read_text()
for _v in ["train_a", "valid_a", "test_a"]:
    _l = _l.replace(f"np.array(self.{_v})", f"np.array(self.{_v}, dtype=object)")
(R / "inductive/load_data.py").write_text(_l)
os.chdir(R / "inductive"); sys.path.insert(0, str(R / "inductive"))
import types
sys.modules.setdefault("ipdb", types.ModuleType("ipdb"))
from load_data import DataLoader
from base_model import BaseModel
from utils import cal_ndcg
seed = 1234
np.random.seed(seed); random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class O:
    pass


opts = O(); opts.lr = 0.00021; opts.lamb = 0.0011; opts.decay_rate = 0.9937; opts.hidden_dim = 48; opts.attn_dim = 1
opts.act = "relu"; opts.n_layer = 3; opts.n_batch = 50; opts.load_ckpt = False; opts.explain = False; opts.seed = seed
opts.perf_file = str(R / "perf.txt")
loader = DataLoader("../SLB", opts)
opts.n_ent, opts.n_rel = loader.n_ent, loader.n_rel
model = BaseModel(opts, loader)


def val_ndcg50():
    model.model.eval(); s_all, o_all = [], []
    gidx = np.array(list(loader.entitypeid2geneid.keys()))
    with torch.no_grad():
        for s in range(0, model.n_valid, 50):
            subs, rels, objs = loader.get_batch_sl(np.arange(s, min(model.n_valid, s + 50)), data="valid")
            sc, _, _ = model.model(subs, rels, mode="transductive", train_mode="eval")
            s_all.append(sc.cpu().numpy()[:, gidx]); o_all.append(objs[:, gidx])
    return cal_ndcg(np.vstack(s_all), np.vstack(o_all), None, n=50)[0]


t0 = time.time(); best, state, bad = -1, None, 0
for ep in range(15):
    model.train_batch()
    if (ep + 1) % 3 == 0:
        v = val_ndcg50()
        print(f"epoch {ep} val NDCG@50 {v:.4f} {time.time() - t0:.0f}s", flush=True)
        if v > best:
            best, state = v, {k: x.detach().clone() for k, x in model.model.state_dict().items()}
        else:
            bad += 1
            if bad == 3:
                break
model.model.load_state_dict(state); model.model.eval()
f = BENCH / f"{SPLIT}.parquet"
if not f.exists():
    f = BENCH / f"{SPLIT}_inputs.parquet"
pr = pd.read_parquet(f, columns=["example_id", "species", "gene_a", "gene_b"]); pr = pr[pr.species == "human"].reset_index(drop=True)
e2i = loader.entity2id_ind; slr = loader.relation2id["SL_GsG"]
ok = pr.gene_a.isin(e2i) & pr.gene_b.isin(e2i)
heads = sorted(set(pr.gene_a[ok]) | set(pr.gene_b[ok]))
row = {}
with torch.no_grad():
    for s in range(0, len(heads), 50):
        hb = heads[s:s + 50]
        sc, _, _ = model.model(np.array([e2i[h] for h in hb]), np.array([slr] * len(hb)), mode="inductive", train_mode="eval")
        sc = sc.cpu().numpy()
        for k, h in enumerate(hb):
            row[h] = sc[k]
score = np.full(len(pr), np.nan)
for i in np.where(ok)[0]:
    a, b = pr.gene_a[i], pr.gene_b[i]
    score[i] = (row[a][e2i[b]] + row[b][e2i[a]]) / 2
out = (ROOT / "results/models" if BENCH.name == "slb1.2" else ROOT / "results/models" / BENCH.name) / f"kr4sl_{SPLIT}.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
res = pd.DataFrame({"example_id": pr.example_id[ok].values, "score": score[ok.values]})
res.to_parquet(out, index=False)
print(f"wrote {out}: {len(res)} of {len(pr)} human rows; best val NDCG@50 {best:.4f}; zero scores {(res.score == 0).mean():.3f}; runtime_sec {int(time.time() - t0)}")
