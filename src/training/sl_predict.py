from pathlib import Path
import argparse, json, pickle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/feng2024/data/preprocessed_data"
OUT = ROOT / "results/sl_predict"
SEED = 123


def csv_matrix(path):
    out = OUT / (path.stem + ".npy")
    if out.exists(): return np.load(out, mmap_mode="r")
    import pandas as pd
    n = 9845
    a = np.lib.format.open_memmap(out, mode="w+", dtype="float32", shape=(n, n))
    row = 0
    for chunk in pd.read_csv(path, index_col=0, chunksize=128):
        x = chunk.to_numpy(dtype="float32", na_value=0.0); x[~np.isfinite(x)] = 0
        a[row:row + len(x)] = x; row += len(x)
    a.flush(); return np.load(out, mmap_mode="r")


def geneformer(meta, checkpoint="Geneformer-V2-104M"):
    from safetensors import safe_open
    base = ROOT / "data/models/weights/Geneformer"
    with open(base / "geneformer/gene_name_id_dict_gc104M.pkl", "rb") as f: names = pickle.load(f)
    with open(base / "geneformer/token_dictionary_gc104M.pkl", "rb") as f: tokens = pickle.load(f)
    with safe_open(base / checkpoint / "model.safetensors", framework="numpy") as f:
        emb = f.get_tensor("bert.embeddings.word_embeddings.weight")
    out = np.zeros((len(meta), emb.shape[1]), "float32"); hit = np.zeros(len(meta), bool)
    for i, symbol in enumerate(meta.symbol):
        ens = names.get(symbol) or names.get(str(symbol).upper())
        tok = tokens.get(ens)
        if tok is not None: out[i] = emb[tok]; hit[i] = True
    return out, hit


def protein(meta):
    import io, pickle, sys, types, zipfile
    path = ROOT / "data/models/weights/SE-600M/protein_embeddings.pt"; z = zipfile.ZipFile(path)
    fake, utils = types.ModuleType("torch"), types.ModuleType("torch._utils")
    fake.FloatStorage = type("FloatStorage", (), {}); utils._rebuild_tensor_v2 = lambda storage, *args: storage
    fake._utils = utils; old = sys.modules.get("torch"); sys.modules["torch"] = fake; sys.modules["torch._utils"] = utils
    u = pickle.Unpickler(io.BytesIO(z.read("archive/data.pkl"))); u.persistent_load = lambda pid: pid[2]; mapping = u.load()
    if old is None: sys.modules.pop("torch", None)
    else: sys.modules["torch"] = old
    sys.modules.pop("torch._utils", None)
    rng = np.random.default_rng(SEED); projection = rng.normal(0, 1/np.sqrt(256), (5120, 256)).astype("float32")
    out = np.zeros((len(meta), 256), "float32"); hit = np.zeros(len(meta), bool)
    for i, symbol in enumerate(meta.symbol):
        key = mapping.get(str(symbol).upper())
        if key is not None:
            out[i] = np.frombuffer(z.read(f"archive/data/{key}"), "float32") @ projection; hit[i] = True
    return out, hit


def load_splits():
    data = {}
    for cv in (1, 2, 3):
        pos, neg = np.load(ROOT / f"data/feng2024/data/data_split/CV{cv}_1.npy", allow_pickle=True)
        for fold in range(5):
            for label, obj in (("pos", pos), ("neg", neg)):
                data[f"cv{cv}_{label}_train_{fold}"] = np.asarray(obj[2][fold], dtype="int16")
                data[f"cv{cv}_{label}_test_{fold}"] = np.asarray(obj[3][fold], dtype="int16")
    return data


def relation_pretrain_pool(ppi_pairs, random_pairs, benchmark_pairs):
    """Build a deterministic relation-training pool disjoint from all benchmark pairs."""
    benchmark_pairs = np.unique(np.sort(benchmark_pairs.astype("int32"), axis=1), axis=0)
    candidates = np.unique(np.sort(np.concatenate((ppi_pairs, random_pairs)).astype("int32"), axis=1), axis=0)
    benchmark_keys = benchmark_pairs[:, 0].astype("int64") * 1_000_003 + benchmark_pairs[:, 1]
    candidate_keys = candidates[:, 0].astype("int64") * 1_000_003 + candidates[:, 1]
    pool = candidates[~np.isin(candidate_keys, benchmark_keys)]
    pool_keys = pool[:, 0].astype("int64") * 1_000_003 + pool[:, 1]
    assert not np.isin(pool_keys, benchmark_keys).any(), "relation pretraining overlaps a benchmark pair"
    return pool


def benchmarks():
    data = {}
    for source, folder in (("full", "data_split"), ("wo_comp", "data_split_wo_comp")):
        for ratio in (1, 5, 20, 50):
            for sampling, suffix in (("random", ""), ("exp", "_Exp"), ("dep", "_Dep")):
                pos, neg = np.load(ROOT / f"data/feng2024/data/{folder}/CV3_{ratio}{suffix}.npy", allow_pickle=True)
                stem = f"{source}_{sampling}_{ratio}"
                for fold in range(5):
                    for label, obj in (("pos", pos), ("neg", neg)):
                        data[f"{stem}_{label}_train_{fold}"] = np.asarray(obj[2][fold], dtype="int16")
                        data[f"{stem}_{label}_test_{fold}"] = np.asarray(obj[3][fold], dtype="int16")
                print(stem, flush=True)
    np.savez_compressed(OUT / "cv3_benchmarks.npz", **data)


def graphs(k=45):
    from scipy import sparse
    mats=(("effect",csv_matrix(RAW/"DepMap_Effect_corr.csv")),("expression",csv_matrix(RAW/"DepMap_Expression_corr.csv")),*((x,np.load(RAW/f"final_gosim_{x}_from_r_9845.npy",mmap_mode="r")) for x in ("bp","cc","mf")))
    out={}
    for name,mat in mats:
        rows=[]; cols=[]; vals=[]
        for lo in range(0,len(mat),128):
            x=np.asarray(mat[lo:lo+128],dtype="float32").copy(); x[np.arange(len(x)),lo+np.arange(len(x))]=-np.inf; at=np.argpartition(x,-k,axis=1)[:,-k:]; v=np.take_along_axis(x,at,1); keep=v>0
            rows.append(np.broadcast_to(np.arange(lo,lo+len(x))[:,None],at.shape)[keep]); cols.append(at[keep]); vals.append(v[keep])
        a=sparse.csr_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=mat.shape); a=a.maximum(a.T)+sparse.eye(a.shape[0]); d=np.asarray(a.sum(1)).ravel()**-.5; a=sparse.diags(d)@a@sparse.diags(d); out[f"{name}_{k}"]=a.tocsr()
        print(name,a.nnz,flush=True)
    ppi=sparse.load_npz(RAW/"ppi_sparse_upper_matrix_without_sl_relation_9845.npz").tocsr(); ppi=ppi.maximum(ppi.T)+sparse.eye(ppi.shape[0]); d=np.asarray(ppi.sum(1)).ravel()**-.5; out["ppi"]=sparse.diags(d)@ppi@sparse.diags(d)
    arrays={}
    for name,a in out.items(): arrays.update({f"{name}_data":a.data.astype("float32"),f"{name}_indices":a.indices.astype("int32"),f"{name}_indptr":a.indptr.astype("int32")})
    np.savez_compressed(OUT/"relation_graphs.npz",n=np.int32(ppi.shape[0]),names=np.array(list(out)),**arrays)


def prepare(anchor_count=32, stem="features", geneformer_checkpoint="Geneformer-V2-104M"):
    import pandas as pd
    from scipy import sparse
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(RAW / "meta_table_9845.csv")
    gf, gf_hit = geneformer(meta, geneformer_checkpoint); esm, esm_hit = protein(meta)
    kg = np.load(RAW / "kg_TransE_l2_entity.npy", mmap_mode="r")[:len(meta)].astype("float32")
    ids = np.load(RAW / "ptgnn_data/ptgnn_encod_by_word_sl_9845_800.npy", mmap_mode="r")
    words = np.load(RAW / "ptgnn_data/trained_word_embedding.npy", mmap_mode="r")
    pt = np.empty((len(meta), 200), "float32")
    for lo in range(0, len(meta), 128):
        x = words[np.asarray(ids[lo:lo + 128])]
        pt[lo:lo + len(x)] = np.concatenate((x.mean(1), x.std(1)), 1)
    rng = np.random.default_rng(SEED); anchors = np.sort(rng.choice(len(meta), anchor_count, replace=False))
    effect = csv_matrix(RAW / "DepMap_Effect_corr.csv")
    expression = csv_matrix(RAW / "DepMap_Expression_corr.csv")
    gos = [np.load(RAW / f"final_gosim_{x}_from_r_9845.npy", mmap_mode="r") for x in ("bp", "cc", "mf")]
    ppi = sparse.load_npz(RAW / "ppi_sparse_upper_matrix_without_sl_relation_9845.npz").tocsr()
    state = np.concatenate((gf, esm, kg, pt, effect[:, anchors], expression[:, anchors],
                            *(g[:, anchors] for g in gos), ppi[:, anchors].toarray()), 1)
    state = np.nan_to_num(state); state = (state - state.mean(0)) / np.maximum(state.std(0), 1e-5)
    splits = load_splits()
    benchmark_pairs = np.unique(np.sort(np.concatenate(list(splits.values())).astype("int32"), axis=1), axis=0)
    upper = sparse.triu(ppi, 1).tocoo(); ppi_pairs = np.stack((upper.row, upper.col), 1)
    if len(ppi_pairs) > 150000: ppi_pairs = ppi_pairs[rng.choice(len(ppi_pairs), 150000, False)]
    random_pairs = rng.integers(0, len(meta), size=(250000, 2), dtype="int32")
    random_pairs.sort(1); random_pairs = random_pairs[random_pairs[:, 0] != random_pairs[:, 1]]
    # The relation task must not be sampled from benchmark membership.  Keep
    # benchmark pairs only for read-only evaluator features, and use PPI plus
    # independent random pairs to train the relation world model.
    pretrain_pairs = relation_pretrain_pool(ppi_pairs, random_pairs, benchmark_pairs)
    pairs = np.unique(np.concatenate((benchmark_pairs, pretrain_pairs)), axis=0)

    def relation_targets(pair_array):
        i, j = pair_array.T
        return np.stack((effect[i, j], expression[i, j], *(g[i, j] for g in gos),
                         np.asarray(ppi[i, j]).ravel()), 1).astype("float32")

    rel = relation_targets(pairs)
    pretrain_rel = relation_targets(pretrain_pairs)
    if stem != "features":
        old=np.load(OUT/"features.npz")
        pairs=old["pairs"]
        rel=old["relations"]
        pretrain_pairs=old["pretrain_pairs"] if "pretrain_pairs" in old.files else pairs
        pretrain_rel=old["pretrain_relations"] if "pretrain_relations" in old.files else rel
    np.savez_compressed(OUT / f"{stem}.npz", state=state.astype("float16"), pairs=pairs,
                        relations=rel.astype("float16"), pretrain_pairs=pretrain_pairs,
                        pretrain_relations=pretrain_rel.astype("float16"), anchors=anchors,
                        gf_hit=gf_hit, esm_hit=esm_hit)
    np.savez_compressed(OUT / "splits.npz", **splits)
    (OUT / f"{stem}.json").write_text(json.dumps({"genes": len(meta), "state_dim": state.shape[1],
        "relation_pairs": len(pairs), "pretrain_relation_pairs": len(pretrain_pairs),
        "geneformer_coverage": int(gf_hit.sum()), "protein_coverage": int(esm_hit.sum()),
        "geneformer_checkpoint": geneformer_checkpoint,
        "modalities": {"geneformer": 768, "protein": 256, "kg": 400, "ptgnn": 200, "anchors": 6*anchor_count}}, indent=2))


def prepare_v1(): prepare(256,"features_v1")


def prepare_cancer(): prepare(256,"features_cancer","Geneformer-V2-104M_CLcancer")


def prepare_spectral(k=32):
    from scipy import sparse
    from scipy.sparse.linalg import eigsh
    base=np.load(OUT/"features.npz"); graph=np.load(OUT/"relation_graphs.npz"); n=int(graph["n"]); views=[]
    for name in graph["names"]:
        a=sparse.csr_matrix((graph[f"{name}_data"],graph[f"{name}_indices"],graph[f"{name}_indptr"]),shape=(n,n)); values,vectors=eigsh(a,k=k,which="LA",v0=np.linspace(1,2,n)); order=np.argsort(values)[::-1]; z=vectors[:,order]*values[order]; views.append(z.astype("float32")); print(name,float(values[order[0]]),float(values[order[-1]]),flush=True)
    state=np.concatenate((base["state"][:,:1624].astype("float32"),*views),1); state=(state-state.mean(0))/np.maximum(state.std(0),1e-5); pretrain_pairs=base["pretrain_pairs"] if "pretrain_pairs" in base.files else base["pairs"]; pretrain_relations=base["pretrain_relations"] if "pretrain_relations" in base.files else base["relations"]; np.savez_compressed(OUT/"features_spectral.npz",state=state.astype("float16"),pairs=base["pairs"],relations=base["relations"],pretrain_pairs=pretrain_pairs,pretrain_relations=pretrain_relations,gf_hit=base["gf_hit"],esm_hit=base["esm_hit"]); (OUT/"features_spectral.json").write_text(json.dumps({"genes":n,"state_dim":state.shape[1],"relation_pairs":len(base["pairs"]),"pretrain_relation_pairs":len(pretrain_pairs),"spectral_dimensions_per_view":k,"views":graph["names"].tolist()},indent=2))


def prepare_spectral_safe():
    pack=np.load(OUT/"features_spectral.npz"); state=pack["state"].copy(); state[:,1024:1424]=0; pretrain_pairs=pack["pretrain_pairs"] if "pretrain_pairs" in pack.files else pack["pairs"]; pretrain_relations=pack["pretrain_relations"] if "pretrain_relations" in pack.files else pack["relations"]; np.savez_compressed(OUT/"features_spectral_safe.npz",state=state,pairs=pack["pairs"],relations=pack["relations"],pretrain_pairs=pretrain_pairs,pretrain_relations=pretrain_relations,gf_hit=pack["gf_hit"],esm_hit=pack["esm_hit"]); (OUT/"features_spectral_safe.json").write_text(json.dumps({"genes":len(state),"state_dim":state.shape[1],"excluded":"ambiguous frozen knowledge-graph embedding","retained":"Geneformer, State/ESM, PPI/GO PT-GNN, six non-SL spectral graph views"},indent=2))


def prepare_spectral_scgpt():
    import csv
    pack=np.load(OUT/"features_spectral_safe.npz"); state=pack["state"].copy(); symbols=[r["symbol"] for r in csv.DictReader(open(RAW/"meta_table_9845.csv"))]; emb=pickle.load(open(ROOT/"data/models/MuSL/processed_data/all_emb_scgpt.pkl","rb")); hit=np.asarray([g in emb for g in symbols]); block=np.zeros((len(state),400),"float32"); block[hit]=np.asarray([emb[g][:400] for g in np.asarray(symbols)[hit]]); mean=block[hit].mean(0); sd=block[hit].std(0); block[hit]=(block[hit]-mean)/(sd+1e-6); state[:,1024:1424]=block.astype(state.dtype); pretrain_pairs=pack["pretrain_pairs"] if "pretrain_pairs" in pack.files else pack["pairs"]; pretrain_relations=pack["pretrain_relations"] if "pretrain_relations" in pack.files else pack["relations"]; np.savez_compressed(OUT/"features_spectral_scgpt.npz",state=state,pairs=pack["pairs"],relations=pack["relations"],pretrain_pairs=pretrain_pairs,pretrain_relations=pretrain_relations,gf_hit=pack["gf_hit"],esm_hit=pack["esm_hit"],scgpt_hit=hit); (OUT/"features_spectral_scgpt.json").write_text(json.dumps({"genes":len(state),"scgpt_genes":int(hit.sum()),"state_dim":state.shape[1],"source":"MuSL all_emb_scgpt.pkl; first 400 coordinates standardized without SL labels","retained":"Geneformer, State/ESM, scGPT, PPI/GO PT-GNN, six non-SL spectral graph views"},indent=2))


def pair_features(state, relation, pairs):
    sections = ((0, 768), (768, 1024), (1024, 1424), (1424, 1624), (1624, state.shape[1]))
    a, b = state[pairs[:, 0]].astype("float32"), state[pairs[:, 1]].astype("float32")
    fs = []
    for lo, hi in sections:
        x, y = a[:, lo:hi], b[:, lo:hi]
        fs += [(x*y).mean(1), np.abs(x-y).mean(1), ((x-y)**2).mean(1),
               (x*y).sum(1)/(np.linalg.norm(x, axis=1)*np.linalg.norm(y, axis=1)+1e-6)]
    keys = pairs[:, 0].astype("int64") * state.shape[0] + pairs[:, 1]
    rel_keys, rel_values = relation; at = np.searchsorted(rel_keys, keys)
    if np.any(at == len(rel_keys)) or np.any(rel_keys[at] != keys): raise KeyError("missing relation pair")
    return np.column_stack((*fs, rel_values[at]))


def baseline():
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
    OUT.mkdir(parents=True, exist_ok=True)
    pack = np.load(OUT / "features.npz"); state = pack["state"]
    keys = pack["pairs"][:, 0].astype("int64") * len(state) + pack["pairs"][:, 1]
    order = np.argsort(keys); relation = (keys[order], pack["relations"].astype("float32")[order])
    split = np.load(OUT / "splits.npz"); rows = []
    for cv in (1, 2, 3):
        for fold in range(5):
            trp, trn = split[f"cv{cv}_pos_train_{fold}"], split[f"cv{cv}_neg_train_{fold}"]
            tep, ten = split[f"cv{cv}_pos_test_{fold}"], split[f"cv{cv}_neg_test_{fold}"]
            train = np.concatenate((trp, trn)).astype("int32"); test = np.concatenate((tep, ten)).astype("int32")
            ytr = np.r_[np.ones(len(trp)), np.zeros(len(trn))]; y = np.r_[np.ones(len(tep)), np.zeros(len(ten))]
            model = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=4, max_features=None,
                                         n_jobs=1, class_weight="balanced", random_state=SEED)
            model.fit(pair_features(state, relation, train), ytr)
            score = model.predict_proba(pair_features(state, relation, test))[:, 1]
            rows.append((cv, fold, roc_auc_score(y, score), average_precision_score(y, score),
                         f1_score(y, score >= .5)))
            print(rows[-1], flush=True)
    import pandas as pd
    out = pd.DataFrame(rows, columns="cv fold auroc aupr f1".split())
    out.to_csv(OUT / "baseline.tsv", sep="\t", index=False)
    print(out.groupby("cv")[["auroc", "aupr", "f1"]].agg(["mean", "std"]))


def spectral_ensemble(spectral="native_spectral_guard", suffix="spectral"):
    from sklearn.metrics import auc, average_precision_score, f1_score, precision_recall_curve, roc_auc_score
    names=("native_base","native_rel256_guard",spectral); packs=[np.load(OUT/n/"tabular_metrics.npz") for n in names]; rows=[]; saved={}
    for fold in range(5):
        pairs=packs[0][f"fold{fold}_pairs"]; y=packs[0][f"fold{fold}_label"]
        if any(not np.array_equal(pairs,p[f"fold{fold}_pairs"]) or not np.array_equal(y,p[f"fold{fold}_label"]) for p in packs[1:]): raise ValueError("fold mismatch")
        score=np.mean([p[f"fold{fold}_score"] for p in packs],0); precision,recall,_=precision_recall_curve(y,score); rows.append({"fold":fold,"auroc":roc_auc_score(y,score),"average_precision":average_precision_score(y,score),"pr_auc":auc(recall,precision),"f1_at_0.5":f1_score(y,score>=.5),"f1_max":np.nanmax(2*precision*recall/(precision+recall+1e-12))}); saved.update({f"fold{fold}_pairs":pairs,f"fold{fold}_label":y,f"fold{fold}_score":score.astype("float32")})
    mean={k:float(np.mean([r[k] for r in rows])) for k in ("auroc","average_precision","pr_auc","f1_at_0.5","f1_max")}; (OUT/f"representation_ensemble_{suffix}.json").write_text(json.dumps({"components":names,"folds":rows,"mean":mean},indent=2)); np.savez_compressed(OUT/f"representation_ensemble_{suffix}.npz",**saved); print(mean)


def spectral_safe_ensemble(): spectral_ensemble("native_spectral_safe_external_guard","spectral_safe")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("command", choices=("prepare", "prepare_v1", "prepare_cancer", "prepare_spectral", "prepare_spectral_safe", "prepare_spectral_scgpt", "spectral_ensemble", "spectral_safe_ensemble", "baseline", "benchmarks", "graphs")); a = p.parse_args()
    globals()[a.command]()
