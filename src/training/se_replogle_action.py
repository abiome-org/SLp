import csv, json, math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from se_in_silico_deletion import encode_inputs
from tahoe_transition_source import ROOT, SE, StateEncoder, load_se

OUT=ROOT/"results/sl_predict/se_replogle_action.json"

def huber(x):
    a=np.abs(x); return float(np.where(a<1,.5*a*a,a-.5).mean())

def main():
    torch.manual_seed(731); torch.set_grad_enabled(False); device=torch.device("cuda"); dtype=torch.bfloat16
    meta=list(csv.DictReader(open(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv")))
    ens={r["ensembl_gene_id"].split(".")[0]:r["symbol"].upper() for r in meta if r["ensembl_gene_id"]}
    vocab=list(torch.load(SE/"protein_embeddings.pt",map_location="cpu",weights_only=True)); vid={g:i for i,g in enumerate(vocab)}
    model=StateEncoder(); load_se(model); model.to(device,dtype=dtype).eval(); sources=[]; corpus=[]
    for sid,name in enumerate(("replogle2022_k562","replogle2022_rpe1")):
        z=np.load(ROOT/f"data/perturbseq_sources/base/{name}.npz"); raw_symbols=[ens.get(str(x).split(".")[0],str(x).upper()) for x in z["feature_name"]]
        first={}; [first.setdefault(g,i) for i,g in enumerate(raw_symbols)]; symbols=list(first); cols=np.asarray([first[g] for g in symbols]); fmap={g:i for i,g in enumerate(symbols)}
        a=np.char.upper(z["endpoint_a"].astype(str)); roles=z["role"].astype(str); cardinality=z["cardinality"]; future=z["future_state"][:,cols]; genes=[]; split=[]; values=[]
        for gene in sorted(set(a[(cardinality==1)&np.isin(roles,("train","intrinsic_test","intrinsic_validation"))])):
            if gene not in vid: continue
            rows=(a==gene)&(cardinality==1); role=set(roles[rows]); role="train" if "train" in role else ("intrinsic_test" if "intrinsic_test" in role else "intrinsic_validation")
            genes.append(gene); split.append(role); values.append(future[rows].mean(0))
        values=np.vstack((z["control_mean"][cols],values)).astype("float32"); tokens,counts,overlap=encode_inputs(values,symbols,vocab); encoded=[]
        for at in range(0,len(tokens),4):
            encoded.append(model(torch.as_tensor(tokens[at:at+4],device=device),torch.as_tensor(counts[at:at+4],device=device,dtype=dtype)).float().cpu())
        response=(torch.cat(encoded)[1:]-torch.cat(encoded)[:1]).numpy(); token_ids=torch.as_tensor([vid[g] for g in genes],device=device)
        gene_token=(model.encoder(torch.nn.functional.normalize(model.pe(token_ids),dim=1))*math.sqrt(2048)).float().cpu().numpy()
        expression=np.asarray([z["control_mean"][cols[fmap[g]]] if g in fmap else 0 for g in genes],"float32")[:,None]
        x=np.column_stack((gene_token,expression)); train=np.asarray(split)=="train"; test=np.asarray(split)=="intrinsic_test"
        pca=PCA(n_components=32,svd_solver="randomized",random_state=731).fit(response[train]); y=pca.transform(response)
        scaler=StandardScaler().fit(x[train]); ridge=Ridge(alpha=10).fit(scaler.transform(x[train]),y[train]); pred=pca.inverse_transform(ridge.predict(scaler.transform(x[test]))); truth=response[test]
        cosine=np.sum(pred*truth,1)/(np.linalg.norm(pred,axis=1)*np.linalg.norm(truth,axis=1)+1e-8); zero=huber(truth); loss=huber(pred-truth)
        row={"source":name,"expression_genes_in_se":overlap,"train_genes":int(train.sum()),"test_genes":int(test.sum()),"mean_cosine":float(cosine.mean()),
             "median_cosine":float(np.median(cosine)),"zero_huber":zero,"huber":loss,"huber_improvement":1-loss/zero,
             "effect_magnitude_spearman":float(spearmanr(np.linalg.norm(pred,axis=1),np.linalg.norm(truth,axis=1)).statistic)}; sources.append(row)
        corpus.append((np.asarray(genes,dtype="U"),np.full(len(genes),sid,"int8"),np.asarray(split,dtype="U"),x.astype("float16"),response.astype("float16")))
    improvement=float(np.mean([x["huber_improvement"] for x in sources])); result={"sources":sources,"source_macro_cosine":float(np.mean([x["mean_cosine"] for x in sources])),
      "source_macro_huber_improvement":improvement,"admitted":bool(all(x["mean_cosine"]>=.1 and x["huber_improvement"]>0 for x in sources) and improvement>=.05),
      "double_perturbation_data_used":False,"sl_labels_used":False}
    OUT.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    np.savez_compressed(OUT.with_name("se_replogle_state.npz"),gene=np.concatenate([x[0] for x in corpus]),source=np.concatenate([x[1] for x in corpus]),role=np.concatenate([x[2] for x in corpus]),features=np.concatenate([x[3] for x in corpus]),response=np.concatenate([x[4] for x in corpus]))

if __name__=="__main__": main()
