import csv, hashlib, json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from tahoe_transition_source import ROOT, SE, StateEncoder, load_se

OUT=ROOT/"results/sl_predict/se_in_silico_deletion.json"

def order(x): return hashlib.sha256(str(x).encode()).hexdigest()

def encode_inputs(values,symbols,vocab):
    index={g:i for i,g in enumerate(vocab)}; keep=[i for i,g in enumerate(symbols) if g in index]
    values=values[:,keep]; ids=np.asarray([index[symbols[i]] for i in keep]); top=np.argsort(-values,1,kind="stable")[:,:2047]
    tokens=np.concatenate((np.full((len(values),1),ids[3]),ids[top]),1)
    weights=values/values.sum(1,keepdims=True)
    counts=np.concatenate((weights[:,3:4],np.take_along_axis(weights,top,1)),1)*100
    return tokens,counts,len(keep)

def main():
    torch.manual_seed(731); torch.set_grad_enabled(False); device=torch.device("cuda"); dtype=torch.bfloat16
    meta=list(csv.DictReader(open(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv")))
    ens={r["ensembl_gene_id"].split(".")[0]:r["symbol"].upper() for r in meta if r["ensembl_gene_id"]}
    vocab=list(torch.load(SE/"protein_embeddings.pt",map_location="cpu",weights_only=True)); records=[]; audits=[]
    for sid,name in enumerate(("replogle2022_k562","replogle2022_rpe1")):
        z=np.load(ROOT/f"data/perturbseq_sources/base/{name}.npz"); raw_symbols=[ens.get(str(x).split(".")[0],str(x).upper()) for x in z["feature_name"]]
        first={}; [first.setdefault(g,i) for i,g in enumerate(raw_symbols)]; symbols=list(first); cols=np.asarray([first[g] for g in symbols])
        control=z["control_mean"][cols].astype("float32"); a=np.char.upper(z["endpoint_a"].astype(str)); test=(z["cardinality"]==1)&(z["role"].astype(str)=="intrinsic_test")
        genes=sorted(set(a[test])&set(symbols)&set(vocab),key=order); fmap={g:i for i,g in enumerate(symbols)}
        deleted=[]; measured=[]
        for gene in genes:
            x=control.copy(); x[fmap[gene]]=0; deleted.append(x)
            measured.append(z["future_state"][(a==gene)&test][:,cols].mean(0))
        values=np.vstack((control,deleted,measured)).astype("float32"); tokens,counts,overlap=encode_inputs(values,symbols,vocab)
        records.append((sid,name,genes,tokens,counts,overlap))
    model=StateEncoder(); load_se(model); model.to(device,dtype=dtype).eval(); all_rows=[]
    for sid,name,genes,tokens,counts,overlap in records:
        encoded=[]
        for at in range(0,len(tokens),2):
            encoded.append(model(torch.as_tensor(tokens[at:at+2],device=device),torch.as_tensor(counts[at:at+2],device=device,dtype=dtype)).float().cpu())
        encoded=torch.cat(encoded); n=len(genes); pred=encoded[1:1+n]-encoded[:1]; measured=encoded[1+n:]-encoded[:1]
        cosine=torch.nn.functional.cosine_similarity(pred,measured).numpy(); pn=pred.norm(dim=1).numpy(); mn=measured.norm(dim=1).numpy()
        shuffled=measured[np.roll(np.arange(n),1)]; shuffle_cos=torch.nn.functional.cosine_similarity(pred,shuffled).numpy()
        audit={"source":name,"genes":n,"expression_genes_in_se":overlap,"nonzero_predicted_actions":int((pn>0).sum()),
               "mean_cosine":float(cosine.mean()),"median_cosine":float(np.median(cosine)),"shuffled_mean_cosine":float(shuffle_cos.mean()),
               "effect_magnitude_spearman":float(spearmanr(pn,mn).statistic)}; audits.append(audit)
        all_rows.append((np.asarray(genes,dtype="U"),np.full(n,sid,"int8"),pred.numpy().astype("float16"),measured.numpy().astype("float16")))
    macro=float(np.mean([x["mean_cosine"] for x in audits])); shuffled=float(np.mean([x["shuffled_mean_cosine"] for x in audits]))
    result={"sources":audits,"source_macro_mean_cosine":macro,"source_macro_shuffled_cosine":shuffled,"margin_over_shuffled":macro-shuffled,
            "admitted":bool(all(x["mean_cosine"]>0 for x in audits) and macro>=.1 and macro-shuffled>=.05),
            "double_perturbation_data_used":False,"sl_labels_used":False}
    OUT.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    if result["admitted"]: np.savez_compressed(OUT.with_suffix(".npz"),gene=np.concatenate([x[0] for x in all_rows]),source=np.concatenate([x[1] for x in all_rows]),predicted=np.concatenate([x[2] for x in all_rows]),measured=np.concatenate([x[3] for x in all_rows]))

if __name__=="__main__": main()
