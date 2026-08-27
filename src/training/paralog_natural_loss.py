import csv, hashlib, json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from natural_loss import aligned, residualize

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest().upper()
def estimate(loss,known,effect,lineage,subset,source,target):
    state=residualize(effect,lineage,subset); out=np.full(len(source),np.nan,"float32")
    for gene in np.unique(source):
        low=subset&known[:,gene]&loss[:,gene]; wt=subset&known[:,gene]&~loss[:,gene]
        if low.sum()<3 or wt.sum()<20: continue
        rows=np.flatnonzero(source==gene); partners=target[rows]; out[rows]=np.nanmean(state[low][:,partners],0)-np.nanmean(state[wt][:,partners],0)
    return out

def metrics(a,b):
    keep=np.isfinite(a)&np.isfinite(b); x=a[keep]; y=b[keep]
    return {"shared_directions":int(keep.sum()),"pearson":float(np.corrcoef(x,y)[0,1]),"spearman":float(spearmanr(x,y).statistic),"sign_agreement":float(np.mean(np.signbit(x)==np.signbit(y)))},keep

def controls(source,target,pairs,seed):
    partner=target[np.random.default_rng(seed).permutation(len(target))]; keep=np.asarray([a!=b and (min(a,b),max(a,b)) not in pairs for a,b in zip(source,partner)])
    return source[keep],partner[keep]

def main():
    meta=list(csv.DictReader(open(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv",encoding="utf-8"))); symbols=np.asarray([r["symbol"].upper() for r in meta]); basal=np.load(OUT/"basal_context.npz"); all_model=basal["model_ids"].astype(str); effect=basal["dependency"].astype("float32"); effect[~basal["dependency_known"]]=np.nan
    cn_id,cn,_=aligned(ROOT/"data/depmap24q2/OmicsAbsoluteCNGene.csv",symbols); mut_id,mut,_=aligned(ROOT/"data/depmap24q2/OmicsSomaticMutationsMatrixDamaging.csv",symbols); cr={x:i for i,x in enumerate(cn_id)}; mr={x:i for i,x in enumerate(mut_id)}; keep=np.asarray([x in cr and x in mr for x in all_model]); model=all_model[keep]; effect=effect[keep]; cn=cn[[cr[x] for x in model]]; mut=mut[[mr[x] for x in model]]; known=np.isfinite(cn)&np.isfinite(mut); loss=(cn<.5)|(mut>.5)
    lineage_map={r["ModelID"]:r["OncotreeLineage"] or "unknown" for r in csv.DictReader(open(ROOT/"data/depmap24q2/Model.csv"))}; lineage=np.asarray([lineage_map.get(x,"unknown") for x in model]); half=np.asarray([hashlib.sha256(x.encode()).digest()[0]&1 for x in model]); rel=np.load(OUT/"ensembl_paralogs.npz"); i=rel["i"].astype("int64"); j=rel["j"].astype("int64"); source=np.r_[i,j]; target=np.r_[j,i]; pairs=set(zip(i.tolist(),j.tolist()))
    h=[estimate(loss,known,effect,lineage,half==k,source,target) for k in range(2)]; candidate,valid=metrics(*h); control={}
    for seed in (821,822):
        a,b=controls(source,target,pairs,seed); x=[estimate(loss,known,effect,lineage,half==k,a,b) for k in range(2)]; control[str(seed)],_=metrics(*x)
    limits=json.loads((OUT/"paralog_natural_loss_protocol.json").read_text()); admit=limits["admission"]; maxp=max(x["pearson"] for x in control.values()); maxs=max(x["spearman"] for x in control.values()); advantage={"pearson":candidate["pearson"]-maxp,"spearman":candidate["spearman"]-maxs}
    admitted=candidate["shared_directions"]>=limits["support"]["minimum_shared_directions"] and candidate["pearson"]>=admit["minimum_split_half_pearson"] and candidate["spearman"]>=admit["minimum_split_half_spearman"] and candidate["sign_agreement"]>=admit["minimum_split_half_sign_agreement"] and advantage["pearson"]>=admit["minimum_pearson_advantage_over_each_control"] and advantage["spearman"]>=admit["minimum_spearman_advantage_over_each_control"]
    np.savez_compressed(OUT/"paralog_natural_loss.npz",loss_gene=source[valid].astype("int16"),dependency_gene=target[valid].astype("int16"),delta=((h[0][valid]+h[1][valid])/2).astype("float16"))
    result={"schema":"sl-predict-paralog-natural-loss-v1","aligned_models":len(model),"candidate_directions":len(source),"candidate":candidate,"controls":control,"advantage_over_stronger_control":advantage,"admitted":bool(admitted),"double_perturbation_data_used":False,"sl_labels_used":False,"benchmark_pairs_used":False,"protocol_sha256":sha(OUT/"paralog_natural_loss_protocol.json"),"ensembl_relation_sha256":sha(OUT/"ensembl_paralogs.npz")}; (OUT/"paralog_natural_loss.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
