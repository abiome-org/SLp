import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def summarize(data, rng):
    pair=data.groupby("sorted_gene_pair",sort=False).agg(score=("score","mean"),gi=("mean_norm_gi",lambda x:-x.mean()),hit=("is_bassik_hit","max"))
    rho=float(spearmanr(pair.score,pair.gi).statistic); n=len(pair); pair_boot=[]
    for _ in range(2000):
        ix=rng.integers(0,n,n); pair_boot.append(spearmanr(pair.score.iloc[ix],pair.gi.iloc[ix]).statistic)
    pair_null=np.asarray([spearmanr(rng.permutation(pair.score),pair.gi).statistic for _ in range(10000)])
    cells=[]
    for cell,d in data.groupby("cell_line_label"):
        y=d.is_bassik_hit.to_numpy("int8")
        if np.unique(y).size==2: cells.append({"cell_line":cell,"pairs":len(d),"positives":int(y.sum()),"auroc":float(roc_auc_score(y,d.score)),"average_precision":float(average_precision_score(y,d.score))})
    y=data.is_bassik_hit.to_numpy("int8"); py=pair.hit.to_numpy("int8")
    observed=float(np.mean([x["auroc"] for x in cells])); crng=np.random.default_rng(20260826); auc=np.asarray([x["auroc"] for x in cells]); cell_boot=auc[crng.integers(0,len(auc),(10000,len(auc)))].mean(1); cell_null=np.zeros(9999)
    supported=0
    for _,d in data.groupby("cell_line_label"):
        ycell=d.is_bassik_hit.to_numpy("int8"); m=int(ycell.sum()); ranks=rankdata(d.score.to_numpy())
        if not 0<m<len(ranks):continue
        chosen=np.argpartition(crng.random((len(cell_null),len(ranks))),m-1,axis=1)[:,:m]; cell_null+=(ranks[chosen].sum(1)-m*(m+1)/2)/(m*(len(ranks)-m)); supported+=1
    cell_null/=supported; secondary={"status":"exploratory_post_result_inference","cell_line_bootstrap_95ci":np.quantile(cell_boot,[.025,.975]).tolist(),"within_cell_permutations":len(cell_null),"one_sided_p":float((1+(cell_null>=observed).sum())/(len(cell_null)+1)),"null_mean":float(cell_null.mean())}
    return {"measurements":len(data),"pairs":n,"positives":int(y.sum()),"primary_spearman":rho,
            "primary_spearman_bootstrap_95ci":[float(x) for x in np.nanquantile(pair_boot,[.025,.975])],
            "primary_permutation_p_two_sided":float((1+(np.abs(pair_null)>=abs(rho)).sum())/(len(pair_null)+1)),
            "cell_line_macro":{"eligible_cell_lines":len(cells),"auroc":float(np.mean([x["auroc"] for x in cells])),"average_precision":float(np.mean([x["average_precision"] for x in cells]))},
            "pooled":{"auroc":float(roc_auc_score(y,data.score)),"average_precision":float(average_precision_score(y,data.score))},
            "pair_any_hit":{"positives":int(py.sum()),"auroc":float(roc_auc_score(py,pair.score)),"average_precision":float(average_precision_score(py,pair.score))},"cell_line_macro_inference":secondary,"cell_lines":cells}


@torch.no_grad()
def run(args):
    sys.path[:0]=[str(Path(__file__).parents[1]/"training"),str(Path(__file__).parent)]
    from musl import score
    from world_model import SLPredict, ResidualInteraction, encode_genes, interaction_head, load_residual_endpoint, residual_interaction_inputs
    device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(args.features)["state"].astype("float32")
    meta=list(csv.DictReader(open(args.meta))); ids={r["symbol"].upper():i for i,r in enumerate(meta)}
    sd=torch.load(args.model,map_location="cpu",weights_only=True); context_dim=sd["context_proj.weight"].shape[1] if "context_proj.weight" in sd else 0; model=SLPredict(args.d,args.latent,args.layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],context_dim).to(device); model.load_state_dict(sd); model.eval()
    ds=torch.load(args.decoder,map_location="cpu",weights_only=True); decoder=torch.nn.Linear(args.latent,ds["weight"].shape[0]).to(device); decoder.load_state_dict(ds); decoder.eval(); interaction=None
    if args.interaction_head:interaction=interaction_head(args.latent).to(device); interaction.load_state_dict(torch.load(args.interaction_head,map_location="cpu",weights_only=True)); interaction.eval()
    genes=torch.as_tensor(encode_genes(model,state,device),device=device); ensemble=None; residual=None
    if getattr(args,"ensemble_model",None):
        es=torch.load(args.ensemble_model,map_location="cpu",weights_only=True); em=SLPredict(args.d,args.latent,args.layers,es["cell.weight"].shape[0],es["outcome.weight"].shape[0],state.shape[1],es["context_proj.weight"].shape[1]).to(device); em.load_state_dict(es); em.eval(); eg=torch.as_tensor(encode_genes(em,state,device),device=device); eh=interaction_head(args.latent).to(device); eh.load_state_dict(torch.load(args.ensemble_interaction_head,map_location="cpu",weights_only=True)); eh.eval(); ensemble=(em,eh,eg)
    if getattr(args,"residual_model",None):
        endpoint=load_residual_endpoint(args.residual_model,state.shape[1],device,args.d,args.latent,args.layers); rg=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); rh=ResidualInteraction(interaction_head(args.latent).to(device)).to(device); rh.load_state_dict(torch.load(args.residual_interaction_head,map_location="cpu",weights_only=True)); rh.eval(); residual=(endpoint,rh,rg)
    data=pd.read_csv(args.data,sep="\t"); mapped=data.targetA.astype(str).str.upper().isin(ids)&data.targetB.astype(str).str.upper().isin(ids)
    data=data[mapped&np.isfinite(data.mean_norm_gi)&data.targetA__is_single_depleted.eq(0)&data.targetB__is_single_depleted.eq(0)].copy()
    context_audit=None
    if args.context_pack:
        z=np.load(args.context_pack); states=dict(zip(z["model_ids"].astype(str),z["cell_state"])); known=data.depMapID.astype(str).isin(states); context_audit={"eligible_cell_lines":int(data.cell_line_label.nunique()),"known_cell_lines":int(data.loc[known,"cell_line_label"].nunique()),"excluded_cell_lines":sorted(data.loc[~known,"cell_line_label"].unique())}; data=data[known].copy(); data["score"]=np.nan
        for dep,x in data.groupby("depMapID",sort=True):
            unique=x[["sorted_gene_pair","targetA","targetB"]].drop_duplicates("sorted_gene_pair"); pairs=np.asarray([(ids[a.upper()],ids[b.upper()]) for a,b in unique[["targetA","targetB"]].itertuples(index=False)],"int64")
            if interaction is None:_,fixed=score(model,decoder,genes,pairs,device,np.asarray([states[str(dep)]],"float32"))
            else:
                p=torch.as_tensor(pairs,device=device); cs=torch.as_tensor(states[str(dep)],dtype=torch.float32,device=device).expand(len(p),-1); fixed=-interaction(model.transition(genes[p[:,0]],genes[p[:,1]],context_state=cs)[0])[:,0]
                if ensemble is not None:
                    em,eh,eg=ensemble; second=-eh(em.transition(eg[p[:,0]],eg[p[:,1]],context_state=cs)[0])[:,0]; fixed=(1-args.ensemble_weight)*fixed+args.ensemble_weight*second
                if residual is not None:
                    endpoint,rh,rg=residual; rz,rr=residual_interaction_inputs(endpoint,rg,p,cs); fixed=-rh(rz,rr)[:,0]
                fixed=fixed.cpu().numpy()
            scores=dict(zip(unique.sorted_gene_pair,fixed)); data.loc[x.index,"score"]=x.sorted_gene_pair.map(scores)
    else:
        unique=data[["sorted_gene_pair","targetA","targetB"]].drop_duplicates("sorted_gene_pair"); pairs=np.asarray([(ids[a.upper()],ids[b.upper()]) for a,b in unique[["targetA","targetB"]].itertuples(index=False)],"int64"); _,fixed=score(model,decoder,genes,pairs,device); scores=dict(zip(unique.sorted_gene_pair,fixed)); data["score"]=data.sorted_gene_pair.map(scores)
    unique=data[["sorted_gene_pair","targetA","targetB"]].drop_duplicates("sorted_gene_pair"); pairs=np.asarray([(ids[a.upper()],ids[b.upper()]) for a,b in unique[["targetA","targetB"]].itertuples(index=False)],"int64")
    perturb=np.load(args.perturb); seen=set(perturb["pairs"][perturb["role"]==0].ravel()); seen.discard(-1)
    if args.interaction_training:
        z=np.load(args.interaction_training,allow_pickle=True); q=z["pairs"].astype("int64"); held=np.arange(q.max()+1)%5==0; rows=z["context_known"][z["context"]]&(~held[q]).all(1); seen.update(q[rows].ravel())
    cold={p for p,(a,b) in zip(unique.sorted_gene_pair,pairs) if a not in seen and b not in seen}
    rng=np.random.default_rng(1234); result={"protocol":json.loads(Path(args.protocol).read_text()),"device":device,"context_coverage":context_audit,"eligible_before_intervention_filter":{"measurements":len(data),"pairs":data.sorted_gene_pair.nunique(),"cell_lines":data.cell_line_label.nunique()},"intervention_cold":summarize(data[data.sorted_gene_pair.isin(cold)],rng),"all_mapped":summarize(data,rng)}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k not in ("protocol",)},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--model",required=True); p.add_argument("--decoder",required=True); p.add_argument("--interaction-head"); p.add_argument("--ensemble-model"); p.add_argument("--ensemble-interaction-head"); p.add_argument("--ensemble-weight",type=float,default=.75); p.add_argument("--residual-model"); p.add_argument("--residual-interaction-head"); p.add_argument("--interaction-training"); p.add_argument("--data",default="data/sanger2025/DATA/postprocessing/combined_gene_level_results.tsv"); p.add_argument("--features",default="results/sl_predict/features_spectral_safe.npz"); p.add_argument("--meta",default="data/feng2024/data/preprocessed_data/meta_table_9845.csv"); p.add_argument("--perturb",default="results/sl_predict/perturbseq_world.npz"); p.add_argument("--context-pack"); p.add_argument("--protocol",default="results/sl_predict/sanger2025_protocol.json"); p.add_argument("--output",default="results/sl_predict/sanger2025_confirmatory.json"); p.add_argument("--d",type=int,default=384); p.add_argument("--latent",type=int,default=128); p.add_argument("--layers",type=int,default=6); run(p.parse_args())
