import argparse, csv, hashlib, json, pickle, sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def metrics(y, score):
    k=int(y.sum()); pred=np.zeros(len(y),"int8"); pred[np.argsort(score)[-k:]]=1
    return {"auroc":float(roc_auc_score(y,score)),"aupr":float(average_precision_score(y,score)),"f1_at_prevalence":float(2*(pred*y).sum()/(pred.sum()+y.sum()))}


@torch.no_grad()
def score(model, decoder, genes, pairs, device, context_states=None,context_summaries=False):
    unknown=[]; source=[]; top3=[]; maximum=[]; stop=32 if decoder.out_features==33 else decoder.out_features
    for at in range(0,len(pairs),2048):
        p=torch.as_tensor(pairs[at:at+2048],device=device); a,b=genes[p[:,0]],genes[p[:,1]]; joint=decoder(model.transition(a,b)[0]); la=model.transition(a)[0]; lb=model.transition(b)[0]; ab=decoder(model.transition(b,state=la)[0]); ba=decoder(model.transition(a,state=lb)[0]); unknown.append(torch.linalg.vector_norm(joint[:,:stop]-(ab[:,:stop]+ba[:,:stop])/2,dim=1).cpu())
        contextual=[]
        for c in range(4 if context_states is None else len(context_states)):
            ctx=torch.full((len(p),),c,device=device) if context_states is None else None; cs=None if context_states is None else torch.as_tensor(context_states[c],device=device).expand(len(p),-1); joint=decoder(model.transition(a,b,context=ctx,context_state=cs)[0]); la=model.transition(a,context=ctx,context_state=cs)[0]; lb=model.transition(b,context=ctx,context_state=cs)[0]; ab=decoder(model.transition(b,state=la,context=ctx,context_state=cs)[0]); ba=decoder(model.transition(a,state=lb,context=ctx,context_state=cs)[0]); contextual.append(torch.linalg.vector_norm(joint[:,:stop]-(ab[:,:stop]+ba[:,:stop])/2,dim=1))
        contextual=torch.stack(contextual); source.append(contextual.mean(0).cpu())
        if context_summaries: top3.append(contextual.topk(min(3,len(contextual)),0).values.mean(0).cpu()); maximum.append(contextual.max(0).values.cpu())
    if context_summaries:return torch.cat(unknown).numpy(),{"basal_mean_sequential":torch.cat(source).numpy(),"basal_top3_sequential":torch.cat(top3).numpy(),"basal_max_sequential":torch.cat(maximum).numpy()}
    return torch.cat(unknown).numpy(),torch.cat(source).numpy()


@torch.no_grad()
def tolerated_score(model,decoder,tolerance,genes,pairs,context_states,device):
    out={"tolerated_mean_sequential":[],"tolerated_top3_sequential":[],"tolerated_max_sequential":[]}
    for at in range(0,len(pairs),2048):
        p=torch.as_tensor(pairs[at:at+2048],device=device); a,b=genes[p[:,0]],genes[p[:,1]]; contextual=[]
        for state in context_states:
            cs=torch.as_tensor(state,device=device).expand(len(p),-1); joint=decoder(model.transition(a,b,context_state=cs)[0]); la=model.transition(a,context_state=cs)[0]; lb=model.transition(b,context_state=cs)[0]; ab=decoder(model.transition(b,state=la,context_state=cs)[0]); ba=decoder(model.transition(a,state=lb,context_state=cs)[0]); residual=torch.linalg.vector_norm(joint-(ab+ba)/2,dim=1); gate=torch.sigmoid(8*(tolerance(la).squeeze(1)+.5))*torch.sigmoid(8*(tolerance(lb).squeeze(1)+.5)); contextual.append(residual*gate)
        contextual=torch.stack(contextual); out["tolerated_mean_sequential"].append(contextual.mean(0).cpu()); out["tolerated_top3_sequential"].append(contextual.topk(min(3,len(contextual)),0).values.mean(0).cpu()); out["tolerated_max_sequential"].append(contextual.max(0).values.cpu())
    return {k:torch.cat(v).numpy() for k,v in out.items()}


@torch.no_grad()
def interaction_score(model,head,genes,pairs,context_states,device):
    out={f"interaction_{target}_{summary}":[] for target in ("depletion","magnitude") for summary in ("mean","top3","max")}
    for at in range(0,len(pairs),2048):
        p=torch.as_tensor(pairs[at:at+2048],device=device); a,b=genes[p[:,0]],genes[p[:,1]]; contextual=[]
        for state in context_states:
            cs=torch.as_tensor(state,device=device).expand(len(p),-1); contextual.append(head(model.transition(a,b,context_state=cs)[0]))
        contextual=torch.stack(contextual)
        values=(-contextual[:,:,0],contextual[:,:,1])
        for name,value in zip(("depletion","magnitude"),values):
            out[f"interaction_{name}_mean"].append(value.mean(0).cpu()); out[f"interaction_{name}_top3"].append(value.topk(min(3,len(value)),0).values.mean(0).cpu()); out[f"interaction_{name}_max"].append(value.max(0).values.cpu())
    return {k:torch.cat(v).numpy() for k,v in out.items()}


def rank01(x):
    return (rankdata(x,method="average")-1)/max(1,len(x)-1)


def public_relation_cv3(codependency_path,tcga_path,meta_path,musl_meta,out_path,seeds=(42,432),codependency_weight=.7404387016238704):
    codep=np.load(codependency_path); tcga=np.load(tcga_path); codep_pos=np.full(9845,-1,"int32"); codep_pos[codep["genes"].astype("int64")]=np.arange(len(codep["genes"])); tcga_pos=np.full(9845,-1,"int32"); tcga_pos[tcga["genes"].astype("int64")]=np.arange(len(tcga["genes"])); n=len(tcga["genes"]); ids={r["symbol"]:i for i,r in enumerate(csv.DictReader(open(meta_path)))}; remap=np.asarray([ids[r["symbol"]] for r in csv.DictReader(open(musl_meta))],"int64"); root=Path(musl_meta).parent; root=root if (root/"test_pairs_seed42.pkl").exists() else root/"data"/"CV3_bins_32"/"fold_data"; rows=[]
    def values(pair,pos,half0,half1,triangular=False):
        a,b=pos[pair[:,0]],pos[pair[:,1]]; known=(a>=0)&(b>=0)&(a!=b); out=np.full(len(pair),np.nan,"float32")
        if triangular:
            lo=np.minimum(a[known],b[known]).astype("int64"); hi=np.maximum(a[known],b[known]).astype("int64"); ix=lo*(2*n-lo-1)//2+hi-lo-1; out[known]=(half0[ix]+half1[ix])/2
        else:out[known]=(half0[a[known],b[known]].astype("float32")+half1[a[known],b[known]].astype("float32"))/2
        out[~known]=np.nanmedian(out); return out,known
    for seed in seeds:
        pairs=pickle.load(open(root/f"test_pairs_seed{seed}.pkl","rb")); labels=pickle.load(open(root/f"test_labels_seed{seed}.pkl","rb"))
        for fold,(pair,y) in enumerate(zip(pairs,labels)):
            pair=remap[np.asarray(pair,"int64")]; y=np.asarray(y,"int8"); c,ck=values(pair,codep_pos,codep["half0"],codep["half1"]); t,tk=values(pair,tcga_pos,tcga["half0"],tcga["half1"],True); fused=codependency_weight*rank01(c)+(1-codependency_weight)*rank01(t); row={"benchmark":"MuSL-CV3","seed":seed,"fold":fold,"pairs":len(pair),"positives":int(y.sum()),"codependency_coverage":float(ck.mean()),"tcga_coverage":float(tk.mean()),"public_fusion":metrics(y,fused),"codependency":metrics(y,c),"tcga":metrics(y,t)}; rows.append(row); print(json.dumps(row),flush=True)
    mean={name:{metric:float(np.mean([r[name][metric] for r in rows])) for metric in ("auroc","aupr","f1_at_prevalence")} for name in ("public_fusion","codependency","tcga")}; result={"schema":"sl-predict-public-relation-musl-v1","protocol":"One locked evaluation on both official MuSL CV3 seeds; positive split-half-mean DepMap co-dependency and TCGA adjusted mutual exclusivity, median imputation for unsupported fold pairs, separate within-fold average ranks, and normalized split-half Spearman weights fixed before exposure; no double perturbation, SL fitting, sign selection, weight selection, calibration, subset or score-family variant","codependency_weight":codependency_weight,"tcga_weight":1-codependency_weight,"rows":rows,"mean":mean,"advanced":bool(mean["public_fusion"]["auroc"]>.6417025456 and mean["public_fusion"]["aupr"]>.6350067764),"double_perturbation_data_used":False,"sl_labels_used_for_fitting_or_selection":False}; Path(out_path).write_text(json.dumps(result,indent=2)); print(json.dumps({"mean":mean,"advanced":result["advanced"]},indent=2)); return result


def tcga_relation_cv3(state32_path,head32_path,state96_path,head96_path,residual_path,relation_path,feature_path,meta_path,musl_meta,context_pack,out_path,reliability=.17417520342635268,d=384,latent=128,layers=6,seeds=(42,432),contexts=32,crossmodal_path=None,crossmodal_reliability=.4215634152929089):
    from sklearn.cluster import KMeans
    sys.path.insert(0,str(Path(__file__).parents[1]/"training")); from tcga_relation_decoder import Head; from world_model import SLPredict,encode_genes,interaction_head,load_residual_endpoint
    device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(feature_path)["state"].astype("float32")
    def load_world(path,head_path):
        sd=torch.load(path,map_location="cpu",weights_only=True); world=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],sd["context_proj.weight"].shape[1]).to(device); world.load_state_dict(sd); world.eval(); head=interaction_head(latent).to(device); head.load_state_dict(torch.load(head_path,map_location="cpu",weights_only=True)); head.eval(); return world,head,torch.as_tensor(encode_genes(world,state,device),device=device)
    world32,head32,genes32=load_world(state32_path,head32_path); world96,head96,genes96=load_world(state96_path,head96_path); endpoint=load_residual_endpoint(residual_path,state.shape[1],device,d,latent,layers); actions=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); saved=torch.load(relation_path,map_location="cpu",weights_only=True); relation=Head(latent).to(device); relation.load_state_dict(saved["state_dict"]); relation.eval(); z=np.load(context_pack); pool=z["cell_state"][z["train_cell"]]; fit=KMeans(contexts,random_state=731,n_init=10).fit(pool); context_states=np.asarray([pool[np.argmin(((pool-c)**2).sum(1))] for c in fit.cluster_centers_],"float32"); ids={r["symbol"]:i for i,r in enumerate(csv.DictReader(open(meta_path)))}; remap=np.asarray([ids[r["symbol"]] for r in csv.DictReader(open(musl_meta))],"int64"); root=Path(musl_meta).parent; root=root if (root/"test_pairs_seed42.pkl").exists() else root/"data"/"CV3_bins_32"/"fold_data"; rows=[]; crossmodal=None
    if crossmodal_path:
        cm=np.load(crossmodal_path); pos=np.full(len(state),-1,"int32"); pos[cm["genes"].astype("int64")]=np.arange(len(cm["genes"])); crossmodal=(cm,pos)
    for seed in seeds:
        pairs=pickle.load(open(root/f"test_pairs_seed{seed}.pkl","rb")); labels=pickle.load(open(root/f"test_labels_seed{seed}.pkl","rb"))
        for fold,(pair,y) in enumerate(zip(pairs,labels)):
            pair=remap[np.asarray(pair,"int64")]; y=np.asarray(y,"int8"); s32=interaction_score(world32,head32,genes32,pair,context_states,device)["interaction_depletion_mean"]; s96=interaction_score(world96,head96,genes96,pair,context_states,device)["interaction_depletion_mean"]; depletion=.25*s32+.75*s96; predicted=[]
            with torch.no_grad():
                for lo in range(0,len(pair),8192):
                    p=torch.as_tensor(pair[lo:lo+8192],device=device); predicted.append(relation(actions,p[:,0],p[:,1]).cpu())
            predicted=torch.cat(predicted).numpy(); fused=(1-reliability)*rank01(depletion)+reliability*rank01(predicted); row={"benchmark":"MuSL-CV3","seed":seed,"fold":fold,"pairs":len(pair),"positives":int(y.sum()),"fixed_fusion":metrics(y,fused),"retained_dual_world":metrics(y,depletion),"tcga_relation":metrics(y,predicted)}
            if crossmodal is not None:
                cm,pos=crossmodal; a,b=pos[pair[:,0]],pos[pair[:,1]]; known=(a>=0)&(b>=0)&(a!=b); value=np.zeros(len(pair),"float32"); value[known]=(cm["half0"][a[known],b[known]].astype("float32")+cm["half1"][a[known],b[known]].astype("float32"))/2; final=(1-crossmodal_reliability)*rank01(fused)+crossmodal_reliability*rank01(value); row.update({"fixed_crossmodal_fusion":metrics(y,final),"retained_tcga_fusion":row["fixed_fusion"],"crossmodal_relation":metrics(y,value),"crossmodal_coverage":float(known.mean())})
            rows.append(row); print(json.dumps(row),flush=True)
    names=("fixed_crossmodal_fusion","retained_tcga_fusion","crossmodal_relation") if crossmodal is not None else ("fixed_fusion","retained_dual_world","tcga_relation"); mean={name:{metric:float(np.mean([r[name][metric] for r in rows])) for metric in ("auroc","aupr","f1_at_prevalence")} for name in names}
    if crossmodal is not None: result={"schema":"sl-predict-depmap-crossmodal-musl-v1","protocol":"One locked label-free evaluation on both official MuSL CV3 seeds: within-fold average ranks of the admitted cellular-state/TCGA fusion and positive directly measured DepMap expression-to-dependency conditional vulnerability, weighted by its independently measured split-half Spearman reliability; unsupported pairs are zero; no fitting, sign, weight, score-family, source, context, missing-value or fold selection","crossmodal_reliability_weight":crossmodal_reliability,"rows":rows,"mean":mean,"advanced":bool(mean["fixed_crossmodal_fusion"]["auroc"]>.6417025456081681 and mean["fixed_crossmodal_fusion"]["aupr"]>.6350067763570974),"double_perturbation_data_used":False,"sl_labels_used_for_fitting_or_selection":False}
    else: result={"schema":"sl-predict-tcga-relation-musl-v1","protocol":"One locked label-free evaluation on both official MuSL CV3 seeds: within-fold average ranks of retained dual-world depletion and frozen TCGA relation prediction, weighted by the independently measured TCGA held-gene Spearman reliability; component metrics are prespecified diagnostics; no fitting, sign, weight, score-family, context, subset or fold selection","reliability_weight":reliability,"rows":rows,"mean":mean,"advanced":bool(mean["fixed_fusion"]["auroc"]>.6319731520319056 and mean["fixed_fusion"]["aupr"]>.6282295488576508),"double_perturbation_data_used":False,"sl_labels_used_for_fitting":False}
    Path(out_path).write_text(json.dumps(result,indent=2)); print(json.dumps({"mean":mean,"advanced":result["advanced"]},indent=2)); return result


@torch.no_grad()
def residual_interaction_score(endpoint,head,genes,pairs,context_states,device):
    from world_model import residual_interaction_inputs
    out={f"residual_interaction_{target}_{summary}":[] for target in ("depletion","magnitude") for summary in ("mean","top3","max")}
    for at in range(0,len(pairs),2048):
        p=torch.as_tensor(pairs[at:at+2048],device=device); contextual=[]
        for state in context_states:
            cs=torch.as_tensor(state,device=device).expand(len(p),-1); z,r=residual_interaction_inputs(endpoint,genes,p,cs); contextual.append(head(z,r))
        contextual=torch.stack(contextual)
        for name,value in zip(("depletion","magnitude"),(-contextual[:,:,0],contextual[:,:,1])):
            out[f"residual_interaction_{name}_mean"].append(value.mean(0).cpu()); out[f"residual_interaction_{name}_top3"].append(value.topk(min(3,len(value)),0).values.mean(0).cpu()); out[f"residual_interaction_{name}_max"].append(value.max(0).values.cpu())
    return {k:torch.cat(v).numpy() for k,v in out.items()}


@torch.no_grad()
def source_sequential_score(world,head,genes,pairs,context_states,device):
    out=[]
    for at in range(0,len(pairs),1024):
        p=torch.as_tensor(pairs[at:at+1024],device=device); a,b=genes[p[:,0]],genes[p[:,1]]; contextual=[]
        for state in context_states:
            cs=torch.as_tensor(state,device=device).expand(len(p),-1); joint=world.transition(a,b,context_state=cs)[0]; la=world.transition(a,context_state=cs)[0]; lb=world.transition(b,context_state=cs)[0]; ab=world.transition(b,state=la,context_state=cs)[0]; ba=world.transition(a,state=lb,context_state=cs)[0]; source=[]
            for decoder in head.decoders: source.append((decoder(joint)-(decoder(ab)+decoder(ba))/2).square().mean(1).sqrt())
            contextual.append(torch.stack(source).mean(0))
        out.append(torch.stack(contextual).mean(0).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def conditional_viability_score(world,head,genes,pairs,context_states,device):
    out=[]
    for at in range(0,len(pairs),1024):
        p=torch.as_tensor(pairs[at:at+1024],device=device); a,b=genes[p[:,0]],genes[p[:,1]]; contextual=[]
        for state in context_states:
            cs=torch.as_tensor(state,device=device).expand(len(p),-1); za=world.transition(a,context_state=cs)[0]; zb=world.transition(b,context_state=cs)[0]; qab=head(world.transition(b,state=za,context_state=cs)[0]).squeeze(1); qba=head(world.transition(a,state=zb,context_state=cs)[0]).squeeze(1); contextual.append((head(za).squeeze(1)+head(zb).squeeze(1)-qab-qba)/2)
        out.append(torch.stack(contextual).mean(0).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def context_dependency_score(world,dependency,shift,genes,pairs,context_states,device):
    out=[]
    for at in range(0,len(pairs),1024):
        p=torch.as_tensor(pairs[at:at+1024],device=device); a,b=genes[p[:,0]],genes[p[:,1]]; contextual=[]
        for state in context_states:
            cs=torch.as_tensor(state,device=device).expand(len(p),-1); da=torch.nn.functional.normalize(shift(a,cs),dim=1); db=torch.nn.functional.normalize(shift(b,cs),dim=1); qa=dependency(world.transition(a,context_state=cs)[0]).squeeze(1); qb=dependency(world.transition(b,context_state=cs)[0]).squeeze(1); qa_b=dependency(world.transition(a,context_state=cs+db)[0]).squeeze(1); qb_a=dependency(world.transition(b,context_state=cs+da)[0]).squeeze(1); contextual.append(-((qa_b-qa)+(qb_a-qb))/2)
        out.append(torch.stack(contextual).mean(0).cpu())
    return torch.cat(out).numpy()


def context_dependency_cv3(endpoint_path,dependency_path,shift_path,feature_path,meta_path,musl_meta,context_pack,out_path,d=384,latent=128,layers=6,seeds=(42,432),contexts=32):
    from sklearn.cluster import KMeans
    sys.path.insert(0,str(Path(__file__).parents[1]/"training")); from world_model import encode_genes,load_residual_endpoint,tolerance_head; from context_transition import Shift
    device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(feature_path)["state"].astype("float32"); endpoint=load_residual_endpoint(endpoint_path,state.shape[1],device,d,latent,layers); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); dependency=tolerance_head(latent).to(device); dependency.load_state_dict(torch.load(dependency_path,map_location="cpu",weights_only=True)); dependency.eval(); shift=Shift().to(device); shift.load_state_dict(torch.load(shift_path,map_location="cpu",weights_only=True)); shift.eval(); z=np.load(context_pack); pool=z["cell_state"][z["train_cell"]]; fit=KMeans(contexts,random_state=731,n_init=10).fit(pool); context_states=np.asarray([pool[np.argmin(((pool-c)**2).sum(1))] for c in fit.cluster_centers_],"float32"); ids={r["symbol"]:i for i,r in enumerate(csv.DictReader(open(meta_path)))}; remap=np.asarray([ids[r["symbol"]] for r in csv.DictReader(open(musl_meta))],"int64"); root=Path(musl_meta).parent; root=root if (root/"test_pairs_seed42.pkl").exists() else root/"data"/"CV3_bins_32"/"fold_data"; rows=[]
    for seed in seeds:
        pairs=pickle.load(open(root/f"test_pairs_seed{seed}.pkl","rb")); labels=pickle.load(open(root/f"test_labels_seed{seed}.pkl","rb"))
        for fold,(pair,y) in enumerate(zip(pairs,labels)):
            pair=remap[np.asarray(pair,"int64")]; y=np.asarray(y,"int8"); row={"benchmark":"MuSL-CV3","seed":seed,"fold":fold,"pairs":len(pair),"positives":int(y.sum()),**metrics(y,context_dependency_score(endpoint.world,dependency,shift,genes,pair,context_states,device))}; rows.append(row); print(json.dumps(row),flush=True)
    mean={k:float(np.mean([r[k] for r in rows])) for k in ("auroc","aupr","f1_at_prevalence")}; result={"protocol":"First and only MuSL exposure of the preregistered extended single-intervention context model; fixed score is the negative symmetric change in frozen single-gene dependency after adding the partner's unit molecular-response direction to each of 32 deterministic DepMap state medoids; no double perturbation or SL fitting, sign reversal, magnitude or context tuning, calibration, score mixing or fold selection","rows":rows,"mean":mean,"exceeds_retained_label_free_dual_world":bool(mean["auroc"]>.631973 and mean["aupr"]>.628230)}; Path(out_path).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); return result


def conditional_viability_cv3(endpoint_path,head_path,feature_path,meta_path,musl_meta,context_pack,out_path,d=384,latent=128,layers=6,seeds=(42,432),contexts=32):
    from sklearn.cluster import KMeans
    sys.path.insert(0,str(Path(__file__).parents[1]/"training")); from world_model import encode_genes,load_residual_endpoint,tolerance_head
    device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(feature_path)["state"].astype("float32"); endpoint=load_residual_endpoint(endpoint_path,state.shape[1],device,d,latent,layers); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); head=tolerance_head(latent).to(device); head.load_state_dict(torch.load(head_path,map_location="cpu",weights_only=True)); head.eval(); z=np.load(context_pack); pool=z["cell_state"][z["train_cell"]]; fit=KMeans(contexts,random_state=731,n_init=10).fit(pool); context_states=np.asarray([pool[np.argmin(((pool-c)**2).sum(1))] for c in fit.cluster_centers_],"float32"); ids={r["symbol"]:i for i,r in enumerate(csv.DictReader(open(meta_path)))}; remap=np.asarray([ids[r["symbol"]] for r in csv.DictReader(open(musl_meta))],"int64"); root=Path(musl_meta).parent; root=root if (root/"test_pairs_seed42.pkl").exists() else root/"data"/"CV3_bins_32"/"fold_data"; rows=[]
    for seed in seeds:
        pairs=pickle.load(open(root/f"test_pairs_seed{seed}.pkl","rb")); labels=pickle.load(open(root/f"test_labels_seed{seed}.pkl","rb"))
        for fold,(pair,y) in enumerate(zip(pairs,labels)):
            pair=remap[np.asarray(pair,"int64")]; y=np.asarray(y,"int8"); row={"benchmark":"MuSL-CV3","seed":seed,"fold":fold,"pairs":len(pair),"positives":int(y.sum()),**metrics(y,conditional_viability_score(endpoint.world,head,genes,pair,context_states,device))}; rows.append(row); print(json.dumps(row),flush=True)
    mean={k:float(np.mean([r[k] for r in rows])) for k in ("auroc","aupr","f1_at_prevalence")}; result={"protocol":"Frozen symmetric excess conditional single-gene dependency after sequential state transitions, averaged over 32 deterministic DepMap state medoids; head selected only on jointly unseen gene-cell CRISPR effects; no double perturbation or SL labels, sign reversal, calibration, context selection, score mixing or fold selection","rows":rows,"mean":mean,"advanced":bool(mean["auroc"]>.631973 and mean["aupr"]>.628230)}; Path(out_path).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); return result


def source_landmark_cv3(endpoint_path,source_path,feature_path,meta_path,musl_meta,context_pack,out_path,d=384,latent=128,layers=6,seeds=(42,432),contexts=32):
    from sklearn.cluster import KMeans
    sys.path.insert(0,str(Path(__file__).parents[1]/"training")); from world_model import SourceEndpoint, encode_genes, load_residual_endpoint
    device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(feature_path)["state"].astype("float32"); endpoint=load_residual_endpoint(endpoint_path,state.shape[1],device,d,latent,layers); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); head=SourceEndpoint(5,latent,32).to(device); head.load_state_dict(torch.load(source_path,map_location="cpu",weights_only=True)); head.eval(); z=np.load(context_pack); pool=z["cell_state"][z["train_cell"]]; fit=KMeans(contexts,random_state=731,n_init=10).fit(pool); context_states=np.asarray([pool[np.argmin(((pool-c)**2).sum(1))] for c in fit.cluster_centers_],"float32"); ids={r["symbol"]:i for i,r in enumerate(csv.DictReader(open(meta_path)))}; remap=np.asarray([ids[r["symbol"]] for r in csv.DictReader(open(musl_meta))],"int64"); root=Path(musl_meta).parent; root=root if (root/"test_pairs_seed42.pkl").exists() else root/"data"/"CV3_bins_32"/"fold_data"; rows=[]
    for seed in seeds:
        pairs=pickle.load(open(root/f"test_pairs_seed{seed}.pkl","rb")); labels=pickle.load(open(root/f"test_labels_seed{seed}.pkl","rb"))
        for fold,(pair,y) in enumerate(zip(pairs,labels)):
            pair=remap[np.asarray(pair,"int64")]; y=np.asarray(y,"int8"); row={"benchmark":"MuSL-CV3","seed":seed,"fold":fold,"pairs":len(pair),"positives":int(y.sum()),**metrics(y,source_sequential_score(endpoint.world,head,genes,pair,context_states,device))}; rows.append(row); print(json.dumps(row),flush=True)
    mean={k:float(np.mean([r[k] for r in rows])) for k in ("auroc","aupr","f1_at_prevalence")}; result={"protocol":"Retrospective label-free evaluation on both official MuSL CV3 seeds; fixed mean source-specific molecular disagreement between simultaneous and order-averaged sequential knockouts, averaged over 32 deterministic DepMap state medoids; no SL fitting, calibration, sign selection, context selection or fold selection","rows":rows,"mean":mean,"advanced":bool(mean["auroc"]>.5382 and mean["aupr"]>.5310)}; Path(out_path).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); return result


def combiseq_cv3(endpoint_path,decoder_path,composer_path,bridge_path,feature_path,meta_path,musl_meta,context_pack,out_path,d=384,latent=128,layers=6,seeds=(42,432)):
    sys.path.insert(0,str(Path(__file__).parents[1]/"training")); from world_model import encode_genes, load_residual_endpoint; from combiseq import StateComposition
    device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(feature_path)["state"].astype("float32"); endpoint=load_residual_endpoint(endpoint_path,state.shape[1],device,d,latent,layers); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); bridge=np.load(bridge_path); width=bridge["residual"].shape[1]; decoder=torch.nn.Sequential(torch.nn.LayerNorm(latent),torch.nn.Linear(latent,latent),torch.nn.GELU(),torch.nn.Linear(latent,width)).to(device); decoder.load_state_dict(torch.load(decoder_path,map_location="cpu",weights_only=True)); decoder.eval(); composer=StateComposition(width).to(device); composer.load_state_dict(torch.load(composer_path,map_location="cpu",weights_only=True)); composer.eval(); basal=np.load(context_pack); k562=torch.as_tensor(basal["cell_state"][np.flatnonzero(basal["model_ids"].astype(str)=="ACH-000551")[0]],device=device); offset=torch.as_tensor(bridge["decoder_offset"],device=device)
    singles=[]
    with torch.no_grad():
        for start in range(0,len(genes),2048):
            action=genes[start:start+2048]; cs=k562.expand(len(action),-1); singles.append((decoder(endpoint.world.transition(action,context_state=cs)[0])+offset).cpu())
    singles=torch.cat(singles).to(device); ids={r["symbol"]:i for i,r in enumerate(csv.DictReader(open(meta_path)))}; remap=np.asarray([ids[r["symbol"]] for r in csv.DictReader(open(musl_meta))],"int64"); root=Path(musl_meta).parent; root=root if (root/"test_pairs_seed42.pkl").exists() else root/"data"/"CV3_bins_32"/"fold_data"; rows=[]
    for seed in seeds:
        pairs=pickle.load(open(root/f"test_pairs_seed{seed}.pkl","rb")); labels=pickle.load(open(root/f"test_labels_seed{seed}.pkl","rb"))
        for fold,(pair,y) in enumerate(zip(pairs,labels)):
            pair=remap[np.asarray(pair,"int64")]; score=[]
            with torch.no_grad():
                for start in range(0,len(pair),4096):
                    p=torch.as_tensor(pair[start:start+4096],device=device); score.append(torch.linalg.vector_norm(composer(singles[p[:,0]],singles[p[:,1]]),dim=1).cpu())
            y=np.asarray(y,"int8"); row={"benchmark":"MuSL-CV3","seed":seed,"fold":fold,"pairs":len(pair),"positives":int(y.sum()),**metrics(y,torch.cat(score).numpy())}; rows.append(row); print(json.dumps(row),flush=True)
    mean={k:float(np.mean([r[k] for r in rows])) for k in ("auroc","aupr","f1_at_prevalence")}; result={"protocol":"Retrospective first SL exposure of the frozen Combi-Seq response-space composer on both official MuSL CV3 seeds; fixed score is the positive L2 norm of its predicted correction from exact K562 single-gene states; no genetic double-perturbation training, SL fitting, calibration, sign selection, context search or fold selection","rows":rows,"mean":mean,"exceeds_retained_label_free_dual_world":bool(mean["auroc"]>.631973 and mean["aupr"]>.628230)}; Path(out_path).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); return result


@torch.no_grad()
def tolerance_effects(model,head,genes,context_states,device):
    genes=torch.as_tensor(genes,dtype=torch.float32,device=device); out=[]
    for state in context_states:
        values=[]
        for at in range(0,len(genes),2048):
            action=genes[at:at+2048]; cs=torch.as_tensor(state,device=device).expand(len(action),-1); values.append(head(model.transition(action,context_state=cs)[0]).squeeze(1).cpu())
        out.append(torch.cat(values).numpy())
    return np.stack(out,1)


def tolerance_pair_summary(effect,pairs):
    a,b=effect[pairs[:,0]],effect[pairs[:,1]]; values=(np.minimum(a,b),np.maximum(a,b),(a+b)/2,np.abs(a-b)); out=[]
    for x in values:out.extend((x.mean(1),x.std(1),x.min(1),x.max(1),np.quantile(x,.25,axis=1),np.quantile(x,.75,axis=1)))
    out.extend((((a>-.5)&(b>-.5)).mean(1),((a<-.5)|(b<-.5)).mean(1)))
    return np.column_stack(out).astype("float32")


@torch.no_grad()
def sequential_features(model,genes,pairs,device,batch=4096):
    genes=torch.as_tensor(genes,dtype=torch.float32,device=device); out=[]
    for at in range(0,len(pairs),batch):
        p=torch.as_tensor(pairs[at:at+batch].astype("int64"),device=device); a,b=genes[p[:,0]],genes[p[:,1]]; la=model.transition(a)[0]; lb=model.transition(b)[0]; ab=model.transition(b,state=la)[0]; ba=model.transition(a,state=lb)[0]; out.append(torch.cat(((a-b).abs(),a*b,ab,ba,(ab-ba).abs(),ab*ba,model.outcome(ab),model.outcome(ba)),1).cpu())
    return torch.cat(out).numpy()


def run(args):
    sys.path.insert(0,str(Path(__file__).parents[1]/"training")); from world_model import SLPredict, ResidualInteraction, encode_genes, interaction_head, load_residual_endpoint, tolerance_head
    device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(args.features)["state"].astype("float32"); base=list(csv.DictReader(open(args.meta))); ids={r["symbol"]:i for i,r in enumerate(base)}; musl=list(csv.DictReader(open(args.musl_meta))); remap=np.asarray([ids[r["symbol"]] for r in musl],"int64")
    sd=torch.load(args.model,map_location="cpu",weights_only=True); context_dim=sd["context_proj.weight"].shape[1] if "context_proj.weight" in sd else 0; model=SLPredict(args.d,args.latent,args.layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],context_dim).to(device); model.load_state_dict(sd); model.eval(); ds=torch.load(args.decoder,map_location="cpu",weights_only=True); decoder=torch.nn.Linear(args.latent,ds["weight"].shape[0]).to(device); decoder.load_state_dict(ds); decoder.eval(); genes=torch.as_tensor(encode_genes(model,state,device),device=device); context_states=None; contextual_name="source_sequential"
    if args.context_pack:
        from sklearn.cluster import KMeans
        z=np.load(args.context_pack); pool=z["cell_state"][z["train_cell"]]; fit=KMeans(args.contexts,random_state=731,n_init=10).fit(pool); context_states=np.asarray([pool[np.argmin(((pool-c)**2).sum(1))] for c in fit.cluster_centers_],"float32"); contextual_name="basal_marginal_sequential"
    tolerance=None
    if args.tolerance_head:
        tolerance=tolerance_head(args.latent).to(device); tolerance.load_state_dict(torch.load(args.tolerance_head,map_location="cpu",weights_only=True)); tolerance.eval()
    interaction=None; ensemble=None; residual=None
    if args.interaction_head:
        interaction=interaction_head(args.latent).to(device); interaction.load_state_dict(torch.load(args.interaction_head,map_location="cpu",weights_only=True)); interaction.eval()
    if getattr(args,"ensemble_model",None):
        es=torch.load(args.ensemble_model,map_location="cpu",weights_only=True); ensemble_model=SLPredict(args.d,args.latent,args.layers,es["cell.weight"].shape[0],es["outcome.weight"].shape[0],state.shape[1],es["context_proj.weight"].shape[1]).to(device); ensemble_model.load_state_dict(es); ensemble_model.eval(); ensemble_genes=torch.as_tensor(encode_genes(ensemble_model,state,device),device=device); ensemble_head=interaction_head(args.latent).to(device); ensemble_head.load_state_dict(torch.load(args.ensemble_interaction_head,map_location="cpu",weights_only=True)); ensemble_head.eval(); ensemble=(ensemble_model,ensemble_head,ensemble_genes)
    if getattr(args,"residual_model",None):
        endpoint=load_residual_endpoint(args.residual_model,state.shape[1],device,args.d,args.latent,args.layers); residual_genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); residual_base=interaction_head(args.latent).to(device); residual_head=ResidualInteraction(residual_base).to(device); residual_head.load_state_dict(torch.load(args.residual_interaction_head,map_location="cpu",weights_only=True)); residual_head.eval(); residual=(endpoint,residual_head,residual_genes)
    root=Path(args.musl_meta).parent; root=root if (root/"test_pairs_seed42.pkl").exists() else root/"data"/"CV3_bins_32"/"fold_data"; rows=[]
    for seed in args.seeds:
        pairs=pickle.load(open(root/f"test_pairs_seed{seed}.pkl","rb")); labels=pickle.load(open(root/f"test_labels_seed{seed}.pkl","rb"))
        for fold,(pair,y) in enumerate(zip(pairs,labels)):
            pair=remap[np.asarray(pair,"int64")]; y=np.asarray(y,"int8"); unknown,contextual=score(model,decoder,genes,pair,device,context_states,context_states is not None); row={"benchmark":"MuSL-CV3","seed":seed,"fold":fold,"pairs":len(pair),"positives":int(y.sum()),"unknown_sequential":metrics(y,unknown)}
            if isinstance(contextual,dict):row.update({name:metrics(y,value) for name,value in contextual.items()})
            else:row[contextual_name]=metrics(y,contextual)
            if tolerance is not None:row.update({name:metrics(y,value) for name,value in tolerated_score(model,decoder,tolerance,genes,pair,context_states,device).items()})
            if interaction is not None:
                primary=interaction_score(model,interaction,genes,pair,context_states,device); row.update({name:metrics(y,value) for name,value in primary.items()})
                if ensemble is not None:
                    secondary=interaction_score(*ensemble,pair,context_states,device); row.update({f"resolution_ensemble_{name}":metrics(y,(1-args.ensemble_weight)*value+args.ensemble_weight*secondary[name]) for name,value in primary.items()})
            if residual is not None:row.update({name:metrics(y,value) for name,value in residual_interaction_score(*residual,pair,context_states,device).items()})
            rows.append(row); print(json.dumps(row),flush=True)
    names=[x for x in rows[0] if isinstance(rows[0][x],dict)]; summary={name:{metric:float(np.mean([r[name][metric] for r in rows])) for metric in ("auroc","aupr","f1_at_prevalence")} for name in names}; protocol="Official MuSL pan-cancer CV3 files; both genes absent from each fold's training genes; seeds fixed before labels were opened"; protocol += f"; contextual scores summarize {args.contexts} deterministic DepMap state medoids by mean, top-three mean and maximum" if context_states is not None else ""
    if interaction is not None:protocol="Retrospective development evaluation on both official MuSL pan-cancer CV3 seeds; both genes absent from each fold's training genes; outcomes were inspected in earlier readout experiments; fixed score is negative continuous double-knockout depletion from a frozen world model, averaged over 32 deterministic DepMap state medoids; no MuSL labels, calibration, sign selection or fold selection"; protocol += f"; dual-resolution score uses a molecularly fixed {args.ensemble_weight:.2f} weight on the second world model" if ensemble is not None else ""
    if residual is not None:protocol="Retrospective first exposure of the frozen residual-endpoint correction on both official MuSL pan-cancer CV3 seeds; both genes absent from each fold's training genes; fixed score is negative continuous double-knockout depletion averaged over 32 deterministic DepMap state medoids; no MuSL labels, calibration, sign selection, context selection or fold selection"
    result={"protocol":protocol,"rows":rows,"mean":summary}; Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(result,indent=2)); print(json.dumps(summary,indent=2))


def calibrated(model_path,feature_path,meta_path,musl_meta,out_path,d=384,latent=128,layers=6,seeds=(42,432),aux_path=None,aux_sequential=False,tolerance_path=None,context_pack=None,contexts=32,codependency_path=None,tcga_path=None,silencing_path=None,response_path=None,stacked=False):
    from sklearn.ensemble import ExtraTreesClassifier
    from lightgbm import LGBMClassifier
    sys.path.insert(0,str(Path(__file__).parents[1]/"training")); from world_model import SLPredict, embed_pairs, encode_genes, fit_head, observed_relations, pair_summary, tolerance_head
    device="cuda" if torch.cuda.is_available() else "cpu"; pack=np.load(feature_path); state=pack["state"].astype("float32"); ids={r["symbol"]:i for i,r in enumerate(csv.DictReader(open(meta_path)))}; musl=list(csv.DictReader(open(musl_meta))); remap=np.asarray([ids[r["symbol"]] for r in musl],"int64"); sd=torch.load(model_path,map_location="cpu",weights_only=True); context_dim=sd["context_proj.weight"].shape[1] if "context_proj.weight" in sd else 0; model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],context_dim).to(device); model.load_state_dict(sd); model.eval(); root=Path(musl_meta).parent; root=root if (root/"train_pairs_seed42.pkl").exists() else root/"data"/"CV3_bins_32"/"fold_data"; rows=[]; external=[]
    if codependency_path and tcga_path:
        codep=np.load(codependency_path); tcga=np.load(tcga_path); cp=np.full(len(state),-1,"int32"); cp[codep["genes"].astype("int64")]=np.arange(len(codep["genes"])); tp=np.full(len(state),-1,"int32"); tp[tcga["genes"].astype("int64")]=np.arange(len(tcga["genes"])); external.extend(((codep,cp,False),(tcga,tp,True)))
    if silencing_path:
        silencing=np.load(silencing_path); sp=np.full(len(state),-1,"int32"); sp[silencing["genes"].astype("int64")]=np.arange(len(silencing["genes"])); external.append((silencing,sp,False))
    response=np.load(response_path) if response_path else None
    def external_features(pair):
        features=[]
        for source,pos,triangular in external:
            a,b=pos[pair[:,0]],pos[pair[:,1]]; known=(a>=0)&(b>=0)&(a!=b); mean=np.zeros(len(pair),"float32"); disagreement=np.zeros(len(pair),"float32")
            if triangular:
                n=len(source["genes"]); lo=np.minimum(a[known],b[known]).astype("int64"); hi=np.maximum(a[known],b[known]).astype("int64"); ix=lo*(2*n-lo-1)//2+hi-lo-1; x=source["half0"][ix]; y=source["half1"][ix]
            else:x=source["half0"][a[known],b[known]].astype("float32"); y=source["half1"][a[known],b[known]].astype("float32")
            mean[known]=(x+y)/2; disagreement[known]=np.abs(x-y); features.extend((mean,disagreement,known.astype("float32")))
        return np.column_stack(features)
    if aux_path:
        aux=np.load(aux_path); auxp=aux["pairs"].astype("int64"); auxy=aux["label"].astype("int8"); gene_latent=encode_genes(model,state,device) if aux_sequential else None; auxx=sequential_features(model,gene_latent,auxp,device) if aux_sequential else np.column_stack((embed_pairs(model,state,auxp,device),pair_summary(state,auxp)))
    tolerance_effect=None
    if tolerance_path:
        from sklearn.cluster import KMeans
        head=tolerance_head(latent).to(device); head.load_state_dict(torch.load(tolerance_path,map_location="cpu",weights_only=True)); head.eval(); z=np.load(context_pack); pool=z["cell_state"][z["train_cell"]]; fit=KMeans(contexts,random_state=731,n_init=10).fit(pool); context_states=np.asarray([pool[np.argmin(((pool-c)**2).sum(1))] for c in fit.cluster_centers_],"float32"); tolerance_effect=tolerance_effects(model,head,encode_genes(model,state,device),context_states,device)
    def components(x,y,z,fold):
        forest=ExtraTreesClassifier(n_estimators=256,min_samples_leaf=3,max_features=.5,class_weight="balanced",n_jobs=-1,random_state=123).fit(x,y).predict_proba(z)[:,1]; boost=LGBMClassifier(n_estimators=500,num_leaves=31,learning_rate=.03,colsample_bytree=.7,reg_lambda=1,min_child_samples=30,n_jobs=8,verbosity=-1,random_state=123).fit(x,y).predict_proba(z)[:,1]; mlp=[]
        for i in range(3):torch.manual_seed(123+10*fold+i); mlp.append(fit_head(x,y,z,device))
        return np.column_stack((forest,boost,np.mean(mlp,0)))
    for seed in seeds:
        train_pairs=pickle.load(open(root/f"train_pairs_seed{seed}.pkl","rb")); test_pairs=pickle.load(open(root/f"test_pairs_seed{seed}.pkl","rb")); train_labels=pickle.load(open(root/f"train_labels_seed{seed}.pkl","rb")); test_labels=pickle.load(open(root/f"test_labels_seed{seed}.pkl","rb"))
        for fold,(tr,te,ytr,y) in enumerate(zip(train_pairs,test_pairs,train_labels,test_labels)):
            tr=remap[np.asarray(tr,"int64")]; te=remap[np.asarray(te,"int64")]; ytr=np.asarray(ytr,"int8"); y=np.asarray(y,"int8"); response_feature=lambda p:(lambda a,b:(np.sum(a*b,1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)).clip(1e-8))[:,None])(*(response["profile"][p[:,k]].astype("float32") for k in (0,1))) if response is not None else np.empty((len(p),0)); xtr=np.column_stack((embed_pairs(model,state,tr,device),pair_summary(state,tr),observed_relations(pack,tr,len(state)),tolerance_pair_summary(tolerance_effect,tr) if tolerance_effect is not None else np.empty((len(tr),0)),external_features(tr) if external else np.empty((len(tr),0)),response_feature(tr))); xte=np.column_stack((embed_pairs(model,state,te,device),pair_summary(state,te),observed_relations(pack,te,len(state)),tolerance_pair_summary(tolerance_effect,te) if tolerance_effect is not None else np.empty((len(te),0)),external_features(te) if external else np.empty((len(te),0)),response_feature(te)))
            kept=None
            if aux_path:
                held=np.unique(te); kept=~np.isin(auxp[:,0],held)&~np.isin(auxp[:,1],held)
                if aux_sequential:
                    qtr=sequential_features(model,gene_latent,tr,device); qte=sequential_features(model,gene_latent,te,device); torch.manual_seed(20260826+seed+fold); extra=fit_head(auxx[kept],auxy[kept],np.row_stack((qtr,qte)),device,epochs=20); xtr=np.column_stack((xtr,extra[:len(tr)])); xte=np.column_stack((xte,extra[len(tr):]))
                else:
                    external=LGBMClassifier(n_estimators=300,num_leaves=31,learning_rate=.03,colsample_bytree=.7,reg_lambda=1,min_child_samples=30,n_jobs=8,verbosity=-1,random_state=20260826).fit(auxx[kept],auxy[kept]); width=auxx.shape[1]; xtr=np.column_stack((xtr,external.predict_proba(xtr[:,:width])[:,1])); xte=np.column_stack((xte,external.predict_proba(xte[:,:width])[:,1]))
            if stacked:
                held=np.zeros(len(state),bool)
                for g in np.unique(tr):held[g]=hashlib.sha256(f"{seed}:{fold}:{int(g)}".encode()).digest()[0]%5==0
                inner_train=~held[tr].any(1); inner_valid=held[tr].all(1); assert len(np.unique(ytr[inner_valid]))==2 and inner_valid.sum()>=256
                inner=components(xtr[inner_train],ytr[inner_train],xtr[inner_valid],fold); candidates=[np.asarray((i/20,j/20,(20-i-j)/20)) for i in range(21) for j in range(21-i)]; losses=[log_loss(ytr[inner_valid],np.clip(inner@w,1e-6,1-1e-6)) for w in candidates]; weight=candidates[int(np.argmin(losses))]; score=components(xtr,ytr,xte,fold)@weight
            else:weight=np.ones(3)/3; score=components(xtr,ytr,xte,fold)@weight
            row={"benchmark":"MuSL-CV3","seed":seed,"fold":fold,"pairs":len(te),**metrics(y,score)}
            if stacked:row.update({"inner_training_pairs":int(inner_train.sum()),"inner_validation_pairs":int(inner_valid.sum()),"inner_log_loss":float(min(losses)),"weights":weight.tolist()})
            if kept is not None: row["hap1_auxiliary_pairs"]=int(kept.sum()); row["hap1_auxiliary_positives"]=int(auxy[kept].sum())
            rows.append(row); print(json.dumps(row),flush=True)
    protocol="Official MuSL fold-local SL training; fixed deterministic SL-Predict readout; no test-fold selection"
    if stacked:protocol+="; ExtraTrees, LightGBM and three-seed neural-mean weights selected on one deterministic gene-cold split inside each training fold by a 0.05 simplex grid minimizing binary log loss"
    if aux_path: protocol += f"; one HAP1-derived {'sequential-transition neural' if aux_sequential else 'static tree'} feature fitted after removing every gene present in the corresponding MuSL test fold"
    if tolerance_path:protocol += f"; 26 fixed summaries of raw single-gene dependency predicted across {contexts} deterministic DepMap state medoids"
    if codependency_path and tcga_path:protocol += "; six fixed public relation features: split-half mean, disagreement and support for DepMap co-dependency and TCGA adjusted mutual exclusivity"
    if silencing_path:protocol += "; three additional fixed features: split-half mean, disagreement and support for lineage-adjusted DepMap expression silencing"
    if response_path:protocol += "; one independently admitted positive cosine between exact-context full-transcriptome response profiles"
    result={"protocol":protocol,"rows":rows,"mean":{k:float(np.mean([r[k] for r in rows])) for k in ("auroc","aupr","f1_at_prevalence")}}; Path(out_path).write_text(json.dumps(result,indent=2)); print(json.dumps(result["mean"],indent=2)); return result


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--model",required=True); p.add_argument("--decoder",required=True); p.add_argument("--features",default="results/sl_predict/features_spectral_safe.npz"); p.add_argument("--meta",default="data/feng2024/data/preprocessed_data/meta_table_9845.csv"); p.add_argument("--musl-meta",default="data/models/MuSL/processed_data/meta_table_7684.csv"); p.add_argument("--output",default="results/sl_predict/musl_cv3_confirmatory.json"); p.add_argument("--seeds",type=int,nargs="+",default=(42,432)); p.add_argument("--context-pack"); p.add_argument("--tolerance-head"); p.add_argument("--interaction-head"); p.add_argument("--ensemble-model"); p.add_argument("--ensemble-interaction-head"); p.add_argument("--ensemble-weight",type=float,default=.75); p.add_argument("--residual-model"); p.add_argument("--residual-interaction-head"); p.add_argument("--contexts",type=int,default=32); p.add_argument("--d",type=int,default=384); p.add_argument("--latent",type=int,default=128); p.add_argument("--layers",type=int,default=6); run(p.parse_args())
