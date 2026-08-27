import argparse, csv, json, pickle, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


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


def source_landmark_cv3(endpoint_path,source_path,feature_path,meta_path,musl_meta,context_pack,out_path,d=384,latent=128,layers=6,seeds=(42,432),contexts=32):
    from sklearn.cluster import KMeans
    sys.path.insert(0,str(Path(__file__).parents[1]/"training")); from world_model import SourceEndpoint, encode_genes, load_residual_endpoint
    device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(feature_path)["state"].astype("float32"); endpoint=load_residual_endpoint(endpoint_path,state.shape[1],device,d,latent,layers); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); head=SourceEndpoint(5,latent,32).to(device); head.load_state_dict(torch.load(source_path,map_location="cpu",weights_only=True)); head.eval(); z=np.load(context_pack); pool=z["cell_state"][z["train_cell"]]; fit=KMeans(contexts,random_state=731,n_init=10).fit(pool); context_states=np.asarray([pool[np.argmin(((pool-c)**2).sum(1))] for c in fit.cluster_centers_],"float32"); ids={r["symbol"]:i for i,r in enumerate(csv.DictReader(open(meta_path)))}; remap=np.asarray([ids[r["symbol"]] for r in csv.DictReader(open(musl_meta))],"int64"); root=Path(musl_meta).parent; root=root if (root/"test_pairs_seed42.pkl").exists() else root/"data"/"CV3_bins_32"/"fold_data"; rows=[]
    for seed in seeds:
        pairs=pickle.load(open(root/f"test_pairs_seed{seed}.pkl","rb")); labels=pickle.load(open(root/f"test_labels_seed{seed}.pkl","rb"))
        for fold,(pair,y) in enumerate(zip(pairs,labels)):
            pair=remap[np.asarray(pair,"int64")]; y=np.asarray(y,"int8"); row={"benchmark":"MuSL-CV3","seed":seed,"fold":fold,"pairs":len(pair),"positives":int(y.sum()),**metrics(y,source_sequential_score(endpoint.world,head,genes,pair,context_states,device))}; rows.append(row); print(json.dumps(row),flush=True)
    mean={k:float(np.mean([r[k] for r in rows])) for k in ("auroc","aupr","f1_at_prevalence")}; result={"protocol":"Retrospective label-free evaluation on both official MuSL CV3 seeds; fixed mean source-specific molecular disagreement between simultaneous and order-averaged sequential knockouts, averaged over 32 deterministic DepMap state medoids; no SL fitting, calibration, sign selection, context selection or fold selection","rows":rows,"mean":mean,"advanced":bool(mean["auroc"]>.5382 and mean["aupr"]>.5310)}; Path(out_path).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); return result


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


def calibrated(model_path,feature_path,meta_path,musl_meta,out_path,d=384,latent=128,layers=6,seeds=(42,432),aux_path=None,aux_sequential=False,tolerance_path=None,context_pack=None,contexts=32):
    from sklearn.ensemble import ExtraTreesClassifier
    from lightgbm import LGBMClassifier
    sys.path.insert(0,str(Path(__file__).parents[1]/"training")); from world_model import SLPredict, embed_pairs, encode_genes, fit_head, observed_relations, pair_summary, tolerance_head
    device="cuda" if torch.cuda.is_available() else "cpu"; pack=np.load(feature_path); state=pack["state"].astype("float32"); ids={r["symbol"]:i for i,r in enumerate(csv.DictReader(open(meta_path)))}; musl=list(csv.DictReader(open(musl_meta))); remap=np.asarray([ids[r["symbol"]] for r in musl],"int64"); sd=torch.load(model_path,map_location="cpu",weights_only=True); context_dim=sd["context_proj.weight"].shape[1] if "context_proj.weight" in sd else 0; model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],context_dim).to(device); model.load_state_dict(sd); model.eval(); root=Path(musl_meta).parent; root=root if (root/"train_pairs_seed42.pkl").exists() else root/"data"/"CV3_bins_32"/"fold_data"; rows=[]
    if aux_path:
        aux=np.load(aux_path); auxp=aux["pairs"].astype("int64"); auxy=aux["label"].astype("int8"); gene_latent=encode_genes(model,state,device) if aux_sequential else None; auxx=sequential_features(model,gene_latent,auxp,device) if aux_sequential else np.column_stack((embed_pairs(model,state,auxp,device),pair_summary(state,auxp)))
    tolerance_effect=None
    if tolerance_path:
        from sklearn.cluster import KMeans
        head=tolerance_head(latent).to(device); head.load_state_dict(torch.load(tolerance_path,map_location="cpu",weights_only=True)); head.eval(); z=np.load(context_pack); pool=z["cell_state"][z["train_cell"]]; fit=KMeans(contexts,random_state=731,n_init=10).fit(pool); context_states=np.asarray([pool[np.argmin(((pool-c)**2).sum(1))] for c in fit.cluster_centers_],"float32"); tolerance_effect=tolerance_effects(model,head,encode_genes(model,state,device),context_states,device)
    for seed in seeds:
        train_pairs=pickle.load(open(root/f"train_pairs_seed{seed}.pkl","rb")); test_pairs=pickle.load(open(root/f"test_pairs_seed{seed}.pkl","rb")); train_labels=pickle.load(open(root/f"train_labels_seed{seed}.pkl","rb")); test_labels=pickle.load(open(root/f"test_labels_seed{seed}.pkl","rb"))
        for fold,(tr,te,ytr,y) in enumerate(zip(train_pairs,test_pairs,train_labels,test_labels)):
            tr=remap[np.asarray(tr,"int64")]; te=remap[np.asarray(te,"int64")]; ytr=np.asarray(ytr,"int8"); y=np.asarray(y,"int8"); xtr=np.column_stack((embed_pairs(model,state,tr,device),pair_summary(state,tr),observed_relations(pack,tr,len(state)),tolerance_pair_summary(tolerance_effect,tr) if tolerance_effect is not None else np.empty((len(tr),0)))); xte=np.column_stack((embed_pairs(model,state,te,device),pair_summary(state,te),observed_relations(pack,te,len(state)),tolerance_pair_summary(tolerance_effect,te) if tolerance_effect is not None else np.empty((len(te),0))))
            kept=None
            if aux_path:
                held=np.unique(te); kept=~np.isin(auxp[:,0],held)&~np.isin(auxp[:,1],held)
                if aux_sequential:
                    qtr=sequential_features(model,gene_latent,tr,device); qte=sequential_features(model,gene_latent,te,device); torch.manual_seed(20260826+seed+fold); extra=fit_head(auxx[kept],auxy[kept],np.row_stack((qtr,qte)),device,epochs=20); xtr=np.column_stack((xtr,extra[:len(tr)])); xte=np.column_stack((xte,extra[len(tr):]))
                else:
                    external=LGBMClassifier(n_estimators=300,num_leaves=31,learning_rate=.03,colsample_bytree=.7,reg_lambda=1,min_child_samples=30,n_jobs=8,verbosity=-1,random_state=20260826).fit(auxx[kept],auxy[kept]); width=auxx.shape[1]; xtr=np.column_stack((xtr,external.predict_proba(xtr[:,:width])[:,1])); xte=np.column_stack((xte,external.predict_proba(xte[:,:width])[:,1]))
            forest=ExtraTreesClassifier(n_estimators=256,min_samples_leaf=3,max_features=.5,class_weight="balanced",n_jobs=-1,random_state=123).fit(xtr,ytr).predict_proba(xte)[:,1]; boost=LGBMClassifier(n_estimators=500,num_leaves=31,learning_rate=.03,colsample_bytree=.7,reg_lambda=1,min_child_samples=30,n_jobs=8,verbosity=-1,random_state=123).fit(xtr,ytr).predict_proba(xte)[:,1]; mlp=[]
            for i in range(3): torch.manual_seed(123+10*fold+i); mlp.append(fit_head(xtr,ytr,xte,device))
            score=(forest+boost+np.mean(mlp,0))/3; row={"benchmark":"MuSL-CV3","seed":seed,"fold":fold,"pairs":len(te),**metrics(y,score)}
            if kept is not None: row["hap1_auxiliary_pairs"]=int(kept.sum()); row["hap1_auxiliary_positives"]=int(auxy[kept].sum())
            rows.append(row); print(json.dumps(row),flush=True)
    protocol="Official MuSL fold-local SL training; fixed deterministic SL-Predict readout; no test-fold selection"
    if aux_path: protocol += f"; one HAP1-derived {'sequential-transition neural' if aux_sequential else 'static tree'} feature fitted after removing every gene present in the corresponding MuSL test fold"
    if tolerance_path:protocol += f"; 26 fixed summaries of raw single-gene dependency predicted across {contexts} deterministic DepMap state medoids"
    result={"protocol":protocol,"rows":rows,"mean":{k:float(np.mean([r[k] for r in rows])) for k in ("auroc","aupr","f1_at_prevalence")}}; Path(out_path).write_text(json.dumps(result,indent=2)); print(json.dumps(result["mean"],indent=2)); return result


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--model",required=True); p.add_argument("--decoder",required=True); p.add_argument("--features",default="results/sl_predict/features_spectral_safe.npz"); p.add_argument("--meta",default="data/feng2024/data/preprocessed_data/meta_table_9845.csv"); p.add_argument("--musl-meta",default="data/models/MuSL/processed_data/meta_table_7684.csv"); p.add_argument("--output",default="results/sl_predict/musl_cv3_confirmatory.json"); p.add_argument("--seeds",type=int,nargs="+",default=(42,432)); p.add_argument("--context-pack"); p.add_argument("--tolerance-head"); p.add_argument("--interaction-head"); p.add_argument("--ensemble-model"); p.add_argument("--ensemble-interaction-head"); p.add_argument("--ensemble-weight",type=float,default=.75); p.add_argument("--residual-model"); p.add_argument("--residual-interaction-head"); p.add_argument("--contexts",type=int,default=32); p.add_argument("--d",type=int,default=384); p.add_argument("--latent",type=int,default=128); p.add_argument("--layers",type=int,default=6); run(p.parse_args())
