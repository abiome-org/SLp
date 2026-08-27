import csv, json, pickle, sys
from pathlib import Path
import numpy as np, torch


def ranking(rows, ks=(1,3,5,10,15,20)):
    out={f"recall@{k}":[] for k in ks}; out.update({f"precision@{k}":[] for k in ks}); rr=[]
    for labels,scores in rows:
        order=np.argsort(scores)[::-1]; y=labels[order]; positives=y.sum()
        if not positives: continue
        rr.append(1/(np.flatnonzero(y)[0]+1))
        for k in ks: out[f"recall@{k}"].append(y[:k].sum()/positives); out[f"precision@{k}"].append(y[:k].sum()/min(positives,k))
    return {"queries":len(rows),"positive_queries":len(rr),"mrr":float(np.mean(rr)),**{k:float(np.mean(v)) for k,v in out.items()}}


def text_embeddings(path):
    import ast
    rows=csv.DictReader(open(path,encoding="utf-8")); emb={r["gene_name"]:np.asarray(ast.literal_eval(r["Embedding"]),"float32") for r in rows if r["gene_name"]!="gene_name"}; return {g:x/(np.linalg.norm(x)+1e-8) for g,x in emb.items() if np.isfinite(x).all()}


def query_rows(labels,sizes,scores):
    rows=[]; at=0
    for n in sizes: rows.append((labels[at:at+n],scores[at:at+n])); at+=n
    return rows


def rank_fusion(scores,sizes,weights):
    out=[]; at=0
    for n in sizes:
        rank=lambda x:np.argsort(np.argsort(x))/max(1,len(x)-1); out.append(sum(w*rank(s[at:at+n]) for w,s in zip(weights,scores))); at+=n
    return np.concatenate(out)


def tune_fusion(labels,sizes,scores):
    choices=[(i/4,j/4,(4-i-j)/4) for i in range(5) for j in range(5-i)]; best=None
    for weights in choices:
        metric=ranking(query_rows(labels,sizes,rank_fusion(scores,sizes,weights))); key=(np.nan_to_num(metric["mrr"],nan=-1.),np.nan_to_num(metric["recall@20"],nan=-1.))
        if best is None or key>best[0]: best=(key,weights)
    return best[1]


def fit_probe(train,valid,test):
    from lightgbm import LGBMRanker, early_stopping, log_evaluation
    _,y,g,*_,x=train; _,yv,gv,*_,xv=valid; *_,xt=test
    if len(np.unique(y))<2: return np.zeros(len(xt),"float32")
    model=LGBMRanker(objective="lambdarank",n_estimators=400,learning_rate=.03,num_leaves=15,max_depth=5,min_child_samples=20,subsample=.8,colsample_bytree=.7,reg_lambda=2,verbosity=-1,n_jobs=8,random_state=123)
    model.fit(x,y.astype("int8"),group=[n for n in g if n],eval_set=[(xv,yv.astype("int8"))],eval_group=[[n for n in gv if n]],eval_at=(1,3,5,10,20),callbacks=[early_stopping(40,verbose=False),log_evaluation(0)])
    return model.predict(xt,num_iteration=model.best_iteration_)


def evaluate_text(split_files,text_files,out_path):
    results=[]
    for split_file,text_file in zip(split_files,text_files):
        emb=text_embeddings(text_file); cell=Path(split_file).name.split("_scenario",1)[0]
        for fold,(_,_,test) in enumerate(pickle.load(open(split_file,"rb"))):
            rows=[]
            for query,partners in test.items():
                kept=[(label=="SL",float(emb[query]@emb[p])) for p,_,label in partners if query in emb and p in emb]; rows.append((np.asarray([x[0] for x in kept]),np.asarray([x[1] for x in kept])))
            row={"cell_line":cell,"fold":fold,**ranking(rows)}; results.append(row); print(json.dumps(row),flush=True)
    Path(out_path).write_text(json.dumps(results,indent=2)); return results


@torch.no_grad()
def evaluate_residual_interaction(endpoint_path,head_path,feature_path,meta_path,context_path,split_files,out_path,d=384,latent=128,layers=6):
    sys.path.insert(0,"/root"); from world_model import ResidualInteraction, batches, encode_genes, interaction_head, load_residual_endpoint, residual_interaction_inputs
    device="cuda"; state=np.load(feature_path)["state"].astype("float32"); symbols=[r["symbol"] for r in csv.DictReader(open(meta_path))]; ids=dict(zip(symbols,range(len(symbols)))); endpoint=load_residual_endpoint(endpoint_path,state.shape[1],device,d,latent,layers); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); head=ResidualInteraction(interaction_head(latent).to(device)).to(device); head.load_state_dict(torch.load(head_path,map_location="cpu",weights_only=True)); head.eval(); context=np.load(context_path); ci={x:i for i,x in enumerate(context["model_ids"].astype(str))}; model_ids={"A549":"ACH-000681","JURKAT":"ACH-000995","K562":"ACH-000551"}; results=[]
    for split_file in split_files:
        cell=Path(split_file).name.split("_scenario",1)[0].upper(); cs=torch.as_tensor(context["cell_state"][ci[model_ids[cell]]],device=device)
        for fold,(_,_,test) in enumerate(pickle.load(open(split_file,"rb"))):
            pairs=[]; labels=[]; sizes=[]
            for query,partners in test.items():
                kept=[(ids[query],ids[p],label=="SL") for p,_,label in partners if query in ids and p in ids]; pairs.extend((a,b) for a,b,_ in kept); labels.extend(y for _,_,y in kept); sizes.append(len(kept))
            pairs=np.asarray(pairs,"int64"); exact=[]; unknown=[]
            for ix in batches(len(pairs),4096,False):
                p=torch.as_tensor(pairs[ix],device=device); x=cs.expand(len(p),-1); z,r=residual_interaction_inputs(endpoint,genes,p,x); exact.append(-head(z,r)[:,0].cpu()); z,r=residual_interaction_inputs(endpoint,genes,p,torch.zeros_like(x)); unknown.append(-head(z,r)[:,0].cpu())
            labels=np.asarray(labels); exact=torch.cat(exact).numpy(); unknown=torch.cat(unknown).numpy(); row={"cell_line":cell,"fold":fold,"mapped_pairs":len(pairs),"context_model":model_ids[cell],**ranking(query_rows(labels,sizes,exact))}; row.update({f"unknown_{k}":v for k,v in ranking(query_rows(labels,sizes,unknown)).items()}); results.append(row); print(json.dumps(row),flush=True)
    Path(out_path).write_text(json.dumps(results,indent=2)); return results


@torch.no_grad()
def evaluate(model_path,feature_path,meta_path,split_files,out_path,outcome_path=None,d=384,latent=128,layers=6,text_files=None,perturb_path=None):
    sys.path.insert(0,"/root"); from world_model import SLPredict, encode_genes, batches
    device="cuda"; state=np.load(feature_path)["state"].astype("float32"); symbols=[r["symbol"] for r in csv.DictReader(open(meta_path))]; ids=dict(zip(symbols,range(len(symbols)))); sd=torch.load(model_path,map_location="cpu",weights_only=True); context_dim=sd["context_proj.weight"].shape[1] if "context_proj.weight" in sd else 0; model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],context_dim).to(device); model.load_state_dict(sd); model.eval(); genes=torch.as_tensor(encode_genes(model,state,device),device=device); decoder_state=torch.load(perturb_path,map_location="cpu",weights_only=True) if perturb_path else None; decoder=torch.nn.Linear(latent,decoder_state["weight"].shape[0]).to(device) if decoder_state else None; decoder.load_state_dict(decoder_state) if decoder is not None else None; decoder.eval() if decoder is not None else None; outcome=np.load(outcome_path,allow_pickle=True) if outcome_path else None; contexts=outcome["contexts"] if outcome is not None else []; observed=set(outcome["context"].tolist()) if outcome is not None else set(); results=[]
    for file_id,split_file in enumerate(split_files):
        cell=Path(split_file).name.split("_scenario",1)[0]; text=text_embeddings(text_files[file_id]) if text_files else None; source_ids=range(4) if cell.upper()=="K562" else ()
        context_ids=[i for i,x in enumerate(contexts) if i in observed and str(x).split("|")[-1].upper()==cell.upper()]; semantic_ids=[i for i,x in enumerate(contexts) if str(x).split("|")[-1].upper()==cell.upper() and "context_state" in outcome.files and np.linalg.norm(outcome["context_state"][i])>0] if outcome is not None else []; semantic_state=torch.as_tensor(outcome["context_state"][semantic_ids].mean(0,dtype=np.float32),device=device) if semantic_ids and model.context_proj is not None else None; semantic_gene=torch.as_tensor(outcome["gene_context_state"][semantic_ids].mean(0,dtype=np.float32),device=device) if semantic_ids and outcome is not None and "gene_context_state" in outcome.files else None
        def predict(group):
            if isinstance(group,list):
                grouped={}
                for query,partner,score,label in group: grouped.setdefault(query,[]).append((partner,score,label))
                group=grouped
            pairs=[]; labels=[]; sizes=[]; text_scores=[]
            for query,partners in group.items():
                kept=[(ids[query],ids[p],label=="SL",p) for p,_,label in partners if query in ids and p in ids]; pairs.extend((a,b) for a,b,_,_ in kept); labels.extend(y for _,_,y,_ in kept); sizes.append(len(kept)); text_scores.extend(text[query]@text[p] for *_,p in kept) if text is not None else None
            pairs=np.asarray(pairs,"int64"); scores=[]; matched=[]; semantic=[]; dynamics=[]; source_dynamics=[]; sequential=[]; source_sequential=[]; surprise=[]; order_effect=[]; fitness=[]; features=[]
            for ix in batches(len(pairs),4096,False):
                p=torch.as_tensor(pairs[ix],device=device); a,b=genes[p[:,0]],genes[p[:,1]]; joint,joint_logsd=model.transition(a,b); scores.append(model.outcome(joint)[:,1].cpu()); raw=[(a-b).abs(),a*b,model.relation_score(a,b,joint),model.outcome(joint)]; matched.append(torch.stack([model.outcome(model.transition(a,b,context=torch.full((len(p),),c,device=device))[0])[:,1] for c in context_ids]).mean(0).cpu() if context_ids else scores[-1]); cs=semantic_state.expand(len(p),-1) if semantic_state is not None else None
                if cs is not None and semantic_gene is not None: ga,gb=semantic_gene[p[:,0]],semantic_gene[p[:,1]]; direct=(ga+gb,(ga-gb).abs(),ga*gb,torch.maximum(ga,gb)); cs=torch.cat((cs,*direct),1); raw.extend(direct)
                if decoder is not None:
                    la,_=model.transition(a); lb,_=model.transition(b); sab,sab_logsd=model.transition(b,state=la); sba,sba_logsd=model.transition(a,state=lb); pj,pa,pb,pab,pba=map(decoder,(joint,la,lb,sab,sba)); stop=32 if decoder.out_features==33 else decoder.out_features; residual=pj[:,:stop]-(pab[:,:stop]+pba[:,:stop])/2; variance=joint_logsd.mul(2).exp()@decoder.weight[:stop].square().T+(sab_logsd.mul(2).exp()+sba_logsd.mul(2).exp())@decoder.weight[:stop].square().T/4; dynamics.append(torch.linalg.vector_norm(pj[:,:stop]-pa[:,:stop]-pb[:,:stop],dim=1).cpu()); sequential.append(torch.linalg.vector_norm(residual,dim=1).cpu()); surprise.append(torch.sqrt((residual.square()/(variance+1e-6)).sum(1)).cpu()); order_effect.append(torch.linalg.vector_norm(pab[:,:stop]-pba[:,:stop],dim=1).cpu()); fitness.append((pa[:,32]+pb[:,32]-pj[:,32]).cpu() if decoder.out_features==33 else dynamics[-1])
                    contextual=[]; contextual_sequential=[]
                    for c in source_ids:
                        ctx=torch.full((len(p),),c,device=device); lj=model.transition(a,b,context=ctx)[0]; lca=model.transition(a,context=ctx)[0]; lcb=model.transition(b,context=ctx)[0]; cj,ca,cb=map(decoder,(lj,lca,lcb)); cab=decoder(model.transition(b,state=lca,context=ctx)[0]); cba=decoder(model.transition(a,state=lcb,context=ctx)[0]); contextual.append(torch.linalg.vector_norm(cj[:,:stop]-ca[:,:stop]-cb[:,:stop],dim=1)); contextual_sequential.append(torch.linalg.vector_norm(cj[:,:stop]-(cab[:,:stop]+cba[:,:stop])/2,dim=1))
                    source_dynamics.append(torch.stack(contextual).mean(0).cpu() if contextual else dynamics[-1]); source_sequential.append(torch.stack(contextual_sequential).mean(0).cpu() if contextual_sequential else sequential[-1])
                else: dynamics.append(scores[-1]); source_dynamics.append(scores[-1]); sequential.append(scores[-1]); source_sequential.append(scores[-1]); surprise.append(scores[-1]); order_effect.append(scores[-1]); fitness.append(scores[-1])
                raw.extend((dynamics[-1].to(device)[:,None],fitness[-1].to(device)[:,None])); features.append(torch.cat(raw,1).cpu())
                semantic.append(model.outcome(model.transition(a,b,context_state=cs)[0])[:,1].cpu() if cs is not None else scores[-1])
            arrays=[torch.cat(x).numpy() for x in (scores,matched,semantic)]; dynamics=torch.cat(dynamics).numpy(); source_dynamics=torch.cat(source_dynamics).numpy(); sequential=torch.cat(sequential).numpy(); source_sequential=torch.cat(source_sequential).numpy(); surprise=torch.cat(surprise).numpy(); order_effect=torch.cat(order_effect).numpy(); fitness=torch.cat(fitness).numpy(); feature=np.column_stack((torch.cat(features).numpy(),*arrays,np.asarray(text_scores),dynamics,source_dynamics,sequential,source_sequential,surprise,order_effect,fitness)); return pairs,np.asarray(labels),sizes,*arrays,np.asarray(text_scores),dynamics,source_dynamics,sequential,source_sequential,surprise,order_effect,fitness,feature
        for fold,(train,valid,test) in enumerate(pickle.load(open(split_file,"rb"))):
            tr=predict(train); va=predict(valid); te=predict(test); pairs,labels,sizes,scores,matched,semantic,text_scores,dynamics,source_dynamics,sequential,source_sequential,surprise,order_effect,fitness,_=te; weights=tune_fusion(va[1],va[2],(va[3],va[5],va[6])); validated=rank_fusion((scores,semantic,text_scores),sizes,weights); fused=rank_fusion((scores,semantic,text_scores),sizes,(.5,0.,.5)); state_text=rank_fusion((dynamics,text_scores),sizes,(.5,.5)); probe=fit_probe(tr,va,te)
            row={"cell_line":cell,"fold":fold,"mapped_pairs":len(pairs),"matched_contexts":len(context_ids),"semantic_contexts":len(semantic_ids),"source_contexts":len(source_ids),"validated_weights":weights,**ranking(query_rows(labels,sizes,scores))}; row.update({f"matched_{k}":v for k,v in ranking(query_rows(labels,sizes,matched)).items()}); row.update({f"semantic_{k}":v for k,v in ranking(query_rows(labels,sizes,semantic)).items()}); row.update({f"text_{k}":v for k,v in ranking(query_rows(labels,sizes,text_scores)).items()}); row.update({f"state_{k}":v for k,v in ranking(query_rows(labels,sizes,dynamics)).items()}); row.update({f"source_state_{k}":v for k,v in ranking(query_rows(labels,sizes,source_dynamics)).items()}); row.update({f"sequential_{k}":v for k,v in ranking(query_rows(labels,sizes,sequential)).items()}); row.update({f"source_sequential_{k}":v for k,v in ranking(query_rows(labels,sizes,source_sequential)).items()}); row.update({f"surprise_{k}":v for k,v in ranking(query_rows(labels,sizes,surprise)).items()}); row.update({f"order_{k}":v for k,v in ranking(query_rows(labels,sizes,order_effect)).items()}); row.update({f"state_text_{k}":v for k,v in ranking(query_rows(labels,sizes,state_text)).items()}); row.update({f"fitness_{k}":v for k,v in ranking(query_rows(labels,sizes,fitness)).items()}); row.update({f"fused_{k}":v for k,v in ranking(query_rows(labels,sizes,fused)).items()}); row.update({f"validated_{k}":v for k,v in ranking(query_rows(labels,sizes,validated)).items()}); row.update({f"probe_{k}":v for k,v in ranking(query_rows(labels,sizes,probe)).items()}); results.append(row); print(json.dumps(row),flush=True)
    Path(out_path).write_text(json.dumps(results,indent=2)); return results
