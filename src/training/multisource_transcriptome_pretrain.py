import json,sys
from pathlib import Path
import numpy as np,torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; RUN=OUT/"native_spectral_safe_scaled_d768_z256_l8_p12_single_only_d3_t10_r3"; sys.path.insert(0,str(Path(__file__).parent))
from full_transcriptome_reinforce import endpoint_metrics,preservation
from world_model import SLPredict,SourceEndpoint,batches,encode_genes

def grouped(data,role):
    rows=np.flatnonzero(data["role"]==role); genes=np.unique(data["gene"][rows]); target=np.stack([data["target_med"][rows[data["gene"][rows]==g]].mean(0) for g in genes]).astype("float32"); return genes.astype("int64"),target

def main(epochs=6):
    torch.manual_seed(107); np.random.seed(107); device="cuda" if torch.cuda.is_available() else "cpu"; full=np.load(OUT/"full_transcriptome_single_endpoint32.npz"); gse=np.load(OUT/"gse337988_moi_state.npz",allow_pickle=True); gtrain_gene,gtrain_target=grouped(gse,0); gvalid_gene,gvalid_target=grouped(gse,1); state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); sd=torch.load(RUN/"world_model.pt",map_location="cpu",weights_only=True)
    def load_world():
        w=SLPredict(768,256,8,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],sd["context_proj.weight"].shape[1]).to(device); w.load_state_dict(sd); return w.eval()
    world=load_world().requires_grad_(False); baseline_world=load_world().requires_grad_(False); full_head=SourceEndpoint(5,256,32).to(device); full_head.load_state_dict(torch.load(RUN/"full_transcriptome_single_endpoint32.pt",map_location="cpu",weights_only=True)); full_head.eval().requires_grad_(False); gse_head=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256),nn.GELU(),nn.Linear(256,64)).to(device); gse_head.load_state_dict(torch.load(RUN/"gse337988_single_endpoint.pt",map_location="cpu",weights_only=True)); gse_head.eval().requires_grad_(False); genes=torch.as_tensor(encode_genes(world,state,device),device=device)
    for module in (world.state_up,world.dist,world.context_proj):module.requires_grad_(True)
    for parameter in (world.basal,world.context,world.time):parameter.requires_grad_(True)
    trainable=[p for p in world.parameters() if p.requires_grad]; assert sum(p.numel() for p in trainable)==691712; gene=torch.as_tensor(full["gene"].astype("int64"),device=device); source=torch.as_tensor(full["source"].astype("int64"),device=device); context=torch.as_tensor(full["context_state"].astype("float32"),device=device); target=torch.as_tensor(full["target"].astype("float32"),device=device); basal=np.load(OUT/"basal_context.npz"); dld1=torch.as_tensor(basal["cell_state"][np.flatnonzero(basal["model_ids"].astype(str)=="ACH-001061")[0]],device=device); gt=torch.as_tensor(gtrain_target,device=device); full_train=np.flatnonzero(full["role"]==0); record_source=np.concatenate((full["source"][full_train],np.full(len(gtrain_gene),5))); record_kind=np.concatenate((np.zeros(len(full_train),"int8"),np.ones(len(gtrain_gene),"int8"))); record_row=np.concatenate((full_train,np.arange(len(gtrain_gene)))); count=np.bincount(record_source); weight=1/count[record_source]; weight/=weight.sum(); opt=torch.optim.AdamW(trainable,1e-5,weight_decay=1e-3); history=[]; saved=[]
    @torch.no_grad()
    def gse_metric(exact):
        pred=[]
        for at in batches(len(gvalid_gene),2048,False):
            ix=torch.as_tensor(gvalid_gene[at],device=device); cs=dld1[None].expand(len(at),-1) if exact else None; pred.append(gse_head(world.transition(genes[ix],context_state=cs)[0]).cpu())
        pred=torch.cat(pred); truth=torch.as_tensor(gvalid_target); return {"huber":float(nn.functional.huber_loss(pred,truth)),"cosine":float(nn.functional.cosine_similarity(pred,truth).mean()),"rows":len(truth)}
    @torch.no_grad()
    def evaluate(epoch):
        f=endpoint_metrics(world,full_head,genes,full,device); g={name:gse_metric(exact) for name,exact in (("unknown",False),("exact",True))}; macro={}
        for c in ("unknown","exact"):macro[c]={"source_macro_huber":(5*f[c]["source_macro_huber"]+g[c]["huber"])/6,"source_macro_cosine":(5*f[c]["source_macro_cosine"]+g[c]["cosine"])/6}
        row={"epoch":epoch,"full":f,"gse":g,"macro":macro,"selection_loss":macro["unknown"]["source_macro_huber"]+macro["exact"]["source_macro_huber"]}; history.append(row); saved.append({n:p.detach().cpu().clone() for n,p in world.named_parameters() if p.requires_grad}); print(json.dumps(row),flush=True)
    evaluate(0)
    for epoch in range(epochs):
        chosen=np.random.default_rng(107+epoch).choice(len(record_row),len(record_row),replace=True,p=weight); total=0.
        for at in batches(len(chosen),256,False):
            take=chosen[at]; kind=record_kind[take]; rows=record_row[take]; losses=[]; trusts=[]; sizes=[]
            fk=np.flatnonzero(kind==0)
            if len(fk):
                ix=rows[fk]; cs=context[source[ix]]; cs=torch.where((torch.rand(len(ix),device=device)<.5)[:,None],torch.zeros_like(cs),cs); mu,_=world.transition(genes[gene[ix]],context_state=cs)
                with torch.no_grad():base,_=baseline_world.transition(genes[gene[ix]],context_state=cs)
                pred=full_head(mu,source[ix]); losses.append(nn.functional.huber_loss(pred,target[ix])+.05*(1-nn.functional.cosine_similarity(pred,target[ix]).mean())); trusts.append(nn.functional.mse_loss(mu,base)); sizes.append(len(ix))
            gk=np.flatnonzero(kind==1)
            if len(gk):
                ix=rows[gk]; ids=torch.as_tensor(gtrain_gene[ix],device=device); cs=dld1[None].expand(len(ix),-1); cs=torch.where((torch.rand(len(ix),device=device)<.5)[:,None],torch.zeros_like(cs),cs); mu,_=world.transition(genes[ids],context_state=cs)
                with torch.no_grad():base,_=baseline_world.transition(genes[ids],context_state=cs)
                pred=gse_head(mu); losses.append(nn.functional.huber_loss(pred,gt[ix])+.05*(1-nn.functional.cosine_similarity(pred,gt[ix]).mean())); trusts.append(nn.functional.mse_loss(mu,base)); sizes.append(len(ix))
            n=sum(sizes); loss=sum(s*x for s,x in zip(sizes,losses))/n+10*sum(s*x for s,x in zip(sizes,trusts))/n; opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(trainable,1.); opt.step(); total+=loss.item()*n
        evaluate(epoch+1); history[-1]["training_loss"]=total/len(chosen)
    selected=min(range(len(history)),key=lambda i:history[i]["selection_loss"]); chosen=history[selected]; base=history[0]
    with torch.no_grad():
        for name,p in world.named_parameters():
            if name in saved[selected]:p.copy_(saved[selected][name])
    baseline_state=preservation(baseline_world,genes,state,device); candidate_state=preservation(world,genes,state,device); macro_ok=selected>0 and all(chosen["macro"][c]["source_macro_huber"]<=.99*base["macro"][c]["source_macro_huber"] and chosen["macro"][c]["source_macro_cosine"]>=base["macro"][c]["source_macro_cosine"] for c in ("unknown","exact")); family_ok=all(chosen["full"][c]["source_macro_huber"]<=.995*base["full"][c]["source_macro_huber"] and chosen["full"][c]["source_macro_cosine"]>=base["full"][c]["source_macro_cosine"] and chosen["gse"][c]["huber"]<=.995*base["gse"][c]["huber"] and chosen["gse"][c]["cosine"]>=base["gse"][c]["cosine"] for c in ("unknown","exact")); sources_ok=all(x["cosine"]>0 and x["cosine"]>=y["cosine"]-.02 for c in ("unknown","exact") for x,y in zip(chosen["full"][c]["sources"],base["full"][c]["sources"])); preserved=all(candidate_state[k]<=1.01*baseline_state[k] for k in baseline_state); advanced=macro_ok and family_ok and sources_ok and preserved; torch.save(world.state_dict(),RUN/"world_model_multisource_transcriptome_pretrained.pt"); result={"schema":"sl-predict-multisource-transcriptome-pretrain-v1","trainable_parameters":sum(p.numel() for p in trainable),"fitting_rows":len(record_row),"selected":chosen,"baseline":base,"baseline_state":baseline_state,"candidate_state":candidate_state,"macro_advanced":bool(macro_ok),"families_advanced":bool(family_ok),"sources_preserved":bool(sources_ok),"state_preserved":bool(preserved),"advanced":bool(advanced),"double_perturbation_data_used":False,"sl_labels_used":False,"history":history}; (RUN/"multisource_transcriptome_pretrain_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2))

if __name__=="__main__":main()
