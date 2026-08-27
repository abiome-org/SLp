import json,sys
from pathlib import Path

import numpy as np,torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; RUN=OUT/"native_spectral_safe_scaled_d768_z256_l8_p12_single_only_d3_t10_r3"; sys.path.insert(0,str(Path(__file__).parent))
from full_transcriptome_reinforce import endpoint_metrics,preservation
from world_model import SLPredict,SourceEndpoint,batches,encode_genes

class TransitionWorld(SLPredict):
    """Experimental transition-only capacity without changing the v1 contract."""
    def __init__(self,*args):
        super().__init__(*args); d=self.config.d; self.transition_adapter=nn.ModuleList([nn.TransformerEncoderLayer(d,6,4*d,.1,batch_first=True,norm_first=True,activation="gelu")])
        block=self.transition_adapter[0]; nn.init.zeros_(block.self_attn.out_proj.weight); nn.init.zeros_(block.self_attn.out_proj.bias); nn.init.zeros_(block.linear2.weight); nn.init.zeros_(block.linear2.bias)
    def transition(self,action,second=None,state=None,context=None,context_state=None):
        b=len(action); start=self.basal[None].expand(b,-1) if state is None else self.state_up(state); seq=[start+self.token_type.weight[2],self.state_up(action)+self.token_type.weight[3]]
        if second is not None:seq.append(self.state_up(second)+self.token_type.weight[3])
        if context_state is not None:ctx=self.context[None]+self.context_proj(context_state)
        elif context is None:ctx=self.context[None].expand(b,-1)
        else:ctx=torch.where((context>=0)[:,None],self.cell(context.clamp_min(0)),self.context[None])
        h=self.core(torch.stack(seq+[self.time[None].expand(b,-1)+self.token_type.weight[4],ctx],1)); h=self.transition_adapter[0](h); mu,logsd=self.dist(h[:,0]).chunk(2,1); return mu,logsd.clamp(-5,2)

def grouped(data,role):
    rows=np.flatnonzero(data["role"]==role); genes=np.unique(data["gene"][rows]); target=np.stack([data["target_med"][rows[data["gene"][rows]==g]].mean(0) for g in genes]).astype("float32"); return genes.astype("int64"),target

def main(epochs=8):
    torch.manual_seed(113); np.random.seed(113); device="cuda" if torch.cuda.is_available() else "cpu"; full=np.load(OUT/"full_transcriptome_single_endpoint32.npz"); gse=np.load(OUT/"gse337988_moi_state.npz",allow_pickle=True); gtrain_gene,gtrain_target=grouped(gse,0); gvalid_gene,gvalid_target=grouped(gse,1); state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); sd=torch.load(RUN/"world_model.pt",map_location="cpu",weights_only=True); args=(768,256,8,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],sd["context_proj.weight"].shape[1])
    baseline=SLPredict(*args).to(device); baseline.load_state_dict(sd); baseline.eval().requires_grad_(False); world=TransitionWorld(*args).to(device); world.load_state_dict(sd,strict=False); world.eval().requires_grad_(False); world.transition_adapter.requires_grad_(True); trainable=list(world.transition_adapter.parameters()); assert sum(p.numel() for p in trainable)==7087872 and world.count()==66824416
    full_head=SourceEndpoint(5,256,32).to(device); full_head.load_state_dict(torch.load(RUN/"full_transcriptome_single_endpoint32.pt",map_location="cpu",weights_only=True)); full_head.eval().requires_grad_(False); gse_head=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256),nn.GELU(),nn.Linear(256,64)).to(device); gse_head.load_state_dict(torch.load(RUN/"gse337988_single_endpoint.pt",map_location="cpu",weights_only=True)); gse_head.eval().requires_grad_(False); genes=torch.as_tensor(encode_genes(baseline,state,device),device=device)
    gene=torch.as_tensor(full["gene"].astype("int64"),device=device); source=torch.as_tensor(full["source"].astype("int64"),device=device); context=torch.as_tensor(full["context_state"].astype("float32"),device=device); target=torch.as_tensor(full["target"].astype("float32"),device=device); gt=torch.as_tensor(gtrain_target,device=device); basal=np.load(OUT/"basal_context.npz"); dld1=torch.as_tensor(basal["cell_state"][np.flatnonzero(basal["model_ids"].astype(str)=="ACH-001061")[0]],device=device); full_train=np.flatnonzero(full["role"]==0); record_source=np.r_[full["source"][full_train],np.full(len(gtrain_gene),5)]; record_kind=np.r_[np.zeros(len(full_train),"int8"),np.ones(len(gtrain_gene),"int8")]; record_row=np.r_[full_train,np.arange(len(gtrain_gene))]; weight=1/np.bincount(record_source)[record_source]; weight/=weight.sum(); opt=torch.optim.AdamW(trainable,3e-5,weight_decay=1e-3); history=[]
    @torch.no_grad()
    def gse_metric(exact):
        pred=[]
        for at in batches(len(gvalid_gene),2048,False):
            ix=torch.as_tensor(gvalid_gene[at],device=device); cs=dld1[None].expand(len(at),-1) if exact else None; pred.append(gse_head(world.transition(genes[ix],context_state=cs)[0]).cpu())
        pred=torch.cat(pred); truth=torch.as_tensor(gvalid_target); return {"huber":float(nn.functional.huber_loss(pred,truth)),"cosine":float(nn.functional.cosine_similarity(pred,truth).mean()),"rows":len(truth)}
    @torch.no_grad()
    def evaluate(epoch):
        f=endpoint_metrics(world,full_head,genes,full,device); g={n:gse_metric(x) for n,x in (("unknown",False),("exact",True))}; macro={c:{"source_macro_huber":(5*f[c]["source_macro_huber"]+g[c]["huber"])/6,"source_macro_cosine":(5*f[c]["source_macro_cosine"]+g[c]["cosine"])/6} for c in ("unknown","exact")}; row={"epoch":epoch,"full":f,"gse":g,"macro":macro,"selection_loss":sum(macro[c]["source_macro_huber"] for c in ("unknown","exact"))}; history.append(row); print(json.dumps(row),flush=True); return row
    base=evaluate(0); best=base; best_state={k:v.detach().cpu().clone() for k,v in world.transition_adapter.state_dict().items()}
    for epoch in range(epochs):
        chosen=np.random.default_rng(113+epoch).choice(len(record_row),len(record_row),replace=True,p=weight); total=0.
        for at in batches(len(chosen),128,False):
            take=chosen[at]; kind=record_kind[take]; rows=record_row[take]; losses=[]; trusts=[]; sizes=[]
            fk=np.flatnonzero(kind==0)
            if len(fk):
                ix=rows[fk]; cs=context[source[ix]]; cs=torch.where((torch.rand(len(ix),device=device)<.5)[:,None],torch.zeros_like(cs),cs); mu,_=world.transition(genes[gene[ix]],context_state=cs)
                with torch.no_grad():ref,_=baseline.transition(genes[gene[ix]],context_state=cs)
                pred=full_head(mu,source[ix]); losses.append(nn.functional.huber_loss(pred,target[ix])+.05*(1-nn.functional.cosine_similarity(pred,target[ix]).mean())); trusts.append(nn.functional.mse_loss(mu,ref)); sizes.append(len(ix))
            gk=np.flatnonzero(kind==1)
            if len(gk):
                ix=rows[gk]; ids=torch.as_tensor(gtrain_gene[ix],device=device); cs=dld1[None].expand(len(ix),-1); cs=torch.where((torch.rand(len(ix),device=device)<.5)[:,None],torch.zeros_like(cs),cs); mu,_=world.transition(genes[ids],context_state=cs)
                with torch.no_grad():ref,_=baseline.transition(genes[ids],context_state=cs)
                pred=gse_head(mu); losses.append(nn.functional.huber_loss(pred,gt[ix])+.05*(1-nn.functional.cosine_similarity(pred,gt[ix]).mean())); trusts.append(nn.functional.mse_loss(mu,ref)); sizes.append(len(ix))
            n=sum(sizes); loss=sum(s*x for s,x in zip(sizes,losses))/n+.5*sum(s*x for s,x in zip(sizes,trusts))/n; opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(trainable,1.); opt.step(); total+=loss.item()*n
        row=evaluate(epoch+1); row["training_loss"]=total/len(chosen)
        if row["selection_loss"]<best["selection_loss"]:best=row; best_state={k:v.detach().cpu().clone() for k,v in world.transition_adapter.state_dict().items()}
    world.transition_adapter.load_state_dict(best_state); baseline_state=preservation(baseline,genes,state,device); candidate_state=preservation(world,genes,state,device); macro_ok=best["epoch"]>0 and all(best["macro"][c]["source_macro_huber"]<=.99*base["macro"][c]["source_macro_huber"] and best["macro"][c]["source_macro_cosine"]>=base["macro"][c]["source_macro_cosine"] for c in ("unknown","exact")); family_ok=all(best["full"][c]["source_macro_huber"]<=.995*base["full"][c]["source_macro_huber"] and best["full"][c]["source_macro_cosine"]>=base["full"][c]["source_macro_cosine"] and best["gse"][c]["huber"]<=.995*base["gse"][c]["huber"] and best["gse"][c]["cosine"]>=base["gse"][c]["cosine"] for c in ("unknown","exact")); sources_ok=all(x["cosine"]>0 and x["cosine"]>=y["cosine"]-.02 for c in ("unknown","exact") for x,y in zip(best["full"][c]["sources"],base["full"][c]["sources"])); preserved=all(candidate_state[k]<=1.01*baseline_state[k] for k in baseline_state); advanced=macro_ok and family_ok and sources_ok and preserved; torch.save(best_state,RUN/"transition_transformer_adapter.pt"); result={"schema":"sl-predict-transition-transformer-adapter-v1","base_parameters":59736544,"trainable_parameters":sum(p.numel() for p in trainable),"total_parameters":world.count(),"fitting_records":len(record_row),"selected":best,"baseline":base,"baseline_state":baseline_state,"candidate_state":candidate_state,"macro_advanced":bool(macro_ok),"families_advanced":bool(family_ok),"sources_preserved":bool(sources_ok),"state_preserved":bool(preserved),"advanced":bool(advanced),"double_perturbation_data_used":False,"sl_labels_used":False,"history":history}; (RUN/"transition_transformer_adapter_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2))

if __name__=="__main__":main()
