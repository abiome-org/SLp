from pathlib import Path
import json,sys
import numpy as np,torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; sys.path.insert(0,str(Path(__file__).parent))
from depmap_crossmodal_relation import pairs
from tcga_relation_decoder import Head
from world_model import batches,dependency_loss,encode_genes,load_residual_endpoint,residual_endpoint_score


class Adapter(torch.nn.Module):
    def __init__(self,dim=64,latent=128):
        super().__init__(); self.proj=torch.nn.Linear(dim,latent,bias=False); torch.nn.init.zeros_(self.proj.weight)
    def forward(self,feature,base): return base+.05*base.square().mean(1,keepdim=True).sqrt()*torch.tanh(self.proj(feature))


def fit(epochs=12,epoch_pairs=1000000,batch=8192):
    torch.manual_seed(799); np.random.seed(799); device="cuda" if torch.cuda.is_available() else "cpu"; model_dir=OUT/"native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); endpoint=load_residual_endpoint(model_dir/"world_model.pt",state.shape[1],device); world=endpoint.world.eval(); base=torch.as_tensor(encode_genes(world,state,device),device=device); source=np.load(OUT/"depmap_expression_silencing.npz"); supported=source["genes"].astype("int64"); half0=source["half0"]; half1=source["half1"]; train=np.flatnonzero(supported%5!=0); valid=np.flatnonzero(supported%5==0); rng=np.random.default_rng(799); anchors=np.sort(rng.choice(train,256,replace=False)); profile=half0[:,anchors].astype("float32"); pca=PCA(64,whiten=True,svd_solver="randomized",random_state=799).fit(profile[train]); feature=np.zeros((len(state),64),"float32"); feature[supported]=pca.transform(profile).astype("float32"); feature=torch.as_tensor(feature,device=device); adapter=Adapter().to(device); head=Head(128).to(device); opt=torch.optim.AdamW((*adapter.parameters(),*head.parameters()),3e-4,weight_decay=1e-3); si,sj=pairs(rng,len(train),min(2000000,len(train)*(len(train)-1)//2)); stats=half0[train[si],train[sj]].astype("float32"); mean=float(stats.mean()); scale=float(stats.std()); vi,vj=np.triu_indices(len(valid),1); history=[]; saved=[]
    @torch.no_grad()
    def evaluate(epoch):
        action=adapter(feature[supported],base[supported]); pred=[]; head.eval()
        for lo in range(0,len(vi),batch):
            a=torch.as_tensor(valid[vi[lo:lo+batch]],device=device); b=torch.as_tensor(valid[vj[lo:lo+batch]],device=device); pred.append(head(action,a,b).cpu())
        pred=torch.cat(pred).numpy(); truth=(half1[valid[vi],valid[vj]].astype("float32")-mean)/scale; row={"epoch":epoch,"validation_pairs":len(vi),"huber":float(torch.nn.functional.huber_loss(torch.from_numpy(pred),torch.from_numpy(truth))),"pearson":float(np.corrcoef(truth,pred)[0,1]) if pred.std()>0 else 0.,"spearman":float(spearmanr(truth,pred).statistic) if pred.std()>0 else 0.}; history.append(row); saved.append({"adapter":{k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()},"head":{k:v.detach().cpu().clone() for k,v in head.state_dict().items()}}); print(json.dumps(row),flush=True)
    evaluate(0)
    for epoch in range(epochs):
        rng=np.random.default_rng(799+epoch); a=rng.choice(train,epoch_pairs); b=rng.choice(train,epoch_pairs); same=a==b
        while same.any(): b[same]=rng.choice(train,same.sum()); same=a==b
        adapter.train(); head.train()
        for lo in range(0,epoch_pairs,batch):
            aa=torch.as_tensor(a[lo:lo+batch],device=device); bb=torch.as_tensor(b[lo:lo+batch],device=device); action=adapter(feature[supported],base[supported]); y=torch.as_tensor((half0[a[lo:lo+batch],b[lo:lo+batch]].astype("float32")-mean)/scale,device=device); loss=torch.nn.functional.huber_loss(head(action,aa,bb),y); opt.zero_grad(); loss.backward(); opt.step()
        evaluate(epoch+1)
    selected=max(range(len(history)),key=lambda q:(history[q]["pearson"],history[q]["spearman"])); adapter.load_state_dict(saved[selected]["adapter"]); head.load_state_dict(saved[selected]["head"]); adapter.eval(); head.eval(); chosen=history[selected];
    with torch.no_grad(): adapted=adapter(feature,base)
    pack=np.load(OUT/"features_spectral_safe.npz"); rp=torch.as_tensor(pack["pairs"].astype("int64"),device=device); rt=torch.as_tensor(pack["relations"].astype("float32"),device=device); rv=torch.nonzero((rp[:,0]*1000003+rp[:,1])%20==0).squeeze(1)
    @torch.no_grad()
    def relation(action):
        total=0.
        for at in batches(len(rv),4096,False):
            ix=rv[at]; p=rp[ix]; a,b=action[p[:,0]],action[p[:,1]]; total+=torch.nn.functional.smooth_l1_loss(world.relation_score(a,b,world.transition(a,b)[0]),rt[ix],reduction="sum").item()
        return total/len(rv)/rt.shape[1]
    perturb=np.load(OUT/"perturbseq_world_v3_nested96.npz"); dep=np.load(OUT/"basal_context.npz"); cells=np.flatnonzero(dep["train_cell"]&(np.arange(len(dep["train_cell"]))%10==0)); gene=np.flatnonzero(dep["train_gene"]); cs=torch.as_tensor(dep["cell_state"].astype("float32"),device=device); target=torch.as_tensor(dep["dependency"].astype("float32"),device=device); baseline={"relation":relation(base),"dependency":dependency_loss(world,base,cells,gene,cs,target,dep["dependency_known"],device,799),"legacy":sum(residual_endpoint_score(endpoint,base,perturb,device,0,x) for x in (False,True)),"residual":sum(residual_endpoint_score(endpoint,base,perturb,device,1,x) for x in (False,True))}; candidate={"relation":relation(adapted),"dependency":dependency_loss(world,adapted,cells,gene,cs,target,dep["dependency_known"],device,799),"legacy":sum(residual_endpoint_score(endpoint,adapted,perturb,device,0,x) for x in (False,True)),"residual":sum(residual_endpoint_score(endpoint,adapted,perturb,device,1,x) for x in (False,True))}; relation_ok=chosen["pearson"]>=.15 and chosen["spearman"]>=.15; preservation=all(candidate[k]<=1.01*baseline[k] for k in baseline); delta=adapted-base; rms_ratio=float(delta.square().mean().sqrt()/base.square().mean().sqrt()); max_coordinate_ratio=float((delta.abs()/base.square().mean(1,keepdim=True).sqrt().clamp_min(1e-8)).max()); result={"schema":"sl-predict-silencing-action-adapter-v1","parameters":sum(p.numel() for p in adapter.parameters())+sum(p.numel() for p in head.parameters()),"adapter_parameters":sum(p.numel() for p in adapter.parameters()),"profile_anchors":len(anchors),"profile_dimensions":64,"pca_fit_genes":len(train),"pca_explained_variance":float(pca.explained_variance_ratio_.sum()),"selected":chosen,"baseline_state":baseline,"adapted_state":candidate,"action_rms_ratio":rms_ratio,"maximum_coordinate_to_base_rms_ratio":max_coordinate_ratio,"relation_advanced":bool(relation_ok),"preserved":bool(preservation),"advanced":bool(relation_ok and preservation),"world_parameters_changed":0,"double_perturbation_data_used_for_fitting":False,"sl_labels_used":False,"history":history}; torch.save({"adapter":adapter.state_dict(),"head":head.state_dict(),"profile_feature":feature.cpu(),"anchors":torch.as_tensor(supported[anchors]),"target_mean":mean,"target_scale":scale},model_dir/"silencing_action_adapter.pt"); (model_dir/"silencing_action_adapter_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2)); return result


if __name__=="__main__": fit()
