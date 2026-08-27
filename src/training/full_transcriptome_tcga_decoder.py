import json,sys
from pathlib import Path
import numpy as np,torch
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; RUN=OUT/"native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; sys.path.insert(0,str(Path(__file__).parent))
from tcga_relation_decoder import Head
from world_model import encode_genes,load_residual_endpoint

def main(epochs=12,epoch_pairs=1_000_000,batch=8192):
    torch.manual_seed(106); np.random.seed(106); device="cuda" if torch.cuda.is_available() else "cpu"; data=np.load(OUT/"tcga_mutual_exclusivity.npz"); response=np.load(OUT/"full_transcriptome_codependency.npz"); supported=data["genes"].astype("int64"); i,j=np.triu_indices(len(supported),1); i=i.astype("uint16"); j=j.astype("uint16"); held=np.isin(supported,response["held"]); train=np.flatnonzero(~held[i]&~held[j]); valid=np.flatnonzero(held[i]&held[j]); assert held.sum()==126 and len(valid)==7875
    mean=float(data["half0"][train].mean()); scale=float(data["half0"][train].std()); target0=(data["half0"]-mean)/scale; target1=(data["half1"]-mean)/scale; state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); world=load_residual_endpoint(RUN/"world_model.pt",state.shape[1],device).world; action=encode_genes(world,state,device)[supported]
    profile=response["profile"][supported].astype("float32"); center=profile[~held].mean(0); spread=profile[~held].std(0).clip(.01); profile=(profile-center)/spread; base=torch.as_tensor(np.concatenate((action,np.zeros_like(profile)),1),device=device); augmented=torch.as_tensor(np.concatenate((action,profile),1),device=device); heads=torch.nn.ModuleDict({"baseline":Head(base.shape[1]),"candidate":Head(augmented.shape[1])}).to(device); opt=torch.optim.AdamW(heads.parameters(),3e-4,weight_decay=1e-3); history=[]; saved=[]
    @torch.no_grad()
    def evaluate(epoch):
        row={"epoch":epoch,"validation_pairs":len(valid)}; heads.eval(); truth=target1[valid]
        for name,x in (("baseline",base),("candidate",augmented)):
            pred=[]
            for lo in range(0,len(valid),batch):
                ix=valid[lo:lo+batch]; pred.append(heads[name](x,torch.as_tensor(i[ix].astype("int64"),device=device),torch.as_tensor(j[ix].astype("int64"),device=device)).cpu())
            pred=torch.cat(pred).numpy(); row[name]={"huber":float(torch.nn.functional.huber_loss(torch.from_numpy(pred),torch.from_numpy(truth))),"pearson":float(np.corrcoef(truth,pred)[0,1]) if pred.std()>0 else 0.,"spearman":float(spearmanr(truth,pred).statistic) if pred.std()>0 else 0.}
        history.append(row); saved.append({name:{k:v.detach().cpu().clone() for k,v in head.state_dict().items()} for name,head in heads.items()}); print(json.dumps(row),flush=True)
    evaluate(0)
    for epoch in range(epochs):
        chosen=np.random.default_rng(106+epoch).choice(train,epoch_pairs); heads.train()
        for lo in range(0,len(chosen),batch):
            ix=chosen[lo:lo+batch]; a=torch.as_tensor(i[ix].astype("int64"),device=device); b=torch.as_tensor(j[ix].astype("int64"),device=device); y=torch.as_tensor(target0[ix],device=device); loss=sum(torch.nn.functional.huber_loss(heads[name](x,a,b),y) for name,x in (("baseline",base),("candidate",augmented))); opt.zero_grad(); loss.backward(); opt.step()
        evaluate(epoch+1)
    chosen={name:max(range(len(history)),key=lambda q:(history[q][name]["pearson"],history[q][name]["spearman"])) for name in heads}; selected={name:history[chosen[name]] for name in heads}; b=selected["baseline"]["baseline"]; c=selected["candidate"]["candidate"]; advanced=c["pearson"]>=b["pearson"]+.01 and c["spearman"]>=b["spearman"]+.01 and c["huber"]<=b["huber"] and c["pearson"]>=.15 and c["spearman"]>=.15; heads["candidate"].load_state_dict(saved[chosen["candidate"]]["candidate"]); torch.save({"state_dict":heads["candidate"].state_dict(),"target_mean":mean,"target_scale":scale,"supported_genes":torch.as_tensor(supported),"profile_center":torch.as_tensor(center),"profile_scale":torch.as_tensor(spread)},RUN/"full_transcriptome_tcga_head.pt")
    result={"schema":"sl-predict-full-transcriptome-tcga-decoder-v1","parameters_per_head":sum(p.numel() for p in heads["candidate"].parameters()),"training_pairs":len(train),"held_genes":int(held.sum()),"validation_pairs":len(valid),"selected_baseline":selected["baseline"],"selected_candidate":selected["candidate"],"pearson_gain":c["pearson"]-b["pearson"],"spearman_gain":c["spearman"]-b["spearman"],"advanced":bool(advanced),"double_perturbation_data_used":False,"sl_labels_used":False,"history":history}; (RUN/"full_transcriptome_tcga_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2))

if __name__=="__main__":main()
