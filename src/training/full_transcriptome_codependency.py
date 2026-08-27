import json,sys
from pathlib import Path
import numpy as np,torch
from scipy.stats import pearsonr,spearmanr

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; RUN=OUT/"native_spectral_safe_scaled_d768_z256_l8_p12_single_only_d3_t10_r3"; sys.path.insert(0,str(Path(__file__).parent))
from world_model import SLPredict,SourceEndpoint,batches,encode_genes

def main():
    device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); data=np.load(OUT/"full_transcriptome_single_endpoint32.npz"); dep=np.load(OUT/"depmap_codependency.npz"); sd=torch.load(RUN/"world_model.pt",map_location="cpu",weights_only=True)
    world=SLPredict(768,256,8,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],sd["context_proj.weight"].shape[1]).to(device); world.load_state_dict(sd); world.eval().requires_grad_(False); head=SourceEndpoint(len(data["sources"]),256,32).to(device); head.load_state_dict(torch.load(RUN/"full_transcriptome_single_endpoint32.pt",map_location="cpu",weights_only=True)); head.eval().requires_grad_(False)
    held=np.unique(data["gene"][data["role"]==1]).astype("int64"); pos={g:i for i,g in enumerate(dep["genes"].astype("int64"))}; genes=np.asarray([g for g in held if g in pos],"int64"); dep_pos=np.asarray([pos[g] for g in genes],"int64"); all_genes=np.arange(len(state)); encoded=torch.as_tensor(encode_genes(world,state,device),device=device); context=torch.as_tensor(data["context_state"].astype("float32"),device=device); profiles=[]
    with torch.no_grad():
        for s in range(len(data["sources"])):
            pred=[]
            for at in batches(len(all_genes),2048,False):
                ix=torch.as_tensor(all_genes[at],device=device); source=torch.full((len(at),),s,device=device,dtype=torch.long); z=world.transition(encoded[ix],context_state=context[s].expand(len(at),-1))[0]; q=head(z,source); pred.append(torch.nn.functional.normalize(q,dim=1).cpu())
            profiles.append(torch.cat(pred).numpy())
    all_profile=np.concatenate(profiles,1); all_profile/=np.linalg.norm(all_profile,axis=1,keepdims=True).clip(1e-8); profile=all_profile[genes]; n=len(genes); total=n*(n-1)//2; count=min(2_000_000,total); k=np.random.default_rng(967).choice(total,count,replace=False); q=2*n-1; i=np.floor((q-np.sqrt(q*q-8*k))/2).astype("int64"); before=i*(2*n-i-1)//2; j=k-before+i+1; score=np.sum(profile[i]*profile[j],1); metrics=[]
    for half in ("half0","half1"):
        target=dep[half][dep_pos[i],dep_pos[j]].astype("float32"); metrics.append({"half":half,"pearson":float(pearsonr(score,target).statistic),"spearman":float(spearmanr(score,target).statistic)})
    result={"schema":"sl-predict-full-transcriptome-codependency-v1","eligible_held_genes":n,"sample_pairs":count,"sources":data["sources"].astype(str).tolist(),"metrics":metrics,"admitted":bool(all(m["pearson"]>=.15 and m["spearman"]>=.15 for m in metrics)),"double_perturbation_data_used":False,"sl_labels_used":False}; (OUT/"full_transcriptome_codependency.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    if result["admitted"]: np.savez_compressed(OUT/"full_transcriptome_codependency.npz",genes=all_genes.astype("int16"),profile=all_profile.astype("float16"),held=genes.astype("int16"))

if __name__=="__main__": main()
