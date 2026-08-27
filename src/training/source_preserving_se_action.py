import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; RUN=OUT/"native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; sys.path.insert(0,str(Path(__file__).parent))
from context_transition import Shift
from world_model import encode_genes,load_residual_endpoint

def cos(a,b): return float(np.mean(np.sum(a*b,1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-12)))
def tangent(delta,base):
    delta=delta-(delta*base).sum(1,keepdims=True)*base; norm=np.linalg.norm(delta,axis=1,keepdims=True); return delta*np.minimum(1,.25/(norm+1e-12))
def fold(source,gene): return int.from_bytes(hashlib.sha256(f"{source}:{gene}".encode()).digest()[:4],"big")%4

def main():
    np.random.seed(827); device="cuda" if torch.cuda.is_available() else "cpu"; data=np.load(OUT/"context_transition.npz"); feature=np.load(OUT/"se_replogle_gene_features.npz"); state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); endpoint=load_residual_endpoint(RUN/"world_model.pt",state.shape[1],device); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); shift=Shift().to(device); shift.load_state_dict(torch.load(RUN/"extended_unit_context_transition_head.pt",map_location="cpu",weights_only=True)); shift.eval().requires_grad_(False)
    gene=data["gene"].astype("int64"); source=data["source"].astype("int64"); cid=(source>=4).astype("int64"); raw=np.stack([feature["features"][g,32*c:32*(c+1)] for g,c in zip(gene,cid)]).astype("float32"); known=feature["known"][gene].astype(bool); target=np.asarray(data["target"],"float32"); target/=np.linalg.norm(target,axis=1,keepdims=True).clip(1e-12); context=torch.as_tensor(data["context_state"],device=device)
    with torch.no_grad(): base=torch.nn.functional.normalize(shift(genes[torch.as_tensor(gene,device=device)],context[torch.as_tensor(source,device=device)]),dim=1).cpu().numpy()
    gates=np.asarray([0,.125,.25,.5,.75,1.]); coefs=np.zeros((5,128,32),"float32"); means=np.zeros((5,32),"float32"); scales=np.ones((5,32),"float32"); selected=np.zeros(5,"float32"); rows=[]
    for sid,name in enumerate(data["sources"].astype(str)):
        tr=np.flatnonzero((data["role"]==0)&(source==sid)&known); va=np.flatnonzero((data["role"]==1)&(source==sid)); residual=target-base; oof=np.zeros((len(tr),128),"float32")
        groups=np.asarray([fold(sid,g) for g in gene[tr]])
        for f in range(4):
            fit=tr[groups!=f]; tune=np.flatnonzero(groups==f); mean=raw[fit].mean(0); scale=raw[fit].std(0).clip(1e-6); model=Ridge(alpha=10,fit_intercept=False).fit((raw[fit]-mean)/scale,residual[fit]); oof[tune]=tangent(model.predict((raw[tr[tune]]-mean)/scale),base[tr[tune]])
        scores=np.asarray([cos((base[tr]+g*oof)/(np.linalg.norm(base[tr]+g*oof,axis=1,keepdims=True)+1e-12),target[tr]) for g in gates]); choice=np.flatnonzero(scores>=scores.max()-1e-12)[0]; gate=gates[choice]; mean=raw[tr].mean(0); scale=raw[tr].std(0).clip(1e-6); model=Ridge(alpha=10,fit_intercept=False).fit((raw[tr]-mean)/scale,residual[tr]); delta=np.zeros((len(va),128),"float32"); use=known[va]; delta[use]=tangent(model.predict((raw[va[use]]-mean)/scale),base[va[use]]); pred=base[va]+gate*delta; pred/=np.linalg.norm(pred,axis=1,keepdims=True).clip(1e-12); coefs[sid]=model.coef_; means[sid]=mean; scales[sid]=scale; selected[sid]=gate; rows.append({"source":name,"training_rows":len(tr),"held_rows":len(va),"held_known_rows":int(use.sum()),"gate":float(gate),"oof_baseline_cosine":cos(base[tr],target[tr]),"oof_selected_cosine":float(scores[choice]),"baseline_cosine":cos(base[va],target[va]),"cosine":cos(pred,target[va])})
    macro=float(np.mean([r["cosine"] for r in rows])); advanced=macro>=.3905860043 and all(r["cosine"]>=.10 and r["cosine"]>=r["baseline_cosine"]-.02 for r in rows); np.savez_compressed(RUN/"source_preserving_se_action_head.npz",coef=coefs,mean=means,scale=scales,gate=selected); result={"schema":"sl-predict-source-preserving-se-action-v1","source_macro_cosine":macro,"baseline_source_macro_cosine":float(np.mean([r["baseline_cosine"] for r in rows])),"sources":rows,"advanced":bool(advanced),"double_perturbation_data_used":False,"sl_labels_used":False}; (RUN/"source_preserving_se_action_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
