import csv,json,pickle,sys
from pathlib import Path
import numpy as np,torch
from sklearn.cluster import KMeans

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; RUN=OUT/"native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; sys.path.insert(0,str(ROOT/"src/training")); sys.path.insert(0,str(Path(__file__).parent))
from context_transition import Shift
from musl import metrics
from world_model import encode_genes,load_residual_endpoint,tolerance_head

@torch.no_grad()
def main():
    device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); endpoint=load_residual_endpoint(RUN/"world_model.pt",state.shape[1],device); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); dependency=tolerance_head(128).to(device); dependency.load_state_dict(torch.load(RUN/"cold_tolerance_head.pt",map_location="cpu",weights_only=True)); dependency.eval(); shift=Shift().to(device); shift.load_state_dict(torch.load(RUN/"extended_unit_context_transition_head.pt",map_location="cpu",weights_only=True)); shift.eval()
    action=np.load(OUT/"se_replogle_gene_features.npz"); head=np.load(RUN/"source_preserving_se_action_head.npz"); molecular=json.loads((RUN/"source_preserving_se_action_metrics.json").read_text()); raw=np.stack([action["features"][:,:32],action["features"][:,32:]],1).astype("float32"); proposed=[]
    for sid in range(5):
        cid=int(sid>=4); proposed.append(torch.as_tensor((((raw[:,cid]-head["mean"][sid])/head["scale"][sid])@head["coef"][sid].T)*action["known"][:,None],device=device))
    gate=torch.as_tensor(head["gate"],device=device); gain=np.asarray([max(0,r["oof_selected_cosine"]-r["oof_baseline_cosine"]) for r in molecular["sources"]]); kw=gain[:4]/gain[:4].sum(); family_weights=(torch.as_tensor(kw,device=device,dtype=torch.float32),torch.ones(1,device=device))
    basal=np.load(OUT/"basal_context.npz"); pool=basal["cell_state"][basal["train_cell"]]; km=KMeans(32,random_state=731,n_init=10).fit(pool); contexts=np.asarray([pool[np.argmin(((pool-c)**2).sum(1))] for c in km.cluster_centers_],"float32"); assay=np.load(OUT/"context_transition.npz")["context_state"][[0,4]]; families=np.argmin(((contexts[:,None]-assay[None])**2).sum(2),1)
    def corrected(g,ids,cs,fam):
        base=torch.nn.functional.normalize(shift(g,cs),dim=1); sources=range(4) if fam==0 else (4,); weights=family_weights[fam]; correction=torch.zeros_like(base)
        for j,sid in enumerate(sources):
            delta=proposed[sid][ids]; delta=delta-(delta*base).sum(1,keepdim=True)*base; delta=delta*torch.clamp(.25/(delta.norm(dim=1,keepdim=True)+1e-12),max=1); correction+=weights[j]*gate[sid]*delta
        return torch.nn.functional.normalize(base+correction,dim=1)
    ids={r["symbol"]:i for i,r in enumerate(csv.DictReader(open(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv")))}; remap=np.asarray([ids[r["symbol"]] for r in csv.DictReader(open(ROOT/"data/models/MuSL/processed_data/meta_table_7684.csv"))],"int64"); folds=ROOT/"data/models/MuSL/processed_data/data/CV3_bins_32/fold_data"; rows=[]
    for seed in (42,432):
        pairs=pickle.load(open(folds/f"test_pairs_seed{seed}.pkl","rb")); labels=pickle.load(open(folds/f"test_labels_seed{seed}.pkl","rb"))
        for fold,(pair,y) in enumerate(zip(pairs,labels)):
            pair=remap[np.asarray(pair,"int64")]; y=np.asarray(y,"int8"); values=[]
            for lo in range(0,len(pair),1024):
                p=torch.as_tensor(pair[lo:lo+1024],device=device); a,b=genes[p[:,0]],genes[p[:,1]]; scores=[]
                for c,(state128,fam) in enumerate(zip(contexts,families)):
                    cs=torch.as_tensor(state128,device=device).expand(len(p),-1); da=corrected(a,p[:,0],cs,fam); db=corrected(b,p[:,1],cs,fam); qa=dependency(endpoint.world.transition(a,context_state=cs)[0]).squeeze(1); qb=dependency(endpoint.world.transition(b,context_state=cs)[0]).squeeze(1); qa_b=dependency(endpoint.world.transition(a,context_state=cs+db)[0]).squeeze(1); qb_a=dependency(endpoint.world.transition(b,context_state=cs+da)[0]).squeeze(1); scores.append(-((qa_b-qa)+(qb_a-qb))/2)
                values.append(torch.stack(scores).mean(0).cpu())
            row={"benchmark":"MuSL-CV3","seed":seed,"fold":fold,"pairs":len(pair),"positives":int(y.sum()),**metrics(y,torch.cat(values).numpy())}; rows.append(row); print(json.dumps(row),flush=True)
    mean={k:float(np.mean([r[k] for r in rows])) for k in ("auroc","aupr","f1_at_prevalence")}; result={"schema":"sl-predict-source-preserving-context-musl-v1","protocol":"One fixed label-free evaluation after molecular admission; negative symmetric dependency change from source-preserving single-action directions over 32 deterministic DepMap medoids; nearest K562/RPE1 context family and positive fitting-gene OOF-gain weights; no SL fitting, calibration, sign or weight selection, fusion, or genetic double perturbations","context_family_counts":{"k562":int((families==0).sum()),"rpe1":int((families==1).sum())},"k562_source_weights":kw.tolist(),"rows":rows,"mean":mean,"advanced":bool(mean["auroc"]>.6417025456 and mean["aupr"]>.6350067764),"double_perturbation_data_used":False,"sl_labels_used_for_fitting_or_selection":False}; (RUN/"musl_cv3_source_preserving_context.json").write_text(json.dumps(result,indent=2)); print(json.dumps({"mean":mean,"advanced":result["advanced"]},indent=2))

if __name__=="__main__": main()
