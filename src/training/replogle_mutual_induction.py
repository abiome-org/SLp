import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy.stats import pearsonr,spearmanr
from sklearn.decomposition import PCA

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; RUN=OUT/"native_spectral_safe_scaled_d768_z256_l8_p12_single_only_d3_t10_r3"; sys.path.insert(0,str(Path(__file__).parent))
from world_model import SLPredict,SourceEndpoint,encode_genes

def corr(a,b): return {"pairs":len(a),"pearson":float(pearsonr(a,b).statistic),"spearman":float(spearmanr(a,b).statistic)}
def pair_values(x,columns):
    q=x[:,columns]; i,j=np.triu_indices(len(columns),1); return ((q[i,j]+q[j,i])/2).astype("float32")

def main():
    meta=pd.read_csv(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"); symbols=meta.symbol.astype(str).str.upper().to_numpy(); ids={g:i for i,g in enumerate(symbols)}; ens={str(e).split('.')[0]:str(g).upper() for e,g in zip(meta.ensembl_gene_id,meta.symbol) if pd.notna(e)}; isolated=set(json.loads((OUT/"slkb_outcomes_intervention_external.json").read_text())["symbols"]); val=lambda g:int.from_bytes(hashlib.sha256(g.encode()).digest()[:4],"big")%5==0; names=("replogle2022_k562","replogle2022_rpe1"); raw=[]
    for name in names:
        z=np.load(ROOT/f"data/perturbseq_sources/base/{name}.npz"); a=np.char.upper(z["endpoint_a"].astype(str)); role=np.asarray([(1 if val(g) else 0) if g in ids and g not in isolated else -1 for g in a],"int8"); role[z["cardinality"]!=1]=-1; delta=z["future_state"].astype("float32")-z["control_mean"].astype("float32"); fit=role==0; center=delta[fit].mean(0); scale=delta[fit].std(0).clip(.05); x=(delta-center)/scale; pca=PCA(32,svd_solver="randomized",random_state=967).fit(x[fit]); latent_scale=pca.transform(x[fit]).std(0).clip(.05); feature=np.asarray([ens.get(str(g).split('.')[0],str(g).upper()) for g in z["feature_name"]]); fmap={g:i for i,g in enumerate(feature)}; observed={g:x[a==g].mean(0) for g in np.unique(a[role>=0])}; raw.append((observed,fmap,pca,latent_scale))
    common=sorted(set(raw[0][0])&set(raw[1][0])&set(raw[0][1])&set(raw[1][1])&set(ids)); train=np.asarray([g for g in common if not val(g)]); held=np.asarray([g for g in common if val(g)]); device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); sd=torch.load(RUN/"world_model.pt",map_location="cpu",weights_only=True); world=SLPredict(768,256,8,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],sd["context_proj.weight"].shape[1]).to(device); world.load_state_dict(sd); world.eval().requires_grad_(False); head=SourceEndpoint(5,256,32).to(device); head.load_state_dict(torch.load(RUN/"full_transcriptome_single_endpoint32.pt",map_location="cpu",weights_only=True)); head.eval().requires_grad_(False); encoded=torch.as_tensor(encode_genes(world,state,device),device=device); context=torch.as_tensor(np.load(OUT/"full_transcriptome_single_endpoint32.npz")["context_state"].astype("float32"),device=device); observed_scores=[]; predicted_scores=[]; held_columns=[]
    with torch.no_grad():
        for sid,(observed,fmap,pca,latent_scale) in enumerate(raw,3):
            obs=np.stack([observed[g] for g in held]); cols=np.asarray([fmap[g] for g in held]); ix=torch.as_tensor([ids[g] for g in held],device=device); source=torch.full((len(held),),sid,device=device,dtype=torch.long); z=world.transition(encoded[ix],context_state=context[sid].expand(len(held),-1))[0]; q=head(z,source).cpu().numpy(); pred=(q*latent_scale)@pca.components_+pca.mean_; observed_scores.append(pair_values(obs,cols)); predicted_scores.append(pair_values(pred,cols)); held_columns.append(cols)
    train_scores=[]
    for observed,fmap,_,_ in raw: train_scores.append(pair_values(np.stack([observed[g] for g in train]),np.asarray([fmap[g] for g in train])))
    source_agreement=corr(*train_scores); held_metrics=[{"source":name,**corr(pred,obs)} for name,pred,obs in zip(names,predicted_scores,observed_scores)]; consensus=corr(np.mean(predicted_scores,0),np.mean(observed_scores,0)); advanced=source_agreement["pearson"]>=.10 and source_agreement["spearman"]>=.10 and all(x["pearson"]>=.10 and x["spearman"]>=.10 for x in held_metrics) and consensus["pearson"]>=.15 and consensus["spearman"]>=.15; support=np.asarray(sorted(set(raw[0][1])&set(raw[1][1])&set(ids))); result={"schema":"sl-predict-replogle-mutual-induction-v1","common_genes":len(common),"fitting_genes":len(train),"held_genes":len(held),"supported_genes":len(support),"source_agreement":source_agreement,"held_prediction":held_metrics,"held_consensus":consensus,"admitted":bool(advanced),"double_perturbation_data_used":False,"sl_labels_used":False}; (OUT/"replogle_mutual_induction.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    if advanced:
        matrices=[]
        with torch.no_grad():
            for sid,(_,fmap,pca,latent_scale) in enumerate(raw,3):
                ix=torch.as_tensor([ids[g] for g in support],device=device); source=torch.full((len(support),),sid,device=device,dtype=torch.long); z=world.transition(encoded[ix],context_state=context[sid].expand(len(support),-1))[0]; q=head(z,source).cpu().numpy(); cols=np.asarray([fmap[g] for g in support]); pred=(q*latent_scale)@pca.components_[:,cols]+pca.mean_[cols]; matrix=(pred+pred.T)/2; np.fill_diagonal(matrix,0); matrices.append(matrix.astype("float16"))
        np.savez_compressed(OUT/"replogle_mutual_induction.npz",genes=np.asarray([ids[g] for g in support],"int16"),half0=matrices[0],half1=matrices[1])

if __name__=="__main__": main()
