import csv,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; RUN=OUT/"native_spectral_safe_scaled_d768_z256_l8_p12_single_only_d3_t10_r3"; sys.path.insert(0,str(Path(__file__).parent))
from world_model import SLPredict,encode_genes

def held(x): return int.from_bytes(hashlib.sha256(x.upper().encode()).digest()[:4],"big")%5==0
def huber(x):
    x=np.abs(x); return float(np.mean(np.where(x<=1,.5*x*x,x-.5)))

@torch.no_grad()
def features(world,genes,pair,context,device):
    out=[]
    for lo in range(0,len(pair),512):
        p=torch.as_tensor(pair[lo:lo+512],device=device); cs=torch.as_tensor(context[lo:lo+512],device=device); a,b=genes[p[:,0]],genes[p[:,1]]; za=world.transition(a,context_state=cs)[0]; zb=world.transition(b,context_state=cs)[0]; joint=world.transition(a,b,context_state=cs)[0]; seq=(world.transition(b,state=za,context_state=cs)[0]+world.transition(a,state=zb,context_state=cs)[0])/2; out.append(torch.cat((joint,seq,joint-seq,(za-zb).abs(),za*zb),1).cpu())
    return torch.cat(out).numpy()

def main():
    np.random.seed(829); device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); sd=torch.load(RUN/"world_model.pt",map_location="cpu",weights_only=True); world=SLPredict(768,256,8,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],sd["context_proj.weight"].shape[1]).to(device); world.load_state_dict(sd); world.eval(); genes=torch.as_tensor(encode_genes(world,state,device),device=device); ids={r["symbol"].upper():i for i,r in enumerate(csv.DictReader(open(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv")))}; basal=np.load(OUT/"basal_context.npz"); contexts=dict(zip(basal["model_ids"].astype(str),basal["cell_state"].astype("float32")))
    data=pd.read_csv(ROOT/"data/sanger2025/DATA/postprocessing/combined_gene_level_results.tsv",sep="\t"); keep=data.targetA.astype(str).str.upper().isin(ids)&data.targetB.astype(str).str.upper().isin(ids)&data.depMapID.astype(str).isin(contexts)&np.isfinite(data.mean_norm_gi)&data.targetA__is_single_depleted.eq(0)&data.targetB__is_single_depleted.eq(0); data=data[keep].copy(); a=data.targetA.astype(str).str.upper().to_numpy(); b=data.targetB.astype(str).str.upper().to_numpy(); role=np.asarray([int(held(x))+int(held(y)) for x,y in zip(a,b)]); train=np.flatnonzero(role==0); valid=np.flatnonzero(role==2); pair=np.asarray([(ids[x],ids[y]) for x,y in zip(a,b)],"int64"); context=np.stack([contexts[x] for x in data.depMapID.astype(str)]); xtr=features(world,genes,pair[train],context[train],device); xva=features(world,genes,pair[valid],context[valid],device); y=-data.mean_norm_gi.to_numpy("float32"); scaler=StandardScaler().fit(xtr); model=Ridge(alpha=100).fit(scaler.transform(xtr),y[train]); pred=model.predict(scaler.transform(xva)); pearson=float(np.corrcoef(pred,y[valid])[0,1]); spearman=float(spearmanr(pred,y[valid]).statistic); zero=huber(y[valid]); loss=huber(pred-y[valid]); hit=data.is_bassik_hit.to_numpy("int8")[valid]; auc=float(roc_auc_score(hit,pred)); frame=pd.DataFrame({"pair":data.sorted_gene_pair.to_numpy()[valid],"pred":pred,"target":y[valid]}).groupby("pair").mean(); pair_spearman=float(spearmanr(frame.pred,frame.target).statistic); advanced=pearson>=.15 and spearman>=.15 and 1-loss/zero>=.05 and auc>=.60 and pair_spearman>=.15; np.savez_compressed(RUN/"sanger_quantitative_cold_decoder.npz",mean=scaler.mean_,scale=scaler.scale_,coef=model.coef_,intercept=model.intercept_); result={"schema":"sl-predict-sanger-quantitative-cold-decoder-v1","eligible_measurements":len(data),"training_measurements":len(train),"both_held_measurements":len(valid),"training_pairs":int(data.sorted_gene_pair.iloc[train].nunique()),"both_held_pairs":len(frame),"pearson":pearson,"spearman":spearman,"zero_huber":zero,"huber":loss,"huber_improvement":1-loss/zero,"pooled_hit_auroc":auc,"pair_mean_spearman":pair_spearman,"advanced":bool(advanced),"muSL_labels_used":False}; (RUN/"sanger_quantitative_cold_decoder.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
