from pathlib import Path
import json,sys
import numpy as np,torch
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr

sys.path.insert(0,str(Path(__file__).parent))
from world_model import encode_genes,load_residual_endpoint

def pair_features(actions,a,b): return np.column_stack((actions[a]*actions[b],np.abs(actions[a]-actions[b])))

def fit(model_path,out_dir,feature_path="/root/data/features_spectral_safe.npz",tcga_path="/root/data/tcga_mutual_exclusivity.npz",samples=1500000,batch=65536):
    data=np.load(tcga_path); supported=data["genes"].astype("int64"); i,j=np.triu_indices(len(supported),1); held=supported%5==0; train=np.flatnonzero(~held[i]&~held[j]); valid=np.flatnonzero(held[i]&held[j]); mean=float(data["half0"][train].mean()); scale=float(data["half0"][train].std()); rng=np.random.default_rng(829); chosen=rng.choice(train,samples,replace=False); state=np.load(feature_path)["state"].astype("float32"); device="cuda" if torch.cuda.is_available() else "cpu"; world=load_residual_endpoint(model_path,state.shape[1],device).world; actions=encode_genes(world,state,device)[supported].astype("float32"); x=pair_features(actions,i[chosen],j[chosen]); y=((data["half0"][chosen]-mean)/scale).astype("float32"); model=LGBMRegressor(objective="huber",n_estimators=500,num_leaves=31,learning_rate=.03,colsample_bytree=.7,reg_lambda=1,min_child_samples=30,max_bin=63,n_jobs=8,verbosity=-1,random_state=829).fit(x,y); del x,y; pred=[]
    for lo in range(0,len(valid),batch):
        ix=valid[lo:lo+batch]; pred.append(model.predict(pair_features(actions,i[ix],j[ix])).astype("float32"))
    pred=np.concatenate(pred); truth=(data["half1"][valid]-mean)/scale; pearson=float(np.corrcoef(truth,pred)[0,1]); spearman=float(spearmanr(truth,pred).statistic); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); model.booster_.save_model(str(out/"tcga_relation_lgbm.txt")); result={"schema":"sl-predict-tcga-tree-decoder-v1","training_pairs_available":int(len(train)),"training_pairs_sampled":int(samples),"validation_pairs":int(len(valid)),"features":int(2*actions.shape[1]),"huber":float(torch.nn.functional.huber_loss(torch.from_numpy(pred),torch.from_numpy(truth))),"pearson":pearson,"spearman":spearman,"advanced":bool(pearson>=.18604423537715636 and spearman>=.18417520342635268),"world_parameters_changed":0,"double_perturbation_data_used":False,"sl_labels_used":False}; (out/"tcga_relation_lgbm_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True); return result

if __name__=="__main__": fit(*sys.argv[1:])
