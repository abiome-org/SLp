from pathlib import Path
import ast,csv,hashlib,json,re
import numpy as np,pandas as pd,torch
from scipy.stats import pearsonr,spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from tahoe_transition_source import SE,ST,StateEncoder,Transition,action_map,load_se,load_transition

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; DATA=ROOT/"data/models/tahoe-x1-70m"; EXPR=ROOT/"data/depmap24q2/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
def norm(x):return re.sub("[^a-z0-9]","",str(x).lower()).replace("hydrochloride","")
def split(x):return int(hashlib.sha256(x.encode()).hexdigest(),16)%2==0
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()

def encoder_inputs(frame,vocab):
    symbols=[x.rsplit(" (",1)[0] for x in frame.columns]; index={g:i for i,g in enumerate(vocab)}; keep=[i for i,g in enumerate(symbols) if g in index]; values=frame.iloc[:,keep].to_numpy("float32")*np.float32(np.log(2)); ids=np.asarray([index[symbols[i]] for i in keep]); top=np.argsort(-values,1,kind="stable")[:,:2047]; tokens=np.concatenate((np.full((len(frame),1),index[symbols[keep[3]]]),ids[top]),1); weights=values/values.sum(1,keepdims=True); counts=np.concatenate((weights[:,3:4],np.take_along_axis(weights,top,1)),1)*100; return tokens,counts,len(keep)

def main(batch=16):
    torch.manual_seed(122); np.random.seed(122); torch.set_grad_enabled(False); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); dtype=torch.bfloat16; meta=pd.read_parquet(DATA/"drug_metadata.parquet"); genes=list(csv.DictReader(open(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"))); symbol=np.asarray([r["symbol"].upper() for r in genes]); gid={g:i for i,g in enumerate(symbol)}; state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); amap=action_map(); actions={}
    for key in amap:
        name,dose,_=ast.literal_eval(str(key))[0]; actions.setdefault(norm(name),[]).append((abs(float(dose)-2.5),str(key),key))
    rows=[]
    for r in meta.itertuples():
        if getattr(r,"_3")!="inhibitor/antagonist" or pd.isna(r.targets) or norm(r.drug) not in actions:continue
        raw=[x.strip().upper() for x in str(r.targets).split(",") if x.strip()]; mapped=[gid[x] for x in raw if x in gid]
        if mapped:rows.append({"drug":str(r.drug),"normalized":norm(r.drug),"targets":raw,"mapped":mapped,"action":min(actions[norm(r.drug)])[2]})
    panel=json.loads((OUT/"tahoe_x1_context.json").read_text())["panel_model_ids"]; found=[]
    for chunk in pd.read_csv(EXPR,index_col=0,chunksize=64):
        take=[x for x in panel if x in chunk.index]
        if take:found.append(chunk.loc[take])
    expression=pd.concat(found).loc[panel]; vocab=list(torch.load(SE/"protein_embeddings.pt",map_location="cpu",weights_only=True)); tokens,counts,overlap=encoder_inputs(expression,vocab); encoder=StateEncoder(); load_se(encoder); encoder.to(device,dtype=dtype).eval(); basal=[]
    for at in range(0,len(panel),2):basal.append(encoder(torch.as_tensor(tokens[at:at+2],device=device),torch.as_tensor(counts[at:at+2],device=device,dtype=dtype)).float().cpu())
    basal=torch.cat(basal); del encoder; torch.cuda.empty_cache(); transition=Transition(); load_transition(transition); transition.to(device,dtype=dtype).eval(); control=next(k for k in amap if "DMSO_TF', 0.0" in str(k))
    def predict(key):
        output=[]
        for at in range(0,len(basal),batch):
            b=basal[at:at+batch].to(device,dtype=dtype); a=amap[key].to(device,dtype=dtype)[None,None].expand(len(b),256,-1); output.append(transition(b[:,None].expand(-1,256,-1),a).mean(1).float().cpu())
        return torch.cat(output)
    base=predict(control); response=[]
    for i,r in enumerate(rows):
        response.append((predict(r["action"])-base).mean(0).numpy()); print(json.dumps({"actions_complete":i+1,"actions_total":len(rows)}),flush=True)
    response=np.asarray(response,"float32"); train=np.asarray([split(r["normalized"]) for r in rows]); pca=PCA(n_components=32,svd_solver="randomized",random_state=122).fit(response[train]); y=pca.transform(response); x=np.stack([state[r["mapped"]].mean(0) for r in rows]); scale=StandardScaler().fit(x[train]); model=Ridge(alpha=10).fit(scale.transform(x[train]),y[train]); pred=model.predict(scale.transform(x[~train])); truth=y[~train]; train_targets=set(sum([r["targets"] for r,k in zip(rows,train) if k],[])); isolated=np.asarray([not bool(set(r["targets"])&train_targets) for r,k in zip(rows,train) if not k])
    def metrics(a,b):
        c=np.sum(a*b,1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-8); return {"drugs":len(a),"mean_cosine":float(c.mean()),"pearson":float(pearsonr(a.ravel(),b.ravel()).statistic),"spearman":float(spearmanr(a.ravel(),b.ravel()).statistic)}
    held=metrics(pred,truth); cold=metrics(pred[isolated],truth[isolated]); train_unique=len(set(sum([r["targets"] for r,k in zip(rows,train) if k],[]))); admitted=held["mean_cosine"]>=.15 and held["spearman"]>=.15 and cold["drugs"]>=20 and cold["mean_cosine"]>0 and cold["spearman"]>0 and train.sum()>=100 and train_unique>=75; result={"schema":"sl-predict-tahoe-targeted-action-v1","metadata_rows":len(meta),"eligible_inhibitor_drugs":len(rows),"mapped_targets":len(set(sum([r["mapped"] for r in rows],[]))),"train_drugs":int(train.sum()),"held_drugs":int((~train).sum()),"train_unique_targets":train_unique,"contexts":len(panel),"expression_overlap":overlap,"components":32,"held":held,"target_isolated_held":cold,"admitted":bool(admitted),"drug_metadata_sha256":sha(DATA/"drug_metadata.parquet"),"protocol_sha256":sha(OUT/"tahoe_targeted_action_protocol.json"),"double_perturbation_data_used":False,"viability_or_sl_labels_used":False}; (OUT/"tahoe_targeted_action.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    if admitted:
        atlas=model.predict(scale.transform(state)).astype("float16"); np.savez_compressed(OUT/"tahoe_targeted_action.npz",profile=atlas,reliability=np.float32(cold["mean_cosine"]),drugs=np.asarray([r["drug"] for r in rows]),response=response.astype("float16")); torch.save({"pca_mean":pca.mean_,"pca_components":pca.components_,"scale_mean":scale.mean_,"scale_scale":scale.scale_,"coef":model.coef_,"intercept":model.intercept_},OUT/"tahoe_targeted_action.pt")
    return result

if __name__=="__main__":main()
