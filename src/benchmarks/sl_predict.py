import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import auc, ndcg_score, precision_recall_curve, roc_auc_score


@torch.no_grad()
def score_all(model, state, device, batch=32768, block=24):
    model.eval(); state=torch.as_tensor(state,dtype=torch.float32); z=[]
    for lo in range(0,len(state),2048): z.append(model.encode(state[lo:lo+2048].to(device)).cpu())
    z=torch.cat(z).to(device); n=len(z); score=np.full((n,n),-1e9,"float32")
    for lo in range(0,n,block):
        ii=np.concatenate([np.full(n-i-1,i,"int32") for i in range(lo,min(lo+block,n))])
        jj=np.concatenate([np.arange(i+1,n,dtype="int32") for i in range(lo,min(lo+block,n))])
        for at in range(0,len(ii),batch):
            a=torch.as_tensor(ii[at:at+batch],device=device); b=torch.as_tensor(jj[at:at+batch],device=device)
            s=model.outcome(model.transition(z[a],z[b])[0])[:,-1]
            score[ii[at:at+batch],jj[at:at+batch]]=s.cpu().numpy()
        print(json.dumps({"phase":"score_matrix","rows":min(lo+block,n),"genes":n}),flush=True)
    return np.maximum(score,score.T)


def ap_at_k(row):
    hit=np.flatnonzero(row)
    return float(np.mean((np.arange(1,len(hit)+1)/(hit+1)))) if len(hit) else 0.


def metrics(score, pos, neg, seen):
    ps=score[pos[:,0],pos[:,1]]; ns=score[neg[:,0],neg[:,1]]; y=np.r_[np.ones(len(ps)),np.zeros(len(ns))]; s=np.r_[ps,ns]
    precision,recall,_=precision_recall_curve(y,s); f1=np.nanmax(2*precision*recall/(precision+recall+1e-12))
    seen=np.unique(seen,axis=0); old=score[seen[:,0],seen[:,1]].copy(); score[seen[:,0],seen[:,1]]=score[seen[:,1],seen[:,0]]=-1e9
    truth={}
    for a,b in pos:
        truth.setdefault(int(a),set()).add(int(b)); truth.setdefault(int(b),set()).add(int(a))
    labels=[]; ranked=[]
    for gene,partners in truth.items():
        top=np.argpartition(score[gene],-100)[-100:]; top=top[np.argsort(score[gene,top])[::-1]]
        labels.append(np.isin(top,list(partners))); ranked.append(score[gene,top])
    score[seen[:,0],seen[:,1]]=old; score[seen[:,1],seen[:,0]]=old
    labels=np.asarray(labels); ranked=np.asarray(ranked); counts=np.asarray([len(x) for x in truth.values()]); out=[]
    for k in (10,20,50): out.append(float(ndcg_score(labels,ranked,k=k)))
    for k in (10,20,50): out.append(float((labels[:,:k].sum(1)/counts).mean()))
    for k in (10,20,50): out.append(float((labels[:,:k].sum(1)/np.minimum(counts,k)).mean()))
    for k in (10,20,50): out.append(float(np.mean([ap_at_k(x[:k]) for x in labels])))
    return {"auroc":float(roc_auc_score(y,s)),"aupr":float(auc(recall,precision)),"f1":float(f1),
            **dict(zip((f"{m}{k}" for m in ("ndcg","recall","precision","map") for k in (10,20,50)),out))}


def evaluate(model_path, feature_path, split_path, out_path, d=384, latent=128, layers=6):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"training")); from world_model import SLPredict
    state=np.load(feature_path)["state"].astype("float32"); splits=np.load(split_path); sd=torch.load(model_path,map_location="cpu",weights_only=True)
    model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0]).cuda(); model.load_state_dict(sd)
    score=score_all(model,state,"cuda"); stems=sorted({k.rsplit("_pos_train_",1)[0] for k in splits.files if "_pos_train_" in k}); rows=[]
    for stem in stems:
        for fold in range(5):
            pos=splits[f"{stem}_pos_test_{fold}"].astype("int32"); neg=splits[f"{stem}_neg_test_{fold}"].astype("int32")
            seen=np.concatenate((splits[f"{stem}_pos_train_{fold}"],splits[f"{stem}_neg_train_{fold}"])).astype("int32")
            row={"protocol":stem,"fold":fold,**metrics(score,pos,neg,seen)}; rows.append(row); print(json.dumps(row),flush=True)
    Path(out_path).write_text(json.dumps(rows,indent=2)); return rows
