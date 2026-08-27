from pathlib import Path
import csv,hashlib,json,sys
import numpy as np,torch
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; DATA=ROOT/"data/depmap24q2"


def md5(path):
    h=hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def pairs(rng,n,count):
    keys=np.empty(0,"int64")
    while len(keys)<count:
        x=rng.integers(0,n,(int(1.2*(count-len(keys)))+1000,2)); x.sort(1); x=x[x[:,0]!=x[:,1]]; keys=np.unique(np.r_[keys,x[:,0]*n+x[:,1]])
    keys=keys[:count]; return keys//n,keys%n


def expression(model_ids):
    path=DATA/"OmicsExpressionProteinCodingGenesTPMLogp1.csv"; meta=list(csv.DictReader((ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv").open())); symbols=[r["symbol"].upper() for r in meta]
    with path.open(encoding="utf-8-sig",newline="") as f:
        header=next(csv.reader(f)); columns={x.split(" (")[0].upper():i for i,x in enumerate(header[1:],1)}; positions=[columns.get(x) for x in symbols]; index={x:i for i,x in enumerate(model_ids)}; out=np.full((len(model_ids),len(symbols)),np.nan,"float32")
        for row in csv.reader(f):
            at=index.get(row[0]);
            if at is not None: out[at]=[float(row[i]) if i is not None and row[i] else np.nan for i in positions]
    return out,path


def normalized(x,known):
    count=known.sum(0).clip(1); mean=(np.where(known,x,0).sum(0)/count); z=np.where(known,x-mean,0); return z/np.sqrt((z*z).sum(0)).clip(1e-6)


def build(sample_pairs=2000000):
    source=np.load(OUT/"basal_context.npz"); ids=source["model_ids"].astype(str); ex,path=expression(ids); dependency=source["dependency"].astype("float32"); known=source["dependency_known"]; parity=np.asarray([hashlib.sha256(x.encode()).digest()[0]&1 for x in ids]); halves=[source["train_cell"]&(parity==q) for q in (0,1)]; eligible=np.isfinite(ex[halves[0]]).mean(0)>=.8; eligible&=np.isfinite(ex[halves[1]]).mean(0)>=.8; eligible&=np.nanstd(ex[halves[0]],0)>1e-4; eligible&=np.nanstd(ex[halves[1]],0)>1e-4; eligible&=known[halves[0]].mean(0)>=.8; eligible&=known[halves[1]].mean(0)>=.8; genes=np.flatnonzero(eligible); device="cuda" if torch.cuda.is_available() else "cpu"; matrices=[]
    for h in halves:
        x=normalized(ex[h][:,genes],np.isfinite(ex[h][:,genes])); y=normalized(dependency[h][:,genes],known[h][:,genes]); xt=torch.as_tensor(x,device=device); yt=torch.as_tensor(y,device=device); cross=xt.T@yt; relation=((cross+cross.T)/2).clamp(-1,1); relation.fill_diagonal_(0); matrices.append(relation.half().cpu().numpy()); del xt,yt,cross,relation
        if device=="cuda": torch.cuda.empty_cache()
    rng=np.random.default_rng(761); i,j=pairs(rng,len(genes),min(sample_pairs,len(genes)*(len(genes)-1)//2)); a=matrices[0][i,j].astype("float32"); b=matrices[1][i,j].astype("float32"); qa,qb=np.quantile(a,.99),np.quantile(b,.99); pearson=float(np.corrcoef(a,b)[0,1]); spearman=float(spearmanr(a,b).statistic); enrichment=float(np.mean((a>=qa)&(b>=qb))/.0001); admitted=pearson>=.15 and spearman>=.15 and enrichment>=3; np.savez_compressed(OUT/"depmap_crossmodal_relation.npz",genes=genes.astype("int16"),half0=matrices[0],half1=matrices[1]); result={"schema":"sl-predict-depmap-crossmodal-relation-v1","expression_file":str(path.relative_to(ROOT)).replace('\\','/'),"expression_md5":md5(path),"cells_per_half":[int(h.sum()) for h in halves],"eligible_genes":len(genes),"eligible_pairs":len(genes)*(len(genes)-1)//2,"validation_sample_pairs":len(i),"split_half_pearson":pearson,"split_half_spearman":spearman,"top_one_percent_overlap_enrichment":enrichment,"admitted":bool(admitted),"double_perturbation_data_used":False,"sl_labels_used":False}; (OUT/"depmap_crossmodal_relation.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); return result


if __name__=="__main__": build(int(sys.argv[1]) if len(sys.argv)>1 else 2000000)
