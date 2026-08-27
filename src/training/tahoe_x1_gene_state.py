from pathlib import Path
import csv,hashlib,json,sys
import numpy as np,torch
from safetensors import safe_open
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; MODEL=ROOT/"data/models/tahoe-x1-70m"
NAMES=("codependency","tcga","silencing"); PATHS=("depmap_codependency.npz","tcga_mutual_exclusivity.npz","depmap_expression_silencing.npz")

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=torch.nn.Sequential(torch.nn.LayerNorm(512),torch.nn.Linear(512,128)); self.heads=torch.nn.ModuleList(torch.nn.Sequential(torch.nn.LayerNorm(256),torch.nn.Linear(256,128),torch.nn.GELU(),torch.nn.Linear(128,1)) for _ in NAMES)
    def forward(self,h,a,b,task):return self.heads[task](torch.cat((h[a]*h[b],(h[a]-h[b]).abs()),1)).squeeze(1)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()

def main(epochs=12,epoch_pairs=250000,batch=4096):
    torch.manual_seed(121); np.random.seed(121); device="cuda" if torch.cuda.is_available() else "cpu"; contextual="--contextual" in sys.argv; raw=[np.load(OUT/p) for p in PATHS]; genes=[x["genes"].astype("int64") for x in raw]; meta=list(csv.DictReader(open(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"))); vocab=json.loads((MODEL/"vocab.json").read_text()); tensor=safe_open(MODEL/"model.safetensors",framework="pt"); weights=np.load(OUT/"tahoe_x1_context.npz")["gene_embeddings"].astype("float32") if contextual else tensor.get_tensor("model.gene_encoder.embedding.weight").numpy(); ens=np.asarray([r["ensembl_gene_id"].split(".")[0] for r in meta]); mapped=np.asarray([x in vocab and np.isfinite(weights[vocab[x]]).all() for x in ens]); feature=np.zeros((len(meta),512),"float32"); feature[mapped]=weights[[vocab[x] for x in ens[mapped]]]; feature[mapped]/=np.linalg.norm(feature[mapped],axis=1,keepdims=True).clip(1e-8)
    common=np.asarray(sorted(set.intersection(*(set(x) for x in genes),set(np.flatnonzero(mapped)))),"int64"); maps=[{g:i for i,g in enumerate(gs)} for gs in genes]; positions=[np.asarray([m[g] for g in common],"int32") for m in maps]; arrays=[(x["half0"],x["half1"]) for x in raw]; train=np.flatnonzero(common%5!=0); held=np.flatnonzero(common%5==0); vi,vj=np.triu_indices(len(held),1); vi=held[vi]; vj=held[vj]
    def values(task,half,a,b):
        p=positions[task]; lo=np.minimum(p[a],p[b]).astype("int64"); hi=np.maximum(p[a],p[b]).astype("int64"); x=arrays[task][half]
        return x[lo,hi].astype("float32") if x.ndim==2 else x[lo*len(genes[task])-lo*(lo+1)//2+hi-lo-1].astype("float32")
    ti,tj=np.triu_indices(len(train),1); ti=train[ti]; tj=train[tj]; stats=[]; truth=[]
    for task in range(len(NAMES)):
        y=values(task,0,ti,tj); mean=float(y.mean()); scale=float(y.std()); stats.append((mean,scale)); truth.append((values(task,1,vi,vj)-mean)/scale)
    del ti,tj; feature=torch.as_tensor(feature[common],device=device); model=Model().to(device); opt=torch.optim.AdamW(model.parameters(),3e-4,weight_decay=1e-3); history=[]; saved=[]
    @torch.no_grad()
    def evaluate(epoch):
        model.eval(); h=model.encoder(feature); row={"epoch":epoch,"held_genes":len(held),"held_pairs":len(vi),"sources":{}}
        for task,name in enumerate(NAMES):
            pred=[]
            for lo in range(0,len(vi),16384):pred.append(model(h,torch.as_tensor(vi[lo:lo+16384],device=device),torch.as_tensor(vj[lo:lo+16384],device=device),task).cpu())
            p=torch.cat(pred).numpy(); y=truth[task]; row["sources"][name]={"huber":float(torch.nn.functional.huber_loss(torch.from_numpy(p),torch.from_numpy(y))),"pearson":float(np.corrcoef(y,p)[0,1]) if p.std()>0 else 0.,"spearman":float(spearmanr(y,p).statistic) if p.std()>0 else 0.}
        row["selection_loss"]=sum(x["huber"] for x in row["sources"].values()); history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()}); print(json.dumps(row),flush=True)
    evaluate(0)
    for epoch in range(epochs):
        rng=np.random.default_rng(121+epoch); sampled=[]
        for task in range(len(NAMES)):
            a=rng.choice(train,epoch_pairs); b=rng.choice(train,epoch_pairs); same=a==b
            while same.any():b[same]=rng.choice(train,same.sum()); same=a==b
            sampled.append((a,b,(values(task,0,a,b)-stats[task][0])/stats[task][1]))
        model.train()
        for lo in range(0,epoch_pairs,batch):
            h=model.encoder(feature); loss=0
            for task,(a,b,y) in enumerate(sampled):loss+=torch.nn.functional.huber_loss(model(h,torch.as_tensor(a[lo:lo+batch],device=device),torch.as_tensor(b[lo:lo+batch],device=device),task),torch.as_tensor(y[lo:lo+batch],device=device))/len(NAMES)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        evaluate(epoch+1)
    selected=min(range(len(history)),key=lambda i:history[i]["selection_loss"]); metric=history[selected]; model.load_state_dict(saved[selected]); s=metric["sources"]; thresholds=all(s[n][k]>=.10 for n in NAMES for k in ("pearson","spearman")) and s["tcga"]["pearson"]>=.186044 and s["tcga"]["spearman"]>=.184175 and s["codependency"]["pearson"]>=.15 and s["codependency"]["spearman"]>=.10 and s["silencing"]["pearson"]>=.10 and s["silencing"]["spearman"]>=.10; improved=metric["selection_loss"]<1.223246157169342; advanced=thresholds and improved; run=OUT/"tahoe_x1_70m"; run.mkdir(exist_ok=True); suffix="contextual" if contextual else "static"; torch.save({"state_dict":model.state_dict(),"genes":torch.as_tensor(common)},run/f"gene_state_{suffix}.pt"); files={p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in MODEL.iterdir() if p.is_file()}; result={"schema":"sl-predict-tahoe-x1-gene-state-v1","representation":"normalized_32_cell_contextual" if contextual else "normalized_static_gene_token","checkpoint_parameters":int(sum(np.prod(tensor.get_slice(k).get_shape()) for k in tensor.keys())),"checkpoint_files":files,"vocabulary_tokens":len(vocab),"fixed_universe_overlap":int(mapped.sum()),"shared_genes":len(common),"training_genes":len(train),"held_genes":len(held),"held_pairs_per_source":len(vi),"readout_parameters":sum(p.numel() for p in model.parameters()),"prior_selection_loss":1.223246157169342,"selected":metric,"thresholds_passed":bool(thresholds),"improved_over_prior":bool(improved),"advanced":bool(advanced),"double_perturbation_data_used":False,"sl_labels_used":False,"benchmark_pairs_used":False,"protocol_sha256":sha(OUT/"tahoe_x1_gene_state_protocol.json"),"history":history}; (run/f"metrics_{suffix}.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k not in ("history","checkpoint_files")},indent=2)); return result

if __name__=="__main__":main()
