from pathlib import Path
import csv,json,pickle,sys,time
import numpy as np,pandas as pd,torch

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; MODEL=ROOT/"data/models/weights/Geneformer/Geneformer-V2-104M_CLcancer"; DICT=MODEL.parent/"geneformer"; sys.path.insert(0,str(Path(__file__).parent))
from context_transition import Shift
from world_model import encode_genes,load_residual_endpoint

def build(batch=8):
    from transformers import BertModel
    meta=list(csv.DictReader(open(ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv"))); ens=np.asarray([r["ensembl_gene_id"].split(".")[0] for r in meta]); tok=pickle.load(open(DICT/"token_dictionary_gc104M.pkl","rb")); median=pickle.load(open(DICT/"gene_median_dictionary_gc104M.pkl","rb")); path=ROOT/"data/depmap24q2/OmicsExpressionProteinCodingGenesTPMLogp1.csv"; header=pd.read_csv(path,nrows=0).columns; named={x.rsplit(" (",1)[0].upper():x for x in header[1:]}; use=[header[0]]+[named[r["symbol"].upper()] for r in meta if r["symbol"].upper() in named]; frame=pd.read_csv(path,usecols=use).set_index(header[0]); models=["ACH-000551","ACH-002462","ACH-002463","ACH-002464","ACH-002465","ACH-002467"]; expression=np.zeros((6,len(meta)),"float32")
    for j,r in enumerate(meta):
        col=named.get(r["symbol"].upper())
        if col is not None: expression[:,j]=frame.loc[models,col]
    raw=np.expm1(expression); contexts=(raw[0],raw[1:].mean(0)); pack=np.load(OUT/"context_transition.npz"); wanted=(np.unique(pack["gene"][pack["source"]<4]),np.unique(pack["gene"][pack["source"]==4])); device="cuda" if torch.cuda.is_available() else "cpu"; model=BertModel.from_pretrained(MODEL,torch_dtype=torch.float16 if device=="cuda" else torch.float32,attn_implementation="sdpa").to(device).eval(); delta=np.zeros((2,len(meta),768),"float16"); known=np.zeros((2,len(meta)),bool); coverage=[]; started=time.time()
    for cid,x in enumerate(contexts):
        at=np.asarray([i for i,e in enumerate(ens) if e in tok and e in median and x[i]>0]); score=x[at]/np.asarray([median[ens[i]] for i in at]); order=at[np.argsort(-score)[:4094]]; sequence=np.asarray([tok["<cls>"],*map(lambda i:tok[ens[i]],order),tok["<eos>"]],"int64"); pos={g:i+1 for i,g in enumerate(order)}; genes=np.asarray([g for g in wanted[cid] if g in pos]); base=torch.as_tensor(sequence,device=device).unsqueeze(0)
        with torch.inference_mode(): original=model(base,attention_mask=torch.ones_like(base)).last_hidden_state[:,1:-1].mean(1)
        for lo in range(0,len(genes),batch):
            gs=genes[lo:lo+batch]; deleted=np.stack([np.delete(sequence,pos[int(g)]) for g in gs]); ids=torch.as_tensor(deleted,device=device)
            with torch.inference_mode(): perturbed=model(ids,attention_mask=torch.ones_like(ids)).last_hidden_state[:,1:-1].mean(1); d=(original-perturbed).float().cpu().numpy()
            delta[cid,gs]=d.astype("float16"); known[cid,gs]=True
        coverage.append({"context":"K562" if cid==0 else "RPE1","sequence_genes":len(order),"requested_genes":len(wanted[cid]),"delta_genes":len(genes)}); print(json.dumps(coverage[-1]),flush=True)
    np.savez_compressed(OUT/"geneformer_isp.npz",delta=delta,known=known,coverage=np.asarray([x["delta_genes"] for x in coverage])); audit={"schema":"sl-predict-geneformer-isp-v1","coverage":coverage,"seconds":time.time()-started,"double_perturbation_data_used":False,"sl_labels_used":False}; (OUT/"geneformer_isp.json").write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2)); return audit

class Residual(torch.nn.Module):
    def __init__(self): super().__init__(); self.net=torch.nn.Sequential(torch.nn.LayerNorm(768),torch.nn.Linear(768,64),torch.nn.GELU(),torch.nn.Linear(64,128)); torch.nn.init.zeros_(self.net[-1].weight); torch.nn.init.zeros_(self.net[-1].bias)
    def forward(self,base,x,known): return torch.nn.functional.normalize(base+known[:,None]*self.net(x),dim=1)

def fit(epochs=60):
    torch.manual_seed(739); np.random.seed(739); device="cuda" if torch.cuda.is_available() else "cpu"; data=np.load(OUT/"context_transition.npz"); isp=np.load(OUT/"geneformer_isp.npz"); state=np.load(OUT/"features_spectral_safe.npz")["state"].astype("float32"); model_dir=OUT/"native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; endpoint=load_residual_endpoint(model_dir/"world_model.pt",state.shape[1],device); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); shift=Shift().to(device); shift.load_state_dict(torch.load(model_dir/"extended_unit_context_transition_head.pt",map_location="cpu",weights_only=True)); shift.eval(); gene=torch.as_tensor(data["gene"].astype("int64"),device=device); source=torch.as_tensor(data["source"].astype("int64"),device=device); context=torch.as_tensor(data["context_state"],device=device); target=torch.nn.functional.normalize(torch.as_tensor(data["target"],device=device),dim=1); cid=np.where(data["source"]<4,0,1); raw=isp["delta"][cid,data["gene"]].astype("float32"); known=isp["known"][cid,data["gene"]].astype("float32"); train=np.flatnonzero(data["role"]==0); valid=np.flatnonzero(data["role"]==1); hit=train[known[train]>0]; mean=raw[hit].mean(0); scale=raw[hit].std(0).clip(1e-6); x=torch.as_tensor(((raw-mean)/scale)*known[:,None],device=device); known_t=torch.as_tensor(known,device=device)
    with torch.no_grad(): base=torch.nn.functional.normalize(shift(genes[gene],context[source]),dim=1)
    head=Residual().to(device); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3); counts=np.bincount(data["source"][train],minlength=5); history=[]; saved=[]
    @torch.no_grad()
    def evaluate(epoch):
        pred=head(base[valid],x[valid],known_t[valid]); rows=[]
        for sid,name in enumerate(data["sources"].astype(str)):
            at=torch.as_tensor(np.flatnonzero(data["source"][valid]==sid),device=device); y=target[torch.as_tensor(valid,device=device)[at]]; rows.append({"source":name,"rows":len(at),"known_rows":int(known[valid][at.cpu().numpy()].sum()),"baseline_cosine":float(torch.nn.functional.cosine_similarity(base[valid][at],y).mean()),"cosine":float(torch.nn.functional.cosine_similarity(pred[at],y).mean())})
        row={"epoch":epoch,"source_macro_cosine":float(np.mean([r["cosine"] for r in rows])),"sources":rows}; history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()}); return row
    print(json.dumps(evaluate(0)),flush=True)
    for epoch in range(epochs):
        order=np.random.default_rng(739+epoch).permutation(train); head.train()
        for lo in range(0,len(order),512):
            ix=torch.as_tensor(order[lo:lo+512],device=device); loss_raw=1-torch.nn.functional.cosine_similarity(head(base[ix],x[ix],known_t[ix]),target[ix]); weight=torch.as_tensor(1/counts[data["source"][order[lo:lo+512]]],dtype=torch.float32,device=device); loss=(loss_raw*weight).sum()/weight.sum(); opt.zero_grad(); loss.backward(); opt.step()
        head.eval(); row=evaluate(epoch+1); print(json.dumps({k:v for k,v in row.items() if k!="sources"}),flush=True)
    best=max(range(len(history)),key=lambda i:history[i]["source_macro_cosine"]); selected=history[best]; head.load_state_dict(saved[best]); advanced=selected["source_macro_cosine"]>=.390586 and all(r["cosine"]>=.10 and r["cosine"]>=r["baseline_cosine"]-.02 for r in selected["sources"]); checkpoint={"state_dict":head.state_dict(),"mean":torch.as_tensor(mean),"scale":torch.as_tensor(scale)}; torch.save(checkpoint,model_dir/"geneformer_isp_transition_head.pt"); result={"schema":"sl-predict-geneformer-isp-transition-v1","parameters":sum(p.numel() for p in head.parameters()),"selected":selected,"advanced":bool(advanced),"double_perturbation_data_used":False,"sl_labels_used":False,"history":history}; (model_dir/"geneformer_isp_transition_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2)); return result

if __name__=="__main__": {"build":build,"fit":fit}.get(sys.argv[1] if len(sys.argv)>1 else "build")()
