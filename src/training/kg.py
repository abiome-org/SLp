from pathlib import Path
import argparse,csv,hashlib,json,math,zipfile
from collections import Counter,defaultdict
import numpy as np
import torch
from torch import nn

LABEL_RELATIONS={"SL_GsG","SR_GsrG","NONSL_GnsG"}

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest().upper()

def safe_relations(path):
    rows=list(csv.DictReader(open(path,encoding="utf-8"))); return [r["type"] for r in sorted(rows,key=lambda x:int(x["id"])) if r["type"] not in LABEL_RELATIONS]

def symbol_map(names_path,map_path):
    names={int(r["_id"]):r["name"] for r in csv.DictReader(open(names_path,encoding="utf-8-sig"))}; by=defaultdict(list)
    with open(map_path,encoding="utf-8") as f:
        next(f)
        for line in f:
            db,kg=map(int,line.split()); by[names[db]].append(kg)
    return by

def audit(args):
    triples=np.loadtxt(args.triples,dtype=np.int64); names=safe_relations(args.relations); raw=Counter()
    with zipfile.ZipFile(args.raw) as z, z.open("kg_triplet.csv") as f:
        rows=csv.reader((x.decode("utf-8-sig") for x in f)); next(rows)
        for row in rows:raw[row[1]]+=1
    counts=np.bincount(triples[:,1],minlength=len(names)); assert len(names)==24 and triples[:,1].min()==0 and triples[:,1].max()==23
    assert all(counts[i]<=raw[name] for i,name in enumerate(names)) and not LABEL_RELATIONS.intersection(names)
    symbols=[r["symbol"] for r in csv.DictReader(open(args.meta,encoding="utf-8-sig"))]; by=symbol_map(args.db_names,args.db_map); hit=[g for g in symbols if len(by.get(g,()))==1]
    result={"schema":"sl-predict-safe-kg-v1","triples":len(triples),"entities":int(max(triples[:,0].max(),triples[:,2].max())+1),"relations":[{"id":i,"name":name,"triples":int(counts[i]),"raw_rows":raw[name]} for i,name in enumerate(names)],"excluded_relations":{name:raw[name] for name in sorted(LABEL_RELATIONS)},"raw_safe_rows":sum(v for k,v in raw.items() if k not in LABEL_RELATIONS),"remap_dropped_rows":sum(v for k,v in raw.items() if k not in LABEL_RELATIONS)-len(triples),"benchmark_genes":len(symbols),"unique_symbol_matches":len(hit),"triples_sha256":sha(args.triples),"raw_sha256":sha(args.raw),"db_map_sha256":sha(args.db_map),"db_names_sha256":sha(args.db_names)}
    Path(args.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

class TransE(nn.Module):
    def __init__(self,n,r,d):
        super().__init__(); self.entity=nn.Embedding(n,d); self.relation=nn.Embedding(r,d); nn.init.uniform_(self.entity.weight,-1/math.sqrt(d),1/math.sqrt(d)); nn.init.zeros_(self.relation.weight)
    def score(self,h,r,t):return 6-(self.entity(h)+self.relation(r)-self.entity(t)).abs().sum(-1)

def same_type(nodes,k,types,pools):
    out=torch.empty((len(nodes),k),dtype=torch.long,device=nodes.device); node_type=types[nodes]
    for t,pool in enumerate(pools):
        at=torch.nonzero(node_type==t).squeeze(1)
        if len(at):out[at]=pool[torch.randint(len(pool),(len(at),k),device=nodes.device)]
    return out

@torch.no_grad()
def validate(model,triples,types,pools,n=10000,k=100):
    model.eval(); x=triples[:min(n,len(triples))]; ranks=[]
    for start in range(0,len(x),512):
        b=x[start:start+512]; h,r,t=b.T; neg=same_type(t,k,types,pools); pos=model.score(h,r,t); score=model.score(h[:,None].expand_as(neg),r[:,None].expand_as(neg),neg); ranks.append(1+(score>=pos[:,None]).sum(1))
    rank=torch.cat(ranks).float(); return {"sampled_mrr":float((1/rank).mean()),"sampled_hits10":float((rank<=10).float().mean()),"validation_triples":len(x),"negatives_per_triple":k}

def fit(args):
    torch.manual_seed(20260826); np.random.seed(20260826); device="cuda" if torch.cuda.is_available() else "cpu"; x=np.loadtxt(args.triples,dtype=np.int64); typ=np.loadtxt(args.types,dtype=np.int64); valid=(x[:,0]*1000003+x[:,2]*9176+x[:,1])%100==0; train=torch.as_tensor(x[~valid],device=device); val=torch.as_tensor(x[valid],device=device); types=torch.as_tensor(typ,device=device); pools=[torch.nonzero(types==t).squeeze(1) for t in range(int(types.max())+1)]; model=TransE(len(typ),int(x[:,1].max())+1,args.dim).to(device); opt=torch.optim.AdamW(model.parameters(),2e-3,weight_decay=1e-5); rows=[]; best=None; best_mrr=-1.
    for epoch in range(args.epochs):
        model.train(); order=torch.randperm(len(train),device=device); total=seen=0
        for start in range(0,len(order),args.batch):
            h,r,t=train[order[start:start+args.batch]].T; nh=same_type(h,4,types,pools); nt=same_type(t,4,types,pools); corrupt=torch.rand_like(nh,dtype=torch.float32)<.5; hh=torch.where(corrupt,nh,h[:,None]); tt=torch.where(corrupt,t[:,None],nt); pos=model.score(h,r,t); neg=model.score(hh,r[:,None].expand_as(hh),tt); loss=nn.functional.softplus(-pos).mean()+nn.functional.softplus(neg).mean(); opt.zero_grad(); loss.backward(); opt.step(); total+=loss.item()*len(h); seen+=len(h)
        metrics=validate(model,val,types,pools); row={"epoch":epoch+1,"train_loss":total/seen,**metrics}; rows.append(row); print(json.dumps(row),flush=True)
        if metrics["sampled_mrr"]>best_mrr:best_mrr=metrics["sampled_mrr"]; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); torch.save(best,out); result={"parameters":sum(p.numel() for p in model.parameters()),"dimension":args.dim,"train_triples":len(train),"held_triples":len(val),"best":max(rows,key=lambda r:r["sampled_mrr"]),"epochs":rows}; out.with_suffix(".json").write_text(json.dumps(result,indent=2)); return result

def features(args):
    base=np.load(args.base); state=base["state"].astype("float32"); emb=torch.load(args.checkpoint,map_location="cpu",weights_only=True)["entity.weight"].numpy(); symbols=[r["symbol"] for r in csv.DictReader(open(args.meta,encoding="utf-8-sig"))]; by=symbol_map(args.db_names,args.db_map); hit=np.asarray([len(by.get(g,()))==1 for g in symbols]); ids=np.asarray([by[g][0] for g in np.asarray(symbols)[hit]]); z=emb[ids]; z=(z-z.mean(0))/(z.std(0)+1e-6); state[:,1024:1424]=0; state[hit,1024:1024+z.shape[1]]=z; np.savez_compressed(args.output,state=state.astype("float16"),pairs=base["pairs"],relations=base["relations"],gf_hit=base["gf_hit"],esm_hit=base["esm_hit"],kg_hit=hit); result={"schema":"sl-predict-safe-kg-features-v1","genes":len(state),"matched_genes":int(hit.sum()),"dimension":z.shape[1],"checkpoint_sha256":sha(args.checkpoint)}; Path(args.output).with_suffix(".json").write_text(json.dumps(result,indent=2)); print(result)

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True); a=sub.add_parser("audit"); a.add_argument("--triples",required=True); a.add_argument("--types",required=True); a.add_argument("--relations",required=True); a.add_argument("--raw",required=True); a.add_argument("--db-map",required=True); a.add_argument("--db-names",required=True); a.add_argument("--meta",required=True); a.add_argument("--output",required=True); f=sub.add_parser("fit"); f.add_argument("--triples",required=True); f.add_argument("--types",required=True); f.add_argument("--output",required=True); f.add_argument("--epochs",type=int,default=4); f.add_argument("--dim",type=int,default=128); f.add_argument("--batch",type=int,default=8192); b=sub.add_parser("features"); b.add_argument("--base",required=True); b.add_argument("--checkpoint",required=True); b.add_argument("--db-map",required=True); b.add_argument("--db-names",required=True); b.add_argument("--meta",required=True); b.add_argument("--output",required=True); args=p.parse_args(); {"audit":audit,"fit":fit,"features":features}[args.command](args)

if __name__=="__main__":main()
