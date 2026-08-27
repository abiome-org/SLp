from pathlib import Path
import hashlib,json
import h5py,numpy as np,torch
from safetensors.torch import load_file
from sklearn.cluster import KMeans

ROOT=Path(__file__).resolve().parents[2]; MODEL=ROOT/"data/models/tahoe-x1-70m"; OUT=ROOT/"results/sl_predict"

class Block(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.norm1=torch.nn.LayerNorm(512); self.norm2=torch.nn.LayerNorm(512); self.self_attn=torch.nn.Module(); self.self_attn.Wqkv=torch.nn.Linear(512,1536); self.self_attn.out_proj=torch.nn.Linear(512,512); self.up_proj=torch.nn.Linear(512,2048); self.down_proj=torch.nn.Linear(2048,512)
    def forward(self,x):
        y=self.norm1(x); q,k,v=self.self_attn.Wqkv(y).chunk(3,-1); shape=(*q.shape[:2],8,64); q,k,v=(z.view(shape).transpose(1,2) for z in (q,k,v)); x=x+self.self_attn.out_proj(torch.nn.functional.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape_as(y)); return x+self.down_proj(torch.relu(self.up_proj(self.norm2(x))))

class Tx1(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.gene_encoder=torch.nn.Module(); self.gene_encoder.embedding=torch.nn.Embedding(62720,512,padding_idx=0); self.gene_encoder.enc_norm=torch.nn.LayerNorm(512); self.expression_encoder=torch.nn.Module(); self.expression_encoder.linear1=torch.nn.Linear(1,512); self.expression_encoder.linear2=torch.nn.Linear(512,512); self.expression_encoder.norm=torch.nn.LayerNorm(512); self.transformer_encoder=torch.nn.Module(); self.transformer_encoder.layers=torch.nn.ModuleList(Block() for _ in range(12)); self.transformer_encoder.norm=torch.nn.LayerNorm(512)
    def forward(self,gene,value):
        x=self.gene_encoder.enc_norm(self.gene_encoder.embedding(gene)); y=self.expression_encoder.linear2(torch.relu(self.expression_encoder.linear1(value.unsqueeze(-1)))); x=x+self.expression_encoder.norm(y)
        for layer in self.transformer_encoder.layers:x=layer(x)
        return self.transformer_encoder.norm(x)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()

def main(batch=4):
    torch.manual_seed(121); np.random.seed(121); device="cuda" if torch.cuda.is_available() else "cpu"; vocab=json.loads((MODEL/"vocab.json").read_text()); f=h5py.File(MODEL/"depmap_counts.h5ad","r"); ids=np.char.decode(f["obs/ModelID"][:].astype("S")); feature=np.char.decode(f["var/feature_id"][:].astype("S")); gene=np.asarray([vocab.get(x,-1) for x in feature]); rows={x:i for i,x in enumerate(ids)}; basal=np.load(OUT/"basal_context.npz"); eligible=np.asarray([i for i,x in enumerate(basal["model_ids"].astype(str)) if basal["train_cell"][i] and x in rows]); pool=basal["cell_state"][eligible]; fit=KMeans(32,random_state=731,n_init=10).fit(pool); chosen=eligible[np.asarray([np.argmin(((pool-c)**2).sum(1)) for c in fit.cluster_centers_])]; models=basal["model_ids"][chosen].astype(str); xdata=f["X/data"][:]; xindex=f["X/indices"][:]; xptr=f["X/indptr"][:]; samples=[]
    for model in models:
        r=rows[model]; lo,hi=xptr[r:r+2]; col=xindex[lo:hi]; keep=gene[col]>=0; g=torch.as_tensor(np.r_[vocab["<cls>"],gene[col[keep]]]); value=torch.as_tensor(np.r_[-2.,xdata[lo:hi][keep]],dtype=torch.float32); grades=torch.linspace(0,1,50); bins=torch.quantile(value[1:],grades); value[1:]=torch.bucketize(value[1:],bins,right=False)+1; order=torch.cat((torch.zeros(1,dtype=torch.long),torch.randperm(len(g)-1)[:1023]+1)); samples.append((g[order],value[order]))
    model=Tx1(); state=load_file(MODEL/"model.safetensors"); relevant={k.removeprefix("model."):v for k,v in state.items() if k.startswith(("model.gene_encoder.","model.expression_encoder.","model.transformer_encoder."))}; missing,unexpected=model.load_state_dict(relevant,strict=False); allowed={"expression_encoder.dropout"}; assert not unexpected and not set(missing)-allowed,(missing,unexpected); model.to(device).eval(); total=torch.zeros((len(vocab),512),device=device); count=torch.zeros(len(vocab),device=device); peak=0
    with torch.no_grad(),torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=="cuda"):
        for at in range(0,len(samples),batch):
            g=torch.stack([z[0] for z in samples[at:at+batch]]).to(device); v=torch.stack([z[1] for z in samples[at:at+batch]]).to(device); h=torch.nn.functional.normalize(model(g,v).float(),dim=-1); flat=g.flatten(); total.index_add_(0,flat,h.flatten(0,1)); count.index_add_(0,flat,torch.ones_like(flat,dtype=torch.float32)); peak=max(peak,torch.cuda.max_memory_allocated() if device=="cuda" else 0); print(json.dumps({"embedded_cells":min(at+batch,len(samples)),"panel_cells":len(samples)}),flush=True)
    out=torch.full_like(total,float("nan")); known=count>0; out[known]=torch.nn.functional.normalize(total[known]/count[known,None],dim=1); result={"schema":"sl-predict-tahoe-x1-context-v1","panel_rule":"32 medoids from count-covered DepMap training cells; KMeans random_state 731, n_init 10","panel_model_ids":models.tolist(),"eligible_cells":len(eligible),"sampled_tokens_per_cell":1024,"covered_vocabulary_tokens":int(known.sum()),"peak_gpu_bytes":int(peak),"finite":bool(torch.isfinite(out[known]).all()),"checkpoint_sha256":sha(MODEL/"model.safetensors"),"counts_sha256":sha(MODEL/"depmap_counts.h5ad"),"double_perturbation_data_used":False,"sl_labels_used":False}; np.savez_compressed(OUT/"tahoe_x1_context.npz",gene_embeddings=out.cpu().numpy().astype("float16"),gene_counts=count.cpu().numpy().astype("int16"),panel_model_ids=models); (OUT/"tahoe_x1_context.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); return result

if __name__=="__main__":main()
