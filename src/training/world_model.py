import json, math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

SEED = 123
class SLPredict(nn.Module):
    def __init__(self, d=384, latent=128, layers=6, contexts=28, outcomes=2, state_dim=1816, context_dim=0):
        super().__init__()
        anchor=(state_dim-1624)//6; self.slices=((0,768),(768,1024),(1024,1424),(1424,1624),*((1624+i*anchor,1624+(i+1)*anchor) for i in range(6)))
        self.proj = nn.ModuleList(nn.Linear(hi-lo, d) for lo, hi in self.slices)
        self.cls = nn.Parameter(torch.randn(d)/math.sqrt(d)); self.basal = nn.Parameter(torch.zeros(d))
        self.context = nn.Parameter(torch.zeros(d)); self.cell = nn.Embedding(contexts, d); self.context_proj=nn.Linear(context_dim,d,bias=False) if context_dim else None; self.time = nn.Parameter(torch.zeros(d))
        self.state_up = nn.Linear(latent, d); self.token_type = nn.Embedding(5, d)
        layer = nn.TransformerEncoderLayer(d, 6, 4*d, .1, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.core = nn.TransformerEncoder(layer, layers, nn.LayerNorm(d))
        self.gene = nn.Linear(d, latent); self.dist = nn.Linear(d, 2*latent)
        self.reconstruct = nn.Linear(latent, state_dim); self.decode_state = nn.Linear(latent, 6*anchor)
        self.relation = nn.Sequential(nn.Linear(3*latent, 256), nn.GELU(), nn.Linear(256, 6))
        self.outcome = nn.Linear(latent, outcomes)

    def encode(self, x, mask=.0):
        tok = torch.stack([p(x[:, lo:hi]) for p, (lo, hi) in zip(self.proj, self.slices)], 1)
        if self.training and mask: tok = tok * (torch.rand(tok.shape[:2], device=x.device) > mask)[..., None]
        cls = self.cls[None, None].expand(len(x), 1, -1) + self.token_type.weight[0]
        return self.gene(self.core(torch.cat((cls, tok + self.token_type.weight[1]), 1))[:, 0])

    def transition(self, action, second=None, state=None, context=None, context_state=None):
        b = len(action); start = self.basal[None].expand(b, -1) if state is None else self.state_up(state)
        seq = [start + self.token_type.weight[2], self.state_up(action) + self.token_type.weight[3]]
        if second is not None: seq.append(self.state_up(second) + self.token_type.weight[3])
        if context_state is not None: ctx=self.context[None]+self.context_proj(context_state)
        elif context is None: ctx = self.context[None].expand(b, -1)
        else:
            ctx = self.cell(context.clamp_min(0)); ctx = torch.where((context>=0)[:,None],ctx,self.context[None])
        seq += [self.time[None].expand(b, -1) + self.token_type.weight[4], ctx]
        h = self.core(torch.stack(seq, 1))[:, 0]; mu, logsd = self.dist(h).chunk(2, 1)
        return mu, logsd.clamp(-5, 2)

    def transition_set(self, actions, mask, state=None, context=None, context_state=None):
        b=len(actions); start=self.basal[None].expand(b,-1) if state is None else self.state_up(state); action=self.state_up(actions)+self.token_type.weight[3]; ctx=self.context[None]+self.context_proj(context_state) if context_state is not None else (self.context[None].expand(b,-1) if context is None else torch.where((context>=0)[:,None],self.cell(context.clamp_min(0)),self.context[None])); seq=torch.cat((start[:,None]+self.token_type.weight[2],action,self.time[None,None].expand(b,1,-1)+self.token_type.weight[4],ctx[:,None]),1); pad=torch.cat((torch.zeros((b,1),device=mask.device,dtype=torch.bool),~mask,torch.zeros((b,2),device=mask.device,dtype=torch.bool)),1); h=self.core(seq,src_key_padding_mask=pad)[:,0]; mu,logsd=self.dist(h).chunk(2,1); return mu,logsd.clamp(-5,2)

    def relation_score(self, a, b, joint):
        return self.relation(torch.cat((a*b, (a-b).abs(), joint), 1))

    def count(self): return sum(p.numel() for p in self.parameters())


class ResidualEndpoint(nn.Module):
    def __init__(self,world,legacy_decoder,residual_dim=64):
        super().__init__(); self.world=world.requires_grad_(False); self.legacy_decoder=legacy_decoder.requires_grad_(False); latent=world.gene.out_features
        self.residual_decoder=nn.Sequential(nn.LayerNorm(latent),nn.Linear(latent,latent),nn.GELU(),nn.Linear(latent,residual_dim))

    def forward(self,action,second=None,context=None,context_state=None,sample=None):
        mu,logsd=self.world.transition(action,second,context=context,context_state=context_state); z=mu if sample is None else mu+logsd.exp()*sample
        return torch.cat((self.legacy_decoder(mu),self.residual_decoder(z)),1),mu,logsd


class ResidualInteraction(nn.Module):
    def __init__(self,base_head,dim=128):
        super().__init__(); self.base_head=base_head.requires_grad_(False); self.correction=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,64),nn.GELU(),nn.Linear(64,2)); nn.init.zeros_(self.correction[-1].weight); nn.init.zeros_(self.correction[-1].bias)
    def forward(self,z,residual):return self.base_head(z)+self.correction(residual)


class RankedResidualInteraction(nn.Module):
    def __init__(self,base_head,dim=128):
        super().__init__(); self.base_head=base_head.requires_grad_(False); self.rank=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,16),nn.GELU(),nn.Linear(16,1)); nn.init.zeros_(self.rank[-1].weight); nn.init.zeros_(self.rank[-1].bias)
    def forward(self,z,residual):
        base=self.base_head(z,residual); return torch.cat((base[:,:1]+self.rank(residual),base[:,1:]),1)


class SourceEndpoint(nn.Module):
    def __init__(self,sources=5,latent=128,state=32):
        super().__init__(); self.decoders=nn.ModuleList(nn.Sequential(nn.LayerNorm(latent),nn.Linear(latent,latent),nn.GELU(),nn.Linear(latent,state)) for _ in range(sources)); [nn.init.zeros_(x) for head in self.decoders for x in (head[-1].weight,head[-1].bias)]
    def forward(self,z,source):
        all_state=torch.stack([head(z) for head in self.decoders],1); return all_state[torch.arange(len(z),device=z.device),source]


def dependency_landscape_head(latent=128,state=64):
    head=nn.Sequential(nn.LayerNorm(latent),nn.Linear(latent,latent),nn.GELU(),nn.Linear(latent,state)); nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias); return head


class DependencyActionAdapter(nn.Module):
    def __init__(self,state_dim=1816,latent=128,target=64):
        super().__init__(); self.adapter=nn.Sequential(nn.LayerNorm(state_dim),nn.Linear(state_dim,64,bias=False),nn.GELU(),nn.Linear(64,latent,bias=False)); nn.init.zeros_(self.adapter[-1].weight); self.decoder=dependency_landscape_head(latent,target)
    def action(self,state,base):return base+self.adapter(state)


class DependencyInteraction(nn.Module):
    def __init__(self,base):
        super().__init__(); self.base=base.requires_grad_(False); self.correction=nn.Sequential(nn.LayerNorm(32),nn.Linear(32,16),nn.GELU(),nn.Linear(16,2)); nn.init.zeros_(self.correction[-1].weight); nn.init.zeros_(self.correction[-1].bias)
    def forward(self,z,residual,dependency):return self.base(z,residual)+self.correction(dependency)


def pair_transition_adapter(latent=128,bottleneck=16):
    module=nn.Sequential(nn.LayerNorm(latent),nn.Linear(latent,bottleneck),nn.GELU(),nn.Linear(bottleneck,latent,bias=False)); nn.init.zeros_(module[-1].weight); return module


class DiagonalActionCalibration(nn.Module):
    def __init__(self,latent=128):super().__init__(); self.log_scale=nn.Parameter(torch.zeros(latent))
    def forward(self,action):return action*torch.exp(.1*torch.tanh(self.log_scale))


class LowRankActionRotation(nn.Module):
    def __init__(self,latent=128,rank=8):super().__init__(); self.down=nn.Linear(latent,rank,bias=False); self.up=nn.Linear(rank,latent,bias=False); nn.init.zeros_(self.up.weight)
    def forward(self,action):
        delta=torch.tanh(self.up(nn.functional.gelu(self.down(nn.functional.layer_norm(action,(action.shape[-1],)))))); return action+.1*action.square().mean(1,keepdim=True).sqrt()*delta


class SymmetricPairFusion(nn.Module):
    def __init__(self,latent=128,relations=6,rank=8):super().__init__(); self.down=nn.Linear(3*latent+relations,rank,bias=False); self.up=nn.Linear(rank,96,bias=False); nn.init.zeros_(self.up.weight)
    def forward(self,a,b,joint,relation):
        x=torch.cat((a*b,(a-b).abs(),joint,relation),1); return .25*torch.tanh(self.up(nn.functional.gelu(self.down(nn.functional.layer_norm(x,(x.shape[-1],))))))


class SymmetricPairLatentFusion(nn.Module):
    def __init__(self,latent=128,relations=6,rank=8):super().__init__(); self.down=nn.Linear(3*latent+relations,rank,bias=False); self.up=nn.Linear(rank,latent,bias=False); nn.init.zeros_(self.up.weight)
    def forward(self,a,b,joint,relation):
        x=torch.cat((a*b,(a-b).abs(),joint,relation),1); delta=torch.tanh(self.up(nn.functional.gelu(self.down(nn.functional.layer_norm(x,(x.shape[-1],)))))); return .1*joint.square().mean(1,keepdim=True).sqrt()*delta


class MultiActionComposition(nn.Module):
    def __init__(self,latent=128,rank=16,max_actions=8):
        super().__init__(); width=3*latent+max_actions; self.norm=nn.LayerNorm(width); self.down=nn.Linear(width,rank,bias=False); self.up=nn.Linear(rank,latent,bias=False); self.max_actions=max_actions; nn.init.zeros_(self.up.weight)
    def forward(self,actions,mask,joint):
        weight=mask[...,None]; count=weight.sum(1).clamp_min(1); mean=(actions*weight).sum(1)/count; variance=(actions.square()*weight).sum(1)/count-mean.square(); card=nn.functional.one_hot(mask.sum(1).clamp(1,self.max_actions)-1,self.max_actions).float(); delta=torch.tanh(self.up(nn.functional.gelu(self.down(self.norm(torch.cat((joint,mean,variance,card),1)))))); return .1*joint.square().mean(1,keepdim=True).sqrt()*delta


class GraphHead(nn.Module):
    def __init__(self,dim,views=6):
        super().__init__(); self.input=nn.Linear(dim,256); self.skip=nn.Linear(256,128); self.one=nn.ModuleList(nn.Linear(256,256) for _ in range(views)); self.two=nn.ModuleList(nn.Linear(256,128) for _ in range(views)); self.att=nn.Parameter(torch.zeros(views)); self.scale=nn.Parameter(torch.tensor(2.3)); self.bias=nn.Parameter(torch.zeros(()))
    def forward(self,x,graphs):
        base=nn.functional.gelu(self.input(x)); hs=[]
        for graph,one,two in zip(graphs,self.one,self.two): hs.append(two(torch.sparse.mm(graph,nn.functional.gelu(one(torch.sparse.mm(graph,base))))))
        return nn.functional.normalize(self.skip(base)+sum(w*h for w,h in zip(self.att.softmax(0),hs)),dim=1)
    def score(self,z,p): return (z[p[:,0]]*z[p[:,1]]).sum(1)*self.scale.exp()+self.bias


class SymmetricHead(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.encode=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,384),nn.GELU(),nn.Dropout(.1),nn.Linear(384,128)); self.score=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,1))
    def forward(self,x,p):
        a,b=self.encode(x[p[:,0]]),self.encode(x[p[:,1]]); return self.score(torch.cat(((a-b).abs(),a*b),1)).squeeze(1)


def batches(n, size, shuffle=True):
    ix = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, size): yield ix[i:i+size]


def fit_batch(model,batch):
    return max(256,batch*384//model.cls.numel()//2) if model.cls.numel()>384 else batch


@torch.no_grad()
def relation_loss(model,state,pairs,relations,valid,device):
    model.eval(); total=0.
    for ix in batches(len(valid),4096,False):
        p=pairs[valid[ix]]; a,b=model.encode(torch.cat((state[p[:,0]],state[p[:,1]])).to(device)).chunk(2); total+=nn.functional.smooth_l1_loss(model.relation_score(a,b,model.transition(a,b)[0]),relations[valid[ix]].to(device),reduction="sum").item()
    return total/len(valid)/relations.shape[1]


@torch.no_grad()
def outcome_loss(model,state,pairs,context,target,held,device,context_state=None,gene_context=None):
    model.eval(); total=0.
    for at in batches(len(held),4096,False):
        ix=held[at]; p=pairs[ix]; a,b=model.encode(torch.cat((state[p[:,0]],state[p[:,1]])).to(device)).chunk(2); c=context[ix].to(device); cs=context_features(context_state,gene_context,c,p) if context_state is not None else None; total+=nn.functional.huber_loss(model.outcome(model.transition(a,b,context=c,context_state=cs)[0]),target[ix].to(device),reduction="sum").item()
    return total/len(held)/target.shape[1]


def context_features(context_state,gene_context,context,pairs):
    cs=context_state[context.clamp_min(0)]
    if gene_context is not None:
        pairs=pairs.to(context.device); a=gene_context[context.clamp_min(0),pairs[:,0]]; b=gene_context[context.clamp_min(0),pairs[:,1]]; cs=torch.cat((cs,a+b,(a-b).abs(),a*b,torch.maximum(a,b)),1)
    return torch.where((context>=0)[:,None],cs,torch.zeros_like(cs))


def outcome_split(pairs,cold=False):
    if not cold:
        valid=(pairs[:,0].long()*1000003+pairs[:,1].long())%10==0
        return torch.nonzero(~valid).squeeze(1),torch.nonzero(valid).squeeze(1)
    held_gene=torch.arange(int(pairs.max())+1)%5==0; seen=~held_gene[pairs.long()]
    return torch.nonzero(seen.all(1)).squeeze(1),torch.nonzero((~seen).all(1)).squeeze(1)


@torch.no_grad()
def dependency_loss(model,genes,cells,gene_ids,state,target,known,device,seed=123,n=100000):
    rng=np.random.default_rng(seed); c=rng.choice(cells,n*2); g=rng.choice(gene_ids,n*2); keep=known[c,g]; c,g=c[keep][:n],g[keep][:n]; total=0.
    model.eval()
    for ix in batches(len(c),4096,False):
        ci=torch.as_tensor(c[ix],device=device); gi=torch.as_tensor(g[ix],device=device); pred=model.outcome(model.transition(genes[gi],context_state=state[ci])[0])[:,0]; total+=nn.functional.huber_loss(pred,target[ci,gi],reduction="sum").item()
    return total/len(c)


def pretrain_dependency(model,gene_state,data,device,epochs=3,epoch_pairs=500000,batch=2048):
    genes=torch.as_tensor(encode_genes(model,gene_state,device),device=device); state=torch.as_tensor(data["cell_state"],dtype=torch.float32,device=device); target=torch.as_tensor(data["dependency"].astype("float32"),device=device); known=data["dependency_known"]; allowed=data["train_cell"]; train=np.flatnonzero(allowed&(np.arange(len(allowed))%10!=0)); valid=np.flatnonzero(allowed&(np.arange(len(allowed))%10==0)); gene=np.flatnonzero(data["train_gene"]); model.requires_grad_(False)
    for module in (model.context_proj,model.state_up,model.dist,model.outcome): module.requires_grad_(True)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),3e-4,weight_decay=1e-3); best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; best_val=dependency_loss(model,genes,valid,gene,state,target,known,device)
    for epoch in range(epochs):
        rng=np.random.default_rng(SEED+epoch); total=0.; seen=0; model.eval()
        for _ in range(math.ceil(epoch_pairs/batch)):
            c=rng.choice(train,batch*2); g=rng.choice(gene,batch*2); keep=known[c,g]; c,g=c[keep][:batch],g[keep][:batch]
            if not len(c): continue
            ci=torch.as_tensor(c,device=device); gi=torch.as_tensor(g,device=device); pred=model.outcome(model.transition(genes[gi],context_state=state[ci])[0])[:,0]; loss=nn.functional.huber_loss(pred,target[ci,gi]); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); total+=loss.item()*len(c); seen+=len(c)
        val=dependency_loss(model,genes,valid,gene,state,target,known,device)
        if val<best_val: best_val=val; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(json.dumps({"phase":"dependency","epoch":epoch+1,"loss":total/seen,"val":val}),flush=True)
    model.load_state_dict(best); model.requires_grad_(True); return best_val


def tolerance_head(latent):return nn.Sequential(nn.LayerNorm(latent),nn.Linear(latent,latent),nn.GELU(),nn.Linear(latent,1))


def interaction_head(latent):return nn.Sequential(nn.LayerNorm(latent),nn.Linear(latent,latent),nn.GELU(),nn.Linear(latent,2))


def interaction_depletion_head(latent):return nn.Sequential(nn.LayerNorm(5*latent),nn.Linear(5*latent,2*latent),nn.GELU(),nn.Linear(2*latent,1))


def interaction_depletion_features(model,a,b,context_state):
    la=model.transition(a,context_state=context_state)[0]; lb=model.transition(b,context_state=context_state)[0]; joint=model.transition(a,b,context_state=context_state)[0]; return torch.cat((joint,(la-lb).abs(),la*lb,(a-b).abs(),a*b),1)


class SymmetricQGIHead(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.gene=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,256),nn.GELU(),nn.Linear(256,64)); self.pair=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,128),nn.GELU(),nn.Linear(128,1))
    def forward(self,x,p):
        a,b=self.gene(x[p[:,0]]),self.gene(x[p[:,1]]); return self.pair(torch.cat(((a-b).abs(),a*b),1)).squeeze(1)


@torch.no_grad()
def tolerance_validation(model,head,genes,data,cells,gene_ids,device,seed=731,n=200000):
    rng=np.random.default_rng(seed); c=rng.choice(cells,n*2); g=rng.choice(gene_ids,n*2); keep=data["dependency_known"][c,g]; c,g=c[keep][:n],g[keep][:n]; pred=[]
    for ix in batches(len(c),4096,False):
        ci=torch.as_tensor(c[ix],device=device); gi=torch.as_tensor(g[ix],device=device); z=model.transition(genes[gi],context_state=torch.as_tensor(data["cell_state"][c[ix]],device=device))[0]; pred.append(head(z).squeeze(1).cpu())
    pred=torch.cat(pred).numpy(); target=data["dependency"][c,g].astype("float32"); return {"huber":float(nn.functional.huber_loss(torch.from_numpy(pred),torch.from_numpy(target))),"dependent_auroc":float(roc_auc_score(target<-.5,-pred)),"correlation":float(np.corrcoef(target,pred)[0,1])}


def fit_tolerance_head(data_dir,model_path,out_dir,epochs=5,epoch_pairs=1000000,d=384,latent=128,layers=6):
    device="cuda" if torch.cuda.is_available() else "cpu"; pack=np.load(Path(data_dir)/"features_spectral_safe.npz"); state=pack["state"].astype("float32"); data=np.load(Path(data_dir)/"depmap_tolerance.npz"); sd=torch.load(model_path,map_location="cpu",weights_only=True); context_dim=sd["context_proj.weight"].shape[1]; model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],context_dim).to(device); model.load_state_dict(sd); model.eval().requires_grad_(False); genes=torch.as_tensor(encode_genes(model,state,device),device=device); head=tolerance_head(latent).to(device); train=np.flatnonzero(data["train_cell"]&(np.arange(len(data["train_cell"]))%10!=0)); valid=np.flatnonzero(data["train_cell"]&(np.arange(len(data["train_cell"]))%10==0)); gene=np.flatnonzero(data["train_gene"]); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3); best=None; best_metrics=None
    for epoch in range(epochs):
        rng=np.random.default_rng(SEED+epoch); total=seen=0; head.train()
        for _ in range(math.ceil(epoch_pairs/2048)):
            c=rng.choice(train,4096); g=rng.choice(gene,4096); keep=data["dependency_known"][c,g]; c,g=c[keep][:2048],g[keep][:2048]
            if not len(c):continue
            target=torch.as_tensor(data["dependency"][c,g].astype("float32"),device=device); ci=torch.as_tensor(c,device=device); gi=torch.as_tensor(g,device=device)
            with torch.no_grad():z=model.transition(genes[gi],context_state=torch.as_tensor(data["cell_state"][c],device=device))[0]
            raw=nn.functional.huber_loss(head(z).squeeze(1),target,reduction="none"); weight=1+2*(target<-.5); loss=(raw*weight).sum()/weight.sum(); opt.zero_grad(); loss.backward(); opt.step(); total+=raw.sum().item(); seen+=len(c)
        head.eval(); metrics=tolerance_validation(model,head,genes,data,valid,gene,device)
        if best_metrics is None or metrics["huber"]<best_metrics["huber"]:best={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}; best_metrics=metrics
        print(json.dumps({"phase":"tolerance","epoch":epoch+1,"train_huber":total/seen,**metrics}),flush=True)
    head.load_state_dict(best); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); torch.save(head.state_dict(),out/"tolerance_head.pt"); (out/"tolerance_metrics.json").write_text(json.dumps(best_metrics,indent=2)); return best_metrics


def fit_hap1_qgi_head(data_dir,model_path,out_dir,epochs=8,d=384,latent=128,layers=6):
    device="cuda" if torch.cuda.is_available() else "cpu"; pack=np.load(Path(data_dir)/"features_spectral_safe.npz"); state=pack["state"].astype("float32"); data=np.load(Path(data_dir)/"hap1_quantitative.npz"); context=np.load(Path(data_dir)/"hap1_context.npz"); sd=torch.load(model_path,map_location="cpu",weights_only=True); context_dim=sd["context_proj.weight"].shape[1]; model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],context_dim).to(device); model.load_state_dict(sd); model.eval().requires_grad_(False); genes=torch.as_tensor(encode_genes(model,state,device),device=device); pairs=torch.as_tensor(data["pairs"].astype("int64"),device=device); target=torch.as_tensor(data["target"].astype("float32"),device=device); cs=torch.as_tensor(context["cell_state"][0],device=device); features=torch.empty((len(pairs),latent),device=device)
    with torch.no_grad():
        for start in range(0,len(pairs),8192):
            p=pairs[start:start+8192]; features[start:start+len(p)]=model.transition(genes[p[:,0]],genes[p[:,1]],context_state=cs.expand(len(p),-1))[0]
    held=torch.arange(len(state),device=device)%5==0; train=torch.nonzero((~held[pairs]).all(1)).squeeze(1); valid=torch.nonzero(held[pairs].all(1)).squeeze(1); rows=[]; saved={}
    for name in ("unweighted","magnitude_weighted"):
        torch.manual_seed(20260826); head=tolerance_head(latent).to(device); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3)
        for _ in range(epochs):
            order=train[torch.randperm(len(train),device=device)]; head.train()
            for start in range(0,len(order),8192):
                ix=order[start:start+8192]; raw=nn.functional.huber_loss(head(features[ix]).squeeze(1),target[ix],reduction="none"); weight=1+4*target[ix].abs().clamp_max(1) if name=="magnitude_weighted" else 1.; loss=(raw*weight).mean(); opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():pred=head(features[valid]).squeeze(1).cpu().numpy()
        truth=target[valid].cpu().numpy(); row={"candidate":name,"train_pairs":len(train),"validation_pairs":len(valid),"validation_huber":float(nn.functional.huber_loss(torch.from_numpy(pred),torch.from_numpy(truth))),"validation_pearson":float(np.corrcoef(truth,pred)[0,1]),"validation_spearman":float(spearmanr(truth,pred).statistic)}; rows.append(row); saved[name]={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}; print(json.dumps(row),flush=True)
    selected=sorted(rows,key=lambda x:(-x["validation_spearman"],-x["validation_pearson"],x["candidate"]))[0]; out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); torch.save(saved[selected["candidate"]],out/"hap1_qgi_head.pt"); result={"selected":selected["candidate"],"candidates":rows}; (out/"hap1_qgi_metrics.json").write_text(json.dumps(result,indent=2)); return result


def fit_hap1_qgi_feature_head(data_dir,model_path,out_dir,epochs=8,d=384,latent=128,layers=6):
    device="cuda" if torch.cuda.is_available() else "cpu"; pack=np.load(Path(data_dir)/"features_spectral_safe.npz"); state=pack["state"].astype("float32"); data=np.load(Path(data_dir)/"hap1_quantitative.npz"); sd=torch.load(model_path,map_location="cpu",weights_only=True); context_dim=sd["context_proj.weight"].shape[1]; model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],context_dim).to(device); model.load_state_dict(sd); model.eval().requires_grad_(False); encoded=encode_genes(model,state,device); pairs=torch.as_tensor(data["pairs"].astype("int64"),device=device); target=torch.as_tensor(data["target"].astype("float32"),device=device); held=torch.arange(len(state),device=device)%5==0; train=torch.nonzero((~held[pairs]).all(1)).squeeze(1); valid=torch.nonzero(held[pairs].all(1)).squeeze(1); rows=[]; saved={}
    for name,x in (("world_gene",encoded),("world_static",np.column_stack((encoded,state)))):
        torch.manual_seed(20260826); x=torch.as_tensor(x,dtype=torch.float32,device=device); head=SymmetricQGIHead(x.shape[1]).to(device); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3)
        for _ in range(epochs):
            order=train[torch.randperm(len(train),device=device)]; head.train()
            for start in range(0,len(order),8192):
                ix=order[start:start+8192]; raw=nn.functional.huber_loss(head(x,pairs[ix]),target[ix],reduction="none"); weight=1+4*target[ix].abs().clamp_max(1); loss=(raw*weight).mean(); opt.zero_grad(); loss.backward(); opt.step()
        head.eval(); pred=[]
        with torch.no_grad():
            for start in range(0,len(valid),8192): pred.append(head(x,pairs[valid[start:start+8192]]).cpu())
        pred=torch.cat(pred).numpy(); truth=target[valid].cpu().numpy(); row={"candidate":name,"parameters":sum(p.numel() for p in head.parameters()),"train_pairs":len(train),"validation_pairs":len(valid),"validation_huber":float(nn.functional.huber_loss(torch.from_numpy(pred),torch.from_numpy(truth))),"validation_pearson":float(np.corrcoef(truth,pred)[0,1]),"validation_spearman":float(spearmanr(truth,pred).statistic)}; rows.append(row); saved[name]={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}; print(json.dumps(row),flush=True)
    selected=sorted(rows,key=lambda x:(-x["validation_spearman"],-x["validation_pearson"],x["candidate"]))[0]; out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); torch.save(saved[selected["candidate"]],out/"hap1_qgi_feature_head.pt"); result={"selected":selected["candidate"],"candidates":rows}; (out/"hap1_qgi_feature_metrics.json").write_text(json.dumps(result,indent=2)); return result


@torch.no_grad()
def interaction_validation(model,head,genes,data,rows,device):
    pred=[]
    for at in batches(len(rows),4096,False):
        ix=rows[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); cs=torch.as_tensor(data["context_state"][data["context"][ix]].astype("float32"),device=device); pred.append(head(model.transition(genes[p[:,0]],genes[p[:,1]],context_state=cs)[0]).cpu())
    pred=torch.cat(pred); target=torch.as_tensor(data["target"][rows].astype("float32")); return {"rows":int(len(rows)),"huber":float(nn.functional.huber_loss(pred,target)),"depletion_huber":float(nn.functional.huber_loss(pred[:,0],target[:,0])),"magnitude_huber":float(nn.functional.huber_loss(pred[:,1],target[:,1])),"depletion_correlation":float(np.corrcoef(target[:,0],pred[:,0])[0,1]),"depletion_spearman":float(spearmanr(target[:,0],pred[:,0]).statistic),"magnitude_correlation":float(np.corrcoef(target[:,1],pred[:,1])[0,1])}


def fit_interaction_head(data_dir,model_path,out_dir,epochs=20,epoch_pairs=100000,d=384,latent=128,layers=6):
    device="cuda" if torch.cuda.is_available() else "cpu"; pack=np.load(Path(data_dir)/"features_spectral_safe.npz"); state=pack["state"].astype("float32"); data=np.load(Path(data_dir)/"slkb_outcomes_intervention_depmap_world.npz",allow_pickle=True); sd=torch.load(model_path,map_location="cpu",weights_only=True); context_dim=sd["context_proj.weight"].shape[1]; model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],context_dim).to(device); model.load_state_dict(sd); model.eval().requires_grad_(False); genes=torch.as_tensor(encode_genes(model,state,device),device=device); head=interaction_head(latent).to(device); held=np.arange(int(data["pairs"].max())+1)%5==0; known=data["context_known"][data["context"]]; train=np.flatnonzero(known&(~held[data["pairs"]]).all(1)); valid=np.flatnonzero(known&held[data["pairs"]].all(1)); source=data["context"][train]; weight=1/np.bincount(source)[source]; weight/=weight.sum(); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3); best=None; best_metrics=None
    for epoch in range(epochs):
        rng=np.random.default_rng(SEED+epoch); chosen=rng.choice(train,epoch_pairs,replace=True,p=weight); total=seen=0; head.train()
        for at in batches(len(chosen),2048,False):
            ix=chosen[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); cs=torch.as_tensor(data["context_state"][data["context"][ix]].astype("float32"),device=device); target=torch.as_tensor(data["target"][ix].astype("float32"),device=device)
            with torch.no_grad():z=model.transition(genes[p[:,0]],genes[p[:,1]],context_state=cs)[0]
            raw=nn.functional.huber_loss(head(z),target,reduction="none"); loss=raw.mean(); opt.zero_grad(); loss.backward(); opt.step(); total+=raw.sum().item(); seen+=raw.numel()
        head.eval(); metrics=interaction_validation(model,head,genes,data,valid,device)
        if best_metrics is None or metrics["huber"]<best_metrics["huber"]:best={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}; best_metrics={"epoch":epoch+1,"train_rows":int(len(train)),**metrics}
        print(json.dumps({"phase":"interaction","epoch":epoch+1,"train_huber":total/seen,**metrics}),flush=True)
    head.load_state_dict(best); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); torch.save(head.state_dict(),out/"interaction_head.pt"); (out/"interaction_metrics.json").write_text(json.dumps(best_metrics,indent=2)); return best_metrics


def pair_targets(data,rows):
    keys=[tuple(x) for x in data["pairs"][rows]]; sums={}; counts={}
    for key,value in zip(keys,data["target"][rows,0]):sums[key]=sums.get(key,0.)+float(value); counts[key]=counts.get(key,0)+1
    return np.asarray([sums[key]/counts[key] for key in keys],"float32"),keys


@torch.no_grad()
def interaction_pair_validation(model,head,genes,data,rows,device):
    pred=[]
    for at in batches(len(rows),4096,False):
        ix=rows[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); cs=torch.as_tensor(data["context_state"][data["context"][ix]].astype("float32"),device=device); value=head(model.transition(genes[p[:,0]],genes[p[:,1]],context_state=cs)[0]); pred.append(value[:,0].cpu())
    pred=torch.cat(pred).numpy(); row=data["target"][rows,0].astype("float32"); mean,keys=pair_targets(data,rows); groups={}
    for key,p,y in zip(keys,pred,mean):groups.setdefault(key,[[],y])[0].append(float(p))
    pair_pred=np.asarray([np.mean(x[0]) for x in groups.values()]); pair_true=np.asarray([x[1] for x in groups.values()]); return {"rows":int(len(rows)),"pairs":int(len(groups)),"row_huber":float(nn.functional.huber_loss(torch.from_numpy(pred),torch.from_numpy(row))),"row_correlation":float(np.corrcoef(row,pred)[0,1]),"row_spearman":float(spearmanr(row,pred).statistic),"pair_huber":float(nn.functional.huber_loss(torch.from_numpy(pair_pred),torch.from_numpy(pair_true))),"pair_correlation":float(np.corrcoef(pair_true,pair_pred)[0,1]),"pair_spearman":float(spearmanr(pair_true,pair_pred).statistic)}


def interaction_array_metrics(pred,data,rows):
    row=data["target"][rows,0].astype("float32"); mean,keys=pair_targets(data,rows); groups={}
    for key,p,y in zip(keys,pred,mean):groups.setdefault(key,[[],y])[0].append(float(p))
    pair_pred=np.asarray([np.mean(x[0]) for x in groups.values()]); pair_true=np.asarray([x[1] for x in groups.values()]); return {"row_huber":float(nn.functional.huber_loss(torch.from_numpy(pred),torch.from_numpy(row))),"row_correlation":float(np.corrcoef(row,pred)[0,1]),"row_spearman":float(spearmanr(row,pred).statistic),"pair_huber":float(nn.functional.huber_loss(torch.from_numpy(pair_pred),torch.from_numpy(pair_true))),"pair_correlation":float(np.corrcoef(pair_true,pair_pred)[0,1]),"pair_spearman":float(spearmanr(pair_true,pair_pred).statistic)}


@torch.no_grad()
def evaluate_resolution_ensemble(data_dir,model_paths,head_paths,out_path,weights=(0.,.25,.5,.75,1.),d=384,latent=128,layers=6):
    device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(Path(data_dir)/"features_spectral_safe.npz")["state"].astype("float32"); data=np.load(Path(data_dir)/"slkb_outcomes_intervention_depmap_world.npz",allow_pickle=True); held=np.arange(int(data["pairs"].max())+1)%5==0; rows=np.flatnonzero(data["context_known"][data["context"]]&held[data["pairs"]].all(1)); predictions=[]
    for model_path,head_path in zip(model_paths,head_paths):
        sd=torch.load(model_path,map_location="cpu",weights_only=True); model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],sd["context_proj.weight"].shape[1]).to(device); model.load_state_dict(sd); model.eval(); genes=torch.as_tensor(encode_genes(model,state,device),device=device); head=interaction_head(latent).to(device); head.load_state_dict(torch.load(head_path,map_location="cpu",weights_only=True)); head.eval(); pred=[]
        for at in batches(len(rows),4096,False):
            ix=rows[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); cs=torch.as_tensor(data["context_state"][data["context"][ix]].astype("float32"),device=device); pred.append(head(model.transition(genes[p[:,0]],genes[p[:,1]],context_state=cs)[0])[:,0].cpu())
        predictions.append(torch.cat(pred).numpy())
    result=[]
    for weight in weights:
        pred=(1-weight)*predictions[0]+weight*predictions[1]; metrics=interaction_array_metrics(pred,data,rows); result.append({"state96_weight":weight,"rows":len(rows),"pairs":66,**metrics,"selection_loss":metrics["row_huber"]+metrics["pair_huber"]})
    selected=min(result,key=lambda x:(x["selection_loss"],x["state96_weight"])); output={"selection":"minimum row plus pair-mean Huber; smallest weight breaks ties","selected":selected,"candidates":result}; Path(out_path).write_text(json.dumps(output,indent=2)); return output


def fit_interaction_shrinkage_head(data_dir,model_path,out_dir,epochs=20,epoch_pairs=100000,shrinkage=.15,d=384,latent=128,layers=6,feature_name="features_spectral_safe.npz",data_name="slkb_outcomes_intervention_depmap_world.npz",head_name="interaction_shrinkage"):
    device="cuda" if torch.cuda.is_available() else "cpu"; pack=np.load(Path(data_dir)/feature_name); state=pack["state"].astype("float32"); data=np.load(Path(data_dir)/data_name,allow_pickle=True); sd=torch.load(model_path,map_location="cpu",weights_only=True); context_dim=sd["context_proj.weight"].shape[1]; model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],context_dim).to(device); model.load_state_dict(sd); model.eval().requires_grad_(False); genes=torch.as_tensor(encode_genes(model,state,device),device=device); head=interaction_head(latent).to(device); held=np.arange(int(data["pairs"].max())+1)%5==0; known=data["context_known"][data["context"]]; train=np.flatnonzero(known&(~held[data["pairs"]]).all(1)); valid=np.flatnonzero(known&held[data["pairs"]].all(1)); mean,_=pair_targets(data,train); source=data["context"][train]; weight=1/np.bincount(source)[source]; weight/=weight.sum(); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3); best=None; best_metrics=None
    for epoch in range(epochs):
        rng=np.random.default_rng(SEED+epoch); chosen=rng.choice(len(train),epoch_pairs,replace=True,p=weight); total=seen=0; head.train()
        for at in batches(len(chosen),2048,False):
            take=chosen[at]; ix=train[take]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); cs=torch.as_tensor(data["context_state"][data["context"][ix]].astype("float32"),device=device); target=torch.as_tensor(data["target"][ix].astype("float32"),device=device); target[:,0]=(1-shrinkage)*target[:,0]+shrinkage*torch.as_tensor(mean[take],device=device)
            with torch.no_grad():z=model.transition(genes[p[:,0]],genes[p[:,1]],context_state=cs)[0]
            raw=nn.functional.huber_loss(head(z),target,reduction="none"); loss=raw.mean(); opt.zero_grad(); loss.backward(); opt.step(); total+=raw.sum().item(); seen+=raw.numel()
        head.eval(); metrics=interaction_validation(model,head,genes,data,valid,device); pair=interaction_pair_validation(model,head,genes,data,valid,device); score=metrics["depletion_huber"]+pair["pair_huber"]
        metrics.update({k:v for k,v in pair.items() if k.startswith("pair_")}); metrics["selection_loss"]=score
        if best_metrics is None or score<best_metrics["selection_loss"]:best={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}; best_metrics={"epoch":epoch+1,"train_rows":int(len(train)),"shrinkage":shrinkage,**metrics}
        print(json.dumps({"phase":"interaction_shrinkage","epoch":epoch+1,"train_huber":total/seen,**metrics}),flush=True)
    head.load_state_dict(best); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); torch.save(head.state_dict(),out/f"{head_name}_head.pt"); (out/f"{head_name}_metrics.json").write_text(json.dumps(best_metrics,indent=2)); return best_metrics


def fit_kg_interaction_residual(data_dir,model_path,head_path,out_dir,d=384,latent=128,layers=6,alpha=100.):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    device="cuda" if torch.cuda.is_available() else "cpu"; pack=np.load(Path(data_dir)/"features_spectral_safe.npz"); kg=np.load(Path(data_dir)/"features_spectral_kg_safe.npz"); state=pack["state"].astype("float32"); data=np.load(Path(data_dir)/"slkb_outcomes_intervention_depmap_world.npz",allow_pickle=True); sd=torch.load(model_path,map_location="cpu",weights_only=True); model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],sd["context_proj.weight"].shape[1]).to(device); model.load_state_dict(sd); model.eval().requires_grad_(False); genes=torch.as_tensor(encode_genes(model,state,device),device=device); head=interaction_head(latent).to(device); head.load_state_dict(torch.load(head_path,map_location="cpu",weights_only=True)); head.eval(); held=np.arange(int(data["pairs"].max())+1)%5==0; known=data["context_known"][data["context"]]; train=np.flatnonzero(known&(~held[data["pairs"]]).all(1)); valid=np.flatnonzero(known&held[data["pairs"]].all(1)); pred=[]
    with torch.no_grad():
        for at in batches(len(data["pairs"]),4096,False):
            p=torch.as_tensor(data["pairs"][at].astype("int64"),device=device); cs=torch.as_tensor(data["context_state"][data["context"][at]].astype("float32"),device=device); pred.append(head(model.transition(genes[p[:,0]],genes[p[:,1]],context_state=cs)[0])[:,0].cpu())
    base=torch.cat(pred).numpy(); emb=kg["state"][:,1024:1152].astype("float32"); hit=kg["kg_hit"]
    def x(p):
        a,b=emb[p[:,0]],emb[p[:,1]]; den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-6; return np.column_stack(((a-b).__abs__(),a*b,(a*b).sum(1)/den,np.abs(a-b).mean(1),((a-b)**2).mean(1),hit[p[:,0]],hit[p[:,1]],hit[p[:,0]]&hit[p[:,1]]))
    pairs,inv=np.unique(data["pairs"][train],axis=0,return_inverse=True); residual=data["target"][train,0]-base[train]; target=np.bincount(inv,residual)/np.bincount(inv); scale=StandardScaler().fit(x(pairs)); ridge=Ridge(alpha=alpha).fit(scale.transform(x(pairs)),target); adjusted=base.copy(); adjusted[valid]+=ridge.predict(scale.transform(x(data["pairs"][valid])))
    def metrics(score,rows):
        y=data["target"][rows,0]; pairs,inv=np.unique(data["pairs"][rows],axis=0,return_inverse=True); yp=np.bincount(inv,score[rows])/np.bincount(inv); yt=np.bincount(inv,y)/np.bincount(inv); return {"row_huber":float(nn.functional.huber_loss(torch.from_numpy(score[rows]),torch.from_numpy(y))),"row_correlation":float(np.corrcoef(y,score[rows])[0,1]),"row_spearman":float(spearmanr(y,score[rows]).statistic),"pair_huber":float(nn.functional.huber_loss(torch.from_numpy(yp),torch.from_numpy(yt))),"pair_correlation":float(np.corrcoef(yt,yp)[0,1]),"pair_spearman":float(spearmanr(yt,yp).statistic)}
    result={"alpha":alpha,"features":int(scale.mean_.size),"train_rows":len(train),"train_pairs":len(pairs),"validation_rows":len(valid),"base":metrics(base,valid),"kg_residual":metrics(adjusted,valid)}; out=Path(out_dir); np.savez(out/"kg_interaction_residual.npz",mean=scale.mean_,scale=scale.scale_,coef=ridge.coef_,intercept=ridge.intercept_); (out/"kg_interaction_residual_metrics.json").write_text(json.dumps(result,indent=2)); return result


@torch.no_grad()
def depletion_validation(model,head,genes,data,rows,device):
    pred=[]
    for at in batches(len(rows),4096,False):
        ix=rows[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); cs=torch.as_tensor(data["context_state"][data["context"][ix]].astype("float32"),device=device); pred.append(head(interaction_depletion_features(model,genes[p[:,0]],genes[p[:,1]],cs)).squeeze(1).cpu())
    pred=torch.cat(pred); target=torch.as_tensor(data["target"][rows,0].astype("float32")); return {"rows":int(len(rows)),"depletion_huber":float(nn.functional.huber_loss(pred,target)),"depletion_correlation":float(np.corrcoef(target,pred)[0,1])}


def fit_interaction_depletion_head(data_dir,model_path,out_dir,epochs=20,epoch_pairs=100000,d=384,latent=128,layers=6):
    device="cuda" if torch.cuda.is_available() else "cpu"; pack=np.load(Path(data_dir)/"features_spectral_safe.npz"); state=pack["state"].astype("float32"); data=np.load(Path(data_dir)/"slkb_outcomes_intervention_depmap_world.npz",allow_pickle=True); sd=torch.load(model_path,map_location="cpu",weights_only=True); context_dim=sd["context_proj.weight"].shape[1]; model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],context_dim).to(device); model.load_state_dict(sd); model.eval().requires_grad_(False); genes=torch.as_tensor(encode_genes(model,state,device),device=device); head=interaction_depletion_head(latent).to(device); held=np.arange(int(data["pairs"].max())+1)%5==0; known=data["context_known"][data["context"]]; train=np.flatnonzero(known&(~held[data["pairs"]]).all(1)); valid=np.flatnonzero(known&held[data["pairs"]].all(1)); source=data["context"][train]; weight=1/np.bincount(source)[source]; weight/=weight.sum(); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3); best=None; best_metrics=None
    for epoch in range(epochs):
        rng=np.random.default_rng(SEED+epoch); chosen=rng.choice(train,epoch_pairs,replace=True,p=weight); total=seen=0; head.train()
        for at in batches(len(chosen),1024,False):
            ix=chosen[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); cs=torch.as_tensor(data["context_state"][data["context"][ix]].astype("float32"),device=device); target=torch.as_tensor(data["target"][ix,0].astype("float32"),device=device)
            with torch.no_grad():features=interaction_depletion_features(model,genes[p[:,0]],genes[p[:,1]],cs)
            raw=nn.functional.huber_loss(head(features).squeeze(1),target,reduction="none"); loss=raw.mean(); opt.zero_grad(); loss.backward(); opt.step(); total+=raw.sum().item(); seen+=len(raw)
        head.eval(); metrics=depletion_validation(model,head,genes,data,valid,device)
        if best_metrics is None or metrics["depletion_huber"]<best_metrics["depletion_huber"]:best={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}; best_metrics={"epoch":epoch+1,"train_rows":int(len(train)),**metrics}
        print(json.dumps({"phase":"interaction_depletion","epoch":epoch+1,"train_huber":total/seen,**metrics}),flush=True)
    head.load_state_dict(best); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); torch.save(head.state_dict(),out/"interaction_depletion_head.pt"); (out/"interaction_depletion_metrics.json").write_text(json.dumps(best_metrics,indent=2)); return best_metrics


@torch.no_grad()
def perturbseq_layout(data):
    width=data["target"].shape[1]; state_dim=int(data["state_dimensions"]) if "state_dimensions" in data.files else (32 if width==33 else width); fitness_index=int(data["fitness_index"]) if "fitness_index" in data.files else (32 if width==33 else -1)
    if state_dim<1 or state_dim>width or fitness_index>=width: raise ValueError("Invalid Perturb-seq target layout")
    return state_dim,fitness_index


def perturbseq_state_error(raw,data,state_dim,block=None):
    if block is not None:
        ends=data["state_blocks"].astype("int64"); lo=0 if block==0 else int(ends[block-1]); return raw[...,lo:int(ends[block])].mean(-1)
    if "state_blocks" not in data.files:return raw[...,:state_dim].mean(-1)
    ends=data["state_blocks"].astype("int64"); weights=data["state_block_weights"].astype("float32"); return sum(float(w)*raw[...,(0 if i==0 else int(ends[i-1])):int(ends[i])].mean(-1) for i,w in enumerate(weights))


def perturbseq_loss(model,decoder,genes,data,device,singles=None,residual_weight=0.,source_context=False,state_block=None):
    model.eval(); decoder.eval(); keep=np.flatnonzero(data["role"]==1); total=pair_total=0.; seen=pairs_seen=0; state_dim,fitness_index=perturbseq_layout(data)
    for at in batches(len(keep),4096,False):
        ix=keep[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); a=genes[p[:,0]]; double=p[:,1]>=0; c=torch.as_tensor(data["source"][ix],device=device,dtype=torch.long) if source_context and "context_state" not in data.files else None; cs=torch.as_tensor(data["context_state"][data["source"][ix]],device=device) if source_context and "context_state" in data.files else None; pred=torch.empty((len(ix),decoder.out_features),device=device)
        if (~double).any(): pred[~double]=decoder(model.transition(a[~double],context=c[~double] if c is not None else None,context_state=cs[~double] if cs is not None else None)[0])
        if double.any(): pred[double]=decoder(model.transition(a[double],genes[p[double,1]],context=c[double] if c is not None else None,context_state=cs[double] if cs is not None else None)[0])
        target=torch.as_tensor(data["target"][ix],device=device); raw=nn.functional.huber_loss(pred,target,reduction="none"); loss=perturbseq_state_error(raw,data,state_dim,state_block)+(.5*raw[:,fitness_index] if fitness_index>=0 else 0)
        if residual_weight and double.any():
            si=singles[np.asarray(ix)[double.cpu().numpy()]]; dc=c[double] if c is not None else None; ds=cs[double] if cs is not None else None; ca=decoder(model.transition(a[double],context=dc,context_state=ds)[0]); cb=decoder(model.transition(genes[p[double,1]],context=dc,context_state=ds)[0]); truth=target[double]-torch.as_tensor(data["target"][si[:,0]],device=device)-torch.as_tensor(data["target"][si[:,1]],device=device); residual=nn.functional.huber_loss(pred[double,:state_dim]-ca[:,:state_dim]-cb[:,:state_dim],truth[:,:state_dim],reduction="none"); loss[double]+=residual_weight*perturbseq_state_error(residual,data,state_dim,state_block)
        total+=loss.sum().item(); seen+=len(ix); pair_total+=loss[double].sum().item(); pairs_seen+=double.sum().item()
    return total/seen,pair_total/max(1,pairs_seen)


def pretrain_perturbseq(model,gene_state,data,device,epochs=10,rl_epochs=3,batch=512,residual_weight=0.,context_selection=False):
    genes=torch.as_tensor(encode_genes(model,gene_state,device),device=device); decoder=nn.Linear(model.gene.out_features,data["target"].shape[1]).to(device); state_dim,fitness_index=perturbseq_layout(data); model.requires_grad_(False)
    for module in (model.state_up,model.dist,model.cell,model.context_proj):
        if module is not None: module.requires_grad_(True)
    for parameter in (model.basal,model.time,model.context): parameter.requires_grad_(True)
    lookup={(int(s),int(a)):i for i,(s,(a,b)) in enumerate(zip(data["source"],data["pairs"])) if b<0}; singles=np.asarray([[lookup.get((int(s),int(a)),-1),lookup.get((int(s),int(b)),-1)] for s,(a,b) in zip(data["source"],data["pairs"])],"int64"); train=np.flatnonzero(data["role"]==0); source=data["source"][train]; double=data["cardinality"][train]==2; weights=1/np.maximum(1,np.bincount(source)[source]); weights*=np.where(double,5.,1.); weights/=weights.sum(); opt=torch.optim.AdamW([*filter(lambda p:p.requires_grad,model.parameters()),*decoder.parameters()],3e-4,weight_decay=1e-3); best=None; best_score=float("inf")
    for epoch in range(epochs):
        rng=np.random.default_rng(SEED+epoch); chosen=rng.choice(train,len(train),replace=True,p=weights); model.eval(); decoder.train(); total=0.
        for at in batches(len(chosen),batch,False):
            ix=chosen[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); a=genes[p[:,0]]; second=p[:,1]>=0; c=torch.as_tensor(data["source"][ix].astype("int64"),device=device) if "context_state" not in data.files else None; cs=torch.as_tensor(data["context_state"][data["source"][ix]],device=device) if "context_state" in data.files else None; drop=torch.rand(len(ix),device=device)<.5; c=torch.where(drop,-1,c) if c is not None else None; cs=torch.where(drop[:,None],torch.zeros_like(cs),cs) if cs is not None else None; pred=torch.empty((len(ix),decoder.out_features),device=device)
            if (~second).any(): pred[~second]=decoder(model.transition(a[~second],context=c[~second] if c is not None else None,context_state=cs[~second] if cs is not None else None)[0])
            if second.any(): pred[second]=decoder(model.transition(a[second],genes[p[second,1]],context=c[second] if c is not None else None,context_state=cs[second] if cs is not None else None)[0])
            target=torch.as_tensor(data["target"][ix],device=device); raw=nn.functional.huber_loss(pred,target,reduction="none"); loss=perturbseq_state_error(raw,data,state_dim)+(.5*raw[:,fitness_index] if fitness_index>=0 else 0)
            if residual_weight and second.any():
                si=singles[np.asarray(ix)[second.cpu().numpy()]]; dc=c[second] if c is not None else None; ds=cs[second] if cs is not None else None; ca=decoder(model.transition(a[second],context=dc,context_state=ds)[0]); cb=decoder(model.transition(genes[p[second,1]],context=dc,context_state=ds)[0]); truth=target[second]-torch.as_tensor(data["target"][si[:,0]],device=device)-torch.as_tensor(data["target"][si[:,1]],device=device); residual=nn.functional.huber_loss(pred[second,:state_dim]-ca[:,:state_dim]-cb[:,:state_dim],truth[:,:state_dim],reduction="none"); loss[second]+=residual_weight*perturbseq_state_error(residual,data,state_dim)
            loss=loss.mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_([*model.parameters(),*decoder.parameters()],1.); opt.step(); total+=loss.item()*len(ix)
        val,pair=perturbseq_loss(model,decoder,genes,data,device,singles,residual_weight); source_val,source_pair=perturbseq_loss(model,decoder,genes,data,device,singles,residual_weight,True); score=val+pair+(source_val+source_pair if context_selection else 0)
        if score<best_score: best_score=score; best=({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},{k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()})
        print(json.dumps({"phase":"perturbseq","epoch":epoch+1,"loss":total/len(chosen),"val":val,"val_double":pair,"source_val":source_val,"source_val_double":source_pair}),flush=True)
    model.load_state_dict(best[0]); decoder.load_state_dict(best[1]); opt=torch.optim.AdamW([*filter(lambda p:p.requires_grad,model.parameters()),*decoder.parameters()],2e-5,weight_decay=1e-3)
    for epoch in range(rl_epochs):
        rng=np.random.default_rng(SEED+100+epoch); chosen=rng.choice(train,len(train),replace=True,p=weights); model.eval(); decoder.train(); total=0.
        for at in batches(len(chosen),batch,False):
            ix=chosen[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); a=genes[p[:,0]]; second=p[:,1]>=0; c=torch.full((len(ix),),-1,device=device,dtype=torch.long) if "context_state" not in data.files else None; cs=torch.as_tensor(data["context_state"][data["source"][ix]],device=device) if "context_state" in data.files else None; drop=torch.rand(len(ix),device=device)<.5; cs=torch.where(drop[:,None],torch.zeros_like(cs),cs) if cs is not None else None; mu=torch.empty((len(ix),model.gene.out_features),device=device); logsd=torch.empty_like(mu)
            if (~second).any(): mu[~second],logsd[~second]=model.transition(a[~second],context=c[~second] if c is not None else None,context_state=cs[~second] if cs is not None else None)
            if second.any(): mu[second],logsd[second]=model.transition(a[second],genes[p[second,1]],context=c[second] if c is not None else None,context_state=cs[second] if cs is not None else None)
            sd=logsd.exp(); sample=(mu[:,None]+sd[:,None]*torch.randn(len(ix),4,mu.shape[1],device=device)).detach(); pred=decoder(sample.reshape(-1,mu.shape[1])).view(len(ix),4,-1); target=torch.as_tensor(data["target"][ix],device=device); error=perturbseq_state_error((pred[:,:,:state_dim]-target[:,None,:state_dim]).square(),data,state_dim)+(.5*(pred[:,:,fitness_index]-target[:,None,fitness_index]).square() if fitness_index>=0 else 0)
            if residual_weight and second.any():
                si=singles[np.asarray(ix)[second.cpu().numpy()]]; ca=decoder(model.transition(a[second])[0]); cb=decoder(model.transition(genes[p[second,1]])[0]); truth=target[second]-torch.as_tensor(data["target"][si[:,0]],device=device)-torch.as_tensor(data["target"][si[:,1]],device=device); error[second]+=residual_weight*perturbseq_state_error((pred[second,:,:state_dim]-ca[:,None,:state_dim]-cb[:,None,:state_dim]-truth[:,None,:state_dim]).square(),data,state_dim)
            reward=-error; advantage=(reward-reward.mean(1,keepdim=True))/(reward.std(1,keepdim=True)+1e-5); logp=torch.distributions.Normal(mu[:,None],sd[:,None]).log_prob(sample).mean(2); loss=-(advantage.detach()*logp).mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_([*model.parameters(),*decoder.parameters()],1.); opt.step(); total+=loss.item()*len(ix)
        val,pair=perturbseq_loss(model,decoder,genes,data,device,singles,residual_weight); source_val,source_pair=perturbseq_loss(model,decoder,genes,data,device,singles,residual_weight,True); score=val+pair+(source_val+source_pair if context_selection else 0)
        if score<best_score: best_score=score; best=({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},{k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()})
        print(json.dumps({"phase":"perturbseq_reinforce","epoch":epoch+1,"loss":total/len(chosen),"val":val,"val_double":pair,"source_val":source_val,"source_val_double":source_pair}),flush=True)
    model.load_state_dict(best[0]); decoder.load_state_dict(best[1]); model.requires_grad_(True); return decoder,best_score


@torch.no_grad()
def perturbseq_scores(model,decoder,genes,pairs,device):
    genes=torch.as_tensor(genes,dtype=torch.float32); pairs=torch.as_tensor(pairs.astype("int64")); out=[]; fitness=[]; sequential=[]; order=[]; model.eval(); decoder.eval()
    for ix in batches(len(pairs),4096,False):
        p=pairs[ix]; a,b=genes[p[:,0]].to(device),genes[p[:,1]].to(device); la,lb=model.transition(a)[0],model.transition(b)[0]; joint=decoder(model.transition(a,b)[0]); single_a,single_b=decoder(la),decoder(lb); ab,ba=decoder(model.transition(b,state=la)[0]),decoder(model.transition(a,state=lb)[0]); stop=32 if decoder.out_features==33 else decoder.out_features; out.append(torch.linalg.vector_norm(joint[:,:stop]-single_a[:,:stop]-single_b[:,:stop],dim=1).cpu()); sequential.append(torch.linalg.vector_norm(joint[:,:stop]-(ab[:,:stop]+ba[:,:stop])/2,dim=1).cpu()); order.append(torch.linalg.vector_norm(ab[:,:stop]-ba[:,:stop],dim=1).cpu()); fitness.append((single_a[:,32]+single_b[:,32]-joint[:,32]).cpu()) if decoder.out_features==33 else None
    return torch.cat(out).numpy(),torch.cat(fitness).numpy() if fitness else None,torch.cat(sequential).numpy(),torch.cat(order).numpy()


def perturbseq_block_scores(model,decoder,gene_state,data,device):
    if "state_blocks" not in data.files or len(data["state_blocks"])<2:return {}
    genes=torch.as_tensor(encode_genes(model,gene_state,device),device=device); result={}
    for block,end in enumerate(data["state_blocks"]):
        val,pair=perturbseq_loss(model,decoder,genes,data,device,state_block=block); source_val,source_pair=perturbseq_loss(model,decoder,genes,data,device,source_context=True,state_block=block); result[f"state_block_{int(end)}_score"]=val+pair+source_val+source_pair
    return result


@torch.no_grad()
def residual_endpoint_score(endpoint,genes,data,device,block,source_context=False):
    keep=np.flatnonzero(data["role"]==1); lo=0 if block==0 else int(data["state_blocks"][block-1]); hi=int(data["state_blocks"][block]); total=pair_total=seen=pairs_seen=0
    for at in batches(len(keep),4096,False):
        ix=keep[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); double=p[:,1]>=0; cs=torch.as_tensor(data["context_state"][data["source"][ix]],device=device) if source_context else None; pred=torch.empty((len(ix),hi-lo),device=device)
        if (~double).any():pred[~double]=endpoint(genes[p[~double,0]],context_state=cs[~double] if cs is not None else None)[0][:,lo:hi]
        if double.any():pred[double]=endpoint(genes[p[double,0]],genes[p[double,1]],context_state=cs[double] if cs is not None else None)[0][:,lo:hi]
        loss=nn.functional.huber_loss(pred,torch.as_tensor(data["target"][ix,lo:hi],device=device),reduction="none").mean(1); total+=loss.sum().item(); seen+=len(ix); pair_total+=loss[double].sum().item(); pairs_seen+=double.sum().item()
    return total/seen+pair_total/max(1,pairs_seen)


def fit_residual_endpoint(data_dir,model_path,decoder_path,out_dir,epochs=10,stochastic_epochs=3,d=384,latent=128,layers=6,perturb_name="perturbseq_world_v3_nested96.npz"):
    torch.manual_seed(731); np.random.seed(731); device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(Path(data_dir)/"features_spectral_safe.npz")["state"].astype("float32"); data=np.load(Path(data_dir)/perturb_name); sd=torch.load(model_path,map_location="cpu",weights_only=True); world=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],sd["context_proj.weight"].shape[1]).to(device); world.load_state_dict(sd); ds=torch.load(decoder_path,map_location="cpu",weights_only=True); legacy=nn.Linear(latent,ds["weight"].shape[0]).to(device); legacy.load_state_dict(ds); endpoint=ResidualEndpoint(world,legacy,int(data["state_blocks"][-1]-data["state_blocks"][0])).to(device).eval(); genes=torch.as_tensor(encode_genes(world,state,device),device=device); lo,hi=map(int,data["state_blocks"])
    train=np.flatnonzero(data["role"]==0); source=data["source"][train]; double=data["cardinality"][train]==2; weights=1/np.maximum(1,np.bincount(source)[source]); weights*=np.where(double,5.,1.); weights/=weights.sum(); opt=torch.optim.AdamW(endpoint.residual_decoder.parameters(),3e-4,weight_decay=1e-3); best=None; best_score=float("inf"); history=[]
    def validate(phase,epoch):
        nonlocal best,best_score
        legacy_score=sum(residual_endpoint_score(endpoint,genes,data,device,0,x) for x in (False,True)); residual_score=sum(residual_endpoint_score(endpoint,genes,data,device,1,x) for x in (False,True)); score=(legacy_score+residual_score)/2; row={"phase":phase,"epoch":epoch,"legacy_score":legacy_score,"residual_score":residual_score,"combined_score":score}; history.append(row); print(json.dumps(row),flush=True)
        if score<best_score:best_score=score; best={k:v.detach().cpu().clone() for k,v in endpoint.state_dict().items()}
    validate("initial",0)
    for phase,n,lr,samples in (("supervised",epochs,3e-4,1),("stochastic",stochastic_epochs,2e-5,4)):
        for group in opt.param_groups:group["lr"]=lr
        for epoch in range(n):
            chosen=np.random.default_rng(731+epoch+(100 if phase=="stochastic" else 0)).choice(train,len(train),replace=True,p=weights); endpoint.train(); endpoint.world.eval(); endpoint.legacy_decoder.eval()
            for at in batches(len(chosen),512,False):
                ix=chosen[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); second=p[:,1]>=0; cs=torch.as_tensor(data["context_state"][data["source"][ix]],device=device); drop=torch.rand(len(ix),device=device)<.5; cs=torch.where(drop[:,None],torch.zeros_like(cs),cs); target=torch.as_tensor(data["target"][ix,lo:hi],device=device); loss=0.
                for _ in range(samples):
                    pred=torch.empty_like(target)
                    if (~second).any():pred[~second]=endpoint(genes[p[~second,0]],context_state=cs[~second],sample=torch.randn((~second).sum(),latent,device=device) if samples>1 else None)[0][:,lo:hi]
                    if second.any():pred[second]=endpoint(genes[p[second,0]],genes[p[second,1]],context_state=cs[second],sample=torch.randn(second.sum(),latent,device=device) if samples>1 else None)[0][:,lo:hi]
                    loss=loss+nn.functional.huber_loss(pred,target)/samples
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(endpoint.residual_decoder.parameters(),1.); opt.step()
            validate(phase,epoch+1)
    endpoint.load_state_dict(best); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); torch.save(endpoint.state_dict(),out/"world_model.pt"); metrics={"parameters":sum(p.numel() for p in endpoint.parameters()),"trainable_parameters":sum(p.numel() for p in endpoint.parameters() if p.requires_grad),"best_combined_score":best_score,"history":history}; (out/"basal_metrics.json").write_text(json.dumps(metrics,indent=2)); return metrics


def load_residual_endpoint(path,state_dim,device,d=384,latent=128,layers=6):
    sd=torch.load(path,map_location="cpu",weights_only=True); ws={k[6:]:v for k,v in sd.items() if k.startswith("world.")}; world=SLPredict(d,latent,layers,ws["cell.weight"].shape[0],ws["outcome.weight"].shape[0],state_dim,ws["context_proj.weight"].shape[1]).to(device); legacy=nn.Linear(latent,sd["legacy_decoder.weight"].shape[0]).to(device); endpoint=ResidualEndpoint(world,legacy,sd["residual_decoder.3.weight"].shape[0]).to(device); endpoint.load_state_dict(sd); return endpoint.eval().requires_grad_(False)


@torch.no_grad()
def residual_interaction_inputs(endpoint,genes,pairs,context_state):
    a,b=genes[pairs[:,0]],genes[pairs[:,1]]; z=endpoint.world.transition(a,b,context_state=context_state)[0]; joint=endpoint.residual_decoder(z); sa=endpoint.residual_decoder(endpoint.world.transition(a,context_state=context_state)[0]); sb=endpoint.residual_decoder(endpoint.world.transition(b,context_state=context_state)[0]); return z,torch.cat((joint,joint-sa-sb),1)


@torch.no_grad()
def full_endpoint_composition(endpoint,genes,pairs,context_state):
    a,b=genes[pairs[:,0]],genes[pairs[:,1]]; z=endpoint.world.transition(a,b,context_state=context_state)[0]; za=endpoint.world.transition(a,context_state=context_state)[0]; zb=endpoint.world.transition(b,context_state=context_state)[0]
    decode=lambda x:torch.cat((endpoint.legacy_decoder(x),endpoint.residual_decoder(x)),1); joint,sa,sb=decode(z),decode(za),decode(zb); residual=torch.cat((joint[:,32:],joint[:,32:]-sa[:,32:]-sb[:,32:]),1); return z,residual,torch.cat((joint,joint-sa-sb),1)


def fit_residual_interaction(data_dir,endpoint_path,base_head_path,out_dir,epochs=20,epoch_pairs=100000,shrinkage=.15,d=384,latent=128,layers=6):
    torch.manual_seed(731); np.random.seed(731); device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(Path(data_dir)/"features_spectral_safe.npz")["state"].astype("float32"); data=np.load(Path(data_dir)/"slkb_outcomes_intervention_depmap_world.npz",allow_pickle=True); endpoint=load_residual_endpoint(endpoint_path,state.shape[1],device,d,latent,layers); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); base=interaction_head(latent).to(device); base.load_state_dict(torch.load(base_head_path,map_location="cpu",weights_only=True)); head=ResidualInteraction(base).to(device); held=np.arange(int(data["pairs"].max())+1)%5==0; known=data["context_known"][data["context"]]; train=np.flatnonzero(known&(~held[data["pairs"]]).all(1)); valid=np.flatnonzero(known&held[data["pairs"]].all(1)); mean,_=pair_targets(data,train); source=data["context"][train]; weight=1/np.bincount(source)[source]; weight/=weight.sum(); opt=torch.optim.AdamW(head.correction.parameters(),3e-4,weight_decay=1e-3); candidates=[]; saved=[]
    @torch.no_grad()
    def validate(epoch):
        pred=[]
        for at in batches(len(valid),4096,False):
            ix=valid[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); cs=torch.as_tensor(data["context_state"][data["context"][ix]].astype("float32"),device=device); z,r=residual_interaction_inputs(endpoint,genes,p,cs); pred.append(head(z,r)[:,0].cpu())
        metrics={"epoch":epoch,**interaction_array_metrics(torch.cat(pred).numpy(),data,valid)}; metrics["selection_loss"]=metrics["row_huber"]+metrics["pair_huber"]; candidates.append(metrics); saved.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()}); print(json.dumps(metrics),flush=True)
    validate(0)
    for epoch in range(epochs):
        chosen=np.random.default_rng(731+epoch).choice(len(train),epoch_pairs,replace=True,p=weight); head.train(); endpoint.eval()
        for at in batches(len(chosen),2048,False):
            take=chosen[at]; ix=train[take]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); cs=torch.as_tensor(data["context_state"][data["context"][ix]].astype("float32"),device=device); target=torch.as_tensor(data["target"][ix].astype("float32"),device=device); target[:,0]=(1-shrinkage)*target[:,0]+shrinkage*torch.as_tensor(mean[take],device=device)
            with torch.no_grad():z,r=residual_interaction_inputs(endpoint,genes,p,cs)
            loss=nn.functional.huber_loss(head(z,r),target); opt.zero_grad(); loss.backward(); opt.step()
        head.eval(); validate(epoch+1)
    baseline=candidates[0]; eligible=[i for i,x in enumerate(candidates) if x["row_huber"]<=1.01*baseline["row_huber"] and x["pair_huber"]<=1.01*baseline["pair_huber"] and x["row_correlation"]>=.99*baseline["row_correlation"] and x["pair_correlation"]>=.99*baseline["pair_correlation"]]; selected=min(eligible,key=lambda i:candidates[i]["selection_loss"]); head.load_state_dict(saved[selected]); metric=candidates[selected]; advanced=(metric["row_correlation"]-baseline["row_correlation"]>=.01 or metric["pair_correlation"]-baseline["pair_correlation"]>=.01 or metric["row_huber"]<=.98*baseline["row_huber"] or metric["pair_huber"]<=.98*baseline["pair_huber"]); out=Path(out_dir); torch.save(head.state_dict(),out/"interaction_residual_head.pt"); result={"baseline":baseline,"selected":metric,"advanced":advanced,"candidates":candidates}; (out/"interaction_residual_metrics.json").write_text(json.dumps(result,indent=2)); return result


def fit_interaction_rank_adapter(data_dir,endpoint_path,base_head_path,out_dir,epochs=10,epoch_pairs=100000,margin=.25,shrinkage=.15,d=384,latent=128,layers=6):
    torch.manual_seed(731); np.random.seed(731); device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(Path(data_dir)/"features_spectral_safe.npz")["state"].astype("float32"); data=np.load(Path(data_dir)/"slkb_outcomes_intervention_depmap_world.npz",allow_pickle=True); endpoint=load_residual_endpoint(endpoint_path,state.shape[1],device,d,latent,layers); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); base=ResidualInteraction(interaction_head(latent).to(device)).to(device); base.load_state_dict(torch.load(base_head_path,map_location="cpu",weights_only=True)); head=RankedResidualInteraction(base).to(device); held=np.arange(int(data["pairs"].max())+1)%5==0; known=data["context_known"][data["context"]]; train=np.flatnonzero(known&(~held[data["pairs"]]).all(1)); valid=np.flatnonzero(known&held[data["pairs"]].all(1)); contexts=np.unique(data["context"][train]); pools={c:train[data["context"][train]==c] for c in contexts}; mean,_=pair_targets(data,train); target=data["target"][:,0].astype("float32").copy(); target[train]=(1-shrinkage)*target[train]+shrinkage*mean; opt=torch.optim.AdamW(head.rank.parameters(),1e-4,weight_decay=1e-3); candidates=[]; saved=[]
    @torch.no_grad()
    def validate(epoch):
        pred=[]
        for at in batches(len(valid),4096,False):
            ix=valid[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); cs=torch.as_tensor(data["context_state"][data["context"][ix]].astype("float32"),device=device); z,r=residual_interaction_inputs(endpoint,genes,p,cs); pred.append(head(z,r)[:,0].cpu())
        metric={"epoch":epoch,**interaction_array_metrics(torch.cat(pred).numpy(),data,valid)}; candidates.append(metric); saved.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()}); print(json.dumps(metric),flush=True)
    def comparisons(rng):
        left=[]; right=[]
        while sum(map(len,left))<epoch_pairs:
            for c in contexts:
                n=max(1,epoch_pairs//len(contexts)); a=rng.choice(pools[c],n); b=rng.choice(pools[c],n); keep=np.abs(data["target"][a,0]-data["target"][b,0])>=margin; left.append(a[keep]); right.append(b[keep])
        return np.concatenate(left)[:epoch_pairs],np.concatenate(right)[:epoch_pairs]
    validate(0)
    for epoch in range(epochs):
        left,right=comparisons(np.random.default_rng(731+epoch)); head.train(); endpoint.eval()
        for at in batches(epoch_pairs,2048,False):
            rows=np.r_[left[at],right[at]]; p=torch.as_tensor(data["pairs"][rows].astype("int64"),device=device); cs=torch.as_tensor(data["context_state"][data["context"][rows]].astype("float32"),device=device)
            with torch.no_grad():z,r=residual_interaction_inputs(endpoint,genes,p,cs)
            pred=head(z,r)[:,0]; one,two=pred.chunk(2); sign=torch.as_tensor(np.sign(data["target"][left[at],0]-data["target"][right[at],0]),device=device); absolute=torch.as_tensor(target[rows],device=device); loss=nn.functional.softplus(-sign*(one-two)/margin).mean()+nn.functional.huber_loss(pred,absolute); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(head.rank.parameters(),1.); opt.step()
        head.eval(); validate(epoch+1)
    baseline=candidates[0]; eligible=[i for i,x in enumerate(candidates) if x["row_huber"]<=1.01*baseline["row_huber"] and x["pair_huber"]<=1.01*baseline["pair_huber"] and x["row_correlation"]>=baseline["row_correlation"]-.005 and x["pair_correlation"]>=baseline["pair_correlation"]-.005 and x["row_spearman"]>=baseline["row_spearman"]-.005 and x["pair_spearman"]>=baseline["pair_spearman"]-.005]; selected=max(eligible,key=lambda i:(candidates[i]["row_spearman"]+candidates[i]["pair_spearman"],-i)); head.load_state_dict(saved[selected]); metric=candidates[selected]; advanced=(metric["row_spearman"]-baseline["row_spearman"]>=.01 or metric["pair_spearman"]-baseline["pair_spearman"]>=.01) and selected>0; out=Path(out_dir); torch.save(head.state_dict(),out/"interaction_rank_head.pt"); result={"parameters":sum(p.numel() for p in head.rank.parameters()),"baseline":baseline,"selected":metric,"advanced":advanced,"eligible_epochs":eligible,"candidates":candidates}; (out/"interaction_rank_metrics.json").write_text(json.dumps(result,indent=2)); return result


def fit_compositional_ridge(data_dir,endpoint_path,head_path,out_dir,alphas=(.1,1.,10.,100.,1000.),d=384,latent=128,layers=6):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(Path(data_dir)/"features_spectral_safe.npz")["state"].astype("float32"); data=np.load(Path(data_dir)/"slkb_outcomes_intervention_depmap_world.npz",allow_pickle=True); endpoint=load_residual_endpoint(endpoint_path,state.shape[1],device,d,latent,layers); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); head=ResidualInteraction(interaction_head(latent).to(device)).to(device); head.load_state_dict(torch.load(head_path,map_location="cpu",weights_only=True)); head.eval().requires_grad_(False); known=data["context_known"][data["context"]]; held=np.arange(int(data["pairs"].max())+1)%5==0; train=np.flatnonzero(known&(~held[data["pairs"]]).all(1)); valid=np.flatnonzero(known&held[data["pairs"]].all(1)); rows=np.r_[train,valid]; feature=np.zeros((len(data["pairs"]),192),"float32"); base=np.zeros(len(data["pairs"]),"float32")
    for at in batches(len(rows),2048,False):
        ix=rows[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); cs=torch.as_tensor(data["context_state"][data["context"][ix]].astype("float32"),device=device); z,r,x=full_endpoint_composition(endpoint,genes,p,cs); feature[ix]=x.cpu(); base[ix]=head(z,r)[:,0].cpu()
    def weights(ix):
        c=data["context"][ix]; return len(ix)/(len(np.unique(c))*np.bincount(c)[c])
    def fit_predict(alpha,tr,va):
        scale=StandardScaler().fit(feature[tr]); model=Ridge(alpha=alpha).fit(scale.transform(feature[tr]),data["target"][tr,0]-base[tr],sample_weight=weights(tr)); return base[va]+model.predict(scale.transform(feature[va])),scale,model
    inner=[]
    for alpha in alphas:
        folds=[]
        for fold in range(4):
            role=(np.arange(int(data["pairs"].max())+1)//5)%4; tr=train[(role[data["pairs"][train]]!=fold).all(1)]; va=train[(role[data["pairs"][train]]==fold).all(1)]
            pred,_,_=fit_predict(alpha,tr,va); metric=interaction_array_metrics(pred,data,va); folds.append({"fold":fold,"train_rows":len(tr),"validation_rows":len(va),**metric})
        inner.append({"alpha":alpha,"selection_loss":float(np.mean([x["row_huber"]+x["pair_huber"] for x in folds])),"folds":folds})
    selected=min(inner,key=lambda x:(x["selection_loss"],-x["alpha"])); pred,scale,model=fit_predict(selected["alpha"],train,valid); baseline=interaction_array_metrics(base[valid],data,valid); candidate=interaction_array_metrics(pred,data,valid); preserved=candidate["row_huber"]<=1.01*baseline["row_huber"] and candidate["pair_huber"]<=1.01*baseline["pair_huber"] and candidate["row_correlation"]>=baseline["row_correlation"]-.005 and candidate["pair_correlation"]>=baseline["pair_correlation"]-.005; improved=candidate["row_huber"]<=.98*baseline["row_huber"] or candidate["pair_huber"]<=.98*baseline["pair_huber"] or candidate["row_correlation"]>=baseline["row_correlation"]+.01 or candidate["pair_correlation"]>=baseline["pair_correlation"]+.01; out=Path(out_dir); np.savez(out/"interaction_compositional_ridge.npz",mean=scale.mean_,scale=scale.scale_,coef=model.coef_,intercept=model.intercept_,alpha=selected["alpha"]); result={"features":192,"train_rows":len(train),"validation_rows":len(valid),"selected_alpha":selected["alpha"],"baseline":baseline,"candidate":candidate,"preserved":bool(preserved),"improved":bool(improved),"advanced":bool(preserved and improved),"inner":inner}; (out/"interaction_compositional_ridge_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True); return result


def fit_source_endpoint(data_dir,endpoint_path,out_dir,epochs=12,d=384,latent=128,layers=6):
    torch.manual_seed(731); np.random.seed(731); device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(Path(data_dir)/"features_spectral_safe.npz")["state"].astype("float32"); data=np.load(Path(data_dir)/"perturbseq_source_landmark.npz"); endpoint=load_residual_endpoint(endpoint_path,state.shape[1],device,d,latent,layers); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); head=SourceEndpoint(len(data["sources"]),latent,data["target"].shape[1]).to(device); train=np.flatnonzero(data["role"]==0); valid=np.flatnonzero(data["role"]==1); source=data["source"][train]; weight=1/np.bincount(source)[source]; weight*=np.where(data["cardinality"][train]==2,5.,1.); weight/=weight.sum(); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3); history=[]; saved=[]
    @torch.no_grad()
    def metrics(epoch):
        rows=[]
        for contextual in (False,True):
            pred=[]
            for at in batches(len(valid),2048,False):
                ix=valid[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); second=p[:,1]>=0; cs=torch.as_tensor(data["context_state"][data["source"][ix]],device=device) if contextual else None; z=torch.empty((len(ix),latent),device=device); z[~second]=endpoint.world.transition(genes[p[~second,0]],context_state=cs[~second] if cs is not None else None)[0]; z[second]=endpoint.world.transition(genes[p[second,0]],genes[p[second,1]],context_state=cs[second] if cs is not None else None)[0]; pred.append(head(z,torch.as_tensor(data["source"][ix].astype("int64"),device=device)).cpu())
            pred=torch.cat(pred).numpy(); target=data["target"][valid]; by=[]
            for s in range(len(data["sources"])):
                keep=data["source"][valid]==s
                if keep.any(): by.append({"source":str(data["sources"][s]),"rows":int(keep.sum()),"huber":float(nn.functional.huber_loss(torch.from_numpy(pred[keep]),torch.from_numpy(target[keep]))),"cosine":float(np.mean(np.sum(pred[keep]*target[keep],1)/(np.linalg.norm(pred[keep],axis=1)*np.linalg.norm(target[keep],axis=1)+1e-8)))})
            double=data["cardinality"][valid]==2; rows.append({"context":"exact" if contextual else "unknown","source_macro_huber":float(np.mean([x["huber"] for x in by])),"source_macro_cosine":float(np.mean([x["cosine"] for x in by])),"double_huber":float(nn.functional.huber_loss(torch.from_numpy(pred[double]),torch.from_numpy(target[double]))),"sources":by})
        row={"epoch":epoch,"unknown":rows[0],"exact":rows[1]}; row["selection_loss"]=sum(row[c][k] for c in ("unknown","exact") for k in ("source_macro_huber","double_huber")); history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()}); print(json.dumps(row),flush=True)
    zero={}; target=data["target"][valid]
    for contextual in ("unknown","exact"):
        by=[float(nn.functional.huber_loss(torch.zeros_like(torch.from_numpy(target[data["source"][valid]==s])),torch.from_numpy(target[data["source"][valid]==s]))) for s in range(len(data["sources"])) if (data["source"][valid]==s).any()]; double=data["cardinality"][valid]==2; zero[contextual]={"source_macro_huber":float(np.mean(by)),"source_macro_cosine":0.,"double_huber":float(nn.functional.huber_loss(torch.zeros_like(torch.from_numpy(target[double])),torch.from_numpy(target[double])))}
    metrics(0)
    for epoch in range(epochs):
        chosen=np.random.default_rng(731+epoch).choice(train,len(train),replace=True,p=weight); head.train()
        for at in batches(len(chosen),512,False):
            ix=chosen[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); second=p[:,1]>=0; cs=torch.as_tensor(data["context_state"][data["source"][ix]],device=device); cs=torch.where((torch.rand(len(ix),device=device)<.5)[:,None],torch.zeros_like(cs),cs); z=torch.empty((len(ix),latent),device=device)
            with torch.no_grad(): z[~second]=endpoint.world.transition(genes[p[~second,0]],context_state=cs[~second])[0]; z[second]=endpoint.world.transition(genes[p[second,0]],genes[p[second,1]],context_state=cs[second])[0]
            pred=head(z,torch.as_tensor(data["source"][ix].astype("int64"),device=device)); loss=nn.functional.huber_loss(pred,torch.as_tensor(data["target"][ix],device=device)); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(head.parameters(),1.); opt.step()
        head.eval(); metrics(epoch+1)
    selected=min(range(len(history)),key=lambda i:history[i]["selection_loss"]); head.load_state_dict(saved[selected]); chosen=history[selected]; advanced=all(chosen[c]["source_macro_huber"]<zero[c]["source_macro_huber"] and chosen[c]["double_huber"]<zero[c]["double_huber"] and chosen[c]["source_macro_cosine"]>0 for c in ("unknown","exact")); out=Path(out_dir); torch.save(head.state_dict(),out/"source_landmark_endpoint.pt"); result={"parameters":sum(p.numel() for p in head.parameters()),"zero":zero,"selected":chosen,"advanced":bool(advanced),"history":history}; (out/"source_landmark_endpoint_metrics.json").write_text(json.dumps(result,indent=2)); return result


def fit_dependency_landscape_endpoint(data_dir,endpoint_path,out_dir,epochs=20,d=384,latent=128,layers=6,target_dimensions=64,stem="dependency_landscape"):
    torch.manual_seed(731); np.random.seed(731); device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(Path(data_dir)/"features_spectral_safe.npz")["state"].astype("float32"); data=np.load(Path(data_dir)/"dependency_landscape.npz"); endpoint=load_residual_endpoint(endpoint_path,state.shape[1],device,d,latent,layers); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); z=[]
    with torch.no_grad():
        for at in batches(len(genes),2048,False): z.append(endpoint.world.transition(genes[at])[0].cpu())
    z=torch.cat(z).to(device); target=torch.as_tensor(data["target"][:,:target_dimensions],device=device); train=np.flatnonzero(data["train"]); valid=np.flatnonzero(data["valid"]); excluded=np.flatnonzero(data["excluded"]); head=dependency_landscape_head(latent,target.shape[1]).to(device); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-3); history=[]; saved=[]
    @torch.no_grad()
    def metric(rows):
        ix=torch.as_tensor(rows,device=device); pred=head(z[ix]); truth=target[ix]; return {"genes":int(len(rows)),"huber":float(nn.functional.huber_loss(pred,truth)),"cosine":float(nn.functional.cosine_similarity(pred,truth).mean()),"correlation":float(np.corrcoef(pred.cpu().numpy().ravel(),truth.cpu().numpy().ravel())[0,1]) if pred.square().sum() else 0.}
    zero=metric(valid); history.append({"epoch":0,**zero}); saved.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()}); print(json.dumps({"phase":"dependency_landscape","epoch":0,**zero}),flush=True)
    for epoch in range(epochs):
        head.train(); chosen=np.random.default_rng(731+epoch).permutation(train); total=0.
        for at in batches(len(chosen),512,False):
            ix=torch.as_tensor(chosen[at],device=device); pred=head(z[ix]); loss=nn.functional.huber_loss(pred,target[ix]); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(head.parameters(),1.); opt.step(); total+=loss.item()*len(ix)
        head.eval(); row={"epoch":epoch+1,"train_huber":total/len(chosen),**metric(valid)}; history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()}); print(json.dumps({"phase":"dependency_landscape",**row}),flush=True)
    selected=min(range(len(history)),key=lambda i:history[i]["huber"]); head.load_state_dict(saved[selected]); chosen=history[selected]; isolated=metric(excluded); ex=torch.as_tensor(excluded,device=device); isolated_zero=float(nn.functional.huber_loss(torch.zeros_like(target[ex]),target[ex])); cosine_floor=.2 if target_dimensions==16 else .1; advanced=chosen["huber"]<=.95*zero["huber"] and chosen["cosine"]>=cosine_floor and isolated["huber"]<=.95*isolated_zero and isolated["cosine"]>0; out=Path(out_dir); torch.save(head.state_dict(),out/f"{stem}_endpoint.pt"); result={"target_dimensions":target_dimensions,"parameters":sum(p.numel() for p in head.parameters()),"zero":zero,"selected":chosen,"intervention_isolated":isolated,"intervention_isolated_zero_huber":isolated_zero,"advanced":bool(advanced),"history":history}; (out/f"{stem}_endpoint_metrics.json").write_text(json.dumps(result,indent=2)); return result


def fit_dependency_action_adapter(data_dir,endpoint_path,decoder_path,out_dir,epochs=30,d=384,latent=128,layers=6):
    torch.manual_seed(731); np.random.seed(731); device="cuda" if torch.cuda.is_available() else "cpu"; pack=np.load(Path(data_dir)/"features_spectral_safe.npz"); state=pack["state"].astype("float32"); data=np.load(Path(data_dir)/"dependency_landscape.npz"); perturb=np.load(Path(data_dir)/"perturbseq_world_v3_nested96.npz"); endpoint=load_residual_endpoint(endpoint_path,state.shape[1],device,d,latent,layers); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); static=torch.as_tensor(state,device=device); target=torch.as_tensor(data["target"],device=device); train=np.flatnonzero(data["train"]); valid=np.flatnonzero(data["valid"]); excluded=np.flatnonzero(data["excluded"]); candidate=DependencyActionAdapter(state.shape[1],latent,target.shape[1]).to(device); candidate.decoder.load_state_dict(torch.load(decoder_path,map_location="cpu",weights_only=True)); opt=torch.optim.AdamW(({"params":candidate.adapter.parameters(),"lr":3e-4},{"params":candidate.decoder.parameters(),"lr":1e-4}),weight_decay=1e-3); history=[]; saved=[]
    @torch.no_grad()
    def metric(rows):
        pred=[]
        for at in batches(len(rows),2048,False):
            ix=torch.as_tensor(rows[at],device=device); pred.append(candidate.decoder(endpoint.world.transition(candidate.action(static[ix],genes[ix]))[0]).cpu())
        pred=torch.cat(pred); truth=target[torch.as_tensor(rows,device=device)].cpu(); return {"genes":int(len(rows)),"huber":float(nn.functional.huber_loss(pred,truth)),"cosine":float(nn.functional.cosine_similarity(pred,truth).mean()),"correlation":float(np.corrcoef(pred.numpy().ravel(),truth.numpy().ravel())[0,1])}
    candidate.eval(); initial=metric(valid); history.append({"epoch":0,**initial}); saved.append({k:v.detach().cpu().clone() for k,v in candidate.state_dict().items()}); print(json.dumps({"phase":"dependency_action_adapter","epoch":0,**initial}),flush=True)
    for epoch in range(epochs):
        candidate.train(); chosen=np.random.default_rng(731+epoch).permutation(train); total=0.
        for at in batches(len(chosen),512,False):
            ix=torch.as_tensor(chosen[at],device=device); action=candidate.action(static[ix],genes[ix]); pred=candidate.decoder(endpoint.world.transition(action)[0]); loss=nn.functional.huber_loss(pred,target[ix]); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(candidate.parameters(),1.); opt.step(); total+=loss.item()*len(ix)
        candidate.eval(); row={"epoch":epoch+1,"train_huber":total/len(chosen),**metric(valid)}; history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in candidate.state_dict().items()}); print(json.dumps({"phase":"dependency_action_adapter",**row}),flush=True)
    selected=min(range(len(history)),key=lambda i:history[i]["huber"]); candidate.load_state_dict(saved[selected]); candidate.eval(); chosen=history[selected]; isolated=metric(excluded); vi=torch.as_tensor(valid,device=device); ex=torch.as_tensor(excluded,device=device); zero={"selection_huber":float(nn.functional.huber_loss(torch.zeros_like(target[vi]),target[vi])),"isolated_huber":float(nn.functional.huber_loss(torch.zeros_like(target[ex]),target[ex]))}
    relation_pairs=torch.as_tensor(pack["pairs"].astype("int64"),device=device); relation_target=torch.as_tensor(pack["relations"].astype("float32"),device=device); relation_valid=torch.nonzero((relation_pairs[:,0]*1000003+relation_pairs[:,1])%20==0).squeeze(1)
    @torch.no_grad()
    def relation(adapted):
        total=0.
        for at in batches(len(relation_valid),4096,False):
            ix=relation_valid[at]; p=relation_pairs[ix]; a,b=genes[p[:,0]],genes[p[:,1]]
            if adapted:a,b=candidate.action(static[p[:,0]],a),candidate.action(static[p[:,1]],b)
            total+=nn.functional.smooth_l1_loss(endpoint.world.relation_score(a,b,endpoint.world.transition(a,b)[0]),relation_target[ix],reduction="sum").item()
        return total/len(relation_valid)/relation_target.shape[1]
    adapted_genes=candidate.action(static,genes); baseline_state={"relation":relation(False),"legacy":sum(residual_endpoint_score(endpoint,genes,perturb,device,0,x) for x in (False,True)),"residual":sum(residual_endpoint_score(endpoint,genes,perturb,device,1,x) for x in (False,True))}; adapted_state={"relation":relation(True),"legacy":sum(residual_endpoint_score(endpoint,adapted_genes,perturb,device,0,x) for x in (False,True)),"residual":sum(residual_endpoint_score(endpoint,adapted_genes,perturb,device,1,x) for x in (False,True))}; dependency_ok=chosen["huber"]<=.95*zero["selection_huber"] and chosen["cosine"]>=.1 and isolated["huber"]<=.95*zero["isolated_huber"] and isolated["cosine"]>0; preservation=all(adapted_state[k]<=1.01*baseline_state[k] for k in baseline_state); advanced=dependency_ok and preservation; out=Path(out_dir); torch.save(candidate.state_dict(),out/"dependency_action_adapter.pt"); result={"parameters":sum(p.numel() for p in candidate.parameters()),"selected":chosen,"intervention_isolated":isolated,"zero":zero,"baseline_state":baseline_state,"adapted_state":adapted_state,"dependency_advanced":bool(dependency_ok),"preserved":bool(preservation),"advanced":bool(advanced),"history":history}; (out/"dependency_action_adapter_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"}),flush=True); return result


@torch.no_grad()
def dependency_interaction_inputs(endpoint,decoder,genes,pairs,context_state):
    a,b=genes[pairs[:,0]],genes[pairs[:,1]]; z=endpoint.world.transition(a,b,context_state=context_state)[0]; za=endpoint.world.transition(a,context_state=context_state)[0]; zb=endpoint.world.transition(b,context_state=context_state)[0]; joint,sa,sb=decoder(z),decoder(za),decoder(zb); _,residual=residual_interaction_inputs(endpoint,genes,pairs,context_state); return z,residual,torch.cat((joint,joint-sa-sb),1)


def fit_dependency_core_interaction(data_dir,endpoint_path,base_path,decoder_path,out_dir,epochs=20,epoch_pairs=100000,shrinkage=.15,d=384,latent=128,layers=6):
    torch.manual_seed(731); np.random.seed(731); device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(Path(data_dir)/"features_spectral_safe.npz")["state"].astype("float32"); data=np.load(Path(data_dir)/"slkb_outcomes_intervention_depmap_world.npz",allow_pickle=True); endpoint=load_residual_endpoint(endpoint_path,state.shape[1],device,d,latent,layers); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); decoder=dependency_landscape_head(latent,16).to(device); decoder.load_state_dict(torch.load(decoder_path,map_location="cpu",weights_only=True)); decoder.eval().requires_grad_(False); base=ResidualInteraction(interaction_head(latent).to(device)).to(device); base.load_state_dict(torch.load(base_path,map_location="cpu",weights_only=True)); head=DependencyInteraction(base).to(device); held=np.arange(int(data["pairs"].max())+1)%5==0; known=data["context_known"][data["context"]]; train=np.flatnonzero(known&(~held[data["pairs"]]).all(1)); valid=np.flatnonzero(known&held[data["pairs"]].all(1)); mean,_=pair_targets(data,train); target=data["target"].astype("float32").copy(); target[train,0]=(1-shrinkage)*target[train,0]+shrinkage*mean; source=data["context"][train]; weight=1/np.bincount(source)[source]; weight/=weight.sum(); opt=torch.optim.AdamW(head.correction.parameters(),3e-4,weight_decay=1e-3); candidates=[]; saved=[]
    @torch.no_grad()
    def validate(epoch):
        pred=[]
        for at in batches(len(valid),4096,False):
            ix=valid[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); cs=torch.as_tensor(data["context_state"][data["context"][ix]].astype("float32"),device=device); z,r,q=dependency_interaction_inputs(endpoint,decoder,genes,p,cs); pred.append(head(z,r,q)[:,0].cpu())
        metric={"epoch":epoch,**interaction_array_metrics(torch.cat(pred).numpy(),data,valid)}; metric["selection_loss"]=metric["row_huber"]+metric["pair_huber"]; candidates.append(metric); saved.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()}); print(json.dumps({"phase":"dependency_core_interaction",**metric}),flush=True)
    head.eval(); validate(0)
    for epoch in range(epochs):
        chosen=np.random.default_rng(731+epoch).choice(train,epoch_pairs,replace=True,p=weight); head.train()
        for at in batches(len(chosen),2048,False):
            ix=chosen[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); cs=torch.as_tensor(data["context_state"][data["context"][ix]].astype("float32"),device=device); z,r,q=dependency_interaction_inputs(endpoint,decoder,genes,p,cs); pred=head(z,r,q); loss=nn.functional.huber_loss(pred,torch.as_tensor(target[ix],device=device)); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(head.correction.parameters(),1.); opt.step()
        head.eval(); validate(epoch+1)
    selected=min(range(len(candidates)),key=lambda i:candidates[i]["selection_loss"]); head.load_state_dict(saved[selected]); baseline=candidates[0]; metric=candidates[selected]; preserved=all(metric[k]<=1.01*baseline[k] for k in ("row_huber","pair_huber")) and all(metric[k]>=baseline[k]-.005 for k in ("row_correlation","pair_correlation","row_spearman","pair_spearman")); improved=any(metric[k]<=.98*baseline[k] for k in ("row_huber","pair_huber")) or any(metric[k]>=baseline[k]+.01 for k in ("row_correlation","pair_correlation","row_spearman","pair_spearman")); advanced=selected>0 and preserved and improved; out=Path(out_dir); torch.save(head.state_dict(),out/"dependency_core_interaction_head.pt"); result={"parameters":sum(p.numel() for p in head.correction.parameters()),"baseline":baseline,"selected":metric,"preserved":bool(preserved),"improved":bool(improved),"advanced":bool(advanced),"candidates":candidates}; (out/"dependency_core_interaction_metrics.json").write_text(json.dumps(result,indent=2)); return result


def fit_pair_transition_adapter(data_dir,endpoint_path,out_dir,epochs=30,d=384,latent=128,layers=6):
    torch.manual_seed(731); np.random.seed(731); device="cuda" if torch.cuda.is_available() else "cpu"; pack=np.load(Path(data_dir)/"features_spectral_safe.npz"); state=pack["state"].astype("float32"); data=np.load(Path(data_dir)/"perturbseq_world_v3_nested96.npz"); endpoint=load_residual_endpoint(endpoint_path,state.shape[1],device,d,latent,layers); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); adapter=pair_transition_adapter(latent).to(device); train=np.flatnonzero((data["role"]==0)&(data["cardinality"]==2)); valid=np.flatnonzero((data["role"]==1)&(data["cardinality"]==2)); source=data["source"][train]; weight=1/np.bincount(source)[source]; weight/=weight.sum(); opt=torch.optim.AdamW(adapter.parameters(),1e-4,weight_decay=1e-2); history=[]; saved=[]
    decode=lambda z:torch.cat((endpoint.legacy_decoder(z),endpoint.residual_decoder(z)),1)
    @torch.no_grad()
    def validate(epoch):
        values={}
        for contextual in (False,True):
            pred=[]
            for at in batches(len(valid),512,False):
                ix=valid[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); cs=torch.as_tensor(data["context_state"][data["source"][ix]],device=device) if contextual else None; z=endpoint.world.transition(genes[p[:,0]],genes[p[:,1]],context_state=cs)[0]; pred.append(decode(z+adapter(z)).cpu())
            pred=torch.cat(pred); truth=torch.as_tensor(data["target"][valid]); stem="exact" if contextual else "unknown"; values[f"{stem}_legacy_huber"]=float(nn.functional.huber_loss(pred[:,:32],truth[:,:32])); values[f"{stem}_residual_huber"]=float(nn.functional.huber_loss(pred[:,32:96],truth[:,32:96]))
        row={"epoch":epoch,**values,"selection_loss":float(np.mean(list(values.values())))}; history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}); print(json.dumps({"phase":"pair_transition_adapter",**row}),flush=True)
    adapter.eval(); validate(0)
    for epoch in range(epochs):
        chosen=np.random.default_rng(731+epoch).choice(train,len(train),replace=True,p=weight); adapter.train(); total=0.
        for at in batches(len(chosen),128,False):
            ix=chosen[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); cs=torch.as_tensor(data["context_state"][data["source"][ix]],device=device); cs=torch.where((torch.rand(len(ix),device=device)<.5)[:,None],torch.zeros_like(cs),cs); target=torch.as_tensor(data["target"][ix],device=device)
            with torch.no_grad():z=endpoint.world.transition(genes[p[:,0]],genes[p[:,1]],context_state=cs)[0]
            pred=decode(z+adapter(z)); loss=.5*nn.functional.huber_loss(pred[:,:32],target[:,:32])+.5*nn.functional.huber_loss(pred[:,32:96],target[:,32:96]); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(adapter.parameters(),1.); opt.step(); total+=loss.item()*len(ix)
        adapter.eval(); validate(epoch+1); history[-1]["train_huber"]=total/len(chosen)
    selected=min(range(len(history)),key=lambda i:history[i]["selection_loss"]); adapter.load_state_dict(saved[selected]); adapter.eval(); relation_pairs=torch.as_tensor(pack["pairs"].astype("int64"),device=device); relation_target=torch.as_tensor(pack["relations"].astype("float32"),device=device); relation_valid=torch.nonzero((relation_pairs[:,0]*1000003+relation_pairs[:,1])%20==0).squeeze(1)
    @torch.no_grad()
    def relation(adapted):
        total=0.
        for at in batches(len(relation_valid),4096,False):
            ix=relation_valid[at]; p=relation_pairs[ix]; a,b=genes[p[:,0]],genes[p[:,1]]; z=endpoint.world.transition(a,b)[0]; z=z+adapter(z) if adapted else z; total+=nn.functional.smooth_l1_loss(endpoint.world.relation_score(a,b,z),relation_target[ix],reduction="sum").item()
        return total/len(relation_valid)/relation_target.shape[1]
    baseline=history[0]; metric=history[selected]; keys=("unknown_legacy_huber","unknown_residual_huber","exact_legacy_huber","exact_residual_huber"); preserved=all(metric[k]<=1.01*baseline[k] for k in keys); improved=metric["selection_loss"]<=.98*baseline["selection_loss"]; relation_loss={"baseline":relation(False),"adapted":relation(True)}; relation_preserved=relation_loss["adapted"]<=1.01*relation_loss["baseline"]; advanced=selected>0 and preserved and improved and relation_preserved; out=Path(out_dir); torch.save(adapter.state_dict(),out/"pair_transition_adapter.pt"); result={"parameters":sum(p.numel() for p in adapter.parameters()),"training_doubles":len(train),"validation_doubles":len(valid),"baseline":baseline,"selected":metric,"relation":relation_loss,"preserved":bool(preserved),"improved":bool(improved),"relation_preserved":bool(relation_preserved),"advanced":bool(advanced),"history":history}; (out/"pair_transition_adapter_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"}),flush=True); return result


def fit_dixit_pair_adapter(data_dir,endpoint_path,out_dir,decoder_epochs=20,adapter_epochs=30,d=384,latent=128,layers=6):
    torch.manual_seed(731); np.random.seed(731); device="cuda" if torch.cuda.is_available() else "cpu"; pack=np.load(Path(data_dir)/"features_spectral_safe.npz"); state=pack["state"].astype("float32"); data=np.load(Path(data_dir)/"perturbseq_world_v4_dixit_context.npz"); endpoint=load_residual_endpoint(endpoint_path,state.shape[1],device,d,latent,layers); endpoint.eval().requires_grad_(False); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); decoder=nn.Sequential(nn.LayerNorm(latent),nn.Linear(latent,96)).to(device); train=np.flatnonzero(data["role"]==0); held_single=np.flatnonzero((data["role"]==1)&(data["cardinality"]==1)); source=data["source"][train]; weight=1/np.bincount(source)[source]; weight*=np.where(data["cardinality"][train]==2,2.,1.); weight/=weight.sum(); opt=torch.optim.AdamW(decoder.parameters(),3e-4,weight_decay=1e-3); candidates=[]; saved=[]
    @torch.no_grad()
    def single_validation(epoch):
        losses=[]
        for contextual in (False,True):
            pred=[]
            for at in batches(len(held_single),512,False):
                ix=held_single[at]; p=torch.as_tensor(data["pairs"][ix,0].astype("int64"),device=device); cs=torch.as_tensor(data["context_state"][data["source"][ix]],device=device) if contextual else None; pred.append(decoder(endpoint.world.transition(genes[p],context_state=cs)[0]).cpu())
            losses.append(float(nn.functional.huber_loss(torch.cat(pred),torch.as_tensor(data["target"][held_single]))))
        row={"epoch":epoch,"unknown_huber":losses[0],"exact_huber":losses[1],"selection_loss":float(np.mean(losses))}; candidates.append(row); saved.append({k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()}); print(json.dumps({"phase":"dixit_decoder",**row}),flush=True)
    for epoch in range(decoder_epochs):
        chosen=np.random.default_rng(731+epoch).choice(train,len(train),replace=True,p=weight); decoder.train()
        for at in batches(len(chosen),256,False):
            ix=chosen[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); a=genes[p[:,0]]; second=p[:,1]>=0; cs=torch.as_tensor(data["context_state"][data["source"][ix]],device=device); cs=torch.where((torch.rand(len(ix),device=device)<.5)[:,None],torch.zeros_like(cs),cs); z=torch.empty((len(ix),latent),device=device)
            if (~second).any():z[~second]=endpoint.world.transition(a[~second],context_state=cs[~second])[0]
            if second.any():z[second]=endpoint.world.transition(a[second],genes[p[second,1]],context_state=cs[second])[0]
            loss=perturbseq_state_error(nn.functional.huber_loss(decoder(z),torch.as_tensor(data["target"][ix],device=device),reduction="none"),data,96); opt.zero_grad(); loss.mean().backward(); nn.utils.clip_grad_norm_(decoder.parameters(),1.); opt.step()
        decoder.eval(); single_validation(epoch+1)
    chosen=min(range(len(candidates)),key=lambda i:candidates[i]["selection_loss"]); decoder.load_state_dict(saved[chosen]); decoder.eval().requires_grad_(False); adapter=pair_transition_adapter(latent).to(device); train_pair=np.flatnonzero((data["role"]==0)&(data["cardinality"]==2)); valid=np.flatnonzero((data["role"]==1)&(data["cardinality"]==2)); source=data["source"][train_pair]; weight=1/np.bincount(source)[source]; weight/=weight.sum(); opt=torch.optim.AdamW(adapter.parameters(),1e-4,weight_decay=1e-2); history=[]; saved=[]
    @torch.no_grad()
    def pair_validation(epoch):
        values={}
        for contextual in (False,True):
            pred=[]
            for at in batches(len(valid),512,False):
                ix=valid[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); cs=torch.as_tensor(data["context_state"][data["source"][ix]],device=device) if contextual else None; z=endpoint.world.transition(genes[p[:,0]],genes[p[:,1]],context_state=cs)[0]; pred.append(decoder(z+adapter(z)).cpu())
            pred=torch.cat(pred); truth=torch.as_tensor(data["target"][valid]); stem="exact" if contextual else "unknown"
            for group,mask in (("all",np.ones(len(valid),bool)),("retained",data["source"][valid]<7),("dixit",data["source"][valid]>=7)):
                for name,lo,hi in (("legacy",0,32),("residual",32,96)):values[f"{stem}_{group}_{name}"]=float(nn.functional.huber_loss(pred[mask,lo:hi],truth[mask,lo:hi]))
        selection=np.mean([v for k,v in values.items() if "_all_" in k]); row={"epoch":epoch,**values,"selection_loss":float(selection)}; history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}); print(json.dumps({"phase":"dixit_pair_adapter",**row}),flush=True)
    adapter.eval(); pair_validation(0)
    for epoch in range(adapter_epochs):
        rows=np.random.default_rng(751+epoch).choice(train_pair,len(train_pair),replace=True,p=weight); adapter.train()
        for at in batches(len(rows),128,False):
            ix=rows[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); cs=torch.as_tensor(data["context_state"][data["source"][ix]],device=device); cs=torch.where((torch.rand(len(ix),device=device)<.5)[:,None],torch.zeros_like(cs),cs)
            with torch.no_grad():z=endpoint.world.transition(genes[p[:,0]],genes[p[:,1]],context_state=cs)[0]
            loss=perturbseq_state_error(nn.functional.huber_loss(decoder(z+adapter(z)),torch.as_tensor(data["target"][ix],device=device),reduction="none"),data,96); opt.zero_grad(); loss.mean().backward(); nn.utils.clip_grad_norm_(adapter.parameters(),1.); opt.step()
        adapter.eval(); pair_validation(epoch+1)
    selected=min(range(len(history)),key=lambda i:history[i]["selection_loss"]); adapter.load_state_dict(saved[selected]); baseline=history[0]; metric=history[selected]; all_keys=[k for k in metric if "_all_" in k]; retained=[k for k in metric if "_retained_" in k]; preserved=all(metric[k]<=1.01*baseline[k] for k in all_keys+retained); improved=metric["selection_loss"]<=.98*baseline["selection_loss"]; relation_pairs=torch.as_tensor(pack["pairs"].astype("int64"),device=device); relation_target=torch.as_tensor(pack["relations"].astype("float32"),device=device); rv=torch.nonzero((relation_pairs[:,0]*1000003+relation_pairs[:,1])%20==0).squeeze(1)
    @torch.no_grad()
    def relation(adapted):
        total=0.
        for at in batches(len(rv),4096,False):
            ix=rv[at]; p=relation_pairs[ix]; a,b=genes[p[:,0]],genes[p[:,1]]; z=endpoint.world.transition(a,b)[0]; z=z+adapter(z) if adapted else z; total+=nn.functional.smooth_l1_loss(endpoint.world.relation_score(a,b,z),relation_target[ix],reduction="sum").item()
        return total/len(rv)/relation_target.shape[1]
    rel={"baseline":relation(False),"adapted":relation(True)}; advanced=selected>0 and improved and preserved and rel["adapted"]<=1.01*rel["baseline"]; out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); torch.save(decoder.state_dict(),out/"dixit_decoder.pt"); torch.save(adapter.state_dict(),out/"dixit_pair_adapter.pt"); result={"decoder_parameters":sum(p.numel() for p in decoder.parameters()),"adapter_parameters":sum(p.numel() for p in adapter.parameters()),"decoder_selected":candidates[chosen],"training_doubles":len(train_pair),"validation_doubles":len(valid),"baseline":baseline,"selected":metric,"relation":rel,"preserved":bool(preserved),"improved":bool(improved),"advanced":bool(advanced),"decoder_history":candidates,"history":history}; (out/"metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k not in ("history","decoder_history")},indent=2),flush=True); return result


def fit_dixit_symmetric_latent(data_dir,endpoint_path,decoder_path,out_dir,epochs=30,d=384,latent=128,layers=6,residual=False):
    torch.manual_seed(731); np.random.seed(731); device="cuda" if torch.cuda.is_available() else "cpu"; pack=np.load(Path(data_dir)/"features_spectral_safe.npz"); state=pack["state"].astype("float32"); data=np.load(Path(data_dir)/"perturbseq_world_v4_dixit_context.npz"); endpoint=load_residual_endpoint(endpoint_path,state.shape[1],device,d,latent,layers); endpoint.eval().requires_grad_(False); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); decoder=nn.Sequential(nn.LayerNorm(latent),nn.Linear(latent,96)).to(device); decoder.load_state_dict(torch.load(decoder_path,map_location="cpu",weights_only=True)); decoder.eval().requires_grad_(False); head=SymmetricPairLatentFusion(latent,6,8).to(device); train=np.flatnonzero((data["role"]==0)&(data["cardinality"]==2)); valid=np.flatnonzero((data["role"]==1)&(data["cardinality"]==2)); singles={}
    for s in range(len(data["sources"])):
        rows=np.flatnonzero((data["source"]==s)&(data["cardinality"]==1))
        for g in np.unique(data["pairs"][rows,0]):singles[(s,int(g))]=data["target"][rows[data["pairs"][rows,0]==g]].mean(0)
    supported=lambda rows:np.asarray([((int(data["source"][i]),int(data["pairs"][i,0])) in singles) and ((int(data["source"][i]),int(data["pairs"][i,1])) in singles) for i in rows]); train=train[supported(train)] if residual else train; valid=valid[supported(valid)] if residual else valid; source=data["source"][train]; weight=1/np.bincount(source)[source]; weight/=weight.sum(); opt=torch.optim.AdamW(head.parameters(),1e-4,weight_decay=1e-2); history=[]; saved=[]
    @torch.no_grad()
    def validate(epoch):
        values={}
        for contextual in (False,True):
            pred=[]
            for at in batches(len(valid),512,False):
                ix=valid[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); a,b=genes[p[:,0]],genes[p[:,1]]; cs=torch.as_tensor(data["context_state"][data["source"][ix]],device=device) if contextual else None; z=endpoint.world.transition(a,b,context_state=cs)[0]; pred.append(decoder(z+head(a,b,z,endpoint.world.relation_score(a,b,z))).cpu())
            joint=torch.cat(pred); full_truth=torch.as_tensor(data["target"][valid]); stem="exact" if contextual else "unknown"; pred=joint; truth=full_truth
            if residual:
                p=torch.as_tensor(data["pairs"][valid].astype("int64"),device=device); a,b=genes[p[:,0]],genes[p[:,1]]; cs=torch.as_tensor(data["context_state"][data["source"][valid]],device=device) if contextual else None; pred=joint-decoder(endpoint.world.transition(a,context_state=cs)[0]).cpu()-decoder(endpoint.world.transition(b,context_state=cs)[0]).cpu(); truth=full_truth-torch.as_tensor(np.stack([singles[(int(data["source"][i]),int(data["pairs"][i,0]))]+singles[(int(data["source"][i]),int(data["pairs"][i,1]))] for i in valid]))
            for group,mask in (("all",np.ones(len(valid),bool)),("retained",data["source"][valid]<7),("dixit",data["source"][valid]>=7)):
                for name,lo,hi in (("legacy",0,32),("residual",32,96)):values[f"{stem}_{group}_{name}"]=float(nn.functional.huber_loss(pred[mask,lo:hi],truth[mask,lo:hi]))
                if residual:
                    for name,lo,hi in (("legacy",0,32),("residual",32,96)):values[f"full_{stem}_{group}_{name}"]=float(nn.functional.huber_loss(joint[mask,lo:hi],full_truth[mask,lo:hi]))
        row={"epoch":epoch,**values,"selection_loss":float(np.mean([v for k,v in values.items() if k.startswith(("unknown_all_","exact_all_"))]))}; history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()}); print(json.dumps({"phase":"dixit_residual_latent" if residual else "dixit_symmetric_latent",**row}),flush=True)
    head.eval(); validate(0)
    for epoch in range(epochs):
        rows=np.random.default_rng(781+epoch).choice(train,len(train),replace=True,p=weight); head.train()
        for at in batches(len(rows),128,False):
            ix=rows[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); a,b=genes[p[:,0]],genes[p[:,1]]; cs=torch.as_tensor(data["context_state"][data["source"][ix]],device=device); cs=torch.where((torch.rand(len(ix),device=device)<.5)[:,None],torch.zeros_like(cs),cs)
            with torch.no_grad():z=endpoint.world.transition(a,b,context_state=cs)[0]; relation=endpoint.world.relation_score(a,b,z)
            pred=decoder(z+head(a,b,z,relation)); target=torch.as_tensor(data["target"][ix],device=device)
            if residual:
                pred=pred-decoder(endpoint.world.transition(a,context_state=cs)[0])-decoder(endpoint.world.transition(b,context_state=cs)[0]); target=target-torch.as_tensor(np.stack([singles[(int(data["source"][i]),int(data["pairs"][i,0]))]+singles[(int(data["source"][i]),int(data["pairs"][i,1]))] for i in ix]),device=device)
            loss=perturbseq_state_error(nn.functional.huber_loss(pred,target,reduction="none"),data,96); opt.zero_grad(); loss.mean().backward(); nn.utils.clip_grad_norm_(head.parameters(),1.); opt.step()
        head.eval(); validate(epoch+1)
    selected=min(range(len(history)),key=lambda i:history[i]["selection_loss"]); head.load_state_dict(saved[selected]); baseline=history[0]; metric=history[selected]; keys=[k for k in metric if (k.startswith("full_") if residual else ("_all_" in k or "_retained_" in k))]; preserved=all(metric[k]<=1.01*baseline[k] for k in keys); improved=metric["selection_loss"]<=.98*baseline["selection_loss"]; rp=torch.as_tensor(pack["pairs"].astype("int64"),device=device); rt=torch.as_tensor(pack["relations"].astype("float32"),device=device); rv=torch.nonzero((rp[:,0]*1000003+rp[:,1])%20==0).squeeze(1)
    @torch.no_grad()
    def relation(adapted):
        total=0.
        for at in batches(len(rv),4096,False):
            ix=rv[at]; p=rp[ix]; a,b=genes[p[:,0]],genes[p[:,1]]; z=endpoint.world.transition(a,b)[0]; z=z+head(a,b,z,endpoint.world.relation_score(a,b,z)) if adapted else z; total+=nn.functional.smooth_l1_loss(endpoint.world.relation_score(a,b,z),rt[ix],reduction="sum").item()
        return total/len(rv)/rt.shape[1]
    rel={"baseline":relation(False),"adapted":relation(True)}; advanced=selected>0 and improved and preserved and rel["adapted"]<=1.01*rel["baseline"]; out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); stem="dixit_residual_latent" if residual else "dixit_symmetric_latent"; torch.save(head.state_dict(),out/f"{stem}.pt"); result={"parameters":sum(p.numel() for p in head.parameters()),"training_doubles":len(train),"validation_doubles":len(valid),"baseline":baseline,"selected":metric,"relation":rel,"preserved":bool(preserved),"improved":bool(improved),"advanced":bool(advanced),"history":history}; (out/"metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2),flush=True); return result


def fit_transition_mixture(data_dir,endpoint_path,out_dir,d=384,latent=128,layers=6):
    device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(Path(data_dir)/"features_spectral_safe.npz")["state"].astype("float32"); data=np.load(Path(data_dir)/"perturbseq_world_v3_nested96.npz"); endpoint=load_residual_endpoint(endpoint_path,state.shape[1],device,d,latent,layers); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); train=np.flatnonzero((data["role"]==0)&(data["cardinality"]==2)); valid=np.flatnonzero((data["role"]==1)&(data["cardinality"]==2)); decode=lambda z:torch.cat((endpoint.legacy_decoder(z),endpoint.residual_decoder(z)),1)
    @torch.no_grad()
    def paths(rows,contextual):
        joint=[]; sequential=[]; additive=[]
        for start in range(0,len(rows),256):
            ix=rows[start:start+256]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); a,b=genes[p[:,0]],genes[p[:,1]]; cs=torch.as_tensor(data["context_state"][data["source"][ix]],device=device) if contextual else None; za=endpoint.world.transition(a,context_state=cs)[0]; zb=endpoint.world.transition(b,context_state=cs)[0]; j=decode(endpoint.world.transition(a,b,context_state=cs)[0]); ab=decode(endpoint.world.transition(b,state=za,context_state=cs)[0]); ba=decode(endpoint.world.transition(a,state=zb,context_state=cs)[0]); joint.append(j.cpu()); sequential.append(((ab+ba)/2).cpu()); additive.append((decode(za)+decode(zb)).cpu())
        return tuple(torch.cat(x).numpy() for x in (joint,sequential,additive))
    train_paths={k:paths(train,k=="exact") for k in ("unknown","exact")}; valid_paths={k:paths(valid,k=="exact") for k in ("unknown","exact")}; target=data["target"].astype("float32"); grid=[(round(a/10,1),round(b/10,1)) for a in range(11) for b in range(11-a)]
    def huber(x):
        q=np.abs(x); return float(np.mean(np.where(q<=1,.5*q*q,q-.5)))
    choices={}; fits={}
    for name,lo,hi in (("legacy",0,32),("residual",32,96)):
        scored=[]
        for ws,wa in grid:
            losses=[]
            for context,(j,s,a) in train_paths.items():
                pred=j+ws*(s-j)+wa*(a-j)
                for source in np.unique(data["source"][train]):
                    keep=data["source"][train]==source; losses.append(huber(pred[keep,lo:hi]-target[train][keep,lo:hi]))
            scored.append({"sequential_weight":ws,"additive_weight":wa,"source_macro_huber":float(np.mean(losses))})
        choices[name]=min(scored,key=lambda x:(x["source_macro_huber"],x["sequential_weight"]+x["additive_weight"])); fits[name]=scored
    def evaluate(mixed):
        out={}
        for context,(j,s,a) in valid_paths.items():
            for name,lo,hi in (("legacy",0,32),("residual",32,96)):
                w=choices[name]; pred=j+mixed*(w["sequential_weight"]*(s-j)+w["additive_weight"]*(a-j)); out[f"{context}_{name}_huber"]=huber(pred[:,lo:hi]-target[valid,lo:hi])
        out["selection_loss"]=float(np.mean(list(out.values()))); return out
    baseline=evaluate(0); selected=evaluate(1); keys=("unknown_legacy_huber","unknown_residual_huber","exact_legacy_huber","exact_residual_huber"); preserved=all(selected[k]<=1.01*baseline[k] for k in keys); improved=selected["selection_loss"]<=.98*baseline["selection_loss"]; advanced=preserved and improved; out=Path(out_dir); np.savez(out/"transition_mixture.npz",legacy=np.array([choices["legacy"]["sequential_weight"],choices["legacy"]["additive_weight"]]),residual=np.array([choices["residual"]["sequential_weight"],choices["residual"]["additive_weight"]])); result={"parameters":4,"training_doubles":len(train),"validation_doubles":len(valid),"weights":choices,"baseline":baseline,"selected":selected,"preserved":bool(preserved),"improved":bool(improved),"advanced":bool(advanced),"fit_grid":fits}; (out/"transition_mixture_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="fit_grid"}),flush=True); return result


def fit_perturbseq_action_calibration(data_dir,endpoint_path,dependency_path,out_dir,epochs=20,d=384,latent=128,layers=6):
    torch.manual_seed(731); np.random.seed(731); device="cuda" if torch.cuda.is_available() else "cpu"; pack=np.load(Path(data_dir)/"features_spectral_safe.npz"); state=pack["state"].astype("float32"); data=np.load(Path(data_dir)/"perturbseq_world_v3_nested96.npz"); dependency=np.load(Path(data_dir)/"dependency_landscape.npz"); endpoint=load_residual_endpoint(endpoint_path,state.shape[1],device,d,latent,layers); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); calibration=DiagonalActionCalibration(latent).to(device); train=np.flatnonzero(data["role"]==0); valid=np.flatnonzero(data["role"]==1); source=data["source"][train]; weight=1/np.bincount(source)[source]; weight*=np.where(data["cardinality"][train]==2,5.,1.); weight/=weight.sum(); opt=torch.optim.AdamW(calibration.parameters(),5e-3,weight_decay=1e-3); decode=lambda z:torch.cat((endpoint.legacy_decoder(z),endpoint.residual_decoder(z)),1); history=[]; saved=[]
    @torch.no_grad()
    def expression(epoch):
        row={"epoch":epoch}
        for contextual in (False,True):
            pred=[]
            for start in range(0,len(valid),512):
                ix=valid[start:start+512]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); second=p[:,1]>=0; cs=torch.as_tensor(data["context_state"][data["source"][ix]],device=device) if contextual else None; value=torch.empty((len(ix),96),device=device); a=calibration(genes[p[:,0]])
                if (~second).any():value[~second]=decode(endpoint.world.transition(a[~second],context_state=cs[~second] if cs is not None else None)[0])
                if second.any():value[second]=decode(endpoint.world.transition(a[second],calibration(genes[p[second,1]]),context_state=cs[second] if cs is not None else None)[0])
                pred.append(value.cpu())
            pred=torch.cat(pred); truth=torch.as_tensor(data["target"][valid]); double=torch.as_tensor(data["cardinality"][valid]==2); context="exact" if contextual else "unknown"
            for name,lo,hi in (("legacy",0,32),("residual",32,96)):
                row[f"{context}_{name}_all_huber"]=float(nn.functional.huber_loss(pred[:,lo:hi],truth[:,lo:hi])); row[f"{context}_{name}_double_huber"]=float(nn.functional.huber_loss(pred[double,lo:hi],truth[double,lo:hi]))
        row["legacy_score"]=sum(row[k] for k in row if "legacy" in k); row["residual_score"]=sum(row[k] for k in row if "residual" in k); row["selection_loss"]=(row["legacy_score"]+row["residual_score"])/2; row["double_mean"]=float(np.mean([row[k] for k in row if "double_huber" in k])); history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in calibration.state_dict().items()}); print(json.dumps({"phase":"perturbseq_action_calibration",**row}),flush=True)
    calibration.eval(); expression(0)
    for epoch in range(epochs):
        chosen=np.random.default_rng(731+epoch).choice(train,len(train),replace=True,p=weight); calibration.train(); total=0.
        for start in range(0,len(chosen),512):
            ix=chosen[start:start+512]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); second=p[:,1]>=0; cs=torch.as_tensor(data["context_state"][data["source"][ix]],device=device); cs=torch.where((torch.rand(len(ix),device=device)<.5)[:,None],torch.zeros_like(cs),cs); target=torch.as_tensor(data["target"][ix],device=device); pred=torch.empty_like(target); a=calibration(genes[p[:,0]])
            if (~second).any():pred[~second]=decode(endpoint.world.transition(a[~second],context_state=cs[~second])[0])
            if second.any():pred[second]=decode(endpoint.world.transition(a[second],calibration(genes[p[second,1]]),context_state=cs[second])[0])
            loss=.5*nn.functional.huber_loss(pred[:,:32],target[:,:32])+.5*nn.functional.huber_loss(pred[:,32:],target[:,32:]); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(calibration.parameters(),1.); opt.step(); total+=loss.item()*len(ix)
        calibration.eval(); expression(epoch+1); history[-1]["train_huber"]=total/len(chosen)
    selected=min(range(len(history)),key=lambda i:history[i]["selection_loss"]); calibration.load_state_dict(saved[selected]); calibration.eval(); adapted=calibration(genes); relation_pairs=torch.as_tensor(pack["pairs"].astype("int64"),device=device); relation_target=torch.as_tensor(pack["relations"].astype("float32"),device=device); relation_valid=torch.nonzero((relation_pairs[:,0]*1000003+relation_pairs[:,1])%20==0).squeeze(1)
    @torch.no_grad()
    def relation(actions):
        total=0.
        for at in batches(len(relation_valid),4096,False):
            ix=relation_valid[at]; p=relation_pairs[ix]; a,b=actions[p[:,0]],actions[p[:,1]]; total+=nn.functional.smooth_l1_loss(endpoint.world.relation_score(a,b,endpoint.world.transition(a,b)[0]),relation_target[ix],reduction="sum").item()
        return total/len(relation_valid)/relation_target.shape[1]
    decoder=dependency_landscape_head(latent,16).to(device); decoder.load_state_dict(torch.load(dependency_path,map_location="cpu",weights_only=True)); decoder.eval().requires_grad_(False)
    @torch.no_grad()
    def dependency_metric(actions,rows):
        pred=[]
        for at in batches(len(rows),2048,False):pred.append(decoder(endpoint.world.transition(actions[torch.as_tensor(rows[at],device=device)])[0]).cpu())
        truth=torch.as_tensor(dependency["target"][rows,:16]); return {"huber":float(nn.functional.huber_loss(torch.cat(pred),truth)),"cosine":float(nn.functional.cosine_similarity(torch.cat(pred),truth).mean())}
    relation_metric={"baseline":relation(genes),"adapted":relation(adapted)}; dependency_metric_all={name:{"baseline":dependency_metric(genes,rows),"adapted":dependency_metric(adapted,rows)} for name,rows in (("generic",np.flatnonzero(dependency["valid"])),("intervention_isolated",np.flatnonzero(dependency["excluded"])))}; baseline=history[0]; metric=history[selected]; expression_preserved=metric["legacy_score"]<=1.005*baseline["legacy_score"] and metric["residual_score"]<=1.005*baseline["residual_score"]; expression_improved=metric["selection_loss"]<=.995*baseline["selection_loss"] and metric["double_mean"]<=.98*baseline["double_mean"]; relation_preserved=relation_metric["adapted"]<=1.01*relation_metric["baseline"]; dependency_preserved=all(x["adapted"]["huber"]<=1.01*x["baseline"]["huber"] for x in dependency_metric_all.values()); advanced=selected>0 and expression_preserved and expression_improved and relation_preserved and dependency_preserved; scale=torch.exp(.1*torch.tanh(calibration.log_scale)).detach().cpu(); out=Path(out_dir); torch.save(calibration.state_dict(),out/"perturbseq_action_calibration.pt"); result={"parameters":128,"training_rows":len(train),"validation_rows":len(valid),"selected":metric,"baseline":baseline,"scale":{"minimum":float(scale.min()),"maximum":float(scale.max()),"mean":float(scale.mean())},"relation":relation_metric,"dependency":dependency_metric_all,"expression_preserved":bool(expression_preserved),"expression_improved":bool(expression_improved),"relation_preserved":bool(relation_preserved),"dependency_preserved":bool(dependency_preserved),"advanced":bool(advanced),"history":history}; (out/"perturbseq_action_calibration_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"}),flush=True); return result


def fit_single_trained_action_rotation(data_dir,endpoint_path,dependency_path,out_dir,epochs=30,d=384,latent=128,layers=6):
    torch.manual_seed(731); np.random.seed(731); device="cuda" if torch.cuda.is_available() else "cpu"; pack=np.load(Path(data_dir)/"features_spectral_safe.npz"); state=pack["state"].astype("float32"); data=np.load(Path(data_dir)/"perturbseq_world_v3_nested96.npz"); dependency=np.load(Path(data_dir)/"dependency_landscape.npz"); endpoint=load_residual_endpoint(endpoint_path,state.shape[1],device,d,latent,layers); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); adapter=LowRankActionRotation(latent,8).to(device); train=np.flatnonzero((data["role"]==0)&(data["cardinality"]==1)); valid=np.flatnonzero((data["role"]==1)&(data["cardinality"]==1)); held_double=np.flatnonzero((data["role"]==1)&(data["cardinality"]==2)); source=data["source"][train]; weight=1/np.bincount(source)[source]; weight/=weight.sum(); opt=torch.optim.AdamW(adapter.parameters(),1e-3,weight_decay=1e-3); decode=lambda z:torch.cat((endpoint.legacy_decoder(z),endpoint.residual_decoder(z)),1); history=[]; saved=[]
    @torch.no_grad()
    def single_metric(epoch):
        row={"epoch":epoch}
        for contextual in (False,True):
            pred=[]
            for start in range(0,len(valid),512):
                ix=valid[start:start+512]; a=adapter(genes[torch.as_tensor(data["pairs"][ix,0].astype("int64"),device=device)]); cs=torch.as_tensor(data["context_state"][data["source"][ix]],device=device) if contextual else None; pred.append(decode(endpoint.world.transition(a,context_state=cs)[0]).cpu())
            pred=torch.cat(pred); truth=torch.as_tensor(data["target"][valid]); context="exact" if contextual else "unknown"
            for name,lo,hi in (("legacy",0,32),("residual",32,96)):row[f"{context}_{name}_huber"]=float(nn.functional.huber_loss(pred[:,lo:hi],truth[:,lo:hi]))
        row["legacy_score"]=row["unknown_legacy_huber"]+row["exact_legacy_huber"]; row["residual_score"]=row["unknown_residual_huber"]+row["exact_residual_huber"]; row["selection_loss"]=(row["legacy_score"]+row["residual_score"])/2; history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}); print(json.dumps({"phase":"single_trained_action_rotation",**row}),flush=True)
    adapter.eval(); single_metric(0)
    for epoch in range(epochs):
        chosen=np.random.default_rng(731+epoch).choice(train,len(train),replace=True,p=weight); adapter.train(); total=0.
        for start in range(0,len(chosen),512):
            ix=chosen[start:start+512]; a=adapter(genes[torch.as_tensor(data["pairs"][ix,0].astype("int64"),device=device)]); cs=torch.as_tensor(data["context_state"][data["source"][ix]],device=device); cs=torch.where((torch.rand(len(ix),device=device)<.5)[:,None],torch.zeros_like(cs),cs); pred=decode(endpoint.world.transition(a,context_state=cs)[0]); target=torch.as_tensor(data["target"][ix],device=device); loss=.5*nn.functional.huber_loss(pred[:,:32],target[:,:32])+.5*nn.functional.huber_loss(pred[:,32:],target[:,32:]); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(adapter.parameters(),1.); opt.step(); total+=loss.item()*len(ix)
        adapter.eval(); single_metric(epoch+1); history[-1]["train_huber"]=total/len(chosen)
    selected=min(range(len(history)),key=lambda i:history[i]["selection_loss"])
    @torch.no_grad()
    def double_metric(state_dict):
        adapter.load_state_dict(state_dict); row={}
        for contextual in (False,True):
            p=torch.as_tensor(data["pairs"][held_double].astype("int64"),device=device); cs=torch.as_tensor(data["context_state"][data["source"][held_double]],device=device) if contextual else None; pred=decode(endpoint.world.transition(adapter(genes[p[:,0]]),adapter(genes[p[:,1]]),context_state=cs)[0]).cpu(); truth=torch.as_tensor(data["target"][held_double]); context="exact" if contextual else "unknown"
            for name,lo,hi in (("legacy",0,32),("residual",32,96)):row[f"{context}_{name}_huber"]=float(nn.functional.huber_loss(pred[:,lo:hi],truth[:,lo:hi]))
        row["mean_huber"]=float(np.mean(list(row.values()))); return row
    double={"baseline":double_metric(saved[0]),"adapted":double_metric(saved[selected])}; adapter.load_state_dict(saved[selected]); adapter.eval(); adapted=adapter(genes); relation_pairs=torch.as_tensor(pack["pairs"].astype("int64"),device=device); relation_target=torch.as_tensor(pack["relations"].astype("float32"),device=device); relation_valid=torch.nonzero((relation_pairs[:,0]*1000003+relation_pairs[:,1])%20==0).squeeze(1)
    @torch.no_grad()
    def relation(actions):
        total=0.
        for at in batches(len(relation_valid),4096,False):
            ix=relation_valid[at]; p=relation_pairs[ix]; a,b=actions[p[:,0]],actions[p[:,1]]; total+=nn.functional.smooth_l1_loss(endpoint.world.relation_score(a,b,endpoint.world.transition(a,b)[0]),relation_target[ix],reduction="sum").item()
        return total/len(relation_valid)/relation_target.shape[1]
    decoder=dependency_landscape_head(latent,16).to(device); decoder.load_state_dict(torch.load(dependency_path,map_location="cpu",weights_only=True)); decoder.eval().requires_grad_(False)
    @torch.no_grad()
    def dep(actions,rows):
        pred=[]
        for at in batches(len(rows),2048,False):pred.append(decoder(endpoint.world.transition(actions[torch.as_tensor(rows[at],device=device)])[0]).cpu())
        truth=torch.as_tensor(dependency["target"][rows,:16]); return {"huber":float(nn.functional.huber_loss(torch.cat(pred),truth)),"cosine":float(nn.functional.cosine_similarity(torch.cat(pred),truth).mean())}
    relation_metric={"baseline":relation(genes),"adapted":relation(adapted)}; dependency_metric={name:{"baseline":dep(genes,rows),"adapted":dep(adapted,rows)} for name,rows in (("generic",np.flatnonzero(dependency["valid"])),("intervention_isolated",np.flatnonzero(dependency["excluded"])))}; baseline=history[0]; metric=history[selected]; single_preserved=metric["legacy_score"]<=1.005*baseline["legacy_score"] and metric["residual_score"]<=1.005*baseline["residual_score"]; single_improved=metric["selection_loss"]<=.995*baseline["selection_loss"]; double_preserved=all(double["adapted"][k]<=1.01*double["baseline"][k] for k in double["adapted"] if k!="mean_huber"); double_improved=double["adapted"]["mean_huber"]<=.98*double["baseline"]["mean_huber"]; relation_preserved=relation_metric["adapted"]<=1.01*relation_metric["baseline"]; dependency_preserved=all(x["adapted"]["huber"]<=1.01*x["baseline"]["huber"] for x in dependency_metric.values()); displacement=(adapted-genes).norm(dim=1)/(genes.norm(dim=1)+1e-8); advanced=selected>0 and single_preserved and single_improved and double_preserved and double_improved and relation_preserved and dependency_preserved; out=Path(out_dir); torch.save(adapter.state_dict(),out/"single_trained_action_rotation.pt"); result={"parameters":sum(p.numel() for p in adapter.parameters()),"training_singles":len(train),"validation_singles":len(valid),"held_doubles":len(held_double),"baseline":baseline,"selected":metric,"double":double,"relation":relation_metric,"dependency":dependency_metric,"displacement":{"mean":float(displacement.mean()),"maximum":float(displacement.max())},"single_preserved":bool(single_preserved),"single_improved":bool(single_improved),"double_preserved":bool(double_preserved),"double_improved":bool(double_improved),"relation_preserved":bool(relation_preserved),"dependency_preserved":bool(dependency_preserved),"advanced":bool(advanced),"history":history}; (out/"single_trained_action_rotation_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"}),flush=True); return result


def fit_safe_symmetric_pair_fusion(data_dir,endpoint_path,out_dir,epochs=30,d=384,latent=128,layers=6):
    torch.manual_seed(731); np.random.seed(731); device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(Path(data_dir)/"features_spectral_safe.npz")["state"].astype("float32"); data=np.load(Path(data_dir)/"perturbseq_world_v3_nested96.npz"); endpoint=load_residual_endpoint(endpoint_path,state.shape[1],device,d,latent,layers); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); head=SymmetricPairFusion(latent,6,8).to(device); train=np.flatnonzero((data["role"]==0)&(data["cardinality"]==2)); valid=np.flatnonzero((data["role"]==1)&(data["cardinality"]==2)); source=data["source"][train]; weight=1/np.bincount(source)[source]; weight/=weight.sum(); opt=torch.optim.AdamW(head.parameters(),1e-3,weight_decay=1e-3); decode=lambda z:torch.cat((endpoint.legacy_decoder(z),endpoint.residual_decoder(z)),1); history=[]; saved=[]
    def predict(rows,contextual,grad=False):
        values=[]
        context=torch.enable_grad() if grad else torch.no_grad()
        with context:
            for start in range(0,len(rows),256):
                ix=rows[start:start+256]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); a,b=genes[p[:,0]],genes[p[:,1]]; cs=torch.as_tensor(data["context_state"][data["source"][ix]],device=device) if contextual else None; z=endpoint.world.transition(a,b,context_state=cs)[0]; values.append(decode(z)+head(a,b,z,endpoint.world.relation_score(a,b,z)))
        return torch.cat(values)
    @torch.no_grad()
    def validate(epoch):
        row={"epoch":epoch}; truth=torch.as_tensor(data["target"][valid],device=device)
        for contextual in (False,True):
            pred=predict(valid,contextual); context="exact" if contextual else "unknown"
            for name,lo,hi in (("legacy",0,32),("residual",32,96)):row[f"{context}_{name}_huber"]=float(nn.functional.huber_loss(pred[:,lo:hi],truth[:,lo:hi]))
        row["selection_loss"]=float(np.mean([v for k,v in row.items() if k.endswith("_huber")])); history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()}); print(json.dumps({"phase":"safe_symmetric_pair_fusion",**row}),flush=True)
    head.eval(); validate(0)
    for epoch in range(epochs):
        chosen=np.random.default_rng(731+epoch).choice(train,len(train),replace=True,p=weight); head.train(); total=0.
        for start in range(0,len(chosen),128):
            ix=chosen[start:start+128]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); a,b=genes[p[:,0]],genes[p[:,1]]; cs=torch.as_tensor(data["context_state"][data["source"][ix]],device=device); cs=torch.where((torch.rand(len(ix),device=device)<.5)[:,None],torch.zeros_like(cs),cs)
            with torch.no_grad():z=endpoint.world.transition(a,b,context_state=cs)[0]; base=decode(z); relation=endpoint.world.relation_score(a,b,z)
            pred=base+head(a,b,z,relation); target=torch.as_tensor(data["target"][ix],device=device); loss=.5*nn.functional.huber_loss(pred[:,:32],target[:,:32])+.5*nn.functional.huber_loss(pred[:,32:],target[:,32:]); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(head.parameters(),1.); opt.step(); total+=loss.item()*len(ix)
        head.eval(); validate(epoch+1); history[-1]["train_huber"]=total/len(chosen)
    selected=min(range(len(history)),key=lambda i:history[i]["selection_loss"]); head.load_state_dict(saved[selected]); baseline=history[0]; metric=history[selected]; keys=("unknown_legacy_huber","unknown_residual_huber","exact_legacy_huber","exact_residual_huber"); preserved=all(metric[k]<=1.01*baseline[k] for k in keys); improved=metric["selection_loss"]<=.98*baseline["selection_loss"]; advanced=selected>0 and preserved and improved; out=Path(out_dir); torch.save(head.state_dict(),out/"safe_symmetric_pair_fusion.pt"); result={"parameters":sum(p.numel() for p in head.parameters()),"training_doubles":len(train),"validation_doubles":len(valid),"baseline":baseline,"selected":metric,"preserved":bool(preserved),"improved":bool(improved),"advanced":bool(advanced),"history":history}; (out/"safe_symmetric_pair_fusion_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!="history"}),flush=True); return result


@torch.no_grad()
def source_invariant_features(endpoint,head,genes,pairs,context_state):
    world=endpoint.world; joint=world.transition(genes[pairs[:,0]],genes[pairs[:,1]],context_state=context_state)[0]; a=world.transition(genes[pairs[:,0]],context_state=context_state)[0]; b=world.transition(genes[pairs[:,1]],context_state=context_state)[0]; features=[]
    rms=lambda x:(x.square().mean(1)+1e-8).sqrt(); cosine=lambda x,y:(x*y).sum(1)/(x.norm(dim=1)*y.norm(dim=1)+1e-8)
    for decoder in head.decoders:
        j,sa,sb=decoder(joint),decoder(a),decoder(b); additive=sa+sb; residual=j-additive
        features.extend((rms(j),rms(residual),rms(additive),rms(sa-sb),cosine(sa,sb),cosine(j,additive),cosine(j,residual)))
    return torch.stack(features,1)


def fit_source_invariant_ridge(data_dir,endpoint_path,interaction_path,source_path,out_dir,alphas=(1.,10.,100.,1000.,10000.),d=384,latent=128,layers=6):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    device="cuda" if torch.cuda.is_available() else "cpu"; state=np.load(Path(data_dir)/"features_spectral_safe.npz")["state"].astype("float32"); data=np.load(Path(data_dir)/"slkb_outcomes_intervention_depmap_world.npz",allow_pickle=True); endpoint=load_residual_endpoint(endpoint_path,state.shape[1],device,d,latent,layers); genes=torch.as_tensor(encode_genes(endpoint.world,state,device),device=device); interaction=ResidualInteraction(interaction_head(latent).to(device)).to(device); interaction.load_state_dict(torch.load(interaction_path,map_location="cpu",weights_only=True)); interaction.eval().requires_grad_(False); source=SourceEndpoint(5,latent,32).to(device); source.load_state_dict(torch.load(source_path,map_location="cpu",weights_only=True)); source.eval().requires_grad_(False); known=data["context_known"][data["context"]]; held=np.arange(int(data["pairs"].max())+1)%5==0; train=np.flatnonzero(known&(~held[data["pairs"]]).all(1)); valid=np.flatnonzero(known&held[data["pairs"]].all(1)); rows=np.r_[train,valid]; feature=np.zeros((len(data["pairs"]),35),"float32"); base=np.zeros(len(data["pairs"]),"float32")
    for at in batches(len(rows),2048,False):
        ix=rows[at]; p=torch.as_tensor(data["pairs"][ix].astype("int64"),device=device); cs=torch.as_tensor(data["context_state"][data["context"][ix]].astype("float32"),device=device); z,r=residual_interaction_inputs(endpoint,genes,p,cs); base[ix]=interaction(z,r)[:,0].cpu(); feature[ix]=source_invariant_features(endpoint,source,genes,p,cs).cpu()
    def weights(ix):
        c=data["context"][ix]; return len(ix)/(len(np.unique(c))*np.bincount(c)[c])
    def fit_predict(alpha,tr,va):
        scale=StandardScaler().fit(feature[tr]); model=Ridge(alpha=alpha).fit(scale.transform(feature[tr]),data["target"][tr,0]-base[tr],sample_weight=weights(tr)); return base[va]+model.predict(scale.transform(feature[va])),scale,model
    inner=[]; role=(np.arange(int(data["pairs"].max())+1)//5)%4
    for alpha in alphas:
        folds=[]
        for fold in range(4):
            tr=train[(role[data["pairs"][train]]!=fold).all(1)]; va=train[(role[data["pairs"][train]]==fold).all(1)]; pred,_,_=fit_predict(alpha,tr,va); folds.append({"fold":fold,"train_rows":len(tr),"validation_rows":len(va),**interaction_array_metrics(pred,data,va)})
        inner.append({"alpha":alpha,"selection_loss":float(np.mean([x["row_huber"]+x["pair_huber"] for x in folds])),"folds":folds})
    selected=min(inner,key=lambda x:(x["selection_loss"],-x["alpha"])); pred,scale,model=fit_predict(selected["alpha"],train,valid); baseline=interaction_array_metrics(base[valid],data,valid); candidate=interaction_array_metrics(pred,data,valid); preserved=all(candidate[k]<=1.01*baseline[k] for k in ("row_huber","pair_huber")) and all(candidate[k]>=baseline[k]-.005 for k in ("row_correlation","pair_correlation","row_spearman","pair_spearman")); improved=any(candidate[k]<=.98*baseline[k] for k in ("row_huber","pair_huber")) or any(candidate[k]>=baseline[k]+.01 for k in ("row_correlation","pair_correlation","row_spearman","pair_spearman")); out=Path(out_dir); np.savez(out/"source_landmark_invariant_ridge.npz",mean=scale.mean_,scale=scale.scale_,coef=model.coef_,intercept=model.intercept_,alpha=selected["alpha"]); result={"features":35,"train_rows":len(train),"validation_rows":len(valid),"selected_alpha":selected["alpha"],"baseline":baseline,"candidate":candidate,"preserved":bool(preserved),"improved":bool(improved),"advanced":bool(preserved and improved),"inner":inner}; (out/"source_landmark_invariant_ridge_metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True); return result


def pretrain(model, state, pairs, relations, device, epochs=12, batch=2048, epoch_pairs=160000):
    batch=fit_batch(model,batch)
    opt = torch.optim.AdamW(model.parameters(), 2e-4, weight_decay=1e-3)
    state = torch.as_tensor(state, dtype=torch.float32); pairs = torch.as_tensor(pairs.astype("int64"))
    relations = torch.as_tensor(relations, dtype=torch.float32)
    valid=torch.nonzero((pairs[:,0]*1000003+pairs[:,1])%20==0).squeeze(1); keep=torch.ones(len(pairs),dtype=torch.bool); keep[valid]=False; pool=torch.nonzero(keep).squeeze(1)
    best=None; best_val=float("inf")
    for epoch in range(epochs):
        model.train(); total = 0; chosen = pool[torch.randperm(len(pool))[:min(epoch_pairs, len(pool))]]
        for pick in batches(len(chosen), batch):
            ix = chosen[pick]
            p = pairs[ix]; x, y = state[p[:, 0]].to(device), state[p[:, 1]].to(device); target = relations[ix].to(device)
            a, b = model.encode(torch.cat((x, y)), .15).chunk(2)
            sa, sb = model.transition(torch.cat((a, b)))[0].chunk(2); joint, logsd = model.transition(a, b)
            n = max(1, len(ix)//4); seq, _ = model.transition(torch.cat((b[:n], a[:n])),
                state=torch.cat((sa[:n], sb[:n]))); ab, ba = seq.chunk(2)
            loss = .05*(nn.functional.mse_loss(model.reconstruct(a), x)+nn.functional.mse_loss(model.reconstruct(b), y))
            loss += nn.functional.mse_loss(model.decode_state(sa), x[:, -model.decode_state.out_features:])
            loss += nn.functional.mse_loss(model.decode_state(sb), y[:, -model.decode_state.out_features:])
            loss += nn.functional.smooth_l1_loss(model.relation_score(a, b, joint), target)
            loss += .1*(nn.functional.mse_loss(joint[:n], ab)+nn.functional.mse_loss(joint[:n], ba))
            loss += 1e-4*(joint.square().mean()+logsd.exp().mean())
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
            total += loss.item()*len(ix)
        val=relation_loss(model,state,pairs,relations,valid,device)
        if val<best_val: best_val=val; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(json.dumps({"phase":"pretrain","epoch":epoch+1,"loss":total/len(chosen),"val_relation":val}), flush=True)
    model.load_state_dict(best)


def reinforce(model, state, pairs, relations, device, epochs=3, batch=1024, group=4, epoch_pairs=160000):
    batch=fit_batch(model,batch)
    opt = torch.optim.AdamW(model.parameters(), 2e-5, weight_decay=1e-3)
    state = torch.as_tensor(state, dtype=torch.float32); pairs = torch.as_tensor(pairs.astype("int64"))
    relations = torch.as_tensor(relations, dtype=torch.float32)
    valid=torch.nonzero((pairs[:,0]*1000003+pairs[:,1])%20==0).squeeze(1); pool=torch.nonzero((pairs[:,0]*1000003+pairs[:,1])%20!=0).squeeze(1)
    best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; best_val=relation_loss(model,state,pairs,relations,valid,device)
    for epoch in range(epochs):
        model.train(); total = 0; chosen = pool[torch.randperm(len(pool))[:min(epoch_pairs, len(pool))]]
        for pick in batches(len(chosen), batch):
            ix = chosen[pick]
            p = pairs[ix]; x, y = state[p[:, 0]].to(device), state[p[:, 1]].to(device); target = relations[ix].to(device)
            a, b = model.encode(torch.cat((x, y))).chunk(2); mu, logsd = model.transition(a, b); sd = logsd.exp()
            sample = (mu[:, None] + sd[:, None]*torch.randn(len(ix), group, mu.shape[1], device=device)).detach()
            aa = a[:, None].expand(-1, group, -1).reshape(-1, a.shape[1]); bb = b[:, None].expand_as(a[:, None].expand(-1, group, -1)).reshape_as(aa)
            pred = model.relation_score(aa, bb, sample.reshape(-1, mu.shape[1])).view(len(ix), group, -1)
            reward = -(pred-target[:, None]).square().mean(2)
            advantage = (reward-reward.mean(1, keepdim=True))/(reward.std(1, keepdim=True)+1e-5)
            logp = torch.distributions.Normal(mu[:, None], sd[:, None]).log_prob(sample).mean(2)
            loss = -(advantage.detach()*logp).mean() - .001*torch.distributions.Normal(mu, sd).entropy().mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
            total += loss.item()*len(ix)
        val=relation_loss(model,state,pairs,relations,valid,device)
        if val<best_val: best_val=val; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(json.dumps({"phase":"reinforce","epoch":epoch+1,"loss":total/len(chosen),"val_relation":val}), flush=True)
    model.load_state_dict(best)


def train_outcomes(model, state, data, device, epochs=5, batch=2048, cold=False, head_only=False):
    batch=fit_batch(model,batch)
    if head_only:
        torch.manual_seed(SEED+7); model.outcome.reset_parameters(); model.requires_grad_(False); model.outcome.requires_grad_(True)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),1e-4,weight_decay=1e-3); state=torch.as_tensor(state,dtype=torch.float32)
    pairs=torch.as_tensor(data["pairs"].astype("int64")); context=torch.as_tensor(data["context"].astype("int64")); target=torch.as_tensor(data["target"],dtype=torch.float32)
    context_state=torch.as_tensor(data["context_state"],dtype=torch.float32,device=device) if "context_state" in data.files else None
    gene_context=torch.as_tensor(data["gene_context_state"],dtype=torch.float32,device=device) if "gene_context_state" in data.files else None
    train,held=outcome_split(pairs,cold)
    best=None; best_val=float("inf")
    for epoch in range(epochs):
        model.train(not head_only); model.outcome.train(); total=0
        for at in batches(len(train),batch):
            ix=train[at]
            p=pairs[ix]; a,b=model.encode(torch.cat((state[p[:,0]],state[p[:,1]])).to(device)).chunk(2)
            c=context[ix].to(device); c=torch.where(torch.rand(len(c),device=device)<.2,-1,c)
            cs=context_features(context_state,gene_context,c,p) if context_state is not None else None; joint,_=model.transition(a,b,context=c,context_state=cs); pred=model.outcome(joint)
            loss=nn.functional.huber_loss(pred,target[ix].to(device))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); total+=loss.item()*len(ix)
        val=outcome_loss(model,state,pairs,context,target,held,device,context_state,gene_context)
        if val<best_val: best_val=val; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(json.dumps({"phase":"phenotype","epoch":epoch+1,"loss":total/len(train),"val":val}),flush=True)
    model.load_state_dict(best)
    model.requires_grad_(True)


def reinforce_outcomes(model, state, data, device, epochs=3, batch=1024, group=4, cold=False):
    batch=fit_batch(model,batch)
    opt=torch.optim.AdamW(model.parameters(),2e-5,weight_decay=1e-3); state=torch.as_tensor(state,dtype=torch.float32)
    pairs=torch.as_tensor(data["pairs"].astype("int64")); context=torch.as_tensor(data["context"].astype("int64")); target=torch.as_tensor(data["target"],dtype=torch.float32)
    context_state=torch.as_tensor(data["context_state"],dtype=torch.float32,device=device) if "context_state" in data.files else None
    gene_context=torch.as_tensor(data["gene_context_state"],dtype=torch.float32,device=device) if "gene_context_state" in data.files else None
    pool,held=outcome_split(pairs,cold)
    best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; best_val=outcome_loss(model,state,pairs,context,target,held,device,context_state,gene_context)
    for epoch in range(epochs):
        model.train(); total=0
        for at in batches(len(pool),batch):
            ix=pool[at]
            p=pairs[ix]; a,b=model.encode(torch.cat((state[p[:,0]],state[p[:,1]])).to(device)).chunk(2)
            c=context[ix].to(device); cs=context_features(context_state,gene_context,c,p) if context_state is not None else None; mu,logsd=model.transition(a,b,context=c,context_state=cs); sd=logsd.exp()
            sample=(mu[:,None]+sd[:,None]*torch.randn(len(ix),group,mu.shape[1],device=device)).detach()
            pred=model.outcome(sample.reshape(-1,mu.shape[1])).view(len(ix),group,-1)
            reward=-(pred-target[ix,None].to(device)).square().mean(2)
            adv=(reward-reward.mean(1,keepdim=True))/(reward.std(1,keepdim=True)+1e-5)
            logp=torch.distributions.Normal(mu[:,None],sd[:,None]).log_prob(sample).mean(2)
            loss=-(adv.detach()*logp).mean()-.001*torch.distributions.Normal(mu,sd).entropy().mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); total+=loss.item()*len(ix)
        val=outcome_loss(model,state,pairs,context,target,held,device,context_state,gene_context)
        if val<best_val: best_val=val; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(json.dumps({"phase":"phenotype_reinforce","epoch":epoch+1,"loss":total/len(pool),"val":val}),flush=True)
    model.load_state_dict(best)


@torch.no_grad()
def embed_pairs(model, genes, pairs, device, batch=4096):
    model.eval(); genes = torch.as_tensor(genes, dtype=torch.float32); pairs = torch.as_tensor(pairs.astype("int64")); out=[]
    for ix in batches(len(pairs), batch, False):
        p=pairs[ix]; a=model.encode(genes[p[:,0]].to(device)); b=model.encode(genes[p[:,1]].to(device))
        joint, logsd=model.transition(a,b); rel=model.relation_score(a,b,joint); outcome=model.outcome(joint)
        out.append(torch.cat(((a-b).abs(),a*b,joint,logsd,rel,outcome),1).cpu())
    return torch.cat(out).numpy()


def fit_head(xtr, ytr, xte, device, epochs=40):
    xtr=torch.as_tensor(xtr,dtype=torch.float32,device=device); ytr=torch.as_tensor(ytr,dtype=torch.float32,device=device)
    head=nn.Sequential(nn.LayerNorm(xtr.shape[1]),nn.Linear(xtr.shape[1],256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,1)).to(device)
    opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-2)
    for _ in range(epochs):
        head.train()
        for ix in batches(len(xtr),2048):
            loss=nn.functional.binary_cross_entropy_with_logits(head(xtr[ix]).squeeze(1),ytr[ix])
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval(); scores=[]
    xt=torch.as_tensor(xte,dtype=torch.float32)
    with torch.no_grad():
        for ix in batches(len(xt),4096,False): scores.append(head(xt[ix].to(device)).sigmoid().cpu())
    return torch.cat(scores).squeeze(1).numpy()


@torch.no_grad()
def encode_genes(model,state,device):
    x=torch.as_tensor(state,dtype=torch.float32); return torch.cat([model.encode(x[i:i+2048].to(device)).cpu() for i in range(0,len(x),2048)]).numpy()


@torch.no_grad()
def outcome_scores(model,genes,pairs,device,contexts,batch=256):
    model.eval(); genes=torch.as_tensor(genes,dtype=torch.float32,device=device); pairs=torch.as_tensor(pairs.astype("int64")); names=("direct_strength","depletion_mean","depletion_top3","depletion_max","excess_mean","excess_top3","excess_max","bliss_mean","bliss_top3","bliss_max","strength_mean","strength_top3","strength_max"); out={k:[] for k in names}
    for ix in batches(len(pairs),batch,False):
        p=pairs[ix].to(device); a,b=genes[p[:,0]],genes[p[:,1]]; out["direct_strength"].append(model.outcome(model.transition(a,b)[0])[:,1].cpu()); n=len(p); ctx=torch.arange(contexts,device=device).repeat(n); a=a[:,None].expand(-1,contexts,-1).reshape(-1,a.shape[1]); b=b[:,None].expand(-1,contexts,-1).reshape(-1,b.shape[1]); double=model.outcome(model.transition(a,b,context=ctx)[0]).reshape(n,contexts,-1); one=model.outcome(model.transition(a,context=ctx)[0]).reshape(n,contexts,-1); two=model.outcome(model.transition(b,context=ctx)[0]).reshape(n,contexts,-1)
        harm=-double[...,0]; one_harm=-one[...,0]; two_harm=-two[...,0]; values=(harm,harm-torch.maximum(one_harm,two_harm),harm-one_harm-two_harm,double[...,1])
        for stem,value in zip(("depletion","excess","bliss","strength"),values):
            out[f"{stem}_mean"].append(value.mean(1).cpu()); out[f"{stem}_top3"].append(value.topk(min(3,contexts),1).values.mean(1).cpu()); out[f"{stem}_max"].append(value.max(1).values.cpu())
    return {k:torch.cat(v).numpy() for k,v in out.items()}


def fit_metric(genes,tr,ytr,te,device,epochs=30):
    genes=torch.as_tensor(genes,dtype=torch.float32,device=device); tr=torch.as_tensor(tr.astype("int64"),device=device); ytr=torch.as_tensor(ytr,dtype=torch.float32,device=device)
    head=nn.Sequential(nn.LayerNorm(genes.shape[1]),nn.Linear(genes.shape[1],512),nn.GELU(),nn.Linear(512,128)).to(device); scale=nn.Parameter(torch.tensor(2.3,device=device)); bias=nn.Parameter(torch.zeros((),device=device)); opt=torch.optim.AdamW([*head.parameters(),scale,bias],3e-4,weight_decay=1e-3)
    for _ in range(epochs):
        head.train()
        for ix in batches(len(tr),2048):
            p=tr[ix]; a=nn.functional.normalize(head(genes[p[:,0]]),dim=1); b=nn.functional.normalize(head(genes[p[:,1]]),dim=1); score=(a*b).sum(1)*scale.exp()+bias
            loss=nn.functional.binary_cross_entropy_with_logits(score,ytr[ix]); opt.zero_grad(); loss.backward(); opt.step()
    head.eval(); z=nn.functional.normalize(head(genes),dim=1); te=torch.as_tensor(te.astype("int64"),device=device); return (((z[te[:,0]]*z[te[:,1]]).sum(1)*scale.exp()+bias).sigmoid().detach().cpu().numpy())


def fit_symmetric(genes,tr,ytr,te,device,epochs=40):
    genes=torch.as_tensor(genes,dtype=torch.float32,device=device); tr=torch.as_tensor(tr.astype("int64"),device=device); ytr=torch.as_tensor(ytr,dtype=torch.float32,device=device); head=SymmetricHead(genes.shape[1]).to(device); opt=torch.optim.AdamW(head.parameters(),3e-4,weight_decay=1e-2)
    for _ in range(epochs):
        head.train()
        for ix in batches(len(tr),2048):
            loss=nn.functional.binary_cross_entropy_with_logits(head(genes,tr[ix]),ytr[ix]); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(head.parameters(),2.); opt.step()
    head.eval(); te=torch.as_tensor(te.astype("int64")); out=[]
    with torch.no_grad():
        for ix in batches(len(te),4096,False): out.append(head(genes,te[ix].to(device)).sigmoid().cpu())
    return torch.cat(out).numpy()


def observed_relations(pack, pairs, n_genes):
    keys=pack["pairs"][:,0].astype("int64")*n_genes+pack["pairs"][:,1]; order=np.argsort(keys)
    query=pairs[:,0].astype("int64")*n_genes+pairs[:,1]; at=np.searchsorted(keys[order],query)
    return pack["relations"].astype("float32")[order[at]]


def pair_summary(state,pairs):
    a,b=state[pairs[:,0]].astype("float32"),state[pairs[:,1]].astype("float32"); out=[]
    anchor=(state.shape[1]-1624)//6
    for lo,hi in ((0,768),(768,1024),(1024,1424),(1424,1624),*((1624+i*anchor,1624+(i+1)*anchor) for i in range(6))):
        x,y=a[:,lo:hi],b[:,lo:hi]
        out += [(x*y).mean(1),np.abs(x-y).mean(1),((x-y)**2).mean(1),(x*y).sum(1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)+1e-6)]
    return np.column_stack(out)


def evaluate(model,state,pack,split,cvs,device,rich=False,perturb_decoder=None):
    rows=[]; perturb_genes=encode_genes(model,state,device) if perturb_decoder is not None else None
    for cv in cvs:
        for fold in range(5):
            trp,trn=split[f"cv{cv}_pos_train_{fold}"],split[f"cv{cv}_neg_train_{fold}"]; tep,ten=split[f"cv{cv}_pos_test_{fold}"],split[f"cv{cv}_neg_test_{fold}"]
            tr=np.concatenate((trp,trn)); te=np.concatenate((tep,ten)); ytr=np.r_[np.ones(len(trp)),np.zeros(len(trn))]; y=np.r_[np.ones(len(tep)),np.zeros(len(ten))]
            ztr=embed_pairs(model,state,tr,device); zte=embed_pairs(model,state,te,device); emergent=zte[:,-1]
            xtr=np.column_stack((ztr,pair_summary(state,tr) if rich else np.empty((len(tr),0)),observed_relations(pack,tr,len(state))))
            xte=np.column_stack((zte,pair_summary(state,te) if rich else np.empty((len(te),0)),observed_relations(pack,te,len(state))))
            score=fit_head(xtr,ytr,xte,device); row={"cv":cv,"fold":fold,"auroc":roc_auc_score(y,score),"aupr":average_precision_score(y,score),"f1":f1_score(y,score>=.5),"emergent_auroc":roc_auc_score(y,emergent),"emergent_aupr":average_precision_score(y,emergent)}
            if perturb_decoder is not None:
                state_score,fitness_score,sequential_score,order_score=perturbseq_scores(model,perturb_decoder,perturb_genes,te,device); row.update({"state_auroc":roc_auc_score(y,state_score),"state_aupr":average_precision_score(y,state_score),"sequential_auroc":roc_auc_score(y,sequential_score),"sequential_aupr":average_precision_score(y,sequential_score),"order_auroc":roc_auc_score(y,order_score),"order_aupr":average_precision_score(y,order_score)})
                if fitness_score is not None: row.update({"fitness_auroc":roc_auc_score(y,fitness_score),"fitness_aupr":average_precision_score(y,fitness_score)})
            rows.append(row); print(json.dumps(row),flush=True)
    return rows


def run(data_dir, out_dir, pretrain_epochs=12, rl_epochs=3, cvs=(1,2,3), d=384, latent=128, layers=6, feature_name="features.npz", outcome_name="slkb_outcomes.npz"):
    torch.manual_seed(SEED); np.random.seed(SEED); device="cuda" if torch.cuda.is_available() else "cpu"
    pack=np.load(Path(data_dir)/feature_name); split=np.load(Path(data_dir)/"splits.npz"); slkb_path=Path(data_dir)/outcome_name
    state=pack["state"].astype("float32"); slkb=np.load(slkb_path,allow_pickle=True) if slkb_path.exists() else None
    outcomes=slkb["target"].shape[1] if slkb is not None and slkb["target"].ndim==2 else 1
    model=SLPredict(d=d,latent=latent,layers=layers,contexts=len(slkb["contexts"]) if slkb is not None else 28,outcomes=outcomes,state_dim=state.shape[1]).to(device)
    print(json.dumps({"device":device,"parameters":model.count(),"d":d,"latent":latent,"layers":layers}),flush=True)
    pretrain(model,state,pack["pairs"],pack["relations"],device,pretrain_epochs)
    if slkb is not None: train_outcomes(model,state,slkb,device,max(2,pretrain_epochs//2))
    reinforce(model,state,pack["pairs"],pack["relations"],device,rl_epochs)
    if slkb is not None: reinforce_outcomes(model,state,slkb,device,rl_epochs)
    rows=evaluate(model,state,pack,split,cvs,device); Path(out_dir).mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),Path(out_dir)/"world_model.pt")
    Path(out_dir,"metrics.json").write_text(json.dumps(rows,indent=2))
    return rows


def run_pretrain(data_dir,out_dir,epochs=12,d=384,latent=128,layers=6,feature_name="features.npz"):
    torch.manual_seed(SEED); np.random.seed(SEED); device="cuda" if torch.cuda.is_available() else "cpu"; pack=np.load(Path(data_dir)/feature_name); state=pack["state"].astype("float32"); model=SLPredict(d,latent,layers,28,2,state.shape[1]).to(device)
    print(json.dumps({"device":device,"parameters":model.count(),"stage":"pretrain"}),flush=True); pretrain(model,state,pack["pairs"],pack["relations"],device,epochs); Path(out_dir).mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),Path(out_dir)/"world_model.pt"); return {"val_relation":relation_loss(model,torch.as_tensor(state),torch.as_tensor(pack["pairs"].astype("int64")),torch.as_tensor(pack["relations"],dtype=torch.float32),torch.nonzero((torch.as_tensor(pack["pairs"][:,0].astype("int64"))*1000003+torch.as_tensor(pack["pairs"][:,1].astype("int64")))%20==0).squeeze(1),device)}


def resume_outcomes(data_dir,model_path,out_dir,mode="cold_joint",epochs=6,rl_epochs=3,d=384,latent=128,layers=6,feature_name="features.npz",outcome_name="slkb_outcomes.npz"):
    torch.manual_seed(SEED); np.random.seed(SEED); device="cuda"; pack=np.load(Path(data_dir)/feature_name); state=pack["state"].astype("float32"); data=np.load(Path(data_dir)/outcome_name,allow_pickle=True); sd=torch.load(model_path,map_location="cpu",weights_only=True); context_dim=data["context_state"].shape[1]+(4*data["gene_context_state"].shape[2] if "gene_context_state" in data.files else 0) if "context_state" in data.files else 0; model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],context_dim).to(device); model.load_state_dict(sd,strict=False); head_only=mode=="cold_head"; train_outcomes(model,state,data,device,epochs,cold=True,head_only=head_only)
    if not head_only: reinforce_outcomes(model,state,data,device,rl_epochs,cold=True)
    rows=evaluate(model,state,pack,np.load(Path(data_dir)/"splits.npz"),(3,),device); Path(out_dir).mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),Path(out_dir)/"world_model.pt"); Path(out_dir,"metrics.json").write_text(json.dumps(rows,indent=2)); return rows


def resume_depmap_world(data_dir,model_path,out_dir,dependency_epochs=3,outcome_epochs=6,rl_epochs=3,d=384,latent=128,layers=6):
    torch.manual_seed(SEED); np.random.seed(SEED); device="cuda"; pack=np.load(Path(data_dir)/"features_spectral_safe.npz"); state=pack["state"].astype("float32"); dep=np.load(Path(data_dir)/"depmap_world.npz",allow_pickle=True); outcome=np.load(Path(data_dir)/"slkb_outcomes_intervention_depmap_world.npz",allow_pickle=True); sd=torch.load(model_path,map_location="cpu",weights_only=True); model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],outcome["context_state"].shape[1]).to(device); model.load_state_dict(sd,strict=False); dep_val=pretrain_dependency(model,state,dep,device,dependency_epochs); train_outcomes(model,state,outcome,device,outcome_epochs,cold=True); reinforce_outcomes(model,state,outcome,device,rl_epochs,cold=True); rows=evaluate(model,state,pack,np.load(Path(data_dir)/"splits.npz"),(3,),device); Path(out_dir).mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),Path(out_dir)/"world_model.pt"); Path(out_dir,"metrics.json").write_text(json.dumps(rows,indent=2)); Path(out_dir,"dependency_metrics.json").write_text(json.dumps({"validation_huber":dep_val},indent=2)); return rows


def resume_perturbseq(data_dir,model_path,out_dir,perturb_epochs=10,perturb_rl_epochs=3,outcome_epochs=6,d=384,latent=128,layers=6,perturb_name="perturbseq_world.npz",residual_weight=0.,context_selection=False,feature_name="features_spectral_safe.npz"):
    torch.manual_seed(SEED); np.random.seed(SEED); device="cuda"; pack=np.load(Path(data_dir)/feature_name); state=pack["state"].astype("float32"); perturb=np.load(Path(data_dir)/perturb_name,allow_pickle=True); outcome=np.load(Path(data_dir)/"slkb_outcomes_intervention_external.npz",allow_pickle=True); sd=torch.load(model_path,map_location="cpu",weights_only=True); model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1]).to(device); model.load_state_dict(sd); decoder,perturb_val=pretrain_perturbseq(model,state,perturb,device,perturb_epochs,perturb_rl_epochs,residual_weight=residual_weight,context_selection=context_selection); train_outcomes(model,state,outcome,device,outcome_epochs,cold=True,head_only=True); rows=evaluate(model,state,pack,np.load(Path(data_dir)/"splits.npz"),(3,),device,perturb_decoder=decoder); Path(out_dir).mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),Path(out_dir)/"world_model.pt"); torch.save(decoder.state_dict(),Path(out_dir)/"perturb_decoder.pt"); Path(out_dir,"metrics.json").write_text(json.dumps(rows,indent=2)); Path(out_dir,"perturbseq_metrics.json").write_text(json.dumps({"validation_score":perturb_val},indent=2)); return rows


def resume_basal_perturbseq(data_dir,model_path,out_dir,dependency_epochs=3,perturb_epochs=10,perturb_rl_epochs=3,outcome_epochs=6,d=384,latent=128,layers=6,feature_name="features_spectral_safe.npz",evaluate_cv3=True,perturb_name="perturbseq_world_v3.npz"):
    torch.manual_seed(SEED); np.random.seed(SEED); device="cuda"; pack=np.load(Path(data_dir)/feature_name); state=pack["state"].astype("float32"); dep=np.load(Path(data_dir)/"basal_context.npz"); perturb=np.load(Path(data_dir)/perturb_name); outcome=np.load(Path(data_dir)/"slkb_outcomes_intervention_external.npz",allow_pickle=True); sd=torch.load(model_path,map_location="cpu",weights_only=True); model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1],dep["cell_state"].shape[1]).to(device); model.load_state_dict(sd,strict=False); dependency_val=pretrain_dependency(model,state,dep,device,dependency_epochs); decoder,perturb_val=pretrain_perturbseq(model,state,perturb,device,perturb_epochs,perturb_rl_epochs,context_selection=True); block_scores=perturbseq_block_scores(model,decoder,state,perturb,device); train_outcomes(model,state,outcome,device,outcome_epochs,cold=True,head_only=True); rows=evaluate(model,state,pack,np.load(Path(data_dir)/"splits.npz"),(3,),device,perturb_decoder=decoder) if evaluate_cv3 else []; Path(out_dir).mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),Path(out_dir)/"world_model.pt"); torch.save(decoder.state_dict(),Path(out_dir)/"perturb_decoder.pt"); Path(out_dir,"metrics.json").write_text(json.dumps(rows,indent=2)); Path(out_dir,"basal_metrics.json").write_text(json.dumps({"dependency_validation_huber":dependency_val,"perturbseq_validation_score":perturb_val,**block_scores},indent=2)); return rows


def resume_perturb_evaluate(data_dir,model_path,decoder_path,out_path,d=384,latent=128,layers=6):
    device="cuda"; pack=np.load(Path(data_dir)/"features_spectral_safe.npz"); state=pack["state"].astype("float32"); sd=torch.load(model_path,map_location="cpu",weights_only=True); model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1]).to(device); model.load_state_dict(sd); ds=torch.load(decoder_path,map_location="cpu",weights_only=True); decoder=nn.Linear(latent,ds["weight"].shape[0]).to(device); decoder.load_state_dict(ds); rows=evaluate(model,state,pack,np.load(Path(data_dir)/"splits.npz"),(3,),device,perturb_decoder=decoder); Path(out_path).write_text(json.dumps(rows,indent=2)); return rows


def resume(data_dir,model_path,out_path,d=384,latent=128,layers=6):
    device="cuda"; pack=np.load(Path(data_dir)/"features.npz"); state=pack["state"].astype("float32"); split=np.load(Path(data_dir)/"splits.npz")
    sd=torch.load(model_path,map_location="cpu",weights_only=True); state_dim=sum(v.shape[1] for k,v in sd.items() if k.startswith("proj.") and k.endswith(".weight")); model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state_dim).to(device); model.load_state_dict(sd)
    rows=evaluate(model,state,pack,split,(3,),device,True); Path(out_path).write_text(json.dumps(rows,indent=2)); return rows


def resume_metric(data_dir,model_path,out_path,d=384,latent=128,layers=6,feature_name="features.npz"):
    device="cuda"; pack=np.load(Path(data_dir)/feature_name); state=pack["state"].astype("float32"); split=np.load(Path(data_dir)/"splits.npz"); sd=torch.load(model_path,map_location="cpu",weights_only=True)
    model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1]).to(device); model.load_state_dict(sd); genes=np.column_stack((encode_genes(model,state,device),state)); rows=[]
    for fold in range(5):
        trp,trn=split[f"cv3_pos_train_{fold}"],split[f"cv3_neg_train_{fold}"]; tep,ten=split[f"cv3_pos_test_{fold}"],split[f"cv3_neg_test_{fold}"]; tr=np.concatenate((trp,trn)); te=np.concatenate((tep,ten)); ytr=np.r_[np.ones(len(trp)),np.zeros(len(trn))]; y=np.r_[np.ones(len(tep)),np.zeros(len(ten))]
        score=fit_metric(genes,tr,ytr,te,device); row={"cv":3,"fold":fold,"auroc":roc_auc_score(y,score),"aupr":average_precision_score(y,score),"f1":f1_score(y,score>=.5)}; rows.append(row); print(json.dumps(row),flush=True)
    Path(out_path).write_text(json.dumps(rows,indent=2)); return rows


def resume_pair(data_dir,model_path,out_path,d=384,latent=128,layers=6,feature_name="features.npz"):
    device="cuda"; pack=np.load(Path(data_dir)/feature_name); state=pack["state"].astype("float32"); split=np.load(Path(data_dir)/"splits.npz"); sd=torch.load(model_path,map_location="cpu",weights_only=True); model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1]).to(device); model.load_state_dict(sd); genes=np.column_stack((encode_genes(model,state,device),state)); rows=[]
    for fold in range(5):
        torch.manual_seed(SEED+fold); trp,trn=split[f"cv3_pos_train_{fold}"],split[f"cv3_neg_train_{fold}"]; tep,ten=split[f"cv3_pos_test_{fold}"],split[f"cv3_neg_test_{fold}"]; tr=np.concatenate((trp,trn)); te=np.concatenate((tep,ten)); ytr=np.r_[np.ones(len(trp)),np.zeros(len(trn))]; y=np.r_[np.ones(len(tep)),np.zeros(len(ten))]; score=fit_symmetric(genes,tr,ytr,te,device); y=np.r_[np.ones(len(tep)),np.zeros(len(ten))]; row={"cv":3,"fold":fold,"auroc":roc_auc_score(y,score),"aupr":average_precision_score(y,score),"f1":f1_score(y,score>=.5)}; rows.append(row); print(json.dumps(row),flush=True)
    Path(out_path).write_text(json.dumps(rows,indent=2)); return rows


def resume_emergent(data_dir,model_path,out_path,d=384,latent=128,layers=6,feature_name="features.npz"):
    device="cuda"; pack=np.load(Path(data_dir)/feature_name); state=pack["state"].astype("float32"); split=np.load(Path(data_dir)/"splits.npz"); sd=torch.load(model_path,map_location="cpu",weights_only=True); model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1]).to(device); model.load_state_dict(sd); genes=encode_genes(model,state,device); rows=[]; predictions={}
    for fold in range(5):
        tep,ten=split[f"cv3_pos_test_{fold}"],split[f"cv3_neg_test_{fold}"]; te=np.concatenate((tep,ten)); y=np.r_[np.ones(len(tep)),np.zeros(len(ten))]; scores=outcome_scores(model,genes,te,device,sd["cell.weight"].shape[0]); row={"cv":3,"fold":fold}
        for name,score in scores.items(): row[name]={"auroc":roc_auc_score(y,score),"aupr":average_precision_score(y,score)}; predictions[f"fold{fold}_{name}"]=score.astype("float32")
        predictions.update({f"fold{fold}_pairs":te,f"fold{fold}_label":y.astype("int8")})
        rows.append(row); print(json.dumps(row),flush=True)
    Path(out_path).write_text(json.dumps(rows,indent=2)); np.savez_compressed(Path(out_path).with_suffix(".npz"),**predictions); return rows


def resume_transfer(data_dir,model_path,out_path,d=384,latent=128,layers=6,feature_name="features.npz"):
    device="cuda"; pack=np.load(Path(data_dir)/feature_name); state=pack["state"].astype("float32"); split=np.load(Path(data_dir)/"splits.npz"); ext=np.load(Path(data_dir)/"slkb_labels.npz"); sd=torch.load(model_path,map_location="cpu",weights_only=True)
    model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1]).to(device); model.load_state_dict(sd); ep=ext["pairs"]; ez=np.column_stack((embed_pairs(model,state,ep,device),pair_summary(state,ep))); rows=[]; rng=np.random.default_rng(SEED)
    for fold in range(5):
        trp,trn=split[f"cv3_pos_train_{fold}"],split[f"cv3_neg_train_{fold}"]; tep,ten=split[f"cv3_pos_test_{fold}"],split[f"cv3_neg_test_{fold}"]; tr=np.concatenate((trp,trn)); te=np.concatenate((tep,ten)); ytr=np.r_[np.ones(len(trp)),np.zeros(len(trn))]; y=np.r_[np.ones(len(tep)),np.zeros(len(ten))]
        unseen=np.zeros(len(state),bool); unseen[np.unique(te)]=True; keep=~(unseen[ep[:,0]]|unseen[ep[:,1]]); pos=np.flatnonzero(keep&(ext["label"]==1)); neg=np.flatnonzero(keep&(ext["label"]==0)); neg=rng.choice(neg,min(len(neg),5*len(pos)),False); ix=np.r_[pos,neg]
        xtr=np.column_stack((embed_pairs(model,state,tr,device),pair_summary(state,tr))); xte=np.column_stack((embed_pairs(model,state,te,device),pair_summary(state,te))); score=fit_head(np.concatenate((xtr,ez[ix])),np.r_[ytr,ext["label"][ix]],xte,device)
        row={"cv":3,"fold":fold,"external_pairs":len(ix),"auroc":roc_auc_score(y,score),"aupr":average_precision_score(y,score),"f1":f1_score(y,score>=.5)}; rows.append(row); print(json.dumps(row),flush=True)
    Path(out_path).write_text(json.dumps(rows,indent=2)); return rows


def resume_tabular(data_dir,model_path,out_path,d=384,latent=128,layers=6,feature_name="features.npz"):
    from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
    from lightgbm import LGBMClassifier
    device="cuda"; pack=np.load(Path(data_dir)/feature_name); state=pack["state"].astype("float32"); split=np.load(Path(data_dir)/"splits.npz"); sd=torch.load(model_path,map_location="cpu",weights_only=True)
    model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1]).to(device); model.load_state_dict(sd); rows=[]; predictions={}
    for fold in range(5):
        trp,trn=split[f"cv3_pos_train_{fold}"],split[f"cv3_neg_train_{fold}"]; tep,ten=split[f"cv3_pos_test_{fold}"],split[f"cv3_neg_test_{fold}"]; tr=np.concatenate((trp,trn)); te=np.concatenate((tep,ten)); ytr=np.r_[np.ones(len(trp)),np.zeros(len(trn))]; y=np.r_[np.ones(len(tep)),np.zeros(len(ten))]
        xtr=np.column_stack((embed_pairs(model,state,tr,device),pair_summary(state,tr),observed_relations(pack,tr,len(state)))); xte=np.column_stack((embed_pairs(model,state,te,device),pair_summary(state,te),observed_relations(pack,te,len(state))))
        forest=ExtraTreesClassifier(n_estimators=256,min_samples_leaf=3,max_features=.5,class_weight="balanced",n_jobs=-1,random_state=SEED).fit(xtr,ytr).predict_proba(xte)[:,1]
        boost=HistGradientBoostingClassifier(max_iter=150,learning_rate=.05,l2_regularization=1,max_leaf_nodes=31,random_state=SEED).fit(xtr,ytr).predict_proba(xte)[:,1]; lgbm=LGBMClassifier(n_estimators=500,num_leaves=31,learning_rate=.03,colsample_bytree=.7,reg_lambda=1,min_child_samples=30,n_jobs=8,verbosity=-1,random_state=SEED).fit(xtr,ytr).predict_proba(xte)[:,1]; mlps=[]
        for seed in range(3): torch.manual_seed(SEED+10*fold+seed); mlps.append(fit_head(xtr,ytr,xte,device))
        mlp=np.mean(mlps,0); row={"cv":3,"fold":fold}
        ensemble=(forest+lgbm+mlp)/3
        for name,score in (("forest",forest),("boost",boost),("lgbm",lgbm),("ensemble",ensemble)):
            row[name]={"auroc":roc_auc_score(y,score),"aupr":average_precision_score(y,score),"f1":f1_score(y,score>=.5)}
        predictions.update({f"fold{fold}_pairs":te,f"fold{fold}_label":y.astype("int8"),f"fold{fold}_score":ensemble.astype("float32")}); rows.append(row); print(json.dumps(row),flush=True)
    Path(out_path).write_text(json.dumps(rows,indent=2)); np.savez_compressed(Path(out_path).with_suffix(".npz"),**predictions); return rows


def load_graphs(path,device):
    p=np.load(path); n=int(p["n"]); out=[]
    for name in p["names"]:
        indptr=p[f"{name}_indptr"]; row=np.repeat(np.arange(n,dtype="int64"),np.diff(indptr)); col=p[f"{name}_indices"].astype("int64"); ix=torch.as_tensor(np.stack((row,col)),device=device); val=torch.as_tensor(p[f"{name}_data"],device=device); out.append(torch.sparse_coo_tensor(ix,val,(n,n),device=device).coalesce())
    return out


def resume_graph(data_dir,model_path,out_path,d=384,latent=128,layers=6,feature_name="features.npz",epochs=120):
    from scipy import sparse
    device="cuda"; pack=np.load(Path(data_dir)/feature_name); state=pack["state"].astype("float32"); split=np.load(Path(data_dir)/"splits.npz"); sd=torch.load(model_path,map_location="cpu",weights_only=True); model=SLPredict(d,latent,layers,sd["cell.weight"].shape[0],sd["outcome.weight"].shape[0],state.shape[1]).to(device); model.load_state_dict(sd)
    genes=np.column_stack((encode_genes(model,state,device),state[:,:1624])); graphs=load_graphs(Path(data_dir)/"relation_graphs.npz",device); rows=[]
    for fold in range(5):
        trp,trn=split[f"cv3_pos_train_{fold}"],split[f"cv3_neg_train_{fold}"]; tep,ten=split[f"cv3_pos_test_{fold}"],split[f"cv3_neg_test_{fold}"]; tr=np.concatenate((trp,trn)).astype("int64"); te=np.concatenate((tep,ten)).astype("int64"); ytr=np.r_[np.ones(len(trp)),np.zeros(len(trn))]; y=np.r_[np.ones(len(tep)),np.zeros(len(ten))]
        edge=np.concatenate((trp,trp[:,::-1])); a=sparse.csr_matrix((np.ones(len(edge),"float32"),(edge[:,0],edge[:,1])),shape=(len(state),len(state))); a=sparse.diags(1/np.maximum(1,np.asarray(a.sum(1)).ravel()))@a; x=torch.as_tensor(np.column_stack((genes,a@genes)),dtype=torch.float32,device=device); p=torch.as_tensor(tr,device=device); yt=torch.as_tensor(ytr,dtype=torch.float32,device=device); head=GraphHead(x.shape[1],len(graphs)).to(device); opt=torch.optim.AdamW(head.parameters(),1e-3,weight_decay=1e-3)
        for _ in range(epochs):
            head.train(); z=head(x,graphs); loss=nn.functional.binary_cross_entropy_with_logits(head.score(z,p),yt); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(head.parameters(),2.); opt.step()
        head.eval()
        with torch.no_grad(): z=head(x,graphs); score=head.score(z,torch.as_tensor(te,device=device)).sigmoid().cpu().numpy()
        row={"cv":3,"fold":fold,"auroc":roc_auc_score(y,score),"aupr":average_precision_score(y,score),"f1":f1_score(y,score>=.5)}; rows.append(row); print(json.dumps(row),flush=True)
    Path(out_path).write_text(json.dumps(rows,indent=2)); return rows


def fit_gse337988_set_composer(data_dir,endpoint_path,set_path,out_dir,decoder_epochs=20,composer_epochs=30,d=384,latent=128,layers=6,condition_means=False):
    torch.manual_seed(731); np.random.seed(731); device="cuda" if torch.cuda.is_available() else "cpu"; root=Path(data_dir); feature=np.load(root/"features_spectral_safe.npz"); state=feature["state"].astype("float32"); data=np.load(set_path); dep=np.load(root/"depmap_world.npz"); context=torch.as_tensor(dep["cell_state"][np.flatnonzero(dep["model_ids"].astype(str)=="ACH-001061")[0]],device=device); endpoint=load_residual_endpoint(endpoint_path,state.shape[1],device,d,latent,layers); world=endpoint.world; genes=torch.as_tensor(encode_genes(world,state,device),device=device); members=data["members"].astype("int64"); card=data["cardinality"]; role=data["role"]; source=data["source"]; target=data["target"].astype("float32").copy()
    @torch.no_grad()
    def latents(exact):
        out=[]
        for at in batches(len(members),1024,False):
            ix=at.numpy(); p=torch.as_tensor(members[ix],device=device); mask=p>=0; actions=genes[p.clamp_min(0)]; cs=context[None].expand(len(ix),-1) if exact else None; out.append(world.transition_set(actions,mask,context_state=cs)[0].cpu())
        return torch.cat(out)
    unknown,exact=latents(False),latents(True); test=np.flatnonzero((role==1)&(card==2))[:8]; p=torch.as_tensor(members[test,:2],device=device); equality=float((world.transition(genes[p[:,0]],genes[p[:,1]])[0]-unknown[test].to(device)).abs().max()); assert equality<1e-4
    decoder=nn.Sequential(nn.LayerNorm(latent),nn.Linear(latent,latent),nn.GELU(),nn.Linear(latent,target.shape[1])).to(device); train_single=np.flatnonzero((role==0)&(card==1)); valid_single=np.flatnonzero((role==1)&(card==1)); counts=np.bincount(source[train_single]); weight=1/counts[source[train_single]]; weight/=weight.sum(); opt=torch.optim.AdamW(decoder.parameters(),3e-4,weight_decay=1e-3); decoder_history=[]; decoder_saved=[]
    @torch.no_grad()
    def macro(rows,z,composition=None):
        losses=[]; sources=[]
        for at in batches(len(rows),2048,False):
            ix=rows[at.numpy()]; joint=z[ix].to(device)
            if composition is not None:
                p=torch.as_tensor(members[ix],device=device); mask=p>=0; joint=joint+composition(genes[p.clamp_min(0)],mask,joint)
            losses.append(nn.functional.huber_loss(decoder(joint),torch.as_tensor(target[ix],device=device),reduction="none").mean(1).cpu().numpy()); sources.append(source[ix])
        loss=np.concatenate(losses); src=np.concatenate(sources); return float(np.mean([loss[src==s].mean() for s in np.unique(src)]))
    def decoder_validation(epoch):
        decoder.eval(); u=macro(valid_single,unknown); e=macro(valid_single,exact); row={"epoch":epoch,"unknown_huber":u,"exact_huber":e,"selection_loss":.5*(u+e)}; decoder_history.append(row); decoder_saved.append({k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()}); print(json.dumps({"phase":"gse337988_decoder",**row}),flush=True)
    for epoch in range(decoder_epochs):
        decoder.train(); chosen=np.random.default_rng(731+epoch).choice(train_single,len(train_single),replace=True,p=weight)
        for at in batches(len(chosen),1024,False):
            ix=chosen[at.numpy()]; drop=torch.rand(len(ix))<.5; z=torch.where(drop[:,None],unknown[ix],exact[ix]).to(device); loss=nn.functional.huber_loss(decoder(z),torch.as_tensor(target[ix],device=device)); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(decoder.parameters(),1.); opt.step()
        decoder_validation(epoch+1)
    selected_decoder=min(range(len(decoder_history)),key=lambda i:decoder_history[i]["selection_loss"]); decoder.load_state_dict(decoder_saved[selected_decoder]); decoder.eval().requires_grad_(False); composer=MultiActionComposition(latent).to(device)
    def condition_rows(rows):
        groups={}
        for i in rows:groups.setdefault((int(source[i]),*members[i,:2]),[]).append(i)
        out=[]
        for values in groups.values():target[values[0]]=target[values].mean(0); out.append(values[0])
        return np.asarray(out,"int64")
    train_pair=np.flatnonzero((role==0)&(card==2)); valid_pair=np.flatnonzero((role==1)&(card==2))
    if condition_means: train_pair=condition_rows(train_pair); valid_pair=condition_rows(valid_pair); train_set=np.r_[train_pair,np.flatnonzero((role==0)&(card>=3))]
    else:train_set=np.flatnonzero((role==0)&(card>=2))
    group=source[train_set]*9+card[train_set]; group_count=np.bincount(group); weight=1/group_count[group]; weight/=weight.sum(); opt=torch.optim.AdamW(composer.parameters(),1e-4,weight_decay=1e-2); history=[]; saved=[]
    def composition_validation(epoch):
        composer.eval(); u=macro(valid_pair,unknown,composer); e=macro(valid_pair,exact,composer); row={"epoch":epoch,"unknown_pair_huber":u,"exact_pair_huber":e,"selection_loss":.5*(u+e)}; history.append(row); saved.append({k:v.detach().cpu().clone() for k,v in composer.state_dict().items()}); print(json.dumps({"phase":"gse337988_set_composition",**row}),flush=True)
    composition_validation(0)
    for epoch in range(composer_epochs):
        composer.train(); chosen=np.random.default_rng(831+epoch).choice(train_set,len(train_set),replace=True,p=weight)
        for at in batches(len(chosen),512,False):
            ix=chosen[at.numpy()]; p=torch.as_tensor(members[ix],device=device); mask=p>=0; actions=genes[p.clamp_min(0)]; drop=torch.rand(len(ix))<.5; joint=torch.where(drop[:,None],unknown[ix],exact[ix]).to(device); pred=decoder(joint+composer(actions,mask,joint)); loss=nn.functional.huber_loss(pred,torch.as_tensor(target[ix],device=device)); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(composer.parameters(),1.); opt.step()
        composition_validation(epoch+1)
    selected=min(range(len(history)),key=lambda i:history[i]["selection_loss"]); composer.load_state_dict(saved[selected]); composer.eval(); baseline=history[0]; chosen=history[selected]; old=np.load(root/"perturbseq_world_v4_dixit_context.npz"); old_rows=np.flatnonzero((old["role"]==1)&(old["cardinality"]==2)); decode=lambda z:torch.cat((endpoint.legacy_decoder(z),endpoint.residual_decoder(z)),1)
    @torch.no_grad()
    def retained(exact_context,adapted):
        losses=[]; sources=[]
        for at in batches(len(old_rows),1024,False):
            ix=old_rows[at.numpy()]; pair=torch.as_tensor(old["pairs"][ix].astype("int64"),device=device); actions=genes[pair]; cs=torch.as_tensor(old["context_state"][old["source"][ix]],device=device) if exact_context else None; joint=world.transition(actions[:,0],actions[:,1],context_state=cs)[0]; joint=joint+composer(actions,torch.ones(actions.shape[:2],device=device,dtype=torch.bool),joint) if adapted else joint; losses.append(nn.functional.huber_loss(decode(joint),torch.as_tensor(old["target"][ix],device=device),reduction="none").mean(1).cpu().numpy()); sources.append(old["source"][ix])
        loss=np.concatenate(losses); src=np.concatenate(sources); return float(np.mean([loss[src==s].mean() for s in np.unique(src)]))
    @torch.no_grad()
    def relation(adapted):
        pairs=torch.as_tensor(feature["pairs"].astype("int64"),device=device); truth=torch.as_tensor(feature["relations"].astype("float32"),device=device); rows=torch.nonzero((pairs[:,0]*1000003+pairs[:,1])%20==0).squeeze(1); total=0.
        for at in batches(len(rows),2048,False):
            ix=rows[at]; p=pairs[ix]; actions=genes[p]; joint=world.transition(actions[:,0],actions[:,1])[0]; joint=joint+composer(actions,torch.ones(actions.shape[:2],device=device,dtype=torch.bool),joint) if adapted else joint; total+=nn.functional.huber_loss(world.relation_score(actions[:,0],actions[:,1],joint),truth[ix],reduction="sum").item()
        return total/(len(rows)*truth.shape[1])
    preservation={"unknown":{"baseline":retained(False,False),"candidate":retained(False,True)},"exact":{"baseline":retained(True,False),"candidate":retained(True,True)},"relation":{"baseline":relation(False),"candidate":relation(True)}}; preserved=all(x["candidate"]<=1.01*x["baseline"] for x in preservation.values()); improved=chosen["selection_loss"]<=.98*baseline["selection_loss"]; advanced=selected>0 and preserved and improved; out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); torch.save(decoder.state_dict(),out/"gse337988_decoder.pt"); torch.save(composer.state_dict(),out/"gse337988_set_composer.pt"); result={"condition_means":bool(condition_means),"decoder_parameters":sum(p.numel() for p in decoder.parameters()),"composer_parameters":sum(p.numel() for p in composer.parameters()),"transition_set_pair_max_error":equality,"training_examples":len(train_set),"validation_pair_examples":len(valid_pair),"training_cardinalities":{str(n):int((card[train_set]==n).sum()) for n in np.unique(card)},"decoder_selected":decoder_history[selected_decoder],"baseline":baseline,"selected":chosen,"preservation":preservation,"single_state_invariant":True,"preserved":bool(preserved),"improved":bool(improved),"advanced":bool(advanced),"decoder_history":decoder_history,"history":history}; (out/"metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k not in ("decoder_history","history")},indent=2),flush=True); return result
