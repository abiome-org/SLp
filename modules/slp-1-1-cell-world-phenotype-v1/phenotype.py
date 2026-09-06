"""Continuous, context-queried single-intervention fitness observation model."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file,load_file


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


class FitnessObservation(nn.Module):
    def __init__(self,inputs=544,context=128,width=512,rank=64):
        super().__init__()
        self.gene=nn.Sequential(nn.Linear(inputs,width),nn.GELU(),nn.Dropout(.1),nn.Linear(width,192),nn.GELU(),nn.Linear(192,rank+1))
        self.context=nn.Sequential(nn.Linear(context,128),nn.GELU(),nn.Linear(128,rank+1))
        self.rank=rank

    def forward(self,world_state,context):
        g=self.gene(world_state);c=self.context(context)
        return g[:,:-1]@c[:,:-1].T/(self.rank**.5)+g[:,-1,None]+c[None,:,-1]


def loss(pred,target,mask):
    m=mask.float();res=(pred-target)*m
    mean=res.sum(1,keepdim=True)/m.sum(1,keepdim=True).clamp_min(1)
    return (res.square().sum()/m.sum().clamp_min(1))+4*((res-mean).square()*m).sum()/m.sum().clamp_min(1)


def matrix_scores(pred,target,mask,baseline):
    m=mask.astype(float);n=m.sum(1,keepdims=True).clip(1)
    def center(x):return x-(x*m).sum(1,keepdims=True)/n
    p=center(pred);t=center(target);b=center(np.broadcast_to(baseline,target.shape))
    pearson=(p*t*m).sum()/np.sqrt((p*p*m).sum()*(t*t*m).sum()).clip(1e-12)
    return {'mse':float(((pred-target)**2*m).sum()/m.sum()),'mean_baseline_mse':float(((baseline-target)**2*m).sum()/m.sum()),
        'gene_centered_mse':float(((p-t)**2*m).sum()/m.sum()),'gene_centered_baseline_mse':float(((b-t)**2*m).sum()/m.sum()),
        'gene_centered_correlation':float(pearson),'genes':len(pred),'cells':pred.shape[1]}


@torch.inference_mode()
def predict(model,x,c,batch=256):
    return torch.cat([model(x[i:i+batch],c).cpu() for i in range(0,len(x),batch)]).numpy()


def train(a):
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(731);rng=np.random.default_rng(731)
    with np.load(a.train) as z:tr={k:z[k] for k in z.files}
    with np.load(a.validation) as z:va={k:z[k] for k in z.files}
    if set(tr['gene_ids'])&set(va['gene_ids']):raise ValueError('fitness development genes overlap fitting')
    if not np.array_equal(tr['context_ids'],va['context_ids']):raise ValueError('development context panel differs')
    xm=tr['states'].mean(0);xs=tr['states'].std(0).clip(.01);cm=tr['contexts'].mean(0);cs=tr['contexts'].std(0).clip(.01)
    tensor=lambda x:torch.as_tensor(x,device=a.device)
    x=tensor(((tr['states']-xm)/xs).astype('f4'));v=tensor(((va['states']-xm)/xs).astype('f4'));c=tensor(((tr['contexts']-cm)/cs).astype('f4'))
    y=tensor(tr['targets']);mask=tensor(tr['known']);vy=tensor(va['targets']);vm=tensor(va['known'])
    model=FitnessObservation(inputs=x.shape[1],context=c.shape[1]).to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    best=float('inf');start=time.time();history=[];best_step=None
    for step in range(1,a.steps+1):
        model.train();gi=rng.integers(len(x),size=128);ci=rng.choice(len(c),128,replace=False)
        p=model(x[gi],c[ci]);objective=loss(p,y[gi][:,ci],mask[gi][:,ci])
        opt.zero_grad(set_to_none=True);objective.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%250==0 or step==a.steps:
            model.eval();pv=predict(model,v,c);value=float(loss(tensor(pv),vy,vm))
            row={'step':step,'fit_loss':float(objective.detach()),'validation_loss':value,'seconds':time.time()-start};history.append(row);print(json.dumps(row),flush=True)
            if value<best:
                best=value;best_step=step;save_file({k:t.detach().cpu().contiguous() for k,t in model.state_dict().items()},str(a.output/'model.safetensors'))
    model.load_state_dict(load_file(str(a.output/'model.safetensors')));model.eval();pv=predict(model,v,c)
    baseline=(tr['targets']*tr['known']).sum(0)/tr['known'].sum(0).clip(1)
    np.savez_compressed(a.output/'normalizer.npz',state_mean=xm,state_scale=xs,context_mean=cm,context_scale=cs,baseline=baseline)
    report={'source':'continuous single-intervention fitness','parameters':sum(t.numel() for t in model.parameters()),'steps':a.steps,'seed':731,
        'best_step':best_step,'selected_by':'molecular fitness development objective only; no genetic-pair labels','seconds':time.time()-start,
        'train_sha256':sha(a.train),'validation_sha256':sha(a.validation),'source_sha256':sha(__file__),'history':history,
        'validation':matrix_scores(pv,va['targets'],va['known'],baseline),'input_width':x.shape[1],'context_width':c.shape[1]}
    (a.output/'training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['validation']),flush=True)


class PhenotypeModel:
    def __init__(self,directory,device='cpu'):
        root=Path(directory);self.device=device;self.report=json.loads((root/'training.json').read_text())
        self.model=FitnessObservation(self.report['input_width'],self.report['context_width']).to(device).eval()
        self.model.load_state_dict(load_file(str(root/'model.safetensors'),device=device))
        with np.load(root/'normalizer.npz') as z:self.normalizer={k:z[k] for k in z.files}

    def predict(self,states,contexts):
        n=self.normalizer
        x=torch.tensor((states-n['state_mean'])/n['state_scale'],device=self.device,dtype=torch.float32)
        c=torch.tensor((contexts-n['context_mean'])/n['context_scale'],device=self.device,dtype=torch.float32)
        return predict(self.model,x,c)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True);p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=5000);p.add_argument('--device',default='cuda');train(p.parse_args())
