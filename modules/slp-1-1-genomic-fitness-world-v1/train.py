"""Joint quantitative human/yeast world training; no application labels."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file,load_file
from model import FitnessWorld,State
from prepare import DTYPE


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def tensor(x,device):return torch.as_tensor(x,device=device)
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def expand(state,n):return State(state.latent.expand(n,-1),state.assay.expand(n))


def yeast_arrays(path):
    z=np.fromfile(path,dtype=DTYPE)
    return np.stack((z['a'],z['b']),1).astype('i8'),np.log(np.stack((z['single_a'],z['single_b'],z['double']),1).clip(.05)).astype('f4')


def load(root):
    arrays={}
    for name in ('genes','human-train','human-validation'):
        with np.load(root/(name+'.npz')) as z:arrays[name]={k:z[k] for k in z.files}
    for role in ('train','validation'):arrays['yeast-'+role]=yeast_arrays(root/f'yeast-{role}.bin')
    # Whole-intervention disjointness, including query genes in double outcomes.
    if set(arrays['yeast-train'][0].ravel())&set(arrays['yeast-validation'][0].ravel()):raise ValueError('yeast outcome overlap')
    if set(arrays['human-train']['gene_indices'])&set(arrays['human-validation']['gene_indices']):raise ValueError('human outcome overlap')
    return arrays


def correlation(p,t):
    p=p-p.mean();t=t-t.mean();return float((p*t).sum()/np.sqrt((p*p).sum()*(t*t).sum()).clip(1e-12))


@torch.inference_mode()
def evaluate(model,xh,xy,ch,cy,data,device,batch=4096):
    model.eval();gh=model.action(xh);gy=model.action(xy);base=model.encode(cy,1)
    pairs,y=data['yeast-validation'];outputs=[]
    for start in range(0,len(pairs),batch):
        p=pairs[start:start+batch];a,b=gy[p[:,0]],gy[p[:,1]];s=expand(base,len(p))
        sa,sb=model.observe(s,a),model.observe(s,b)
        ab,ba=model.conditional(s,a,b),model.conditional(s,b,a)
        outputs.append(torch.stack((sa,sb,(sa+ab+sb+ba)*.5,(ab-sb+ba-sa)*.5),1).cpu().numpy())
    p=np.concatenate(outputs);interaction=y[:,2]-y[:,0]-y[:,1]
    raw=float(np.mean((p[:,2]-y[:,2])**2));imse=float(np.mean((p[:,3]-interaction)**2))
    fit_y=data['yeast-train'][1];mean=float(fit_y[:,2].mean());imean=float((fit_y[:,2]-fit_y[:,0]-fit_y[:,1]).mean())
    report={'yeast':{'n':len(y),'double_log_fitness_mse':raw,'fitting_mean_mse':float(np.mean((y[:,2]-mean)**2)),
        'measured_single_additive_mse':float(np.mean(interaction**2)),'interaction_mse':imse,
        'interaction_mean_mse':float(np.mean((interaction-imean)**2)),'interaction_correlation':correlation(p[:,3],interaction),
        'double_log_fitness_correlation':correlation(p[:,2],y[:,2])}}
    h=data['human-validation'];hi=h['gene_indices'];out=[]
    cs=model.encode(ch,0)
    for gi in hi:
        out.append(model.observe(cs,gh[int(gi)].expand(len(ch),-1)).cpu().numpy())
    hp=np.stack(out);hy=h['targets'];m=h['known'];n=m.sum(1,keepdims=True).clip(1)
    err=(hp-hy)*m;center=err-err.sum(1,keepdims=True)/n
    hmse=float((err**2).sum()/m.sum());hc=float((center**2*m).sum()/m.sum())
    tr=data['human-train'];baseline=(tr['targets']*tr['known']).sum(0)/tr['known'].sum(0).clip(1)
    report['human']={'genes':len(hi),'contexts':len(ch),'mse':hmse,'gene_centered_mse':hc,
        'fitting_context_mean_mse':float((((baseline-hy)**2)*m).sum()/m.sum())}
    # Fixed balanced objective, species-native units. No test-dependent weights.
    report['selection_objective']=raw+4*imse+.3*(hmse+4*hc)
    return report,{'yeast_predicted':p,'yeast_observed':y,'yeast_pairs':pairs,'human_predicted':hp}


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(731);rng=np.random.default_rng(731)
    data=load(a.data);g=data['genes'];tr=data['human-train'];tp,ty=data['yeast-train']
    hm=tr['contexts'].mean(0);hs=tr['contexts'].std(0).clip(.01)
    fitting_yeast=np.unique(tp);fitting_human=tr['gene_indices']
    sigfit=np.concatenate((g['human_signatures'][fitting_human],g['yeast_signatures'][fitting_yeast]))
    sm=sigfit.mean(0);ss=sigfit.std(0).clip(.01)
    xh=tensor(np.concatenate((g['human_raw'],(g['human_signatures']-sm)/ss),1).astype('f4'),a.device)
    xy=tensor(np.concatenate((g['yeast_raw'],(g['yeast_signatures']-sm)/ss),1).astype('f4'),a.device)
    ch=tensor(((tr['contexts']-hm)/hs).astype('f4'),a.device);cy=tensor(g['yeast_context'][None].astype('f4'),a.device)
    hp=tensor(tr['targets'],a.device);mask=tensor(tr['known'],a.device);yp=tensor(tp,a.device);yt=tensor(ty,a.device)
    architecture=FitnessWorld
    if a.architecture=='capacity':
        from capacity_model import FitnessWorld as architecture
    model=architecture().to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    np.savez_compressed(a.output/'normalizer.npz',signature_mean=sm,signature_scale=ss,context_mean=hm,context_scale=hs,
        feature_mean=g['feature_mean'],feature_scale=g['feature_scale'])
    start=time.time();best=float('inf');history=[];beststep=None
    for step in range(1,a.steps+1):
        model.train();sel=torch.randint(len(yp),(a.batch,),device=a.device);pairs=yp[sel];y=yt[sel]
        ga=model.action(xy[pairs[:,0]]);gb=model.action(xy[pairs[:,1]]);state=expand(model.encode(cy,1),len(pairs))
        sa,sb=model.observe(state,ga),model.observe(state,gb)
        ab,ba=model.conditional(state,ga,gb),model.conditional(state,gb,ga)
        residual=y[:,2]-y[:,0]-y[:,1]
        yl=.25*((sa-y[:,0]).square().mean()+(sb-y[:,1]).square().mean())
        yl=yl+.5*((ab-(y[:,2]-y[:,0])).square().mean()+(ba-(y[:,2]-y[:,1])).square().mean())
        yl=yl+a.interaction_weight*.5*((ab-sb-residual).square().mean()+(ba-sa-residual).square().mean())+.1*(sa+ab-sb-ba).square().mean()
        gi=rng.integers(len(fitting_human),size=32);ci=rng.choice(len(ch),32,replace=False)
        hgenes=model.action(xh[fitting_human[gi]]);hstate=model.encode(ch[ci],0)
        hstate=State(hstate.latent[None].expand(32,-1,-1).reshape(-1,256),hstate.assay.repeat(32))
        pred=model.observe(hstate,hgenes[:,None].expand(-1,32,-1).reshape(-1,256)).reshape(32,32)
        target=hp[gi][:,ci];m=mask[gi][:,ci].float();err=(pred-target)*m;den=m.sum().clamp_min(1)
        centered=err-err.sum(1,keepdim=True)/m.sum(1,keepdim=True).clamp_min(1)
        hl=err.square().sum()/den+4*(centered.square()*m).sum()/den
        rec=(model.reconstruct_gene(ga)-xy[pairs[:,0],:642]).square().mean()+(model.reconstruct_gene(hgenes)-xh[fitting_human[gi],:642]).square().mean()
        objective=yl+.3*hl+.05*rec
        opt.zero_grad(set_to_none=True);objective.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%a.evaluate_every==0 or step==a.steps:
            report,_=evaluate(model,xh,xy,ch,cy,data,a.device)
            selection=report['yeast']['double_log_fitness_mse']+a.interaction_weight*report['yeast']['interaction_mse']+.3*(report['human']['mse']+4*report['human']['gene_centered_mse'])
            row={'step':step,'seconds':time.time()-start,'fit_objective':float(objective.detach()),'checkpoint_selection_objective':selection,**report};history.append(row);print(json.dumps(row),flush=True)
            if selection<best:
                best=selection;beststep=step
                save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(a.output/'model.safetensors'))
        if time.time()-start>a.max_seconds:break
    model.load_state_dict(load_file(str(a.output/'model.safetensors')));report,pred=evaluate(model,xh,xy,ch,cy,data,a.device)
    np.savez_compressed(a.output/'validation.npz',**pred)
    write(a.output/'training.json',{'schema':'slp.genomic-fitness-world/v1','architecture':a.architecture,'interaction_weight':a.interaction_weight,'seed':731,'steps':step,'selected_step':beststep,
        'parameters':sum(v.numel() for v in model.parameters()),'seconds':time.time()-start,'batch':a.batch,'history':history,'validation':report,
        'selection':f'minimum fixed yeast raw+{a.interaction_weight}interaction MSE + .3(human raw+4gene-centered MSE), development outcomes only',
        'endpoint':'human raw CRISPR gene effect; yeast natural-log relative fitness with fixed .05 floor',
        'corpus_sha256':sha(a.data/'manifest.json'),'source_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        'weights_sha256':sha(a.output/'model.safetensors')})
    print(json.dumps({'selected_step':beststep,'validation':report}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=20000);p.add_argument('--batch',type=int,default=1024);p.add_argument('--evaluate-every',type=int,default=1000)
    p.add_argument('--max-seconds',type=int,default=3000);p.add_argument('--device',default='cuda')
    p.add_argument('--architecture',choices=['conditional','capacity'],default='conditional');p.add_argument('--interaction-weight',type=float,default=4.)
    main(p.parse_args())
