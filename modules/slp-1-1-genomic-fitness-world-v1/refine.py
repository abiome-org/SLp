"""Interaction-focused quantitative curriculum after basal fitness learning."""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file,save_file
from model import FitnessWorld,State
from train import load,tensor,expand,evaluate,sha,write


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(732);rng=np.random.default_rng(732)
    data=load(a.data);g=data['genes'];tr=data['human-train'];tp,ty=data['yeast-train']
    with np.load(a.initial/'normalizer.npz') as z:n={k:z[k] for k in z.files}
    xh=tensor(np.concatenate((g['human_raw'],(g['human_signatures']-n['signature_mean'])/n['signature_scale']),1).astype('f4'),a.device)
    xy=tensor(np.concatenate((g['yeast_raw'],(g['yeast_signatures']-n['signature_mean'])/n['signature_scale']),1).astype('f4'),a.device)
    ch=tensor(((tr['contexts']-n['context_mean'])/n['context_scale']).astype('f4'),a.device);cy=tensor(g['yeast_context'][None].astype('f4'),a.device)
    hp=tensor(tr['targets'],a.device);mask=tensor(tr['known'],a.device);yp=tensor(tp,a.device);yt=tensor(ty,a.device)
    model=FitnessWorld().to(a.device);model.load_state_dict(load_file(str(a.initial/'model.safetensors')))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01);history=[];start=time.time()
    def selection(report):return 100*report['yeast']['interaction_mse']+report['human']['mse']+4*report['human']['gene_centered_mse']
    report,_=evaluate(model,xh,xy,ch,cy,data,a.device);best=selection(report);beststep=0;initial_report=report
    shutil.copyfile(a.initial/'model.safetensors',a.output/'model.safetensors');shutil.copyfile(a.initial/'normalizer.npz',a.output/'normalizer.npz')
    for step in range(1,a.steps+1):
        model.train();sel=torch.randint(len(yp),(1024,),device=a.device);pairs=yp[sel];y=yt[sel]
        ga=model.action(xy[pairs[:,0]]);gb=model.action(xy[pairs[:,1]]);s=expand(model.encode(cy,1),len(pairs))
        sa,sb=model.observe(s,ga),model.observe(s,gb);ab,ba=model.conditional(s,ga,gb),model.conditional(s,gb,ga)
        residual=y[:,2]-y[:,0]-y[:,1];weight=1+(residual.abs()/.05).clamp(max=10)
        interaction=((ab-sb-residual).square()+(ba-sa-residual).square())*.5
        yl=100*(interaction*weight).sum()/weight.sum()+.1*(sa+ab-sb-ba).square().mean()
        yl=yl+.25*((sa-y[:,0]).square().mean()+(sb-y[:,1]).square().mean())
        gi=rng.integers(len(tr['gene_indices']),size=32);ci=rng.choice(len(ch),32,replace=False)
        hg=model.action(xh[tr['gene_indices'][gi]]);hs=model.encode(ch[ci],0)
        hs=State(hs.latent[None].expand(32,-1,-1).reshape(-1,256),hs.assay.repeat(32))
        pred=model.observe(hs,hg[:,None].expand(-1,32,-1).reshape(-1,256)).reshape(32,32)
        m=mask[gi][:,ci].float();err=(pred-hp[gi][:,ci])*m;den=m.sum().clamp_min(1)
        centered=err-err.sum(1,keepdim=True)/m.sum(1,keepdim=True).clamp_min(1)
        hl=err.square().sum()/den+4*(centered.square()*m).sum()/den
        rec=(model.reconstruct_gene(ga)-xy[pairs[:,0],:642]).square().mean()+(model.reconstruct_gene(hg)-xh[tr['gene_indices'][gi],:642]).square().mean()
        objective=yl+hl+.05*rec
        opt.zero_grad(set_to_none=True);objective.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%1000==0 or step==a.steps:
            report,_=evaluate(model,xh,xy,ch,cy,data,a.device);value=selection(report)
            row={'step':step,'seconds':time.time()-start,'fit_objective':float(objective.detach()),'curriculum_selection':value,**report};history.append(row);print(json.dumps(row),flush=True)
            if value<best:
                best=value;beststep=step;save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(a.output/'model.safetensors'))
        if time.time()-start>3000:break
    model.load_state_dict(load_file(str(a.output/'model.safetensors')));report,pred=evaluate(model,xh,xy,ch,cy,data,a.device)
    np.savez_compressed(a.output/'validation.npz',**pred)
    write(a.output/'training.json',{'schema':'slp.genomic-fitness-world/v1','phase':'interaction curriculum','seed':732,'steps':step,'selected_step':beststep,
        'parameters':sum(v.numel() for v in model.parameters()),'seconds':time.time()-start,'history':history,'validation':report,'initial_validation':initial_report,
        'selection':'100*yeast unweighted interaction MSE + human raw MSE +4*human centered MSE; includes initial checkpoint',
        'fit_weights':'continuous residual magnitude weighting 1+min(abs(log fAB-log fA-log fB)/.05,10), no binary labels',
        'initial_weights_sha256':sha(a.initial/'model.safetensors'),'initial_training_sha256':sha(a.initial/'training.json'),
        'corpus_sha256':sha(a.data/'manifest.json'),'source_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'weights_sha256':sha(a.output/'model.safetensors')})
    print(json.dumps({'selected_step':beststep,'validation':report}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('data','initial','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--device',default='cuda');main(p.parse_args())
