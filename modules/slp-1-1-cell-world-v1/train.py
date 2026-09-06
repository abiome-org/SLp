"""Train the shared molecular state model on molecular observations only."""
from __future__ import annotations
import argparse,copy,json,math,os,random,shutil,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from safetensors.torch import save_file
from model import CellWorld,Config
from data import Corpus
from prepare import sha,write


def tensors(batch,device):
    return {k:torch.as_tensor(np.array(v,copy=True),device=device) if isinstance(v,np.ndarray) and v.dtype.kind in 'biuf' else v for k,v in batch.items()}


def context(b):return b['assay'],b['taxon'],b['mechanism']


def encode(model,b,values):
    e=b['encoder_indices']
    return model.encode(values[:,e],b['basal'][:,e],b['query'][e],b['modality'][e],b['observed'][:,e],*context(b),control=b['control'][:,e])


def decode(model,b,state):
    d=b['decoder_indices'];return model.decode(state,b['query'][d],b['modality'][d],*context(b),control=b['control'][:,d])


def balanced(error,b):
    d=b['decoder_indices'];valid=b['observed'][:,d];modality=b['modality'][d]
    losses=[]
    for m in (0,1):
        mask=valid&(modality[None]==m)
        if mask.any():losses.append(error[mask].float().mean())
    return torch.stack(losses).mean()


def objective(model,teacher,b,rng,step,warmup):
    d=b['decoder_indices'];ctx=context(b)
    target_state=encode(model,b,b['target'])
    reconstruct,logvar=decode(model,b,target_state)
    target_delta=b['target'][:,d]-b['basal'][:,d]
    reconstruction=balanced((reconstruct-target_delta).square(),b)
    variance=balanced(.5*((reconstruct.detach()-target_delta).square()*torch.exp(-logvar)+logvar),b)
    # Decoding withheld molecular coordinates grounds the latent in observations.
    loss=reconstruction+.05*variance
    metrics={'reconstruction':float(reconstruction.detach()),'variance':float(variance.detach())}
    if model.config.observation_likelihood and b['cell']:
        rna=(b['modality'][d]==0)[None]&b['observed'][:,d]
        present,mu,lv=model.observation_parameters(target_state,b['query'][d],b['modality'][d],*ctx,control=b['control'][:,d])
        native=b['target'][:,d]*b['scale'][d]
        detected=native>0
        detection=torch.nn.functional.binary_cross_entropy_with_logits(present[rna].float(),detected[rna].float())
        positive=rna&detected
        logvalue=native.clamp_min(1e-8).log()
        positive_nll=.5*((logvalue-mu).square()*torch.exp(-lv)+lv)
        expression=positive_nll[positive].float().mean() if positive.any() else positive_nll.sum()*0
        observation=detection+.25*expression
        loss=loss+.5*observation
        metrics['rna_detection']=float(detection.detach());metrics['rna_positive_nll']=float(expression.detach())
    if step<warmup:return loss,metrics
    initial=encode(model,b,b['initial'])
    with torch.no_grad():endpoint=encode(teacher,b,b['target'])
    if b['cell']:
        # Control and perturbed cells were not biologically paired. This creates
        # a minibatch transport coupling within one intervention and context.
        distance=torch.cdist(initial.detach().float().flatten(1),endpoint.float().flatten(1)).cpu().numpy()
        ii,jj=linear_sum_assignment(distance);order=np.empty(len(ii),np.int64);order[jj]=ii
        order=torch.as_tensor(order,device=initial.device);initial=initial[order]
        initial_values=b['initial'][order]
    else:initial_values=b['initial']
    mean=model.transition(initial,b['actions'],b['action_mask'],*ctx)
    base,_=decode(model,b,initial);forecast,_=decode(model,b,mean)
    predicted=initial_values[:,d]+forecast-base
    error=predicted-b['target'][:,d]
    if b['cell']:
        # Unpaired cellular endpoints supervise their distribution's mean.
        error=error.mean(0,keepdim=True).expand_as(error)
    molecular=balanced(error.square(),b)
    latent=(mean-endpoint).square().mean()
    loss=loss+2*molecular+.2*latent
    # Rectified flow learns a conditional distribution of molecular latent states.
    residual=(endpoint-mean).detach();noise=torch.randn_like(residual)*.3
    t=torch.rand(len(residual),device=residual.device)
    interpolated=(1-t[:,None,None])*noise+t[:,None,None]*residual
    velocity=model.flow_velocity(interpolated,t,initial.detach(),mean.detach(),b['actions'],b['action_mask'],*ctx)
    flow=(velocity-(residual-noise)).square().mean()
    loss=loss+.3*flow
    metrics.update(molecular=float(molecular.detach()),latent=float(latent.detach()),flow=float(flow.detach()))
    if 'parent_mask' in b and b['parent_mask'].any():
        mask=b['action_mask'];first=torch.zeros_like(mask);column=step%2 if model.config.observation_likelihood else 0;first[:,column]=mask[:,column];second=mask&~first
        parent_target=b['parent_target'] if column==0 else b['other_parent_target']
        intermediate=model.transition(initial,b['actions'],first,*ctx)
        composed=model.transition(intermediate,b['actions'],second,*ctx)
        composed_decoded,_=decode(model,b,composed)
        measured_parent=encode(model,b,parent_target)
        observed_next=model.transition(measured_parent,b['actions'],second,*ctx)
        measured_decoded,_=decode(model,b,measured_parent);observed_decoded,_=decode(model,b,observed_next)
        valid=b['parent_mask'][:,None]&b['observed'][:,d]
        chain=(initial_values[:,d]+composed_decoded-base-b['target'][:,d]).square()
        parent=(parent_target[:,d]+observed_decoded-measured_decoded-b['target'][:,d]).square()
        composition=(chain[valid].mean()+parent[valid].mean())*.5
        loss=loss+composition
        metrics['composition']=float(composition.detach())
    return loss,metrics


def save(output,model,teacher,optimizer,step,rng,metadata):
    folder=output/f'checkpoint-{step:06d}';folder.mkdir(exist_ok=False)
    save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(folder/'model.safetensors'))
    save_file({k:v.detach().cpu().contiguous() for k,v in teacher.state_dict().items()},str(folder/'teacher.safetensors'))
    torch.save({'optimizer':optimizer.state_dict(),'step':step,'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':rng.bit_generator.state},folder/'resume.pt')
    write(folder/'config.json',asdict(model.config));write(folder/'training.json',metadata)
    write(folder/'manifest.json',{'files':{p.name:sha(p) for p in folder.iterdir() if p.is_file()}})
    write(output/'latest.json',{'checkpoint':folder.name,'step':step})
    return folder


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--index',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=16000);p.add_argument('--seconds',type=int,default=3300)
    p.add_argument('--batch',type=int,default=16);p.add_argument('--encoder-queries',type=int,default=192);p.add_argument('--decoder-queries',type=int,default=256)
    p.add_argument('--warmup',type=int,default=1000);p.add_argument('--seed',type=int,default=731);p.add_argument('--resume',type=Path)
    p.add_argument('--initialize',type=Path)
    p.add_argument('--learning-rate',type=float,default=2e-4)
    p.add_argument('--stage',choices=('state','generative'),default='generative')
    p.add_argument('--schedule-steps',type=int)
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('this run explicitly requires native CUDA')
    if not 0<a.seconds<=3500:raise ValueError('local training must be bounded below one hour')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    rng=np.random.default_rng(a.seed);start=time.monotonic()
    corpus=Corpus(a.source_root,a.index);print(json.dumps({'event':'corpus_ready','seconds':time.monotonic()-start,'sources':corpus.report()}),flush=True)
    model=CellWorld(Config(observation_likelihood=a.stage=='generative')).cuda();teacher=copy.deepcopy(model).eval();teacher.requires_grad_(False)
    new_parameters=[v for k,v in model.named_parameters() if k.startswith(('observation_head.','query_control.'))]
    core_parameters=[v for k,v in model.named_parameters() if not k.startswith(('observation_head.','query_control.'))]
    optimizer=torch.optim.AdamW([{'params':core_parameters,'lr_multiplier':1.},
        {'params':new_parameters,'lr_multiplier':3. if a.initialize else 1.}],lr=a.learning_rate,weight_decay=.01);first=0
    if a.resume and a.initialize:raise ValueError('choose initialization or optimizer continuation')
    if a.initialize:
        from safetensors.torch import load_file
        result=model.load_state_dict(load_file(str(a.initialize/'model.safetensors')),strict=False)
        if result.unexpected_keys or any(not key.startswith(('observation_head.','query_control.')) for key in result.missing_keys):raise ValueError('incompatible initialization')
        teacher.load_state_dict(model.state_dict())
    if a.resume:
        from safetensors.torch import load_file
        model.load_state_dict(load_file(str(a.resume/'model.safetensors')));teacher.load_state_dict(load_file(str(a.resume/'teacher.safetensors')))
        state=torch.load(a.resume/'resume.pt',weights_only=False,map_location='cpu');optimizer.load_state_dict(state['optimizer']);first=int(state['step'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);rng.bit_generator.state=state['numpy_rng']
        # The within-shard cache is intentionally reconstructed on resume; all
        # resumed draws are recorded as a new continuation, not bitwise replay.
    module=Path(__file__).parent;captured=a.output/'source';captured.mkdir()
    for path in module.iterdir():
        if path.is_file():shutil.copy2(path,captured/path.name)
    write(a.output/'protocol.json',{'schema':'slp.cell-world-training/v1','arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        'config':asdict(model.config),'parameters':sum(p.numel() for p in model.parameters()),'index_sha256':sha(a.index/'manifest.json'),
        'source_hashes':{p.name:sha(p) for p in captured.iterdir()},'torch':torch.__version__,'numpy':np.__version__,
        'device':torch.cuda.get_device_name(),'executor':'native-windows-cuda','world_objectives':['masked molecular reconstruction','RNA to protein imputation',
        'intervention-conditioned molecular transition','latent endpoint prediction','conditional latent rectified flow','measured and latent combination composition'],
        'hypothesis':'A jointly learned molecular state supports reconstructing withheld coordinates and action-conditioned generation without fitted response priors.',
        'advancement_rule':'Retain the trained generative artifact and report matched molecular development baselines; do not infer emergence or SOTA from training loss.',
        'benchmark_labels':False,'fitted_response_prior':False,
        'initialization_sha256':sha(a.initialize/'model.safetensors') if a.initialize else None,
        'sparse_rna_observation_distribution':model.config.observation_likelihood})
    accum={};counts={};last=first;deadline=start+a.seconds
    log=(a.output/'training.jsonl').open('w')
    for step in range(first+1,a.steps+1):
        if time.monotonic()>deadline-45:break
        batch=tensors(corpus.draw(rng,a.batch,a.encoder_queries,a.decoder_queries),'cuda')
        optimizer.zero_grad(set_to_none=True)
        horizon=a.schedule_steps or a.steps
        lr=a.learning_rate*min(step/500,1.)*(.15+.85*.5*(1+math.cos(math.pi*min(step,horizon)/horizon)))
        for group in optimizer.param_groups:group['lr']=lr*group.get('lr_multiplier',1.)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=objective(model,teacher,batch,rng,step,a.warmup)
        if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss {step} {batch["name"]}')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError(f'nonfinite gradient {step}')
        optimizer.step()
        with torch.no_grad():
            for target,source in zip(teacher.parameters(),model.parameters()):target.lerp_(source,.005 if step>=a.warmup else .05)
        name=batch['name'];counts[name]=counts.get(name,0)+1
        for key,value in metrics.items():
            metric=name+'/'+key;accum.setdefault(metric,[]).append(value)
        last=step
        if step%100==0 or step==first+1:
            record={'step':step,'seconds':time.monotonic()-start,'loss':float(loss.detach()),'gradient_norm':float(norm),
                'lr':lr,'metrics':{k:float(np.mean(v)) for k,v in accum.items()},'draws':dict(counts),
                'peak_cuda_mib':torch.cuda.max_memory_allocated()/2**20}
            line=json.dumps(record);log.write(line+'\n');log.flush();print(line,flush=True);accum={}
        if step%4000==0 and step<a.steps:
            preview=a.output/f'preview-{step:06d}';preview.mkdir()
            save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(preview/'model.safetensors'))
            write(preview/'config.json',asdict(model.config))
    if last==first:raise RuntimeError('no training updates completed')
    metadata={'steps':last,'elapsed_seconds':time.monotonic()-start,'peak_cuda_mib':torch.cuda.max_memory_allocated()/2**20,
              'draws':counts,'sources':corpus.report(),'parameters':sum(p.numel() for p in model.parameters()),'status':'trained_research_candidate'}
    folder=save(a.output,model,teacher,optimizer,last,rng,metadata)
    write(a.output/'summary.json',dict(metadata,checkpoint=folder.name));log.close()
    print(json.dumps({'event':'complete','checkpoint':str(folder),**metadata}),flush=True)


if __name__=='__main__':main()
