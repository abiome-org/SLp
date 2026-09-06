"""Run a molecular request using only a standalone world-model bundle."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from inference import WorldModel


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--request',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path);p.add_argument('--device',default='cpu');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    with np.load(a.request,allow_pickle=False) as z:r={k:z[k] for k in z.files}
    world=WorldModel(a.bundle,a.device)
    state=world.encode(r['observed'],r['basal'],r['query_descriptors'],modality=r['modality'],scale=r['scale'],
        assay=int(r['assay']),taxon=int(r['taxon']),mechanism=int(r['mechanism']),encoder_indices=r['encoder_indices'],normalized_descriptors=True)
    changed=world.intervene(state,r['actions'],r['action_mask'],normalized_descriptors=True)
    output={'prediction':world.decode(changed)['values'],'reconstruction':world.reconstruct(state)}
    if world.model.config.observation_likelihood:
        for seed in (731,732):
            generated=world.generate(state,r['actions'],r['action_mask'],seed=seed,steps=16,normalized_descriptors=True)
            output[f'generated_{seed}']=generated['values']
    empty=world.intervene(state,r['actions'],np.zeros_like(r['action_mask']),normalized_descriptors=True)
    error=float(np.max(np.abs(world.decode(empty)['values']-r['observed'])))
    if error!=0:raise AssertionError('empty action failed exact observation identity')
    reverse=world.intervene(state,r['actions'][:,::-1].copy(),r['action_mask'][:,::-1].copy(),normalized_descriptors=True)
    permutation_error=float(np.max(np.abs(world.decode(reverse)['values']-output['prediction'])))
    if permutation_error>1e-5:raise AssertionError('action-set permutation changed prediction')
    replay_errors={}
    if a.reference:
        with np.load(a.reference,allow_pickle=False) as z:
            for key,value in output.items():
                replay_errors[key]=float(np.max(np.abs(value-z[key])))
                if replay_errors[key]>1e-5:raise AssertionError(f'cross-runtime replay drift: {key}')
    np.savez_compressed(a.output/'outputs.npz',**output)
    with (a.bundle/'model.safetensors').open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    report={'status':'passed','checkpoint_sha256':digest,'empty_action_max_error':error,'action_permutation_max_error':permutation_error,
        'reference_max_errors':replay_errors,'torch':torch.__version__,'numpy':np.__version__,'device':a.device,
        'sample_difference':float(np.max(np.abs(output['generated_731']-output['generated_732']))) if 'generated_731' in output else None}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    (a.output/'metrics.json').write_text(json.dumps({'empty_action_max_error':error,'action_permutation_max_error':permutation_error,
        'portability_max_error':max(replay_errors.values(),default=0.),'sample_difference':report['sample_difference']},indent=2)+'\n')


if __name__=='__main__':main()
