"""Freeze quantitative fitness forecasts and evaluate held gene/cell observations."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from phenotype import FitnessObservation,PhenotypeModel,sha,matrix_scores


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    model=PhenotypeModel(a.model,'cuda')
    if sha(a.train)!=model.report['train_sha256']:raise ValueError('wrong fitting reference')
    with np.load(a.train) as z:
        fit_states=z['states'];training_mean=float((z['targets']*z['known']).sum()/z['known'].sum())
    with np.load(a.requests) as z:gene_ids=z['gene_ids'];states=z['states'];contexts=z['contexts'];context_ids=z['context_ids'];selected=z['reference_cells']
    with np.load(a.held) as z:held={k:z[k] for k in z.files}
    results={}
    for arm in ('trained','untrained'):
        if arm=='untrained':
            torch.manual_seed(1731);model.model=FitnessObservation(model.report['input_width'],model.report['context_width']).to('cuda').eval()
        prediction=model.predict(states,contexts[selected]);reference=model.predict(fit_states,contexts[selected]).mean(0)
        profile=prediction-reference;profile-=profile.mean(1,keepdims=True)
        np.savez_compressed(a.output/f'{arm}.npz',gene_ids=gene_ids,context_ids=context_ids[selected],prediction=prediction,
                            reference_profile=reference,profile=profile)
        hp=model.predict(held['states'],held['contexts']);hb=np.full(len(held['contexts']),training_mean,np.float32)
        results[arm]=matrix_scores(hp,held['targets'],held['known'],hb)
        if arm=='trained':np.savez_compressed(a.output/'query-bank.npz',contexts=contexts[selected],context_ids=context_ids[selected],reference_profile=reference)
    report={'model_sha256':sha(a.model/'model.safetensors'),'source_sha256':sha(__file__),'held_gene_and_cell':results,
        'profile_rule':'predict continuous single-intervention fitness; subtract fitting-gene mean prediction per cell, then center each gene across fixed non-target cell contexts',
        'files':{p.name:sha(p) for p in a.output.glob('*.npz')},'inputs':{str(p):sha(p) for p in (a.train,a.requests,a.held)},
        'genetic_pair_labels_used':False}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(results,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('model','train','requests','held','output'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
