"""Artifact replay of molecular state, conditional fitness and SL decoding."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch


def main(a):
    sys.path.insert(0,str(a.bundle.resolve()));from functional_predict import SLpWorld
    torch.set_num_threads(4);model=SLpWorld(a.bundle,a.device)
    with np.load(a.request) as z:pairs=z['gene_pairs'].astype(str)
    result=model.predict_pairs(pairs);reverse=model.predict_pairs(pairs[:,::-1])
    symmetry={k:float(abs(v-reverse[k]).max()) for k,v in result.items()}
    errors={}
    if a.reference:
        with np.load(a.reference) as z:errors={k:float(abs(v-z[k]).max()) for k,v in result.items()}
    sim=model.functional_simulator;world=model.functional;s=world.expand(sim.states[0],4);actions=sim.actions[:4];query=sim.actions[4:8]
    sa=world.intervene(s,actions);sab=world.intervene(sa,query);sba=world.intervene(world.intervene(s,query),actions)
    conditional=world.observe(sa,query);total=world.decode(sab,s)
    composition=float(abs(total-world.observe(s,actions)-conditional).max());order=float(abs(total-world.decode(sba,s)).max())
    # Replay arbitrary-gene route through actual RNA simulations, not registry IDs.
    with np.load(a.bundle/'genes.npz') as z:
        n=world.norm;raw=z['human_raw'][:4]*n['feature_scale']+n['feature_mean']
    regenerated=model.actions_from_descriptors(raw);action_error=float(abs(regenerated-actions).max())
    with np.load(a.bundle/'molecular_contexts/k562-context.npz') as z:
        composite=model.encode(z['basal'][None],z['basal'][None],z['query_descriptors'],fitness_context=sim.contexts[:1],
            modality=np.zeros(len(z['basal']),int),scale=z['scale'],assay=int(z['assay']),mechanism=int(z['mechanism']),
            control=z['control'][None],encoder_indices=z['encoder_indices'])
    unchanged=model.intervene(composite,raw[:2][None],np.zeros((1,2),bool));assert unchanged is composite
    combined=model.intervene(composite,raw[:2][None],np.ones((1,2),bool));decoded=model.decode(combined)
    joint_finite=bool(np.isfinite(decoded['molecular']['values']).all() and np.isfinite(decoded['fitness_effect']).all())
    threshold=lambda k:2e-4 if k=='features' else 1e-5
    passed=all(np.isfinite(v).all() for v in result.values()) and all(v<=threshold(k) for k,v in symmetry.items())
    passed=passed and all(v<=threshold(k) for k,v in errors.items()) and composition<1e-5 and order<1e-5 and action_error<2e-4 and joint_finite
    a.output.mkdir(parents=True,exist_ok=False);np.savez_compressed(a.output/'predictions.npz',**result)
    report={'passed':bool(passed),'pair_order_max_errors':symmetry,'reference_max_errors':errors,'fitness_path_composition_error':composition,
        'fitness_endpoint_order_error':order,'molecular_signature_action_replay_error':action_error,'joint_molecular_fitness_state_replay':joint_finite,
        'device':a.device,'pairs':len(pairs),'torch':torch.__version__,'numpy':np.__version__}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if not passed:raise AssertionError('functional world replay failed')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('bundle','request','reference','output'):p.add_argument('--'+name,type=Path,required=name!='reference')
    p.add_argument('--device',default='cpu');main(p.parse_args())
