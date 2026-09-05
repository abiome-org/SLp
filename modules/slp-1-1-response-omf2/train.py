import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from response_model import fit, save

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--rank',type=int,required=True); p.add_argument('--alpha',type=float,required=True); a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False); contexts={}
    with threadpool_limits(2):
      for source in ('k562','rpe1'):
        with np.load(a.data/f'{source}.npz',allow_pickle=False) as d: arrays={k:np.asarray(d[k]) for k in d.files}
        rank=min(len(arrays['features'])-1,arrays['features'].shape[1]) if a.rank==0 else a.rank
        model=fit(arrays['features'],arrays['residual_targets'],rank=rank,alpha=a.alpha)
        name=f'model-{source}.npz'; save(a.output/name,model,query_ids=arrays['query_ids'],source_id=source)
        contexts[source]={'model':name,'sha256':digest(a.output/name),'fittingGenes':len(arrays['features']),'queries':len(arrays['query_ids']),'rank':model.rank}
    for name in ('response_model.py','inference.py','requirements.lock'): shutil.copyfile(Path(__file__).with_name(name),a.output/name)
    manifest={'schema':'slp.response-omf2-model/v1','rankRequested':a.rank,'alpha':a.alpha,'contexts':contexts,'scope':'panel-specific feature-linear molecular response; no unmeasured-query or dynamics claim'}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+'\n')
if __name__=='__main__': main()
