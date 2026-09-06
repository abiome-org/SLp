"""Execute a standalone world-to-SL predictor and compare a saved reference."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch


def main(a):
    sys.path.insert(0,str(a.bundle.resolve()))
    from predict import SLPredictor
    torch.set_num_threads(4)
    with np.load(a.request) as z:genes=z['gene_descriptors'];pairs=z['pair_indices']
    predictor=SLPredictor(a.bundle,a.device);result=predictor.predict(genes,pairs)
    reverse=predictor.predict(genes,pairs[:,::-1].copy())
    a.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(a.output/'predictions.npz',**result)
    symmetry=float(abs(result['sl_score']-reverse['sl_score']).max());errors={}
    if a.reference:
        with np.load(a.reference) as z:
            for k in result:errors[k]=float(abs(result[k]-z[k]).max())
    passed=symmetry<=1e-7 and all(np.isfinite(v).all() for v in result.values())
    if errors:passed=passed and all(e<= (2e-4 if k=='features' else 1e-5) for k,e in errors.items())
    report={'passed':bool(passed),'pair_order_score_max_error':symmetry,'reference_max_errors':errors,
        'pairs':len(pairs),'device':a.device,'torch':torch.__version__,'numpy':np.__version__}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if not passed:raise AssertionError('world-to-SL replay differs')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('bundle','request','output','reference'):p.add_argument('--'+name,type=Path,required=name!='reference')
    p.add_argument('--device',default='cpu');main(p.parse_args())
