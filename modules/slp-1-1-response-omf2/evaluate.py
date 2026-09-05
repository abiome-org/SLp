import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from inference import load_bundle

def center(x):
    x=x-x[:1]; return x-x.mean(0)
def score(y,p,a):
    mse=float(np.square(y-p).mean())
    x=center(y-a); z=center(p-a); x-=x.mean(1,keepdims=True); z-=z.mean(1,keepdims=True)
    den=np.linalg.norm(x,axis=1)*np.linalg.norm(z,axis=1); good=den>1e-12
    return {'geneProfileMse':mse,'independentlyQueryCenteredResidualPearson':float((np.sum(x[good]*z[good],axis=1)/den[good]).mean()),'finiteCorrelationGenes':int(good.sum())}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--model',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    bundle=load_bundle(a.model); metrics={}; checks=[]; compatibility=[]
    for source in ('k562','rpe1'):
      with np.load(a.data/f'{source}.npz',allow_pickle=False) as d: q={k:np.asarray(d[k]) for k in d.files}
      model_path=a.model/bundle.manifest['contexts'][source]['model']
      with np.load(model_path,allow_pickle=False) as fitted:
       model_queries=np.asarray(fitted['query_ids']).astype(str)
      pred=bundle.predict(source,q['features'],q['control_prediction']); cand=score(q['truth'],pred,q['control_prediction']); ridge=score(q['truth'],q['static_ridge_prediction'],q['control_prediction'])
      compatibility += [bool(np.array_equal(model_queries,q['query_ids'].astype(str))),bool(pred.shape==q['truth'].shape),bool(np.isfinite(pred).all())]
      metrics[source]={'candidate':cand,'staticRidge':ridge}; checks += [cand['geneProfileMse']<=.99*ridge['geneProfileMse'],cand['independentlyQueryCenteredResidualPearson']>=.1,cand['independentlyQueryCenteredResidualPearson']>=ridge['independentlyQueryCenteredResidualPearson']]
    flat={'k562_mse':metrics['k562']['candidate']['geneProfileMse'],'rpe1_mse':metrics['rpe1']['candidate']['geneProfileMse'],'k562_centered_pearson':metrics['k562']['candidate']['independentlyQueryCenteredResidualPearson'],'rpe1_centered_pearson':metrics['rpe1']['candidate']['independentlyQueryCenteredResidualPearson'],'passed':bool(all(checks)),'compatibilityPassed':bool(all(compatibility))}
    (a.output/'metrics.json').write_text(json.dumps(flat,indent=2,sort_keys=True,allow_nan=False)+'\n')
    (a.output/'report.json').write_text(json.dumps({'schema':'slp.response-omf2-evaluation/v1','metrics':metrics,'compatibilityChecks':compatibility},indent=2,sort_keys=True,allow_nan=False)+'\n')
if __name__=='__main__': main()
