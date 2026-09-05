"""Self-contained adaptive development evaluation for a joint-world bundle."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from inference import JointWorldBundle

def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:np.asarray(a[k]) for k in a.files}
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def center(x):
    x=np.asarray(x,np.float64);x=x-x[:1];return x-x.mean(0,keepdims=True)
def pearson(a,b):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
    a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True)
    den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1);ok=den>1e-12
    return (float(np.mean(np.sum(a[ok]*b[ok],1)/den[ok])) if ok.any() else None,int(ok.sum()))
def response_metrics(y,p,a):
    r,n=pearson(center(y-a),center(p-a))
    return {'mse':float(np.square(np.asarray(y,np.float64)-p).mean()),'centeredPearson':r,'finiteRows':n}
def combination_metrics(y,p,additive):
    r,n=pearson(np.asarray(p)-additive,np.asarray(y)-additive)
    cr,cn=pearson(center(np.asarray(p)-additive),center(np.asarray(y)-additive))
    return {'mse':float(np.square(np.asarray(y,np.float64)-p).mean()),'nonadditivePearson':r,
            'centeredNonadditivePearson':cr,'finiteRows':n,'finiteCenteredRows':cn}
def single(features):
    actions=np.zeros((len(features),2,features.shape[1]),np.float32);mask=np.zeros((len(features),2),bool)
    actions[:,0]=features;mask[:,0]=True;return actions,mask
def prior_only(bundle,context,actions,mask,observed):
    flat=bundle.priors[context].predict(actions.reshape(-1,actions.shape[-1])).reshape(len(actions),actions.shape[1],observed.shape[1])
    return np.asarray(observed,np.float64)+(flat*mask[...,None]).sum(1)
def held_metrics(truth,prediction,observed,gene_ids):
    truth=np.asarray(truth,np.float64);prediction=np.asarray(prediction,np.float64);observed=np.asarray(observed,bool);genes=np.asarray(gene_ids).astype(str)
    if truth.shape!=prediction.shape or observed.shape!=truth.shape or genes.shape!=(len(truth),):raise ValueError('held metric arrays must align')
    if not observed.any(1).all():raise ValueError('every development view needs measured queries')
    view_mse=np.asarray([np.square(truth[i,observed[i]]-prediction[i,observed[i]]).mean() for i in range(len(truth))])
    view_corr=[]
    for i in range(len(truth)):
        x=truth[i,observed[i]];y=prediction[i,observed[i]];x=x-x.mean();y=y-y.mean();den=np.linalg.norm(x)*np.linalg.norm(y)
        if den>1e-12:view_corr.append(float(x@y/den))
    gene_mse=[];gene_corr=[]
    for gene in sorted(set(genes)):
        take=genes==gene;count=observed[take].sum(0);keep=count>0;yt=np.where(observed[take],truth[take],0.).sum(0)[keep]/count[keep];yp=np.where(observed[take],prediction[take],0.).sum(0)[keep]/count[keep]
        gene_mse.append(float(np.square(yt-yp).mean()));yt=yt-yt.mean();yp=yp-yp.mean();den=np.linalg.norm(yt)*np.linalg.norm(yp)
        if den>1e-12:gene_corr.append(float(yt@yp/den))
    common=observed.all(0)
    centered_view, centered_view_count = (None, 0)
    centered_gene, centered_gene_count = (None, 0)
    if common.any():
        centered_view, centered_view_count = pearson(center(truth[:,common]),center(prediction[:,common]))
        unique=sorted(set(genes))
        gene_truth=np.stack([truth[genes==gene][:,common].mean(0) for gene in unique])
        gene_prediction=np.stack([prediction[genes==gene][:,common].mean(0) for gene in unique])
        centered_gene, centered_gene_count=pearson(center(gene_truth),center(gene_prediction))
    return {'populationViewEqualWeightMse':float(view_mse.mean()),'populationViewProfilePearson':float(np.mean(view_corr)) if view_corr else None,'populationViews':len(truth),'finitePopulationViewPearsons':len(view_corr),'uniqueGeneEqualWeightMse':float(np.mean(gene_mse)),'uniqueGeneMeanProfilePearson':float(np.mean(gene_corr)) if gene_corr else None,'uniqueGenes':len(gene_mse),'finiteUniqueGenePearsons':len(gene_corr),
            'populationViewIndependentlyQueryCenteredPearson':centered_view,'finiteCenteredPopulationViews':centered_view_count,
            'uniqueGeneIndependentlyQueryCenteredPearson':centered_gene,'finiteCenteredUniqueGenes':centered_gene_count,
            'commonMeasuredQueries':int(common.sum())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--development',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');a=p.parse_args()
    if (a.output/'report.json').exists() or (a.output/'predictions.npz').exists():raise FileExistsError('evaluation artifacts already exist')
    if a.device=='cpu':torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    bundle=JointWorldBundle(a.model,a.checkpoint,a.device);reports={};arrays={}
    for context in ('k562','rpe1'):
        d=load(a.development/f'{context}.npz');ref=load(a.reference/f'development-forecast-{context}.npz')
        if not np.array_equal(bundle.query_ids(context),d['query_ids'].astype(str)) or not np.array_equal(ref['query_ids'].astype(str),d['query_ids'].astype(str)) or not np.array_equal(ref['gene_ids'].astype(str),d['gene_ids'].astype(str)):raise ValueError(f'{context}: evaluation identity mismatch')
        actions,mask=single(d['features']);anchor=np.asarray(d['control_prediction'],np.float64)
        candidate=bundle.predict(context,actions,mask,anchor);truth=np.asarray(d['truth'],np.float64)
        reports[context]={name:response_metrics(truth,value,anchor) for name,value in {
            'jointWorld':candidate,'staticRidge':d['static_ridge_prediction'],'retainedRank32':ref['rank32_prediction'],'priorOnly':prior_only(bundle,context,actions,mask,anchor)}.items()}
        arrays.update({f'{context}_query_ids':d['query_ids'],f'{context}_gene_ids':d['gene_ids'],f'{context}_truth':truth,
                       f'{context}_control':anchor,f'{context}_joint_world':candidate,f'{context}_static_ridge':d['static_ridge_prediction'],f'{context}_retained_rank32':ref['rank32_prediction']})
    d=load(a.development/'norman.npz');fold=int(bundle.settings.get('training',{}).get('norman_fold',0));which=np.flatnonzero(d['combination_fold']==fold)
    rows=d['combination_rows'][which];parents=d['combination_single_rows'][which];common=d['combination_common_query_mask'][which].all(0)
    if not common.any() or not np.array_equal(bundle.query_ids('norman'),d['query_ids'].astype(str)):raise ValueError('Norman query support mismatch')
    basal=np.asarray(d['basal'][rows],np.float32);truth=np.asarray(d['targets'][rows],np.float64);left=np.asarray(d['targets'][parents[:,0]],np.float32);right=np.asarray(d['targets'][parents[:,1]],np.float32)
    pair=np.asarray(d['action_features'][rows],np.float32);pairmask=np.asarray(d['action_mask'][rows],bool);aa,onemask=single(pair[:,0]);bb,_=single(pair[:,1])
    pred_a=bundle.predict('norman',aa,onemask,basal);pred_b=bundle.predict('norman',bb,onemask,basal)
    observed_ab=bundle.predict('norman',bb,onemask,basal,observed=left);observed_ba=bundle.predict('norman',aa,onemask,basal,observed=right)
    autonomous_ab=bundle.predict('norman',bb,onemask,basal,observed=pred_a);autonomous_ba=bundle.predict('norman',aa,onemask,basal,observed=pred_b)
    values={'observedParentAThenB':observed_ab,'observedParentBThenA':observed_ba,'observedParentAverage':.5*(observed_ab+observed_ba),
            'directTwoActions':bundle.predict('norman',pair,pairmask,basal),
            'autonomousAThenB':autonomous_ab,'autonomousBThenA':autonomous_ba,'autonomousAverage':.5*(autonomous_ab+autonomous_ba),
            'observedAdditive':left.astype(np.float64)+right.astype(np.float64)-basal,
            'predictedAdditive':pred_a+pred_b-basal,'priorOnly':prior_only(bundle,'norman',pair,pairmask,basal),
            'observedParentPrior':.5*(prior_only(bundle,'norman',bb,onemask,left)+prior_only(bundle,'norman',aa,onemask,right))}
    norman_metrics={name:combination_metrics(truth[:,common],value[:,common],values['observedAdditive'][:,common]) for name,value in values.items()}
    reports['norman']={'fold':fold,'heldCombinations':len(rows),'commonQueries':int(common.sum()),'metrics':norman_metrics}
    arrays.update({'norman_query_ids':d['query_ids'],'norman_rows':rows,'norman_parents':parents,'norman_common':common,'norman_truth':truth,**{f'norman_{k}':v for k,v in values.items()}})
    for context in ('gwps','hepg2'):
        if context not in bundle.settings['contexts'] or not (a.development/f'{context}.npz').is_file():continue
        extra=load(a.development/f'{context}.npz');query_ids=extra['query_ids'].astype(str);gene_ids=extra['gene_ids'].astype(str)
        if not np.array_equal(bundle.query_ids(context),query_ids):raise ValueError(f'{context}: query axis mismatch')
        actions=extra['features'][:,None,:];action_mask=np.ones((len(actions),1),bool);basal=np.asarray(extra['basal'],np.float64);control=np.broadcast_to(extra['control_context_values'],basal.shape);control_mask=np.broadcast_to(extra['control_context_mask'],basal.shape)
        candidate=bundle.predict(context,actions,action_mask,basal,control_context_values=control,control_context_mask=control_mask);prior=prior_only(bundle,context,actions,action_mask,basal);truth=np.asarray(extra['truth'],np.float64);observed=np.asarray(extra['observed'],bool) & bundle.supported_query_mask(context)[None,:]
        reports[context]={'jointWorld':held_metrics(truth,candidate,observed,gene_ids),'priorOnly':held_metrics(truth,prior,observed,gene_ids),'zeroResponse':held_metrics(truth,basal,observed,gene_ids),'targetUnits':str(extra['target_units'])}
        arrays.update({f'{context}_query_ids':query_ids,f'{context}_gene_ids':gene_ids,f'{context}_truth':truth,f'{context}_observed':observed,f'{context}_joint_world':candidate,f'{context}_prior_only':prior,f'{context}_zero_response':basal})
    a.output.mkdir(parents=True,exist_ok=True);predictions=a.output/'predictions.npz';np.savez_compressed(predictions,**arrays)
    checkpoint=a.model/'checkpoints'/a.checkpoint
    report={'schema':'slp.joint-world-omf2-development-evaluation/v1','adaptiveDevelopment':True,'independentConfirmation':False,
            'protectedTestAccessed':False,'checkpoint':{'name':a.checkpoint,'sha256':digest(checkpoint),'device':a.device},'sources':reports,
            'predictions':{'path':predictions.name,'sha256':digest(predictions)}}
    (a.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    metrics={'k562_mse':reports['k562']['jointWorld']['mse'],'rpe1_mse':reports['rpe1']['jointWorld']['mse'],
             'k562_centered_pearson':reports['k562']['jointWorld']['centeredPearson'],'rpe1_centered_pearson':reports['rpe1']['jointWorld']['centeredPearson'],
             'norman_autonomous_mse':norman_metrics['autonomousAverage']['mse'],'norman_observed_parent_mse':norman_metrics['observedParentAverage']['mse']}
    for context in ('gwps','hepg2'):
        if context in reports:
            metrics[f'{context}_population_view_mse']=reports[context]['jointWorld']['populationViewEqualWeightMse']
            metrics[f'{context}_unique_gene_mse']=reports[context]['jointWorld']['uniqueGeneEqualWeightMse']
    (a.output/'metrics.json').write_text(json.dumps(metrics,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(metrics,sort_keys=True))
if __name__=='__main__':main()
