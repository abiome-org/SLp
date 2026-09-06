"""Fixed, no-SL-training molecular-response decoder on two benchmark suites."""
import argparse,csv,json,pickle
from pathlib import Path
import numpy as np
from bridge import sha,write
from decoder import metrics


def normalized(x):
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-10)


def representations(directory):
    blocks=[];ids=None
    for name in ('k562','rpe1','hepg2'):
        with np.load(directory/f'{name}-single-responses.npz') as z:
            if ids is None:ids=z['gene_ids'].astype(str)
            else:assert np.array_equal(ids,z['gene_ids'])
            blocks.append(normalized(z['responses']))
    return ids,np.concatenate(blocks,1)/np.float32(np.sqrt(3))


def pair_scores(embedding,pairs):
    values=[]
    for start in range(0,len(pairs),2048):
        p=pairs[start:start+2048];values.append((embedding[p[:,0]]*embedding[p[:,1]]).sum(1))
    return np.concatenate(values)


def rank_metrics(y,score):
    if not y.sum():return None
    order=np.argsort(-score,kind='stable');truth=y[order]
    return {'mrr':float(1/(np.flatnonzero(truth)[0]+1)),'recall20':float(truth[:20].sum()/y.sum())}


def run(a,representation_loader=representations,model_receipts=None,scorer=pair_scores):
    a.output.mkdir(parents=True,exist_ok=False)
    ids,w=representation_loader(a.world);ri,r=representation_loader(a.random);assert np.array_equal(ids,ri)
    lookup={g:i for i,g in enumerate(ids)}
    with np.load(a.roster/'genes.npz') as z:
        assert np.array_equal(ids,z['gene_ids']);raw=z['action_features']
    with np.load(a.bundle/'normalizer.npz') as z:s=normalized((raw-z['mean'])/z['scale'])
    embeddings={'world_label_free':w,'random_label_free':r,'static_cosine':s}
    musl=list(csv.DictReader((a.root/'data/models/MuSL/processed_data/meta_table_7684.csv').open(encoding='utf-8-sig')))
    mapped=np.array([lookup.get(row['ensembl_gene_id'].split('.')[0],-1) for row in musl])
    held=set((a.index/'excluded-human-action-ids.txt').read_text().splitlines())
    rows=[];receipts={};arrays={}
    for protocol in ('CV1','CV2','CV3'):
        path=a.root/f'data/models/MuSL/processed_data/data/{protocol}_bins_32/fold_data'
        for seed in (42,432):
            pp=path/f'test_pairs_seed{seed}.pkl';lp=path/f'test_labels_seed{seed}.pkl'
            receipts[str(pp)]=sha(pp);receipts[str(lp)]=sha(lp)
            with pp.open('rb') as f:foldpairs=pickle.load(f)
            with lp.open('rb') as f:foldlabels=pickle.load(f)
            for fold,(pair,y) in enumerate(zip(foldpairs,foldlabels)):
                pair=mapped[np.asarray(pair,int)];y=np.asarray(y).reshape(-1);supported=(pair>=0).all(1)
                pair=pair[supported];y=y[supported]
                cold=np.array([ids[l] in held and ids[rr] in held for l,rr in pair])
                key=f'{protocol}-{seed}-{fold}';arrays[key+'-y']=y;arrays[key+'-pairs']=ids[pair];arrays[key+'-molecular-held']=cold
                for arm,emb in embeddings.items():
                    score=scorer(emb,pair);arrays[key+'-'+arm]=score
                    for subset,mask in (('all',np.ones(len(pair),bool)),('both_molecular_held',cold)):
                        if len(np.unique(y[mask]))<2:continue
                        rows.append({'suite':'MuSL','protocol':protocol,'seed':seed,'fold':fold,'arm':arm,'subset':subset,
                            'n':int(mask.sum()),'excluded_missing_static':int((~supported).sum()),**metrics(y[mask],score[mask])})
    # SLAMR is evaluated with the same fixed pan-context score. No target-cell
    # context, training pair, validation outcome or benchmark tuning is used.
    meta=list(csv.DictReader((a.root/'data/feng2024/data/preprocessed_data/meta_table_9845.csv').open(encoding='utf-8-sig')))
    symbols={row['symbol']:lookup[row['ensembl_gene_id'].split('.')[0]] for row in meta if row['ensembl_gene_id'].split('.')[0] in lookup}
    rank_rows=[]
    for cell,study in (('A549','28319113'),('JURKAT','30033366'),('K562','30033366')):
        path=a.root/f'data/models/SLAMR/data_slb_filtered/{study}/{cell}_scenario3_fold5_seed88.pkl';receipts[str(path)]=sha(path)
        with path.open('rb') as f:folds=pickle.load(f)
        for fold,(_,_,test) in enumerate(folds):
            per_arm={arm:[] for arm in embeddings};excluded=0;total=0
            for query,partners in test.items():
                total+=len(partners);kept=[(symbols[query],symbols[g],label=='SL') for g,_,label in partners if query in symbols and g in symbols]
                excluded+=len(partners)-len(kept)
                if not kept:continue
                pair=np.array([[l,rr] for l,rr,_ in kept]);y=np.array([label for _,_,label in kept])
                for arm,emb in embeddings.items():
                    score=scorer(emb,pair);result=rank_metrics(y,score)
                    if result is not None:per_arm[arm].append(result)
            for arm,rr in per_arm.items():rank_rows.append({'cell':cell,'fold':fold,'arm':arm,'queries_with_positive':len(rr),
                'total_pairs':total,'excluded_missing_static':excluded,**{k:float(np.mean([x[k] for x in rr])) if rr else None for k in ('mrr','recall20')}})
    macro={}
    for protocol in ('CV1','CV2','CV3'):
        for subset in ('all','both_molecular_held'):
            for arm in embeddings:
                rr=[x for x in rows if x['protocol']==protocol and x['subset']==subset and x['arm']==arm]
                macro[f'{protocol}/{subset}/{arm}']={'folds':len(rr),'n':sum(x['n'] for x in rr),**{k:float(np.mean([x[k] for x in rr])) for k in ('auroc','ap','pr_auc')}}
    np.savez_compressed(a.output/'predictions.npz',**arrays)
    write(a.output/'scores.json',{'rule':getattr(a,'rule','equal-context positive single-response cosine, frozen before any evaluation; no SL training'),
        'additional_model_receipts':model_receipts or {},
        'musl':rows,'musl_macro':macro,'slamr':rank_rows,'receipts':receipts,'source_sha256':sha(__file__),
        'molecular_world_sha256':sha(a.bundle/'model.safetensors'),'predictions_sha256':sha(a.output/'predictions.npz'),
        'scope':'retrospective benchmark; both_molecular_held restricts both genes to the original global molecular exclusion roster; SLAMR uses pan-context predictions'})
    print(json.dumps({'musl_macro':macro,'slamr':rank_rows},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','world','random','roster','bundle','index','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
