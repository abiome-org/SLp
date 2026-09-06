"""Gene-cluster uncertainty for frozen, fold-local SL decoder comparisons."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from bridge import sha,write
from decoder import labels


def main(a):
    lock=json.loads((a.fit/'fit-manifest.json').read_text());folds=[];all_genes=set()
    for seed in (42,432):
        for fold in range(5):
            file=f'seed{seed}-fold{fold}-predictions.npz'
            if sha(a.fit/file)!=lock['files'][file]:raise ValueError('changed predictions')
            with np.load(a.fit/file) as z:
                ids=z['pair_ids'].astype(str);all_genes.update(ids.reshape(-1))
                folds.append({'ids':ids,'y':labels(a.labels/f'test_labels_seed{seed}.pkl',fold)[z['source_row']],
                              'p':np.stack([z[arm] for arm in ('world','random','static')],1)})
    lookup={g:i for i,g in enumerate(sorted(all_genes))}
    for fold in folds:fold['ix']=np.array([[lookup[g] for g in row] for row in fold['ids']])
    rng=np.random.default_rng(731);draws=[]
    for _ in range(500):
        gene_weight=rng.poisson(1.,len(lookup));metrics=[]
        for fold in folds:
            ix=fold['ix'];weight=gene_weight[ix[:,0]]*gene_weight[ix[:,1]]
            metrics.append([roc_auc_score(fold['y'],fold['p'][:,j],sample_weight=weight) for j in range(3)])
        value=np.mean(metrics,axis=0);draws.append(np.r_[value,value[0]-value[1],value[0]-value[2]])
    point=np.mean([[roc_auc_score(f['y'],f['p'][:,j]) for j in range(3)] for f in folds],axis=0)
    point=np.r_[point,point[0]-point[1],point[0]-point[2]];draws=np.array(draws)
    report={'method':'500 seed731 Poisson gene-weight draws shared across all ten folds; macro AUROC recomputed per draw',
        'scope':'retrospective gene-cluster uncertainty with fixed trained models; does not include model-retraining or biological-replication uncertainty',
        'fit_manifest_sha256':sha(a.fit/'fit-manifest.json'),'scores':{name:{'auroc':float(point[j]),'interval95':np.quantile(draws[:,j],[.025,.975]).tolist()}
            for j,name in enumerate(('world','random','static','world_minus_random','world_minus_static'))}}
    write(a.output,report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('fit','labels','output'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
