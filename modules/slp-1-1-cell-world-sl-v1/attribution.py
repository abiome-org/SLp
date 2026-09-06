"""Pretraining attribution with gene-resampled label-free benchmark uncertainty."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from bridge import sha,write


def main(a):
    with np.load(a.predictions) as z:
        rows={};occurrences=[];conflicts=set();repeat_error=0.
        for seed in (42,432):
            for fold in range(5):
                key=f'CV3-{seed}-{fold}'
                for ids,y,w,r,s in zip(z[key+'-pairs'],z[key+'-y'],z[key+'-world_label_free'],z[key+'-random_label_free'],z[key+'-static_cosine']):
                    pair=tuple(sorted(ids));value=(int(y),float(w),float(r),float(s))
                    if pair in rows:
                        repeat_error=max(repeat_error,float(np.max(np.abs(np.array(rows[pair][1:])-value[1:]))))
                        if not np.allclose(rows[pair][1:],value[1:],rtol=1e-5,atol=2e-6):raise ValueError('inconsistent repeated prediction')
                        if rows[pair][0]!=value[0]:conflicts.add(pair)
                    rows[pair]=value
                    occurrences.append((pair,value))
    # Preserve official labels, including contradictory cross-seed labels;
    # bootstrap both occurrences together through shared gene weights.
    pairs=np.array([x[0] for x in occurrences]);values=np.array([x[1] for x in occurrences]);y=values[:,0];p=values[:,1:]
    genes,inverse=np.unique(pairs,return_inverse=True);ix=inverse.reshape(-1,2);rng=np.random.default_rng(731)
    draws=[]
    for _ in range(500):
        weights=rng.poisson(1.,len(genes));weights=weights[ix[:,0]]*weights[ix[:,1]]
        score=np.array([roc_auc_score(y,p[:,j],sample_weight=weights) for j in range(3)])
        draws.append(np.r_[score,score[0]-score[1],score[0]-score[2]])
    draws=np.array(draws)
    report={'pair_occurrences':len(pairs),'unique_pairs':len(rows),'cross_seed_label_conflicts':len(conflicts),'genes':len(genes),'resampling':'500 seed731 Poisson gene weights; pair weight is product of both gene weights; official repeated occurrences preserved',
        'interval_scope':'descriptive two-way gene-cluster bootstrap of this retrospective benchmark, not prospective replication',
        'repeated_score_max_error':repeat_error,
        'scores':{},'predictions_sha256':sha(a.predictions)}
    point=[roc_auc_score(y,p[:,j]) for j in range(3)];point+= [point[0]-point[1],point[0]-point[2]]
    for j,name in enumerate(('world','random','static','world_minus_random','world_minus_static')):
        report['scores'][name]={'auroc':point[j],'interval95':np.quantile(draws[:,j],[.025,.975]).tolist()}
    write(a.output,report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
