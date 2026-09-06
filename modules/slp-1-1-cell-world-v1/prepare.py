"""Build a molecular-only training index over existing immutable source payloads."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve(); out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    static=root/'data/derived/slp11-human-shared-static/ensembl116-source3-esm8m-shared-go-complete-v2/human-static-esm8m-shared-go-mf-cc-features.npz'
    with np.load(static,allow_pickle=False) as z: human_ids=z['entity_id'].astype(str);human=z['feature_values']
    # STRING columns are copied from already aligned public query/action records.
    human=np.pad(human,((0,0),(0,65)));lookup={g:i for i,g in enumerate(human_ids)}
    poproot=root/'data/derived/slp11-joint-world-context-transfer-v2-training-r4'
    popmanifest=json.loads((poproot/'manifest.json').read_text())
    excluded=set(); development_receipts={}
    for name in ('k562','rpe1','gwps','hepg2'):
        path=root/f'data/derived/slp11-joint-world-expanded-development-string-v1/{name}.npz'
        with np.load(path,allow_pickle=False) as z:excluded.update(z['gene_ids'].astype(str))
        development_receipts[path.relative_to(root).as_posix()]=sha(path)
    frroot=root/'data/derived/slp11-frangieh/paired-development-v1'
    with np.load(frroot/'development.npz',allow_pickle=False) as z:
        fr={k:z[k] for k in z.files}
    excluded.update(fr['action_ids'][fr['split_validation']].astype(str))
    pop_records=[]; receipts={static.relative_to(root).as_posix():sha(static),(frroot.relative_to(root)/'development.npz').as_posix():sha(frroot/'development.npz')}
    source_weights={'k562':.04,'rpe1':.04,'gwps':.10,'hepg2':.04,'norman':.07,
                    'mcf10a_full_d0':.02,'mcf10a_full_d6':.02,'mcf10a_tgfb1_d6':.02}
    for name in popmanifest['trainingSources']:
        path=poproot/f'{name}.npz'
        if sha(path)!=popmanifest['sources'][name]['sha256']:raise ValueError(f'changed source: {name}')
        receipts[path.relative_to(root).as_posix()]=popmanifest['sources'][name]['sha256']
        with np.load(path,allow_pickle=False) as z:
            ids=z['action_ids'].astype(str); mask=z['action_mask'];qids=z['query_ids'].astype(str)
            if ids.ndim==1:
                expanded=np.full(mask.shape,'',dtype=ids.dtype)
                if 'action_offsets' in z:
                    offsets=z['action_offsets']
                    if len(offsets)!=len(mask)+1 or offsets[-1]!=len(ids):raise ValueError('invalid action offsets')
                    for i,m in enumerate(mask):
                        row=ids[offsets[i]:offsets[i+1]]
                        if len(row)!=m.sum():raise ValueError('action cardinality mismatch')
                        expanded[i,m]=row
                elif len(ids)==len(mask):expanded[:,0]=ids
                else:raise ValueError('unrecognized action layout')
                ids=expanded
            allowed=np.array([all(g not in excluded for g in row[m]) for row,m in zip(ids,mask)])
            if 'combination_rows' in z.files:
                allowed[z['combination_rows'][z['combination_fold']==0]]=False
            rows=np.flatnonzero(allowed)
            for g,f in zip(qids,z['query_features']):
                if g in lookup:human[lookup[g],577:]=f[577:]
            for row,frow,m in zip(ids,z['action_features'],mask):
                for g,f in zip(row[m],frow[m]):
                    if g in lookup:human[lookup[g],577:]=f[577:]
            mode=int(z['mode_id']) if 'mode_id' in z.files else (1 if name=='norman' else 0)
            assay=int(z['assay_id']) if 'assay_id' in z.files else (1 if name=='norman' else 0)
            if name in ('gwps','hepg2'):assay=2 if name=='gwps' else 3
            namepath=f'{name}-index.npz'; np.savez_compressed(out/namepath,rows=rows,action_ids=ids[rows])
            pop_records.append({'name':name,'path':path.relative_to(root).as_posix(),'index':namepath,
                'rows':len(rows),'weight':source_weights[name],'assay':assay,'mechanism':mode,'taxon':9606})
    with np.load(root/'data/derived/slp11-yeast-atlas-counts/nadal-ribelles-rna-neural-fitting-v1/reference.npz',allow_pickle=False) as z:
        yeast_ids=z['fitting_action_ids'].astype(str);yqids=z['query_ids'].astype(str)
        yfeatures=z['action_features_normalized']*z['feature_std']+z['feature_mean']
        yquery=z['query_features_normalized']*z['feature_std']+z['feature_mean']
    # Equal weight for each species' static fitting-gene distribution.
    fitting_human=set()
    for rec in pop_records:
        with np.load(out/rec['index'],allow_pickle=False) as z:fitting_human.update(z['action_ids'].reshape(-1))
    fitting_human.update(fr['action_ids'][fr['split_train']].astype(str));fitting_human-=excluded;fitting_human.discard('')
    hfit=human[[lookup[g] for g in sorted(fitting_human) if g in lookup]]
    yfit=np.pad(yfeatures,((0,0),(0,65)))
    mean=(hfit.mean(0)+yfit.mean(0))*.5
    variance=(np.square(hfit-mean).mean(0)+np.square(yfit-mean).mean(0))*.5
    scale=np.maximum(np.sqrt(variance),.02)
    np.savez_compressed(out/'features.npz',human_ids=human_ids,human_features=human.astype(np.float32),
        yeast_action_ids=yeast_ids,yeast_action_features=yfit.astype(np.float32),yeast_query_ids=yqids,
        yeast_query_features=np.pad(yquery,((0,0),(0,65))).astype(np.float32),feature_mean=mean,feature_scale=scale)
    raw=[]
    for name,base,rowsfile,countsfile in (
        ('k562_cells','data/derived/slp11-human-k562-essential-count-latent-training-mmap-v1','rows.npz','counts.uint16'),
        ('rpe1_cells','data/derived/slp11-human-rpe1-essential-raw-cells-v1','reconstruction-train-row-metadata.npz','reconstruction-train-counts.uint16')):
        directory=root/base; metadata=directory/rowsfile
        with np.load(metadata,allow_pickle=False) as z:
            ids=z['action_ids'].astype(str);control=z['is_control'];qids=z['query_ids'].astype(str)
            take=np.flatnonzero(control|np.array([g in fitting_human and g in lookup and g not in excluded for g in ids]))
            np.savez_compressed(out/f'{name}-index.npz',rows=take,action_ids=ids[take],is_control=control[take],
                library_size=z['library_size'][take],gem_group=z['gem_group'][take],query_ids=qids)
            shape=[len(ids),len(qids)]
        metadata_hash=sha(metadata); counts_hash=sha(directory/countsfile)
        raw_manifest=json.loads((directory/'manifest.json').read_text())
        expected=raw_manifest['counts']['sha256'] if name=='k562_cells' else raw_manifest['trainingMmap']['sha256']
        if counts_hash!=expected:raise ValueError(f'changed raw cells: {name}')
        receipts[metadata.relative_to(root).as_posix()]=metadata_hash;receipts[(directory/countsfile).relative_to(root).as_posix()]=counts_hash
        raw.append({'name':name,'path':(directory/countsfile).relative_to(root).as_posix(),
            'index':f'{name}-index.npz','shape':shape,'rows':len(take),'weight':.15,'assay':5,'mechanism':0,'taxon':9606})
    frtrain=np.asarray([i for i in fr['split_train'] if str(fr['action_ids'][i]) not in excluded and str(fr['action_ids'][i]) in lookup])
    np.savez_compressed(out/'frangieh-index.npz',rows=frtrain,action_ids=fr['action_ids'][frtrain])
    fr_fitting={k:(v[frtrain] if k in ('rna_targets','rna_observed','protein_targets','protein_observed','action_ids','action_taxon','context_ids','source_target_guide_sets','num_cells','record_ids') else v)
                for k,v in fr.items() if k not in ('split_train','split_validation')}
    np.savez_compressed(out/'frangieh-populations.npz',**fr_fitting)
    (out/'excluded-human-action-ids.txt').write_text('\n'.join(sorted(excluded))+'\n')
    rights=['figshare-replogle-2022-k562-essential-singlecell-cc-by-4.0.yaml','figshare-replogle-2022-rpe1-essential-singlecell-cc-by-4.0.yaml',
            'figshare-replogle-2022-k562-gwps-cc-by-4.0.yaml','nadig-2025-gse264667-public-molecular.yaml',
            'frangieh-2021-scperturb-processed-cc-by-4.0.yaml','slp-1-1-nadal-ribelles-yeast-molecular-summaries-cc-by-4.0.yaml',
            'norman-2019-geo-public-molecular.yaml','gse164996-combinatorial-cropseq-public-molecular.yaml']
    for name in rights:
        path=root/'rights'/name
        if 'trainingAllowed: true' not in path.read_text():raise ValueError(f'rights do not admit training: {name}')
        receipts['rights/'+name]=sha(path)
    yeastroot=root/'data/derived/slp11-yeast-atlas-counts/nadal-ribelles-rna-neural-fitting-v1'
    ym=json.loads((yeastroot/'manifest.json').read_text())
    for key,filename in [('targets','train-targets.npy'),('metadata','train-metadata.npz'),('reference','reference.npz')]:
        path=yeastroot/filename;digest=sha(path)
        if digest!=ym[key]['sha256']:raise ValueError(f'changed yeast {key}')
        receipts[path.relative_to(root).as_posix()]=digest
    frraw=root/'data/derived/slp11-frangieh/paired-singlecell-train-control-v1'
    receipts[(frraw/'manifest.json').relative_to(root).as_posix()]=sha(frraw/'manifest.json')
    write(out/'manifest.json',{'schema':'slp.cell-world-training-index/v1','populations':pop_records,'raw_cells':raw,
        'frangieh':{'path':'data/derived/slp11-frangieh/paired-singlecell-train-control-v1','population_file':'frangieh-populations.npz',
                     'weight':.15,'fitting_populations':len(frtrain),'index':'frangieh-index.npz'},
        'yeast':{'path':'data/derived/slp11-yeast-atlas-counts/nadal-ribelles-rna-neural-fitting-v1','weight':.20},
        'excludedHumanGenes':len(excluded),'sourceReceipts':receipts,'exclusionRosterReceipts':development_receipts,'rights':rights,
        'scope':'Human and yeast molecular observations only; no benchmark labels or fitted response priors.',
        'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'output':str(out),'human_population_rows':sum(x['rows'] for x in pop_records),
                      'replogle_cells':sum(x['rows'] for x in raw),'frangieh_populations':len(frtrain),'excluded_genes':len(excluded)}))

if __name__=='__main__':main()
