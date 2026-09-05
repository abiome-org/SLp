"""Prepare label-free official MuSL CV3 pair rosters for frozen-world features."""
from __future__ import annotations
import argparse, csv, hashlib, json, pickle, re, shutil
from pathlib import Path
import h5py
import numpy as np

SEEDS=(42,432)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read_mapping(musl_meta,feng_meta):
    musl=list(csv.DictReader(Path(musl_meta).open(encoding='utf-8-sig',newline='')))
    feng={row['symbol']:row for row in csv.DictReader(Path(feng_meta).open(encoding='utf-8-sig',newline=''))}
    ids=[]
    for row in musl:
        reference=feng.get(row['symbol'])
        if reference is None or reference['ensembl_gene_id']!=row['ensembl_gene_id']:
            raise ValueError(f"MuSL/Feng stable-ID mapping mismatch for {row['symbol']}")
        stable=row['ensembl_gene_id'].split('.')[0]
        if not stable.startswith('ENSG'):raise ValueError(f'invalid human stable ID {stable}')
        ids.append(stable)
    if len(ids)!=len(set(ids)):raise ValueError('MuSL stable-ID mapping is not one-to-one')
    return np.asarray(ids)

def feature_lookup(static_path,string_path,gtf_path):
    with np.load(static_path,allow_pickle=False) as a:
        ids=a['entity_id'].astype(str);base=np.asarray(a['feature_values'],np.float32)
    if base.shape!=(len(ids),577) or len(ids)!=len(set(ids)):raise ValueError('invalid canonical static577 pack')
    symbols={};pattern=re.compile(r'gene_id "([^"]+)".*gene_name "([^"]+)"')
    with Path(gtf_path).open(encoding='utf-8') as handle:
        for line in handle:
            match=pattern.search(line)
            if match:symbols.setdefault(match.group(1).split('.')[0],match.group(2))
    lookup={}
    with h5py.File(string_path,'r') as h5:
        for stable,vector in zip(ids,base):
            extra=np.zeros(65,np.float32);symbol=symbols.get(stable,'')
            if symbol in h5:extra[:64]=h5[symbol][:];extra[64]=1
            lookup[stable]=np.concatenate((vector,extra))
    receipts=[{'path':str(Path(path).resolve()),'sha256':digest(path)} for path in (static_path,string_path,gtf_path)]
    return lookup,642,receipts

def collect_pairs(fold_root,musl_ids,covered):
    memberships=[];all_pairs=set();summaries=[];input_receipts=[]
    for seed in SEEDS:
        train_path=Path(fold_root)/f'train_pairs_seed{seed}.pkl';test_path=Path(fold_root)/f'test_pairs_seed{seed}.pkl'
        input_receipts.extend({'path':str(p.resolve()),'sha256':digest(p)} for p in (train_path,test_path))
        with train_path.open('rb') as f:train=pickle.load(f)
        with test_path.open('rb') as f:test=pickle.load(f)
        if len(train)!=5 or len(test)!=5:raise ValueError('official MuSL seeds require five folds')
        for fold in range(5):
            rows=[];excluded=[]
            for partition,source in ((0,train[fold]),(1,test[fold])):
                source=np.asarray(source)
                if source.ndim!=2 or source.shape[1]!=2:raise ValueError('MuSL pair array must be [N,2]')
                if np.any(source<0) or np.any(source>=len(musl_ids)):raise ValueError('MuSL pair index out of range')
                for source_row,(a,b) in enumerate(source.astype(np.int64)):
                    pair=tuple(sorted((str(musl_ids[a]),str(musl_ids[b]))))
                    if pair[0]==pair[1]:raise ValueError('self pair in MuSL roster')
                    record=(partition,source_row,pair)
                    if pair[0] in covered and pair[1] in covered:
                        rows.append(record);all_pairs.add(pair)
                    else:excluded.append(record)
            memberships.append((seed,fold,rows))
            summaries.append({'seed':seed,'fold':fold,'sourcePairs':len(rows)+len(excluded),'coveredPairs':len(rows),
                'excludedPairs':len(excluded),'excludedTrainPairs':sum(x[0]==0 for x in excluded),
                'excludedTestPairs':sum(x[0]==1 for x in excluded)})
    return sorted(all_pairs),memberships,summaries,input_receipts

def write_snapshot(output,musl_ids,lookup,width,pairs,memberships,summaries,receipts,k562_adapter):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    genes=sorted({gene for pair in pairs for gene in pair});gene_pos={gene:i for i,gene in enumerate(genes)}
    pair_pos={pair:i for i,pair in enumerate(pairs)}
    np.savez_compressed(output/'genes.npz',gene_ids=np.asarray(genes),action_features=np.stack([lookup[g] for g in genes]),ncbi_taxon=np.asarray(9606))
    np.savez_compressed(output/'pairs.npz',pair_ids=np.asarray(pairs),gene_indices=np.asarray([[gene_pos[a],gene_pos[b]] for a,b in pairs],np.int32))
    for seed,fold,rows in memberships:
        np.savez_compressed(output/f'seed{seed}-fold{fold}.npz',pair_indices=np.asarray([pair_pos[r[2]] for r in rows],np.int32),
            partition=np.asarray([r[0] for r in rows],np.uint8),source_row=np.asarray([r[1] for r in rows],np.int32))
    shutil.copyfile(k562_adapter,output/'k562-adapter.npz')
    with np.load(k562_adapter,allow_pickle=False) as a:
        control=np.asarray(a['control_context_values'],np.float32);mask=np.asarray(a['control_context_mask'],bool)
        query_ids=a['query_ids'].astype(str)
    np.savez_compressed(output/'k562-control.npz',query_ids=query_ids,basal=control*np.float32(np.log(2.)),
        control_context_values=control,control_context_mask=mask)
    files={p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(output.glob('*.npz'))}
    missing=sorted(set(musl_ids)-set(lookup))
    manifest={'schema':'slp.musl-world-pair-roster/v1','benchmark':'MuSL official CV3','seeds':list(SEEDS),'folds':5,
        'featureDim':width,'ncbiTaxon':9606,'stableIdType':'Ensembl gene','uniquePairs':len(pairs),'coveredGenes':len(genes),
        'muslGenes':len(musl_ids),'uncoveredGenes':missing,'foldsSummary':summaries,'sourceReceipts':receipts,
        'files':files,'partitions':{'0':'training','1':'test'},'labelsPresent':False,
        'accessBoundary':'Pair identities and official fold membership only; benchmark label files were not opened.',
        'featureJoin':'Exact stable-ID join to raw 642-coordinate query features; uncovered genes and their pairs are excluded, never zero-filled.',
        'context':'Frozen K562 adapter from the verified eight-context research export; basal equals ln(2) times its control log2(1+CP10K) profile.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return manifest

def main():
    root=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser()
    p.add_argument('--fold-root',type=Path,default=root/'data/models/MuSL/processed_data/data/CV3_bins_32/fold_data')
    p.add_argument('--musl-meta',type=Path,default=root/'data/models/MuSL/processed_data/meta_table_7684.csv')
    p.add_argument('--feng-meta',type=Path,default=root/'data/feng2024/data/preprocessed_data/meta_table_9845.csv')
    p.add_argument('--export',type=Path,default=root/'results/slp11-transition/joint-world-eight-context-research-export-v1')
    p.add_argument('--static',type=Path,default=root/'data/derived/slp11-human-shared-static/ensembl116-source3-esm8m-shared-go-complete-v2/human-static-esm8m-shared-go-mf-cc-features.npz')
    p.add_argument('--string',type=Path,default=root/'data/tooling/slim-5a7e9ade/data/gene_string_embeddings.v0.3.h5')
    p.add_argument('--gtf',type=Path,default=root/'data/sources/replogle-perturbseq-gi-code/data_sharing/cellranger-GRCh38-1.2.0_only_genes.gtf')
    p.add_argument('--output',type=Path,default=root/'data/derived/slp11-musl-world-pair-roster-v1');a=p.parse_args()
    musl_ids=read_mapping(a.musl_meta,a.feng_meta);lookup,width,feature_receipts=feature_lookup(a.static,a.string,a.gtf)
    pairs,memberships,summaries,pair_receipts=collect_pairs(a.fold_root,musl_ids,set(lookup))
    receipts=[{'path':str(a.musl_meta.resolve()),'sha256':digest(a.musl_meta)},
        {'path':str(a.feng_meta.resolve()),'sha256':digest(a.feng_meta)},*pair_receipts,*feature_receipts,
        {'path':str((a.export/'adapters/k562.npz').resolve()),'sha256':digest(a.export/'adapters/k562.npz')}]
    manifest=write_snapshot(a.output,musl_ids,lookup,width,pairs,memberships,summaries,receipts,a.export/'adapters/k562.npz')
    print(json.dumps({'output':str(a.output),'uniquePairs':manifest['uniquePairs'],'coveredGenes':manifest['coveredGenes'],
        'uncoveredGenes':len(manifest['uncoveredGenes']),'excludedPairs':sum(x['excludedPairs'] for x in summaries)},sort_keys=True))
if __name__=='__main__':main()
