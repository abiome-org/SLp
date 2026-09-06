"""Replay a portable SL predictor against locked predictions, without labels."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model','selection','roster','baseline','world','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.model/'manifest.json').read_text())
    if sha(args.selection/'lock.json') != manifest['provenance']['selectionLockSha256']:
        raise ValueError('selection lock differs from exported source')
    lock = json.loads((args.selection/'lock.json').read_text())
    receipts = manifest['provenance']['featureReceipts']
    inputs = {'world-features.npz':args.world,'genes.npz':args.roster/'genes.npz'}
    for name in ('embed_pairs.npy','pair_summary.npy','public_relations.npy'):
        inputs[name] = args.baseline/name
    for name,path in inputs.items():
        if sha(path) != receipts[name]:
            raise ValueError(f'feature input differs from fitted receipt: {name}')
    # Verify executable before importing it; the predictor checks all payload files.
    if sha(args.model/'predictor.py') != manifest['files']['predictor.py']:
        raise ValueError('predictor source differs from bundle manifest')
    spec = importlib.util.spec_from_file_location('portable_sl_predictor',args.model/'predictor.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predictor = module.SLPredictor(args.model)
    baseline = [np.load(args.baseline/name,mmap_mode='r',allow_pickle=False)
                for name in ('embed_pairs.npy','pair_summary.npy','public_relations.npy')]
    with np.load(args.roster/'genes.npz',allow_pickle=False) as source:
        genes, gene_ids = source['action_features'].astype(np.float32),source['gene_ids'].astype(str)
    with np.load(args.roster/'pairs.npz',allow_pickle=False) as source:
        indices, pair_ids = source['gene_indices'],source['pair_ids'].astype(str)
    if not np.array_equal(gene_ids[indices],pair_ids):
        raise ValueError('roster gene and pair identities disagree')
    with np.load(args.world,allow_pickle=False) as source:
        world = source['features'].astype(np.float32)
        if not np.array_equal(source['pair_ids'].astype(str),pair_ids):
            raise ValueError('world feature identities disagree with roster')
    args.output.mkdir(parents=True)
    rows = []
    for item in manifest['models']:
        seed,fold = item['seed'],item['fold']
        name = f'seed{seed}-fold{fold}-selected.npz'
        if sha(args.selection/name) != lock['artifacts'][name]:
            raise ValueError('locked predictions changed')
        with np.load(args.selection/name,allow_pickle=False) as source:
            take = source['pair_indices']
            expected = source['prediction']
            if not np.array_equal(source['pair_ids'].astype(str),pair_ids[take]):
                raise ValueError('selected fold identities disagree with roster')
        features = np.concatenate([np.asarray(block[take],np.float32) for block in baseline],axis=1)
        descriptors = genes[indices[take]]
        actual = predictor.predict_fold(seed,fold,features,descriptors,world[take],pair_ids[take])
        drift = float(np.max(np.abs(actual-expected),initial=0))
        if drift > 1e-12:
            raise ValueError(f'portable prediction drift: seed{seed}/fold{fold}: {drift}')
        rows.append({'seed':seed,'fold':fold,'family':item['family'],'pairs':len(take),'maxAbsDrift':drift})
        if len(rows) == 1:
            np.savez_compressed(args.output/'example-request.npz',retained_baseline=features[:8],
                raw_gene_descriptor_pairs=descriptors[:8],world_features=world[take[:8]],pair_ids=pair_ids[take[:8]])
            np.savez_compressed(args.output/'example-reference.npz',score=actual[:8])
    report = {'schema':'slp.sl-predictor-replay/v1','manifestSha256':sha(args.model/'manifest.json'),
        'testLabelsRead':False,'rows':rows,'pairs':sum(row['pairs'] for row in rows),
        'maxAbsDrift':max(row['maxAbsDrift'] for row in rows)}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
