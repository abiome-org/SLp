"""Export the already selected SL fold models without reading labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def contained(root, name):
    path = (root/name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError('artifact path escapes its source')
    return path


def export(selection, v1, v2, module, output):
    if output.exists():
        raise FileExistsError(output)
    lock = load(selection/'lock.json')
    if lock['schema'] != 'slp.sl-readout-family-lock/v1':
        raise ValueError('unsupported selection lock')
    if lock['rule'] != 'lower inner_logloss per seed/fold; exact tie selects v1':
        raise ValueError('unsupported family selection rule')
    roots = {'v1':v1, 'v2':v2}
    source_params, source_manifests = {}, {}
    for family, root in roots.items():
        record = lock['families'][family]
        for name, key in [('params.json','paramsSha256'), ('fit-manifest.json','fitManifestSha256')]:
            if sha(root/name) != record[key]:
                raise ValueError(f'{family} {name} differs from selection lock')
        source_params[family] = {(int(x['seed']),int(x['fold'])):x for x in load(root/'params.json')['runs']}
        source_manifests[family] = load(root/'fit-manifest.json')
    expected = {(s,f) for s in (42,432) for f in range(5)}
    feature_receipts = {}
    for filename in ('world-features.npz','embed_pairs.npy','pair_summary.npy','public_relations.npy','genes.npz'):
        matches = [value for name,value in source_manifests['v2']['inputs'].items()
                   if name.replace('\\','/').split('/')[-1] == filename]
        if len(matches) != 1:
            raise ValueError(f'expected one feature receipt for {filename}')
        feature_receipts[filename] = matches[0]
        if filename != 'genes.npz':
            other = [value for name,value in source_manifests['v1']['inputs'].items()
                     if name.replace('\\','/').split('/')[-1] == filename]
            if other != matches:
                raise ValueError(f'families use different frozen features: {filename}')
    selected = {(int(x['seed']),int(x['fold'])):x for x in lock['selections']}
    if len(lock['selections']) != 10 or set(selected) != expected:
        raise ValueError('selection must contain all ten distinct runs')
    plan, models = [], []
    for seed, fold in sorted(expected):
        item = selected[seed,fold]
        left, right = source_params['v1'][seed,fold], source_params['v2'][seed,fold]
        chosen = 'v1' if left['inner_logloss'] <= right['inner_logloss'] else 'v2'
        if item['selectedFamily'] != chosen:
            raise ValueError('selection differs from frozen training-validation rule')
        params = source_params[chosen][seed,fold]
        weight_key = 'augmented_blend_weight' if chosen == 'v1' else 'world_augmented_blend_weight'
        weight = float(params[weight_key])
        if not 0 <= weight <= 1:
            raise ValueError('blend weight is outside [0,1]')
        row = {'seed':seed, 'fold':fold, 'family':chosen, 'blendWeight':weight}
        suffixes = ('baseline','augmented') if chosen == 'v1' else ('static-control','world-augmented')
        for arm, suffix in zip(('control','augmented'), suffixes):
            source_name = f'seed{seed}-fold{fold}-{suffix}.txt'
            source = contained(roots[chosen], source_name)
            expected_hash = source_manifests[chosen]['artifacts'][source_name]
            if sha(source) != expected_hash:
                raise ValueError(f'frozen booster changed: {source_name}')
            destination = f'models/seed{seed}-fold{fold}-{arm}.txt'
            plan.append((source,destination,expected_hash))
            row[arm] = destination
        models.append(row)
    for name in ('predictor.py','CONTRACT.md','requirements.lock'):
        source = module/name
        plan.append((source,name,sha(source)))
    output.mkdir(parents=True)
    (output/'models').mkdir()
    files = {}
    for source, destination, expected_hash in plan:
        shutil.copyfile(source, output/destination)
        files[destination] = sha(output/destination)
        if files[destination] != expected_hash:
            raise ValueError('export copy differs from frozen input')
    manifest = {'schema':'slp.sl-predictor/v1',
        'featureWidths':{'retainedBaseline':1081,'geneDescriptors':642,'world':1054},
        'featureComposition':{'staticPair':['sum','absolute_difference','product'],
            'retainedBaseline':['frozen_pair_embedding_1032','state_summary_40','public_relations_9']},
        'models':models, 'files':files,
        'provenance':{'selectionLockSha256':sha(selection/'lock.json'),
            'families':{family:{k:v for k,v in lock['families'][family].items() if k != 'path'} for family in roots},
            'featureReceipts':feature_receipts,
            'testLabelsReadDuringExport':False},
        'scope':'Research ranking from prepared frozen feature blocks; no clinical calibration or ensemble benchmark claim.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(output),'files':len(files),'foldModels':len(models),
        'manifestSha256':sha(output/'manifest.json')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('selection','v1','v2','module','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    export(args.selection,args.v1,args.v2,args.module,args.output)


if __name__ == '__main__':
    main()
