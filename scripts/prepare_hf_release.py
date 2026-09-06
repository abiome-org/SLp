"""Build the explicit SLp-1.1 r1 publication inventories; never copy payloads."""
import argparse
import json
from pathlib import Path
from publish_hf import digest


def build(root, output):
    output.mkdir(parents=True, exist_ok=False)
    data = {}

    def add(name, role, terms):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        data[name] = {'local': name, 'remote': name, 'role': role, 'terms': terms,
                      'size': path.stat().st_size, 'sha256': digest(path)}

    def directory(name, role, terms):
        for path in sorted((root / name).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                add(path.relative_to(root).as_posix(), role, terms)

    index = 'data/derived/slp11-cell-world-training-v5'
    manifest = json.loads((root / index / 'manifest.json').read_text())
    directory(index, 'molecular-index', 'mixed; THIRD_PARTY_NOTICES.md')
    for name, expected in manifest['sourceReceipts'].items():
        add(name, 'source-receipt' if name.startswith('rights/') else 'molecular-source',
            'source-specific; THIRD_PARTY_NOTICES.md')
        if data[name]['sha256'] != expected:
            raise ValueError(f'Changed molecular source: {name}')
    for key in ('frangieh', 'yeast'):
        directory(manifest[key]['path'], 'molecular-source', 'CC-BY-4.0')
    for row in manifest['populations'] + manifest['raw_cells']:
        if row['path'] not in data:
            add(row['path'], 'molecular-source', 'source-specific; THIRD_PARTY_NOTICES.md')
    directory('data/derived/slp11-genomic-fitness-world-v1', 'functional-fitting-development',
              'CC-BY-4.0 / CC0-1.0 / static-feature source terms')
    directory('data/derived/slp11-musl-world-pair-roster-v1', 'application-input',
              'MIT / static-feature source terms')
    add('data/models/MuSL/LICENSE', 'third-party-notice', 'MIT')
    for cv in (1, 2, 3):
        base = f'data/models/MuSL/processed_data/data/CV{cv}_bins_32/fold_data'
        for seed in (42, 432):
            for split in ('train', 'test'):
                for kind in ('pairs', 'labels'):
                    add(f'{base}/{split}_{kind}_seed{seed}.pkl', 'application-benchmark', 'MIT')
    for name in (
        'rights/zenodo-14062629-cc-by-4.0.yaml',
        'rights/slp-1-1-nadal-ribelles-yeast-seus-split-cc-by-4.0.yaml',
        'rights/sgd-stable-id-mapping-cc-by-4.0.yaml',
        'rights/sgd-protein-sequences-r64-5-1-cc-by-4.0.yaml',
        'rights/go-sgd-2022-09-19-cc-by-4.0.yaml',
    ):
        add(name, 'source-receipt', 'CC-BY-4.0')
    prefix = 'results/slp11-transition/'
    for name in ('genomic-fitness-world-contexts-v1',
                 'genomic-capacity-world-sl-decoder-v1'):
        directory(prefix + name, 'application-contexts' if 'contexts' in name else 'decoder-evidence',
                  'MIT predictions/weights; source terms for observed inputs')
    for name in (
        'genomic-capacity-world-v1/training.json',
        'genomic-capacity-world-v1/validation.npz',
        'genomic-capacity-world-label-free-v1/scores.json',
        'genomic-capacity-world-decoder-replay-v1.json',
        'genomic-capacity-world-decoder-uncertainty-v1.json',
        'genomic-capacity-world-label-free-attribution-v1.json',
        'genomic-capacity-world-multiseed-attribution-v1.json',
        'cell-world-v1-generative-development-final/report.json',
        'cellular-genomic-world-request-v1.npz',
        'cellular-genomic-world-native-replay-v2/predictions.npz',
        'cellular-genomic-world-native-replay-v2/report.json',
    ):
        add(prefix + name, 'evaluation-evidence', 'MIT outputs; source terms for observations')

    def write(name, value):
        path = output / name
        path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
        return path.relative_to(root).as_posix()

    inventory = {'schema': 'slp.hf-data-release/v1', 'release': 'SLp-1.1-r1',
        'source_commit': '214ab389a9bde43b1fe36d5bc3d4d308288700b0',
        'files': [{k: v for k, v in data[n].items() if k != 'local'} for n in sorted(data)],
        'scope': 'Prepared world-training inputs, separate application benchmark and evaluation evidence; not a raw-data archive.',
        'notices': 'THIRD_PARTY_NOTICES.md'}
    inventory_path = write('inventory.json', inventory)
    rows = list(data.values())

    def mapped(local, remote):
        return {'local': local, 'remote': remote, 'size': (root / local).stat().st_size,
                'sha256': digest(root / local)}

    rows.extend([mapped(inventory_path, 'inventory.json'),
                 mapped('release/hf-dataset.md', 'README.md'),
                 mapped('release/THIRD_PARTY_NOTICES.md', 'THIRD_PARTY_NOTICES.md')])
    write('dataset-plan.json', {'repo_id': 'potteryrage/SLp-1.1-data', 'repo_type': 'dataset',
        'message': 'Publish SLp-1.1 r1 prepared inputs and research evidence', 'files': rows})

    bundle = root / 'results/slp11-transition/cellular-genomic-world-predictor-v2'
    model_manifest = json.loads((bundle / 'manifest.json').read_text())
    destination = 'checkpoints/v1.1/SLp-1.1-r1/'
    model_rows = []
    for name, expected in model_manifest['files'].items():
        row = mapped((bundle / name).relative_to(root).as_posix(), destination + name)
        if row['sha256'] != expected:
            raise ValueError(f'Changed model payload: {name}')
        model_rows.append(row)
    model_rows.append(mapped((bundle / 'manifest.json').relative_to(root).as_posix(), destination + 'manifest.json'))
    for name in ('LICENSE', 'release/THIRD_PARTY_NOTICES.md'):
        model_rows.append(mapped(name, destination + Path(name).name))
    model_rows.append(mapped('release/hf-model.md', 'README.md'))
    write('model-plan.json', {'repo_id': 'potteryrage/SLp', 'repo_type': 'model',
        'message': 'Publish SLp-1.1 r1 cellular and genomic world model',
        'mutable': ['README.md'], 'files': model_rows})
    print(json.dumps({'data_files': len(rows), 'data_bytes': sum(r['size'] for r in rows),
                      'model_files': len(model_rows), 'model_bytes': sum(r['size'] for r in model_rows)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.root.resolve(), args.output.resolve())
