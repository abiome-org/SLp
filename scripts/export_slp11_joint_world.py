"""Export a self-contained research bundle from a completed joint-world run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'modules/slp-1-1-joint-world-v1/requirements-linux.lock'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export(model, checkpoint, output, requirements=LOCK, inference_override=None):
    model, output, requirements = Path(model), Path(output), Path(requirements)
    config = json.loads((model / 'config.json').read_text(encoding='utf-8'))
    if config.get('training', {}).get('steps_completed', 0) <= 0:
        raise ValueError('a completed training run is required')
    if Path(checkpoint).name != checkpoint or not checkpoint.endswith('.safetensors'):
        raise ValueError('checkpoint must be a safetensor filename')
    required = ['config.json', 'normalizer.npz', 'world_model.py',
                'response_model.py', 'inference.py', 'CONTRACT.md']
    for name in required:
        if not (model / name).is_file():
            raise FileNotFoundError(model / name)
    for name, expected in config.get('code', {}).items():
        if Path(name).name != name or digest(model / name) != expected:
            raise ValueError(f'captured source hash mismatch: {name}')
    weights = model / 'checkpoints' / checkpoint
    if not weights.is_file() or not requirements.is_file():
        raise FileNotFoundError('checkpoint or dependency lock missing')
    if output.exists():
        raise FileExistsError('export destination must be new')
    output.mkdir(parents=True)
    for name in required + ['train.py', 'data-manifest.json']:
        if (model / name).is_file():
            shutil.copyfile(model / name, output / name)
    override_record = None
    if inference_override is not None:
        inference_override = Path(inference_override)
        if not inference_override.is_file():
            raise FileNotFoundError(inference_override)
        original_hash = digest(model / 'inference.py')
        replacement_hash = digest(inference_override)
        shutil.copyfile(inference_override, output / 'inference.py')
        exported_config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
        exported_config.setdefault('code', {})['inference.py'] = replacement_hash
        (output / 'config.json').write_text(json.dumps(exported_config, indent=2, sort_keys=True) + '\n')
        override_record = {'originalInferenceSha256': original_hash,
                           'replacementInferenceSha256': replacement_hash}
    for directory in ('adapters', 'priors'):
        (output / directory).mkdir()
        for context in config['contexts']:
            if Path(context).name != context:
                raise ValueError('invalid context name')
            shutil.copyfile(model / directory / f'{context}.npz',
                            output / directory / f'{context}.npz')
    (output / 'checkpoints').mkdir()
    shutil.copyfile(weights, output / 'checkpoints' / checkpoint)
    shutil.copyfile(requirements, output / 'requirements-linux.lock')
    files = {str(p.relative_to(output)).replace('\\', '/'): digest(p)
             for p in sorted(output.rglob('*')) if p.is_file()}
    manifest = {'schema': 'slp.joint-world-research-export/v1',
                'selectedCheckpoint': checkpoint, 'files': files,
                'trainingConfigSha256': digest(model / 'config.json'),
                'status': 'local unpromoted research model',
                'trainingExecutor': config['runtime'],
                'dependencyPlatform': 'Linux x86_64 CPython 3.12, CUDA 12.8 wheels',
                'noDatasetRequiredForInference': True}
    if override_record is not None:
        manifest['inferenceOverride'] = override_record
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--requirements', type=Path, default=LOCK)
    parser.add_argument('--inference-override', type=Path)
    args = parser.parse_args()
    manifest = export(args.model, args.checkpoint, args.output, args.requirements,
                      args.inference_override)
    print(json.dumps({'output': str(args.output), 'files': len(manifest['files']),
                      'checkpoint': manifest['selectedCheckpoint']}))


if __name__ == '__main__':
    main()
