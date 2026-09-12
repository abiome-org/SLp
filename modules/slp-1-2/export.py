"""Materialize a new, checksummed standalone inference bundle from a trained run."""
import argparse
import json
from pathlib import Path
import shutil
from io_utils import sha, write_json


def export(run, output, checkpoint='best'):
    latest = json.loads((run / 'latest.json').read_text())
    folder = run / latest['path']
    weights = run / 'best.safetensors' if checkpoint == 'best' else folder / 'model.safetensors'
    if not weights.is_file():
        raise FileNotFoundError(weights)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(weights, output / 'model.safetensors')
    shutil.copy2(folder / 'config.json', output / 'config.json')
    for name in ('vocabulary.json', 'normalizers.npz', 'entities.npz', 'contexts.npz', 'corpus.json',
                 'run.json', 'best.json', 'completion.json', 'untrained-development.json', 'development-final.json'):
        if (run / name).is_file():
            shutil.copy2(run / name, output / name)
    # Use the source captured with the most recent training segment, rather
    # than silently exporting a later edited implementation.
    captures = [run / 'source'] + sorted(run.glob('source-resume-*'), key=lambda p: int(p.name.split('-')[-1]))
    source = captures[-1]
    for name in ('model.py', 'io_utils.py', 'inference.py', 'replay.py', 'CONTRACT.md', 'requirements.in', 'requirements-linux-cu128.lock'):
        shutil.copy2(source / name, output / name)
    for path in sorted(run.glob('run-resume-*.json')):
        shutil.copy2(path, output / path.name)
    # CPU packaging may be prepared after the training segment starts; it does
    # not change the captured model implementation or CUDA training receipt.
    for name in ('requirements-linux-cpu.lock', 'requirements-linux-arm64-cpu.lock'):
        cpu_lock = Path(__file__).with_name(name)
        if cpu_lock.exists():
            shutil.copy2(cpu_lock, output / cpu_lock.name)
    for name in ('LICENSE', 'THIRD_PARTY_NOTICES.md'):
        if (run / name).exists():
            shutil.copy2(run / name, output / name)
    if (run / 'evaluation-source').exists():
        shutil.copytree(run / 'evaluation-source', output / 'evaluation-source')
    write_json(output / 'manifest.json', {'schema': 'slp.bundle/v1.2', 'checkpoint': checkpoint,
               'files': {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()},
               'scope': 'Native PyTorch research artifact; not an OMF ModelPackage service deployment.'})
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', choices=['best', 'latest'], default='best')
    args = p.parse_args()
    print(export(args.run, args.output, args.checkpoint))
