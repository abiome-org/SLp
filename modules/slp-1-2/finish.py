"""Evaluate, export and replay a trained run; request GPU cleanup afterwards."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from data import Corpus
from export import export
from inference import World
from io_utils import sha, write_json


def main(args):
    torch.set_num_threads(4)
    subprocess.run([sys.executable, str(Path(__file__).with_name('evaluate.py')), '--root', str(args.root),
                    '--run', str(args.run), '--output', str(args.run / 'development-final.json')], check=True)
    bundle = export(args.run, args.bundle)
    corpus = Corpus(args.root, validation=True, verify=False, seed=120731)
    raw = corpus.molecular(1, 64, 64, source='frangieh_cells')
    example = {k: v for k, v in raw.items() if isinstance(v, np.ndarray)
               and k not in {'target', 'control_target', 'control_anchor', 'observed_fitness_triplet'}}
    np.savez_compressed(bundle / 'example-input.npz', **example)
    world = World(bundle, 'cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    result = world.predict(example, sample=True, steps=8, seed=120731)
    np.savez_compressed(bundle / 'example-output.npz', **result)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    manifest['files'] = {str(p.relative_to(bundle)): sha(p) for p in bundle.rglob('*') if p.is_file() and p != bundle / 'manifest.json'}
    write_json(bundle / 'manifest.json', manifest)
    subprocess.run([sys.executable, str(bundle / 'replay.py'), '--bundle', str(bundle), '--device', 'cuda'], check=True)
    write_json(args.run / 'export-complete.json', {'bundle': str(bundle), 'manifest_sha256': sha(bundle / 'manifest.json')})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--cleanup-request', type=Path)
    args = p.parse_args()
    status = 'failed'
    try:
        main(args)
        status = 'exported'
    finally:
        if args.cleanup_request:
            # The network volume survives deletion. Thirty minutes allows the
            # local collector to copy artifacts before the GPU is released.
            write_json(args.cleanup_request, {'status': status, 'terminate_at': time.time() + 1800})
