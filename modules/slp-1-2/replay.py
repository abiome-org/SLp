"""Replay the bundle's real molecular example on CPU or CUDA."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from inference import World


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    torch.set_num_threads(4)
    world = World(args.bundle, args.device)
    with np.load(args.bundle / 'example-input.npz', allow_pickle=False) as z:
        output = world.predict(dict(z), sample=True, steps=8, seed=120731)
    with np.load(args.bundle / 'example-output.npz', allow_pickle=False) as expected:
        errors = {}
        for name, value in output.items():
            np.testing.assert_allclose(value, expected[name], atol=3e-4, rtol=3e-4)
            errors[name] = float(np.max(np.abs(value - expected[name])))
    print(json.dumps({'replay': 'passed', 'device': args.device, 'maximum_absolute_errors': errors}))
