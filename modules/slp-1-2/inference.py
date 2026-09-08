"""Standalone SLp-1.2 inference. No training corpus or SLp-1.1 model required."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from io_utils import sha
from model import Config, WorldModel


class World:
    def __init__(self, bundle, device='cpu'):
        self.bundle = Path(bundle)
        manifest = json.loads((self.bundle / 'manifest.json').read_text())
        for name, digest in manifest['files'].items():
            path = (self.bundle / name).resolve()
            if not path.is_relative_to(self.bundle.resolve()) or sha(path) != digest:
                raise ValueError('Bundle checksum/path mismatch: ' + name)
        self.device = torch.device(device)
        config = Config(**json.loads((self.bundle / 'config.json').read_text()))
        self.model = WorldModel(config).to(self.device).eval()
        self.model.load_state_dict(load_file(str(self.bundle / 'model.safetensors'), device=str(self.device)))
        self.vocabulary = json.loads((self.bundle / 'vocabulary.json').read_text())
        self.lookup = {name: i for i, name in enumerate(self.vocabulary)}
        with np.load(self.bundle / 'entities.npz', allow_pickle=False) as z:
            self.descriptors = z['descriptors']
        with np.load(self.bundle / 'contexts.npz', allow_pickle=False) as z:
            self.contexts = dict(zip(z['ids'].astype(str), z['standardized']))

    @torch.inference_mode()
    def predict(self, batch, *, sample=False, steps=32, seed=731):
        tensors = {k: torch.as_tensor(v, device=self.device) for k, v in batch.items() if isinstance(v, (np.ndarray, torch.Tensor))}
        out = self.model(tensors)
        result = {k: out[k].cpu().numpy() for k in ('mean', 'log_variance')}
        if sample:
            result['sample'] = self.model.generate(tensors, steps=steps, seed=seed).cpu().numpy()
        return result

    def fitness(self, interventions, *, taxon, context_ids=None, descriptors=None):
        """Predict quantitative fitness for batches of simultaneous knockouts.

        Human units are the source gene-effect score; yeast units are log
        relative fitness. Human combinations have no human combination-fitness
        supervision in this bootstrap run. ``descriptors`` may supply already
        normalized 702-dimensional features for entities outside the vocabulary.
        """
        if taxon not in (9606, 4932):
            raise ValueError('Supported taxa are human 9606 and yeast 4932')
        if not interventions or any(len(set(row)) != len(row) for row in interventions):
            raise ValueError('Use a nonempty batch and unique genes within each action set')
        size, actions = len(interventions), max(1, max(map(len, interventions)))
        z = lambda *shape: np.zeros(shape, np.float32)
        i = lambda *shape: np.zeros(shape, np.int64)
        m = lambda *shape: np.zeros(shape, bool)
        b = dict(observation_features=z(size, 1, 702), observation_values=z(size, 1),
                 observation_ids=i(size, 1), observation_modality=i(size, 1), observation_mask=m(size, 1),
                 action_features=z(size, actions, 702), action_ids=i(size, actions), action_mask=m(size, actions),
                 action_values=z(size, actions, 3), action_known=m(size, actions, 3), action_mechanism=np.full((size, actions), 2),
                 query_features=z(size, 1, 702), query_ids=i(size, 1), query_modality=np.full((size, 1), 2),
                 query_mask=np.ones((size, 1), bool), query_anchor=z(size, 1), query_anchor_known=m(size, 1),
                 context=z(size, 128), context_known=m(size), assay=np.full(size, 8 if taxon == 9606 else 9),
                 taxon=np.full(size, 0 if taxon == 9606 else 1))
        for row, genes in enumerate(interventions):
            for col, gene in enumerate(genes):
                key = f'{taxon}:{gene}'
                index = self.lookup.get(key, 0)
                if not index and key not in (descriptors or {}):
                    raise ValueError('Supply a normalized descriptor for unknown entity: ' + key)
                b['action_ids'][row, col] = index
                b['action_features'][row, col] = self.descriptors[index] if index else descriptors[key]
                b['action_mask'][row, col] = True
        if taxon == 9606:
            if context_ids is None or len(context_ids) != size:
                raise ValueError('Supply one admitted DepMap context ID per human experiment')
            b['context'][:] = [self.contexts[c] for c in context_ids]
            b['context_known'][:] = True
        return self.predict(b)['mean'][:, 0]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True, help='NPZ with normalized observation/action/query arrays')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--sample', action='store_true')
    p.add_argument('--seed', type=int, default=731)
    args = p.parse_args()
    world = World(args.bundle, args.device)
    with np.load(args.input, allow_pickle=False) as z:
        result = world.predict(dict(z), sample=args.sample, seed=args.seed)
    with args.output.open('xb') as f:
        np.savez_compressed(f, **result)
