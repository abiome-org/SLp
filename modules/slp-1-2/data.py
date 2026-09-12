"""Unified molecular / fitness batches over explicitly admitted SLp data.

The release contains legacy simulation signatures; this loader ignores them.
Shared intervention exclusions are applied before any outcome is sampled.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import numpy as np
import torch

from io_utils import load_npz, sha, write_json
from legacy_data import Features, Population, RawCells, PairedCells, Yeast

FITNESS_DTYPE = np.dtype([('a', '<u2'), ('b', '<u2'), ('single_a', '<f4'), ('single_b', '<f4'), ('double', '<f4')])


def batch_template(b, observations, actions, queries, feature_dim=702):
    z = lambda *shape: np.zeros(shape, np.float32)
    i = lambda *shape: np.zeros(shape, np.int64)
    m = lambda *shape: np.zeros(shape, bool)
    return dict(observation_values=z(b, observations), observation_features=z(b, observations, feature_dim),
                observation_ids=i(b, observations), observation_modality=i(b, observations), observation_mask=m(b, observations),
                action_features=z(b, actions, feature_dim), action_ids=i(b, actions), action_mask=m(b, actions),
                action_mechanism=i(b, actions), action_values=z(b, actions, 3), action_known=m(b, actions, 3),
                query_features=z(b, queries, feature_dim), query_ids=i(b, queries), query_modality=i(b, queries),
                query_mask=np.ones((b, queries), bool), query_anchor=z(b, queries), query_anchor_known=m(b, queries),
                context=z(b, 128), context_known=m(b), assay=i(b), taxon=i(b), target=z(b, queries),
                scale=np.ones((b, queries), np.float32), cell=False)


def to_device(batch, device):
    return {key: torch.as_tensor(np.array(value, copy=True), device=device) if isinstance(value, np.ndarray) else value
            for key, value in batch.items()}


def apply_controls(batch, selected):
    """Use observed control coordinates or the assay's neutral fitness reference.

    For cells, the observation panel and target panel come from the SAME real
    control cell. Query anchors use the population control, so the requested
    target coordinates are not disclosed as their own anchors.
    """
    batch['action_mask'][selected] = False
    if batch['task'] == 'molecular':
        batch['target'][selected] = batch['control_target'][selected]
        batch['query_anchor'][selected] = batch['control_anchor'][selected]
        for row in np.flatnonzero(selected):
            # The paired-cell reader sometimes includes protein channels in
            # both panels. Hide overlap when reconstructing this control cell.
            overlap = np.isin(batch['observation_ids'][row], batch['query_ids'][row])
            batch['observation_mask'][row] &= ~overlap
    else:
        batch['target'][selected] = 0.
    batch['is_control'] = selected
    batch.pop('control_target', None)
    batch.pop('control_anchor', None)
    return batch


def partition_source(source, held, validation):
    """Partition every quantitative row by intervention, including both pair members."""
    if isinstance(source, RawCells):
        source.groups = {g: rows for g, rows in source.groups.items() if (g in held) == validation}
        source.genes = sorted(source.groups)
        return bool(source.genes)
    if isinstance(source, PairedCells):
        source.allowed = {g for g in source.allowed if (g in held) == validation}
        return bool(source.allowed)
    ids = source.ids if isinstance(source, Population) else source.z['action_ids'][:, None]
    nonempty = ids != ''
    is_held = np.isin(ids, list(held)) & nonempty
    if isinstance(source, Population):
        # Fit the scale on the final shared training split, then use exactly
        # that scale in development. Legacy indices predate the larger holdout.
        train = ~is_held.any(1)
        residual = (source.target - source.basal)[train]
        measured = source.observed[train]
        if not measured.any():
            return False
        source.scale = float(max(np.sqrt(np.square(residual[measured]).mean()), .03))
    keep = ((is_held | ~nonempty).all(1) & nonempty.any(1)) if validation else ~is_held.any(1)
    if not keep.any():
        return False
    if isinstance(source, Population):
        source.ids = source.ids[keep]
        source.target = source.target[keep]
        source.basal = source.basal[keep]
        source.observed = source.observed[keep]
        # 1.2 learns simultaneous endpoints directly; no synthetic parent trajectories.
        source.parent_rows = np.full((int(keep.sum()), 2), -1, np.int64)
    else:
        source.eligible_rows = np.flatnonzero(keep)
    return True


class Corpus:
    def __init__(self, root, *, validation=False, verify=True, seed=120731, weights=None, control_probability=0.):
        self.root = Path(root)
        self.validation = validation
        self.rng = np.random.default_rng(seed)
        self.draws = {}
        if not 0 <= control_probability <= 1 or (validation and control_probability):
            raise ValueError('Invalid control sampling probability or validation mixture')
        self.control_probability = control_probability
        self.control_examples = {}
        self.paired_state = {}
        self.paired_loaded = {}
        self.index = self.root / 'data/derived/slp11-cell-world-training-v5'
        self.fitness_root = self.root / 'data/derived/slp11-genomic-fitness-world-v1'
        self.manifest = json.loads((self.index / 'manifest.json').read_text())
        fit_manifest = json.loads((self.fitness_root / 'manifest.json').read_text())
        if verify:
            for name, digest in self.manifest['files'].items():
                if sha(self.index / name) != digest:
                    raise ValueError('Changed molecular index: ' + name)
            for name, digest in self.manifest['sourceReceipts'].items():
                if sha(self.root / name) != digest:
                    raise ValueError('Changed molecular input: ' + name)
            for name, digest in fit_manifest['files'].items():
                if sha(self.fitness_root / name) != digest:
                    raise ValueError('Changed fitness input: ' + name)
        self.features = Features(self.index / 'features.npz')
        # Load static gene features only. Never retain the 1.1 simulation signatures.
        with np.load(self.fitness_root / 'genes.npz', allow_pickle=False) as g:
            self.human_ids = g['human_ids'].astype(str)
            self.yeast_ids = g['yeast_ids'].astype(str)
            self.human_features = np.pad(g['human_raw'], ((0, 0), (0, 60))).astype(np.float32)
            self.yeast_features = np.pad(g['yeast_raw'], ((0, 0), (0, 60))).astype(np.float32)
        human_validation = load_npz(self.fitness_root / 'human-validation.npz')
        human_training = load_npz(self.fitness_root / 'human-train.npz')
        self.held_human = set((self.index / 'excluded-human-action-ids.txt').read_text().splitlines())
        self.held_human.update(self.human_ids[human_validation['gene_indices']])
        yp = json.loads((self.fitness_root / 'yeast-partitions.json').read_text())
        self.held_yeast = set(yp['held_gene_ids']) | set(yp.get('previous_molecular_held_ids', []))
        self.human = human_validation if validation else human_training
        allowed = np.array([g not in self.held_human for g in self.human_ids[self.human['gene_indices']]])
        if validation:
            allowed = ~allowed
        for name in ('gene_indices', 'targets', 'known'):
            self.human[name] = self.human[name][allowed]
        self.context_mean = human_training['contexts'].mean(0)
        self.context_scale = human_training['contexts'].std(0).clip(.01)
        self.human['contexts'] = ((self.human['contexts'] - self.context_mean) / self.context_scale).astype(np.float32)
        role = 'validation' if validation else 'train'
        self.yeast = np.fromfile(self.fitness_root / f'yeast-{role}.bin', dtype=FITNESS_DTYPE)
        for col in ('a', 'b'):
            seen = set(self.yeast_ids[self.yeast[col]])
            if validation and not seen <= self.held_yeast:
                raise ValueError('Validation yeast intervention outside held roster')
            if not validation and seen & self.held_yeast:
                raise ValueError('Fitting yeast intervention is held')

        self.vocabulary = ['<unknown>'] + sorted(
            {f'9606:{g}' for g in self.features.human} | {f'4932:{g}' for g in self.features.yeast})
        fr = load_npz(self.index / self.manifest['frangieh']['population_file'])
        self.vocabulary += [f'protein:{g}' for g in fr['protein_channel_ids'].astype(str)]
        self.lookup = {name: i for i, name in enumerate(self.vocabulary)}
        self.human_indices = self.ids(self.human_ids, 9606)
        self.yeast_indices = self.ids(self.yeast_ids, 4932)

        records = [(Population, r) for r in self.manifest['populations']]
        records += [(RawCells, r) for r in self.manifest['raw_cells']]
        records += [(PairedCells, self.manifest['frangieh']), (Yeast, self.manifest['yeast'])]
        self.sources = []
        for cls, record in records:
            source = cls(self.root, record, self.features) if cls is Yeast else cls(self.root, self.index, record, self.features)
            held = self.held_human if source.taxon == 9606 else self.held_yeast
            if partition_source(source, held, validation):
                self.sources.append(source)
        self.source_probabilities = np.array([s.weight for s in self.sources], np.float64)
        self.source_probabilities /= self.source_probabilities.sum()
        weights = weights or {'molecular': .6, 'human_fitness': .2, 'yeast_fitness': .2}
        self.tasks = list(weights)
        self.task_probabilities = np.array(list(weights.values()), np.float64)
        if any(k not in {'molecular', 'human_fitness', 'yeast_fitness'} for k in self.tasks) or np.any(self.task_probabilities <= 0):
            raise ValueError('Invalid source mixture')
        self.task_probabilities /= self.task_probabilities.sum()
        self.receipts = {'molecular_index': sha(self.index / 'manifest.json'),
                         'fitness_index': sha(self.fitness_root / 'manifest.json')}

    def ids(self, ids, taxon, modality=None):
        ids = np.asarray(ids).astype(str)
        if modality is None:
            return np.array([self.lookup.get(f'{taxon}:{g}', 0) for g in ids.ravel()], np.int64).reshape(ids.shape)
        return np.array([self.lookup.get(f'protein:{g}' if m == 1 else f'{taxon}:{g}', 0)
                         for g, m in zip(ids, modality)], np.int64)

    def molecular(self, size, observations, queries, *, source=None):
        for _ in range(100):
            s = (self.sources[self.rng.choice(len(self.sources), p=self.source_probabilities)] if source is None
                 else next(s for s in self.sources if s.name == source))
            if isinstance(s, PairedCells):
                # Amortize compressed-shard I/O while explicitly checkpointing
                # its index and remaining uses. Loading a known index uses no RNG.
                state = self.paired_state.get(s.name, {'remaining': 0})
                if state['remaining'] == 0:
                    state = {'index': int(self.rng.integers(len(s.manifest['shards']))), 'remaining': 48}
                    self.paired_state[s.name] = state
                if self.paired_loaded.get(s.name) != state['index']:
                    s.shard(self.rng, index=state['index'])
                    self.paired_loaded[s.name] = state['index']
                if not s.group_keys:
                    state['remaining'] = 0
                    continue
                state['remaining'] -= 1
                # Suppress the legacy reader's separate periodic shard choice.
                s.draws = 1
            old = s.draw(self.rng, size, observations, queries)
            break
        else:
            raise RuntimeError('No eligible molecular batch')
        e, d = old['encoder_indices'], old['decoder_indices']
        b = batch_template(size, len(e), old['actions'].shape[1], len(d))
        taxon = 9606 if old['taxon'][0] == 0 else 4932
        qi = self.ids(old['query_ids'], taxon, old['modality'])
        b['observation_values'] = old['initial'][:, e] if old['cell'] else old['control'][:, e]
        b['observation_features'][:] = old['query'][e]
        b['observation_ids'][:] = qi[e]
        b['observation_modality'][:] = old['modality'][e]
        b['observation_mask'] = old['observed'][:, e]
        b['action_features'][:, :, :642] = old['actions']
        b['action_ids'] = self.ids(old['action_ids'], taxon)
        b['action_mask'] = old['action_mask']
        b['action_mechanism'][:] = old['mechanism'][:, None]
        b['query_features'][:] = old['query'][d]
        b['query_ids'][:] = qi[d]
        b['query_modality'][:] = old['modality'][d]
        b['query_anchor'] = old['initial'][:, d]
        b['query_anchor_known'][:] = True
        b['query_mask'] = old['observed'][:, d]
        b['target'] = old['target'][:, d]
        b['control_target'] = old['initial'][:, d].copy()
        b['control_anchor'] = old['basal'][:, d].copy()
        b['scale'][:] = old['scale'][d]
        b['assay'], b['taxon'] = old['assay'], old['taxon']
        b['cell'], b['source'] = old['cell'], old['name']
        return b

    def human_fitness(self, size):
        h = self.human
        # Rejection samples only missing outcomes; no zero-filling labels.
        gi, ci = [], []
        while len(gi) < size:
            g = self.rng.integers(len(h['gene_indices']), size=size)
            c = self.rng.integers(len(h['contexts']), size=size)
            keep = h['known'][g, c]
            gi.extend(g[keep].tolist())
            ci.extend(c[keep].tolist())
        gi, ci = np.array(gi[:size]), np.array(ci[:size])
        genes = h['gene_indices'][gi]
        b = batch_template(size, 1, 1, 1)
        b['action_features'][:, 0] = self.human_features[genes]
        b['action_ids'][:, 0] = self.human_indices[genes]
        b['action_mask'][:] = True
        b['action_mechanism'][:] = 2
        b['query_modality'][:] = 2
        b['context'] = h['contexts'][ci]
        b['context_known'][:] = True
        b['target'][:, 0] = h['targets'][gi, ci]
        b['assay'][:] = 8
        b['source'] = 'human_fitness'
        return b

    def yeast_fitness(self, size, *, force_double=False):
        rows = self.yeast[self.rng.integers(len(self.yeast), size=size)]
        genes = np.stack((rows['a'], rows['b']), 1).astype(np.int64)
        b = batch_template(size, 1, 2, 1)
        b['action_features'] = self.yeast_features[genes]
        b['action_ids'] = self.yeast_indices[genes]
        b['action_mask'][:] = True
        b['action_mechanism'][:] = 2
        b['taxon'][:] = 1
        b['assay'][:] = 9
        b['query_modality'][:] = 2
        targets = np.log(np.stack((rows['single_a'], rows['single_b'], rows['double']), 1).clip(.05)).astype(np.float32)
        kind = np.full(size, 2) if force_double else self.rng.choice(3, size=size, p=[.2, .2, .6])
        b['action_mask'][kind == 0, 1] = False
        b['action_mask'][kind == 1, 0] = False
        b['target'][:, 0] = targets[np.arange(size), kind]
        b['source'] = 'yeast_fitness'
        b['observed_fitness_triplet'] = targets
        return b

    def draw(self, size, observations, queries, *, task=None, fitness_size=None):
        task = task or str(self.rng.choice(self.tasks, p=self.task_probabilities))
        if task != 'molecular' and fitness_size is not None:
            size = fitness_size
        if task == 'molecular':
            batch = self.molecular(size, observations, queries)
        elif task == 'human_fitness':
            batch = self.human_fitness(size)
        elif task == 'yeast_fitness':
            batch = self.yeast_fitness(size)
        else:
            raise ValueError(task)
        self.draws[batch['source']] = self.draws.get(batch['source'], 0) + size
        batch['task'] = task
        selected = self.rng.random(size) < self.control_probability if self.control_probability else np.zeros(size, bool)
        batch = apply_controls(batch, selected)
        self.control_examples[batch['source']] = self.control_examples.get(batch['source'], 0) + int(selected.sum())
        return batch

    def state_dict(self):
        return {'rng': copy.deepcopy(self.rng.bit_generator.state), 'draws': dict(self.draws),
                'paired_state': copy.deepcopy(self.paired_state), 'control_examples': dict(self.control_examples)}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state['rng']
        self.draws = dict(state['draws'])
        self.paired_state = copy.deepcopy(state.get('paired_state', {}))
        self.paired_loaded = {}
        self.control_examples = dict(state.get('control_examples', {}))

    def export_metadata(self, output):
        write_json(Path(output) / 'vocabulary.json', self.vocabulary)
        write_json(Path(output) / 'corpus.json', {
            'schema': 'slp.corpus/v1.2', 'receipts': self.receipts,
            'held_human_genes': sorted(self.held_human), 'held_yeast_genes': sorted(self.held_yeast),
            'validation': self.validation, 'molecular_sources': [s.name for s in self.sources],
            'control_probability': self.control_probability,
            'control_reference': 'Observed molecular controls; zero effect in normalized fitness units. Controls do not introduce held intervention outcomes.',
            'molecular_scales': {s.name: np.asarray(s.scale).tolist() for s in self.sources},
            'human_fitting_genes': len(self.human['gene_indices']), 'yeast_rows': len(self.yeast),
            'uses_sl_labels': False, 'uses_frozen_simulation_signatures': False,
            'scope': 'Existing species-native public molecular and fitness corpus, with shared intervention exclusions. Development is retrospective.'})
        np.savez_compressed(Path(output) / 'normalizers.npz',
                            human_context_mean=self.context_mean, human_context_scale=self.context_scale,
                            descriptor_mean=self.features.mean, descriptor_scale=self.features.scale)
        descriptors = np.zeros((len(self.vocabulary), 702), np.float32)
        for taxon, table in ((9606, self.features.human), (4932, self.features.yeast)):
            for gene, value in table.items():
                descriptors[self.lookup[f'{taxon}:{gene}'], :642] = value
        for source in self.sources:
            protein = source.modality == 1
            for name, value in zip(source.query_ids[protein], source.query[protein]):
                descriptors[self.lookup[f'protein:{name}']] = value
        np.savez_compressed(Path(output) / 'entities.npz', descriptors=descriptors)
        np.savez_compressed(Path(output) / 'contexts.npz', ids=self.human['context_ids'],
                            standardized=self.human['contexts'])
