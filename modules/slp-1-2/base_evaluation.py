"""Prepare immutable development panels and compare frozen SLp-1.2 bases.

Preparation alone reads the corpus. Scoring reuses the same persisted inputs,
outcomes, fitting-only baselines and wrong interventions across checkpoints.
This is retrospective development, never an independent test or promotion gate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import numpy as np
import torch
from data import Corpus, batch_template, to_device
from inference import World
from io_utils import load_npz, sha, write_json

SCHEMA = 'slp.base-evaluation/v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def array_digest(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode() + a.tobytes()).hexdigest()


def contract(corpus):
    return dict(receipts=corpus.receipts, vocabulary=digest(corpus.vocabulary),
                normalizers={k: array_digest(v) for k, v in dict(
                    human_context_mean=corpus.context_mean, human_context_scale=corpus.context_scale,
                    descriptor_mean=corpus.features.mean, descriptor_scale=corpus.features.scale).items()},
                held_human=digest(sorted(corpus.held_human)), held_yeast=digest(sorted(corpus.held_yeast)),
                molecular_scales={s.name: digest(np.asarray(s.scale).tolist()) for s in corpus.sources})


def validate_bundle(world, expected):
    corpus = json.loads((world.bundle / 'corpus.json').read_text())
    actual = dict(receipts=corpus['receipts'], vocabulary=digest(world.vocabulary),
                  normalizers={k: array_digest(v) for k, v in load_npz(world.bundle / 'normalizers.npz').items()},
                  held_human=digest(corpus['held_human_genes']), held_yeast=digest(corpus['held_yeast_genes']),
                  molecular_scales={k: digest(v) for k, v in corpus['molecular_scales'].items()})
    if actual != expected:
        raise ValueError('Panel/bundle vocabulary, exclusion, corpus or normalization contract differs')


def action_key(batch, row):
    return tuple(sorted(batch['action_ids'][row, batch['action_mask'][row]].tolist()))


def wrong_actions(batch, donors):
    """Swap identities only; same source, mechanisms, cardinality and conditions.

    Raw-cell batches share one action, so donors must span panels. Never score
    an unchanged action as a successful swap. Unavailable rows are explicit.
    """
    out = {k: v.copy() for k, v in batch.items() if isinstance(v, np.ndarray)}
    available = np.zeros(len(batch['action_ids']), bool)
    rng = np.random.default_rng(120908)
    choices = {}
    for donor in donors:
        for row in range(len(donor['action_ids'])):
            active = np.flatnonzero(donor['action_mask'][row])
            signature = tuple(donor['action_mechanism'][row, active].tolist())
            choices.setdefault(signature, {}).setdefault(action_key(donor, row), (donor, row, active))
    for row in range(len(available)):
        active = np.flatnonzero(batch['action_mask'][row])
        signature = tuple(batch['action_mechanism'][row, active].tolist())
        candidates = list(choices.get(signature, {}).items())
        if candidates:
            offset = int(rng.integers(len(candidates)))
            candidates = candidates[offset:] + candidates[:offset]
        for key, (donor, index, positions) in candidates:
            if key != action_key(batch, row) and len(active):
                for name in ('action_ids', 'action_features'):
                    out[name][row, active] = donor[name][index, positions]
                available[row] = True
                break
    return out, available


class FittingMeans:
    def __init__(self, size):
        self.count = np.zeros(size, np.int64)
        self.target = np.zeros(size, np.float64)
        self.control = np.zeros(size, np.float64)

    def add(self, batch):
        mask = batch['query_mask']
        ids = batch['query_ids'][mask]
        np.add.at(self.count, ids, 1)
        for name, values in (('target', batch['target']), ('control', batch['control_target'])):
            if not np.isfinite(values[mask]).all():
                raise ValueError('Nonfinite fitting observation')
            np.add.at(getattr(self, name), ids, values[mask])

    def predict(self, batch):
        ids = batch['query_ids']
        count = self.count[ids]
        reference = batch['control_anchor']
        return {**{name: np.where(count > 0, getattr(self, name)[ids] / count.clip(1), reference).astype(np.float32)
                   for name in ('target', 'control')}, 'count': count}


def fit_means(corpus, source, config):
    if corpus.validation:
        raise ValueError('Baselines must be fitted on the fitting corpus')
    means = FittingMeans(len(corpus.vocabulary))
    for _ in range(config['baseline_draws']):
        means.add(corpus.molecular(config['baseline_batch_size'], 1, config['baseline_queries'], source=source))
    return means


def human_batch(corpus, rows):
    gi, ci = rows.T
    h = corpus.human
    if not h['known'][gi, ci].all():
        raise ValueError('Unmeasured human outcome')
    genes = h['gene_indices'][gi]
    b = batch_template(len(rows), 1, 1, 1)
    b['action_features'][:, 0] = corpus.human_features[genes]
    b['action_ids'][:, 0] = corpus.human_indices[genes]
    b['action_mask'][:] = True
    b['action_mechanism'][:] = 2
    b['query_modality'][:] = 2
    b['context'] = h['contexts'][ci]
    b['context_known'][:] = True
    b['target'][:, 0] = h['targets'][gi, ci]
    b['assay'][:] = 8
    b['row_key'] = rows
    return b


def split_genes(genes, held, seed):
    if len(set(genes)) != len(genes) or not set(genes) <= set(held):
        raise ValueError('Transfer genes must be unique and excluded from base fitting')
    ranked = sorted(genes, key=lambda g: hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
    if len(ranked) < 2:
        raise ValueError('Need at least two held genes')
    return set(ranked[:len(ranked) // 2]), set(ranked[len(ranked) // 2:])


def save_arrays(path, batch):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **{k: v for k, v in batch.items() if isinstance(v, np.ndarray)})


def capture_source(output):
    folder = output / 'source'
    folder.mkdir()
    names = ('base_evaluation.py', 'transfer_probe.py', 'base-evaluation.json', 'data.py', 'legacy_data.py',
             'inference.py', 'model.py', 'io_utils.py', 'requirements.in')
    paths = [Path(__file__).with_name(name) for name in names]
    paths.extend(sorted(Path(__file__).parent.glob('requirements-*.lock')))
    for path in paths:
        shutil.copy2(path, folder / path.name)


def prepare(root, output, config):
    if config['schema'] != SCHEMA or any(config[k] < 1 for k in (
            'panels_per_source', 'batch_size', 'observations', 'queries', 'baseline_draws',
            'baseline_batch_size', 'baseline_queries', 'fitness_rows', 'probe_evaluation_rows')):
        raise ValueError('Invalid evaluation configuration')
    if not config['probe_budgets'] or any(n < 2 for n in config['probe_budgets']):
        raise ValueError('Probe budgets must contain at least two labels')
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    train = Corpus(root, seed=config['seed'])
    held = Corpus(root, validation=True, seed=config['seed'] + 1)
    manifest = dict(schema=SCHEMA, configuration=config, contract=contract(train), cases=[],
                    independent_test=False, selection_used_development=True,
                    scope='Retrospective development; shared intervention exclusions. Human results primary; yeast diagnostic.')
    for source in held.sources:
        print('Preparing', source.name, flush=True)
        means = fit_means(train, source.name, config)
        panels = [held.molecular(config['batch_size'], config['observations'], config['queries'], source=source.name)
                  for _ in range(config['panels_per_source'])]
        for index, raw in enumerate(panels):
            swapped, available = wrong_actions(raw, panels)
            stats = means.predict(raw)
            raw.update(baseline_training_mean=stats['target'], baseline_control_mean=stats['control'],
                       baseline_reference=raw['control_anchor'], baseline_count=stats['count'],
                       wrong_action_ids=swapped['action_ids'], wrong_action_features=swapped['action_features'],
                       wrong_available=available)
            name = f'{source.name}-{index:04d}.npz'
            save_arrays(output / name, raw)
            manifest['cases'].append(dict(file=name, source=source.name, taxon=source.taxon, cell=source.cell))
    # Exact source-fitting context mean: no held outcome enters this baseline.
    h = train.human
    context_mean = np.where(h['known'], h['targets'], 0).sum(0) / h['known'].sum(0).clip(1)
    if not h['known'].any(0).all() or not np.array_equal(h['context_ids'], held.human['context_ids']):
        raise ValueError('Human context coverage/order differs')
    rng = np.random.default_rng(config['seed'] + 2)
    known_rows = np.argwhere(held.human['known'])
    genes = held.human_ids[held.human['gene_indices']].tolist()
    adaptation, evaluation = split_genes(genes, held.held_human, config['seed'])
    manifest['probe_genes'] = dict(adaptation=sorted(adaptation), evaluation=sorted(evaluation))
    for role, allowed, size in (('adaptation', adaptation, max(config['probe_budgets'])),
                                 ('evaluation', evaluation, config['probe_evaluation_rows'])):
        rows = known_rows[np.isin(np.array(genes)[known_rows[:, 0]], sorted(allowed))]
        if size > len(rows):
            raise ValueError('Requested more unique probe rows than available')
        rows = rows[rng.choice(len(rows), size=size, replace=False)]
        b = human_batch(held, rows)
        b['baseline_training_mean'] = context_mean[rows[:, 1], None].astype(np.float32)
        save_arrays(output / f'probe-{role}.npz', b)
    # The zero-shot human panel is exactly the probe evaluation panel.
    raw = load_npz(output / 'probe-evaluation.npz')
    swapped, available = wrong_actions(raw, [raw])
    raw.update(baseline_reference=np.zeros_like(raw['target']), baseline_control_mean=np.zeros_like(raw['target']),
               baseline_count=np.ones_like(raw['target'], np.int64), wrong_action_ids=swapped['action_ids'],
               wrong_action_features=swapped['action_features'], wrong_available=available)
    save_arrays(output / 'human_fitness.npz', raw)
    manifest['cases'].append(dict(file='human_fitness.npz', source='human_fitness', taxon=9606, cell=False))
    raw = held.yeast_fitness(config['fitness_rows'])
    fit = np.log(np.stack([train.yeast[k] for k in ('single_a', 'single_b', 'double')], 1).clip(.05))
    mean = float(fit.mean(0) @ np.array([.2, .2, .6]))
    swapped, available = wrong_actions(raw, [raw])
    raw.update(baseline_training_mean=np.full_like(raw['target'], mean), baseline_reference=np.zeros_like(raw['target']),
               baseline_control_mean=np.zeros_like(raw['target']), baseline_count=np.ones_like(raw['target'], np.int64),
               wrong_action_ids=swapped['action_ids'], wrong_action_features=swapped['action_features'], wrong_available=available)
    save_arrays(output / 'yeast_fitness.npz', raw)
    manifest['cases'].append(dict(file='yeast_fitness.npz', source='yeast_fitness', taxon=4932, cell=False))
    capture_source(output)
    manifest['elapsed_seconds'] = time.monotonic() - start
    manifest['files'] = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
    write_json(output / 'manifest.json', manifest)
    print('Prepared', len(manifest['cases']), 'cases in', output, flush=True)


def load_panels(path):
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest['schema'] != SCHEMA:
        raise ValueError('Unsupported panel schema')
    for name, value in manifest['files'].items():
        p = (path / name).resolve()
        if not p.is_relative_to(path.resolve()) or sha(p) != value:
            raise ValueError('Panel checksum/path mismatch: ' + name)
    return manifest


@torch.inference_mode()
def predict(model, raw, device, batch_size=64, features=False):
    model.eval()
    means, hidden = [], []
    for start in range(0, len(raw['target']), batch_size):
        batch = {k: v[start:start + batch_size] for k, v in raw.items() if isinstance(v, np.ndarray)}
        result = model(to_device(batch, device))
        means.append(result['mean'].cpu().numpy())
        if features:
            hidden.append(result['features'][:, 0].cpu().numpy())
    return np.concatenate(means), np.concatenate(hidden) if features else None


def score_case(raw, predictions, cell):
    records = []
    for modality, label in ((0, 'rna'), (1, 'protein'), (2, 'fitness')):
        mask = raw['query_mask'] & (raw['query_modality'] == modality)
        for row in range(len(mask)):
            keep = mask[row]
            if not keep.any():
                continue
            values = {name: float(np.square(pred[row, keep] - raw['target'][row, keep]).mean())
                      for name, pred in predictions.items() if name != 'wrong_action' or raw['wrong_available'][row]}
            records.append(dict(modality=label, action=list(action_key(raw, row)), mse=values,
                                baseline_coverage=float((raw['baseline_count'][row, keep] > 0).mean())))
        if cell and mask.any():
            if len({action_key(raw, i) for i in range(len(mask))}) != 1 or not np.all(raw['query_ids'] == raw['query_ids'][0]):
                raise ValueError('Pseudobulk requires a single intervention and shared coordinates')
            keep = mask.all(0)
            if keep.any():
                values = {name: float(np.square(pred[:, keep].mean(0) - raw['target'][:, keep].mean(0)).mean())
                          for name, pred in predictions.items() if name != 'wrong_action' or raw['wrong_available'].all()}
                records.append(dict(modality=label + '_pseudobulk', action=list(action_key(raw, 0)), mse=values,
                                    baseline_coverage=float((raw['baseline_count'][:, keep] > 0).mean())))
    return records


def summarize(records):
    metrics = {}
    for key in sorted({r['source'] + ':' + r['modality'] for r in records}):
        rows = [r for r in records if r['source'] + ':' + r['modality'] == key]
        names = sorted({name for r in rows for name in r['mse']})
        mse = {name: float(np.mean([r['mse'][name] for r in rows if name in r['mse']])) for name in names}
        matched = [r for r in rows if 'wrong_action' in r['mse']]
        metrics[key] = dict(mse=mse, rows=len(rows), unique_action_sets=len({tuple(r['action']) for r in rows}),
                            baseline_coverage=float(np.mean([r['baseline_coverage'] for r in rows])),
                            wrong_action_rows=len(matched),
                            wrong_minus_correct_mse=float(np.mean([r['mse']['wrong_action'] - r['mse']['model'] for r in matched])) if matched else None,
                            improvement_over_training_mean=1 - mse['model'] / mse['training_mean'] if mse['training_mean'] else None)
    return metrics


def score(bundle, panels, output, device):
    start = time.monotonic()
    manifest = load_panels(panels)
    world = World(bundle, device)
    validate_bundle(world, manifest['contract'])
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for case in manifest['cases']:
        print('Scoring', case['file'], flush=True)
        raw = load_npz(panels / case['file'])
        values = {name: raw['baseline_' + name] for name in ('training_mean', 'control_mean', 'reference')}
        values['model'] = predict(world.model, raw, device)[0]
        wrong = dict(raw, action_ids=raw['wrong_action_ids'], action_features=raw['wrong_action_features'])
        values['wrong_action'] = predict(world.model, wrong, device)[0]
        masked = dict(raw, action_mask=np.zeros_like(raw['action_mask']))
        values['action_masked'] = predict(world.model, masked, device)[0]
        if any(not np.isfinite(v[raw['query_mask']]).all() for v in values.values()):
            raise ValueError('Nonfinite scored prediction')
        save_arrays(output / case['file'], values)
        records.extend(dict(source=case['source'], taxon=case['taxon'], case=case['file'], **r)
                       for r in score_case(raw, values, case['cell']))
    report = dict(schema=SCHEMA, phase='frozen_base', independent_test=False,
                  weights_sha256=sha(bundle / 'model.safetensors'), panels_sha256=sha(panels / 'manifest.json'),
                  metrics=summarize(records), records=records, elapsed_seconds=time.monotonic() - start,
                  environment=dict(torch=torch.__version__, numpy=np.__version__, python=platform.python_version(), device=device),
                  scope='Sampled retrospective development; source-native normalized units. Row-weighted means; cell pseudobulks within source/context panels. No independent-test or scaling claim.',
                  baseline_scope='Molecular source-pooled empirical means fitted on training draws; human exact training-context mean; yeast fitting-mixture mean. Missing molecular coordinates use the provided control reference and report coverage.')
    capture_source(output)
    report['files'] = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
    write_json(output / 'report.json', report)
    print(json.dumps(report['metrics'], indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--root', type=Path, required=True)
    a.add_argument('--config', type=Path, default=Path(__file__).with_name('base-evaluation.json'))
    a.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('score')
    a.add_argument('--bundle', type=Path, required=True)
    a.add_argument('--panels', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--device', default='cpu')
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.command == 'prepare':
        prepare(args.root, args.output, json.loads(args.config.read_text()))
    else:
        score(args.bundle, args.panels, args.output, args.device)
