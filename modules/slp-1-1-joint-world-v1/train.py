"""Train a shared population-state model from explicit molecular snapshots."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from safetensors.torch import save_file
import torch
from threadpoolctl import threadpool_limits

from response_model import fit, save
from world_model import Config, SharedWorldModel


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def prepare_context(path, output, fold, rng, observation_queries, response_rank):
    with np.load(path, allow_pickle=False) as archive:
        keys = ['targets', 'basal', 'observed', 'action_features', 'action_mask',
                'query_features', 'query_ids', 'feature_mean', 'feature_scale']
        data = {key: np.asarray(archive[key]) for key in keys}
        source = path.stem
        if source == 'norman':
            for key in ('single_rows', 'combination_rows', 'combination_single_rows', 'combination_fold'):
                data[key] = np.asarray(archive[key])
        for key in ('mode_id','assay_id','control_context_values','control_context_observed'):
            if key in archive.files: data[key] = np.asarray(archive[key])
    n = len(data['targets'])
    data['fit_rows'] = np.arange(n)
    if source == 'norman':
        data['fit_rows'] = np.concatenate((data['single_rows'],
            data['combination_rows'][data['combination_fold'] != fold]))
        prior_rows = data['single_rows']
        parent = np.full((n, data['action_mask'].shape[1]), -1, dtype=np.int64)
        parent[data['combination_rows']] = data['combination_single_rows']
        data['parents'] = parent
    else:
        prior_rows = data['fit_rows']
        data['parents'] = np.full((n, data['action_mask'].shape[1]), -1, dtype=np.int64)

    residual = data['targets'][prior_rows] - data['basal'][prior_rows]
    prior = fit(data['action_features'][prior_rows, 0], residual,
                rank=32 if source == 'norman' else response_rank,
                alpha=100.0 if source == 'norman' else 1000.0)
    save(output/'priors'/f'{source}.npz', prior,
         query_ids=data['query_ids'], source_id=source)
    active_features = data['action_features'].reshape(-1, data['action_features'].shape[-1])
    per_action = prior.predict(active_features).reshape(n, data['action_mask'].shape[1], -1)
    data['prior_per_action'] = (per_action * data['action_mask'][..., None]).astype(np.float32)
    fit_values = data['targets'][data['fit_rows']] - data['basal'][data['fit_rows']]
    fit_mask = data['observed'][data['fit_rows']]
    scale = max(float(np.sqrt(np.mean(fit_values[fit_mask]**2))), 0.02)
    support = np.flatnonzero(fit_mask.all(0))
    if not len(support):
        raise ValueError(f'{source}: no common observed queries')
    obs_indices = np.sort(rng.choice(support, min(observation_queries, len(support)), replace=False))
    if 'control_context_values' in data:
        context_values=np.asarray(data['control_context_values'],np.float32); context_mask=np.asarray(data['control_context_observed'],np.bool_)
    elif source in ('k562','rpe1'):
        context_values=np.asarray(data['basal'][data['fit_rows']].mean(0)/np.log(2.),np.float32); context_mask=fit_mask.all(0)
    else:
        context_values=np.zeros(len(data['query_ids']),np.float32); context_mask=np.zeros(len(data['query_ids']),bool)
    np.savez_compressed(output/'adapters'/f'{source}.npz',
        query_ids=data['query_ids'], query_features=data['query_features'],
        observed_query_mask=fit_mask.all(0), observation_indices=obs_indices,
        control_context_values=context_values, control_context_mask=context_mask)
    data['support'] = support
    data['response_scale'] = scale
    data['mode'] = int(data.get('mode_id', int(source == 'norman')))
    data['assay'] = int(data.get('assay_id', int(source == 'norman')))
    if 'control_context_values' not in data and source in ('k562','rpe1'):
        data['control_context_values']=np.asarray(data['basal']/np.log(2.),np.float32); data['control_context_mask']=np.asarray(data['observed'],np.bool_)
    elif context_values.ndim==1: data['control_context_values']=np.broadcast_to(context_values,(n,len(context_values))).copy(); data['control_context_mask']=np.broadcast_to(context_mask,(n,len(context_mask))).copy()
    else: data['control_context_values']=context_values; data['control_context_mask']=context_mask
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=731)
    parser.add_argument('--norman-fold', type=int, choices=range(3), default=0)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--observation-queries', type=int, default=256)
    parser.add_argument('--decode-queries', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--reconstruction-weight', type=float, default=0.05)
    parser.add_argument('--response-rank', type=int, default=16)
    parser.add_argument('--residual-penalty', type=float, default=1.0)
    parser.add_argument('--max-seconds', type=int, default=2700)
    parser.add_argument('--sources', default='k562,rpe1,norman,gwps,hepg2')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Native CUDA is required; no executor fallback')
    if args.steps <= 0 or args.save_every <= 0:
        raise ValueError('steps and save-every must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('priors', 'adapters', 'checkpoints'):
        (args.output/name).mkdir()
    # Freeze executable source before fitting, including for intermediate checkpoints.
    for name in ('world_model.py', 'response_model.py', 'inference.py', 'train.py',
                 'evaluate.py', 'CONTRACT.md', 'requirements-linux.lock'):
        path = Path(__file__).with_name(name)
        if path.is_file():
            shutil.copyfile(path, args.output/name)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    with threadpool_limits(2):
        source_names=tuple(x.strip() for x in args.sources.split(',') if x.strip())
        if (set(source_names) not in ({'k562','rpe1','norman'},{'k562','rpe1','norman','gwps','hepg2'})):
            raise ValueError('sources must select the original three or expanded five contexts')
        contexts = {name: prepare_context(args.data/f'{name}.npz', args.output,
            args.norman_fold, rng, args.observation_queries, args.response_rank)
            for name in source_names}
    normalizer = contexts['k562']
    mean, scale = normalizer['feature_mean'], normalizer['feature_scale']
    for data in contexts.values():
        np.testing.assert_array_equal(data['feature_mean'], mean)
        np.testing.assert_array_equal(data['feature_scale'], scale)
    np.savez_compressed(args.output/'normalizer.npz', feature_mean=mean, feature_scale=scale)
    config = Config(feature_dim=len(mean), width=args.width, control_context=True,
                    assay_count=max(x['assay'] for x in contexts.values())+1)
    model = SharedWorldModel(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.001)
    metadata = {'schema': 'slp.joint-population-world/v1', 'config': asdict(config),
        'observation_queries': args.observation_queries, 'seed': args.seed,
        'contexts': {name: {key: data[key] for key in ('mode', 'assay', 'response_scale')}
                     for name, data in contexts.items()},
        'training': {'norman_fold': args.norman_fold, 'steps_requested': args.steps,
            'batch_size': args.batch_size, 'decode_queries': args.decode_queries,
            'learning_rate': args.learning_rate, 'reconstruction_weight': args.reconstruction_weight,
            'response_rank': args.response_rank, 'residual_penalty': args.residual_penalty,
            'residual_penalty_crispra_multiplier': 0.1,
            'source_probabilities': ({'k562':.25,'rpe1':.25,'norman':.15,'gwps':.25,'hepg2':.10} if len(contexts)==5 else {'k562':.4,'rpe1':.4,'norman':.2}),
            'objective': 'population endpoint residual MSE plus masked observed-state reconstruction',
            'target_uncertainty': 'not estimated; no single-cell likelihood claim'},
        'inputs': {name: {'sha256': digest(args.data/f'{name}.npz'),
            'fitting_populations': len(data['fit_rows']), 'queries': len(data['query_ids'])}
            for name, data in contexts.items()},
        'runtime': {'torch': torch.__version__, 'numpy': np.__version__,
            'device': torch.cuda.get_device_name(0), 'parameters': sum(p.numel() for p in model.parameters())}}
    write_json(args.output/'config.json', metadata)
    if (args.data/'manifest.json').is_file():
        shutil.copyfile(args.data/'manifest.json', args.output/'data-manifest.json')
    for data in contexts.values():
        data['action_features'] = ((data['action_features']-mean)/scale).astype(np.float32)
        data['action_features'][~data['action_mask']] = 0
        data['query_features'] = ((data['query_features']-mean)/scale).astype(np.float32)
        for key in ('targets', 'basal', 'action_features', 'query_features', 'prior_per_action','control_context_values'):
            data[key] = torch.tensor(data[key], dtype=torch.float32, device='cuda')
        data['action_mask'] = torch.tensor(data['action_mask'], dtype=torch.bool, device='cuda')
        data['control_context_mask'] = torch.tensor(data['control_context_mask'], dtype=torch.bool, device='cuda')
    save_file(model.state_dict(), str(args.output/'checkpoints/step-000000.safetensors'))
    losses = []
    source_names = tuple(contexts)
    source_weights=metadata['training']['source_probabilities']; probabilities=np.asarray([source_weights[x] for x in source_names]);probabilities=probabilities/probabilities.sum()
    model.train()
    for step in range(1, args.steps+1):
        source = source_names[int(rng.choice(len(source_names), p=probabilities))]
        data = contexts[source]
        rows = rng.choice(data['fit_rows'], args.batch_size, replace=True)
        obs_idx = rng.choice(data['support'], min(args.observation_queries, len(data['support'])), replace=False)
        out_idx = rng.choice(data['support'], min(args.decode_queries, len(data['support'])), replace=False)
        target = data['targets'][rows]
        basal = data['basal'][rows]
        observed = basal.clone()
        actions = data['action_features'][rows].clone()
        action_mask = data['action_mask'][rows].clone()
        prior = data['prior_per_action'][rows].sum(1)
        if source == 'norman':
            # Train control->single/double and measured-single->double with one shared operator.
            for position, row in enumerate(rows):
                parents = data['parents'][row]
                if parents[0] >= 0 and rng.random() < 0.67:
                    parent_side = int(rng.integers(2))
                    observed[position] = data['targets'][parents[parent_side]]
                    action_mask[position, parent_side] = False
                    actions[position, parent_side] = 0
                    prior[position] = data['prior_per_action'][row, 1-parent_side]
        mode = torch.full((len(rows),), data['mode'], dtype=torch.long, device='cuda')
        assay = torch.full_like(mode, data['assay'])
        mask = torch.ones((len(rows), len(obs_idx)), dtype=torch.bool, device='cuda')
        response_scale = data['response_scale']
        state = model.encode(observed[:, obs_idx]/response_scale, basal[:, obs_idx]/response_scale,
            data['query_features'][obs_idx], mask, mode, assay, data['control_context_values'][rows][:,obs_idx], data['control_context_mask'][rows][:,obs_idx])
        changed = model.transition(state, actions, action_mask, mode, assay)
        delta = model.decode(changed, data['query_features'][out_idx], assay) - model.decode(
            state, data['query_features'][out_idx], assay)
        target_delta = (target[:, out_idx]-observed[:, out_idx]-prior[:, out_idx])/response_scale
        forecast_loss = (delta-target_delta).square().mean()
        target_state = model.encode(target[:, obs_idx]/response_scale, basal[:, obs_idx]/response_scale,
            data['query_features'][obs_idx], mask, mode, assay, data['control_context_values'][rows][:,obs_idx], data['control_context_mask'][rows][:,obs_idx])
        reconstruction = model.decode(target_state, data['query_features'][out_idx], assay)
        reconstruction_loss = (reconstruction-(target[:, out_idx]-basal[:, out_idx])/response_scale).square().mean()
        penalty = args.residual_penalty * (0.1 if source == 'norman' else 1.0)
        loss = forecast_loss + args.reconstruction_weight*reconstruction_loss + penalty*delta.square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite molecular training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append({'step': step, 'source': source, 'forecast': float(forecast_loss.detach()),
                       'reconstruction': float(reconstruction_loss.detach())})
        if step % 100 == 0:
            print(json.dumps({'step': step, 'elapsed_seconds': round(time.monotonic()-started, 2),
                'forecast_loss_recent': float(np.mean([x['forecast'] for x in losses[-100:]])),
                'gpu_peak_mb': torch.cuda.max_memory_allocated()/1024**2}), flush=True)
        timed_out = time.monotonic()-started >= args.max_seconds
        if step % args.save_every == 0 or step == args.steps or timed_out:
            save_file(model.state_dict(), str(args.output/f'checkpoints/step-{step:06d}.safetensors'))
            write_json(args.output/'training-history.json', losses)
        if timed_out:
            break
    metadata['training'].update(steps_completed=step, elapsed_seconds=time.monotonic()-started)
    metadata['runtime']['gpu_peak_mb'] = torch.cuda.max_memory_allocated()/1024**2
    metadata['code'] = {path.name: digest(path) for path in args.output.glob('*.py')}
    write_json(args.output/'config.json', metadata)
    print(json.dumps({'completed': True, 'steps': step, 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
