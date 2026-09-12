"""Joint SLp-1.2 training on a GPU pod; torchrun supports multiple GPUs.

Checkpoints contain model, optimizer, per-rank sampling/RNG state and source
receipts. The wall-clock bound is independent of steps. No SL benchmark labels
are opened by this program.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from safetensors.torch import load_file, save_file

from data import Corpus, to_device
from io_utils import sha, write_json
from model import Config, WorldModel

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def autocast(device, precision):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and precision == 'bf16' else nullcontext()


def balanced_mean(error, batch):
    valid = batch['query_mask']
    # Each modality and each experimental example gets a vote. A 20-channel
    # protein assay must not disappear behind a 512-gene RNA panel.
    values = []
    for modality in (0, 1, 2):
        mask = valid & (batch['query_modality'] == modality)
        n = mask.sum(1)
        if n.any():
            values.append((torch.where(mask, error, 0.).sum(1) / n.clamp_min(1))[n > 0].mean())
    if not values:
        raise ValueError('No measured target coordinates')
    return torch.stack(values).mean()


def objective(model, batch, *, flow, noise_scale):
    if flow:
        target = torch.where(batch['query_mask'], batch['target'] - batch['query_anchor'], 0.)
        noise = torch.randn_like(target) * noise_scale
        t = torch.rand(len(target), device=target.device)
        noisy = (1 - t[:, None]) * noise + t[:, None] * target
        prediction = model(batch, noisy, t)
        loss = balanced_mean((prediction['velocity'] - (target - noise)).square(), batch)
        return loss, 'flow'
    prediction = model(batch)
    error = torch.where(batch['query_mask'], prediction['mean'] - batch['target'], 0.).square()
    loss = balanced_mean(error, batch)
    # The mean's target is always the measurement. The detached error only fits
    # an observation variance; it cannot reduce mean loss by inflating variance.
    variance = .5 * (error.detach() * torch.exp(-prediction['log_variance']) + prediction['log_variance'])
    return loss + .05 * balanced_mean(variance, batch), 'mean'


@torch.inference_mode()
def evaluate(model, corpus, config, device):
    model.eval()
    saved = corpus.state_dict()
    corpus.rng = np.random.default_rng(120999)
    metrics = {}
    per_task = []
    for task in ('molecular', 'human_fitness', 'yeast_fitness'):
        errors, bases = [], []
        panels = ([source.name for source in corpus.sources] if task == 'molecular' else [task])
        for panel in [name for name in panels for _ in range(config['evaluation_batches'])]:
            raw = (corpus.molecular(config['micro_batch'], config['observation_queries'], config['output_queries'], source=panel)
                   if task == 'molecular' else corpus.draw(config['micro_batch'], config['observation_queries'], config['output_queries'],
                                                          task=task, fitness_size=config.get('fitness_batch', config['micro_batch'])))
            batch = to_device(raw, device)
            with autocast(device, config['precision']):
                output = model(batch)['mean']
            error = float(balanced_mean((output - batch['target']).square(), batch))
            baseline = float(balanced_mean((batch['query_anchor'] - batch['target']).square(), batch))
            errors.append(error)
            bases.append(baseline)
            key = raw['source']
            metrics.setdefault(key, {'mse_sum': 0., 'unchanged_mse_sum': 0., 'batches': 0})
            metrics[key]['mse_sum'] += error
            metrics[key]['unchanged_mse_sum'] += baseline
            metrics[key]['batches'] += 1
            if config.get('evaluate_action_ablation', False):
                batch['action_mask'][:] = False
                with autocast(device, config['precision']):
                    ablated = model(batch)['mean']
                metrics[key]['action_ablated_mse_sum'] = metrics[key].get('action_ablated_mse_sum', 0.) + float(
                    balanced_mean((ablated - batch['target']).square(), batch))
        per_task.append(float(np.mean(errors)))
    # A measured-single additive comparator isolates nonadditivity in native
    # yeast log-fitness. This is explicitly distinct from an intervention-cold
    # deployable comparator, which would have to predict both singles.
    residual_errors, additive_errors, predictions, observations = [], [], [], []
    for _ in range(config['evaluation_batches']):
        raw = corpus.yeast_fitness(config.get('fitness_batch', config['micro_batch']), force_double=True)
        batch = to_device(raw, device)
        target = batch['observed_fitness_triplet']
        estimates = []
        for action in (0, 1, 2):
            batch['action_mask'][:] = True
            if action < 2:
                batch['action_mask'][:, 1 - action] = False
            with autocast(device, config['precision']):
                estimates.append(model(batch)['mean'][:, 0])
        predicted = estimates[2] - estimates[0] - estimates[1]
        observed = target[:, 2] - target[:, 0] - target[:, 1]
        residual_errors.extend((predicted - observed).square().cpu().tolist())
        additive_errors.extend(observed.square().cpu().tolist())
        predictions.extend(predicted.cpu().tolist())
        observations.extend(observed.cpu().tolist())
    for record in metrics.values():
        record['mse'] = record.pop('mse_sum') / record['batches']
        record['unchanged_mse'] = record.pop('unchanged_mse_sum') / record['batches']
        if 'action_ablated_mse_sum' in record:
            record['action_ablated_mse'] = record.pop('action_ablated_mse_sum') / record['batches']
    p, y = np.array(predictions), np.array(observations)
    correlation = float(np.corrcoef(p, y)[0, 1]) if p.std() > 1e-12 and y.std() > 1e-12 else 0.
    corpus.load_state_dict(saved)
    return {'selection_mse': float(np.mean(per_task)), 'source_metrics': metrics,
            'yeast_interaction': {'n': len(y), 'residual_mse': float(np.mean(residual_errors)),
                                  'measured_single_additive_mse': float(np.mean(additive_errors)),
                                  'correlation': correlation},
            'scope': 'Fixed sampled retrospective development panel; no independent or full-corpus claim.'}


def rank_state(corpus, device):
    return {'corpus': corpus.state_dict(), 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state(device).cpu() if device.type == 'cuda' else None,
            'numpy_rng': np.random.get_state(), 'python_rng': random.getstate()}


def checkpoint(output, model, optimizer, corpus, step, best, elapsed, world_size, rank, device, report=None):
    state = rank_state(corpus, device)
    all_states = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(all_states, state)
    else:
        all_states[0] = state
    if rank == 0:
        folder = output / f'checkpoint-{step:07d}'
        temporary = output / f'.checkpoint-{step:07d}.tmp'
        temporary.mkdir(exist_ok=False)
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(temporary / 'model.safetensors'))
        torch.save({'optimizer': optimizer.state_dict(), 'step': step, 'best': best, 'elapsed': elapsed,
                    'world_size': world_size, 'ranks': all_states, 'corpus_receipts': corpus.receipts}, temporary / 'resume.pt')
        write_json(temporary / 'config.json', model.configuration())
        if report is not None:
            write_json(temporary / 'development.json', report)
        write_json(temporary / 'manifest.json', {p.name: sha(p) for p in temporary.iterdir() if p.is_file()})
        temporary.rename(folder)
        write_json(output / 'latest.json', {'path': folder.name, 'step': step, 'elapsed_seconds': elapsed})
        # Bounded disk: retain two resumable checkpoints. Best inference weights
        # are stored separately and never depend on a subsequently pruned folder.
        previous = sorted(output.glob('checkpoint-*'))
        for old in previous[:-2]:
            shutil.rmtree(old)
    if world_size > 1:
        dist.barrier()


def main(args):
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
    if device.type != 'cuda' and not args.allow_cpu:
        raise RuntimeError('Main training requires CUDA. --allow-cpu is for numerical verification only.')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if world_size > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    torch.set_num_threads(4)
    config = json.loads(args.config.read_text())
    t = config['training']
    if args.steps is not None:
        t['steps'] = args.steps
    if args.max_hours is not None:
        t['max_hours'] = args.max_hours
    if t['max_hours'] <= 0 or t['steps'] < 1 or t['accumulation_steps'] < 1:
        raise ValueError('Invalid run limits')
    seed = int(t['seed']) + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=bool(args.resume))
    if world_size > 1:
        dist.barrier()
    print(json.dumps({'event': 'loading_corpus', 'rank': rank}), flush=True)
    corpus = Corpus(args.root, seed=seed, weights=t['source_weights'], control_probability=t.get('control_probability', 0.))
    validation = Corpus(args.root, validation=True, verify=False, seed=120999) if rank == 0 else None
    config['model']['gene_count'] = len(corpus.vocabulary) - 1
    model = WorldModel(Config(**config['model'])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t['learning_rate'], betas=(.9, .95),
                                 weight_decay=t['weight_decay'], fused=device.type == 'cuda')
    first, best, previous_elapsed = 0, float('inf'), 0.
    if args.resume:
        for name, digest in json.loads((args.resume / 'manifest.json').read_text()).items():
            path = (args.resume / name).resolve()
            if not path.is_relative_to(args.resume.resolve()) or sha(path) != digest:
                raise ValueError('Checkpoint checksum/path mismatch: ' + name)
        resume = torch.load(args.resume / 'resume.pt', map_location='cpu', weights_only=False)
        if resume['world_size'] != world_size or resume['corpus_receipts'] != corpus.receipts:
            raise ValueError('Resume requires identical rank count and corpus receipts')
        saved_config = json.loads((args.resume / 'config.json').read_text())
        if saved_config != model.configuration():
            raise ValueError('Resume model configuration differs')
        model.load_state_dict(load_file(str(args.resume / 'model.safetensors')))
        optimizer.load_state_dict(resume['optimizer'])
        first, best, previous_elapsed = resume['step'], resume['best'], resume['elapsed']
        state = resume['ranks'][rank]
        corpus.load_state_dict(state['corpus'])
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(state['cuda_rng'], device)
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
    if rank == 0:
        corpus.export_metadata(args.output)
        for name, path in [('LICENSE', args.root / 'LICENSE'),
                           ('THIRD_PARTY_NOTICES.md', args.root / 'release/THIRD_PARTY_NOTICES.md')]:
            shutil.copy2(path, args.output / name)
        captured = args.output / ('source' if not args.resume else f'source-resume-{first}')
        captured.mkdir(exist_ok=False)
        for path in Path(__file__).parent.iterdir():
            if path.is_file() and path.suffix in {'.py', '.json', '.lock', '.in', '.md'}:
                shutil.copy2(path, captured / path.name)
        write_json(args.output / ('run.json' if not args.resume else f'run-resume-{first}.json'), {'configuration': config, 'world_size': world_size,
                   'parameters': sum(p.numel() for p in model.parameters()), 'torch': torch.__version__,
                   'python': sys.version, 'numpy': np.__version__, 'cuda_runtime': torch.version.cuda,
                   'executor': {'kind': 'native_pytorch', 'runpod_pod_id': os.environ.get('RUNPOD_POD_ID')},
                   'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                   'source_sha256': {p.name: sha(p) for p in captured.iterdir()},
                   'corpus_receipts': corpus.receipts, 'resumed_from': str(args.resume) if args.resume else None,
                   'training_targets': ['molecular endpoints', 'human single fitness', 'yeast single/double fitness']})
    if t.get('compile'):
        model_forward = torch.compile(model, mode=t.get('compile_mode', 'default'))
    else:
        model_forward = model
    if world_size > 1:
        model_forward = DistributedDataParallel(model_forward, device_ids=[local_rank] if device.type == 'cuda' else None)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    start = time.monotonic()
    deadline = start + t['max_hours'] * 3600
    log = (args.output / 'training.jsonl').open('a', buffering=1) if rank == 0 else None
    step, last_checkpoint, report = first, first, None
    window = []
    try:
        if rank == 0 and not args.resume:
            report = evaluate(model, validation, t, device)
            write_json(args.output / 'untrained-development.json', report)
            print(json.dumps({'event': 'untrained_development', **report}), flush=True)
        if world_size > 1:
            dist.barrier()
        for step in range(first + 1, t['steps'] + 1):
            model_forward.train()
            lr = t['learning_rate'] * min(step / t['warmup_steps'], 1.)
            progress = max(0., (step - t['warmup_steps']) / max(1, t['steps'] - t['warmup_steps']))
            lr *= t['min_learning_rate_ratio'] + (1 - t['min_learning_rate_ratio']) * .5 * (1 + math.cos(math.pi * min(progress, 1.)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.
            for micro in range(t['accumulation_steps']):
                raw = corpus.draw(t['micro_batch'], t['observation_queries'], t['output_queries'],
                                  fitness_size=t.get('fitness_batch', t['micro_batch']))
                batch = to_device(raw, device)
                flow = bool(raw['cell'] and corpus.rng.random() < t['flow_probability'])
                sync = model_forward.no_sync() if world_size > 1 and micro + 1 < t['accumulation_steps'] else nullcontext()
                with sync, autocast(device, t['precision']):
                    loss, mode = objective(model_forward, batch, flow=flow, noise_scale=t['flow_noise_scale'])
                    scaled = loss / t['accumulation_steps']
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at {step} in {raw["source"]}')
                scaled.backward()
                loss_sum += float(loss.detach()) / t['accumulation_steps']
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), t['gradient_clip'])
            if not torch.isfinite(grad):
                raise FloatingPointError(f'Nonfinite gradient at {step}')
            optimizer.step()
            window.append(loss_sum)
            elapsed = time.monotonic() - start
            if rank == 0 and (step % t['log_every'] == 0 or step == first + 1):
                record = {'event': 'training', 'step': step, 'elapsed_seconds': elapsed,
                          'total_training_seconds': previous_elapsed + elapsed, 'loss': float(np.mean(window)),
                          'learning_rate': lr, 'gradient_norm': float(grad),
                          'optimizer_steps_per_second': (step - first) / max(elapsed, 1e-6),
                          'source_draws_rank0': corpus.draws,
                          'control_examples_rank0': corpus.control_examples,
                          'peak_cuda_mib': torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == 'cuda' else 0}
                line = json.dumps(record)
                print(line, flush=True)
                log.write(line + '\n')
                window.clear()
            stopping = torch.tensor(int(STOP or time.monotonic() >= deadline - 180 or step == t['steps']
                                        or (args.stop_after_steps is not None and step - first >= args.stop_after_steps)), device=device)
            if world_size > 1:
                dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
            ending = bool(stopping)
            should_evaluate = step % t['evaluate_every'] == 0 or ending
            if should_evaluate:
                if rank == 0:
                    report = evaluate(model, validation, t, device)
                    record = {'event': 'development', 'step': step, **report}
                    print(json.dumps(record), flush=True)
                    log.write(json.dumps(record) + '\n')
                    if report['selection_mse'] < best:
                        best = report['selection_mse']
                        temporary = args.output / '.best.safetensors.tmp'
                        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(temporary))
                        temporary.replace(args.output / 'best.safetensors')
                        write_json(args.output / 'best.json', {'step': step, **report})
                if world_size > 1:
                    dist.barrier()
            if step % t['checkpoint_every'] == 0 or ending:
                checkpoint(args.output, model, optimizer, corpus, step, best,
                           previous_elapsed + time.monotonic() - start, world_size, rank, device, report)
                last_checkpoint = step
            if ending:
                break
        if rank == 0:
            status = ('completed' if step == t['steps'] else 'checkpointed_after_requested_steps'
                      if args.stop_after_steps is not None and step - first >= args.stop_after_steps else 'checkpointed_at_time_limit')
            write_json(args.output / 'completion.json', {'status': status,
                       'step': step, 'last_checkpoint': last_checkpoint, 'best_selection_mse': best,
                       'seconds': time.monotonic() - start, 'source_draws_rank0': corpus.draws})
            print(json.dumps({'event': 'TRAINING_FINISHED', 'step': step, 'output': str(args.output)}), flush=True)
    except Exception as error:
        if rank == 0:
            write_json(args.output / 'failure.json', {'error_type': type(error).__name__, 'message': str(error),
                       'step': step, 'last_checkpoint': last_checkpoint, 'seconds': time.monotonic() - start})
        raise
    finally:
        if log:
            log.close()
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    p.add_argument('--resume', type=Path)
    p.add_argument('--steps', type=int)
    p.add_argument('--max-hours', type=float)
    p.add_argument('--stop-after-steps', type=int, help='Checkpoint after this many additional steps without changing the LR schedule')
    p.add_argument('--allow-cpu', action='store_true')
    main(p.parse_args())
