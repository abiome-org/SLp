"""Frozen-checkpoint, fitting-only audit of shared decoder gradient conflict."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / 'results/slp11-transition/human-essential-count-shared-context-seed731-v1'
OUTPUT = ROOT / 'results/slp11-transition/human-essential-count-objective-gradient-audit-v1'
RUNNER = ROOT / 'scripts/run_slp11_count_world_shared_context.py'
SEED = 2831
BATCHES = 16


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_new(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if path.exists() and path.read_text(encoding='utf-8') != payload:
        raise FileExistsError(path)
    path.write_text(payload, encoding='utf-8')


def prepare():
    runner = load_module(RUNNER, 'objective_audit_runner')
    if sha256(RUNNER) != '8fdf92ecf837eb72c4d107cfc1ed053eff071671172079fd2db001dcdbb10091':
        raise ValueError('frozen training runner changed')
    freeze = json.loads((ARTIFACT / 'FROZEN-FITTING-ONLY.json').read_text(encoding='utf-8'))
    for arm, expected in freeze['modelSha256'].items():
        if sha256(ARTIFACT / f'arms/{arm}.safetensors') != expected:
            raise ValueError('checkpoint changed')
    protocol = {
        'hypothesis': 'Cell ELBO and population-mean supervision exert systematically opposing gradients on the shared query-loading decoder in both jointly trained contexts.',
        'decision': 'Support strong sampled decoder conflict only if the joint checkpoint has median gradient cosine below -0.25 and at least 75 percent negative batch cosines in each source. This is a local optimization diagnostic, not evidence that splitting loadings improves generalization.',
        'sampling': {'seed': SEED, 'batchesPerSourcePerArm': BATCHES, 'cells': 128,
                     'controls': 64, 'targets': 64, 'uniformUniqueFittingGenes': 16},
        'objectives': 'Separate gradients of count ELBO and 0.1 times source-normalized population-mean MSE; exact frozen sampler/training helper, training dropout for counts and dropout-free population prior. Same rows, population selections and torch seed per arm/source.',
        'parameters': 'All shared query_loading parameters; no optimizer step or model mutation.',
        'modalities': 'Fitting-only native raw counts, fitting population moments, measured controls and shared static577 protein/GO features.',
        'fittingFreezeSha256': sha256(ARTIFACT / 'FROZEN-FITTING-ONLY.json'),
        'modelSha256': freeze['modelSha256'],
        'runnerSha256': sha256(RUNNER), 'auditSha256': sha256(Path(__file__)),
        'compute': {'device': 'cuda', 'cpuThreads': 2, 'maximumSeconds': 600},
        'developmentOpened': False, 'testOpened': False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_new(OUTPUT / 'protocol.json', protocol)
    return runner


def gradient_summary(count_gradient, mean_gradient):
    first = torch.cat([part.detach().double().reshape(-1) for part in count_gradient])
    second = torch.cat([part.detach().double().reshape(-1) for part in mean_gradient])
    n1, n2 = first.norm(), second.norm()
    if not torch.isfinite(first).all() or not torch.isfinite(second).all() or min(n1, n2) <= 0:
        raise ValueError('undefined or nonfinite gradient comparison')
    return {'cosine': float(torch.dot(first, second) / (n1 * n2)),
            'countNorm': float(n1), 'weightedMeanNorm': float(n2),
            'weightedMeanToCountNorm': float(n2 / n1)}


def run():
    runner = prepare()
    if (OUTPUT / 'report.json').exists():
        raise FileExistsError('audit already complete')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no fallback')
    panels, _ = runner.load_panels()
    core = load_module(ARTIFACT / 'source/count_latent_state.py', 'objective_audit_core')
    step = load_module(ARTIFACT / 'source/training_step.py', 'objective_audit_step')
    device = torch.device('cuda')
    tensors = runner.panel_tensors(panels, device)
    reports = {}
    started = time.perf_counter()
    for arm in runner.ARMS:
        model = core.CountLatentState(core.Config(**runner.MODEL_CONFIG)).to(device)
        model.load_state_dict(load_file(str(ARTIFACT / f'arms/{arm}.safetensors')))
        model.train()
        reports[arm] = {}
        for source, panel in panels.items():
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            cells_rng, mean_rng = np.random.default_rng(SEED), np.random.default_rng(SEED + 1)
            trace, records = hashlib.sha256(), []
            for _ in range(BATCHES):
                if time.perf_counter() - started > 600:
                    raise TimeoutError('fixed audit wall cap reached')
                cells, rows = runner.as_cell_batch(step, panel, tensors[source], cells_rng, device)
                populations, genes = runner.as_population_batch(step, panel, mean_rng, device)
                trace.update(rows.astype('<i8').tobytes() + genes.astype('<i8').tobytes())
                local = tensors[source]
                result = step.training_losses(model, local['query'], local['basal'], local['mask'],
                                              cells, populations, mean_weight=.1,
                                              fitting_mean_scale=panel.fitting_mean_scale)
                parameters = tuple(model.query_loading.parameters())
                count_grad = torch.autograd.grad(result['count_elbo'], parameters, retain_graph=True)
                mean_grad = torch.autograd.grad(.1 * result['normalized_mean_mse'], parameters)
                records.append(gradient_summary(count_grad, mean_grad))
            cosine = np.array([record['cosine'] for record in records])
            reports[arm][source] = {
                'medianCosine': float(np.median(cosine)), 'meanCosine': float(cosine.mean()),
                'negativeFraction': float(np.mean(cosine < 0)),
                'medianWeightedMeanToCountNorm': float(np.median([v['weightedMeanToCountNorm'] for v in records])),
                'rowsAndPopulationTraceSha256': trace.hexdigest(), 'batches': records,
                'strongConflict': bool(np.median(cosine) < -.25 and np.mean(cosine < 0) >= .75),
            }
        del model
    report = {'protocolSha256': sha256(OUTPUT / 'protocol.json'), 'arms': reports,
              'strongJointConflictBothSources': all(v['strongConflict'] for v in reports['joint-alternating'].values()),
              'seconds': time.perf_counter() - started, 'developmentOpened': False, 'testOpened': False}
    write_new(OUTPUT / 'report.json', report)
    return {**{key: value for key, value in report.items() if key != 'arms'},
            'arms': {arm: {source: {k: v for k, v in values.items() if k != 'batches'}
                            for source, values in sources.items()} for arm, sources in reports.items()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(2):
        result = prepare() if args.command == 'prepare' else run()
    print('prepared' if args.command == 'prepare' else json.dumps(result, indent=2))
