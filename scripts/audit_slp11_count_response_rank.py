"""Fitting-only reduced-rank ridge diagnostic; no development outcome access."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/slp11-transition/human-essential-count-response-rank-audit-v1'
CORE = ROOT / 'modules/slp-1-1-count-static-ridge-v1/count_static_ridge.py'
LOADER = ROOT / 'scripts/run_slp11_control_coexpression_ridge_cv.py'
RANKS = (32, 64, 128, 256)
ALPHA = 1000.0


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reduced_rank_prediction(core, state, features, rank, alpha=ALPHA):
    """Exact rank-constrained ridge in whitened design coordinates.

    The intercept is unpenalized and excluded from the rank constraint.
    W = (D + alpha)^(-1/2) U' X' Y. Its truncated SVD minimizes
    squared error plus alpha times the coefficient Frobenius norm.
    """
    if rank < 1 or alpha <= 0:
        raise ValueError('positive rank and alpha required')
    scale = np.sqrt(state['eigenvalues'] + alpha)
    whitened = state['rhs'] / scale[:, None]
    _, vectors = np.linalg.eigh(whitened @ whitened.T)
    keep = vectors[:, -min(rank, len(vectors)):]
    design = core.transform_features(features, state) - state['design_mean']
    latent = (design @ state['eigenvectors']) / scale
    return state['target_mean'] + (latent @ keep) @ (keep.T @ whitened)


def write_new(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if path.exists() and path.read_text(encoding='utf-8') != payload:
        raise FileExistsError(path)
    path.write_text(payload, encoding='utf-8')


def prepare(output):
    loader = load_module(LOADER, 'rank_audit_pins')
    pins = {CORE: loader.PINS[CORE]}
    for paths in loader.CONTEXTS.values():
        for kind in ('moments', 'static', 'baseline'):
            pins[paths[kind]] = loader.PINS[paths[kind]]
    for path, expected in pins.items():
        if sha256(path) != expected:
            raise ValueError(f'input changed: {path}')
    protocol = {
        'hypothesis': 'A rank-32 intervention response map preserves full static ridge held-fitting-gene MSE within one percent in both K562 and RPE1.',
        'rule': 'Reject rank-32 sufficiency if either source MSE exceeds 1.01 times full ridge. Report predeclared ranks 32,64,128,256 and full; no development forecast or architecture selection on development outcomes.',
        'endpoint': 'ln1p(equal-cell CP10k mean) minus GEM-composition-matched control anchor, all native queries; equal weight per fitting gene.',
        'modalities': 'Static577 protein/GO action features, fitting perturbation moments, reconstruction-training NT controls only.',
        'folds': 'Existing global seed731 three folds, fold-local feature normalization and intercept. All ranks use alpha1000 previously selected in fitting CV; this is adaptive fitting diagnostics.',
        'rankConstraint': 'Exact reduced-rank regularized least squares; unpenalized intercept excluded from response-map rank.',
        'limitations': 'Learned native-panel output loadings do not establish feature-query decoding or prediction of unmeasured queries. Linear rank does not equal nonlinear latent dimension.',
        'compute': {'cpuThreads': 2, 'maximumSeconds': 600},
        'pins': {str(path.relative_to(ROOT)): digest for path, digest in pins.items()},
        'runnerSha256': sha256(Path(__file__)),
        'developmentOpened': False, 'testOpened': False,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_new(output / 'protocol.json', protocol)
    return loader, protocol


def run(output):
    loader, protocol = prepare(output)
    if (output / 'report.json').exists():
        raise FileExistsError('diagnostic already completed')
    core = load_module(CORE, 'rank_audit_ridge')
    started = time.perf_counter()
    reports, arrays = {}, {}
    with threadpool_limits(limits=2):
        for source, paths in loader.CONTEXTS.items():
            moments = loader.load_npz(paths['moments'])
            static = loader.load_npz(paths['static'])
            baseline = loader.load_npz(paths['baseline'])
            genes = moments['action_ids'].astype(str)
            if not np.array_equal(moments['query_ids'], baseline['query_ids']):
                raise ValueError('query axes differ')
            lookup = {gene: i for i, gene in enumerate(static['entity_id'].astype(str))}
            x = static['feature_values'][[lookup[g] for g in genes]]
            y = core.response_from_cp10k_moments(moments['cp10k_sum'], moments['cell_count'])
            y -= core.control_anchor(baseline['basal_rate'], moments['gem_cell_count'])
            folds = np.array([core.global_gene_fold(gene) for gene in genes])
            errors = {str(rank): np.empty(len(genes)) for rank in (*RANKS, 'full')}
            fold_reports = []
            for fold in range(3):
                if time.perf_counter() - started > 600:
                    raise TimeoutError('fixed CPU cap reached')
                fitting, held = folds != fold, folds == fold
                state = core.fit_state(x[fitting], y[fitting])
                local = {}
                for rank in (*RANKS, 'full'):
                    prediction = (core.predict_residual(state, x[held], '1000') if rank == 'full'
                                  else reduced_rank_prediction(core, state, x[held], rank))
                    error = np.mean(np.square(prediction - y[held]), axis=1)
                    errors[str(rank)][held] = error
                    local[str(rank)] = float(error.mean())
                fold_reports.append({'fold': fold, 'heldGenes': int(held.sum()), 'mse': local})
            mse = {rank: float(error.mean()) for rank, error in errors.items()}
            reports[source] = {'genes': len(genes), 'queries': y.shape[1], 'mse': mse,
                               'rank32PreservesWithinOnePercent': mse['32'] <= 1.01 * mse['full'],
                               'folds': fold_reports}
            arrays[f'{source}_gene_ids'] = genes
            arrays[f'{source}_folds'] = folds
            arrays.update({f'{source}_{rank}_mse': error for rank, error in errors.items()})
    np.savez_compressed(output / 'per-gene-mse.npz', **arrays)
    shutil.copyfile(Path(__file__), output / 'executed-source.py')
    report = {'protocolSha256': sha256(output / 'protocol.json'), 'sources': reports,
              'seconds': time.perf_counter() - started, 'developmentOpened': False,
              'testOpened': False, 'perGeneSha256': sha256(output / 'per-gene-mse.npz'),
              'rank32SufficiencyPasses': all(v['rank32PreservesWithinOnePercent'] for v in reports.values()),
              'limitations': protocol['limitations']}
    write_new(output / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = prepare(args.output)[1] if args.command == 'prepare' else run(args.output)
    print(json.dumps(result, indent=2))
