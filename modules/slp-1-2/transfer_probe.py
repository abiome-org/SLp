"""Human-only development transfer probe with frozen pretrained/random bases.

Only a ridge readout sees adaptation labels. Adaptation and evaluation genes
are disjoint, and both were excluded from base fitting. The existing development
pool was used for checkpoint selection, so this is not an independent test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from base_evaluation import capture_source, load_panels, predict, save_arrays, validate_bundle
from inference import World
from io_utils import load_npz, sha, write_json
from model import WorldModel


def ridge_predict(train, target, evaluation, penalty):
    """Standardize on adaptation rows; fixed per-dimension normalized penalty.

    Minimize mean squared loss + penalty * dimension * ||coefficient||^2.
    Centering supplies an unpenalized intercept. Dual solve handles low budgets.
    """
    if penalty <= 0 or len(train) != len(target) or len(train) < 2:
        raise ValueError('Invalid ridge inputs')
    x, z, y = (np.asarray(a, np.float64) for a in (train, evaluation, target))
    mean, scale = x.mean(0), x.std(0).clip(1e-6)
    x, z = (x - mean) / scale, (z - mean) / scale
    center = y.mean(0)
    y = y - center
    regularizer = len(x) * x.shape[1] * penalty
    if len(x) < x.shape[1]:
        gram = x @ x.T
        gram.flat[::len(x) + 1] += regularizer
        coefficient = x.T @ np.linalg.solve(gram, y)
    else:
        gram = x.T @ x
        gram.flat[::x.shape[1] + 1] += regularizer
        coefficient = np.linalg.solve(gram, x.T @ y)
    return z @ coefficient + center


def parameter_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(bundle, panels, output, device):
    start = time.monotonic()
    manifest = load_panels(panels)
    c = manifest['configuration']
    a, e = (load_npz(panels / f'probe-{role}.npz') for role in ('adaptation', 'evaluation'))
    if set(a['action_ids'][:, 0]) & set(e['action_ids'][:, 0]):
        raise ValueError('Probe adaptation/evaluation gene overlap')
    for raw in (a, e):
        if len(np.unique(raw['row_key'], axis=0)) != len(raw['row_key']):
            raise ValueError('Repeated labels in probe budget')
        if not (raw['taxon'] == 0).all() or not (raw['assay'] == 8).all():
            raise ValueError('Probe must contain human fitness only')
    world = World(bundle, device)
    validate_bundle(world, manifest['contract'])
    world.model.requires_grad_(False)
    before = parameter_digest(world.model)
    output.mkdir(parents=True, exist_ok=False)
    features, predictions = {}, {}
    for name, model in (('pretrained', world.model),):
        print('Extracting frozen', name, 'representations', flush=True)
        ap, af = predict(model, a, device, features=True)
        ep, ef = predict(model, e, device, features=True)
        features[name] = (af, ef)
        predictions['pretrained_zero_shot'] = ep[:, 0]
    torch.manual_seed(c['random_seed'])
    random_model = WorldModel(world.model.config).to(device).eval().requires_grad_(False)
    print('Extracting frozen random representations', flush=True)
    features['random'] = tuple(predict(random_model, raw, device, features=True)[1] for raw in (a, e))
    del random_model
    features['descriptors_context'] = tuple(np.concatenate((raw['action_features'][:, 0], raw['context']), 1) for raw in (a, e))
    predictions['training_context_mean'] = e['baseline_training_mean'][:, 0]
    predictions['neutral'] = np.zeros(len(e['target']))
    y = a['target'][:, 0]
    truth = e['target'][:, 0]
    rows = []
    for budget in c['probe_budgets']:
        print('Fitting readouts with', budget, 'human labels', flush=True)
        record = dict(labels=budget, adaptation_genes=int(len(np.unique(a['action_ids'][:budget, 0]))), mse={})
        predictions[f'adaptation_mean_{budget}'] = np.full(len(truth), float(y[:budget].mean()))
        record['mse']['adaptation_mean'] = float(np.square(predictions[f'adaptation_mean_{budget}'] - truth).mean())
        for name, (af, ef) in features.items():
            pred = ridge_predict(af[:budget], y[:budget], ef, c['probe_ridge'])
            if not np.isfinite(pred).all():
                raise ValueError('Nonfinite probe prediction')
            predictions[f'{name}_{budget}'] = pred
            record['mse'][name] = float(np.square(pred - truth).mean())
        rows.append(record)
    after = parameter_digest(world.model)
    if before != after or any(p.grad is not None or p.requires_grad for p in world.model.parameters()):
        raise ValueError('Probe modified base parameters')
    save_arrays(output / 'predictions.npz', dict(predictions, target=truth, evaluation_gene=e['action_ids'][:, 0]))
    for name, (af, ef) in features.items():
        save_arrays(output / f'features-{name}.npz', dict(adaptation=af, evaluation=ef))
    report = dict(schema='slp.human-transfer-probe/v1', independent_test=False, phase='frozen_base_readout_probe',
                  weights_sha256=sha(bundle / 'model.safetensors'), panels_sha256=sha(panels / 'manifest.json'),
                  parameter_digest_before=before, parameter_digest_after=after, base_updated=False,
                  probe_taxon=9606, evaluation_rows=len(truth), evaluation_genes=len(np.unique(e['action_ids'][:, 0])),
                  random_seed=c['random_seed'], ridge_penalty=c['probe_ridge'], budgets=rows,
                  zero_shot_mse={name: float(np.square(predictions[name] - truth).mean()) for name in
                                 ('pretrained_zero_shot', 'training_context_mean', 'neutral')},
                  elapsed_seconds=time.monotonic() - start,
                  scope='One fixed retrospective human gene partition and one random initialization. Unique nested gene/context labels; frozen bases; identical ridge protocol. Same task as pretraining, new intervention genes. Not human SL transfer, final post-training, an independent test, or a yeast-pretraining ablation.')
    capture_source(output)
    report['files'] = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
    write_json(output / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--panels', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    torch.set_num_threads(4)
    run(args.bundle, args.panels, args.output, args.device)
