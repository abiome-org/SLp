"""Retrospective, source-separated development evaluation of a saved 1.2 model."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from safetensors.torch import load_file
from data import Corpus, to_device
from model import Config, WorldModel
from train import evaluate, autocast
from io_utils import sha, write_json


@torch.inference_mode()
def distributions(model, corpus, config, device, panels=32):
    """Compare empirical endpoint distributions; cells are not paired trajectories."""
    model.eval()
    corpus.rng = np.random.default_rng(121007)
    records = {}
    for source in [s for s in corpus.sources if s.cell]:
        for panel in range(panels):
            raw = corpus.molecular(config['micro_batch'], config['observation_queries'], config['output_queries'], source=source.name)
            batch = to_device(raw, device)
            with autocast(device, config['precision']):
                generated = model.generate(batch, steps=32, seed=121007 + panel)
            for modality, label in ((0, 'rna'), (1, 'protein')):
                keep = batch['query_mask'].all(0) & (batch['query_modality'][0] == modality)
                if not keep.any():
                    continue
                x, y, control = generated[:, keep].float(), batch['target'][:, keep], batch['query_anchor'][:, keep]
                def energy(a, b):
                    return (2 * torch.cdist(a, b).mean() - torch.cdist(a, a).mean() - torch.cdist(b, b).mean()) / a.shape[1] ** .5
                values = {'energy_distance': float(energy(x, y)), 'control_energy_distance': float(energy(control, y)),
                          'mean_mse': float((x.mean(0) - y.mean(0)).square().mean()),
                          'control_mean_mse': float((control.mean(0) - y.mean(0)).square().mean()),
                          'variance_mse': float((x.var(0, unbiased=False) - y.var(0, unbiased=False)).square().mean()),
                          'control_variance_mse': float((control.var(0, unbiased=False) - y.var(0, unbiased=False)).square().mean())}
                key = source.name + ':' + label
                records.setdefault(key, []).append(values)
    return {'panels_per_source': panels, 'cells_per_panel': config['micro_batch'], 'flow_steps': 32,
            'metrics': {key: {name: float(np.mean([r[name] for r in rows])) for name in rows[0]} for key, rows in records.items()},
            'scope': 'Fixed retrospective unpaired endpoint panels in normalized assay units. Finite-sample energy and moment estimates; not paired cellular counterfactual accuracy.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batches', type=int, default=128)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    torch.set_num_threads(4)
    latest = json.loads((args.run / 'latest.json').read_text())
    config = json.loads((args.run / latest['path'] / 'config.json').read_text())
    segments = [args.run / 'run.json'] + sorted(args.run.glob('run-resume-*.json'), key=lambda p: int(p.stem.split('-')[-1]))
    training = json.loads(segments[-1].read_text())['configuration']['training']
    training.update(evaluation_batches=args.batches, evaluate_action_ablation=True)
    model = WorldModel(Config(**config)).to(args.device).eval()
    weights = args.run / 'best.safetensors'
    model.load_state_dict(load_file(str(weights), device=args.device))
    corpus = Corpus(args.root, validation=True)
    report = evaluate(model, corpus, training, torch.device(args.device))
    report['single_cell_distributions'] = distributions(model, corpus, training, torch.device(args.device))
    captured = args.run / 'evaluation-source'
    captured.mkdir(exist_ok=False)
    for name in ('evaluate.py', 'train.py', 'data.py', 'legacy_data.py', 'model.py', 'io_utils.py', 'requirements.in', 'requirements-linux-cu128.lock'):
        shutil.copy2(Path(__file__).with_name(name), captured / name)
    report.update(weights_sha256=sha(weights), corpus_receipts=corpus.receipts,
                  evaluation_batches_per_source=args.batches,
                  selection_used_development=True, independent_test=False,
                  evaluation_source_sha256={p.name: sha(p) for p in captured.iterdir()})
    write_json(args.output, report)
    print(json.dumps(report), flush=True)
