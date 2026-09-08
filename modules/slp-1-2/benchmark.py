"""Measure full-architecture CUDA memory/throughput on maximal token shapes.

These synthetic tensors check execution cost only; they are not scientific
training data and no resulting weights are saved or used by the training run.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from data import batch_template, to_device
from model import Config, WorldModel
from train import objective


def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    config = json.loads(Path(__file__).with_name('config.json').read_text())['model']
    config.update(gene_count=args.gene_count, activation_checkpointing=False)
    model = WorldModel(Config(**config)).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    for size, padded in ((4, False), (8, False), (16, True)):
        batch = batch_template(size, 532, 2, 532)
        for name in ('observation_mask', 'action_mask', 'query_mask'):
            batch[name][:] = True
        if padded:
            batch['action_mask'][0] = False
        for name in ('observation_features', 'action_features', 'query_features', 'target'):
            batch[name][:] = .1
        batch = to_device(batch, 'cuda')
        try:
            elapsed = []
            torch.cuda.reset_peak_memory_stats()
            for step in range(8):
                torch.cuda.synchronize()
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, _ = objective(model, batch, flow=step % 2 == 0, noise_scale=1.)
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                if step >= 3:
                    elapsed.append(time.perf_counter() - start)
            print(json.dumps({'micro_batch': size, 'tokens': 1067, 'activation_checkpointing': False,
                    'mixed_padding': padded, 'mean_step_seconds': sum(elapsed) / len(elapsed),
                    'peak_cuda_gib': torch.cuda.max_memory_allocated() / 2**30,
                    'parameters': sum(p.numel() for p in model.parameters()), 'torch': torch.__version__}), flush=True)
        except torch.OutOfMemoryError:
            print(json.dumps({'micro_batch': size, 'out_of_memory': True}), flush=True)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            break


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gene-count', type=int, required=True)
    main(p.parse_args())
