"""Plain-PyTorch training loop for the ESM4SL attention model (AttnModule / AttnWrap, original config
esm4sl/configuration/attn/new.yaml), used instead of the Lightning harness for esm4sl__human.

Why: under the coach_pl Lightning trainer the attention run logged train/loss = nan from the first steps
(bf16-mixed) and ran at ~1.8 it/s, while the same module/batches in a plain loop train finitely at ~5 it/s; we
could not resolve this within the shared-GPU slot. The loop reproduces the harness' protocol: same model +
AttnModule.forward (BCEWithLogits), Adam(lr = BASE_LR * batch/16, weight_decay 1e-4), MultiStepLR([30, 70], 0.2)
stepped per epoch, grad-norm clipping 1.0, class-balanced WeightedRandomSampler, batch 16, checkpoint selection
on the validation 'avgmtr' = (AUROC + AUPRC) / 2 (evaluated every EVAL_EVERY steps and at epoch end).
Writes <OUTPUT_DIR>/csv_log/version_0/test_logits.csv (gene1, gene2, probs) like ClsModule, so collect.py works.

usage (cwd = external/models/esm4sl): python train_attn.py <workdir> <cellfeat.npz|none> <epochs>
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(8)
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv, _args = sys.argv[:1], sys.argv[1:]
import train_slb  # noqa: E402,F401  (registers datasets + NaN fix for the cell cross-attention)
from coach_pl.configuration import CfgNode  # noqa: E402
from coach_pl.dataset.build import build_dataset  # noqa: E402
from coach_pl.module import build_module  # noqa: E402
from pytorch_lightning.trainer.states import RunningStage  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

wd, cell, epochs = Path(_args[0]), _args[1], int(_args[2])
EVAL_EVERY = int(os.environ.get("ESM4SL_EVAL_EVERY", 3000))
torch.manual_seed(42)
np.random.seed(42)
cfg = CfgNode.load_yaml_with_base("esm4sl/configuration/attn/new.yaml")
CfgNode.merge_with_dotlist(cfg, ["DATASET.NAME", "SLBWholeDataset", "DATASET.ESM_ROOT", str(wd),
                                 "DATASET.CELL_LINE", "null" if cell == "none" else cell,
                                 "DATASET.TRAIN_FILE", str(wd / "train.csv"), "DATASET.VAL_FILE", str(wd / "val.csv"),
                                 "DATASET.TEST_FILE", str(wd / "test.csv")])
bs = cfg.DATALOADER.TRAIN.BATCH_SIZE
tr = build_dataset(cfg, RunningStage.TRAINING)
va = build_dataset(cfg, RunningStage.VALIDATING)
te = build_dataset(cfg, RunningStage.TESTING)
dl = lambda d, s: DataLoader(d, batch_size=bs, sampler=s, shuffle=False, collate_fn=d.collate_fn, num_workers=2,
                             pin_memory=True, drop_last=d is tr)
mod = build_module(cfg).cuda()
net = mod.model
opt = torch.optim.Adam(net.parameters(), lr=cfg.MODULE.OPTIMIZER.BASE_LR * bs / 16, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 70], gamma=0.2)


def predict(d):
    net.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in dl(d, None):
            _, logit, *_ = mod.forward(b)
            out.append(torch.sigmoid(logit.float()).cpu())
    net.train()
    return torch.cat(out).numpy()


best, best_state, step, t0 = -1.0, None, 0, time.time()
yv = va.df["2"].to_numpy()


def evaluate():
    global best, best_state
    p = predict(va)
    m = (roc_auc_score(yv, p) + average_precision_score(yv, p)) / 2
    print(f"step {step}: val auroc {roc_auc_score(yv, p):.4f} avgmtr {m:.4f} ({time.time() - t0:.0f}s)", flush=True)
    if m > best:
        best, best_state = m, {k: v.detach().clone() for k, v in net.state_dict().items()}


net.train()
for ep in range(epochs):
    run = []
    for b in dl(tr, tr.sampler):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, *_ = mod.forward(b)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        run.append(loss.item())
        step += 1
        if step % 500 == 0:
            print(f"epoch {ep} step {step}: loss {np.mean(run[-500:]):.4f} ({time.time() - t0:.0f}s)", flush=True)
        if step % EVAL_EVERY == 0:
            evaluate()
    sch.step()
    evaluate()
net.load_state_dict(best_state)
p = predict(te)
o = wd / "out/csv_log/version_0"
o.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"gene1": te.df["0"], "gene2": te.df["1"], "probs": p}).to_csv(o / "test_logits.csv", index=False)
print(f"best val avgmtr {best:.4f}; wrote {o / 'test_logits.csv'} ({time.time() - t0:.0f}s)", flush=True)
