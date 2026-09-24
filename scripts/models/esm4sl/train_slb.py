"""Run the ORIGINAL ESM4SL coach_pl training entry point (train.py -> coach_pl.tool.train.main) with the SLB
datasets from slb_data.py registered.  Must be run with cwd = external/models/esm4sl.

usage: python train_slb.py --config-file esm4sl/configuration/{attn,mlp}/new.yaml --num-gpus 1 KEY VALUE ...
"""
import sys

import torch

import os
from pathlib import Path

torch.set_num_threads(8)
if os.environ.get("ESM4SL_CPU") == "1":
    # CPU mode for the small ESM-2+MLP: the original ClsModule calls .cuda() on every batch; make that a no-op
    # (with CUDA_VISIBLE_DEVICES="" Lightning's accelerator="auto" then trains on CPU). Numerically identical model.
    torch.Tensor.cuda = lambda self, *a, **k: self

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import esm4sl  # noqa: E402,F401  (registers datasets / models / modules)
import slb_data  # noqa: E402,F401  (registers SLBMeanDataset / SLBWholeDataset)
from coach_pl.tool.train import arg_parser, main  # noqa: E402
from esm4sl.model.attention import MultiHeadCrossAttention  # noqa: E402

# Bug fix (documented in notes): with the cell branch, gene_cell_CA calls attention with a query mask and no
# key mask, which fills whole rows of padded query positions with -inf -> softmax NaN -> NaN propagates to every
# sample containing a padded protein (verified: the unmodified AttnWrap returns NaN for any padded input when
# use_cell is set). We drop query-only masks (padded positions are excluded later by the pooling masks).
_orig_fwd = MultiHeadCrossAttention.forward


def _fwd(self, q, q_mask, kv, kv_mask):
    if q_mask is not None and kv is not None and kv_mask is None:
        q_mask = None
    return _orig_fwd(self, q, q_mask, kv, kv_mask)


MultiHeadCrossAttention.forward = _fwd

if __name__ == "__main__":
    main(arg_parser().parse_args())
