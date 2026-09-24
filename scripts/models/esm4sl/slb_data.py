"""SLB datasets for the ESM4SL coach_pl harness (registered into esm4sl's DATASET_REGISTRY).

Differences from esm4sl/dataset/sl_emb.py (documented in notes/models/esm4sl.md):
  * embeddings are held in RAM (loaded once per process from _slb files) instead of one torch.load per item;
  * the cell-line tensor is looked up per ROW (SLB pools all human contexts into one model) instead of one
    fixed cell line per run; pairs from contexts without DepMap data get a zero tensor;
  * the whole-embedding collate pads to the longest protein in the batch (<= 2000) instead of always 2000.
Label / sampler / random-swap logic is inherited unchanged from SLDataset / SLembDataset.
Config keys used: DATASET.TRAIN_FILE/VAL_FILE/TEST_FILE (csv with columns 0,1,2,ctx; 0/1 = int gene index),
DATASET.ESM_ROOT (dir with index.tsv + per-residue <gene>.pt, or a mean parquet), DATASET.CELL_LINE (path to
cellfeat.npz or null), DATALOADER.SAMPLE.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.trainer.states import RunningStage

from coach_pl.dataset import DATASET_REGISTRY
from esm4sl.dataset.sl import SLDataset
from esm4sl.dataset.sl_emb import CollateBatch

_CACHE: dict = {}


def _embeddings(root: str) -> dict[int, torch.Tensor]:
    if root in _CACHE:
        return _CACHE[root]
    p = Path(root)
    if p.suffix == ".parquet":          # mean embeddings (gene, e0..e1279) + index column 'idx'
        df = pd.read_parquet(p)
        x = torch.from_numpy(df.filter(regex=r"^e\d+$").to_numpy(np.float32))
        emb = {int(i): x[k] for k, i in enumerate(df["idx"])}
    else:                               # per-residue fp16 tensors listed in index.tsv (idx, gene, path)
        ix = pd.read_csv(p / "index.tsv", sep="\t")
        emb = {int(i): torch.load(f, weights_only=True) for i, f in zip(ix.idx, ix.path)}
    _CACHE[root] = emb
    return emb


def _cells(path) -> dict[str, np.ndarray] | None:
    if not path:
        return None
    if path not in _CACHE:
        z = np.load(path)
        _CACHE[path] = {k: z[k].astype(np.float32) for k in z.files if k != "genes"}
    return _CACHE[path]


class _SLBBase(SLDataset):
    def __init__(self, cfg, stage):
        # = SLDataset.__init__ minus its hard-coded '/home/qingyuyang/.../clname2embed.npy' load
        self.stage = stage
        self.cell_line = cfg.DATASET.CELL_LINE
        if stage == RunningStage.TRAINING:
            self.df = pd.read_csv(cfg.DATASET.TRAIN_FILE)
            self.df = self.df.sample(frac=1).reset_index(drop=True)  # shuffle
        elif stage == RunningStage.VALIDATING:
            self.df = pd.read_csv(cfg.DATASET.VAL_FILE)
        else:
            self.df = pd.read_csv(cfg.DATASET.TEST_FILE)
        self.sample = cfg.DATALOADER.SAMPLE
        self.emb = _embeddings(cfg.DATASET.ESM_ROOT)
        self.cells = _cells(cfg.DATASET.CELL_LINE)
        if self.cells is not None:
            zero = np.zeros_like(next(iter(self.cells.values())))
            self.cell_rows = [self.cells.get(c, zero) for c in self.df["ctx"]]
        self.g0 = self.df["0"].to_numpy()
        self.g1 = self.df["1"].to_numpy()
        self.y = self.df["2"].to_numpy().astype(np.float32)

    def _item(self, index, mean: bool):
        g1_idx, g2_idx, label = int(self.g0[index]), int(self.g1[index]), float(self.y[index])
        gene1, gene2 = self.emb[g1_idx].float(), self.emb[g2_idx].float()
        if mean and gene1.dim() == 2:
            gene1, gene2 = gene1.mean(0), gene2.mean(0)
        if self.stage == RunningStage.TRAINING and random.uniform(0, 1) > 0.5:
            gene2, gene1 = gene1, gene2
            g2_idx, g1_idx = g1_idx, g2_idx
        if self.cells is None:
            return gene1, gene2, label, g1_idx, g2_idx
        return gene1, gene2, label, g1_idx, g2_idx, self.cell_rows[index]


@DATASET_REGISTRY.register()
class SLBMeanDataset(_SLBBase):
    """ESM-2 + MLP input: mean embeddings (ClsModule concatenates gene1 | gene2)."""

    def __getitem__(self, index):
        return self._item(index, mean=True)


MAXLEN = int(os.environ.get("ESM4SL_MAXLEN", 2000))


class _DynCollate(CollateBatch):
    def padding(self, genes, max_len: int = 2000):
        return super().padding(genes, max_len=min(max_len, MAXLEN, max(g.shape[0] for g in genes)))


@DATASET_REGISTRY.register()
class SLBWholeDataset(_SLBBase):
    """ESM4SL input: per-residue embeddings + (optional) cell-line tensor."""

    def __getitem__(self, index):
        return self._item(index, mean=False)

    @property
    def collate_fn(self):
        return _DynCollate(self.cell_line)
