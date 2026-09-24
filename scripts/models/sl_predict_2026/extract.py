"""Extract the released SL-Predict MAE gene vectors from DepMap 26Q1.

The input file is the exact 26Q1 Chronos matrix used by the authors. The two
K562-derived lines are removed, leaving the checkpoint's 1206 input columns.
The source repository's encoder class is loaded with its frozen weights.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw/sl_predict_2026"
OUT = RAW / "gene_embeddings.parquet"
PROFILE = RAW / "dependency_zscore.npy"
EXCLUDED_CELL_IDS = {"ACH-000551", "ACH-002061"}
SOURCE_SHA256 = "e610a4cefb13a82b5b256b47eb08b63ff14843f8dbd0fb164bc0a32688e5b89e"
CHECKPOINT_SHA256 = "ed8840a654bb9591877d1f25007c07487f682d41c0a5c712264bfe154b0640cc"
META = OUT.with_suffix(".json")


def verify(path: Path, expected: str) -> None:
    with path.open("rb") as stream:
        h = hashlib.file_digest(stream, "sha256").hexdigest()
    if h != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {h}")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class GeneEncoder(nn.Module):
    """Exact `src.models.gene_encoder.encoder.GeneEncoder` layer sequence."""

    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...], embed_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for hidden in hidden_dims:
            layers.append(nn.Sequential(nn.Linear(prev, hidden), nn.LayerNorm(hidden), nn.GELU()))
            prev = hidden
        layers.append(nn.Linear(prev, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    csv = RAW / "CRISPRGeneEffect_26Q1.csv"
    ckpt = RAW / "mae_encoder_d256_leak_repaired.ckpt"
    verify(csv, SOURCE_SHA256)
    verify(ckpt, CHECKPOINT_SHA256)
    fingerprint = {
        "source_csv_sha256": SOURCE_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "extractor_sha256": digest(Path(__file__)),
    }
    if OUT.exists() and PROFILE.exists() and META.exists():
        saved = json.loads(META.read_text())
        if (all(saved.get(k) == v for k, v in fingerprint.items())
                and saved.get("embeddings_sha256") == digest(OUT)
                and saved.get("profile_sha256") == digest(PROFILE)):
            print(f"verified cached {OUT} and {PROFILE}")
            return
    df = pd.read_csv(csv, low_memory=False)
    cell_col = df.columns[0]
    cell_ids = df[cell_col].astype(str)
    if len(df) != 1208 or sum(cell_ids.isin(EXCLUDED_CELL_IDS)) != 2:
        raise ValueError("unexpected DepMap 26Q1 cell-line universe")
    keep = ~cell_ids.isin(EXCLUDED_CELL_IDS)
    genes = []
    gene_cols = []
    for col in df.columns[1:]:
        match = re.fullmatch(r"([A-Za-z0-9_.\-]+)\s+\(\d+\)", col)
        if match:
            genes.append(match.group(1))
            gene_cols.append(col)
    if len(gene_cols) < 18000 or len(set(genes)) != len(genes):
        raise ValueError("unexpected DepMap 26Q1 gene columns")
    x = df.loc[keep, gene_cols].to_numpy(dtype=np.float32).T
    del df
    if x.shape[1] != 1206:
        raise ValueError(f"encoder input dimension mismatch: {x.shape}")
    mean = np.nanmean(x, axis=1, keepdims=True)
    std = np.nanstd(x, axis=1, keepdims=True)
    x = (x - mean) / np.where(std == 0, 1, std)
    x = np.nan_to_num(x, nan=0.0).astype(np.float32)
    np.save(PROFILE, x)

    weights = torch.load(ckpt, map_location="cpu", weights_only=False)
    hp = weights["hyper_parameters"]
    model = GeneEncoder(hp["in_dim"], hp["hidden_dims"], hp["embed_dim"])
    enc_state = {k.removeprefix("encoder."): v for k, v in weights["state_dict"].items()
                 if k.startswith("encoder.")}
    model.load_state_dict(enc_state, strict=True)
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            chunks.append(model(torch.from_numpy(x[start:start + 512])).numpy())
    z = np.concatenate(chunks, axis=0)
    if z.shape != (len(genes), 256) or not np.isfinite(z).all():
        raise ValueError(f"unexpected encoder output: {z.shape}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"gene": genes, **{f"dim_{j:03d}": z[:, j] for j in range(z.shape[1])}}).to_parquet(OUT, index=False)
    META.write_text(json.dumps({
        **fingerprint,
        "embeddings_sha256": digest(OUT),
        "profile_sha256": digest(PROFILE),
        "n_genes": len(genes), "n_cells": x.shape[1], "dim": z.shape[1],
        "excluded_cell_ids": sorted(EXCLUDED_CELL_IDS),
        "preprocessing": "per-gene nanmean/nanstd z-score; NaN to 0",
    }, indent=2))
    print(f"wrote {OUT}: {z.shape}")


if __name__ == "__main__":
    main()
