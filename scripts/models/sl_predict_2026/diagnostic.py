"""Fixed-input MAE-versus-coessentiality ablation for the SL-Predict adapter.

Usage: SLB_BENCH=data/bench/slb1.3 uv run python scripts/models/sl_predict_2026/diagnostic.py dev
Outputs go under results/diagnostics/, outside the ranked model battery.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as model  # noqa: E402

slb = model.slb
OUT = slb.ROOT / "results/diagnostics/sl_predict_2026" / slb.BENCH.name


def main(split: str) -> None:
    index, z, profile = model.vectors()
    train = slb.load("train")
    train = train[train.species == "human"].reset_index(drop=True)
    xt, okt = model.pair_features(train, index, z, profile)
    y = train.label.to_numpy(dtype=np.int8)[okt]
    train = train.loc[okt].reset_index(drop=True)
    target = slb.load(split).copy()
    human = target[target.species == "human"]
    xe, oke = model.pair_features(human, index, z, profile)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cols in [("mae_only", slice(None, -1)), ("coess_only", slice(-1, None))]:
        checkpoint = OUT / f"{name}.joblib"
        classifier = model.load_or_train(checkpoint, name, train, xt[:, cols], y)
        scores = np.zeros(len(target), dtype=float)
        pred = np.full(len(human), np.nan)
        pred[oke] = classifier.predict_proba(xe[:, cols])[:, 1]
        pred[~oke] = np.nanmedian(pred)
        scores[target.species.eq("human").to_numpy()] = pred
        dest = OUT / f"{name}_{split}.parquet"
        pd.DataFrame({"example_id": target.example_id, "score": scores}).to_parquet(dest, index=False)
        report = OUT / f"{name}_{split}.json"
        command = ["uv", "run", "slpbench", "eval", str(dest), "--split", split,
                   "--out", str(report)]
        if split == "test":
            command += ["--boot", "200"]
        subprocess.run(command, check=True, cwd=slb.ROOT,
                       env={**os.environ, "SLB_BENCH": str(slb.BENCH)},
                       stdout=subprocess.DEVNULL)
        result = json.loads(report.read_text())
        print(name, split, "SLB", round(result["slb_score"], 4),
              "human", round(result["species_scores"]["human"], 4), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
