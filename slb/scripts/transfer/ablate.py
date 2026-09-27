"""Which features carry the human signal of H0? Refit gbm_nocode on an arm with one feature group held
constant, and score dev and test. Scored by score.py as ablation/<arm>/<model>__no_<group>.

    uv run python scripts/transfer/ablate.py --arm h0 --group fam
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_model as R

GROUPS = {
    "fam": ["same_family"],
    "fit": ["fit_min", "fit_max"],
    "go": [c for c in __import__("run").FEAT if c.startswith("go_")],
    "net": [c for c in __import__("run").FEAT if c.startswith(("biogrid", "string_", "phys_"))],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", default="gbm_nocode")
    ap.add_argument("--group", required=True, choices=sorted(GROUPS))
    a = ap.parse_args()
    arm = R.ROOT / "data/transfer" / a.arm
    tr = pd.read_parquet(arm / "train.parquet")
    ev = {s: pd.read_parquet(arm / f"{s}_inputs.parquet") for s in R.SPLITS}
    sc = R.run_gbm(tr[tr.label.notna()], pd.concat(ev.values(), ignore_index=True), a.model, drop=GROUPS[a.group])
    out = R.OUT / "ablation" / a.arm
    out.mkdir(parents=True, exist_ok=True)
    for s, e in ev.items():
        pd.DataFrame({"example_id": e.example_id.to_numpy(), "score": sc.reindex(e.example_id).to_numpy()}) \
            .to_parquet(out / f"{a.model}__no_{a.group}_{s}.parquet", index=False)


if __name__ == "__main__":
    main()
