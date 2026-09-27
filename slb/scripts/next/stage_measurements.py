"""Write the measurement tables of candidate human sources for a side-by-side build (data/slb_next).

A candidate source is a parser in src/slbench/sources/staging/<key>.py with a check report in
data/interim/new_human/<key>.json. Only sources whose report says "include" are written (to
data/interim/measurements_next/<key>.parquet); the rest are listed with their reasons.

    uv run python scripts/next/stage_measurements.py [key ...]
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKS = ROOT / "data/interim/new_human"
OUT = ROOT / "data/interim/measurements_next"


def main(keys: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for rep in sorted(CHECKS.glob("*.json")):
        key = rep.stem
        if keys and key not in keys:
            continue
        r = json.loads(rep.read_text())
        if r.get("verdict") != "include":
            print(f"{key:16s} not staged: {r.get('verdict')} ({r.get('verdict_reason', '')[:120]})")
            (OUT / f"{key}.parquet").unlink(missing_ok=True)
            continue
        t = time.time()
        df = importlib.import_module(f"slbench.sources.staging.{key}").load()
        assert df["source"].unique().to_list() == [key], df["source"].unique().to_list()
        df.write_parquet(OUT / f"{key}.parquet")
        print(f"{key:16s} {df.height:>10,d} rows  pos={int((df['label'] == 1).sum()):>7,d}  "
              f"neg={int((df['label'] == 0).sum()):>9,d}  contexts={df['context'].n_unique()}  {time.time() - t:.0f}s",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
