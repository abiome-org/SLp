"""LLMsynthlet (Prosz et al. 2026, bioRxiv 10.64898/2026.01.28.702211) released zero-shot predictions.

Qwen2.5-32B-Instruct, no SL-label training, scored 398,277 'clinically relevant' human gene pairs
(data/raw/llmsynthlet/sorted_hypotheses.jsonl: prompt 'GENE1, GENE2' -> text ending 'SCORE: x').
We parse SCORE (fallback: VERDICT) and join on unordered HGNC pairs. Human only; pairs not in the release are
left missing (median-filled by eval) -> variant llmsynthlet__released.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import slb  # noqa: E402

SCORE = re.compile(r"SCORE:\s*\**\s*([01](?:\.\d+)?)")


def table():
    cache = slb.ROOT / "external/models/_statsl_cache/llmsynthlet_released.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    res = slb.symbol_resolver()
    rows = []
    with open(slb.RAW / "llmsynthlet/sorted_hypotheses.jsonl") as f:
        for line in f:
            o = json.loads(line)
            g = [x.strip() for x in o["prompt"].split(",")]
            if len(g) != 2:
                continue
            m = SCORE.findall(o.get("hypothesis") or "")
            s = float(m[-1]) if m else np.nan
            if np.isnan(s):
                v = (o.get("hypothesis") or "").upper()
                s = 0.75 if "VERDICT: SYNTHETIC-LETHAL" in v else (0.25 if "VERDICT:" in v else np.nan)
            rows.append((res(g[0]) or g[0], res(g[1]) or g[1], s))
    t = pd.DataFrame(rows, columns=["a", "b", "score"])
    t["key"] = slb.pair_key(t.a, t.b).values
    t = t.groupby("key", as_index=False).score.mean()
    t.to_parquet(cache)
    return t


def main(split):
    t = table().set_index("key").score
    d = slb.load(split)
    out = d[["example_id", "species"]].copy()
    out["score"] = np.nan
    h = d.species == "human"
    out.loc[h, "score"] = t.reindex(slb.pair_key(d.gene_a[h], d.gene_b[h]).values).to_numpy()
    print("coverage:", slb.coverage(out), file=sys.stderr)
    slb.write(out, "llmsynthlet__released", split)
    slb.evaluate("llmsynthlet__released", split)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else slb.SPLIT)
