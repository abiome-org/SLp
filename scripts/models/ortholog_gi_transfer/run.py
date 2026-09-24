"""Cross-species GI transfer through orthologs (yeast <-> pombe <-> human), the idea behind Slorth-style
and "conserved genetic interaction" SL prediction (e.g. Dixon 2008, Roguev 2008, Srivas 2016 transfer).

LEAKY by construction: SLB families contain cross-species orthologs, so the ortholog pairs of a dev pair
are themselves dev-family pairs, and their measured GI scores come from the combinatorial screens behind
SLB (including non-included sources). Scored only as a diagnostic of GI conservation; not rankable.

Score for target pair (species s, a, b): over every other species t and every ortholog pair (a', b') of
(a, b) in t measured by a source in data/interim/measurements, take -z where z is the source-standardised
GI score (score / source SD; more negative = stronger negative GI); report the max over matches.
Pairs with no measured ortholog pair get NaN (filled with 0 = neutral in the output; in_model=False).

    SLB_BENCH=... SLB_SPLIT=dev scripts/models/ortholog_gi_transfer/run.sh
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from slpbench.families import edges

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(os.environ.get("SLB_BENCH", ROOT / "data/bench/slb1.2"))
BENCH = BENCH if BENCH.is_absolute() else ROOT / BENCH
MEAS = ROOT / "data/interim/measurements"
# every measurement table in data/interim/measurements; all use "more negative = stronger negative GI / SL"
SOURCES = sorted(p.stem for p in MEAS.glob("*.parquet"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=os.environ.get("SLB_SPLIT", "dev"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    orth = edges()
    try:  # cross-species orthologs for species added after SLB-1.2 (data-orthology)
        from slpbench.families_extra import extra_edges
        orth = pl.concat([orth, extra_edges().select(orth.columns)])
    except Exception as e:  # noqa: BLE001
        print("families_extra unavailable:", e)
    orth = orth.filter(pl.col("kind") == "ortholog")
    omap = defaultdict(set)
    for u, v in zip(orth["u"], orth["v"]):
        omap[u].add(v)
        omap[v].add(u)
    # best (most negative standardised) measured score per (species, unordered pair), over sources
    best: dict[tuple[str, str, str], float] = {}
    for s in SOURCES:
        p = MEAS / f"{s}.parquet"
        if not p.exists():
            continue
        m = pl.read_parquet(p, columns=["species", "gene_a", "gene_b", "score"]).drop_nulls()
        m = m.filter(pl.col("score").is_finite())
        sd = float(m["score"].std())
        m = m.with_columns((pl.col("score") / sd).alias("z"),
                           pl.min_horizontal("gene_a", "gene_b").alias("x"), pl.max_horizontal("gene_a", "gene_b").alias("y"))
        m = m.group_by("species", "x", "y").agg(pl.col("z").min())
        for sp, x, y, z in m.iter_rows():
            k = (sp, x, y)
            if z < best.get(k, np.inf):
                best[k] = z
        print(f"{s}: {m.height:,} pairs (sd {sd:.3f})", flush=True)
    fn = BENCH / (f"{a.split}.parquet" if (BENCH / f"{a.split}.parquet").exists() else f"{a.split}_inputs.parquet")
    ev = pl.read_parquet(fn)
    sc = np.full(ev.height, np.nan)
    for i, (sp, ga, gb) in enumerate(ev.select("species", "gene_a", "gene_b").iter_rows()):
        oa, ob = omap.get(f"{sp}:{ga}", ()), omap.get(f"{sp}:{gb}", ())
        vals = []
        for u in oa:
            su, gu = u.split(":", 1)
            for v in ob:
                sv, gv = v.split(":", 1)
                if su != sv or su == sp:
                    continue
                z = best.get((su, min(gu, gv), max(gu, gv)))
                if z is not None:
                    vals.append(-z)
        if vals:
            sc[i] = max(vals)
    out = pl.DataFrame({"example_id": ev["example_id"], "raw": sc}).with_columns(
        pl.col("raw").is_not_nan().alias("in_model"), pl.col("raw").fill_nan(0.0).alias("score"))
    print(out.join(ev.select("example_id", "species"), on="example_id").group_by("species")
          .agg(pl.len(), pl.col("in_model").sum()))
    outdir = Path(os.environ.get("SLB_OUT") or ("results/models" if BENCH.name == "slb1.2" else f"results/models/{BENCH.name}"))
    path = Path(a.out or (outdir if outdir.is_absolute() else ROOT / outdir) / f"ortholog_gi_transfer_{a.split}.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path)
    print("wrote", path)


if __name__ == "__main__":
    main()
