"""FBA double-gene-deletion scores for SLB pairs (label-free, mechanistic).

For each species with a GEM (gems.LOADERS), every SLB pair whose two genes are both in the model is
simulated: wild type, both single deletions and the double deletion (GPR-aware knock-out, max biomass).
Relative growth f = growth / wild type.

Scores (higher = more SL):
  fba_min   = min(f_a, f_b) - f_ab       (default: drop below the sicker single mutant)
  fba_mult  = f_a * f_b - f_ab           (negative epsilon vs multiplicative expectation)
Pairs outside the model get 0 (= "no predicted interaction"), so the file scores every example.

    python run.py --split dev [--species scer spom spne human] [--medium default] [--score fba_min]
Writes results/models/fba_<split>.parquet (example_id, score, plus in_model/f_a/f_b/f_ab columns).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gems  # noqa: E402

ROOT = gems.ROOT
BENCH = Path(os.environ.get("SLB_BENCH", ROOT / "data/bench/slb1.2"))
BENCH = BENCH if BENCH.is_absolute() else ROOT / BENCH
CACHE = gems.EXT / "cache"

_M = None


def _init(species, medium):
    global _M
    _M = gems.LOADERS[species](medium)[0]
    _M.solver.configuration.timeout = 60  # glpk occasionally stalls on a knock-out LP; record NaN instead


def _grow(genes: tuple[str, ...]) -> float:
    with _M:
        for g in genes:
            _M.genes.get_by_id(g).knock_out()
        v = _M.slim_optimize(error_value=float("nan"))
    return float(v) if v != v else max(float(v), 0.0)


def _job(key):
    return key, _grow(key)


def simulate(species: str, medium: str, pairs: pd.DataFrame, procs: int) -> pd.DataFrame:
    """pairs: gene_a, gene_b (SLB IDs). Returns pairs in the model with f_a, f_b, f_ab (cached)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cpath = CACHE / f"{species}_{medium}.parquet"
    model, mp_ = gems.LOADERS[species](medium)
    inv: dict[str, list[str]] = {}
    for mg, sg in mp_.items():
        inv.setdefault(sg, []).append(mg)
    p = pairs[pairs.gene_a.isin(inv) & pairs.gene_b.isin(inv)].drop_duplicates(["gene_a", "gene_b"])
    cache = pd.read_parquet(cpath) if cpath.exists() else pd.DataFrame(columns=["key", "growth"])
    known = dict(zip(cache.key, cache.growth))
    keys = {(): ""}
    for a, b in zip(p.gene_a, p.gene_b):
        keys[tuple(sorted(inv[a]))] = a
        keys[tuple(sorted(inv[b]))] = b
        keys[tuple(sorted(set(inv[a]) | set(inv[b])))] = (a, b)
    todo = [k for k in keys if "|".join(k) not in known]
    print(f"{species}/{medium}: {len(p):,} pairs in model; {len(todo):,} new simulations", flush=True)
    if todo:
        t = time.time()
        with mp.get_context("fork").Pool(procs, initializer=_init, initargs=(species, medium)) as pool:
            for i, (k, v) in enumerate(pool.imap_unordered(_job, todo, chunksize=4)):
                known["|".join(k)] = v
                if (i + 1) % 5000 == 0:
                    pd.DataFrame({"key": list(known), "growth": list(known.values())}).to_parquet(cpath)
                    print(f"  {i + 1:,}/{len(todo):,} {time.time() - t:.0f}s", flush=True)
        pd.DataFrame({"key": list(known), "growth": list(known.values())}).to_parquet(cpath)
        print(f"  done in {time.time() - t:.0f}s", flush=True)
    wt = known[""]
    g = lambda ks: known["|".join(sorted(ks))] / wt  # noqa: E731
    p = p.copy()
    p["f_a"] = [g(inv[a]) for a in p.gene_a]
    p["f_b"] = [g(inv[b]) for b in p.gene_b]
    p["f_ab"] = [g(set(inv[a]) | set(inv[b])) for a, b in zip(p.gene_a, p.gene_b)]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--species", nargs="*", default=None, help="default: every species in the split with a GEM")
    ap.add_argument("--medium", nargs="*", default=[], help="species=medium overrides, e.g. scer=rich")
    ap.add_argument("--score", default="fba_min", choices=["fba_min", "fba_mult"])
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    med = dict(gems.DEFAULT_MEDIUM) | dict(x.split("=") for x in a.medium)
    fn = BENCH / (f"{a.split}.parquet" if (BENCH / f"{a.split}.parquet").exists() else f"{a.split}_inputs.parquet")
    d = pd.read_parquet(fn, columns=["example_id", "species", "gene_a", "gene_b"])
    parts = []
    for sp in (a.species or list(d.species.unique())):
        if sp not in gems.LOADERS:
            print(f"no GEM for {sp}; skipped")
            continue
        x = d[d.species == sp]
        if x.empty:
            continue
        sim = simulate(sp, med[sp], x[["gene_a", "gene_b"]], a.procs)
        parts.append(x.merge(sim, on=["gene_a", "gene_b"], how="inner"))
    s = pd.concat(parts) if parts else pd.DataFrame(columns=["example_id", "f_a", "f_b", "f_ab"])
    s["fba_min"] = s[["f_a", "f_b"]].min(axis=1) - s.f_ab
    s["fba_mult"] = s.f_a * s.f_b - s.f_ab
    out = d[["example_id", "species"]].merge(s[["example_id", "f_a", "f_b", "f_ab", "fba_min", "fba_mult"]],
                                             on="example_id", how="left")
    out["in_model"] = out.f_ab.notna()
    out["score"] = out[a.score].fillna(0.0)
    print(out.groupby("species").agg(n=("score", "size"), in_model=("in_model", "sum"),
                                     predicted_sl=("score", lambda v: int((v > 0.01).sum()))))
    outdir = Path(os.environ.get("SLB_OUT") or ("results/models" if BENCH.name == "slb1.2" else f"results/models/{BENCH.name}"))
    path = Path(a.out or (outdir if outdir.is_absolute() else ROOT / outdir) / f"fba_{a.split}.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.drop(columns="species").to_parquet(path)
    print("wrote", path)


if __name__ == "__main__":
    main()
