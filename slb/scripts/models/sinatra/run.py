"""SINaTRA (Jacunski, Dixon & Tatonetti 2015, PLoS Comput Biol 11:e1004506) released human predictions.

SINaTRA trains a classifier on PPI-network topology features of YEAST gene pairs with yeast SL labels
(BioGRID / SGA-derived) and transfers it to the human PPI network ("connectivity homology"). The released
human scores (figshare 1501103/1501105/1501115, CC BY 4.0; Entrez-ID pairs) are used as-is.
LEAKY: the classifier was fitted on yeast SL labels, which include pairs from held-out (cross-species)
SLB families. Code is not released, so no SLB retrain. Human only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import slb  # noqa: E402

FILES = sorted((slb.RAW / "sinatra").glob("JacunskiEtAl_SINaTRA_Human_Filtered_Predictions_*.txt.gz"))


def table(needed_entrez: set) -> pd.DataFrame:
    cache = slb.ROOT / f"external/models/_statsl_cache/sinatra_{slb.BENCH.name}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    ids = pl.Series(sorted(needed_entrez), dtype=pl.Int64)
    parts = []
    for f in FILES:
        df = pl.read_csv(f, separator="\t", has_header=False, new_columns=["e1", "e2", "score"],
                         schema_overrides={"e1": pl.Int64, "e2": pl.Int64, "score": pl.Float64})
        parts.append(df.filter(pl.col("e1").is_in(ids) & pl.col("e2").is_in(ids)))
        print(f.name, df.height, file=sys.stderr)
    t = pl.concat(parts).to_pandas()
    t.to_parquet(cache)
    return t


def main(split):
    h = slb.hgnc().dropna(subset=["entrez_id"])
    sym2ent = dict(zip(h.symbol, h.entrez_id.astype(int)))
    ent2sym = {v: k for k, v in sym2ent.items()}
    genes = set()
    for s in ["train", "dev", "test"]:
        d0 = slb.load(s)
        d0 = d0[d0.species == "human"]
        genes |= set(d0.gene_a) | set(d0.gene_b)
    t = table({sym2ent[g] for g in genes if g in sym2ent})
    t["key"] = slb.pair_key(t.e1.map(ent2sym).fillna("?"), t.e2.map(ent2sym).fillna("?")).values
    s = t.groupby("key").score.max()
    d = slb.load(split)
    out = d[["example_id", "species"]].copy()
    out["score"] = np.nan
    hm = d.species == "human"
    out.loc[hm, "score"] = s.reindex(slb.pair_key(d.gene_a[hm], d.gene_b[hm]).values).to_numpy()
    print("coverage:", slb.coverage(out), file=sys.stderr)
    slb.write(out, "sinatra__released", split)
    slb.evaluate("sinatra__released", split)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else slb.SPLIT)
