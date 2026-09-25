"""Reference baselines for SLB.

Unsupervised (no SL labels):
  random            uniform noise
  paralog_identity  max protein sequence identity if the pair are Ensembl 116 paralogs, else 0
  fitness           sickness of the two single mutants: -(f_a + f_b), single-loss effects from
                    fitness.py (DepMap in that line where available, SGA fitness, PomBase viability,
                    single-sgRNA log2FC). SLB balances this away, so it scores ~0.5 by design.
  codependency      correlation of the two genes' DepMap gene-effect profiles (human only)
Supervised (fit on the SLB train split only):
  lgbm              gradient boosting on all features above + species
  fitness_lgbm      the same, on the single-loss covariates only: the strongest fitness-only model,
                    a probe of how much fitness signal the SLB balancing leaves (should be ~0.5)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from slbench import fitness
from slbench.evaluate import BENCH


def features(df: pl.DataFrame) -> pl.DataFrame:
    """Single-loss covariates (fitness.covariates) + paralog identity + DepMap co-dependency, all from the
    benchmark's own files (features/), so the public bundle is enough."""
    from slbench import features as F

    ctx = pl.read_parquet(BENCH / "contexts.parquet")
    out = fitness.covariates(df, ctx, BENCH)
    par = F.read(BENCH, "paralogs").rename({"a": "gene_a", "b": "gene_b", "identity": "paralog_identity"})
    cod = F.read(BENCH, "codependency").with_columns(pl.lit("human").alias("species"))
    out = out.join(par, on=["species", "gene_a", "gene_b"], how="left", maintain_order="left") \
        .join(cod, on=["species", "gene_a", "gene_b"], how="left", maintain_order="left")
    return out.with_columns(pl.col("paralog_identity").fill_null(0.0), (pl.col("f_lo") + pl.col("f_hi")).alias("fit_sum"))


FEATURES = ["f_lo", "f_hi", "pan_lo", "pan_hi", "codependency", "paralog_identity"]


def run(name: str, split: str) -> pl.DataFrame:
    from slbench.evaluate import inputs_path

    df = pl.read_parquet(inputs_path(split, BENCH)).drop("label", "sources", strict=False)
    if name == "random":
        s = np.random.default_rng(0).random(df.height)
    else:
        f = features(df)
        if name == "paralog_identity":
            s = f["paralog_identity"].fill_null(0).fill_nan(0).to_numpy()
        elif name == "fitness":
            s = -f["fit_sum"].fill_nan(None).fill_null(0).to_numpy()
        elif name == "codependency":
            s = f["codependency"].fill_nan(None).fill_null(0).to_numpy()
        elif name == "lgbm":
            s = _lgbm(f, FEATURES)
        elif name == "fitness_lgbm":
            s = _lgbm(f, fitness.COVARIATES)
        else:
            raise SystemExit(f"unknown baseline {name}")
    return pl.DataFrame({"example_id": df["example_id"], "score": s})


def _lgbm(f: pl.DataFrame, feats: list[str]) -> np.ndarray:
    import lightgbm as lgb

    tr = features(pl.read_parquet(BENCH / "train.parquet"))
    # balance species so yeast's 2M rows do not dominate
    tr = pl.concat([d.sample(min(d.height, 150_000), seed=0) for d in tr.partition_by("species")])
    sp = sorted(tr["species"].unique().to_list())

    def mat(d):
        cols = [d[c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in feats]
        cols += [(d["species"] == s).to_numpy().astype(float) for s in sp]
        return np.column_stack(cols)

    w = tr.group_by("species").len().rename({"len": "n"})
    tr = tr.join(w, on="species").with_columns((1.0 / pl.col("n")).alias("w"))
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0, n_jobs=8)
    m.fit(mat(tr), tr["label"].to_numpy(), sample_weight=tr["w"].to_numpy() * tr.height)
    return m.predict_proba(mat(f))[:, 1]
