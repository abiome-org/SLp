"""Accept/reject and random-noise probes for the frozen SLB grader (dev only).

Scores are the normalized SLB score (scorer 3): 0 = chance, 1 = an independent re-measurement. The exact
labels therefore score above 1 (they also carry the measurement noise); raw balanced AUROCs are reported
alongside under raw_*.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from slbench import evaluate as E

PUBLIC = {"dev": "dev_inputs.parquet", "test": "test_inputs.parquet"}


def public_inputs(split: str) -> pl.DataFrame:
    """The split's public input file, without labels or sources."""
    return pl.read_parquet(E.BENCH / PUBLIC[split], columns=["example_id", "species", "context_id", "gene_a", "gene_b"])


def gene_degree(inputs: pl.DataFrame) -> pl.DataFrame:
    """score = -(rows in the same context involving gene_a + rows involving gene_b), from inputs only."""
    k = ["species", "context_id", "gene"]
    deg = pl.concat([inputs.select("example_id", *k[:2], pl.col(g).alias("gene")) for g in ("gene_a", "gene_b")]) \
        .unique().group_by(k).agg(pl.len().alias("d"))
    return inputs.join(deg.rename({"gene": "gene_a", "d": "da"}), on=k[:2] + ["gene_a"], how="left") \
        .join(deg.rename({"gene": "gene_b", "d": "db"}), on=k[:2] + ["gene_b"], how="left") \
        .select("example_id", (-(pl.col("da") + pl.col("db")).cast(pl.Float64)).alias("score"))


def main() -> None:
    gold = E.load_gold("dev")
    y = gold["label"].to_numpy()
    n = len(y)

    def score(x):
        return E.headline(gold.with_columns(pl.Series("score", x)))[0]

    def raw(x):
        return E.headline(gold.with_columns(pl.Series("score", x)), raw=True)[0]

    probes = {
        "label_oracle_accept": score(y.astype(float)),
        "inverse_oracle_reject": score(1.0 - y),
        "constant_reject": score(np.zeros(n)),
        "raw_label_oracle": raw(y.astype(float)),
        "raw_inverse_oracle": raw(1.0 - y),
        "raw_constant": raw(np.zeros(n)),
        "dev_ceilings": E.ceilings(gold, E.species_tiers()[0]),
    }
    random = [score(np.random.default_rng(seed).random(n)) for seed in range(20)]
    probes["random_mean"] = float(np.mean(random))
    probes["random_sd"] = float(np.std(random, ddof=1))
    probes["random_min"] = float(np.min(random))
    probes["random_max"] = float(np.max(random))
    prior = gold.with_columns(pl.col("label").mean().over("context_id", "sources").alias("prior"))["prior"].to_numpy()
    probes["stratum_hit_rate_prior_reject"] = score(prior)
    deg = gold.select("example_id").join(gene_degree(public_inputs("dev")), on="example_id", how="left",
                                         maintain_order="left")["score"].to_numpy()
    probes["gene_degree_reject"] = score(deg)
    fit = E.read_predictions("results/slb/fitness_lgbm_dev.parquet")
    probes["fitness_only_probe"] = E.evaluate(fit, "dev")["slb_score"]
    assert np.isclose(probes["raw_label_oracle"], 1.0) and probes["label_oracle_accept"] > 1.0
    assert np.isclose(probes["raw_inverse_oracle"], 0.0) and probes["inverse_oracle_reject"] < -1.0
    assert np.isclose(probes["constant_reject"], 0.0)
    assert np.isclose(probes["stratum_hit_rate_prior_reject"], 0.0)
    assert abs(probes["random_mean"]) < 0.1
    assert abs(probes["fitness_only_probe"]) < 0.15
    Path("reference/grader_probe.json").write_text(json.dumps(probes, indent=2) + "\n")
    print(json.dumps(probes, indent=2))


if __name__ == "__main__":
    main()
