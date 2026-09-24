"""Accept/reject and random-noise probes for the frozen SLB grader (dev only)."""

import json
from pathlib import Path

import numpy as np
import polars as pl

from slpbench import evaluate as E


def main() -> None:
    gold = E.load_gold("dev")
    y = gold["label"].to_numpy()
    n = len(y)

    def score(x):
        return E.headline(gold.with_columns(pl.Series("score", x)))[0]

    probes = {
        "label_oracle_accept": score(y.astype(float)),
        "inverse_oracle_reject": score(1.0 - y),
        "constant_reject": score(np.zeros(n)),
    }
    random = [score(np.random.default_rng(seed).random(n)) for seed in range(20)]
    probes["random_mean"] = float(np.mean(random))
    probes["random_sd"] = float(np.std(random, ddof=1))
    probes["random_min"] = float(np.min(random))
    probes["random_max"] = float(np.max(random))
    prior = gold.with_columns(pl.col("label").mean().over("context_id", "sources").alias("prior"))["prior"].to_numpy()
    probes["stratum_hit_rate_prior_reject"] = score(prior)
    fit = E.read_predictions("results/slb1.3/fitness_lgbm_dev.parquet")
    probes["fitness_only_probe"] = E.evaluate(fit, "dev")["slb_score"]
    assert np.isclose(probes["label_oracle_accept"], 1.0)
    assert np.isclose(probes["inverse_oracle_reject"], 0.0)
    assert np.isclose(probes["constant_reject"], 0.5)
    assert np.isclose(probes["stratum_hit_rate_prior_reject"], 0.5)
    assert abs(probes["random_mean"] - 0.5) < 0.04
    assert abs(probes["fitness_only_probe"] - 0.5) < 0.05
    Path("reference/grader_probe_slb1.3.json").write_text(json.dumps(probes, indent=2) + "\n")
    print(json.dumps(probes, indent=2))


if __name__ == "__main__":
    main()
