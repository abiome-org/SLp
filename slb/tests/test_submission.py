"""Adversarial, end-to-end checks of scorer input and held-out metric behavior."""

import numpy as np
import polars as pl
import pytest

from slbench.evaluate import _family_weights, species_result, species_score, validated_join


def _gold():
    return pl.DataFrame({"example_id": ["a", "b", "c"], "label": [1, 0, 0]})


def _pred(ids, scores):
    return pl.DataFrame({"example_id": ids, "score": scores}, schema_overrides={"score": pl.Float64})


@pytest.mark.parametrize("ids,scores", [
    (["a", "a", "b"], [0.8, 0.8, 0.1]),
    (["a", "b", "x"], [0.8, 0.1, 0.2]),
    ([None, "b", "c"], [0.8, 0.1, 0.2]),
    (["a", "b", "c"], [0.8, float("inf"), 0.2]),
    (["a", "b", "c"], [0.8, float("-inf"), 0.2]),
    (["a", "b", "c"], [0.8, float("nan"), 0.2]),
])
def test_submission_rejects_invalid_ids_and_scores(ids, scores):
    with pytest.raises(ValueError):
        validated_join(_gold(), _pred(ids, scores))


def test_submission_requires_complete_coverage_and_records_fill():
    with pytest.raises(ValueError):
        validated_join(_gold(), _pred(["a", "b"], [0.8, 0.1]))
    df, missing = validated_join(_gold(), _pred(["a", "b"], [0.8, 0.1]), allow_missing=True)
    assert missing == 1 and df.height == 3 and df["score"].null_count() == 0


def test_same_family_bootstrap_counts_once():
    counts = np.array([2, 3, 5])
    ia = np.array([0, 0, 1, 2])
    ib = np.array([0, 1, 2, 2])
    assert _family_weights(counts, ia, ib).tolist() == [2, 6, 15, 5]


def test_bootstrap_family_indices_are_canonical(tmp_path, monkeypatch):
    from slbench import evaluate as E

    fam = pl.DataFrame({"species": ["human"] * 3,
                        "gene": ["g1", "g2", "g3"], "family": ["z", "a", "m"]})
    fam.write_parquet(tmp_path / "held_out_families.parquet")
    monkeypatch.setattr(E, "BENCH", tmp_path)
    pairs = pl.DataFrame({"example_id": ["p1", "p2"], "species": ["human"] * 2,
                          "gene_a": ["g1", "g3"], "gene_b": ["g2", "g1"]})
    ia, ib, n = E._family_index(pairs)
    assert (ia.tolist(), ib.tolist(), n) == ([2, 1], [0, 2], 3)


def test_unknown_ancestry_is_its_own_human_group():
    d = pl.DataFrame({
        "context_id": ["eur"] * 40 + ["unknown"] * 40,
        "sources": ["screen"] * 80,
        "ancestry_group": ["EUR"] * 40 + ["unknown"] * 40,
        "label": [1] * 20 + [0] * 20 + [1] * 20 + [0] * 20,
        "score": [0.0] * 20 + [1.0] * 20 + [1.0] * 20 + [0.0] * 20,
        "_bw": [1.0] * 80,
    })
    # lines without an ancestry estimate form their own group, averaged in like the others (scorer 3)
    assert species_score(d, "human") == 0.5


def test_normalized_score_uses_the_stratum_noise_ceiling():
    y = [1] * 20 + [0] * 20
    d = pl.DataFrame({
        "context_id": ["a"] * 40 + ["b"] * 40, "sources": ["s1"] * 40 + ["s2"] * 40,
        "ancestry_group": ["n/a"] * 80, "label": y + y,
        "score": [1.0] * 20 + [0.0] * 20 + [1.0] * 10 + [0.0] * 10 + [0.5] * 20,
        "_bw": [1.0] * 80, "_rel": [0.9] * 40 + [0.7] * 40,
    })
    r = species_result(d, "scer")
    assert abs(r["auroc"] - 0.75) < 1e-12           # strata a: 1.0, b: 0.5, equal pair weights
    assert abs(r["ceiling"] - 0.8) < 1e-12          # pair-weighted mean reliability
    assert abs(r["normalized"] - (0.75 - 0.5) / (0.8 - 0.5)) < 1e-12
    assert np.isnan(species_result(d.drop("_rel"), "scer")["normalized"])
