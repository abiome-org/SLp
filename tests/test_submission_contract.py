"""Adversarial, end-to-end checks of scorer input and held-out metric behavior."""

import numpy as np
import polars as pl
import pytest

from slpbench.evaluate import _family_weights, species_score, validated_join


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


def test_unknown_ancestry_is_excluded_from_human_headline():
    d = pl.DataFrame({
        "context_id": ["eur"] * 40 + ["unknown"] * 40,
        "sources": ["screen"] * 80,
        "ancestry_group": ["EUR"] * 40 + ["unknown"] * 40,
        "label": [1] * 20 + [0] * 20 + [1] * 20 + [0] * 20,
        "score": [0.0] * 20 + [1.0] * 20 + [1.0] * 20 + [0.0] * 20,
        "_bw": [1.0] * 80,
    })
    assert species_score(d, "human") == 0.0
