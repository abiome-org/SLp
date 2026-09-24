"""Cohort and scorer checks for the matched ancestry diagnostic."""

import numpy as np
import polars as pl
import pytest

from slpbench import ancestry as A
from slpbench import evaluate as E


def test_precision_at_k_is_order_independent_at_tie_boundary():
    y = np.array([1, 0, 1, 0, 0, 1])
    score = np.array([.9, .9, .8, .8, .8, .1])
    expected = (1 + 2 / 3) / 4  # one .9 hit, plus 2/3 expected from two .8 slots
    for order in (np.arange(6), np.arange(6)[::-1], np.array([4, 2, 1, 5, 0, 3])):
        assert A.precision_at_k_ties(y[order], score[order], 4) == pytest.approx(expected)


def test_exact_panel_matching_and_line_macro_reject_group_prior():
    rows = []
    for context, group, scores in [
        ("human:RKO", "AFR", [3.0, 2.0, 1.0]),
        ("human:COLO 678", "EUR", [1.0, 2.0, 3.0]),
        ("human:HT-29", "EUR", [3.0, 2.0, 1.0]),
    ]:
        for i, (gene_a, gene_b) in enumerate([("A", "B"), ("C", "D"), ("E", "F")]):
            rows.append((f"{context}:{i}", "human", context, group, gene_a, gene_b,
                         "flister2025", int(i == 0), .5, scores[i]))
    rows.append(("extra", "human", "human:COLO 678", "EUR", "G", "H",
                 "flister2025", 1, .5, 100.0))
    d = pl.DataFrame(rows, schema=["example_id", "species", "context_id", "ancestry_group",
                                   "gene_a", "gene_b", "sources", "label", "propensity", "score"],
                     orient="row")
    meta = pl.DataFrame({"species": ["human"] * 3,
                         "context_id": ["human:RKO", "human:COLO 678", "human:HT-29"],
                         "disease": ["Colon carcinoma", "Colon carcinoma", "Colon adenocarcinoma"],
                         "ancestry_basis": ["genotype"] * 3,
                         "cellosaurus_ac": ["CVCL_A", "CVCL_B", "CVCL_C"],
                         "anc_AFR": [.8, .0, .0], "anc_EAS": [.0, .0, .0],
                         "anc_EUR": [.2, 1.0, 1.0]})
    cfg = A.protocol()
    cfg["target_groups"] = ["AFR"]
    bench = A.AncestryBenchmark(d.drop("score"), meta, cfg)
    assert bench.support["AFR"]["by_group"]["EUR"]["pairs"] == 6
    assert bench.support["AFR"]["blocks"][0]["shared_pairs"] == 3
    measured = bench.evaluate(d.select("example_id", "score"))["AFR"]
    assert measured["auroc"] == {"AFR": 1.0, "EUR": .5}
    assert measured["gap_auroc"] == .5
    assert measured["verdict"] == "insufficient_support"
    prior = d.select("example_id", "ancestry_group").with_columns(
        pl.when(pl.col("ancestry_group") == "AFR").then(1.0).otherwise(0.0).alias("score"))
    assert bench.evaluate(prior)["AFR"]["auroc"] == {"AFR": .5, "EUR": .5}


def test_donor_gate_and_bootstrap_on_adequate_synthetic_cohort():
    cfg = A.protocol()
    cfg["inference"]["bootstrap_reps"] = 200
    blocks = []
    for site in ("lung", "colorectal"):
        lines = [{"group": group, "context_id": f"{group}:{site}:{i}",
                  "pairs": 100, "sl": 10, "auroc": .7 if group == "AFR" else .8}
                 for group in ("AFR", "EUR") for i in range(10)]
        blocks.append({"cancer_site": site, "sources": "same_screen",
                       "complete_case_pairs": 100, "scorable": True, "lines": lines})
    support = {"blocks": blocks}
    gate = A.adequacy(support, cfg, "AFR", "EUR", donor_disjoint_training=True,
                      variant_aware_guide_qc=True)
    assert gate["passed"]
    assert A._bootstrap_gap(blocks, "AFR", "EUR", cfg) == pytest.approx([-.1, -.1])
    blocked = A.adequacy(support, cfg, "AFR", "EUR", donor_disjoint_training=False,
                         variant_aware_guide_qc=True)
    assert blocked["failures"] == ["test_cell_lines_not_proven_disjoint_from_model_training"]


@pytest.mark.skipif(not (E.BENCH / "hidden/test_labels.parquet").exists(), reason="private test not built")
def test_frozen_matched_support_and_adequacy_gate():
    bench = A.AncestryBenchmark(E.load_gold("test"), pl.read_parquet(E.BENCH / "contexts.parquet"))
    assert bench.support["AFR"]["by_group"]["AFR"] == {"pairs": 626, "sl": 23, "cell_lines": 2}
    assert bench.support["AFR"]["by_group"]["EUR"] == {"pairs": 3400, "sl": 76, "cell_lines": 14}
    assert {b["cancer_site"] for b in bench.support["AFR"]["blocks"]} == {"colorectal", "lung"}
    assert {(b["cancer_site"], b["complete_case_pairs"])
            for b in bench.support["AFR"]["blocks"]} == {("colorectal", 165), ("lung", 0)}
    assert bench.complete_case_support["AFR"]["by_group"]["AFR"]["pairs"] == 165
    prior = E.load_gold("test").select("example_id", "ancestry_group").with_columns(
        pl.when(pl.col("ancestry_group") == "AFR").then(2.0)
        .when(pl.col("ancestry_group") == "EAS").then(1.0).otherwise(0.0).alias("score"))
    scores = bench.evaluate(prior)
    for target in ("AFR", "EAS"):
        assert list(scores[target]["auroc"].values()) == pytest.approx([.5, .5])
    assert scores["AFR"]["verdict"] == "insufficient_support"
