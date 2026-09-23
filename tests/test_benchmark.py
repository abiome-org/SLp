"""Integration checks on the built benchmark (skipped if the current benchmark version is not built)."""
from pathlib import Path

import polars as pl
import pytest

from slpbench.evaluate import BENCH  # noqa: E402
pytestmark = pytest.mark.skipif(not (BENCH / "manifest.json").exists(), reason="benchmark not built")


def test_splits_are_family_disjoint():
    fam = pl.read_parquet(BENCH / "held_out_families.parquet")
    for split, allowed in [("train.parquet", {"train"}), ("dev.parquet", {"dev"}), ("test_inputs.parquet", {"test"})]:
        d = pl.read_parquet(BENCH / split).select("species", "gene_a", "gene_b")
        for col in ("gene_a", "gene_b"):
            b = d.join(fam.rename({"gene": col}), on=["species", col], how="left")["bucket"]
            assert set(b.unique().to_list()) <= allowed, (split, col)


def test_no_example_in_two_splits():
    ids = [pl.read_parquet(BENCH / f)["example_id"] for f in
           ["train.parquet", "dev.parquet", "dev_semi.parquet", "test_inputs.parquet", "test_semi_inputs.parquet"]]
    allids = pl.concat(ids)
    assert allids.n_unique() == allids.len()


def test_both_classes_every_species_in_test():
    from slpbench.evaluate import load_split
    t = load_split("test")
    c = t.group_by("species").agg((pl.col("label") == 1).sum().alias("p"), (pl.col("label") == 0).sum().alias("n"))
    assert c.height == 4 and (c["p"] > 0).all() and (c["n"] > 0).all()


def test_leakage_catches_paralog_of_heldout_gene():
    from slpbench import leakage
    from slpbench.homology import paralogs
    fam = pl.read_parquet(BENCH / "held_out_families.parquet").filter(pl.col("species") == "human")
    test_g = set(fam.filter(pl.col("bucket") == "test")["gene"])
    known = set(fam["gene"])
    train_g = sorted(set(fam.filter(pl.col("bucket") == "train")["gene"]))[:2]
    p = paralogs().filter((pl.col("species") == "human") & (pl.col("identity") >= 0.3))
    novel = next(b if a in test_g else a for _, a, b, _ in p.iter_rows()
                 if (a in test_g and b not in known) or (b in test_g and a not in known))
    rec = pl.DataFrame({"species": ["human", "human"], "gene_a": [novel, train_g[0]], "gene_b": [train_g[1], train_g[1]]})
    leaks = leakage.check(rec)
    assert leaks.height == 1 and novel in (leaks["gene_a"][0], leaks["gene_b"][0])
