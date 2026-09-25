"""Integration checks on the built benchmark (skipped if the current benchmark version is not built)."""
import polars as pl
import pytest

from slbench.evaluate import BENCH

from pathlib import Path

pytestmark = pytest.mark.skipif(not (BENCH / "manifest.json").exists(), reason="benchmark not built")
private = pytest.mark.skipif(not (BENCH / "hidden/test_labels.parquet").exists(), reason="needs the private test labels")
raw = pytest.mark.skipif(not Path("data/raw/ids/hgnc_complete_set.txt").exists(), reason="needs the raw sources")


def test_splits_are_family_disjoint():
    fam = pl.read_parquet(BENCH / "held_out_families.parquet")
    for split, allowed in [("train.parquet", {"train"}), ("dev_inputs.parquet", {"dev"}), ("test_inputs.parquet", {"test"})]:
        d = pl.read_parquet(BENCH / split).select("species", "gene_a", "gene_b")
        for col in ("gene_a", "gene_b"):
            b = d.join(fam.rename({"gene": col}), on=["species", col], how="left")["bucket"]
            assert set(b.unique().to_list()) <= allowed, (split, col)


@raw
def test_qualifying_homology_edges_never_cross_family_buckets():
    from slbench.families import edges

    fam = pl.read_parquet(BENCH / "held_out_families.parquet").with_columns(
        pl.concat_str("species", "gene", separator=":").alias("node"))
    left = fam.select(pl.col("node").alias("u"), pl.col("family").alias("fa"), pl.col("bucket").alias("ba"))
    right = fam.select(pl.col("node").alias("v"), pl.col("family").alias("fb"), pl.col("bucket").alias("bb"))
    linked = edges().select("u", "v").join(left, on="u", how="inner").join(right, on="v", how="inner")
    assert linked.height > 0
    assert (linked["fa"] == linked["fb"]).all()
    assert (linked["ba"] == linked["bb"]).all()


def test_no_example_in_two_splits():
    ids = [pl.read_parquet(BENCH / f)["example_id"] for f in
           ["train.parquet", "dev_inputs.parquet", "dev_semi_inputs.parquet", "test_inputs.parquet", "test_semi_inputs.parquet"]]
    allids = pl.concat(ids)
    assert allids.n_unique() == allids.len()


@private
def test_both_classes_every_headline_species_in_test():
    from slbench.evaluate import MIN_GROUP_POS, load_split, species_tiers
    t = load_split("test")
    c = t.group_by("species").agg((pl.col("label") == 1).sum().alias("p"), (pl.col("label") == 0).sum().alias("n"))
    c = c.filter(pl.col("species").is_in(species_tiers()[0]))
    assert c.height == len(species_tiers()[0]) and (c["p"] >= MIN_GROUP_POS).all() and (c["n"] > 0).all()


@raw
def test_leakage_catches_paralog_of_heldout_gene():
    from slbench import leakage
    from slbench.homology import paralogs
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


def test_balance_weights_equalise_single_gene_fitness():
    """Pooled over strata (positive-weighted mean of the per-stratum differences), SLB's weights give SL and
    non-SL pairs the same mean single-loss effects. Individual human strata keep some imbalance (README)."""
    from slbench.evaluate import load_gold
    from slbench.fitness import COVARIATES, covariates

    d = covariates(load_gold("dev"), pl.read_parquet(BENCH / "contexts.parquet"), BENCH)
    pos, w = pl.col("label") == 1, pl.col("_bw")
    for (sp,), g in d.group_by(["species"]):
        if (g["label"] == 1).sum() < 20:
            continue
        for c in COVARIATES:
            x = pl.col(c)
            t = g.drop_nulls(c).group_by("context_id", "sources").agg(
                pos.sum().alias("np"),
                (x.filter(pos).mean() - x.filter(~pos).mean()).alias("raw"),
                ((x * w).filter(pos).sum() / w.filter(pos).sum()
                 - (x * w).filter(~pos).sum() / w.filter(~pos).sum()).alias("bal"),
            ).filter(pl.col("np") > 0)
            sd = g[c].cast(pl.Float64).std()
            r0, bal = (abs((t[k] * t["np"]).sum() / t["np"].sum() / sd) for k in ("raw", "bal"))
            assert bal < 0.03 and (bal < r0 / 3 or r0 < 0.03), (sp, c, r0, bal)
