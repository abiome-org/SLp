"""Collect every number the SLB figures plot into figures/data/ (run from slb/: uv run python figures/prepare_data.py).

Reads the built benchmark (data/slb), parsed measurements (data/interim), result JSON (results/), the
replication report and the robustness builds. Uses train/dev labels and aggregate test scores only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from slbench import evaluate as E
from slbench import families, fitness, genome

OUT = Path("figures/data")
MEAS = Path("data/interim/measurements")
SPECIES = ["human", "mmus", "dmel", "cele", "scer", "spom", "bsub"]


def hero_graph() -> None:
    """Benchmark genes (species, gene, family, bucket) and the homology edges between them."""
    fam = pl.read_parquet(E.BENCH / "held_out_families.parquet")
    nodes = set(fam["species"] + ":" + fam["gene"])
    e = pl.concat([families.edges(), families.overlap_edges(fam.select("species", "gene"))])
    e = e.filter(pl.col("u").is_in(list(nodes)) & pl.col("v").is_in(list(nodes)) & (pl.col("u") != pl.col("v")))
    e = e.with_columns(pl.min_horizontal("u", "v").alias("u"), pl.max_horizontal("u", "v").alias("v")).unique(["u", "v"])
    key = fam.select((pl.col("species") + ":" + pl.col("gene")).alias("n"), "family", "bucket")
    e = e.join(key.rename({"n": "u"}), on="u").join(key.rename({"n": "v", "family": "fv", "bucket": "bv"}), on="v")
    fam.write_parquet(OUT / "hero_nodes.parquet")
    e.select("u", "v", "kind", "bucket").write_parquet(OUT / "hero_edges.parquet")
    print("hero", fam.height, "genes", e.height, "edges", e.group_by("bucket").len().rows())


def screen_matrices(bins: int = 700) -> None:
    """Gene x gene measured/SL density for four screens, genes ordered by their SL-partner count."""
    specs = [("scer", "costanzo2016", "costanzo2016", "S. cerevisiae", "Costanzo 2016 · SGA"),
             ("spom", "ryan2012", "ryan2012", "S. pombe", "Ryan 2012 · E-MAP"),
             ("bsub", "koo2025", "koo2025", "B. subtilis", "Koo 2025 · dual CRISPRi"),
             ("human", "slkb", None, "H. sapiens", "8 paralog screens · 50 cell lines")]
    out = {}
    for sp, f, src, name, sub in specs:
        m = pl.read_parquet(MEAS / f"{f}.parquet") if sp != "human" else pl.concat(
            [pl.read_parquet(MEAS / f"{x}.parquet").select("species", "source", "gene_a", "gene_b", "label")
             for x in ("slkb", "ryanlab_zdlfc", "chou2025", "flister2025", "harle2025", "spidr2025")])
        m = m.filter(pl.col("species") == sp)
        if src:
            m = m.filter(pl.col("source") == src)
        m = m.group_by("gene_a", "gene_b").agg(pl.col("label").max().alias("label"))
        pos = m.filter(pl.col("label") == 1)
        deg = pl.concat([pos["gene_a"], pos["gene_b"]]).value_counts()
        genes = pl.concat([m["gene_a"], m["gene_b"]]).unique().to_frame("g").join(
            deg.rename({"": "g"} if "" in deg.columns else {deg.columns[0]: "g"}), on="g", how="left") \
            .with_columns(pl.col("count").fill_null(0)).sort(["count", "g"], descending=[True, False])
        rank = dict(zip(genes["g"], range(genes.height)))
        a = np.array([rank[x] for x in m["gene_a"]]); b = np.array([rank[x] for x in m["gene_b"]])
        n = genes.height; nb = min(bins, n)
        ia, ib = (a * nb // n), (b * nb // n)
        meas = np.zeros((nb, nb)); sl = np.zeros((nb, nb)); neg = np.zeros((nb, nb))
        lab = m["label"].to_numpy()
        for arr, mask in ((meas, np.ones(len(a), bool)), (sl, lab == 1), (neg, lab == 0)):
            np.add.at(arr, (ia[mask], ib[mask]), 1); np.add.at(arr, (ib[mask], ia[mask]), 1)
        out[sp] = dict(meas=meas, sl=sl, neg=neg)
        (OUT / f"screen_{sp}.json").write_text(json.dumps(
            {"species": name, "subtitle": sub, "genes": n, "pairs": m.height, "sl": int((lab == 1).sum()),
             "neg": int((lab == 0).sum())}))
        print("screen", sp, n, m.height)
    np.savez_compressed(OUT / "screens.npz", **{f"{sp}_{k}": v for sp, d in out.items() for k, v in d.items()})


def human_contexts(n_pairs: int = 1400) -> None:
    """Human cell line x gene pair: measured and SL, for the most widely measured pairs; lines grouped by ancestry."""
    from slbench.build import EXCLUDED_SOURCES, _context_table

    m = pl.concat([pl.read_parquet(MEAS / f"{x}.parquet").select("species", "source", "context", "gene_a", "gene_b", "label")
                   for x in ("slkb", "ryanlab_zdlfc", "chou2025", "flister2025", "harle2025", "spidr2025")])
    m = m.filter((pl.col("species") == "human") & ~pl.col("source").is_in(list(EXCLUDED_SOURCES)))
    ctx = _context_table(m)
    m = m.join(ctx.select("source", "context", "context_id", "ancestry_group"), on=["source", "context"])
    m = m.group_by("context_id", "ancestry_group", "gene_a", "gene_b").agg(pl.col("label").max().alias("label"))
    pairs = m.group_by("gene_a", "gene_b").agg(pl.len().alias("n"), (pl.col("label") == 1).mean().alias("sl")) \
        .sort(["n", "sl"], descending=True).head(n_pairs).sort("sl", descending=True).with_row_index("j")
    order = {"AFR": 0, "EAS": 1, "EUR": 2, "unknown": 3}
    lines = m.group_by("context_id", "ancestry_group").agg((pl.col("label") == 1).mean().alias("rate")) \
        .with_columns(pl.col("ancestry_group").replace_strict(order, default=4).alias("o")).sort(["o", "rate"], descending=[False, True]) \
        .with_row_index("i")
    x = m.join(pairs.select("gene_a", "gene_b", "j"), on=["gene_a", "gene_b"]).join(lines.select("context_id", "i"), on="context_id")
    mat = np.full((lines.height, pairs.height), np.nan)
    mat[x["i"].to_numpy(), x["j"].to_numpy()] = x["label"].fill_null(-1).to_numpy()
    np.save(OUT / "human_ctx.npy", mat)
    lines.select("i", "context_id", "ancestry_group").write_csv(OUT / "human_ctx_lines.csv")
    print("human contexts", mat.shape, int(np.nansum(mat == 1)))


def label_bands() -> None:
    """Score distributions with the SLB label of each measured pair, for three representative screens."""
    rows, totals = [], {}
    for f, src, name in (("costanzo2016", "costanzo2016", "Costanzo 2016 SGA (ε)"),
                         ("ryan2012", "ryan2012", "Ryan 2012 E-MAP (S)"),
                         ("ryanlab_zdlfc", "dede2020", "Dede 2020 (zdLFC)")):
        m = pl.read_parquet(MEAS / f"{f}.parquet").filter(pl.col("source") == src).drop_nulls("score")
        totals[name] = m.height
        if m.height > 3_000_000:
            m = m.sample(3_000_000, seed=0)
        rows.append(m.select(pl.lit(name).alias("screen"), "score", "label"))
    pl.concat(rows).write_parquet(OUT / "label_bands.parquet")
    (OUT / "label_bands_totals.json").write_text(json.dumps(totals))


def replication() -> None:
    """Per-source reproducibility evidence, parsed from the audit's decision table."""
    text = Path("results/reports/replication.md").read_text()
    table = text.split("| source | cross-study AUROC", 1)[1].split("\n\n", 1)[0].splitlines()[2:]
    rows = []
    for line in table:
        c = [x.strip() for x in line.strip("|").split("|")]
        def rng(s):
            v = [float(x) for x in re.findall(r"\d\.\d+", s)]
            return (min(v), max(v)) if v else (None, None)
        cross, within = rng(c[1]), rng(c[3])
        rows.append({"source": c[0], "cross_lo": cross[0], "cross_hi": cross[1], "within_lo": within[0],
                     "within_hi": within[1], "vs_included": float(c[2]) if c[2] else None,
                     "included": c[5] == "yes", "reason": c[6]})
    pl.DataFrame(rows).write_csv(OUT / "replication.csv")


def linkage() -> None:
    """SL rate against same-chromosome distance for the yeast screens (measured labels, before unscoring)."""
    rows = []
    for sp, files in (("scer", ("costanzo2016", "scer_emaps", "kuzmin2018", "kuzmin2020")), ("spom", ("ryan2012", "frost2012"))):
        for f in files:
            m = pl.read_parquet(MEAS / f"{f}.parquet").filter(pl.col("label").is_not_null()).select("gene_a", "gene_b", "label")
            d = genome.pair_distance(m, sp)
            m = m.with_columns(d.alias("dist"))
            rows.append(m.with_columns(pl.lit(sp).alias("species"), pl.lit(f).alias("source")))
    x = pl.concat(rows)
    edges = np.array([0, 25e3, 50e3, 75e3, 100e3, 150e3, 200e3, 300e3, 450e3, 700e3, 1.1e6, 1.6e6])
    x = x.with_columns(pl.when(pl.col("dist").is_null()).then(-1).otherwise(
        pl.col("dist").cut(edges[1:].tolist(), labels=[str(i) for i in range(len(edges))]).cast(pl.Int32)).alias("bin"))
    g = x.group_by("species", "source", "bin").agg(pl.len().alias("n"), pl.col("label").mean().alias("rate"),
                                                  pl.col("dist").median().alias("dist_med"))
    g.sort("species", "source", "bin").write_csv(OUT / "linkage.csv")
    np.savetxt(OUT / "linkage_edges.txt", edges)


def funnel() -> None:
    rep = json.loads(Path("data/interim/examples_report.json").read_text())
    man = json.loads((E.BENCH / "manifest.json").read_text())
    rep["splits"] = man["counts"]
    (OUT / "funnel.json").write_text(json.dumps(rep, indent=1))


def balance() -> None:
    """|standardised mean difference| of each balance covariate between SL and non-SL pairs, raw and weighted (dev)."""
    part = fitness.with_degree(E.load_split("dev"))
    ctx = pl.read_parquet(E.BENCH / "contexts.parquet")
    d = fitness.covariates(part, ctx).join(E.load_gold("dev").select("example_id", "_bw"), on="example_id", how="inner")
    pos, w = pl.col("label") == 1, pl.col("_bw")
    rows = []
    for (sp,), g in d.group_by(["species"]):
        if (g["label"] == 1).sum() < 20:
            continue
        for c in fitness.BALANCE_COVARIATES:
            x = pl.col(c)
            t = g.drop_nulls(c).group_by("context_id", "sources").agg(
                pos.sum().alias("np"),
                (x.filter(pos).mean() - x.filter(~pos).mean()).alias("raw"),
                ((x * w).filter(pos).sum() / w.filter(pos).sum() - (x * w).filter(~pos).sum() / w.filter(~pos).sum()).alias("bal"),
            ).filter(pl.col("np") > 0)
            sd = g[c].cast(pl.Float64).std()
            if not sd:
                continue
            raw, bal = (abs(float((t[k] * t["np"]).sum() / t["np"].sum() / sd)) for k in ("raw", "bal"))
            rows.append({"species": sp, "covariate": c, "raw": raw, "balanced": bal})
    pl.DataFrame(rows).write_csv(OUT / "balance.csv")


def _deg_probe(bench: Path, split: str) -> float:
    E.BENCH = bench
    name = f"{split}_inputs.parquet" if (bench / f"{split}_inputs.parquet").exists() else f"{split}.parquet"
    X = pl.read_parquet(bench / name, columns=["example_id", "context_id", "gene_a", "gene_b"])
    n = pl.concat([X.select("context_id", pl.col(c).alias("g")) for c in ("gene_a", "gene_b")]).group_by("context_id", "g").len()
    for s in "ab":
        X = X.join(n.rename({"g": f"gene_{s}", "len": f"n_{s}"}), on=["context_id", f"gene_{s}"], how="left", maintain_order="left")
    p = X.select("example_id", (-(pl.col("n_a") + pl.col("n_b")).cast(pl.Float64)).alias("score"))
    df, _ = E.validated_join(E.load_gold(split), p, inputs=E.input_ids(split))
    return E.headline(df)[0]


def probes() -> None:
    """Reject probes on dev (current build), and the row-count probe on the previous and current builds."""
    new = Path("data/slb")
    gp = json.loads(Path("reference/grader_probe.json").read_text())
    out = {"dev": gp}
    old = Path("/tmp/slb_old")
    out["row_count"] = {"dev_new": _deg_probe(new, "dev"), "test_new": _deg_probe(new, "test")}
    if old.exists():
        out["row_count"] |= {"dev_old": _deg_probe(old, "dev"), "test_old": _deg_probe(old, "test")}
    E.BENCH = new
    (OUT / "probes.json").write_text(json.dumps(out, indent=1))
    print("probes", out["row_count"])


def noise(seeds: int = 60) -> None:
    """SLB score of random per-gene scores (score(a, b) = r_a + r_b), the practical noise floor."""
    rows = []
    for split in ("dev", "test"):
        name = f"{split}_inputs.parquet"
        X = pl.read_parquet(E.BENCH / name, columns=["example_id", "gene_a", "gene_b"])
        gold, ids = E.load_gold(split), E.input_ids(split)
        genes = pl.concat([X["gene_a"], X["gene_b"]]).unique().sort()
        ga = X.select(pl.col("gene_a").alias("g")).join(genes.to_frame("g").with_row_index("i"), on="g", how="left", maintain_order="left")["i"].to_numpy()
        gb = X.select(pl.col("gene_b").alias("g")).join(genes.to_frame("g").with_row_index("i"), on="g", how="left", maintain_order="left")["i"].to_numpy()
        for seed in range(seeds):
            r = np.random.default_rng(seed).standard_normal(genes.len())
            df, _ = E.validated_join(gold, pl.DataFrame({"example_id": X["example_id"], "score": r[ga] + r[gb]}), inputs=ids)
            s, parts = E.headline(df)
            rows.append({"split": split, "seed": seed, "slb": s, **parts})
    pl.DataFrame(rows).write_csv(OUT / "noise.csv")


def robustness() -> None:
    base = ["lgbm", "fitness_lgbm", "fitness", "paralog_identity", "codependency", "random"]
    rows = []
    for b in base:
        r = json.loads(Path(f"results/slb/{b}_test.json").read_text())
        rows.append({"split": "official", "model": b, "slb": r["slb_score"]})
    for d in sorted(Path("data/robustness").iterdir()):
        for b in base:
            r = json.loads((d / f"res_{b}.json").read_text())
            rows.append({"split": d.name, "model": b, "slb": r["slb_score"]})
    pl.DataFrame(rows).write_csv(OUT / "robustness.csv")


def results() -> None:
    """Leaderboard entries with CI, species scores, paralog strata and ancestry groups; dev/test pairs."""
    entries = yaml.safe_load(Path("leaderboard.yaml").read_text())
    rows = []
    for e in entries:
        r = json.loads(Path(e["result"]).read_text())
        st = {s["stratum"]: s for s in r["strata"]}
        def auc(k):
            return (st.get(k) or {}).get("slb_auroc")
        dev = Path(e["predictions"].replace("_test.parquet", "_dev.parquet"))
        dres = E.evaluate(E.read_predictions(dev), "dev", allow_missing=True) if dev.exists() else None
        rows.append({"name": e["name"], "leaky": str(e.get("leaky", False)), "ranked": e.get("ranked", True),
                     "slb": r["slb_score"], "lo": r["slb_score_ci95"][0], "hi": r["slb_score_ci95"][1],
                     **{f"sp_{k}": v for k, v in {**r["species_scores"], **r["auxiliary_species_scores"]}.items()},
                     "paralog": auc("pair=same family (paralogs)"), "nonparalog": auc("pair=different families"),
                     **{f"anc_{g}": auc(f"human ancestry={g}") for g in ("AFR", "EAS", "EUR")},
                     "dev": dres["slb_score"] if dres else None})
    pl.DataFrame(rows, infer_schema_length=None).write_csv(OUT / "leaderboard.csv")
    print(pl.read_csv(OUT / "leaderboard.csv").select("name", "slb", "dev", "paralog").head(8))
    anc = json.loads(Path("reference/ancestry_audit.json").read_text())
    (OUT / "ancestry.json").write_text(json.dumps(anc["models"], indent=1))


if __name__ == "__main__":
    import sys

    OUT.mkdir(parents=True, exist_ok=True)
    steps = {f.__name__: f for f in (hero_graph, screen_matrices, human_contexts, label_bands, replication, linkage, funnel, balance,
                                     probes, noise, robustness, results)}
    for name in sys.argv[1:] or steps:
        print("==", name, flush=True)
        steps[name]()
