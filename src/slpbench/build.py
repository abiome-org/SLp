"""Build SLB-1: measurements -> examples -> family-held-out splits.

    uv run python -m slpbench.build            # all stages
    uv run python -m slpbench.build --stage measurements|examples|splits
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import polars as pl

from slpbench import contexts, families
from slpbench.sources import bacteria, dmel, human, yeast

VERSION = "slb1"
SALT = "slb1-2026-09-22"
INTERIM = Path("data/interim")
OUT = Path("data/bench") / VERSION

# fraction of families per bucket
TEST_FRAC, DEV_FRAC = 0.20, 0.15

PARSERS = {
    "slkb": human.slkb,
    "ryanlab_zdlfc": human.ryanlab_zdlfc,
    "chou2025": human.chou2025,
    "spidr2025": human.spidr2025,
    "harle2025": human.harle2025,
    "flister2025": human.flister2025,
    "costanzo2016": yeast.costanzo2016,
    "ryan2012": yeast.ryan2012,
    "fischer2015": dmel.fischer2015,
    "heigwer2023": dmel.heigwer2023,
    "dualcrispri2025": bacteria.dualcrispri2025,
}


RAW = Path("data/raw")


def stage_prepare() -> None:
    """Unpack archives the parsers read: SLKB SQLite dump and Costanzo pairwise files."""
    import shutil
    import sqlite3
    import zipfile

    db = INTERIM / "slkb/slkb.sqlite"
    if not db.exists():
        db.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(RAW / "slkb/SQL_Dumps.zip") as z:
            sql = z.read("SQL_Dumps/SLKB-sqlite3_dump.sql").decode()
        con = sqlite3.connect(db)
        con.executescript(sql)
        con.close()
        print(f"loaded {db}")
    s1 = INTERIM / "costanzo/S1"
    if not s1.exists():
        tmp = INTERIM / "costanzo/_unzip"
        with zipfile.ZipFile(RAW / "costanzo2016_scer/pairwise.zip") as z:
            z.extractall(tmp)
        inner = next(p for p in tmp.iterdir() if p.is_dir() and not p.name.startswith("__"))
        shutil.move(str(inner), s1)
        shutil.rmtree(tmp)
        print(f"unpacked {s1}")


def stage_measurements(only: list[str] | None = None) -> None:
    d = INTERIM / "measurements"
    d.mkdir(parents=True, exist_ok=True)
    for name, fn in PARSERS.items():
        if only and name not in only:
            continue
        t = time.time()
        df = fn()
        df.write_parquet(d / f"{name}.parquet")
        print(f"{name:16s} {df.height:>10,d} rows  pos={int((df['label'] == 1).sum()):>7,d}  "
              f"neg={int((df['label'] == 0).sum()):>9,d}  {time.time() - t:.0f}s", flush=True)


def _context_table(m: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for species, source, ctx in m.select("species", "source", "context").unique().iter_rows():
        if species == "human":
            name, dep = (ctx.split("|") + [None])[:2]
            r = contexts.lookup(name, dep if dep and dep != "NA" else None)
            if r is None:
                raise ValueError(f"unmapped human context {source}:{ctx}")
            rows.append({
                "species": species, "source": source, "context": ctx,
                "context_id": f"human:{r['cellosaurus_name']}", "cellosaurus_ac": r["cellosaurus_ac"],
                "depmap_id": r["depmap_id"], "disease": r["disease"], "sex": r["sex"],
                "ancestry_group": r["ancestry_group"], "ancestry_basis": r["ancestry_basis"],
                "anc_AFR": r["anc_AFR"], "anc_AMR": r["anc_AMR"], "anc_EAS": r["anc_EAS"],
                "anc_EUR": r["anc_EUR"], "anc_SAS": r["anc_SAS"],
            })
        else:
            rows.append({"species": species, "source": source, "context": ctx,
                         "context_id": f"{species}:{ctx}", "ancestry_group": "n/a", "ancestry_basis": "n/a"})
    return pl.DataFrame(rows, infer_schema_length=None)


def stage_examples() -> None:
    """Collapse measurements of the same (species, context, pair) across sources.

    label = 1 if every labelled measurement says 1, 0 if every one says 0; pairs with
    conflicting labels are dropped and counted in the build report.
    """
    m = pl.concat([pl.read_parquet(p) for p in sorted((INTERIM / "measurements").glob("*.parquet"))])
    ctx = _context_table(m)
    m = m.join(ctx.select("species", "source", "context", "context_id"), on=["species", "source", "context"])
    lab = m.filter(pl.col("label").is_not_null())
    ex = lab.group_by("species", "context_id", "gene_a", "gene_b").agg(
        pl.col("label").min().alias("lmin"), pl.col("label").max().alias("lmax"),
        pl.col("source").unique().sort().str.join(",").alias("sources"),
        pl.len().alias("n_measurements"),
    )
    conflicts = ex.filter(pl.col("lmin") != pl.col("lmax"))
    ex = ex.filter(pl.col("lmin") == pl.col("lmax")).rename({"lmin": "label"}).drop("lmax")
    # cross-study replication among multiply-measured pairs (a label-noise floor for the report)
    multi = lab.group_by("species", "context_id", "gene_a", "gene_b").agg(
        pl.col("source").n_unique().alias("ns"), pl.col("label").min().alias("lmin"), pl.col("label").max().alias("lmax"),
        pl.col("label").max().alias("any_pos"),
    ).filter(pl.col("ns") > 1)
    ctx_meta = ctx.drop("source", "context").unique("context_id").sort("context_id")
    INTERIM.mkdir(exist_ok=True, parents=True)
    ex.write_parquet(INTERIM / "examples.parquet")
    ctx_meta.write_parquet(INTERIM / "contexts.parquet")
    report = {
        "measurements": m.height,
        "labelled_measurements": lab.height,
        "examples": ex.height,
        "conflicting_pairs_dropped": conflicts.height,
        "multi_source_pairs": multi.height,
        "multi_source_positive_pairs": int((multi["any_pos"] == 1).sum()),
        "multi_source_positive_agreement": float(
            ((multi["any_pos"] == 1) & (multi["lmin"] == 1)).sum() / max(1, (multi["any_pos"] == 1).sum())),
    }
    (INTERIM / "examples_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def bucket(family: str) -> str:
    h = int.from_bytes(hashlib.sha256(f"{SALT}:{family}".encode()).digest()[:8], "big") / 2**64
    return "test" if h < TEST_FRAC else "dev" if h < TEST_FRAC + DEV_FRAC else "train"


def stage_splits() -> None:
    ex = pl.read_parquet(INTERIM / "examples.parquet")
    ctx = pl.read_parquet(INTERIM / "contexts.parquet")
    genes = pl.concat([
        ex.select("species", pl.col("gene_a").alias("gene")), ex.select("species", pl.col("gene_b").alias("gene"))
    ]).unique().sort("species", "gene")
    fam = families.assign(genes)
    fam = fam.with_columns(pl.col("family").map_elements(bucket, return_dtype=pl.String).alias("bucket"))
    fa = fam.rename({"gene": "gene_a", "family": "family_a", "bucket": "bucket_a"})
    fb = fam.rename({"gene": "gene_b", "family": "family_b", "bucket": "bucket_b"})
    ex = ex.join(fa, on=["species", "gene_a"]).join(fb, on=["species", "gene_b"])
    ex = ex.with_columns(
        pl.when(pl.col("bucket_a") == pl.col("bucket_b")).then(pl.col("bucket_a"))
        .when(pl.concat_list("bucket_a", "bucket_b").list.sort() == ["test", "train"]).then(pl.lit("test_semi"))
        .when(pl.concat_list("bucket_a", "bucket_b").list.sort() == ["dev", "train"]).then(pl.lit("dev_semi"))
        .otherwise(pl.lit("drop")).alias("split"),
        (pl.col("family_a") == pl.col("family_b")).alias("same_family"),
    ).join(ctx.select("context_id", "ancestry_group"), on="context_id", how="left")
    ex = ex.with_columns(
        (pl.col("species") + "|" + pl.col("context_id") + "|" + pl.col("gene_a") + "|" + pl.col("gene_b")).alias("example_id")
    )
    cols = ["example_id", "species", "context_id", "ancestry_group", "gene_a", "gene_b", "same_family", "sources", "label"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hidden").mkdir(exist_ok=True)
    manifest = {"version": VERSION, "salt": SALT, "test_frac": TEST_FRAC, "dev_frac": DEV_FRAC,
                "paralog_min_identity": families.PARALOG_MIN_IDENTITY, "built": time.strftime("%Y-%m-%d"), "files": {}}
    for split in ["train", "dev", "dev_semi", "test", "test_semi"]:
        part = ex.filter(pl.col("split") == split).select(cols).sort("example_id")
        if split.startswith("test"):
            _write(part.drop("label", "sources"), OUT / f"{split}_inputs.parquet", manifest)
            _write(part.select("example_id", "label", "sources"), OUT / "hidden" / f"{split}_labels.parquet", manifest)
        else:
            _write(part, OUT / f"{split}.parquet", manifest)
    _write(ctx.sort("context_id"), OUT / "contexts.parquet", manifest)
    _write(_single_effect_bins(genes), OUT / "gene_single_effects.parquet", manifest)
    _write(fam.sort("species", "gene"), OUT / "held_out_families.parquet", manifest)
    counts = ex.group_by("split", "species").agg(
        pl.len().alias("n"), (pl.col("label") == 1).sum().alias("pos"), pl.col("context_id").n_unique().alias("contexts")
    ).sort("split", "species")
    manifest["counts"] = counts.to_dicts()
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with pl.Config(tbl_rows=100):
        print(counts)


FITNESS_BINS = 5


def _single_effect_bins(genes: pl.DataFrame) -> pl.DataFrame:
    """Per benchmark gene: reference single-loss effect and its within-species quintile (1 = sickest;
    0 = unknown). Used by the fitness-matched metric; also a legitimate single-gene model input."""
    from slpbench.baselines import single_effects

    se = genes.join(single_effects(), on=["species", "gene"], how="left")
    return se.with_columns(
        pl.when(pl.col("single_effect").is_null()).then(0)
        .otherwise((pl.col("single_effect").rank("ordinal").over("species") - 1) * FITNESS_BINS
                   // pl.col("single_effect").count().over("species") + 1)
        .cast(pl.Int8).alias("fitness_bin")
    ).sort("species", "gene")


def _write(df: pl.DataFrame, path: Path, manifest: dict) -> None:
    df.write_parquet(path)
    manifest["files"][str(path.relative_to(OUT))] = {
        "rows": df.height, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prepare", "measurements", "examples", "splits", "all"], default="all")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.stage in ("prepare", "all"):
        stage_prepare()
    if a.stage in ("measurements", "all"):
        stage_measurements(a.only)
    if a.stage in ("examples", "all"):
        stage_examples()
    if a.stage in ("splits", "all"):
        stage_splits()


if __name__ == "__main__":
    main()
