"""Build data/interim/orthology_extra/edges.parquet (u, v, kind, weight, source).

Steps:
  1. proteomes -> one protein per canonical gene (longest isoform), FASTA headers "species:gene"
  2. DIAMOND blastp (--more-sensitive, e<=1e-5) of every proteome against every proteome; hits symmetrized
  3. ortholog edges  = reciprocal best hits (bitscore; near-ties within RBH_TIE of the best count as best),
                       alignment covering >= MIN_COV of the shorter protein            [source diamond_rbh]
                     + Alliance combined for pairs involving mouse / worm (same filter as families.py)
                                                                                         [source alliance]
  4. paralog edges   = Ensembl BioMart paralogs for mouse / worm (as homology.py)       [source ensembl_biomart]
                     + DIAMOND within-species hits for species without Ensembl paralogs
                       (bacteria, spne, calb): identity = nident / min(qlen, slen) (Ensembl's max(%id, %id_r1)
                       analogue), kept when >= 0.30                                      [source diamond_self]
Usage: uv run python scripts/orthology_extra/build_edges.py [--threads N] [--redo-diamond]
"""

from __future__ import annotations

import argparse
import gzip
import re
import subprocess
from pathlib import Path

import polars as pl

from slbench import families, ids, ids_extra

RAW = Path("data/raw")
PROT = RAW / "orthology_extra/proteomes"
WORK = Path("data/interim/orthology_extra")
DIAMOND = Path("external/bin/diamond")

EVALUE = 1e-5
MIN_COV = 0.5          # aligned length / shorter protein, for ortholog RBH
RBH_TIE = 0.95         # hits with bitscore >= RBH_TIE * best count as best (co-orthologs / near-identical in-paralogs)
PARALOG_MIN_IDENTITY = families.PARALOG_MIN_IDENTITY
MAX_TARGETS = 250      # per query per target proteome

EUK = ["human", "mmus", "cele", "dmel", "scer", "spom", "calb"]
BACT = ["spne", "ecol", "bsub", "mtub", "saur"]
SPECIES = EUK + BACT
ENSEMBL_PARALOGS = {"mmus": "ensembl_mmusculus_paralogs.tsv", "cele": "ensembl_celegans_paralogs.tsv"}
DIAMOND_PARALOGS = ["calb"] + BACT   # species with no Ensembl paralog table in SLB
ALLIANCE_TAXA = {"NCBITaxon:9606": "human", "NCBITaxon:559292": "scer", "NCBITaxon:7227": "dmel",
                 "NCBITaxon:10090": "mmus", "NCBITaxon:6239": "cele"}


# ---------------------------------------------------------------- proteomes
def _fasta(path: Path):
    op = gzip.open if str(path).endswith(".gz") else open
    head, seq = None, []
    with op(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if head is not None:
                    yield head, "".join(seq)
                head, seq = line[1:].rstrip("\n"), []
            else:
                seq.append(line.strip())
    if head is not None:
        yield head, "".join(seq)


def _ensembl_fields(h: str) -> dict[str, str]:
    out = {"id": h.split()[0], "loc": h.split()[2] if len(h.split()) > 2 else ""}
    for tok in h.split():
        k, sep, v = tok.partition(":")
        if sep and k in ("gene", "gene_symbol", "gene_biotype", "transcript_biotype"):
            out[k] = v
    return out


def proteome(sp: str) -> dict[str, str]:
    """canonical gene -> longest protein sequence."""
    best: dict[str, str] = {}

    def add(g, s):
        s = s.rstrip("*").replace("*", "X")
        if g and s and len(s) > len(best.get(g, "")):
            best[g] = s

    if sp in ("human", "mmus", "cele", "dmel", "scer"):
        res = {"human": ids.human, "mmus": ids_extra.mmus, "scer": ids.scer}.get(sp)
        res = res() if res else None
        for h, s in _fasta(PROT / f"{sp}.pep.fa.gz"):
            f = _ensembl_fields(h)
            if f.get("gene_biotype") != "protein_coding" or not (f["loc"].startswith("chromosome") or
                                                                  f["loc"].startswith("primary_assembly")):
                continue
            gid = f["gene"].split(".")[0]
            if sp in ("human", "mmus"):
                g = res(gid) or res(f.get("gene_symbol"))
            elif sp == "scer":
                g = res(gid)
            else:
                g = gid  # WBGene / FBgn
            add(g, s)
    elif sp == "spom":
        res = ids.spom()
        for h, s in _fasta(PROT / "spom.pep.fa.gz"):
            sysid = re.sub(r"\.\d+$", "", h.split()[0].split(":")[0])  # SPAC1002.01.1 -> SPAC1002.01
            add(res(sysid), s)
    elif sp == "calb":
        for h, s in _fasta(PROT / "calb.pep.fa.gz"):
            fid = h.split()[0]
            if fid.endswith("_B"):
                continue
            add(ids_extra.calb_canonical(fid), s)
    else:
        t = ids_extra.bacterial_gene_table(sp)
        for g, s in zip(t["canonical"], t["translation"]):
            add(g, s)
    return best


def write_fastas() -> pl.DataFrame:
    (WORK / "fasta").mkdir(parents=True, exist_ok=True)
    ids_extra.write_bacterial_tables()
    stats = []
    for sp in SPECIES:
        p = proteome(sp)
        with open(WORK / f"fasta/{sp}.fa", "w") as f:
            for g, s in p.items():
                f.write(f">{sp}:{g}\n{s}\n")
        stats.append((sp, len(p)))
        print(sp, len(p), flush=True)
    return pl.DataFrame(stats, schema=["species", "proteins"], orient="row")


# ---------------------------------------------------------------- DIAMOND
COLS = ["q", "s", "pident", "length", "nident", "qlen", "slen", "evalue", "bitscore"]


def run_diamond(threads: int, redo: bool) -> None:
    for sp in SPECIES:
        db = WORK / f"db/{sp}.dmnd"
        if redo or not db.exists():
            db.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([str(DIAMOND), "makedb", "--in", str(WORK / f"fasta/{sp}.fa"), "-d", str(db), "--quiet"],
                           check=True)
    allq = WORK / "fasta/all.fa"
    with open(allq, "w") as out:
        for sp in SPECIES:
            out.write((WORK / f"fasta/{sp}.fa").read_text())
    for sp in SPECIES:
        hits = WORK / f"hits/vs_{sp}.parquet"
        if hits.exists() and not redo:
            continue
        hits.parent.mkdir(parents=True, exist_ok=True)
        tsv = WORK / f"hits/vs_{sp}.tsv"
        print("diamond vs", sp, flush=True)
        subprocess.run([str(DIAMOND), "blastp", "-q", str(allq), "-d", str(WORK / f"db/{sp}.dmnd"), "-o", str(tsv),
                        "--more-sensitive", "--evalue", str(EVALUE), "--max-target-seqs", str(MAX_TARGETS),
                        "--threads", str(threads), "--quiet", "--outfmt", "6", "qseqid", "sseqid", "pident", "length",
                        "nident", "qlen", "slen", "evalue", "bitscore"], check=True)
        pl.read_csv(tsv, separator="\t", has_header=False, new_columns=COLS, quote_char=None,
                    schema_overrides={"evalue": pl.Float64, "pident": pl.Float64, "bitscore": pl.Float64}) \
            .write_parquet(hits)
        tsv.unlink()


def load_hits() -> pl.DataFrame:
    h = pl.concat([pl.read_parquet(WORK / f"hits/vs_{sp}.parquet") for sp in SPECIES])
    h = h.with_columns(pl.col("q").str.split(":").list.first().alias("qsp"),
                       pl.col("s").str.split(":").list.first().alias("ssp"))
    h = h.filter(pl.col("q") != pl.col("s"))
    # DIAMOND's sensitivity is not symmetric (a hit found for a->b may be missed for b->a): symmetrize,
    # then keep the best-scoring HSP per ordered pair
    rev = h.select(pl.col("s").alias("q"), pl.col("q").alias("s"), "pident", "length", "nident",
                   pl.col("slen").alias("qlen"), pl.col("qlen").alias("slen"), "evalue", "bitscore",
                   pl.col("ssp").alias("qsp"), pl.col("qsp").alias("ssp"))
    return pl.concat([h, rev]).sort("bitscore", descending=True).unique(["q", "s"], keep="first")


def rbh(h: pl.DataFrame) -> pl.DataFrame:
    x = h.filter(pl.col("qsp") != pl.col("ssp"))
    x = x.with_columns(pl.col("bitscore").max().over("q", "ssp").alias("best"))
    top = x.filter(pl.col("bitscore") >= RBH_TIE * pl.col("best"))
    fwd = top.select("q", "s", "length", "qlen", "slen")
    rev = top.select(pl.col("s").alias("q"), pl.col("q").alias("s"))
    r = fwd.join(rev, on=["q", "s"], how="inner")
    r = r.filter(pl.col("length") / pl.min_horizontal("qlen", "slen") >= MIN_COV)
    r = r.filter(pl.col("q") < pl.col("s")).select(pl.col("q").alias("u"), pl.col("s").alias("v"))
    return r.with_columns(pl.lit("ortholog").alias("kind"), pl.lit(1.0).alias("weight"), pl.lit("diamond_rbh").alias("source"))


def self_paralogs(h: pl.DataFrame, species: list[str]) -> pl.DataFrame:
    x = h.filter((pl.col("qsp") == pl.col("ssp")) & pl.col("qsp").is_in(species))
    x = x.with_columns((pl.col("nident") / pl.min_horizontal("qlen", "slen")).alias("identity"),
                       pl.min_horizontal("q", "s").alias("u"), pl.max_horizontal("q", "s").alias("v"))
    x = x.group_by("u", "v").agg(pl.col("identity").max()).filter(pl.col("identity") >= PARALOG_MIN_IDENTITY)
    return x.select("u", "v", pl.lit("paralog").alias("kind"), pl.col("identity").alias("weight"),
                    pl.lit("diamond_self").alias("source"))


# ---------------------------------------------------------------- curated resources
def ensembl_paralogs() -> pl.DataFrame:
    parts = []
    for sp, fn in ENSEMBL_PARALOGS.items():
        df = pl.read_csv(RAW / "orthology_extra" / fn, separator="\t", infer_schema_length=0, quote_char=None)
        df = df.filter(~df[df.columns[0]].str.starts_with("[success]"))
        df = df.select(pl.col(df.columns[0]).alias("a"), pl.col(df.columns[2]).alias("b"),
                       pl.max_horizontal(pl.col(df.columns[6]).cast(pl.Float64),
                                         pl.col(df.columns[7]).cast(pl.Float64)).alias("identity")).drop_nulls()
        if sp == "mmus":
            df = df.with_columns(ids_extra.resolve("mmus", df["a"]).alias("a"), ids_extra.resolve("mmus", df["b"]).alias("b"))
        df = df.drop_nulls().filter(pl.col("a") != pl.col("b")).with_columns(
            (pl.lit(sp + ":") + pl.min_horizontal("a", "b")).alias("u"),
            (pl.lit(sp + ":") + pl.max_horizontal("a", "b")).alias("v"), pl.col("identity") / 100)
        df = df.group_by("u", "v").agg(pl.col("identity").max()).filter(pl.col("identity") >= PARALOG_MIN_IDENTITY)
        parts.append(df.select("u", "v", pl.lit("paralog").alias("kind"), pl.col("identity").alias("weight"),
                               pl.lit("ensembl_biomart").alias("source")))
    return pl.concat(parts)


def alliance() -> pl.DataFrame:
    """Alliance combined orthology for pairs involving mouse or worm (human/scer/dmel pairs are in families.py)."""
    hgnc = pl.read_csv(RAW / "ids/hgnc_complete_set.txt", separator="\t", infer_schema_length=0, quote_char=None,
                       columns=["hgnc_id", "symbol"])
    hgnc = dict(zip(hgnc["hgnc_id"], hgnc["symbol"]))
    sgd, mgi = ids.scer(), ids_extra.mmus()

    def conv(gid, sp):
        if sp == "human":
            return hgnc.get(gid)
        if sp == "scer":
            return sgd(gid.removeprefix("SGD:"))
        if sp == "dmel":
            return gid.removeprefix("FB:")
        if sp == "mmus":
            return mgi(gid)
        if sp == "cele":
            return gid.removeprefix("WB:")
        return None

    out = []
    with gzip.open(RAW / "orthology/ORTHOLOGY-ALLIANCE_COMBINED.tsv.gz", "rt") as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if header is None:
                header = p
                continue
            s1, s2 = ALLIANCE_TAXA.get(p[2]), ALLIANCE_TAXA.get(p[6])
            if not s1 or not s2 or s1 == s2 or not ({s1, s2} & {"mmus", "cele"}):
                continue
            if not ((p[11] == "Yes" and p[12] == "Yes") or int(p[9]) >= families.ORTHOLOG_MIN_ALGORITHMS):
                continue
            g1, g2 = conv(p[0], s1), conv(p[4], s2)
            if g1 and g2:
                a, b = sorted([f"{s1}:{g1}", f"{s2}:{g2}"])
                out.append((a, b))
    return pl.DataFrame(out, schema=["u", "v"], orient="row").unique().with_columns(
        pl.lit("ortholog").alias("kind"), pl.lit(1.0).alias("weight"), pl.lit("alliance").alias("source"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--redo-diamond", action="store_true")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    stats = write_fastas()
    stats.write_csv(WORK / "proteome_counts.tsv", separator="\t")
    run_diamond(a.threads, a.redo_diamond)
    h = load_hits()
    e = pl.concat([rbh(h), self_paralogs(h, DIAMOND_PARALOGS), ensembl_paralogs(), alliance()])
    # canonical orientation, one row per (u, v, kind, source)
    e = e.with_columns(pl.min_horizontal("u", "v").alias("u"), pl.max_horizontal("u", "v").alias("v")) \
        .group_by("u", "v", "kind", "source").agg(pl.col("weight").max()).sort("u", "v", "kind", "source") \
        .select("u", "v", "kind", "weight", "source")
    e.write_parquet(WORK / "edges.parquet")
    # all-species DIAMOND self identities (for validation against Ensembl; not part of edges)
    self_paralogs(h, SPECIES).rename({"weight": "identity"}).drop("kind", "source") \
        .write_parquet(WORK / "diamond_self_paralogs_all.parquet")
    sp = e.with_columns(pl.col("u").str.split(":").list.first().alias("su"), pl.col("v").str.split(":").list.first().alias("sv"))
    print(sp.group_by("kind", "source", "su", "sv").len().sort("kind", "source", "su", "sv"))


if __name__ == "__main__":
    main()
