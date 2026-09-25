"""Additional bacterial pairwise genetic-interaction sources and per-species single-gene fitness.

Every source function returns the MEASUREMENT schema via `finalize`; every
`<species>_single()` returns [gene, effect] with negative = sicker.

Reproducibility checks for `slbench audit` (each returns list[dict] with keys check/auroc/pos,
plus n/ci95/source where meaningful): `dualtnseq2025_checks`, `koo2025_checks`,
`crisprtnseq2024_checks`, `spne_cross_checks` (all three pneumococcal screens against each other,
both directions, with bootstrap CIs), `babu2011_checks`, `gagarinova2016_checks`,
`kumar2016_checks`, `cote2016_checks` and the wrapper `ecoli_array_checks`.

Species codes and canonical gene IDs:
  spne  S. pneumoniae D39V. Canonical IDs come from `slbench.ids_extra.spne()` (lead decision
        2026-09-24): the ID used for the original spne labels, i.e. the dual CRISPRi-seq author name where
        there is one (blpU, srf-06, ...), else the SPV_ locus tag. `spne_map` widens the alias
        space that ids_extra covers with the cross-strain tags SPV_RS* (RefSeq NZ_CP027540.1),
        SPD_* (D39, CP000410), SP_* (TIGR4) and spr* (R6). See `spne_resolver()`.
  ecol  E. coli K-12 (MG1655/BW25113), b-number locus tag, e.g. b0002 (= `ids_extra.ecol`;
        `ecol_map` adds the JW/Keio ids).
  bsub  B. subtilis 168, old-style BSU locus tag, e.g. BSU00010 (= `ids_extra.bsub`).
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import numpy as np
import polars as pl

from . import finalize

RAW = Path("data/raw")


# --------------------------------------------------------------------------------------------
# GenBank helpers
# --------------------------------------------------------------------------------------------

def read_genbank_features(path: Path, types: tuple[str, ...] = ("gene", "CDS")) -> list[dict]:
    """Minimal GenBank feature-table reader (handles multi-line qualifiers). Values are strings;
    repeated qualifiers (e.g. several /note) are joined with ' | '."""
    feats: list[dict] = []
    cur: dict | None = None
    key: str | None = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("ORIGIN"):
                break
            if len(line) > 5 and line[5] != " " and line.startswith("     "):
                typ, loc = line[5:21].strip(), line[21:].strip()
                cur = {"type": typ, "loc": loc} if typ in types else None
                if cur is not None:
                    feats.append(cur)
                key = None
                continue
            if cur is None or not line.startswith(" " * 21):
                continue
            s = line[21:].rstrip("\n")
            m = re.match(r'/(\w+)(?:=(.*))?$', s)
            if m:
                key = m.group(1)
                val = (m.group(2) or "").strip('"')
                cur[key] = f"{cur[key]} | {val}" if key in cur else val
            elif key is not None:
                cur[key] = (cur[key] + " " + s.strip().strip('"')).strip()
    return feats


# --------------------------------------------------------------------------------------------
# S. pneumoniae D39V identifiers
# --------------------------------------------------------------------------------------------

# Names used by dual CRISPRi-seq 2025 that postdate the CP027540.1 annotation (PneumoBrowse 2).
_SPNE_NAME_OVERRIDES = {"SPV_0768": "cozE", "SPV_1569": "aqpZ", "SPV_0887": "yfnA", "SPV_0878": "orfX"}


@functools.cache
def spne_table() -> pl.DataFrame:
    """One row per D39V SPV_ locus: [spv, name, canonical, spd, tigr4, r6, refseq_rs].

    canonical = the ID dualcrispri2025 uses for that gene (name or SPV_ tag) where the gene occurs
    there; otherwise the CP027540.1 gene name; otherwise the SPV_ tag.
    """
    ann = RAW / "spne_annot"
    rows: dict[str, dict] = {}
    for f in read_genbank_features(ann / "CP027540.1.gb", ("gene",)):
        t = f.get("locus_tag")
        if not t or not t.startswith("SPV_"):
            continue
        note = f.get("note", "")
        r = rows.setdefault(t, {"spv": t, "name": None, "syn": set(), "spd": set(), "tigr4": set(), "r6": set()})
        if f.get("gene"):
            r["name"] = f["gene"]
        for s in re.split(r"[;,| ]+", f.get("gene_synonym", "")):
            if s:
                r["syn"].add(s)
        for m in re.finditer(r"Corresponds to (\S+) found in (?:INSD |RefSeq )?(\S+?);?(?:\s|$)", note):
            tag = m.group(1).rstrip(";")
            if tag.startswith("SPD_"):
                r["spd"].add(tag)
            elif tag.startswith("SP_"):
                r["tigr4"].add(tag)
            elif tag.startswith("spr"):
                r["r6"].add(tag)
    rs: dict[str, str] = {}
    for f in read_genbank_features(ann / "NZ_CP027540.1.gb", ("gene",)):
        if f.get("old_locus_tag") and f.get("locus_tag"):
            for o in f["old_locus_tag"].split(" | "):
                rs[f["locus_tag"]] = o
    rs_by_spv: dict[str, set] = {}
    for k, v in rs.items():
        rs_by_spv.setdefault(v, set()).add(k)

    # dualcrispri2025 ID usage
    d = pl.read_csv(RAW / "dualcrispri2025_spneumo/mmc4.csv", infer_schema_length=0, null_values=["NA", ""],
                    columns=["SG1.targets", "SG2.targets"])
    dc = set()
    for c in d.columns:
        for v in d[c].drop_nulls().unique():
            dc.update(v.split(","))
    name2spv: dict[str, str] = {}
    for t, r in rows.items():
        if r["name"]:
            name2spv.setdefault(r["name"], t)
    for t, n in _SPNE_NAME_OVERRIDES.items():
        name2spv[n] = t

    out = []
    for t, r in rows.items():
        name = _SPNE_NAME_OVERRIDES.get(t, r["name"])
        if t in dc:
            canon = t
        elif name and name in dc:
            canon = name
        else:
            canon = name or t
        out.append({"spv": t, "name": name, "canonical": canon, "synonyms": ",".join(sorted(r["syn"])),
                    "spd": ",".join(sorted(r["spd"])), "tigr4": ",".join(sorted(r["tigr4"])),
                    "r6": ",".join(sorted(r["r6"])), "refseq_rs": ",".join(sorted(rs_by_spv.get(t, [])))})
    return pl.DataFrame(out)


@functools.cache
def spne_alias_to_spv() -> dict[str, str]:
    """alias -> SPV_ locus tag, for the cross-strain IDs that `ids_extra.spne` does not carry:
    SPD_ (D39 CP000410), SP_ (TIGR4), spr (R6) and SPV_RS (RefSeq NZ_CP027540.1) tags.
    Ambiguous aliases (mapping to more than one locus) are dropped."""
    t = spne_table()
    cand: dict[str, set] = {}

    def add(k, v):
        if k:
            cand.setdefault(str(k), set()).add(v)

    for r in t.iter_rows(named=True):
        add(r["spv"], r["spv"])
        for col in ("spd", "tigr4", "r6", "refseq_rs"):
            for a in (r[col] or "").split(","):
                add(a, r["spv"])
    out = {k: next(iter(v)) for k, v in cand.items() if len(v) == 1}
    out |= {r["spv"]: r["spv"] for r in t.iter_rows(named=True)}
    return out


@functools.cache
def spne_resolver() -> dict[str, str]:
    """alias -> canonical spne ID.

    The canonical ID is whatever `slbench.ids_extra.spne()` returns, which is the ID already used
    in SLB (the dual CRISPRi-seq author name where there is one, else the SPV_ tag) - lead decision
    2026-09-24, so that a gene cannot get two IDs across the three pneumococcal sources. This
    resolver only widens the alias space: anything ids_extra does not know is first mapped to its
    SPV_ tag with `spne_alias_to_spv` and then handed to ids_extra.
    """
    from slbench import ids_extra

    r = ids_extra.spne()
    out: dict[str, str] = {}
    for alias, spv in spne_alias_to_spv().items():
        c = r(alias) or r(spv)
        if c:
            out[alias] = c
    for row in spne_table().iter_rows(named=True):
        for k in (row["spv"], row["name"], row["canonical"]):
            if k:
                c = r(k) or out.get(row["spv"])
                if c:
                    out[k] = c
        for a in (row["synonyms"] or "").split(","):
            if a and a not in out:
                c = r(a) or out.get(row["spv"])
                if c:
                    out[a] = c
    return out


def spne_map(s: pl.Series) -> pl.Series:
    """Map any S. pneumoniae identifier to the canonical spne ID used by SLB (via ids_extra.spne)."""
    from slbench import ids_extra

    m = spne_resolver()
    r = ids_extra.spne()
    uniq = {x: (r(x) or m.get(str(x))) for x in s.unique().to_list()}
    return s.replace_strict(uniq, default=None, return_dtype=pl.String)


# --------------------------------------------------------------------------------------------
# S. pneumoniae D39V: CRISPRi-TnSeq (Jana et al., Nat Microbiol 9:2395-2409, 2024)
# --------------------------------------------------------------------------------------------

_CRTN_QUERIES = ("adk", "atpF", "clpP", "cozE", "fabH", "folA", "ftsH", "ftsZ", "gyrA",
                 "parC", "pbp2x", "rpoB", "rpoC")


def _crisprtnseq2024_long() -> pl.DataFrame:
    """Per-(library tab, transposon-mutant gene) rows of Supplementary Data 5 (MOESM6)."""
    import openpyxl

    wb = openpyxl.load_workbook(RAW / "crisprtnseq2024_spneumo/MOESM6.xlsx", read_only=True)
    recs = []
    for ws in wb.worksheets:
        if ws.title.lower() in ("description", "tab explanation", "genetic interaction") or \
                ws.title.strip().lower() in ("description", "tab explanation", "genetic interaction"):
            continue
        it = ws.iter_rows(values_only=True)
        head = list(next(it))
        if "locus_tag_SPV" not in head:
            continue  # growth-curve / correlation tabs
        col = {"value": head.index("value"), "p": head.index("p-value"), "spv": head.index("locus_tag_SPV"),
               "padj": head.index("padj"), "sig": head.index("Sig"), "z": head.index("z-score")}
        for r in it:
            if r[0] is None:
                continue
            r = list(r) + [None] * 60
            recs.append({"tab": ws.title, "rs": str(r[0]), "fit_iptg": r[1], "fit_noiptg": r[11],
                         **{k: r[v] for k, v in col.items()}})
    df = pl.DataFrame([{k: (str(v) if v is not None else None) for k, v in d.items()} for d in recs])
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False)
                          for c in ("fit_iptg", "fit_noiptg", "value", "p", "padj", "z")])
    df = df.with_columns(pl.col("tab").str.extract(r"^(pbp2x|[A-Za-z]+)").alias("query"))
    gene = spne_map(df["rs"])
    gene2 = spne_map(df["spv"].fill_null(""))
    return df.with_columns(pl.Series("gene", [a if a is not None else b for a, b in zip(gene, gene2)]))


def _crisprtnseq2024_calls() -> pl.DataFrame:
    """The authors' 1,334 called interactions ('genetic interaction' tab): [query, gene, gi_padj, gi_z]."""
    import openpyxl

    wb = openpyxl.load_workbook(RAW / "crisprtnseq2024_spneumo/MOESM6.xlsx", read_only=True)
    ws = wb["genetic interaction"]
    rows = [r[:4] for r in ws.iter_rows(values_only=True)][1:]
    d = pl.DataFrame({"rs": [str(r[0]) for r in rows], "lab": [str(r[1]) for r in rows],
                      "gi_padj": [str(r[2]) for r in rows], "gi_z": [str(r[3]) for r in rows]})
    d = d.with_columns(pl.col("gi_padj").cast(pl.Float64), pl.col("gi_z").cast(pl.Float64),
                       pl.col("lab").str.extract(r"^(\S+) ").replace({"cozEa": "cozE"}).alias("query"))
    return d.with_columns(pl.Series("gene", spne_map(d["rs"]))).drop_nulls("gene") \
            .select("query", "gene", "gi_padj", "gi_z")


def crisprtnseq2024() -> pl.DataFrame:
    """CRISPRi-TnSeq, Jana et al. Nat Microbiol 2024 (PMID 39030344): S. pneumoniae D39V, 13 essential-gene
    CRISPRi strains x genome-wide transposon-mutant libraries, ~24k essential x non-essential pairs.

    Score = the authors' fitness difference W(+IPTG) - W(-IPTG) for the transposon mutant, averaged
    over that query's IPTG levels and experimental repeats (1-4 libraries per query); negative =
    the knockout is sicker when the essential partner is knocked down = aggravating.
    Positive: on the authors' called-interaction list with z < 0 (their 754 negative interactions).
    Negative: not on the list, |fitness difference| below the dataset median (0.022) and p > 0.05.
    Everything else (called positive/alleviating interactions, and the intermediate band) -> null.

    Caveat: these labels are highly reproducible within
    the study (split-half AUROC 0.87-0.99) but are NOT recovered by dual CRISPRi-seq on the same
    strain over 1,599 shared labelled pairs (AUROC 0.50). Recommended as training measurements only.
    """
    long = _crisprtnseq2024_long().drop_nulls("gene")
    agg = long.group_by("query", "gene").agg(
        pl.col("value").mean().alias("score"), pl.col("p").min().alias("p"),
        (pl.col("sig") == "True").sum().alias("n_sig"), pl.len().alias("n_lib"))
    agg = agg.join(_crisprtnseq2024_calls(), on=["query", "gene"], how="left")
    med = agg["score"].abs().median()
    df = agg.select(
        pl.lit("spne").alias("species"), pl.lit("crisprtnseq2024").alias("source"),
        pl.lit("D39V").alias("context"), pl.lit("CRISPRi+Tn").alias("mechanism"),
        pl.col("query").alias("gene_a"), pl.col("gene").alias("gene_b"),
        pl.col("score"), pl.lit("fitness_difference").alias("score_name"),
        pl.col("p").alias("signif"), pl.lit("p_ttest").alias("signif_name"),
        pl.when(pl.col("gi_z") < 0).then(1)
        .when(pl.col("gi_z").is_null() & (pl.col("score").abs() < med) & (pl.col("p") > 0.05)).then(0)
        .otherwise(None).cast(pl.Int8).alias("label"),
    )
    return finalize(df)


def spne_single_tnseq() -> pl.DataFrame:
    """Single-gene fitness for S. pneumoniae D39V from CRISPRi-TnSeq reference arms (no IPTG).

    Median over libraries of the transposon-mutant fitness W without induction; W = 1 is wild-type
    growth, so `effect` = W - 1 (negative = sicker). Covers ~1.9k non-essential genes. The
    benchmark's primary spne effect stays `fitness.spne_single()` (dual CRISPRi-seq single-sgRNA
    log2FC); this one is the independent complement, and correlates with it at Spearman 0.50 over
    411 shared genes.
    """
    long = _crisprtnseq2024_long().drop_nulls("gene")
    return long.group_by("gene").agg((pl.col("fit_noiptg").median() - 1.0).alias("effect"))


# --------------------------------------------------------------------------------------------
# E. coli K-12 identifiers (canonical: b-number locus tag, e.g. b0002)
# --------------------------------------------------------------------------------------------

@functools.cache
def ecol_table() -> pl.DataFrame:
    """One row per E. coli K-12 MG1655 gene: [b, name, synonyms, jw].

    b-number and gene name/synonyms come from the U00096.3 GenBank flatfile; the JW (Keio) id from
    Baba et al. 2006 Supplementary Table 3, which lists ECK number, gene, JW id and b number.
    """
    import pandas as pd

    rows: dict[str, dict] = {}
    for f in read_genbank_features(RAW / "ecol_annot/U00096.3.gb", ("gene",)):
        b = f.get("locus_tag")
        if not b or not re.fullmatch(r"b\d+", b):
            continue
        r = rows.setdefault(b, {"b": b, "name": None, "syn": set()})
        if f.get("gene"):
            r["name"] = f["gene"]
        for s in re.split(r"[;,| ]+", f.get("gene_synonym", "")):
            if s:
                r["syn"].add(s)
    jw: dict[str, str] = {}
    x = pd.read_excel(RAW / "keio2006/S3_keio_mutants.xls", header=None)
    for _, row in x.iterrows():
        b_, j = row[8], row[2]
        if isinstance(b_, str) and re.fullmatch(r"b\d+", b_) and isinstance(j, str) and j.startswith("JW"):
            jw[b_] = j
    return pl.DataFrame([{"b": b, "name": r["name"], "synonyms": ",".join(sorted(r["syn"])),
                          "jw": jw.get(b)} for b, r in rows.items()])


@functools.cache
def ecol_resolver() -> dict[str, str]:
    """alias -> b-number. Gene names, ECK numbers, JW ids and JW ids with a Keio allele suffix
    (JW0001-1, JW0001_1) all resolve; ambiguous aliases are dropped."""
    t = ecol_table()
    cand: dict[str, set] = {}

    def add(k, v):
        if k:
            cand.setdefault(str(k), set()).add(v)

    for r in t.iter_rows(named=True):
        add(r["b"], r["b"])
        add(r["name"], r["b"])
        add(r["jw"], r["b"])
        for a in (r["synonyms"] or "").split(","):
            add(a, r["b"])
    out = {k: next(iter(v)) for k, v in cand.items() if len(v) == 1}
    out |= {r["b"]: r["b"] for r in t.iter_rows(named=True)}
    return out


def ecol_map(s: pl.Series) -> pl.Series:
    """Map E. coli identifiers to b-numbers (the canonical ecol ID, same as `ids_extra.ecol`).

    `ids_extra.ecol` is tried first; this function adds the JW (Keio) ids, which it does not carry,
    and is case-insensitive on gene names. JW allele suffixes (JW0013-1, JW0013_2) are stripped."""
    from slbench import ids_extra

    ie = ids_extra.ecol()
    m = ecol_resolver()
    low = {k.lower(): v for k, v in m.items()}

    def one(x: str | None) -> str | None:
        if x is None:
            return None
        k = str(x).strip()
        for cand in (k, k.lower(), re.sub(r"[-_]\d+$", "", k), re.sub(r"[-_]\d+$", "", k).upper()):
            hit = ie(cand)
            if hit:
                return hit
            if cand in m:
                return m[cand]
            if cand.lower() in low:
                return low[cand.lower()]
        return None

    uniq = {x: one(x) for x in s.unique().to_list()}
    return s.replace_strict(uniq, default=None, return_dtype=pl.String)


# --------------------------------------------------------------------------------------------
# S. pneumoniae D39: Dual Tn-seq (Zik et al., Science 2025)
# --------------------------------------------------------------------------------------------

_ZIK = RAW / "zik2025_spneumo"

# Gene pairs whose midpoints are closer than this are dropped: two lox-bearing transposons close
# together excise the intervening chromosome segment on Cre induction, so double mutants are lost
# for physical reasons. This is the authors' own `notNearby(minDist=6000)` threshold (dblStats.R).
_ZIK_MIN_DIST = 6000


@functools.cache
def _zik_gene_pos() -> dict[str, float]:
    g = pl.read_csv(_ZIK / "small/genes.tab", separator="\t", infer_schema_length=0)
    g = g.with_columns(((pl.col("begin").cast(pl.Int64) + pl.col("end").cast(pl.Int64)) / 2).alias("pos"))
    return dict(zip(g["locusId"], g["pos"]))


def _zik_pairs() -> pl.DataFrame:
    """genepair_stats.tsv.gz (central 10-90% analysis) with the chromosomal-distance filter applied."""
    d = pl.read_csv(_ZIK / "small/genepair_stats.tsv.gz", separator="\t")
    pos = _zik_gene_pos()
    d = d.with_columns((pl.col("locusId1").replace_strict(pos, default=None)
                        - pl.col("locusId2").replace_strict(pos, default=None)).abs().alias("dist"))
    return d.filter(pl.col("dist") > _ZIK_MIN_DIST)


def dualtnseq2025(min_expect_neg: float = 10.0) -> pl.DataFrame:
    """Dual Tn-seq, Zik et al. Science 2025 (doi:10.1126/science.adt7685): S. pneumoniae D39,
    ~1.4 billion double transposon mutants, 894,694 gene pairs scored by double-mutant depletion.

    Two randomly barcoded transposon libraries (different markers) are combined by transformation
    and the two barcodes are joined by Cre-lox, so read/strain counts per barcode pair measure how
    many viable double mutants exist. Score = zStrains = (observed - expected) / sqrt(expected)
    strains for the pair, where the expectation is the product of the two genes' marginal strain
    counts after the authors' chromosomal-position bias adjustment. Negative = fewer double mutants
    than expected = synthetic sick / lethal.

    Positive: the authors' medium-confidence call, zStrains <= -3 and readRatio <= 0.2.
    Negative: |zStrains| < 1 and 0.8 < readRatio < 1.25, with expectStrainsAdj >= `min_expect_neg`
    so the pair had the coverage to detect a depletion.
    Gene pairs less than 6 kb apart on the chromosome are dropped (Cre-lox excision artifact).

    IDs: locusId is a D39 SPD_ tag; `spne_map` takes it to the benchmark's D39V ID. 113 of 1,504
    SPD genes have no D39V counterpart in the CP027540.1 cross-references and are dropped.
    """
    d = _zik_pairs()
    df = d.select(
        pl.lit("spne").alias("species"), pl.lit("dualtnseq2025").alias("source"),
        pl.lit("D39V").alias("context"), pl.lit("dual-Tn").alias("mechanism"),
        pl.Series("gene_a", spne_map(d["locusId1"])), pl.Series("gene_b", spne_map(d["locusId2"])),
        pl.col("zStrains").alias("score"), pl.lit("zStrains").alias("score_name"),
        pl.lit(None, dtype=pl.Float64).alias("signif"), pl.lit(None, dtype=pl.String).alias("signif_name"),
        pl.when((pl.col("zStrains") <= -3) & (pl.col("readRatio") <= 0.2)).then(1)
        .when((pl.col("zStrains").abs() < 1) & (pl.col("readRatio") > 0.8) & (pl.col("readRatio") < 1.25)
              & (pl.col("expectStrainsAdj") >= min_expect_neg)).then(0)
        .otherwise(None).cast(pl.Int8).alias("label"),
    )
    return finalize(df)


def spne_single_dualtnseq() -> pl.DataFrame:
    """Single-gene fitness proxy for spne from the Dual Tn-seq essentiality table (`small/esstable`).

    `normreads` is a gene's transposon reads per nt divided by the genome median, so 1 = wild-type
    tolerance of insertion and values near 0 mean insertions are not tolerated. `effect` =
    log2(normreads + 0.01), clipped at -4 (negative = sicker), i.e. 0 for a neutral gene and
    about -4 for a gene with no usable insertions. This is insertion tolerance, not a growth rate;
    the benchmark's primary spne effect stays `fitness.spne_single()`.
    """
    d = pl.read_csv(_ZIK / "small/esstable", separator="\t", infer_schema_length=0)
    d = d.with_columns(pl.col("normreads").cast(pl.Float64))
    d = d.select(pl.Series("gene", spne_map(d["locusId"])),
                 (pl.col("normreads") + 0.01).log(2).clip(lower_bound=-4.0).alias("effect"))
    return d.drop_nulls("gene").group_by("gene").agg(pl.col("effect").median())


def spne_single() -> pl.DataFrame:
    """Merged single-gene effect for spne covering both S. pneumoniae screens' gene space.

    `fitness.spne_single()` (dual CRISPRi-seq single-sgRNA log2FC) covers only the 530 genes with a
    single-gene sgRNA, i.e. 318 of the 1,377 genes in `dualtnseq2025`. This function keeps that
    log2FC where it exists and fills the rest from the Dual Tn-seq insertion-tolerance effect
    (`spne_single_dualtnseq`), rescaled to the log2FC scale by matching medians and inter-quartile
    ranges on the 318 genes measured by both (their Spearman correlation there is 0.69).
    Negative = sicker. Coverage: ~1.8k genes.
    """
    from slbench import fitness as _f

    a = _f.spne_single().select("gene", pl.col("effect").cast(pl.Float64))
    b = spne_single_dualtnseq()
    j = a.join(b.rename({"effect": "tn"}), on="gene")
    def q(col, p):
        return float(col.quantile(p))

    sa = q(j["effect"], 0.75) - q(j["effect"], 0.25)
    sb = q(j["tn"], 0.75) - q(j["tn"], 0.25)
    scale = sa / sb if sb else 1.0
    b = b.with_columns(((pl.col("effect") - q(j["tn"], 0.5)) * scale + q(j["effect"], 0.5)).alias("effect"))
    return pl.concat([a, b.filter(~pl.col("gene").is_in(a["gene"]))]).select("gene", "effect")


# --------------------------------------------------------------------------------------------
# B. subtilis 168 identifiers (canonical: old-style BSU locus tag, e.g. BSU00010)
# --------------------------------------------------------------------------------------------

@functools.cache
def bsub_table() -> pl.DataFrame:
    """One row per B. subtilis 168 gene: [bsu, new_tag, name, synonyms] from AL009126.3.

    Canonical ID = the old-style tag (`/old_locus_tag`, e.g. BSU00010), which is what the
    double-CRISPRi libraries and SubtiWiki use; the current RefSeq/EMBL tag is BSU_00010.
    """
    rows: dict[str, dict] = {}
    for f in read_genbank_features(RAW / "bsub_annot/AL009126.3.gb", ("gene",)):
        new = f.get("locus_tag")
        old = (f.get("old_locus_tag") or "").split(" | ")[0] or (new or "").replace("_", "")
        if not old:
            continue
        r = rows.setdefault(old, {"bsu": old, "new_tag": new, "name": None, "syn": set()})
        if f.get("gene"):
            r["name"] = f["gene"]
        for s in re.split(r"[;,| ]+", f.get("gene_synonym", "")):
            if s:
                r["syn"].add(s)
    return pl.DataFrame([{"bsu": k, "new_tag": v["new_tag"], "name": v["name"],
                          "synonyms": ",".join(sorted(v["syn"]))} for k, v in rows.items()])


@functools.cache
def bsub_resolver() -> dict[str, str]:
    t = bsub_table()
    cand: dict[str, set] = {}

    def add(k, v):
        if k:
            cand.setdefault(str(k), set()).add(v)

    for r in t.iter_rows(named=True):
        add(r["bsu"], r["bsu"])
        add(r["new_tag"], r["bsu"])
        add(r["name"], r["bsu"])
        for a in (r["synonyms"] or "").split(","):
            add(a, r["bsu"])
    out = {k: next(iter(v)) for k, v in cand.items() if len(v) == 1}
    out |= {r["bsu"]: r["bsu"] for r in t.iter_rows(named=True)}
    return out


def bsub_map(s: pl.Series) -> pl.Series:
    """Map B. subtilis identifiers (BSU00010, BSU_00010, gene names, synonyms) to BSU00010 style,
    the canonical bsub ID (identical to `ids_extra.bsub`, which is tried first)."""
    from slbench import ids_extra

    ie = ids_extra.bsub()
    m = bsub_resolver()
    low = {k.lower(): v for k, v in m.items()}

    def one(x):
        if x is None:
            return None
        k = str(x).strip()
        for c in (k, k.replace("_", ""), k.upper(), k.lower()):
            hit = ie(c)
            if hit:
                return hit
            if c in m:
                return m[c]
            if c.lower() in low:
                return low[c.lower()]
        return None

    return s.replace_strict({x: one(x) for x in s.unique().to_list()}, default=None, return_dtype=pl.String)


# --------------------------------------------------------------------------------------------
# B. subtilis 168: double-CRISPRi (Koo et al., Cell Systems 2025)
# --------------------------------------------------------------------------------------------

_KOO = RAW / "koo2025_bsub_dcrispri"
_KOO_GI = _KOO / "NIHMS2115161-supplement-MMC4.xlsx"     # Table S3: GI scores
_KOO_RF = _KOO / "NIHMS2115161-supplement-MMC3.xlsx"     # Table S2: relative fitness
# The three 5-generation exponential-growth intervals with CRISPRi induced. T0-T4/T0-T6 are
# uninduced or overnight controls and T4-T5/T6-T7 are stationary-phase recovery, so they measure
# different conditions and are not merged in.
_KOO_EXPONENTIAL = ("T0 to T1 GI scores", "T1 to T2 GI scores", "T2 to T3 GI scores")


def _koo_matrix(path: Path, sheet: str) -> tuple[list[str], list[str], np.ndarray]:
    import openpyxl

    ws = openpyxl.load_workbook(path, read_only=True)[sheet]
    it = ws.iter_rows(values_only=True)
    head = list(next(it))
    cols = [(str(x) if x is not None else "") for x in head[1:]]
    rows, vals = [], []

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    for r in it:
        if r[0] is None:
            continue
        rows.append(str(r[0]))
        vals.append([num(v) for v in r[1:len(head)]])
    return rows, cols, np.array(vals)


def _koo_long(sheet: str, path: Path = _KOO_GI) -> pl.DataFrame:
    """Long form of one GI-score matrix: [gene_a, gene_b, score, fwd] with sgRNA labels resolved."""
    rows, cols, a = _koo_matrix(path, sheet)
    ri = np.repeat(np.arange(len(rows)), len(cols))
    ci = np.tile(np.arange(len(cols)), len(rows))
    v = a.ravel()
    ok = ~np.isnan(v)
    d = pl.DataFrame({"r": pl.Series([rows[i] for i in ri[ok]], dtype=pl.String),
                      "c": pl.Series([cols[i] for i in ci[ok]], dtype=pl.String),
                      "score": v[ok]}).filter((pl.col("r") != "") & (pl.col("c") != ""))
    d = d.with_columns(pl.Series("g1", bsub_map(d["r"].str.split("_").list.first())),
                       pl.Series("g2", bsub_map(d["c"].str.split("_").list.first())))
    d = d.drop_nulls(["g1", "g2"]).filter(pl.col("g1") != pl.col("g2"))
    return d.with_columns(pl.min_horizontal("g1", "g2").alias("gene_a"),
                          pl.max_horizontal("g1", "g2").alias("gene_b"),
                          (pl.col("g1") == pl.min_horizontal("g1", "g2")).alias("fwd"))


def koo2025(threshold: float = -1.5) -> pl.DataFrame:
    """Double-CRISPRi, Koo et al. Cell Systems 2025 (PMID 41045937): B. subtilis 168 envelope,
    319 sgRNA1 x 1,310 sgRNA2 targets, ~237k gene pairs per time interval.

    A single strain carries two sgRNAs with a barcode; relative fitness (RF) of every sgRNA pair is
    tracked over a pooled growth experiment (3 flasks per time point). The published GI score is a
    robust z-score of the strain's observed RF against the distribution of RF expected from the two
    single knockdowns, so negative = worse than expected = aggravating.

    Score here = mean GI score over the three 5-generation exponential-growth intervals with
    CRISPRi induced (T0-T1, T1-T2, T2-T3), which replicates far better than any single interval.
    Positive: mean GI <= `threshold` (-1.5, chosen because it is the strongest cut-off whose
    positives are recovered by the independent sgRNA1/sgRNA2 orientation swap at AUROC >= 0.85).
    Negative: |mean GI| below the median |GI| of the merged
    matrix. Both orientations of a pair are averaged by `finalize`.
    """
    parts = [_koo_long(s).with_columns(pl.lit(s).alias("tp")) for s in _KOO_EXPONENTIAL]
    d = pl.concat(parts)
    g = d.group_by("gene_a", "gene_b").agg(pl.col("score").mean(), pl.len().alias("n_obs"))
    med = float(g["score"].abs().median())
    df = g.select(
        pl.lit("bsub").alias("species"), pl.lit("koo2025").alias("source"),
        pl.lit("168").alias("context"), pl.lit("CRISPRi").alias("mechanism"),
        "gene_a", "gene_b",
        pl.col("score"), pl.lit("GI_robust_z").alias("score_name"),
        pl.lit(None, dtype=pl.Float64).alias("signif"), pl.lit(None, dtype=pl.String).alias("signif_name"),
        pl.when(pl.col("score") <= threshold).then(1)
        .when(pl.col("score").abs() < med).then(0)
        .otherwise(None).cast(pl.Int8).alias("label"),
    )
    return finalize(df)


def bsub_single() -> pl.DataFrame:
    """Single-gene knockdown effect for B. subtilis 168 from the Koo 2025 relative-fitness matrix.

    For each gene, the median relative fitness (RF) of all double-CRISPRi strains in which it is one
    of the two targets, over 10 generations of induction (`T0 to T2 RF`), taken as log2 so that
    0 = no cost and negative = sicker. Averaging over ~300-1,300 partners makes this a marginal
    single-knockdown effect, the same quantity the benchmark uses for *S. pneumoniae* (single-sgRNA
    log2FC). 1,171 genes.

    Caveat: it ranks the essential-gene flag in the authors' sgRNA table *backwards* (AUROC 0.35).
    That flag marks genes that cannot be deleted, while these sgRNAs were chosen to knock down
    partially, so a flagged gene often has little growth cost here; the genes this effect calls
    sickest (ftsH, gcaD, rasP, mbl, the dlt operon, accC, spoVE, ponA, cpgA) are the expected
    knockdown-sensitive ones. It measures knockdown cost in this library, not deletion essentiality.
    """
    rows, cols, a = _koo_matrix(_KOO_RF, "T0 to T2 RF")
    acc: dict[str, list[float]] = {}
    for i, rn in enumerate(rows):
        v = a[i, :][~np.isnan(a[i, :])]
        if len(v) >= 20:
            acc.setdefault(rn.split("_")[0], []).append(float(np.median(v)))
    for j, cn in enumerate(cols):
        v = a[:, j][~np.isnan(a[:, j])]
        if len(v) >= 20 and cn:
            acc.setdefault(cn.split("_")[0], []).append(float(np.median(v)))
    genes = pl.Series(list(acc))
    out = pl.DataFrame({"gene": bsub_map(genes),
                        "effect": [float(np.log2(max(np.median(v), 1e-3))) for v in acc.values()]})
    return out.drop_nulls("gene").group_by("gene").agg(pl.col("effect").median())


# --------------------------------------------------------------------------------------------
# E. coli K-12 colony-array double-mutant screens (eSGA / GIANT-coli family)
#
# All four screens below measure every tested pair (interacting or not), so each supplies
# positives and negatives. They do NOT agree with each other: every cross-study label-recovery
# AUROC among them is 0.46-0.51 (`slbench audit`). They are parsed so the lead can keep
# them as training measurements; none of them should supply benchmark labels.
# --------------------------------------------------------------------------------------------

def _xlsx_rows(path: Path, sheet: str | None, skip: int):
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb[sheet] if sheet else wb.worksheets[0]
    it = ws.iter_rows(values_only=True)
    for _ in range(skip):
        next(it)
    return it


def _ecoli_pair_frame(rec, cols) -> pl.DataFrame:
    d = pl.DataFrame(rec, schema=cols, orient="row")
    return d.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in cols[2:]])


def _ecoli_finalize(d: pl.DataFrame, source: str, context: str, score: str, signif: str | None,
                    score_name: str, signif_name: str | None, pos_thr: float,
                    pos_p: float | None = 0.05, neg_p: float | None = 0.25) -> pl.DataFrame:
    """Shared label rule for the colony-array screens: positive = score <= pos_thr and (p < pos_p);
    negative = |score| below the screen median and (p > neg_p)."""
    d = d.with_columns(pl.Series("gene_a", ecol_map(d["g1"])), pl.Series("gene_b", ecol_map(d["g2"])))
    d = d.drop_nulls(["gene_a", "gene_b", score])
    med = float(d[score].abs().median())
    pos = pl.col(score) <= pos_thr
    neg = pl.col(score).abs() < med
    if signif and pos_p is not None:
        pos = pos & (pl.col(signif) < pos_p)
    if signif and neg_p is not None:
        neg = neg & (pl.col(signif) > neg_p)
    return finalize(d.select(
        pl.lit("ecol").alias("species"), pl.lit(source).alias("source"), pl.lit(context).alias("context"),
        pl.lit("deletion").alias("mechanism"), "gene_a", "gene_b",
        pl.col(score).alias("score"), pl.lit(score_name).alias("score_name"),
        (pl.col(signif) if signif else pl.lit(None, dtype=pl.Float64)).alias("signif"),
        pl.lit(signif_name, dtype=pl.String).alias("signif_name"),
        pl.when(pos).then(1).when(neg).then(0).otherwise(None).cast(pl.Int8).alias("label"),
    ))


def babu2011(condition: str = "RM") -> pl.DataFrame:
    """eSGA cell-envelope GI map: Babu et al., PLoS Genet 7:e1002377 (2011). 235,031 gene pairs
    (821 query genes x Keio array) scored in rich (RM) and minimal (MM) media.

    Score = E-score (negative = aggravating); the floor of -20 is the authors' code for a synthetic
    lethal (no double-mutant colony), which is 11% of all pairs. Positive: E <= -2.5 and p < 0.05.
    Negative: |E| below the median and p > 0.25. Hit rate is implausibly high (28% of labelled
    pairs) and the screen does not agree with the other three E. coli array maps.
    """
    col = {"RM": ("rm_e", "rm_p"), "MM": ("mm_e", "mm_p")}[condition]
    rec = []
    for r in _xlsx_rows(RAW / "babu2011_ecoli/TableS3_GI_scores.xlsx", "Table S3-trimmed", 3):
        if not r or not r[0]:
            continue
        g = str(r[0]).split("---")
        if len(g) == 2:
            rec.append((g[0], g[1], r[1], r[2], r[3], r[4]))
    d = _ecoli_pair_frame(rec, ["g1", "g2", "rm_e", "rm_p", "mm_e", "mm_p"])
    return _ecoli_finalize(d, "babu2011", f"BW25113_{condition}", col[0], col[1],
                           "E_score", "p", pos_thr=-2.5)


def gagarinova2016(condition: str = "RM") -> pl.DataFrame:
    """Translation-machinery GI map: Gagarinova et al., Cell Rep 17:904 (2016). All 43,168 pairs
    among ~338 query genes, scored in four conditions (RM rich, MM minimal, LT 23 C, HT 42 C).

    Score = GI score (negative = aggravating). Positive: GI <= -0.2 and p < 0.05; negative:
    |GI| below the median and p > 0.25. Hit rate 4%.
    """
    col = {"RM": ("rm_e", "rm_p"), "MM": ("mm_e", "mm_p"),
           "LT": ("lt_e", "lt_p"), "HT": ("ht_e", "ht_p")}[condition]
    rec = []
    for r in _xlsx_rows(RAW / "gagarinova2016_ecoli/mmc3.xlsx", "TableS3", 4):
        if not r or not r[0]:
            continue
        g = str(r[0]).split("---")
        if len(g) == 2:
            rec.append((g[0].split("__")[0], g[1].split("__")[0], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11]))
    d = _ecoli_pair_frame(rec, ["g1", "g2", "rm_e", "rm_p", "mm_e", "mm_p", "lt_e", "lt_p", "ht_e", "ht_p"])
    return _ecoli_finalize(d, "gagarinova2016", f"BW25113_{condition}", col[0], col[1],
                           "GI_score", "p", pos_thr=-0.2)


def kumar2016(condition: str = "UT") -> pl.DataFrame:
    """Genome-integrity GI map: Kumar et al., Cell Rep 14:648 (2016). 107,147 pairs, untreated (UT)
    and MMS-treated (DNA-damage) conditions, S-scores with p-values.

    Positive: S <= -2.5 and p < 0.05. Negative: |S| below the median and p > 0.25. Hit rate 24%.
    """
    col = {"UT": ("ut_e", "ut_p"), "MMS": ("mms_e", "mms_p")}[condition]
    rec = []
    for r in _xlsx_rows(RAW / "kumar2016_ecoli/Table_S2.xlsx", None, 4):
        if not r or not r[0] or not r[1]:
            continue
        rec.append((str(r[0]), str(r[1]), r[2], r[3], r[4], r[5]))
    d = _ecoli_pair_frame(rec, ["g1", "g2", "ut_e", "ut_p", "mms_e", "mms_p"])
    return _ecoli_finalize(d, "kumar2016", f"BW25113_{condition}", col[0], col[1],
                           "S_score", "p", pos_thr=-2.5)


def cote2016() -> pl.DataFrame:
    """Nutrient-stress GI map: Cote et al., mBio 7:e01714-16 (2016). 82 (+2) nutrient-stress query
    genes crossed with the whole Keio collection: 315,649 double deletions, each scored by the
    synthetic interaction value SIV = observed / expected colony density (1 = no interaction).

    Score reported here is SIV - 1 so that negative = synthetic sick, matching the house convention.
    Positive: SIV more than 2.5 SD below the mean (the authors' cut-off). Negative:
    0.9 < SIV < 1.1. Hit rate 0.4%, the only plausible one of the four array screens; it still does
    not agree with them.
    """
    import openpyxl

    wb = openpyxl.load_workbook(RAW / "cote2016_ecoli/mbo006163075st2.xlsx", read_only=True)
    rec = []
    for name in ("Nutrient stress genes", "panF", "gdhA (M9)"):
        ws = wb[name]
        it = ws.iter_rows(values_only=True)
        qs = [(str(x) if x is not None else "") for x in list(next(it))[1:]]
        for r in it:
            if not r or not r[0]:
                continue
            for q, v in zip(qs, r[1:]):
                if v is None or v == "" or not q:
                    continue
                rec.append((str(r[0]), q, float(v)))
    d = pl.DataFrame(rec, schema=["g1", "g2", "siv"], orient="row")
    thr = float(d["siv"].mean() - 2.5 * d["siv"].std())
    d = d.with_columns(pl.Series("gene_a", ecol_map(d["g1"])), pl.Series("gene_b", ecol_map(d["g2"]))) \
         .drop_nulls(["gene_a", "gene_b"])
    return finalize(d.select(
        pl.lit("ecol").alias("species"), pl.lit("cote2016").alias("source"),
        pl.lit("BW25113_LB").alias("context"), pl.lit("deletion").alias("mechanism"),
        "gene_a", "gene_b", (pl.col("siv") - 1.0).alias("score"), pl.lit("SIV_minus_1").alias("score_name"),
        pl.lit(None, dtype=pl.Float64).alias("signif"), pl.lit(None, dtype=pl.String).alias("signif_name"),
        pl.when(pl.col("siv") <= thr).then(1)
        .when((pl.col("siv") > 0.9) & (pl.col("siv") < 1.1)).then(0)
        .otherwise(None).cast(pl.Int8).alias("label"),
    ))


def ecol_single() -> pl.DataFrame:
    """Single-gene fitness for E. coli K-12 from the Keio collection (Baba et al. 2006).

    Baba's Supplementary Table 3 reports each deletion mutant's cell growth (OD600 after 22 h in
    LB) for 3,912 genes; `effect` = log2(OD600 / median OD600), so 0 = wild-type growth and
    negative = sicker (observed range -4.0 to +0.6). Genes with no viable deletion mutant (the 303
    essential-gene candidates of Supplementary Table 6) are given the 1st percentile of that
    distribution, -1.21, because their true single-loss effect is worse than any measured mutant's
    but the OD scale has no value for them. This mirrors the *S. pombe* treatment of inviable
    deletions. ~4.2k genes.
    """
    import pandas as pd

    v = pd.read_excel(RAW / "keio2006/S3_keio_mutants.xls", header=None, skiprows=4)
    b = v[8].astype(str)
    od = pd.to_numeric(v[17], errors="coerce")
    ok = b.str.match(r"^b\d+$") & od.notna()
    eff = np.log2(od[ok].to_numpy() / float(np.median(od[ok].to_numpy())))
    d = pl.DataFrame({"gene": b[ok].tolist(), "effect": eff.tolist()})
    floor = float(np.percentile(eff, 1))
    e = pd.read_excel(RAW / "keio2006/S6_essential_candidates.xls", header=None, skiprows=4)
    ess = [x for x in e[6].astype(str) if re.fullmatch(r"b\d+", x)]
    ess = pl.DataFrame({"gene": ess, "effect": [floor] * len(ess)}).filter(~pl.col("gene").is_in(d["gene"]))
    out = pl.concat([d, ess])
    return out.with_columns(pl.Series("gene", ecol_map(out["gene"]))).drop_nulls("gene") \
              .group_by("gene").agg(pl.col("effect").median())


# ============================================================================================
# Reproducibility checks (`slbench audit`)
#
# Every `<source>_checks()` returns a list of dicts with the keys used by
# eukaryotes_extra.frost2012_checks(): "check", "auroc", "pos", and where meaningful "n",
# "ci95" and "source". They recompute the reproducibility evidence from the raw files;
# `slbench audit` reports it.
# ============================================================================================

def _auroc(y, s) -> float | None:
    """AUROC with at least 5 of each class, else None. `s` must already point the right way."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.sum() < 5 or (1 - y).sum() < 5:
        return None
    return float(roc_auc_score(y, s))


def _auroc_ci(y, s, n_boot: int = 500, seed: int = 0) -> tuple[float | None, list[float] | None]:
    """AUROC plus a percentile bootstrap 95% CI over pairs (the unit of observation)."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.sum() < 5 or (1 - y).sum() < 5:
        return None, None
    point = float(roc_auc_score(y, s))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    boots = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) == 2:
            boots.append(roc_auc_score(y[b], s[b]))
    ci = [round(float(x), 3) for x in np.percentile(boots, [2.5, 97.5])] if boots else None
    return point, ci


def _cross_check(a: pl.DataFrame, b: pl.DataFrame, label: str, ci: bool = False) -> dict:
    """Labels of source `a` scored by source `b`'s score over their shared unordered pairs.

    Both frames come from the source functions, so gene_a < gene_b already and the score
    convention is negative = aggravating in both; hence the predictor is -score.
    """
    j = a.select("gene_a", "gene_b", "label").drop_nulls("label").join(
        b.select("gene_a", "gene_b", pl.col("score").alias("s2")), on=["gene_a", "gene_b"]).drop_nulls("s2")
    j = j.sort("gene_a", "gene_b")  # deterministic row order, so the bootstrap CI is reproducible
    y = j["label"].to_numpy()
    s = -j["s2"].to_numpy()
    if ci:
        auroc, ci95 = _auroc_ci(y, s)
    else:
        auroc, ci95 = _auroc(y, s), None
    out = {"check": label, "auroc": auroc, "pos": int(y.sum()), "n": j.height}
    if ci95 is not None:
        out["ci95"] = ci95
    return out


# --------------------------------------------------------------------------------------------
# dualtnseq2025 (spne Dual Tn-seq): leave-one-run-out over the 5 independent runs, the
# all-insertion rescoring, and the independent one-versus-all RB-TnSeq assay.

_ZIK_RUNS = ("run1_genepairs_min6", "run2_genepairs_min3", "run3_genepairs_min3",
             "run4_genepairs_min4", "run5_genepairs_min4")


def _zik_run_stats(name: str, min_expect_strains: float = 5.0, nbins: int = 30) -> pl.DataFrame:
    """Re-score one Dual Tn-seq run from its own gene-pair counts with the authors' recipe
    (dblStats.R: marginal-product expectation, then a median ratio per 30x30 chromosomal-position
    bin), giving an independent zStrains/readRatio for that run alone."""
    p = pl.read_csv(_ZIK / f"{name}.tsv.gz", separator="\t")
    t1 = p.group_by("locusId1").agg(pl.col("nStrains").sum().alias("t1S"), pl.col("nReads").sum().alias("t1R"))
    t2 = p.group_by("locusId2").agg(pl.col("nStrains").sum().alias("t2S"), pl.col("nReads").sum().alias("t2R"))
    p = p.join(t1, on="locusId1").join(t2, on="locusId2")
    tot_s, tot_r = p["nStrains"].sum(), p["nReads"].sum()
    p = p.with_columns((pl.col("t1S").cast(pl.Float64) * pl.col("t2S") / tot_s).alias("eS"),
                       (pl.col("t1R").cast(pl.Float64) * pl.col("t2R") / tot_r).alias("eR"))
    fwd = p.filter(pl.col("locusId1") < pl.col("locusId2")).select(
        "locusId1", "locusId2", "nStrains", "nReads", "eS", "eR")
    rev = p.filter(pl.col("locusId1") > pl.col("locusId2")).select(
        pl.col("locusId2").alias("locusId1"), pl.col("locusId1").alias("locusId2"),
        pl.col("nStrains").alias("nS2"), pl.col("nReads").alias("nR2"),
        pl.col("eS").alias("eS2"), pl.col("eR").alias("eR2"))
    d = fwd.join(rev, on=["locusId1", "locusId2"])
    d = d.with_columns((pl.col("nStrains") + pl.col("nS2")).alias("nS"),
                       (pl.col("nReads") + pl.col("nR2")).alias("nR"),
                       (pl.col("eS") + pl.col("eS2")).alias("expS"),
                       (pl.col("eR") + pl.col("eR2")).alias("expR")).filter(pl.col("expS") >= 2)
    pos = _zik_gene_pos()
    d = d.with_columns(pl.col("locusId1").replace_strict(pos, default=None).alias("pos1"),
                       pl.col("locusId2").replace_strict(pos, default=None).alias("pos2")) \
         .drop_nulls(["pos1", "pos2"])
    allpos = np.r_[d["pos1"].to_numpy(), d["pos2"].to_numpy()]
    edges = np.linspace(allpos.min() - 1, allpos.max() + 1, nbins + 1)
    b1 = np.digitize(d["pos1"].to_numpy(), edges)
    b2 = np.digitize(d["pos2"].to_numpy(), edges)
    d = d.with_columns(pl.Series("bin12", [f"{x}.{y}" for x, y in zip(b1, b2)]),
                       (pl.col("nS") / pl.col("expS")).alias("rS"), (pl.col("nR") / pl.col("expR")).alias("rR"))
    med = d.group_by("bin12").agg(pl.col("rS").median().alias("mS"), pl.col("rR").median().alias("mR"))
    d = d.join(med, on="bin12").with_columns((pl.col("expS") * pl.col("mS")).alias("expSadj"),
                                             (pl.col("expR") * pl.col("mR")).alias("expRadj"))
    d = d.filter(pl.col("expSadj") >= min_expect_strains)
    return d.with_columns(
        ((pl.col("nS") - pl.col("expSadj")) / pl.col("expSadj").sqrt()).alias("zStrains"),
        (pl.col("nR") / pl.col("expRadj")).alias("readRatio"),
        (pl.col("pos1") - pl.col("pos2")).abs().alias("dist"),
    ).select("locusId1", "locusId2", "expSadj", "zStrains", "readRatio", "dist")


def dualtnseq2025_checks() -> list[dict]:
    """(1) leave-one-run-out across the 5 independent Dual Tn-seq runs, each re-scored from its own
    counts; (2) the authors' pooled labels scored by each single run; (3) the 10-90% labels scored
    by the independent all-insertion (0-100%) rescoring of table S1; (4) enrichment of the
    independent one-versus-all RB-TnSeq hits (needs `pyreadr`; skipped with auroc None if absent)."""
    key = ["locusId1", "locusId2"]
    runs = {n.split("_")[0]: _zik_run_stats(n) for n in _ZIK_RUNS}
    out: list[dict] = []

    # (1) labels from one run, scored by the mean zStrains of the other four
    for a, held in runs.items():
        others = [r for r in runs if r != a]
        o = runs[others[0]].select(*key, pl.col("zStrains").alias("z0"))
        for i, r in enumerate(others[1:], 1):
            o = o.join(runs[r].select(*key, pl.col("zStrains").alias(f"z{i}")), on=key)
        j = held.join(o, on=key).filter(pl.col("dist") > _ZIK_MIN_DIST)
        zo = np.nanmean(np.c_[[j[f"z{i}"].to_numpy() for i in range(len(others))]], axis=0)
        pos = ((j["zStrains"] <= -3) & (j["readRatio"] <= 0.2)).to_numpy()
        neg = ((j["zStrains"].abs() < 1) & (j["readRatio"] > 0.8) & (j["readRatio"] < 1.25)).to_numpy()
        m = pos | neg
        out.append({"source": "dualtnseq2025", "n": j.height, "pos": int(pos.sum()),
                    "check": f"within-study: labels from {a} alone, scored by the mean of the other 4 runs",
                    "auroc": _auroc(pos[m], -zo[m])})

    # (2) the pooled (published) labels scored by each single run
    c = _zik_pairs()
    for a, d in runs.items():
        j = c.select(*key, "zStrains", "readRatio").join(
            d.select(*key, pl.col("zStrains").alias("zr")), on=key)
        pos = ((j["zStrains"] <= -3) & (j["readRatio"] <= 0.2)).to_numpy()
        neg = ((j["zStrains"].abs() < 1) & (j["readRatio"] > 0.8) & (j["readRatio"] < 1.25)).to_numpy()
        m = pos | neg
        out.append({"source": "dualtnseq2025", "n": int(m.sum()), "pos": int(pos.sum()),
                    "check": f"within-study (optimistic, labels pooled over all runs): scored by {a}",
                    "auroc": _auroc(pos[m], -j["zr"].to_numpy()[m])})

    # (3) 10-90% labels vs the independent all-insertion (0-100%) scoring
    t = pl.read_csv(_ZIK / "tableS1.tsv.gz", separator="\t", null_values=[""])
    t = t.rename({x: x.replace("(", "_").replace(")", "").replace("%", "").replace("-", "_") for x in t.columns})
    t = t.with_columns([pl.col(x).cast(pl.Float64, strict=False) for x in t.columns
                        if x.startswith(("zStrains", "readRatio", "expectStrains"))])
    pos_map = _zik_gene_pos()
    t = t.with_columns((pl.col("locusId1").replace_strict(pos_map, default=None)
                        - pl.col("locusId2").replace_strict(pos_map, default=None)).abs().alias("dist"))
    t = t.filter(pl.col("dist") > _ZIK_MIN_DIST).drop_nulls(
        ["zStrains_10_90", "readRatio_10_90", "zStrains_0_100"])
    pos = ((t["zStrains_10_90"] <= -3) & (t["readRatio_10_90"] <= 0.2)).to_numpy()
    neg = ((t["zStrains_10_90"].abs() < 1) & (t["readRatio_10_90"] > 0.8)
           & (t["readRatio_10_90"] < 1.25)).to_numpy()
    m = pos | neg
    out.append({"source": "dualtnseq2025", "n": int(m.sum()), "pos": int(pos.sum()),
                "check": "within-study: central-10-90% labels scored by the all-insertion (0-100%) rescoring",
                "auroc": _auroc(pos[m], -t["zStrains_0_100"].to_numpy()[m])})

    out.extend(_dualtnseq2025_onevsall_checks())
    return out


def _dualtnseq2025_onevsall_checks() -> list[dict]:
    """Independent assay: one-versus-all RB-TnSeq (library transferred into single deletion
    backgrounds) from figshare's 1vsall.image. Only the significant-hit tables are readable, so the
    computable statistic is how well zStrains ranks those hits among pairs of the same query genes,
    plus their enrichment in the SLB positives."""
    try:
        import pyreadr
    except ImportError:
        return [{"source": "dualtnseq2025", "auroc": None, "pos": None,
                 "check": "independent assay (one-vs-all RB-TnSeq): skipped, pyreadr not installed "
                          "(run with `uv run --with pyreadr`)"}]
    r = pyreadr.read_r(str(_ZIK / "1vsall.image"))
    h = pl.concat([pl.from_pandas(r[k][["deleted", "locusId", "fitnorm"]]) for k in ("ML2hits", "ML3hits")])
    h = h.with_columns(pl.Series("ga", spne_map(h["deleted"])), pl.Series("gb", spne_map(h["locusId"]))) \
         .drop_nulls(["ga", "gb"])
    h = h.with_columns(pl.min_horizontal("ga", "gb").alias("gene_a"), pl.max_horizontal("ga", "gb").alias("gene_b"))
    backgrounds = set(h["ga"].to_list()) | set(h["gb"].to_list())
    hits = h.filter(pl.col("fitnorm") < 0).select("gene_a", "gene_b").unique().with_columns(pl.lit(1).alias("ova"))
    d = dualtnseq2025().filter(pl.col("gene_a").is_in(backgrounds) | pl.col("gene_b").is_in(backgrounds))
    d = d.join(hits, on=["gene_a", "gene_b"], how="left").with_columns(pl.col("ova").fill_null(0))
    lab = d.drop_nulls("label")
    p, n = lab.filter(pl.col("label") == 1), lab.filter(pl.col("label") == 0)
    rate_p = float(p["ova"].mean()) if p.height else float("nan")
    rate_n = float(n["ova"].mean()) if n.height else float("nan")
    fold = round(rate_p / rate_n, 1) if rate_n else None
    return [
        {"source": "dualtnseq2025", "n": lab.height, "pos": int(lab["ova"].sum()),
         "check": "independent assay: zStrains ranks the one-vs-all RB-TnSeq aggravating hits "
                  "(labelled pairs involving a deletion-background gene)",
         "auroc": _auroc(lab["ova"].to_numpy(), -lab["score"].to_numpy())},
        {"source": "dualtnseq2025", "n": lab.height, "pos": int(p["ova"].sum()),
         "check": f"independent assay: one-vs-all aggravating hits are {rate_p:.3%} of SLB positives "
                  f"({p.height}) vs {rate_n:.3%} of negatives ({n.height}), {fold}x enrichment",
         "auroc": _auroc(lab["ova"].to_numpy(), lab["label"].to_numpy())},
    ]


# --------------------------------------------------------------------------------------------
# koo2025 (bsub double-CRISPRi): the sgRNA1 <-> sgRNA2 orientation swap is an independent
# re-measurement of the same gene pair with a different construct and barcode.

def koo2025_checks(thresholds: tuple[float, ...] = (-0.75, -1.0, -1.25, -1.5, -2.0)) -> list[dict]:
    """(1) orientation swap on the mean-of-three-exponential-intervals score at a range of positive
    cut-offs (the SLB rule is -1.5), in both directions; (2) the same for each single interval, to
    show why the intervals are merged; (3) labels from the first interval scored by the
    stationary-phase-recovery interval T4-T5, an independent culture."""
    d = pl.concat([_koo_long(s).with_columns(pl.lit(s).alias("tp")) for s in _KOO_EXPONENTIAL])
    d = d.with_columns((pl.col("g1") == pl.col("gene_a")).alias("is_fwd"))
    o = d.group_by("gene_a", "gene_b", "is_fwd").agg(pl.col("score").mean().alias("score"))
    f = o.filter(pl.col("is_fwd")).select("gene_a", "gene_b", pl.col("score").alias("sf"))
    r = o.filter(~pl.col("is_fwd")).select("gene_a", "gene_b", pl.col("score").alias("sr"))
    j = f.join(r, on=["gene_a", "gene_b"])
    med = float(pl.concat([j["sf"], j["sr"]]).abs().median())
    out: list[dict] = []
    for t in thresholds:
        for lab, sco in (("sgRNA1-side", "sf"), ("sgRNA2-side", "sr")):
            other = "sr" if sco == "sf" else "sf"
            pos = (j[sco] <= t).to_numpy()
            neg = (j[sco].abs() < med).to_numpy()
            m = pos | neg
            out.append({"source": "koo2025", "n": j.height, "pos": int(pos.sum()),
                        "check": f"within-study: orientation swap, labels from the {lab} measurement "
                                 f"(mean GI <= {t}), scored by the other orientation",
                        "auroc": _auroc(pos[m], -j[other].to_numpy()[m])})
    # single intervals, and the independent recovery-phase interval
    per = {s.split()[0] + s.split()[2]: _koo_long(s) for s in
           (*_KOO_EXPONENTIAL, "T4 to T5 GI scores", "T6 to T7 GI scores")}
    coll = {k: v.group_by("gene_a", "gene_b").agg(pl.col("score").mean()) for k, v in per.items()}
    base = coll["T0T1"]
    bmed = float(base["score"].abs().median())
    for other in ("T1T2", "T2T3", "T4T5", "T6T7"):
        jj = base.join(coll[other].select("gene_a", "gene_b", pl.col("score").alias("s2")), on=["gene_a", "gene_b"])
        pos = (jj["score"] <= -1.5).to_numpy()
        neg = (jj["score"].abs() < bmed).to_numpy()
        m = pos | neg
        out.append({"source": "koo2025", "n": jj.height, "pos": int(pos.sum()),
                    "check": f"within-study (weaker): labels from interval T0-T1 alone (GI <= -1.5), "
                             f"scored by interval {other[:2]}-{other[2:]}",
                    "auroc": _auroc(pos[m], -jj["s2"].to_numpy()[m])})
    return out


# --------------------------------------------------------------------------------------------
# crisprtnseq2024 (spne CRISPRi-TnSeq): split-half over the independent transposon libraries
# screened per CRISPRi query.

def crisprtnseq2024_checks() -> list[dict]:
    """Within-study split-half: for each CRISPRi query screened against more than one transposon
    library table, re-derive labels from one subset of its tables and score them with the mean
    fitness difference of the held-out tables. Reported per query and as the median over splits."""
    import itertools

    long = _crisprtnseq2024_long().drop_nulls("gene")
    out: list[dict] = []
    aurocs: list[float] = []
    for query in sorted(set(long["query"].drop_nulls().to_list())):
        q = long.filter(pl.col("query") == query)
        tabs = sorted(set(q["tab"].to_list()))
        if len(tabs) < 2:
            continue
        best = []
        for k in range(1, len(tabs) // 2 + 1):
            for half in itertools.combinations(tabs, k):
                a = q.filter(pl.col("tab").is_in(list(half))).group_by("gene").agg(
                    pl.col("value").mean().alias("score"), pl.col("p").min().alias("p"))
                b = q.filter(~pl.col("tab").is_in(list(half))).group_by("gene").agg(
                    pl.col("value").mean().alias("s2"))
                j = a.join(b, on="gene")
                med = float(j["score"].abs().median())
                pos = ((j["score"] < -0.1) & (j["p"] < 0.05)).to_numpy()
                neg = ((j["score"].abs() < med) & (j["p"] > 0.05)).to_numpy()
                m = pos | neg
                au = _auroc(pos[m], -j["s2"].to_numpy()[m])
                if au is not None:
                    best.append((au, int(pos.sum()), j.height))
        if not best:
            continue
        aus = sorted(x[0] for x in best)
        median_au = aus[len(aus) // 2]
        pick = next(x for x in best if x[0] == median_au)
        aurocs.append(median_au)
        out.append({"source": "crisprtnseq2024", "n": pick[2], "pos": pick[1],
                    "check": f"within-study split-half over transposon libraries, query {query} "
                             f"(median of {len(best)} splits)", "auroc": median_au})
    if aurocs:
        aurocs.sort()
        out.append({"source": "crisprtnseq2024", "n": len(aurocs), "pos": None,
                    "check": f"within-study split-half over transposon libraries, median over the "
                             f"{len(aurocs)} queries with >1 library", "auroc": aurocs[len(aurocs) // 2]})
    out.extend(x for x in spne_cross_checks() if "crisprtnseq2024" in x["check"])
    return out


# --------------------------------------------------------------------------------------------
# Cross-study: the three S. pneumoniae D39V screens against each other, both directions.

def spne_cross_checks() -> list[dict]:
    """All six directed comparisons among the three pneumococcal screens (dualcrispri2025, which is
    SLB's current spne source, crisprtnseq2024 and dualtnseq2025), with bootstrap CIs, plus the
    single-gene-fitness controls that show the disagreement is not an ID-mapping artifact.

    This is the evidence behind the lead's decision to keep all three as measurements only."""
    from slbench import fitness
    from slbench.sources.bacteria import dualcrispri2025

    src = {"dualcrispri2025": dualcrispri2025(), "crisprtnseq2024": crisprtnseq2024(),
           "dualtnseq2025": dualtnseq2025()}
    out: list[dict] = []
    for a, b in ((x, y) for x in src for y in src if x != y):
        c = _cross_check(src[a], src[b], f"cross-study: {a} labels scored by {b}", ci=True)
        c["source"] = a
        if c["auroc"] is None and c["n"] == 0:
            c["check"] += " (no overlapping pairs: Dual Tn-seq needs viable transposon insertions, "
            c["check"] += "CRISPRi-TnSeq targets essential genes)"
        out.append(c)
    # controls: the same three studies' single-gene measures do agree, so the IDs line up
    singles = {"dualcrispri2025": fitness.spne_single(), "crisprtnseq2024": spne_single_tnseq(),
               "dualtnseq2025": spne_single_dualtnseq()}
    for a, b in (("dualcrispri2025", "crisprtnseq2024"), ("dualcrispri2025", "dualtnseq2025"),
                 ("crisprtnseq2024", "dualtnseq2025")):
        j = singles[a].rename({"effect": "ea"}).join(singles[b].rename({"effect": "eb"}), on="gene")
        rho = j.select(pl.corr("ea", "eb", method="spearman")).item()
        out.append({"source": a, "n": j.height, "pos": None, "auroc": None,
                    "check": f"ID-mapping control: single-gene effect of {a} vs {b}, "
                             f"Spearman {rho:.2f} over {j.height} shared genes"})
    return out


# --------------------------------------------------------------------------------------------
# E. coli colony-array screens: within-study across growth conditions, and cross-study against
# each of the other three screens.

_ECOLI_CHECK_SOURCES = {
    "babu2011": (babu2011, ("RM", "MM")),
    "gagarinova2016": (gagarinova2016, ("RM", "MM", "LT", "HT")),
    "kumar2016": (kumar2016, ("UT", "MMS")),
    "cote2016": (cote2016, ()),
}


@functools.cache
def _ecoli_cached(source: str, condition: str | None) -> pl.DataFrame:
    fn, _ = _ECOLI_CHECK_SOURCES[source]
    return fn() if condition is None else fn(condition)


def _ecoli_source_checks(source: str) -> list[dict]:
    conditions = _ECOLI_CHECK_SOURCES[source][1]
    primary = _ecoli_cached(source, conditions[0] if conditions else None)
    out: list[dict] = []
    # within-study: same library, other growth condition. Not independent replicates, but it is the
    # only internal check these supplements allow, and it is the contrast with the cross-study rows.
    for other in conditions[1:]:
        c = _cross_check(primary, _ecoli_cached(source, other),
                         f"within-study (same library, other condition): {source} {conditions[0]} labels "
                         f"scored by {source} {other}")
        c["source"] = source
        out.append(c)
    if len(conditions) > 1:
        c = _cross_check(_ecoli_cached(source, conditions[1]), primary,
                         f"within-study (same library, other condition): {source} {conditions[1]} labels "
                         f"scored by {source} {conditions[0]}")
        c["source"] = source
        out.append(c)
    # cross-study against the other three screens, in this source's labelling direction
    for other in _ECOLI_CHECK_SOURCES:
        if other == source:
            continue
        ocond = _ECOLI_CHECK_SOURCES[other][1]
        c = _cross_check(primary, _ecoli_cached(other, ocond[0] if ocond else None),
                         f"cross-study: {source} labels scored by {other}", ci=True)
        c["source"] = source
        if c["auroc"] is None:
            c["check"] += (" (no usable overlap: disjoint gene sets)" if c["n"] < 10 else
                           f" (undefined: only {c['pos']} of this source's positives fall in the "
                           f"{c['n']}-pair overlap)")
        out.append(c)
    return out


def babu2011_checks() -> list[dict]:
    """eSGA cell-envelope map: rich vs minimal medium within-study, then against the other three
    E. coli array screens. Every cross-study AUROC is at chance, which is why this source is
    measurements-only."""
    return _ecoli_source_checks("babu2011")


def gagarinova2016_checks() -> list[dict]:
    """Translation-machinery map: rich medium labels vs the minimal/23 C/42 C conditions of the same
    library, then against the other three E. coli array screens."""
    return _ecoli_source_checks("gagarinova2016")


def kumar2016_checks() -> list[dict]:
    """Genome-integrity map: untreated vs MMS within-study, then against the other three E. coli
    array screens."""
    return _ecoli_source_checks("kumar2016")


def cote2016_checks() -> list[dict]:
    """Nutrient-stress x Keio map. No within-study check is possible (the biological duplicates were
    averaged before publication), so only the cross-study directions are reported."""
    return _ecoli_source_checks("cote2016")


def ecoli_array_checks() -> list[dict]:
    """All four E. coli colony-array screens' checks in one list (convenience wrapper)."""
    return [c for s in _ECOLI_CHECK_SOURCES for c in _ecoli_source_checks(s)]
