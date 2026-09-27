"""Billmann, Costanzo et al. 2026 Cell (doi 10.1016/j.cell.2026.03.044): HAP1 global GI map.

Design: HAP1 parental + congenic query-KO clones, each screened genome-wide with TKOv3 (query x genome,
hub-structured). Mendeley dataset 10.17632/bpcpfns6vb (CC BY 4.0), Data S4 sheets:
  qGI_scores / qGI_FDR   wide matrices, 17,804 library genes x one column per screen (GIN id + medium)
  pairwise GI_Complete dataset   the authors' called GIs (|qGI|>0.3, FDR<0.1) - not needed here.
Data S1 = query/screen table (medium type per GIN id).

Only the standard-medium arm is used: "rich" = IMDM (the standard HAP1 medium); "min" = DMEM 10 mM glucose,
1 mM glutamine, a metabolic sensitising condition. 166 rich screens / 128 queries. Replicate screens of the
same query in rich medium are averaged (score = mean qGI, signif = median FDR) and exposed separately by
`replicates()` (28 queries, 2-5 screens each). SLX1 (ambiguous SLX1A/SLX1B) does not resolve and is dropped.

Label rule (SLB, stricter than the authors' |qGI|>0.3 & FDR<0.1):
  positive: qGI < -0.5 and FDR < 0.05  (on the replicate-screen mean qGI / median FDR)
  negative: |qGI| < median |qGI| of that query's screen and FDR > 0.5
  else null.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

from slbench.sources import finalize

RAW = Path("data/raw/billmann2026")
S4 = RAW / "Data_S4_HAP1_genetic_interaction_dataset.xlsx"
S1 = RAW / "Data_S1_Query_gene_and_screen_information.xlsx"
CACHE = RAW / "_parsed"

POS_QGI, POS_FDR = -0.5, 0.05
NEG_FDR = 0.5

_CELL = re.compile(rb'<c r="([A-Z]+)(\d+)"(?: s="\d+")?(?: t="(\w+)")?(?: s="\d+")?>(?:<v>([^<]*)</v>)?')


def _col_index(letters: bytes) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + (ch - 64)
    return n - 1


def _shared_strings(z: zipfile.ZipFile) -> list[str]:
    x = z.read("xl/sharedStrings.xml").decode()
    out = []
    for si in re.findall(r"<si>(.*?)</si>", x, flags=re.S):
        out.append("".join(re.findall(r"<t[^>]*>([^<]*)</t>", si)))
    import html
    return [html.unescape(s) for s in out]


def _read_wide(sheet_no: int) -> pl.DataFrame:
    """Read a wide gene x screen sheet of Data S4 into polars (first column = library gene)."""
    with zipfile.ZipFile(S4) as z:
        ss = _shared_strings(z)
        x = z.read(f"xl/worksheets/sheet{sheet_no}.xml")
    rows, cols, vals, types = [], [], [], []
    for m in _CELL.finditer(x):
        cols.append(_col_index(m.group(1)))
        rows.append(int(m.group(2)) - 1)
        types.append(m.group(3) or b"")
        vals.append(m.group(4))
    rows_a, cols_a = np.array(rows), np.array(cols)
    nr, nc = rows_a.max() + 1, cols_a.max() + 1
    num = np.full((nr, nc), np.nan)
    txt: dict[tuple[int, int], str] = {}
    for r, c, t, v in zip(rows, cols, types, vals):
        if v is None:
            continue
        if t == b"s":
            txt[(r, c)] = ss[int(v)]
        elif t in (b"str", b"inlineStr"):
            txt[(r, c)] = v.decode()
        else:
            num[r, c] = float(v)
    header = [txt.get((0, c), f"col{c}") for c in range(nc)]
    genes = [txt.get((r, 0)) for r in range(1, nr)]
    data = {header[0]: genes}
    for c in range(1, nc):
        data[header[c]] = num[1:, c]
    return pl.DataFrame(data)


def _long(which: str) -> pl.DataFrame:
    """Long table: library_gene, screen, value. Cached as parquet."""
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{which}.parquet"
    if not p.exists():
        w = _read_wide({"qGI": 3, "FDR": 4, "query_lfc": 2, "wt_lfc": 1}[which])
        g = w.columns[0]
        w.rename({g: "library_gene"}).unpivot(index="library_gene", variable_name="screen", value_name=which) \
            .drop_nans(which).drop_nulls(which).write_parquet(p)
    return pl.read_parquet(p)


def screens() -> pl.DataFrame:
    """Screen table from Data S1: GIN id, query gene, medium."""
    import openpyxl

    wb = openpyxl.load_workbook(S1, read_only=True)
    rows = list(wb.worksheets[0].iter_rows(values_only=True))
    h = rows[0]
    df = pl.DataFrame([dict(zip(h, r)) for r in rows[1:] if r[0]], infer_schema_length=None)
    return df.select(pl.col("Query Gene").alias("query"), pl.col("Media.type").alias("medium"),
                     pl.col("GIN_ID").alias("gin"), pl.col("Query Description").alias("allele"))


def _screen_map(cols: list[str]) -> pl.DataFrame:
    """qGI column headers are '<QUERY>_<GIN number>_<min|rich>'."""
    out = []
    for c in cols:
        q, n, med = re.match(r"^(.+)_(\d+)_(min|rich)$", c).groups()
        out.append({"screen": c, "query": q, "gin": f"GIN{int(n):03d}", "medium": med})
    return pl.DataFrame(out)


def pair_table() -> pl.DataFrame:
    """Per (query, library gene, screen) qGI + FDR, rich medium only."""
    q = _long("qGI")
    f = _long("FDR")
    d = q.join(f, on=["library_gene", "screen"], how="left")
    sm = _screen_map(d["screen"].unique().to_list())
    d = d.join(sm, on="screen", how="left").filter(pl.col("medium") == "rich")
    return d


def _label(qgi: pl.Expr, fdr: pl.Expr, med: pl.Expr) -> pl.Expr:
    return (pl.when((qgi < POS_QGI) & (fdr < POS_FDR)).then(1)
            .when((qgi.abs() < med) & (fdr > NEG_FDR)).then(0)
            .otherwise(None).cast(pl.Int8))


def load() -> pl.DataFrame:
    d = pair_table()
    # neutral band per screen: |qGI| below that screen's median |qGI|
    d = d.with_columns(pl.col("qGI").abs().median().over("screen").alias("med"))
    # collapse replicate screens of the same query (28 queries have 2-5 rich-medium screens):
    # score = mean qGI, signif = median FDR, neutral cut = mean of the screens' medians
    d = d.group_by("query", "library_gene").agg(pl.col("qGI").mean(), pl.col("FDR").median(), pl.col("med").mean())
    d = d.with_columns(_label(pl.col("qGI"), pl.col("FDR"), pl.col("med")).alias("label"))
    df = d.select(
        pl.lit("human").alias("species"), pl.lit("billmann2026").alias("source"), pl.lit("HAP1").alias("context"),
        pl.lit("CRISPR-KO").alias("mechanism"), pl.col("query").alias("gene_a"), pl.col("library_gene").alias("gene_b"),
        pl.col("qGI").alias("score"), pl.lit("qGI").alias("score_name"),
        pl.col("FDR").alias("signif"), pl.lit("qGI_FDR").alias("signif_name"), "label",
    )
    # finalize merges the two orientations when both genes were queries (score mean, conflicting labels -> null)
    return finalize(df, "human")


def replicates() -> pl.DataFrame:
    """Queries screened >= 2 times in rich medium: one column per independent screen (gi_rep1..)."""
    d = pair_table()
    reps = d.select("query", "screen", "gin").unique().sort("query", "gin")
    reps = reps.with_columns(pl.col("gin").rank("ordinal").over("query").cast(pl.Int32).alias("rep"))
    multi = reps.group_by("query").agg(pl.len().alias("n")).filter(pl.col("n") >= 2)["query"]
    d = d.join(reps.select("screen", "rep"), on="screen").filter(pl.col("query").is_in(multi.implode()))
    w = d.with_columns(pl.format("gi_rep{}", "rep").alias("r")).pivot(
        on="r", index=["query", "library_gene"], values="qGI")
    f = d.with_columns(pl.format("fdr_rep{}", "rep").alias("r")).pivot(
        on="r", index=["query", "library_gene"], values="FDR")
    w = w.join(f, on=["query", "library_gene"])
    return w.select(pl.col("query").alias("gene_a"), pl.col("library_gene").alias("gene_b"),
                    pl.lit("HAP1").alias("context"), *sorted(c for c in w.columns if c.startswith(("gi_rep", "fdr_rep"))))
