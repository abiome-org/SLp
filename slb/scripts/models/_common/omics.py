"""Cached single-gene omics matrices for the statistical SL-inference models (DAISY, ISLE, MiSL, SLIdR, ...).

All matrices are (samples x genes) float32 DataFrames with current HGNC symbols as columns.
Sources (all single-gene, permitted by the leakage rules):
  DepMap 24Q4: CRISPR gene effect, expression (TPM log1p), relative CN, damaging mutations  (data/raw/depmap)
  DEMETER2 combined RNAi gene effect (Achilles + DRIVE + Marcotte)                             (data/raw/demeter2)
  TCGA PanCanAtlas via UCSC Xena: EB++ RNA-seq, GISTIC2 thresholded CN, MC3 non-silent mutations,
  TCGA-CDR survival + cancer type                                                               (data/raw/tcga_pancan)
Cache: external/models/_statsl_cache/*.parquet
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slb  # noqa: E402

CACHE = slb.ROOT / "external/models/_statsl_cache"
CACHE.mkdir(parents=True, exist_ok=True)
RAW = slb.RAW


@functools.cache
def _res():
    return slb.symbol_resolver()


def _depmap_cols(cols) -> list:
    """'TSPAN6 (7105)' -> HGNC symbol via entrez first, then symbol."""
    res = _res()
    out = []
    for c in cols:
        sym = c.split(" (")[0]
        ent = c.split("(")[-1].rstrip(")") if "(" in c else None
        out.append(res(ent) or res(sym) or None)
    return out


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, [c is not None for c in df.columns]]
    return df.loc[:, ~pd.Index(df.columns).duplicated()]


def _cached(name, build):
    p = CACHE / f"{name}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    df = build()
    df.columns = [str(c) for c in df.columns]
    tmp = p.with_suffix(".tmp.parquet")
    df.to_parquet(tmp)
    tmp.rename(p)  # atomic: concurrent readers never see a partial file
    return df


def _depmap_matrix(fname):
    df = pd.read_csv(RAW / "depmap" / fname, index_col=0)
    df.columns = _depmap_cols(df.columns)
    return _dedupe(df).astype("float32")


@functools.cache
def crispr():
    return _cached("depmap_crispr", lambda: _depmap_matrix("CRISPRGeneEffect_24Q4.csv"))


@functools.cache
def ccle_expr():
    def b():
        df = pd.read_csv(RAW / "depmap/OmicsExpressionProteinCodingGenesTPMLogp1_24Q4.csv", index_col=0)
        if "ModelID" in df.columns:  # newer layout
            df = df.set_index("ModelID")
        df = df.select_dtypes("number")
        df.columns = _depmap_cols(df.columns)
        return _dedupe(df).astype("float32")
    return _cached("ccle_expr", b)


@functools.cache
def ccle_cn():
    return _cached("ccle_cn", lambda: _depmap_matrix("OmicsCNGene_24Q4.csv"))


@functools.cache
def ccle_mut():
    return _cached("ccle_mut_damaging", lambda: _depmap_matrix("OmicsSomaticMutationsMatrixDamaging_24Q4.csv"))


@functools.cache
def rnai():
    """DEMETER2 gene effect (lines indexed by DepMap ACH id x genes)."""
    def b():
        df = pd.read_csv(RAW / "demeter2/D2_gene_effect.csv", index_col=0)  # lines (ACH ids) x genes
        df.columns = _depmap_cols(df.columns)
        return _dedupe(df).astype("float32")
    return _cached("demeter2", b)


def _xena_matrix(fname, transform=None):
    df = pd.read_csv(RAW / "tcga_pancan" / fname, sep="\t", index_col=0)
    res = _res()
    df.index = [res(str(i)) for i in df.index]
    df = df[[i is not None for i in df.index]]
    df = df[~df.index.duplicated()]
    df = df.T.astype("float32")
    if transform:
        df = transform(df)
    return df


@functools.cache
def tcga_clin():
    def b():
        s = pd.read_csv(RAW / "tcga_pancan/survival_S1_20171025.tsv", sep="\t")
        s = s.rename(columns={"cancer type abbreviation": "type", "age_at_initial_pathologic_diagnosis": "age"})
        return s[["sample", "type", "age", "gender", "race", "OS", "OS.time"]].set_index("sample")
    return _cached("tcga_clin", b)


def _primary(df):
    # primary tumours (sample type 01-09), one per sample barcode
    keep = [i for i in df.index if len(i) >= 15 and i[13:15].isdigit() and int(i[13:15]) < 10]
    return df.loc[keep]


@functools.cache
def tcga_expr():
    # the Xena EB++ matrix is already log2(normalised RSEM + 1)
    return _cached("tcga_expr", lambda: _primary(_xena_matrix("EBpp_geneExp.xena.gz")))


@functools.cache
def tcga_cna():
    return _cached("tcga_cna", lambda: _primary(_xena_matrix("gistic2_thresholded.by_genes.gz")))


@functools.cache
def tcga_mut():
    return _cached("tcga_mut", lambda: _primary(_xena_matrix("mc3_nonsilentGene.xena.gz")))


def tertiles_by_group(x: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """0/1/2 = low/middle/high tertile of each gene within each group (ISLE's mRNAq2/scnaq2); NaN kept."""
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype="float32")
    for g, idx in groups.groupby(groups).groups.items():
        sub = x.loc[idx]
        q1 = sub.quantile(1 / 3)
        q2 = sub.quantile(2 / 3)
        v = (sub > q1).astype("float32") + (sub > q2).astype("float32")
        out.loc[idx] = v.where(sub.notna())
    return out


if __name__ == "__main__":
    for f in [crispr, ccle_expr, ccle_cn, ccle_mut, rnai, tcga_clin, tcga_expr, tcga_cna, tcga_mut]:
        d = f()
        print(f.__name__, d.shape, float(np.isnan(d.select_dtypes("number").to_numpy()).mean()) if d.shape[1] else None,
              file=sys.stderr)
