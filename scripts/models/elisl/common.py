"""Shared helpers for the ELISL adapter (runs inside external/models/elisl/.venv, python 3.8).

ELISL (Tepeli et al., Bioinformatics 2024) feature families, rebuilt for SLB gene pairs:
  seq      |SeqVec(g1) - SeqVec(g2)|                  (1024-d, context-free)
  ppi      |node2vec(g1) - node2vec(g2)|              (64-d, context-free)
  crispr_dependency_mut / crispr_dependency_expr      (4-d each, cell lines of the context's cancer type)
  tissue   TCGA/GTEx hand-crafted features            (13-d, context's TCGA cancer type)
Only single-gene data are used; labels come exclusively from SLB train.parquet.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(os.environ.get("SLB_BENCH", "data/bench/slb1.2"))
if not BENCH.is_absolute():
    BENCH = ROOT / BENCH
RAW = ROOT / "data/raw"
WORK = ROOT / "external/models/elisl/_slb"
CACHE = WORK / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
SPECIES = "human"  # ELISL's cell-line / TCGA / GTEx features exist only for human


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load(split: str) -> pd.DataFrame:
    """train/dev with labels, test* inputs only. Never reads hidden/."""
    if split.startswith("test"):
        return pd.read_parquet(BENCH / f"{split}_inputs.parquet")
    return pd.read_parquet(BENCH / f"{split}.parquet")


def human(split: str) -> pd.DataFrame:
    d = load(split)
    return d[d.species == SPECIES].reset_index(drop=True)


def contexts() -> pd.DataFrame:
    return pd.read_parquet(BENCH / "contexts.parquet")


# ---------------------------------------------------------------- cancer-type mapping
# DepMap OncotreeLineage (+ OncotreeCode) -> TCGA study code. ELISL used 8 TCGA types (BRCA, CESC, COAD, KIRC,
# LAML, LUAD, OV, SKCM); SLB contexts need a few more (LUSC, READ, PAAD, HNSC, STAD). Leukaemias (myeloid and
# lymphoid) map to LAML exactly as ELISL's 'Leukemia' cell-line group did.
LINEAGE2TCGA = {
    "Breast": "BRCA", "Cervix": "CESC", "Bowel": "COAD", "Kidney": "KIRC", "Myeloid": "LAML", "Lymphoid": "LAML",
    "Lung": "LUAD", "Ovary/Fallopian Tube": "OV", "Skin": "SKCM", "Pancreas": "PAAD", "Head and Neck": "HNSC",
    "Esophagus/Stomach": "STAD", "CNS/Brain": "GBM", "Liver": "LIHC", "Uterus": "UCEC", "Bladder/Urinary Tract": "BLCA",
    "Prostate": "PRAD", "Thyroid": "THCA", "Soft Tissue": "SARC", "Bone": "SARC", "Pleura": "MESO", "Eye": "UVM",
    "Biliary Tract": "CHOL", "Testis": "TGCT", "Adrenal Gland": "ACC", "Peripheral Nervous System": "PCPG",
}
CODE2TCGA = {"LUSC": "LUSC", "READ": "READ", "ESCA": "ESCA", "ESCC": "ESCA", "EGC": "ESCA", "LGG": "LGG",
             "ODG": "LGG", "ASTR": "LGG", "DLBCLNOS": "DLBC", "UCS": "UCS", "CCRCC": "KIRC", "PRCC": "KIRP",
             "CHRCC": "KICH", "HCC": "LIHC"}
DISEASE_KEYWORDS = [  # fallback for contexts without a DepMap model (free-text disease)
    ("melanoma", "SKCM"), ("lung squamous", "LUSC"), ("lung", "LUAD"), ("rectal", "READ"), ("colon", "COAD"),
    ("colorectal", "COAD"), ("cecum", "COAD"), ("pancrea", "PAAD"), ("breast", "BRCA"), ("ovar", "OV"),
    ("cervi", "CESC"), ("leukemia", "LAML"), ("gastric", "STAD"), ("stomach", "STAD"), ("squamous cell carcinoma", "HNSC"),
    ("kidney", "KIRC"), ("renal", "KIRC"), ("glioblastoma", "GBM"), ("liver", "LIHC"), ("hepato", "LIHC"),
]
PANCAN = "PANCAN"  # non-cancer / unmapped contexts (e.g. hTERT-RPE1): pan-cancer tissue + all cell lines


def depmap_models() -> pd.DataFrame:
    m = pd.read_csv(RAW / "depmap/Model_24Q4.csv", usecols=["ModelID", "OncotreeLineage", "OncotreeCode"])
    return m.set_index("ModelID")


def model_tcga(lineage, code) -> str | None:
    if isinstance(code, str) and code in CODE2TCGA:
        return CODE2TCGA[code]
    if isinstance(lineage, str) and lineage in LINEAGE2TCGA:
        return LINEAGE2TCGA[lineage]
    return None


def depmap_tcga() -> pd.Series:
    """DepMap ModelID -> TCGA code (used to define ELISL's per-cancer cell-line groups)."""
    m = depmap_models()
    return pd.Series([model_tcga(l, c) for l, c in zip(m.OncotreeLineage, m.OncotreeCode)], index=m.index)


def context_tcga() -> pd.Series:
    """SLB human context_id -> TCGA code (or PANCAN)."""
    c = contexts()
    c = c[c.species == SPECIES]
    dm = depmap_tcga()
    out = {}
    for cid, did, dis in zip(c.context_id, c.depmap_id, c.disease):
        t = dm.get(did) if isinstance(did, str) else None
        if t is None and isinstance(dis, str):
            for kw, code in DISEASE_KEYWORDS:
                if kw in dis.lower():
                    t = code
                    break
        out[cid] = t or PANCAN
    return pd.Series(out, name="tcga")


def pairs_with_type(df: pd.DataFrame) -> pd.DataFrame:
    ct = context_tcga()
    d = df[["example_id", "context_id", "gene_a", "gene_b"]].copy()
    d["cancer"] = d.context_id.map(ct).fillna(PANCAN)
    return d


def genes_of(*splits) -> list:
    g = set()
    for s in splits:
        d = human(s)
        g |= set(d.gene_a) | set(d.gene_b)
    return sorted(g)


# ---------------------------------------------------------------- symbol resolution
_RES = None


def resolver():
    """any HGNC symbol / previous / alias / Entrez / Ensembl -> current HGNC symbol"""
    global _RES
    if _RES is None:
        h = pd.read_csv(RAW / "ids/hgnc_complete_set.txt", sep="\t", low_memory=False,
                        usecols=["symbol", "ensembl_gene_id", "entrez_id", "prev_symbol", "alias_symbol"])
        m = {}
        for col in ["alias_symbol", "prev_symbol"]:
            for s, v in zip(h.symbol, h[col]):
                if isinstance(v, str):
                    for x in v.split("|"):
                        m.setdefault(x, s)
        for s, e, z in zip(h.symbol, h.ensembl_gene_id, h.entrez_id):
            m[s] = s
            if isinstance(e, str):
                m[e] = s
            if z == z:
                m[str(int(z))] = s
        _RES = m
    return _RES


def resolve_cols(cols) -> list:
    """DepMap 'SYM (entrez)' or plain symbol / entrez -> HGNC symbol (None if unknown)."""
    m = resolver()
    out = []
    for c in cols:
        c = str(c)
        mm = re.match(r"^(.*) \((\d+)\)$", c)
        if mm:
            out.append(m.get(mm.group(2)) or m.get(mm.group(1)))
        else:
            out.append(m.get(c))
    return out


def dedupe_cols(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c is not None for c in df.columns]
    df = df.loc[:, keep]
    return df.loc[:, ~pd.Index(df.columns).duplicated()]


def cached(name, build):
    p = CACHE / f"{name}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    df = build()
    df.columns = [str(c) for c in df.columns]
    df.to_parquet(p)
    return df
