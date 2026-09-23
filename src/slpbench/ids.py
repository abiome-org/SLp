"""Map source gene identifiers to one canonical ID per species.

Canonical IDs:
  human      HGNC approved symbol (resolved via HGNC current/previous/alias symbols, Entrez, Ensembl)
  scer       SGD systematic ORF name, e.g. YAL001C
  spom       PomBase systematic ID, e.g. SPAC1782.04
  dmel       FlyBase gene ID, e.g. FBgn0002441
"""

from __future__ import annotations

import functools
from pathlib import Path

import polars as pl

RAW = Path("data/raw")


class Resolver:
    """symbol-ish string -> canonical ID, with unambiguous-only fallbacks."""

    def __init__(self, primary: dict[str, str], fallbacks: list[dict[str, set[str]]]):
        self.primary = primary
        self.fallbacks = fallbacks

    def __call__(self, name: str | None) -> str | None:
        if name is None:
            return None
        key = str(name).strip()
        for k in (key, key.upper()):
            if k in self.primary:
                return self.primary[k]
        for fb in self.fallbacks:
            for k in (key, key.upper()):
                hits = fb.get(k)
                if hits and len(hits) == 1:
                    return next(iter(hits))
        return None


def _multi(pairs) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for k, v in pairs:
        if k:
            out.setdefault(k, set()).add(v)
    return out


@functools.cache
def human() -> Resolver:
    df = pl.read_csv(RAW / "ids/hgnc_complete_set.txt", separator="\t", infer_schema_length=0, quote_char=None)
    df = df.filter(pl.col("status") == "Approved")
    primary = {s: s for s in df["symbol"]}
    primary |= {s.upper(): s for s in df["symbol"] if s.upper() not in primary}
    for col in ("entrez_id", "ensembl_gene_id", "hgnc_id"):
        for k, s in zip(df[col], df["symbol"]):
            if k:
                primary[k] = s

    def split(col):
        for cell, s in zip(df[col], df["symbol"]):
            for a in (cell or "").strip('"').split("|"):
                if a:
                    yield a.upper(), s

    return Resolver(primary, [_multi(split("prev_symbol")), _multi(split("alias_symbol"))])


@functools.cache
def scer() -> Resolver:
    cols = ["sgdid", "type", "qual", "orf", "std", "alias", "parent", "sec", "chr", "start", "stop",
            "strand", "gpos", "cver", "sver", "desc"]
    df = pl.read_csv(RAW / "ids/SGD_features.tab", separator="\t", has_header=False, new_columns=cols,
                     infer_schema_length=0, quote_char=None)
    df = df.filter(pl.col("type").is_in(["ORF", "tRNA gene", "ncRNA gene", "snoRNA gene", "snRNA gene",
                                         "transposable_element_gene", "pseudogene", "blocked_reading_frame"]))
    primary = {o: o for o in df["orf"] if o}
    primary |= {s.upper(): o for s, o in zip(df["std"], df["orf"]) if s}
    primary |= {g: o for g, o in zip(df["sgdid"], df["orf"]) if g}
    aliases = _multi((a.upper(), o) for cell, o in zip(df["alias"], df["orf"]) for a in (cell or "").split("|") if a)
    return Resolver(primary, [aliases])


@functools.cache
def spom() -> Resolver:
    cols = ["sysid", "sysid_prefixed", "name", "chr", "product", "uniprot", "type", "synonyms"]
    df = pl.read_csv(RAW / "ids/pombase_gene_IDs_names_products.tsv", separator="\t", has_header=False,
                     new_columns=cols, infer_schema_length=0, quote_char=None, comment_prefix="#")
    primary = {s: s for s in df["sysid"]}
    primary |= {s.upper(): s for s in df["sysid"]}
    primary |= {n.upper(): s for n, s in zip(df["name"], df["sysid"]) if n}
    syn = _multi((a.strip().upper(), s) for cell, s in zip(df["synonyms"], df["sysid"])
                 for a in (cell or "").split(",") if a.strip())
    return Resolver(primary, [syn])


@functools.cache
def dmel() -> Resolver:
    """FlyBase current symbols and symbol synonyms (incl. CG annotation IDs) -> FBgn."""
    import gzip

    primary: dict[str, str] = {}
    syn_pairs = []
    with gzip.open(RAW / "ids/fb_synonym.tsv.gz", "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 3 or p[1] != "Dmel" or not p[0].startswith("FBgn"):
                continue
            fb, sym = p[0], p[2]
            primary[fb] = fb
            primary.setdefault(sym, fb)
            primary.setdefault(sym.upper(), fb)
            if len(p) > 5:
                syn_pairs += [(a.upper(), fb) for a in p[5].split("|") if a]
    return Resolver(primary, [_multi(syn_pairs)])


RESOLVERS = {"human": human, "scer": scer, "spom": spom, "dmel": dmel}


def resolve(species: str, names: pl.Series) -> pl.Series:
    r = RESOLVERS[species]()
    uniq = names.unique().to_list()
    m = {n: r(n) for n in uniq}
    return names.replace_strict(m, return_dtype=pl.String, default=None)
