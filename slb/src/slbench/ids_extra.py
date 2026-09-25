"""Canonical gene IDs for the additional species (companion to ids.py).

Canonical IDs (node names in the family graph are "species:ID"):
  mmus   MGI approved marker symbol, e.g. Trp53        (MRK_List2.rpt; MGI IDs, Ensembl IDs, synonyms)
  cele   WormBase gene ID, e.g. WBGene00000001         (WS298 geneIDs: CGC name, sequence name, other IDs)
  calb   CGD Assembly-22 ORF without allele suffix, e.g. C1_00060W  (CGD features: gene name, orf19, aliases)
  ecol   E. coli K-12 MG1655 b-number, e.g. b0002      (GenBank U00096.3: gene, ECK/synonyms)
  bsub   B. subtilis 168 BSU locus tag without underscore, e.g. BSU00010 (the data-bacteria scheme;
         GenBank AL009126.3 old_locus_tag; the new-style BSU_00010 tag and gene names resolve to it)
  mtub   M. tuberculosis H37Rv Rv number, e.g. Rv0001  (GenBank AL123456.3: gene)
  saur   S. aureus NCTC 8325 locus tag, e.g. SAOUHSC_00001  (GenBank CP000253.1: gene)
  spne   S. pneumoniae D39V: gene name when annotated, else SPV_ locus tag (the scheme of
         dualcrispri2025 mmc4.csv / sources.bacteria), e.g. dnaA, SPV_0039  (GenBank CP027540.1)

The bacterial tables data/raw/ids/<sp>_genes.tsv are derived from the GenBank records by
scripts/orthology_extra/build_edges.py (columns canonical, locus_tag, old_locus_tag, gene,
synonyms ('|'-separated), protein_id). Resolution: exact/upper-case primary keys first, then
synonyms only when unambiguous (same policy as ids.Resolver).
"""

from __future__ import annotations

import functools
import gzip
from pathlib import Path

import polars as pl

from slbench.ids import Resolver, _multi

RAW = Path("data/raw")
GENBANK = RAW / "orthology_extra/proteomes"
BACTERIA = {  # species -> (GenBank file, canonical scheme)
    "spne": ("spne_D39V_CP027540.1.gb", "gene_or_locus"),
    "ecol": ("ecol_MG1655_U00096.3.gb", "locus"),
    "bsub": ("bsub_168_AL009126.3.gb", "old_locus"),
    "mtub": ("mtub_H37Rv_AL123456.3.gb", "locus"),
    "saur": ("saur_NCTC8325_CP000253.1.gb", "locus"),
}


def _primary(pairs) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in pairs:
        if k:
            out.setdefault(k, v)
            out.setdefault(k.upper(), v)
    return out


@functools.cache
def mmus() -> Resolver:
    df = pl.read_csv(RAW / "ids/MRK_List2.rpt", separator="\t", infer_schema_length=0, quote_char=None)
    df = df.filter((pl.col("Status") == "O") & pl.col("Marker Type").is_in(["Gene", "Pseudogene"]))
    sym, mgi = df["Marker Symbol"].to_list(), df["MGI Accession ID"].to_list()
    primary = _primary(zip(sym, sym)) | _primary(zip(mgi, sym))
    ens = RAW / "orthology_extra/ensembl_mmusculus_paralogs.tsv"
    if ens.exists():
        e = pl.read_csv(ens, separator="\t", infer_schema_length=0, quote_char=None, columns=[0, 1]).unique()
        known = set(sym)
        primary |= {g: s for g, s in zip(e[:, 0], e[:, 1]) if s in known and g not in primary}
    syn = _multi((a.upper(), s) for cell, s in zip(df["Marker Synonyms (pipe-separated)"], sym)
                 for a in (cell or "").split("|") if a)
    return Resolver(primary, [syn])


@functools.cache
def cele() -> Resolver:
    primary: dict[str, str] = {}
    with gzip.open(RAW / "ids/c_elegans.WS298.geneIDs.txt.gz", "rt") as f:
        rows = [line.rstrip("\n").split(",") for line in f if not line.startswith("#")]
    live = [r for r in rows if len(r) >= 5 and r[4] == "Live"]
    for r in live:
        primary[r[1]] = r[1]
    for r in live:  # CGC names first, then sequence names
        if r[2]:
            primary.setdefault(r[2], r[1])
            primary.setdefault(r[2].upper(), r[1])
    for r in live:
        if r[3]:
            primary.setdefault(r[3], r[1])
            primary.setdefault(r[3].upper(), r[1])
    other = []
    with gzip.open(RAW / "ids/c_elegans.WS298.geneOtherIDs.txt.gz", "rt") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) > 2 and p[1] == "Live":
                other += [(a.upper(), p[0]) for a in p[2:] if a]
                other += [(a.upper().removeprefix("CELE_"), p[0]) for a in p[2:] if a.startswith("CELE_")]
    return Resolver(primary, [_multi(other)])


def calb_canonical(feature: str) -> str:
    """C1_00060W_A -> C1_00060W (haplotype A/B alleles collapse to one gene)."""
    return feature[:-2] if feature.endswith(("_A", "_B")) else feature


@functools.cache
def calb() -> Resolver:
    cols = ["feature", "gene", "aliases", "type"]
    df = pl.read_csv(RAW / "ids/C_albicans_SC5314_A22_chromosomal_feature.tab", separator="\t", has_header=False,
                     comment_prefix="!", infer_schema_length=0, quote_char=None, columns=[0, 1, 2, 3], new_columns=cols)
    df = df.filter(pl.col("feature").str.ends_with("_A") | ~pl.col("feature").str.contains(r"_[AB]$"))
    canon = [calb_canonical(f) for f in df["feature"]]
    primary = _primary(zip(canon, canon)) | _primary(zip(df["feature"], canon))
    primary |= _primary((f[:-2] + "_B", c) for f, c in zip(df["feature"], canon) if f.endswith("_A"))
    primary |= {k: v for k, v in _primary(zip(df["gene"], canon)).items() if k not in primary}
    ali = _multi((a.upper(), c) for cell, c in zip(df["aliases"], canon) for a in (cell or "").split("|") if a)
    return Resolver(primary, [ali])


def parse_genbank(path: Path) -> list[dict]:
    """Minimal GenBank CDS/gene parser: locus_tag, old_locus_tag, gene, gene_synonym, protein_id, translation."""
    feats, cur, key, in_feats = [], None, None, False
    with open(path) as f:
        for line in f:
            if line.startswith("FEATURES"):
                in_feats = True
                continue
            if line.startswith("ORIGIN"):
                in_feats = False
            if not in_feats:
                continue
            if len(line) > 5 and line[5] != " ":
                kind = line[5:21].strip()
                cur = {"kind": kind, "q": {}} if kind in ("CDS", "gene", "tRNA", "rRNA", "ncRNA", "tmRNA",
                                                          "misc_RNA") else None
                if cur is not None:
                    feats.append(cur)
                key = None
                continue
            if cur is None:
                continue
            s = line.strip()
            if s.startswith("/"):
                k, _, v = s[1:].partition("=")
                key = k
                cur["q"].setdefault(k, []).append(v.strip('"'))
            elif key is not None and cur["q"].get(key):
                sep = "" if key == "translation" else " "
                cur["q"][key][-1] = (cur["q"][key][-1] + sep + s).strip('"')
    return feats


def bacterial_gene_table(sp: str) -> pl.DataFrame:
    """One row per locus tag from the GenBank record (CDS merged with its gene/RNA features)."""
    fn, scheme = BACTERIA[sp]
    rows: dict[str, dict] = {}
    for ft in parse_genbank(GENBANK / fn):
        q = ft["q"]
        lt = (q.get("locus_tag") or [None])[0]
        if not lt:
            continue
        r = rows.setdefault(lt, {"locus_tag": lt, "old_locus_tag": set(), "gene": None, "synonyms": set(),
                                 "protein_id": None, "translation": None, "pseudo": False})
        r["old_locus_tag"] |= set(q.get("old_locus_tag", []))
        if q.get("gene") and not r["gene"]:
            r["gene"] = q["gene"][0]
        for s in q.get("gene_synonym", []):
            r["synonyms"] |= {x.strip() for x in s.split(";") if x.strip()}
        if ft["kind"] == "CDS":
            if q.get("translation"):
                r["translation"] = q["translation"][0]
                r["protein_id"] = (q.get("protein_id") or [None])[0]
            if "pseudo" in q or "pseudogene" in q:
                r["pseudo"] = True
    out = []
    for r in rows.values():
        canon = r["gene"] if (scheme == "gene_or_locus" and r["gene"]) else r["locus_tag"]
        if scheme == "old_locus" and len(r["old_locus_tag"]) == 1:
            canon = next(iter(r["old_locus_tag"]))
        out.append({"canonical": canon, "locus_tag": r["locus_tag"], "old_locus_tag": "|".join(sorted(r["old_locus_tag"])),
                    "gene": r["gene"] or "", "synonyms": "|".join(sorted(r["synonyms"])),
                    "protein_id": r["protein_id"] or "", "translation": r["translation"] or ""})
    df = pl.DataFrame(out)
    if scheme == "gene_or_locus":  # duplicated gene names (rare) fall back to locus tags
        dup = df.group_by("canonical").len().filter(pl.col("len") > 1)["canonical"]
        df = df.with_columns(pl.when(pl.col("canonical").is_in(dup.implode())).then(pl.col("locus_tag"))
                             .otherwise(pl.col("canonical")).alias("canonical"))
    if sp == "spne":
        df = _spne_author_names(df)
    return df


def _spne_author_names(df: pl.DataFrame) -> pl.DataFrame:
    """Use the dualcrispri2025 authors' names (mmc4.csv) as canonical where they differ from GenBank.

    The benchmark's spne IDs are the authors' names (an older PneumoBrowse naming: e.g. SPV_0112
    is 'rtgR' in GenBank CP027540.1, aqpZ is 'Pn-aqpA'). Author names are matched to locus tags
    through the GenBank locus tag, gene name and synonyms (unambiguous matches only).
    """
    src = RAW / "dualcrispri2025_spneumo/mmc4.csv"
    if not src.exists():
        return df
    m = pl.read_csv(src, columns=["SG1.targets", "SG2.targets"], infer_schema_length=0)
    names = {t.strip() for col in m.columns for cell in m[col].unique().drop_nulls() for t in cell.split(",") if t.strip()}
    by_key: dict[str, set[str]] = {}
    for lt, g, syn in zip(df["locus_tag"], df["gene"], df["synonyms"]):
        for k in [lt, g, *syn.split("|")]:
            if k:
                by_key.setdefault(k.upper(), set()).add(lt)
    override = {}
    for n in names:
        hits = by_key.get(n.upper(), set())
        if len(hits) == 1:
            override[next(iter(hits))] = n
    return df.with_columns(pl.col("locus_tag").replace_strict(override, default=pl.col("canonical")).alias("canonical"))


def write_bacterial_tables() -> None:
    for sp in BACTERIA:
        bacterial_gene_table(sp).drop("translation").write_csv(RAW / f"ids/{sp}_genes.tsv", separator="\t")


def _bacterial(sp: str) -> Resolver:
    path = RAW / f"ids/{sp}_genes.tsv"
    if not path.exists():
        write_bacterial_tables()
    df = pl.read_csv(path, separator="\t", infer_schema_length=0, empty_string_is_null=False)
    c = df["canonical"].to_list()
    primary = _primary(zip(c, c)) | _primary(zip(df["locus_tag"], c))
    primary |= {k: v for k, v in _primary(zip(df["protein_id"], c)).items() if k not in primary}
    primary |= {k: v for k, v in _primary((o, x) for cell, x in zip(df["old_locus_tag"], c)
                                          for o in cell.split("|") if o).items() if k not in primary}
    # locus-tag spelling variants: BSU_00010 / BSU00010, SPV_0001 / SPV0001
    primary |= {k: v for k, v in _primary((lt.replace("_", ""), x) for lt, x in zip(df["locus_tag"], c)).items()
                if k not in primary}
    genes = _multi((g.upper(), x) for g, x in zip(df["gene"], c) if g)
    syn = _multi((s.upper(), x) for cell, x in zip(df["synonyms"], c) for s in cell.split("|") if s)
    return Resolver(primary, [genes, syn])


@functools.cache
def spne() -> Resolver:
    return _bacterial("spne")


@functools.cache
def ecol() -> Resolver:
    return _bacterial("ecol")


@functools.cache
def bsub() -> Resolver:
    return _bacterial("bsub")


@functools.cache
def mtub() -> Resolver:
    return _bacterial("mtub")


@functools.cache
def saur() -> Resolver:
    return _bacterial("saur")


RESOLVERS = {"mmus": mmus, "cele": cele, "calb": calb, "spne": spne, "ecol": ecol, "bsub": bsub, "mtub": mtub,
             "saur": saur}


def resolver(species: str) -> Resolver:
    """Resolver for any SLB species (base species from ids.py, new ones from here)."""
    from slbench import ids

    if species in RESOLVERS:
        return RESOLVERS[species]()
    return ids.RESOLVERS[species]()


def resolve(species: str, names: pl.Series) -> pl.Series:
    r = resolver(species)
    m = {n: r(n) for n in names.unique().to_list()}
    return names.replace_strict(m, return_dtype=pl.String, default=None)
