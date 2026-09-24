"""Shared per-species feature bundle for the SLB model battery (owner: models-mechanistic).

    uv run python scripts/models/_common/bundle.py [--species human scer ...] [--parts go ppi fitness esm2]

Writes data/interim/bundle/<species>/:
  go.parquet       gene, term, aspect (P/F/C), evidence   direct GO annotations, canonical SLB gene IDs.
                   Dropped: evidence IGI (inferred from genetic interaction: GI labels), ND, NOT-qualified.
  ppi.parquet      gene_a < gene_b, biogrid_phys (n BioGRID physical evidence rows), string_neighborhood,
                   string_fusion, string_cooccurence, string_coexpression, string_database (0-1000).
                   Dropped: BioGRID "genetic" rows; STRING experimental (imports genetic assays from BioGRID;
                   on scer 20% of its edges are BioGRID GIs vs 2.7% of BioGRID physical edges), textmining
                   (paper co-mentions of SL pairs) and combined_score.
  fitness.parquet  gene, effect, source   reference single-loss effect (negative = sicker): slpbench.fitness
                   for the SLB-1.2 species, else the first `<species>_single()` found in slpbench.sources.*
  esm2.parquet     gene, e0..e1279  ESM-2 650M (esm2_t33_650M_UR50D, layer 33) mean residue embedding of one
                   protein per gene (first 1022 aa): a symlink to data/interim/esm2_650m/<species>.parquet,
                   computed by models-features-fm (human: UniProt canonical; others: orthology_extra/fasta).
Shared across species: data/interim/bundle/_go/{terms,edges}.parquet (go-basic: term, namespace, name;
child -> parent over is_a + part_of).  data/interim/bundle/manifest.json: inputs (+sha256) and counts.

Gene IDs are canonical SLB IDs via slpbench.ids_extra.resolver(species) (ids.py for the base species).
A species is added by adding one entry to SPECIES (GAF files, STRING taxon, BioGRID organism file).
Inputs live in data/raw/{go,string_v12,biogrid} (sha256 in data/raw/MANIFEST.tsv).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import json
import time
from collections import Counter
from pathlib import Path

import polars as pl

from slpbench import ids_extra

RAW = Path("data/raw")
OUT = Path("data/interim/bundle")
DROP_EVIDENCE = {"IGI", "ND"}
CH = ["neighborhood", "fusion", "cooccurence", "coexpression", "database"]
ALL_CH = ["neighborhood", "fusion", "cooccurence", "coexpression", "experimental", "database"]

SPECIES = {
    "human": dict(gaf=["go/goa_human.gaf.gz"], string="9606", biogrid="Homo_sapiens"),
    "scer": dict(gaf=["go/sgd.gaf.gz"], string="4932", biogrid="Saccharomyces_cerevisiae_S288c"),
    "spom": dict(gaf=["go/pombase.gaf.gz"], string="284812", biogrid="Schizosaccharomyces_pombe_972h"),
    # GOA proteome of D39 (NCTC 7466); STRING has R6 (spr tags) not D39/D39V: mapped through the resolver
    "spne": dict(gaf=["go/goa_spne_d39.goa"], string="171101", biogrid="Streptococcus_pneumoniae_ATCCBAA255"),
    "cele": dict(gaf=["go/wb.gaf.gz"], string="6239", biogrid="Caenorhabditis_elegans"),
    "dmel": dict(gaf=["go/fb.gaf.gz"], string="7227", biogrid="Drosophila_melanogaster"),
    "mmus": dict(gaf=["go/mgi.gaf.gz"], string="10090", biogrid="Mus_musculus"),
    "ecol": dict(gaf=["go/goa_ecol_mg1655.goa"], string="511145", biogrid="Escherichia_coli_K12_MG1655"),
    "bsub": dict(gaf=["go/goa_bsub_168.goa"], string="224308", biogrid="Bacillus_subtilis_168"),
}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _open(p: Path):
    return gzip.open(p, "rt") if p.suffix == ".gz" else open(p)


class Res:
    """Canonical-ID resolver: ids_extra.resolver, plus (spne) bacteria_extra aliases for D39 SPD_/R6 spr tags."""

    def __init__(self, sp: str):
        self.r = ids_extra.resolver(sp)
        self.extra = {}
        if sp == "spne":
            from slpbench.sources.bacteria_extra import spne_resolver
            self.extra = spne_resolver()
        self.cache: dict[str, str | None] = {}

    def __call__(self, x: str | None) -> str | None:
        if not x:
            return None
        if x not in self.cache:
            g = self.r(x)
            if g is None and x in self.extra:
                g = self.r(self.extra[x]) or self.extra[x]
            self.cache[x] = g
        return self.cache[x]

    def first(self, cands) -> str | None:
        for c in cands:
            g = self(c)
            if g:
                return g
        return None


# ------------------------------------------------------------------ GO
def go_shared(inputs: dict) -> None:
    obo = RAW / "go/go-basic.obo"
    inputs[str(obo)] = sha(obo)
    terms, edges, cur = [], [], None
    for line in open(obo):
        line = line.rstrip("\n")
        if line == "[Term]":
            cur = {"term": None, "namespace": None, "name": None, "obsolete": False, "parents": []}
            terms.append(cur)
        elif line.startswith("["):
            cur = None
        elif cur is not None and ": " in line:
            k, v = line.split(": ", 1)
            if k == "id":
                cur["term"] = v
            elif k in ("namespace", "name"):
                cur[k] = v
            elif k == "is_obsolete" and v == "true":
                cur["obsolete"] = True
            elif k == "is_a":
                cur["parents"].append(v.split(" ! ")[0])
            elif k == "relationship" and v.startswith("part_of "):
                cur["parents"].append(v.split()[1])
    terms = [t for t in terms if not t["obsolete"]]
    (OUT / "_go").mkdir(parents=True, exist_ok=True)
    pl.DataFrame([{k: t[k] for k in ("term", "namespace", "name")} for t in terms]).write_parquet(OUT / "_go/terms.parquet")
    pl.DataFrame([(t["term"], p) for t in terms for p in t["parents"]], schema=["child", "parent"],
                 orient="row").write_parquet(OUT / "_go/edges.parquet")


def go(sp: str, res: Res, inputs: dict) -> pl.DataFrame:
    valid = set(pl.read_parquet(OUT / "_go/terms.parquet")["term"])
    rows = []
    for f in SPECIES[sp]["gaf"]:
        p = RAW / f
        inputs[str(p)] = sha(p)
        with _open(p) as fh:
            for line in fh:
                if line.startswith("!"):
                    continue
                c = line.rstrip("\n").split("\t")
                if len(c) < 15 or "NOT" in c[3] or c[6] in DROP_EVIDENCE or c[4] not in valid:
                    continue
                db_id = c[1] if c[0] != "MGI" else c[1]
                g = res.first([db_id, f"{c[0]}:{c[1]}", c[2], *c[10].split("|")])
                if g:
                    rows.append((g, c[4], c[8], c[6]))
    return pl.DataFrame(rows, schema=["gene", "term", "aspect", "evidence"], orient="row").unique()


# ------------------------------------------------------------------ PPI
def string_map(sp: str, res: Res, inputs: dict) -> dict[str, str]:
    """STRING protein -> canonical gene: majority vote over preferred name, the ID suffix and all aliases."""
    t = SPECIES[sp]["string"]
    votes: dict[str, Counter] = {}
    info = RAW / f"string_v12/{t}.protein.info.v12.0.txt.gz"
    ali = RAW / f"string_v12/{t}.protein.aliases.v12.0.txt.gz"
    for p in (info, ali):
        inputs[str(p)] = sha(p)
    with gzip.open(info, "rt") as fh:
        next(fh)
        for line in fh:
            c = line.split("\t")
            for x in (c[0].split(".", 1)[1], c[1]):
                g = res(x)
                if g:
                    votes.setdefault(c[0], Counter())[g] += 3
    with gzip.open(ali, "rt") as fh:
        next(fh)
        for line in fh:
            c = line.rstrip("\n").split("\t")
            g = res(c[1])
            if g:
                votes.setdefault(c[0], Counter())[g] += 1
    return {k: v.most_common(1)[0][0] for k, v in votes.items()}


def ppi(sp: str, res: Res, inputs: dict) -> pl.DataFrame:
    parts = []
    bg = RAW / f"biogrid/BIOGRID-ORGANISM-{SPECIES[sp]['biogrid']}-5.0.261.tab3.txt"
    if bg.exists():
        inputs[str(bg)] = sha(bg)
        d = pl.read_csv(bg, separator="\t", infer_schema_length=0, quote_char=None,
                        columns=["Systematic Name Interactor A", "Systematic Name Interactor B",
                                 "Official Symbol Interactor A", "Official Symbol Interactor B",
                                 "Entrez Gene Interactor A", "Entrez Gene Interactor B", "Experimental System Type"])
        d = d.filter(pl.col("Experimental System Type") == "physical")
        rows = []
        for sa, sb, oa, ob, ea, eb, _ in d.iter_rows():
            a, b = res.first([sa, oa, ea]), res.first([sb, ob, eb])
            if a and b and a != b:
                rows.append((min(a, b), max(a, b)))
        if rows:
            parts.append(pl.DataFrame(rows, schema=["gene_a", "gene_b"], orient="row")
                         .group_by("gene_a", "gene_b").agg(pl.len().cast(pl.Int32).alias("biogrid_phys")))
    t = SPECIES[sp]["string"]
    st = RAW / f"string_v12/{t}.protein.links.detailed.v12.0.txt.gz"
    if st.exists():
        inputs[str(st)] = sha(st)
        m = string_map(sp, res, inputs)
        best: dict[tuple, list] = {}
        with gzip.open(st, "rt") as fh:
            next(fh)
            for line in fh:
                c = line.split()
                a, b = m.get(c[0]), m.get(c[1])
                if not a or not b or a >= b:
                    continue
                v = dict(zip(ALL_CH, map(int, c[2:8])))
                vals = [v[k] for k in CH]
                if not any(vals):
                    continue
                k = (a, b)
                best[k] = [max(x, y) for x, y in zip(best[k], vals)] if k in best else vals
        parts.append(pl.DataFrame([(a, b, *v) for (a, b), v in best.items()],
                                  schema=["gene_a", "gene_b"] + [f"string_{c}" for c in CH], orient="row")
                     .with_columns(pl.col(f"string_{c}").cast(pl.Int16) for c in CH))
    if not parts:
        return pl.DataFrame(schema={"gene_a": pl.String, "gene_b": pl.String})
    out = parts[0]
    for p in parts[1:]:
        out = out.join(p, on=["gene_a", "gene_b"], how="full", coalesce=True)
    for c in ["biogrid_phys"] + [f"string_{c}" for c in CH]:
        if c not in out.columns:
            out = out.with_columns(pl.lit(0).alias(c))
    return out.fill_null(0).sort("gene_a", "gene_b")


# ------------------------------------------------------------------ fitness
def fitness(sp: str, res: Res) -> pl.DataFrame:
    from slpbench import fitness as F

    ge = F.gene_effects()
    d = ge.filter(pl.col("species") == sp)
    if d.height:
        return d.select("gene", pl.col("effect").cast(pl.Float64), pl.lit(f"slpbench.fitness.gene_effects").alias("source"))
    for mod in ("slpbench.fitness", "slpbench.sources.eukaryotes_extra", "slpbench.sources.bacteria_extra",
                "slpbench.sources.eukaryotes", "slpbench.sources.bacteria"):
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        fn = getattr(m, f"{sp}_single", None)
        if fn is None:
            continue
        try:
            d = fn()
        except Exception as e:  # missing raw data etc.
            print(f"  {mod}.{sp}_single failed: {e}")
            continue
        d = d.with_columns(pl.col("gene").map_elements(lambda x: res(x) or x, return_dtype=pl.String))
        return d.group_by("gene").agg(pl.col("effect").cast(pl.Float64).median()).with_columns(
            pl.lit(f"{mod}.{sp}_single").alias("source"))
    return pl.DataFrame(schema={"gene": pl.String, "effect": pl.Float64, "source": pl.String})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", nargs="*", default=list(SPECIES))
    ap.add_argument("--parts", nargs="*", default=["go", "ppi", "fitness", "esm2"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    mpath = OUT / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"species": {}, "inputs": {}}
    if "go" in a.parts:
        go_shared(manifest["inputs"])
    for sp in a.species:
        t0 = time.time()
        res = Res(sp)
        d = OUT / sp
        d.mkdir(parents=True, exist_ok=True)
        info = manifest["species"].setdefault(sp, {})
        if "go" in a.parts:
            g = go(sp, res, manifest["inputs"])
            g.write_parquet(d / "go.parquet")
            info["go"] = {"genes": g["gene"].n_unique(), "rows": g.height}
        if "fitness" in a.parts:
            f = fitness(sp, res)
            f.write_parquet(d / "fitness.parquet")
            info["fitness"] = {"genes": f.height, "source": f["source"][0] if f.height else None}
        if "esm2" in a.parts:
            src = Path("data/interim/esm2_650m") / f"{sp}.parquet"
            dst = d / "esm2.parquet"
            if src.exists():
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(Path("../../esm2_650m") / f"{sp}.parquet")
                info["esm2"] = {"genes": pl.scan_parquet(src).select(pl.len()).collect().item(),
                                "source": str(src), "sha256": sha(src)}
            else:
                print(f"  {sp}: no {src} yet")
        if "ppi" in a.parts:
            p = ppi(sp, res, manifest["inputs"])
            p.write_parquet(d / "ppi.parquet")
            info["ppi"] = {"edges": p.height, "biogrid_phys_edges": int((p["biogrid_phys"] > 0).sum()) if p.height else 0}
        print(sp, json.dumps(info), f"{time.time() - t0:.0f}s", flush=True)
        mpath.write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
