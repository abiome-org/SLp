"""Genome-scale metabolic models (GEMs) per SLB species, and the map GEM gene -> SLB gene ID.

Each entry returns (cobra.Model with the chosen medium, {model_gene_id: slb_gene_id}).
Add a species by adding a loader to LOADERS. All paths are under external/models/fba (gitignored).
"""

from __future__ import annotations

import csv
import gzip
import logging
import re
from pathlib import Path

import cobra

ROOT = Path(__file__).resolve().parents[3]
EXT = ROOT / "external/models/fba"
GEMS = EXT / "gems"
ANNOT = EXT / "annot"
RAW = ROOT / "data/raw"

logging.getLogger("cobra").setLevel(logging.ERROR)


def _read(path: Path) -> cobra.Model:
    m = cobra.io.read_sbml_model(str(path))
    m.solver = "glpk"
    return m


# ---------------------------------------------------------------- S. cerevisiae: yeast-GEM (Yeast9)
def scer(medium: str = "default"):
    m = _read(GEMS / "yeast-GEM/model/yeast-GEM.xml")
    if medium == "rich":
        _open_rich(m, AA_NUC_NAMES)
    # yeast-GEM genes are SGD systematic ORFs already
    return m, {g.id: g.id for g in m.genes}


# ---------------------------------------------------------------- S. pombe: pomGEM (Elsemman 2022)
def spom(medium: str = "default"):
    m = _read(GEMS / "pombe/models/pcPombe/jupyterNotebookReconstruction/pomGEM_updated_editedManually.xml")
    if medium == "rich":
        _open_rich(m, AA_NUC_NAMES)
    ids = set()
    with open(RAW / "ids/pombase_gene_IDs_names_products.tsv") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            ids.add(r["gene_systematic_id"])
    mp = {}
    for g in m.genes:
        cands = [g.id, g.id.replace("_", "."), re.sub(r"_(?=[^_]*$)", ".", g.id)]
        cands += [c[:-1] + c[-1].lower() for c in cands]
        hit = next((c for c in cands if c in ids), None)
        if hit:
            mp[g.id] = hit
    return m, mp


# ---------------------------------------------------------------- S. pneumoniae: iDS372 (R6) -> D39V
def spne(medium: str = "default"):
    m = _read(GEMS / "iDS372/iDS372.xml")
    # R6 (spr) -> D39V (SPV_) by diamond reciprocal best hit, >= 95% identity, >= 90% coverage
    fwd, rev = {}, {}
    for fn, d in (("R6_vs_D39V.tsv", fwd), ("D39V_vs_R6.tsv", rev)):
        with open(ANNOT / fn) as fh:
            for q, s, pid, ln, ql, sl, *_ in csv.reader(fh, delimiter="\t"):
                if float(pid) >= 95 and int(ln) >= 0.9 * min(int(ql), int(sl)) and q not in d:
                    d[q] = s
    name = {}
    with open(ANNOT / "D39V_genes.tsv") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            name[r["locus_tag"]] = r["gene"] or r["locus_tag"]
    # canonical SLB IDs: shared resolver (slbench.sources.bacteria_extra.spne_resolver, exported to TSV)
    res = {}
    rp = ANNOT / "spne_resolver.tsv"
    if rp.exists():
        with open(rp) as fh:
            res = {r["alias"]: r["canonical"] for r in csv.DictReader(fh, delimiter="\t")}
    mp = {}
    for g in m.genes:
        if g.id in res:  # R6 spr tag annotated in the D39V GenBank notes
            mp[g.id] = res[g.id]
            continue
        spv = fwd.get(g.id)
        if spv and rev.get(spv) == g.id:
            mp[g.id] = res.get(spv, name.get(spv, spv))
    return m, mp


# ---------------------------------------------------------------- human: Human-GEM (Human1 lineage)
HAMS = [  # Ham's F-12-like medium (+ O2, ions); matched on Human-GEM extracellular metabolite names
    "glucose", "arginine", "histidine", "lysine", "methionine", "phenylalanine", "tryptophan", "tyrosine",
    "alanine", "glycine", "serine", "threonine", "aspartate", "glutamate", "asparagine", "glutamine",
    "isoleucine", "leucine", "proline", "valine", "cysteine", "thiamin", "hypoxanthine", "folate",
    "biotin", "pantothenate", "choline", "inositol", "nicotinamide", "pyridoxine", "riboflavin",
    "thymidine", "aquacob(III)alamin", "lipoic acid", "linoleate", "putrescine", "pyruvate",
    "O2", "H2O", "Pi", "sulfate", "Fe2+", "Fe3+", "Na+", "K+", "Ca2+", "Cl-", "HCO3-", "H+", "CO2",
    "NH3", "zinc", "Cu2+", "Mg2+",
    # serum-derived (needed for the Human-GEM biomass pools): vitamin A/E, lipids
    "retinol", "alpha-tocopherol", "gamma-tocopherol", "linolenate", "cholesterol", "oleate", "palmitate",
    "stearate", "arachidonate", "ethanolamine",
]
INORGANIC = {"o2", "h2o", "pi", "sulfate", "fe2+", "fe3+", "na+", "k+", "ca2+", "cl-", "hco3-", "h+", "co2", "nh3",
             "zinc", "cu2+", "mg2+"}


def _hams(m: cobra.Model) -> None:
    """Glucose-limited Ham's-like medium for Human-GEM-lineage models: glucose 1, other organics 0.1
    (mmol/gDW/h), inorganics unlimited, every other exchange closed for uptake."""
    allow = {n.lower() for n in HAMS}
    for r in m.exchanges:
        nm = next(iter(r.metabolites)).name.lower()
        r.lower_bound = 0.0 if nm not in allow else -1000.0 if nm in INORGANIC else -1.0 if nm == "glucose" else -0.1


def _tsv_map(sp: str) -> dict[str, str]:
    p = ANNOT / f"{sp}_gem_map.tsv"  # written by export_maps.py (needs slbench resolvers)
    with open(p) as fh:
        return {r["gem_gene"]: r["gene"] for r in csv.DictReader(fh, delimiter="\t")}


def _animal(sp: str, path: str, medium: str):
    m = _read(GEMS / path)
    if medium == "hams":
        _hams(m)
    mp = _tsv_map(sp)
    return m, {g.id: mp[g.id] for g in m.genes if g.id in mp}


def mmus(medium: str = "hams"):
    """Mouse-GEM (SysBioChalmers, Human-GEM lineage); genes are MGI symbols."""
    return _animal("mmus", "Mouse-GEM/model/Mouse-GEM.xml", medium)


def dmel(medium: str = "hams"):
    """Fruitfly-GEM (SysBioChalmers); gene symbols -> FBgn."""
    return _animal("dmel", "Fruitfly-GEM/model/Fruitfly-GEM.xml", medium)


def cele(medium: str = "hams"):
    """Worm-GEM (SysBioChalmers); sequence names -> WBGene."""
    return _animal("cele", "Worm-GEM/model/Worm-GEM.xml", medium)


def human(medium: str = "hams"):
    m = _read(GEMS / "Human-GEM/model/Human-GEM.xml")
    if medium == "hams":
        _hams(m)
    sym = {}
    with open(RAW / "ids/hgnc_complete_set.txt") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            if r["status"] == "Approved" and r["ensembl_gene_id"]:
                sym[r["ensembl_gene_id"]] = r["symbol"]
    return m, {g.id: sym[g.id] for g in m.genes if g.id in sym}


# ---------------------------------------------------------------- bacteria from BiGG (future species)
def _bigg(fn: str):
    with gzip.open(GEMS / fn, "rt") as fh:
        m = cobra.io.read_sbml_model(fh)
    m.solver = "glpk"
    return m


def ecol(medium: str = "default"):
    """E. coli K-12 MG1655 iML1515. Genes are b-numbers; also mapped by gene name so either ID works."""
    m = _bigg("iML1515.xml.gz")
    return m, {g.id: g.id for g in m.genes}


def bsub(medium: str = "default"):
    """B. subtilis 168 iYO844. Genes are BSU locus tags."""
    m = _bigg("iYO844.xml.gz")
    return m, {g.id: g.id for g in m.genes}


def gene_name_aliases(m: cobra.Model) -> dict[str, str]:
    """model gene id -> gene name (BiGG models carry names), for benchmarks that use names."""
    return {g.id: g.name for g in m.genes if g.name}


# ---------------------------------------------------------------- helpers
AA_NUC_NAMES = [
    "L-alanine", "L-arginine", "L-asparagine", "L-aspartate", "L-cysteine", "L-glutamate", "L-glutamine",
    "glycine", "L-histidine", "L-isoleucine", "L-leucine", "L-lysine", "L-methionine", "L-phenylalanine",
    "L-proline", "L-serine", "L-threonine", "L-tryptophan", "L-tyrosine", "L-valine", "adenine", "uracil",
    "guanine", "cytosine", "thymine", "inositol", "myo-inositol", "nicotinate", "pantothenate", "biotin",
    "thiamine", "riboflavin", "folate", "4-aminobenzoate", "pyridoxine", "choline",
]


def _open_rich(m: cobra.Model, names: list[str], flux: float = 1.0) -> None:
    """Open uptake of amino acids / nucleobases / vitamins (synthetic complete-like medium)."""
    allow = {n.lower() for n in names}
    for r in m.exchanges:
        met = next(iter(r.metabolites))
        nm = re.sub(r"\s*\[.*\]$", "", met.name).lower()
        if nm in allow and r.lower_bound > -flux:
            r.lower_bound = -flux


LOADERS = {"scer": scer, "spom": spom, "spne": spne, "human": human, "ecol": ecol, "bsub": bsub,
           "mmus": mmus, "dmel": dmel, "cele": cele}
DEFAULT_MEDIUM = {"scer": "default", "spom": "default", "spne": "default", "human": "hams", "ecol": "default",
                  "bsub": "default", "mmus": "default", "dmel": "default", "cele": "default"}
# mouse/fly/worm GEMs: the Ham's medium gives ~0 growth (their biomass needs other nutrients), so they use the
# distributed medium (all exchanges open, i.e. a rich medium); only the biomass objective limits growth.
