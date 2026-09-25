"""Cell-line context metadata, including inferred genetic ancestry from Cellosaurus.

Cellosaurus "Genome ancestry" comments carry per-line ancestry fractions inferred from
genotypes (Dutil et al. 2019, PubMed 30894373). We collapse them to 1000 Genomes-style
super-populations and call a line's ancestry group the majority component if it is at
more than ANCESTRY_MAJORITY (a strict majority), otherwise "admixed". Lines with no genotype-based estimate fall
back to the self-reported Cellosaurus "Population" field, flagged as such.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import polars as pl

RAW = Path("data/raw")
ANCESTRY_MAJORITY = 0.50

SUPERPOP = {
    "African": "AFR",
    "Native American": "AMR",
    "East Asian, North": "EAS",
    "East Asian, South": "EAS",
    "South Asian": "SAS",
    "European, North": "EUR",
    "European, South": "EUR",
}

POPULATION_SELF_REPORT = {
    "caucasian": "EUR", "white": "EUR", "european": "EUR",
    "african american": "AFR", "african": "AFR", "black": "AFR",
    "japanese": "EAS", "chinese": "EAS", "korean": "EAS", "east asian": "EAS", "asian": "EAS",
    "thai": "EAS", "vietnamese": "EAS", "filipino": "EAS", "malay": "EAS",
    "south asian": "SAS", "indian": "SAS", "pakistani": "SAS",
    "hispanic": "AMR", "latin american": "AMR", "native american": "AMR", "amerindian": "AMR",
}


def _norm(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", name.upper())


@functools.cache
def cellosaurus_human() -> pl.DataFrame:
    """One row per human Cellosaurus cell line with ancestry and DepMap cross-refs."""
    rows = []
    rec: dict = {}
    with open(RAW / "cellosaurus/cellosaurus.txt", encoding="utf-8", errors="replace") as f:
        for line in f:
            tag, val = line[:2], line[5:].rstrip("\n")
            if tag == "ID":
                rec = {"name": val, "synonyms": [], "depmap": None, "ancestry": None, "population": None,
                       "human": False, "sex": None, "disease": None, "cosmic_clp": None}
            elif tag == "AC":
                rec["accession"] = val
            elif tag == "SY":
                rec["synonyms"] = [s.strip() for s in val.split(";") if s.strip()]
            elif tag == "DR" and val.startswith("DepMap;"):
                rec["depmap"] = val.split(";")[1].strip()
            elif tag == "DR" and val.startswith("Cosmic-CLP;"):
                rec["cosmic_clp"] = int(val.split(";")[1].strip())
            elif tag == "OX" and "NCBI_TaxID=9606" in val:
                rec["human"] = True
            elif tag == "SX":
                rec["sex"] = val
            elif tag == "DI" and rec.get("disease") is None:
                rec["disease"] = val.split(";")[-1].strip()
            elif tag == "CC":
                if val.startswith("Genome ancestry:"):
                    rec["ancestry"] = val
                elif val.startswith("Population:"):
                    rec["population"] = val.split(":", 1)[1].strip().rstrip(".")
            elif tag == "//" and rec.get("human"):
                rows.append(_finish(rec))
    return pl.DataFrame(rows, infer_schema_length=None)


def _finish(rec: dict) -> dict:
    fracs = {p: 0.0 for p in set(SUPERPOP.values())}
    if rec["ancestry"]:
        for part in rec["ancestry"].split(":", 1)[1].split(";"):
            m = re.match(r"\s*(.+?)=([\d.]+)%", part)
            if m and m.group(1) in SUPERPOP:
                fracs[SUPERPOP[m.group(1)]] += float(m.group(2)) / 100
        top = max(fracs, key=fracs.get)
        group, basis = (top if fracs[top] > ANCESTRY_MAJORITY else "admixed"), "genotype"
    elif rec["population"]:
        tokens = [t.strip() for t in re.split(r"[;,/]| and ", rec["population"].lower()) if t.strip()]
        groups = {POPULATION_SELF_REPORT.get(t, POPULATION_SELF_REPORT.get(t.split()[0])) for t in tokens} - {None}
        group = groups.pop() if len(groups) == 1 else "other"
        basis = "self_reported"
        fracs = {p: None for p in fracs}
    else:
        group, basis = "unknown", "none"
        fracs = {p: None for p in fracs}
    return {
        "cellosaurus_ac": rec.get("accession"), "cellosaurus_name": rec["name"], "depmap_id": rec["depmap"],
        "cosmic_clp": rec["cosmic_clp"],
        "synonyms": rec["synonyms"], "sex": rec["sex"], "disease": rec["disease"],
        "population_reported": rec["population"], "ancestry_group": group, "ancestry_basis": basis,
        **{f"anc_{k}": v for k, v in sorted(fracs.items())},
    }


@functools.cache
def _name_index() -> dict[str, list[int]]:
    df = cellosaurus_human()
    idx: dict[str, list[int]] = {}
    for i, (n, syn) in enumerate(zip(df["cellosaurus_name"], df["synonyms"])):
        for k in {_norm(n), *(_norm(s) for s in syn)}:
            idx.setdefault(k, []).append(i)
    return idx


# Manual pins where a screen's cell-line label is ambiguous in Cellosaurus.
PINS = {
    "RPE1": "CVCL_4388",      # hTERT RPE-1
    "RPE": "CVCL_4388",
    "HAP1": "CVCL_Y019",
    "HEK293T": "CVCL_0063",
    "293T": "CVCL_0063",
    "K562": "CVCL_0004",
    "HELA": "CVCL_0030",
    "JURKAT": "CVCL_0065",    # Jurkat, Clone E6-1
    "A549": "CVCL_0023",
    "MEWO": "CVCL_0445",
}


def lookup(label: str, depmap_id: str | None = None) -> dict | None:
    df = cellosaurus_human()
    if depmap_id:
        hit = df.filter(pl.col("depmap_id") == depmap_id)
        if hit.height == 1:
            return hit.row(0, named=True)
    key = _norm(label)
    if key in PINS:
        return df.filter(pl.col("cellosaurus_ac") == PINS[key]).row(0, named=True)
    rows = _name_index().get(key, [])
    # prefer the primary-name match, then a line with a DepMap cross-ref
    exact = [i for i in rows if _norm(df["cellosaurus_name"][i]) == key]
    for cand in (exact, [i for i in rows if df["depmap_id"][i]], rows):
        if len(cand) == 1:
            return df.row(cand[0], named=True)
    return None


KESSLER = RAW / "kessler2019/CNCR-125-2076-s002.xlsx"
KESSLER_GROUPS = {"EUR": "European", "AFR": "African", "EAS": "East_Asian", "SAS": "South_Asian", "AMR": "Native_American"}


def kessler_ancestry() -> pl.DataFrame:
    """Kessler et al. 2019 (Cancer, doi:10.1002/cncr.32020, PMC6541501) genotype ancestry of 1,013 Sanger (COSMIC) lines, Table S2."""
    import openpyxl

    ws = openpyxl.load_workbook(KESSLER, read_only=True)["Table-S2|CellLineAncestryEstima"]
    rows = list(ws.iter_rows(values_only=True))
    k = pl.DataFrame(rows[1:], schema=list(rows[0]), orient="row")
    k = k.select(pl.col("Cell_Line_Name").alias("kessler_name"), pl.col("Cell_Line_ID").cast(pl.Int64).alias("cosmic_clp"),
                 *[pl.col(f"{v}_Ancestry_Proportion").cast(pl.Float64).alias(f"kessler_{g}") for g, v in KESSLER_GROUPS.items()])
    top = pl.concat_list([pl.col(f"kessler_{g}") for g in KESSLER_GROUPS])
    return k.with_columns(
        pl.when(top.list.max() > ANCESTRY_MAJORITY)
        .then(pl.lit(list(KESSLER_GROUPS)).list.get(top.list.arg_max())).otherwise(pl.lit("admixed")).alias("kessler_group"))


def kessler_crosscheck(contexts: pl.DataFrame) -> pl.DataFrame:
    """Human benchmark contexts next to Kessler's independent genotype ancestry, matched by COSMIC cell-line ID."""
    cells = cellosaurus_human().select("cellosaurus_ac", "cosmic_clp")
    h = contexts.filter(pl.col("species") == "human").select("context_id", "cellosaurus_ac", "ancestry_group", "ancestry_basis")
    return h.join(cells, on="cellosaurus_ac", how="left").join(kessler_ancestry(), on="cosmic_clp", how="left") \
        .with_columns((pl.col("kessler_group") == pl.col("ancestry_group")).alias("agree")).sort("context_id")
