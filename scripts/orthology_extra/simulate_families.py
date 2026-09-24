"""Simulate SLB family assignment with the extra homology edges (families_extra) on the current benchmark genes.

Reuses families.DSU / families._cap and build.bucket; nothing under data/bench is written or read from hidden/.
Reports (stdout + data/interim/orthology_extra/simulation.json):
  * baseline check: families.edges() alone reproduces slb1.2 held_out_families.parquet
  * spne genes that join cross-species families
  * component size distribution (benchmark genes per family), before / after
  * genes (esp. currently-test genes) whose bucket changes, per species, under two naming rules:
      naive  - family ID = DSU root = lexicographically smallest node of the component (what families.assign does)
      stable - family ID = smallest node of a species already in SLB-1.2 (human, scer, spom, dmel, spne) when the
               component has one; avoids relabelling families just because e.g. an "ecol:" node joined.
Usage: uv run python scripts/orthology_extra/simulate_families.py [--no-base-pairs]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import polars as pl

from slpbench import families, families_extra
from slpbench.build import bucket

BENCH = Path("data/bench/slb1.2/held_out_families.parquet")
OUT = Path("data/interim/orthology_extra")
OLD_SPECIES = ("human", "scer", "spom", "dmel", "spne")


def assign(genes: pl.DataFrame, e: pl.DataFrame, stable: bool) -> dict[str, str]:
    nodes = [f"{s}:{g}" for s, g in genes.select("species", "gene").iter_rows()]
    dsu = families.DSU()
    for n in nodes:
        dsu.find(n)
    for u, v in e.select("u", "v").iter_rows():
        dsu.union(u, v)
    if stable:
        label: dict[str, str] = {}
        for n in list(dsu.p):
            r = dsu.find(n)
            if n.split(":", 1)[0] in OLD_SPECIES and (r not in label or n < label[r]):
                label[r] = n
        fam = {n: label.get(dsu.find(n), dsu.find(n)) for n in nodes}
    else:
        fam = {n: dsu.find(n) for n in nodes}
    return families._cap(fam, e, set(nodes))


def sizes(fam: dict[str, str]) -> dict:
    c = Counter(fam.values())
    s = pl.Series(list(c.values()))
    bins = {"1": int((s == 1).sum()), "2-5": int(((s >= 2) & (s <= 5)).sum()), "6-20": int(((s >= 6) & (s <= 20)).sum()),
            "21-100": int(((s >= 21) & (s <= 100)).sum()), "101-400": int(((s >= 101) & (s <= 400)).sum()),
            ">400": int((s > 400).sum())}
    top = sorted(c.items(), key=lambda x: -x[1])[:10]
    return {"families": len(c), "genes": len(fam), "max": int(s.max()), "bins": bins,
            "largest": [f"{k} ({v})" for k, v in top], "capped_genes": sum(1 for f in fam.values() if f.startswith("cap:"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-base-pairs", action="store_true", help="drop DIAMOND RBH edges between two base species")
    ap.add_argument("--species", default=None, help="comma-separated species whose extra edges are used (default all)")
    ap.add_argument("--tag", default="", help="suffix for output files")
    a = ap.parse_args()
    tag = a.tag or ("_nobase" if a.no_base_pairs else "")
    cur = pl.read_parquet(BENCH)
    genes = cur.select("species", "gene")
    nodes = [f"{s}:{g}" for s, g in genes.iter_rows()]
    old_bucket = dict(zip(nodes, cur["bucket"]))

    base = families.edges()
    fam0 = assign(genes, base, stable=False)
    reproduced = sum(fam0[n] == f for n, f in zip(nodes, cur["family"]))
    report: dict = {"baseline_reproduces_slb1.2": f"{reproduced}/{len(nodes)}", "before": sizes(fam0)}

    extra = families_extra.extra_edges(base_pairs=not a.no_base_pairs,
                                       species=set(a.species.split(",")) if a.species else None)
    both = pl.concat([base, extra]).unique()
    report["edges"] = {"base": base.height, "extra": extra.height, "combined_unique": both.height}

    # spne linkage: which spne genes share a (pre-cap) component with other species
    dsu = families.DSU()
    for u, v in both.select("u", "v").iter_rows():
        dsu.union(u, v)
    comp_species: dict[str, set] = {}
    comp_bench_species: dict[str, set] = {}
    for n in list(dsu.p):
        comp_species.setdefault(dsu.find(n), set()).add(n.split(":", 1)[0])
    for n in nodes:
        if n in dsu.p:
            comp_bench_species.setdefault(dsu.find(n), set()).add(n.split(":", 1)[0])
    spne = [n for n in nodes if n.startswith("spne:")]
    rs = {n: dsu.find(n) for n in spne if n in dsu.p}
    report["spne"] = {
        "benchmark_genes": len(spne),
        "with_any_edge": len(rs),
        "with_spne_paralog_in_benchmark": sum(1 for n in spne if n in rs and sum(1 for m in spne if rs.get(m) == rs[n]) > 1),
        "in_cross_species_component_any_node": sum(1 for n, r in rs.items() if comp_species[r] - {"spne"}),
        "in_cross_species_component_with_other_benchmark_species": sum(1 for n, r in rs.items() if comp_bench_species[r] - {"spne"}),
        "partner_species_counts": dict(Counter(s for n, r in rs.items() for s in comp_species[r] - {"spne"}).most_common()),
    }

    for stable in (False, True):
        fam1 = assign(genes, both, stable=stable)
        new_bucket = {n: bucket(f) for n, f in fam1.items()}
        key = "stable_naming" if stable else "naive_naming"
        ch = pl.DataFrame({"node": nodes, "species": genes["species"], "old": [old_bucket[n] for n in nodes],
                           "new": [new_bucket[n] for n in nodes]})
        changed = ch.filter(pl.col("old") != pl.col("new"))
        trans = changed.group_by("species", "old", "new").len().sort("species", "old", "new")
        report[key] = {
            "after": sizes(fam1),
            "genes_changing_bucket": changed.height,
            "test_genes_changing_bucket": changed.filter(pl.col("old") == "test").height,
            "test_genes_before": int((ch["old"] == "test").sum()), "test_genes_after": int((ch["new"] == "test").sum()),
            "transitions": trans.to_dicts(),
            "families_merged": len(set(fam0.values())) - len(set(fam1.values())),
        }
        if stable:
            pl.DataFrame({"species": genes["species"], "gene": genes["gene"], "family": [fam1[n] for n in nodes],
                          "bucket": [new_bucket[n] for n in nodes]}).write_parquet(
                OUT / f"simulated_families{tag}.parquet")
    fn = OUT / f"simulation{tag}.json"
    fn.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
