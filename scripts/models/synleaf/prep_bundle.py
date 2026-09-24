"""SynLeaF `allspecies` variant: the KG/RGCN branch (task only_kg) on a per-species knowledge graph built from the
shared feature bundle data/interim/bundle/<species>/{go,ppi}.parquet (+ _go/edges.parquet), trained per species
on SLB train rows of that species. The omics branch needs TCGA (human-only), so it is not part of this variant.

KG relations (no genetic-interaction evidence anywhere: bundle drops GO IGI, BioGRID genetic, STRING experimental
and textmining):
  Gene -GO_P/GO_F/GO_C-> GOTerm, GOTerm -GO_PARENT-> GOTerm (is_a + part_of),
  Gene -PPI_BIOGRID_PHYS- Gene (>=1 BioGRID physical row), Gene -STRING_<channel>- Gene (channel score >= 400).
Writes external/models/synleaf/data/<ct>/ (gene.pt, kg.pt, none.npy dummy omics, cv_1_fold_1/*, eval_<split>.parquet)
with <ct> = $SYNLEAF_CT_PREFIX_<species>. Species = those with SLB train rows and a bundle go+ppi file.
Training rows of very large species are subsampled (MAX_TRAIN rows, stratified by label) to bound GPU time.

usage: .venv/bin/python prep_bundle.py <split>      prints the prepared species, one per line, on stdout
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import slb  # noqa: E402
from prep import build_kg  # noqa: E402

B = slb.ROOT / "data/interim/bundle"
PREFIX = os.environ.get("SYNLEAF_CT_PREFIX", f"bundle_{slb.BENCH.name}")
MAX_TRAIN = int(os.environ.get("SYNLEAF_MAX_TRAIN", 300_000))
CH = ["neighborhood", "fusion", "cooccurence", "coexpression", "database"]


def species_kg(sp: str) -> pd.DataFrame:
    go = pd.read_parquet(B / sp / "go.parquet")
    ppi = pd.read_parquet(B / sp / "ppi.parquet")
    goe = pd.read_parquet(B / "_go/edges.parquet")
    rows = []
    g2 = go[["gene", "term", "aspect"]].drop_duplicates()
    rows.append(pd.DataFrame({"x_id": "g:" + g2.gene, "x_type": "Gene", "x_name": g2.gene, "y_id": "t:" + g2.term,
                              "y_type": "GOTerm", "y_name": g2.term, "relation": "GO_" + g2.aspect}))
    rows.append(pd.DataFrame({"x_id": "t:" + goe.child, "x_type": "GOTerm", "x_name": goe.child, "y_id": "t:" + goe.parent,
                              "y_type": "GOTerm", "y_name": goe.parent, "relation": "GO_PARENT"}))
    sel = {"PPI_BIOGRID_PHYS": ppi.biogrid_phys > 0}
    sel.update({f"STRING_{c.upper()}": ppi[f"string_{c}"] >= 400 for c in CH if f"string_{c}" in ppi})
    for rel, m in sel.items():
        e = ppi.loc[m, ["gene_a", "gene_b"]]
        rows.append(pd.DataFrame({"x_id": "g:" + e.gene_a, "x_type": "Gene", "x_name": e.gene_a, "y_id": "g:" + e.gene_b,
                                  "y_type": "Gene", "y_name": e.gene_b, "relation": rel}))
    kg = pd.concat(rows, ignore_index=True)
    assert not kg.relation.str.contains("SL|GI|GENETIC|EXPERIMENT|TEXT").any()
    return kg


def main(split):
    tr_all = slb.load("train")
    ev_all = slb.load(split)
    done = []
    for sp in sorted(tr_all.species.unique()):
        if not ((B / sp / "go.parquet").exists() and (B / sp / "ppi.parquet").exists()):
            print(f"{sp}: no bundle go/ppi -> skipped", file=sys.stderr)
            continue
        out = slb.ROOT / "external/models/synleaf/data" / f"{PREFIX}_{sp}"
        out.mkdir(parents=True, exist_ok=True)
        if not (out / "cv_1_fold_1/val_sl.npy").exists():
            kg = species_kg(sp)
            genes = sorted(set(kg.loc[kg.x_type == "Gene", "x_name"]) | set(kg.loc[kg.y_type == "Gene", "y_name"]))
            gidx = {g: i for i, g in enumerate(genes)}
            torch.save({i: g for g, i in gidx.items()}, out / "gene.pt")
            torch.save(build_kg(kg, genes), out / "kg.pt")
            np.save(out / "none.npy", np.zeros((len(genes), 1)))  # dummy omics (only_kg ignores it)
            tr = tr_all[(tr_all.species == sp) & tr_all.label.notna()].copy()
            tr["i"], tr["j"] = tr.gene_a.map(gidx), tr.gene_b.map(gidx)
            n0 = len(tr)
            tr = tr.dropna(subset=["i", "j"])
            if len(tr) > MAX_TRAIN:
                tr = tr.groupby("label").sample(frac=MAX_TRAIN / len(tr), random_state=2025)
            rng = np.random.default_rng(2025)
            tg = np.array(sorted(set(tr.gene_a) | set(tr.gene_b)))
            vg = set(rng.choice(tg, size=max(1, len(tg) // 10), replace=False))
            isv = tr.gene_a.isin(vg) | tr.gene_b.isin(vg)
            (out / "cv_1_fold_1").mkdir(exist_ok=True)
            arr = lambda d: d[["i", "j", "label"]].to_numpy(dtype=np.int64)  # noqa: E731
            np.save(out / "cv_1_fold_1/train_sl.npy", arr(tr[~isv]))
            np.save(out / "cv_1_fold_1/val_sl.npy", arr(tr[isv]))
            np.save(out / "cv_1_fold_1/test_sl.npy", arr(tr[isv]))
            print(f"{sp}: KG genes {len(genes):,}; train rows usable {len(tr):,}/{n0:,} (fit {int((~isv).sum()):,}, "
                  f"val {int(isv.sum()):,}; SL {int(tr.label.sum()):,})", file=sys.stderr)
        gidx = {g: i for i, g in torch.load(out / "gene.pt", weights_only=False).items()}
        ev = ev_all[ev_all.species == sp].copy()
        ev["i"], ev["j"] = ev.gene_a.map(gidx), ev.gene_b.map(gidx)
        ev[["example_id", "i", "j"]].to_parquet(out / f"eval_{split}.parquet", index=False)
        print(f"{sp}: {split} rows with both genes in KG {int((ev.i.notna() & ev.j.notna()).sum()):,}/{len(ev):,}",
              file=sys.stderr)
        done.append(sp)
    print("\n".join(done))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
