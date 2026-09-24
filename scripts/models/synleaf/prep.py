"""SynLeaF (Xing et al. 2026) data preparation for SLB, pan-cancer setting.

Re-implements src/preprocess_main.py (pan-cancer: --ct pan --cn_kg TOTAL --omics_types cna exp mut) with:
  * SL labels = SLB human train rows ONLY (no SynLethDB / ELISL labels).
  * SynLethKG 2.0 (SynLethDB 2.0 sldb_complete.csv, from the authors' packaged data_raw.tar.gz) with
    the SL_GsG / NONSL_GnsG / SR_GsrG relations REMOVED (exactly as the original code does; asserted below).
  * SynLethKG's INTERACTS_GiG edges replaced by the shared bundle PPI (see substitute_ppi).
  * Gene symbols in KG, TCGA and UniProt harmonised to current HGNC symbols (SLB uses current symbols).
Writes, under external/models/synleaf/data/$SYNLEAF_CT (default slb)/ (train.py reads ../data/<ct>/ from src/):
  gene.pt, kg.pt, cna.npy, exp.npy, mut.npy, cv_1_fold_1/{train,val,test}_sl.npy, eval_<split>.parquet
The model is context-agnostic (like the original pan-cancer model): one training row per SLB (context, pair).
Validation (early stopping) = train rows touching a random 10% of genes (gene-held-out, like dev); test_sl.npy
is a copy of val (the original script needs one; dev labels are never given to train.py).

usage: .venv/bin/python prep.py <split> [<split> ...]      (splits to prepare eval files for, e.g. dev)
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import slb  # noqa: E402

RAW = slb.RAW / "synleaf/extracted/data_raw"
OUT = slb.ROOT / "external/models/synleaf/data" / os.environ.get("SYNLEAF_CT", "slb")
FORBIDDEN = ["SL_GsG", "NONSL_GnsG", "SR_GsrG"]
OMICS = ["cna", "exp", "mut"]
res = slb.symbol_resolver()


def canon(s):
    return res(s) if isinstance(s, str) else None


def load_kg() -> pd.DataFrame:
    kg = pd.read_csv(RAW / "SLKG2/raw_kg.tsv", sep="\t", low_memory=False)
    n0 = len(kg)
    kg = kg[~kg.relation.isin(FORBIDDEN)].copy()
    print(f"KG edges {n0:,}; dropped SL/nonSL/SR relations -> {len(kg):,}", file=sys.stderr)
    assert not kg.relation.isin(FORBIDDEN).any()
    assert not kg.relation.str.contains("SL|GsG|GnsG|GsrG").any(), kg.relation.unique()
    # harmonise gene names to current HGNC symbols (unmapped names are kept as-is: still valid KG nodes)
    for side in "xy":
        g = kg[f"{side}_type"] == "Gene"
        mapped = kg.loc[g, f"{side}_name"].map(lambda s: canon(s) or s)
        kg.loc[g, f"{side}_name"] = mapped
    # original modify_gene_id_with_duplicate_gene_names: one entity id per gene name (largest id)
    genes = pd.concat([kg.loc[kg.x_type == "Gene", ["x_id", "x_name"]].set_axis(["id", "name"], axis=1),
                       kg.loc[kg.y_type == "Gene", ["y_id", "y_name"]].set_axis(["id", "name"], axis=1)]).drop_duplicates()
    name2id = genes.groupby("name").id.max()
    for side in "xy":
        g = kg[f"{side}_type"] == "Gene"
        kg.loc[g, f"{side}_id"] = kg.loc[g, f"{side}_name"].map(name2id).values
    return kg


PPI_REL = "INTERACTS_GiG"
BUNDLE_CH = ["neighborhood", "fusion", "cooccurence", "coexpression", "database"]


def substitute_ppi(kg: pd.DataFrame) -> pd.DataFrame:
    """Lead's leakage rule: SynLethKG's INTERACTS_GiG (147k gene-gene 'interacts' edges of undocumented
    provenance: 93.5k are BioGRID physical, 121 BioGRID genetic-only, 53k in neither) is replaced by the shared
    bundle PPI (data/interim/bundle/human/ppi.parquet): BioGRID 5.0.261 physical rows only + STRING v12
    neighborhood / fusion / cooccurence / coexpression / database channels (>= 400); no STRING experimental,
    textmining or combined score, no BioGRID genetic rows. Edges are added between KG genes only, one relation
    per evidence channel (as in the allspecies KG)."""
    n = int((kg.relation == PPI_REL).sum())
    kg = kg[kg.relation != PPI_REL]
    ppi = pd.read_parquet(slb.ROOT / "data/interim/bundle/human/ppi.parquet")
    genes = pd.concat([kg.loc[kg.x_type == "Gene", ["x_id", "x_name"]].set_axis(["id", "name"], axis=1),
                       kg.loc[kg.y_type == "Gene", ["y_id", "y_name"]].set_axis(["id", "name"], axis=1)]).drop_duplicates()
    name2id = genes.groupby("name").id.max()
    sel = {"PPI_BIOGRID_PHYS": ppi.biogrid_phys > 0}
    sel.update({f"STRING_{c.upper()}": ppi[f"string_{c}"] >= 400 for c in BUNDLE_CH})
    add = []
    for rel, m in sel.items():
        e = ppi.loc[m, ["gene_a", "gene_b"]]
        e = e[e.gene_a.isin(name2id.index) & e.gene_b.isin(name2id.index)]
        add.append(pd.DataFrame({"x_id": name2id[e.gene_a].values, "x_type": "Gene", "x_name": e.gene_a.values,
                                 "y_id": name2id[e.gene_b].values, "y_type": "Gene", "y_name": e.gene_b.values,
                                 "relation": rel}))
    add = pd.concat(add, ignore_index=True)
    print(f"PPI substitution: dropped {n:,} {PPI_REL} edges, added {len(add):,} bundle PPI edges "
          f"({add.relation.value_counts().to_dict()})", file=sys.stderr)
    return pd.concat([kg, add], ignore_index=True)


def load_tcga() -> dict[str, pd.DataFrame]:
    out = {}
    for o in OMICS:
        df = pd.read_csv(RAW / f"TCGA/pan/{o}.txt", sep="\t", low_memory=False, comment="#")
        if o == "mut":
            df = df[["Hugo_Symbol", "Tumor_Sample_Barcode"]].copy()
            df["Hugo_Symbol"] = df.Hugo_Symbol.map(canon)
            df = df.dropna()
            m = df.groupby(["Hugo_Symbol", "Tumor_Sample_Barcode"]).size().unstack(fill_value=0)  # mutation counts
        else:
            df = df.drop(columns=[c for c in ["Entrez_Gene_Id"] if c in df.columns])
            df["Hugo_Symbol"] = df.Hugo_Symbol.map(canon)
            df = df.dropna(subset=["Hugo_Symbol"]).drop_duplicates("Hugo_Symbol").set_index("Hugo_Symbol")
            m = df.apply(pd.to_numeric, errors="coerce")
        out[o] = m
        print(f"TCGA pan {o}: {m.shape}", file=sys.stderr)
    return out


def uniprot_genes() -> set:
    gs = set()
    for r in SeqIO.parse(RAW / "uniprot/uniprotkb_organism_id_9606_AND_reviewed_2024_10_26.fasta", "fasta"):
        m = re.search(r"GN=(.*?) PE", r.description)
        if m and canon(m.group(1)):
            gs.add(canon(m.group(1)))
    return gs


def build_kg(kg: pd.DataFrame, slgenes: list) -> dict:
    # generate_generic_nodes_data
    nodes = pd.concat([kg[["x_id", "x_name", "x_type"]].set_axis(["id", "name", "type"], axis=1),
                       kg[["y_id", "y_name", "y_type"]].set_axis(["id", "name", "type"], axis=1)]).drop_duplicates()
    ent = {e: i + 2 for i, e in enumerate(sorted(nodes.id.unique()))}
    ent["<PAD>"], ent["<MASK>"] = 0, 1
    idx2ent = {i: e for e, i in ent.items()}
    ge = nodes[nodes.type == "Gene"].drop_duplicates("id")
    eidx2name = {ent[i]: n for i, n in zip(ge.id, ge.name)}
    name2eidx = {n: i for i, n in eidx2name.items()}
    # generate_specific_edges_data (pan: keep all Disease nodes; drop edges of genes outside the model gene set)
    s = set(slgenes)
    xg, yg = kg.x_type == "Gene", kg.y_type == "Gene"
    xin, yin = kg.x_name.isin(s), kg.y_name.isin(s)
    drop = (xg & ~xin & ~yg) | (yg & ~yin & ~xg) | (xg & yg & ~xin & ~yin)
    kg = kg[~drop]
    print(f"KG edges after dropping non-model genes: {len(kg):,}", file=sys.stderr)
    g = HeteroData()
    for t, nd in nodes.groupby("type"):
        g[t].node_id = torch.unique(torch.tensor([ent[i] for i in nd.id], dtype=torch.long))
    for (xt, yt), grp in kg.groupby(["x_type", "y_type"]):
        for rel, r in grp.groupby("relation"):
            ei = torch.tensor([[ent[i] for i in r.x_id], [ent[i] for i in r.y_id]], dtype=torch.long)
            g[(xt, rel, yt)].edge_index = torch.unique(ei, dim=1)
    for (xt, yt), grp in kg.groupby(["x_type", "y_type"]):
        for rel, r in grp.groupby("relation"):
            if rel == "REGULATES_GrG":  # stored bidirectionally in SLKG2
                continue
            ei = torch.tensor([[ent[i] for i in r.y_id], [ent[i] for i in r.x_id]], dtype=torch.long)
            g[(yt, rel + "_reversed", xt)].edge_index = torch.unique(ei, dim=1)
    print(f"KG: {len(ent):,} entities, {len(g.edge_types)} relation types", file=sys.stderr)
    return {"entity_to_index": ent, "index_to_entity": idx2ent, "entity_idx_to_gene_name": eidx2name,
            "gene_name_to_entity_idx": name2eidx, "kg_graph": g}


def pairs_to_idx(df: pd.DataFrame, gidx: dict) -> pd.DataFrame:
    df = df.copy()
    df["i"] = df.gene_a.map(gidx)
    df["j"] = df.gene_b.map(gidx)
    return df


def write_eval(eval_splits, gidx):
    for sp in eval_splits:
        d = slb.load(sp)
        d = pairs_to_idx(d[d.species == "human"], gidx)
        d[["example_id", "i", "j"]].to_parquet(OUT / f"eval_{sp}.parquet", index=False)
        ok = d.i.notna() & d.j.notna()
        print(f"{sp}: human rows with both genes in model {int(ok.sum()):,}/{len(d):,}", file=sys.stderr)


def keep_isolated(kg: pd.DataFrame, genes) -> pd.DataFrame:
    """Model genes whose only KG edges were INTERACTS_GiG get a self-loop (relation SELF_GG) so they stay KG
    entities (their KG embedding is then just their own learned entity embedding)."""
    have = set(kg.loc[kg.x_type == "Gene", "x_name"]) | set(kg.loc[kg.y_type == "Gene", "y_name"])
    iso = sorted(set(genes) - have)
    if not iso:
        return kg
    ids = {n: i for n, i in zip(kg_orig_gene_names["name"], kg_orig_gene_names["id"])}
    add = pd.DataFrame({"x_id": [ids[g] for g in iso], "x_type": "Gene", "x_name": iso,
                        "y_id": [ids[g] for g in iso], "y_type": "Gene", "y_name": iso, "relation": "SELF_GG"})
    print(f"{len(iso)} model genes left without KG edges after the PPI substitution -> self-loops", file=sys.stderr)
    return pd.concat([kg, add], ignore_index=True)


kg_orig_gene_names = None


def main(eval_splits):
    global kg_orig_gene_names
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "cv_1_fold_1/val_sl.npy").exists() and (OUT / "kg.pt").exists():  # model inputs built already
        gidx = {g: i for i, g in torch.load(OUT / "gene.pt", weights_only=False).items()}
        return write_eval(eval_splits, gidx)
    kg = load_kg()
    # gene universe from the SL-free SynLethKG as in the original (before the PPI substitution)
    kg_genes = set(kg.loc[kg.x_type == "Gene", "x_name"]) | set(kg.loc[kg.y_type == "Gene", "y_name"])
    kg_orig_gene_names = pd.concat([kg.loc[kg.x_type == "Gene", ["x_id", "x_name"]].set_axis(["id", "name"], axis=1),
                                    kg.loc[kg.y_type == "Gene", ["y_id", "y_name"]].set_axis(["id", "name"], axis=1)]
                                   ).drop_duplicates("name")
    kg = substitute_ppi(kg)
    left = set(kg.loc[kg.x_type == "Gene", "x_name"]) | set(kg.loc[kg.y_type == "Gene", "y_name"])
    if (OUT / "cv_1_fold_1/val_sl.npy").exists():  # omics + folds built already: only (re)build the KG
        gidx = {g: i for i, g in torch.load(OUT / "gene.pt", weights_only=False).items()}
        kg = keep_isolated(kg, list(gidx))
        torch.save(build_kg(kg, list(gidx)), OUT / "kg.pt")
        return write_eval(eval_splits, gidx)
    tc = load_tcga()
    up = uniprot_genes()
    tcga_genes = set().union(*[set(m.index) for m in tc.values()])
    genes = sorted(kg_genes & up & tcga_genes)
    print(f"genes: KG {len(kg_genes):,} UniProt {len(up):,} TCGA {len(tcga_genes):,} -> intersect {len(genes):,}",
          file=sys.stderr)
    kg = keep_isolated(kg, genes)
    gidx = {g: i for i, g in enumerate(genes)}
    torch.save({i: g for g, i in gidx.items()}, OUT / "gene.pt")
    torch.save(build_kg(kg, genes), OUT / "kg.pt")

    samples = np.sort(pd.Index(np.concatenate([m.columns.astype(str) for m in tc.values()])).unique())
    for o, m in tc.items():
        m.columns = m.columns.astype(str)
        a = m.reindex(index=genes, columns=samples).fillna(0).to_numpy(dtype=np.float64)
        if o == "exp":
            a = np.clip(a, -10.0, None)
        if o == "mut":
            mx = a.max()
            a = np.log1p(a) / np.log1p(mx) if mx > 0 else np.zeros_like(a)
        np.save(OUT / f"{o}.npy", a)
        print(f"{o}.npy {a.shape}", file=sys.stderr)

    tr = slb.load("train")
    tr = pairs_to_idx(tr[tr.species == "human"], gidx)
    n = len(tr)
    tr = tr.dropna(subset=["i", "j", "label"])
    print(f"train rows usable {len(tr):,}/{n:,} ({int(tr.label.sum()):,} SL)", file=sys.stderr)
    rng = np.random.default_rng(2025)
    tg = np.array(sorted(set(tr.gene_a) | set(tr.gene_b)))
    val_genes = set(rng.choice(tg, size=len(tg) // 10, replace=False))
    is_val = tr.gene_a.isin(val_genes) | tr.gene_b.isin(val_genes)
    fold = OUT / "cv_1_fold_1"
    fold.mkdir(exist_ok=True)
    arr = lambda d: d[["i", "j", "label"]].to_numpy(dtype=np.int64)  # noqa: E731
    np.save(fold / "train_sl.npy", arr(tr[~is_val]))
    np.save(fold / "val_sl.npy", arr(tr[is_val]))
    np.save(fold / "test_sl.npy", arr(tr[is_val]))
    print(f"fit rows {int((~is_val).sum()):,} ({int(tr[~is_val].label.sum())} SL); val rows {int(is_val.sum()):,} "
          f"({int(tr[is_val].label.sum())} SL)", file=sys.stderr)
    write_eval(eval_splits, gidx)


if __name__ == "__main__":
    main(sys.argv[1:] or ["dev"])
