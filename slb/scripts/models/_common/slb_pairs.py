"""Shared SLB -> pair-level data for graph/KG/MF SL models (models-graph agent).

Reads SLB_BENCH (default data/slb) and writes, into OUT (default external/models/_slb_work/<bench name>):
  human_train_pairs.parquet  unique unordered human train pairs, contexts pooled:
                             label = 1 if SL in >=1 context, else 0 (all are measured; no unknown=negative);
                             frac_sl, n_ctx; split = fit | valid (gene-family-disjoint internal validation, ~20% of families)
  human_train_rows.parquet   per-context human train rows (for context-aware models) with the same fit/valid split
  human_dev_pairs.parquet    unique unordered human dev pairs (NO labels written)
  human_dev_rows.parquet     human dev example_id, context_id, gene_a, gene_b (NO labels)
Only train labels are read. Dev labels are never touched.
"""
import hashlib, os, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(os.environ.get("SLB_BENCH", ROOT / "data/slb"))
if not BENCH.is_absolute():
    BENCH = ROOT / BENCH
OUT = Path(os.environ.get("SLB_WORK", ROOT / "external/models/_slb_work" / BENCH.name))
VALID_FRAC = float(os.environ.get("SLB_VALID_FRAC", "0.2"))
SP = os.environ.get("SLB_SPECIES", "human")  # any species code present in the benchmark


def _bucket(fam: str) -> str:
    h = int(hashlib.sha256(("slb-graph-valid|" + fam).encode()).hexdigest(), 16) % 1000
    return "valid" if h < VALID_FRAC * 1000 else "fit"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr = pd.read_parquet(BENCH / "train.parquet", columns=["example_id", "species", "context_id", "gene_a", "gene_b", "same_family", "sources", "label"])
    tr = tr[tr.species == SP].copy()
    fam = pd.read_parquet(BENCH / "held_out_families.parquet")
    fam = fam[fam.species == SP].set_index("gene").family.to_dict()
    genes = sorted(set(tr.gene_a) | set(tr.gene_b))
    gsplit = {g: _bucket(fam.get(g, SP + ":" + g)) for g in genes}
    a, b = tr.gene_a.where(tr.gene_a < tr.gene_b, tr.gene_b), tr.gene_b.where(tr.gene_a < tr.gene_b, tr.gene_a)
    tr["gene_a"], tr["gene_b"] = a, b
    sa, sb = tr.gene_a.map(gsplit), tr.gene_b.map(gsplit)
    tr["split"] = "drop"
    tr.loc[(sa == "fit") & (sb == "fit"), "split"] = "fit"
    tr.loc[(sa == "valid") & (sb == "valid"), "split"] = "valid"
    tr.to_parquet(OUT / f"{SP}_train_rows.parquet", index=False)
    p = tr.groupby(["gene_a", "gene_b"]).agg(frac_sl=("label", "mean"), n_ctx=("label", "size"), same_family=("same_family", "first"), split=("split", "first")).reset_index()
    p["label"] = (p.frac_sl > 0).astype(int)
    p.to_parquet(OUT / f"{SP}_train_pairs.parquet", index=False)
    dv = pd.read_parquet(BENCH / "dev.parquet", columns=["example_id", "species", "context_id", "gene_a", "gene_b", "same_family"])
    dv = dv[dv.species == SP].copy()
    a, b = dv.gene_a.where(dv.gene_a < dv.gene_b, dv.gene_b), dv.gene_b.where(dv.gene_a < dv.gene_b, dv.gene_a)
    dv["gene_a"], dv["gene_b"] = a, b
    dv.to_parquet(OUT / f"{SP}_dev_rows.parquet", index=False)
    dv[["gene_a", "gene_b", "same_family"]].drop_duplicates(["gene_a", "gene_b"]).to_parquet(OUT / f"{SP}_dev_pairs.parquet", index=False)
    print(f"bench={BENCH} out={OUT}")
    print(p.groupby("split").label.agg(["size", "sum"]))
    print("dev rows", len(dv), "dev pairs", dv[["gene_a", "gene_b"]].drop_duplicates().shape[0])


if __name__ == "__main__":
    main()
