"""PiLSL inputs for the SLB universe. Plain arrays only (the dict of node features is
written inside the legacy container by make_pilsl_feat.py).
  pilsl_data/pilsl_graph_pairs.npy      training SL pairs (fit positives) = relation-0 edges of the subgraph graph
                                        (original PiLSL design; Feng's release used *all* 48M gene pairs instead)
  pilsl_data/all_pairs_used_9845.npy    pairs whose enclosing subgraphs are extracted: fit + valid (pos and neg) and
                                        every human pair of dev / test inputs (no labels)
  pilsl_data/slb_predict_pairs.npy      the dev / test-input pairs (scored after training, label-free)
  pilsl_data/pilsl_unified_etype_9845.csv  entity type of every (re-mapped) KG entity"""
import os
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(os.environ.get("SLB_BENCH", ROOT / "data/slb"))
if not BENCH.is_absolute():
    BENCH = ROOT / BENCH
WORK = Path(os.environ.get("SLB_WORK", ROOT / "external/models/_slb_work" / BENCH.name))
FD = WORK / "feng/data"
OUT = FD / "preprocessed_data/pilsl_data"; OUT.mkdir(parents=True, exist_ok=True)
z = np.load(FD / "data_split/slb_split_arrays.npz")
gi = pd.read_parquet(FD / "slb_gene_index.parquet"); u = dict(zip(gi.gene, gi.unified_id))
pp = []
for f in ["dev_inputs.parquet", "test_inputs.parquet"]:
    p = BENCH / f
    if p.exists():
        d = pd.read_parquet(p, columns=["species", "gene_a", "gene_b"]); d = d[d.species == "human"]
        a, b = d.gene_a.map(u).values, d.gene_b.map(u).values
        pp.append(np.stack([np.minimum(a, b), np.maximum(a, b)], 1))
pp = np.unique(np.concatenate(pp), axis=0); pp = pp[pp[:, 0] != pp[:, 1]]
db = np.unique(np.concatenate([z["fit_pos"], z["fit_neg"], z["val_pos"], z["val_neg"], pp]), axis=0)
z3 = lambda x: np.hstack([x, np.zeros((len(x), 1), dtype=np.int64)]).astype(np.int64)
np.save(OUT / "all_pairs_used_9845.npy", z3(db)); np.save(OUT / "pilsl_graph_pairs.npy", z3(z["fit_pos"]))
np.save(OUT / "slb_predict_pairs.npy", pp.astype(np.int64))
ents = pd.read_csv(ROOT / "data/raw/feng2024_slbench/extracted/data/preprocessed_data/not_used/fin_entities.csv")
m = np.load(FD / "kg_entity_remap.npy")
n_ent = int(m.max()) + 1
et = np.full(n_ent, 2.0)  # 2 = Gene in Feng's typing
et[m[ents.unified_id.values]] = ents.entity_type.values
np.savetxt(OUT / "pilsl_unified_etype_9845.csv", et)
print(f"db pairs {len(db)}, graph (fit SL) pairs {len(z['fit_pos'])}, predict pairs {len(pp)}, entities {n_ent}")
