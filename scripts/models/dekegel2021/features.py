"""Attach De Kegel et al. 2021 paralog-pair features (Ryan lab, Ensembl 111 rebuild, Zenodo 14973633)
to SLB human pairs. Pairs are matched on unordered HGNC symbols after resolving the table's Ensembl IDs
and symbols to current HGNC symbols. Pairs that are not Ensembl paralogs get no features (NaN).

usage: uv run python scripts/models/dekegel2021/features.py <split> <out.csv>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import slb  # noqa: E402

FEATURES = pd.read_csv(slb.ROOT / "external/models/dekegel_paralog_sl/local_data/feature_list.txt").feature.tolist()


def feature_table() -> pd.DataFrame:
    f = pd.read_csv(slb.RAW / "ryan_paralog_features/ens111_human_allFeatures.csv",
                    usecols=["A1", "A2", "A1_ensembl", "A2_ensembl"] + FEATURES)
    res = slb.symbol_resolver()
    a = [res(e) or res(s) or s for e, s in zip(f.A1_ensembl, f.A1)]
    b = [res(e) or res(s) or s for e, s in zip(f.A2_ensembl, f.A2)]
    f["key"] = slb.pair_key(pd.Series(a), pd.Series(b))
    for c in FEATURES:
        if f[c].dtype == bool or f[c].dtype == object:
            f[c] = f[c].astype(float)
    return f.drop_duplicates("key").set_index("key")[FEATURES]


def main(split: str, out: str) -> None:
    d = slb.load(split)
    d = d[d.species == "human"].reset_index(drop=True)
    d["key"] = slb.pair_key(d.gene_a, d.gene_b)
    f = feature_table()
    x = f.reindex(d.key.values).reset_index(drop=True)
    cols = ["example_id", "context_id", "gene_a", "gene_b"] + (["label"] if "label" in d else [])
    res = pd.concat([d[cols], x], axis=1)
    res["has_features"] = x[FEATURES[0]].notna().values
    res.to_csv(out, index=False)
    print(f"{split}: {res.has_features.sum():,}/{len(res):,} human rows have paralog features", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
