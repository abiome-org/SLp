"""Fit one pooled multi-species model on one transfer arm and score the dev and test inputs.

The models are the two pooled SLB adapters, reused unchanged except for how often they train:
the adapters retrain for every split, while this fits once per arm and scores dev and test together.

  ontotype        scripts/models/ontotype pooled variant (shared GO-term ontotype, one LightGBM)
  gbm             scripts/models/go_ppi_gbm pooled variant, species code included (as on the leaderboard)
  gbm_nocode      the same without the species code: in H0 the human code is never seen in training,
                  so every arm of the transfer track uses this species-agnostic version
  gbm_xs          gbm_nocode with each gene-level feature (single-loss effect, PPI/STRING degree, number
                  of GO terms) replaced by the gene's percentile within its species. Raw scales differ by
                  species (DepMap Chronos vs yeast colony fitness; human PPI is far denser), which a model
                  trained without human rows cannot calibrate. Uses no labels.

GO/PPI pair features are computed once for every train, dev and test row of data/slb and cached in
data/transfer/_cache/; arms select their rows by example_id. Model selection uses only the adapters'
internal validation split of each arm's train rows (dev labels are never read).

    uv run python scripts/transfer/run_model.py --arm h0 --model gbm_xs
Writes results/transfer/<arm>/<model>_{dev,test}.parquet (example_id, score).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "scripts/models"
sys.path.insert(0, str(MODELS / "_common"))
sys.path.insert(0, str(MODELS / "go_ppi_gbm"))
# TRANSFER_BENCH selects the benchmark build (default data/slb; e.g. data/slb_next for a candidate build)
BENCH = ROOT / os.environ.get("TRANSFER_BENCH", "data/slb")
TAG = "" if BENCH.name == "slb" else f"_{BENCH.name}"
CACHE = ROOT / f"data/transfer/_cache{TAG}"
OUT = ROOT / "results/transfer"
SPLITS = ("dev", "test")
GENE_FEATS = {"fit_min": "fit", "fit_max": "fit", "phys_deg_min": "phys", "phys_deg_max": "phys",
              "string_deg_min": "strn", "string_deg_max": "strn", "go_n_min": "go", "go_n_max": "go"}

_SD = None


def _feat_chunk(d: pd.DataFrame) -> np.ndarray:
    return _SD.features(d)


def gbm_features() -> pd.DataFrame:
    """Raw GO/PPI features of every train/dev/test row of data/slb, plus per-species gene percentiles
    for the gene-level columns (suffix _pct). Cached."""
    global _SD
    p = CACHE / "go_ppi_features.parquet"
    if p.exists():
        return pd.read_parquet(p)
    import run as G  # scripts/models/go_ppi_gbm/run.py
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = pd.concat([pd.read_parquet(BENCH / "train.parquet", columns=["example_id", "species", "gene_a",
                                                                                   "gene_b", "same_family"])]
                     + [pd.read_parquet(BENCH / f"{s}_inputs.parquet", columns=["example_id", "species",
                                                                                         "gene_a", "gene_b", "same_family"])
                        for s in SPLITS + ("dev_semi", "test_semi") if (BENCH / f"{s}_inputs.parquet").exists()],
                     ignore_index=True)
    # species codes as in the adapter: index in the sorted species list of train + evaluated split
    allsp = sorted(rows.species.unique())
    parts = []
    for i, sp in enumerate(allsp):
        import bundle_io as B
        if not (B.has_bundle(sp, "go") or B.has_bundle(sp, "ppi")):
            continue
        t0 = time.time()
        _SD = G.SpeciesData(sp, i)
        d = rows[rows.species == sp].reset_index(drop=True)
        chunks = np.array_split(np.arange(len(d)), max(1, len(d) // 20000))
        with mp.get_context("fork").Pool(int(os.environ.get("SLB_PROCS", "16"))) as pool:
            x = np.vstack(pool.map(_feat_chunk, [d.iloc[c] for c in chunks]))
        f = pd.DataFrame(x, columns=G.FEAT)
        # gene-level percentiles within the species (bundle genes; labels not involved)
        pops = {"fit": np.array([v for v in _SD.fit.values() if not np.isnan(v)]),
                "phys": np.array([len(v) for v in _SD.phys.values()]),
                "strn": np.array([len(v) for v in _SD.strn.values()]),
                "go": np.array([len(v) for v in _SD.direct.values()])}
        for col, pop in GENE_FEATS.items():
            ref = np.sort(pops[pop]) if len(pops[pop]) else np.zeros(1)
            v = f[col].to_numpy()
            # genes absent from a network have degree 0; their percentile is the share of genes at 0
            pct = np.searchsorted(ref, v, side="right") / len(ref)
            f[col + "_pct"] = np.where(np.isnan(v), np.nan, pct).astype(np.float32)
        f.insert(0, "example_id", d.example_id.to_numpy())
        parts.append(f)
        print(f"features {sp}: {len(d):,} rows in {time.time() - t0:.0f}s", flush=True)
    out = pd.concat(parts, ignore_index=True).drop_duplicates("example_id")
    out.to_parquet(p, index=False)
    return out


def gbm_matrix(variant: str) -> pd.DataFrame:
    import run as G
    F = gbm_features().set_index("example_id")
    X = F[list(G.FEAT)].copy()
    if variant in ("gbm_nocode", "gbm_xs"):
        X["species_code"] = 0.0
    if variant == "gbm_xs":
        for c in GENE_FEATS:
            X[c] = F[c + "_pct"]
    return X


def run_gbm(tr: pd.DataFrame, ev: pd.DataFrame, variant: str, save: Path | None = None,
            drop: list[str] = (), seed: int = 0) -> pd.Series:
    """drop: feature columns set to a constant (used by ablate.py)."""
    import run as G
    X = gbm_matrix(variant)
    for c in drop:
        X[c] = 0.0
    tr = tr[tr.example_id.isin(X.index)]
    xt = X.loc[tr.example_id].to_numpy(np.float32)
    y = tr.label.to_numpy().astype(int)
    m, auc, r = G.fit(xt, y, (tr.species + "|" + tr.gene_a).to_numpy(), (tr.species + "|" + tr.gene_b).to_numpy(),
                      seed=seed)
    print(f"{variant}: internal-val AUROC {auc:.4f} rounds {r}", flush=True)
    if save:
        m.save_model(str(save))
    e = ev[ev.example_id.isin(X.index)]
    return pd.Series(m.predict(X.loc[e.example_id].to_numpy(np.float32)), index=e.example_id.to_numpy())


def run_ontotype(tr: pd.DataFrame, ev: pd.DataFrame) -> pd.Series:
    import importlib.util  # ontotype/run.py, loaded by path: `run` is already the go_ppi_gbm module
    spec = importlib.util.spec_from_file_location("ontotype_run", MODELS / "ontotype/run.py")
    O = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(O)
    r = O.pooled(tr, ev)
    return pd.Series(r.score.to_numpy(), index=r.example_id.to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", required=True, choices=["ontotype", "gbm", "gbm_nocode", "gbm_xs"])
    ap.add_argument("--seed", type=int, default=0, help="GBM seed (bagging and internal validation genes); "
                    "seed > 0 writes <model>_seed<N>, to measure how much refitting alone moves the score")
    a = ap.parse_args()
    name = a.model if a.seed == 0 else f"{a.model}_seed{a.seed}"
    arm = ROOT / "data/transfer" / a.arm
    tr = pd.read_parquet(arm / "train.parquet")
    tr = tr[tr.label.notna()]
    ev = {s: pd.read_parquet(arm / f"{s}_inputs.parquet") for s in SPLITS}
    both = pd.concat(ev.values(), ignore_index=True)
    t0 = time.time()
    if a.model == "ontotype":
        sc = run_ontotype(tr, both)
    else:
        (OUT / a.arm).mkdir(parents=True, exist_ok=True)
        sc = run_gbm(tr, both, a.model, save=OUT / a.arm / f"{name}.lgb.txt", seed=a.seed)
    (OUT / a.arm).mkdir(parents=True, exist_ok=True)
    for s, e in ev.items():
        # rows a model cannot score (species without a bundle) get that species' median; none expected
        d = pd.DataFrame({"example_id": e.example_id.to_numpy(), "score": sc.reindex(e.example_id).to_numpy()})
        med = d.groupby(e.species.to_numpy()).score.transform("median").fillna(0.0)
        d["score"] = d.score.fillna(med)
        d.to_parquet(OUT / a.arm / f"{name}_{s}.parquet", index=False)
    print(f"{a.arm}/{name}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
