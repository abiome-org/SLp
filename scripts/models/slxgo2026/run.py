"""Score the reproducible GO-PCA branch of SLxGO on SLB.

The full released SLxGO model also uses an ANN pretrained to imitate SLant
network features. That checkpoint is excluded here because those targets can
carry genetic-interaction labels. This is an explicit ablation, not a full
SLxGO reproduction. The authors' PANTHER/GO-derived embeddings are retained,
so the variant is *possibly* leaky until its GO evidence is audited.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import slb  # noqa: E402


SOURCE = slb.ROOT / "external/models/slxgo2026/BioBERT"
CHECKPOINT = slb.ROOT / "external/models/slxgo2026/_slb" / slb.BENCH.name / "go_pca.joblib"


def gene_vectors() -> tuple[dict[str, int], np.ndarray]:
    go = pd.read_csv(SOURCE / "hum_go.csv", usecols=["Gene Name/Symbol"])
    e = pd.read_csv(SOURCE / "hum_embedded.csv").to_numpy(dtype=np.float32)
    if len(go) != len(e):
        raise ValueError("SLxGO GO and embedding rows differ")
    symbols = go["Gene Name/Symbol"].str.split(";", regex=False).str[1]
    usable = symbols.notna() & ~symbols.duplicated(keep=False)
    ids = {str(s): int(i) for i, s in symbols[usable].items()}
    return ids, e


def pair_matrix(df: pd.DataFrame, ids: dict[str, int], e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ai = df.gene_a.map(ids).fillna(-1).to_numpy(dtype=np.int32)
    bi = df.gene_b.map(ids).fillna(-1).to_numpy(dtype=np.int32)
    ok = (ai >= 0) & (bi >= 0)
    a, b = e[ai[ok]], e[bi[ok]]
    return np.concatenate((a, b), axis=1), ok


def main() -> None:
    split = slb.SPLIT
    train = slb.load("train")
    train = train.loc[train.species == "human"].reset_index(drop=True)
    ids, e = gene_vectors()
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT.exists():
        model = joblib.load(CHECKPOINT)
    else:
        X, ok = pair_matrix(train, ids, e)
        y = train.label.to_numpy(dtype=np.int8)[ok]
        # Authors augment with both orientations, then use a HistGBM classifier.
        # Fixed representative parameters avoid optimizing on an easy pair split.
        X = np.concatenate((X, np.concatenate((X[:, 50:], X[:, :50]), axis=1)), axis=0)
        y = np.concatenate((y, y))
        model = HistGradientBoostingClassifier(
            learning_rate=.05, max_iter=250, max_depth=5,
            min_samples_leaf=30, l2_regularization=.05,
            early_stopping=True, random_state=42,
        )
        model.fit(X, y)
        joblib.dump(model, CHECKPOINT)
        print(f"trained SLxGO GO-PCA on {ok.sum():,} human SLB train pairs")
    target = slb.load(split).copy()
    target["score"] = np.nan
    human = target.loc[target.species == "human"]
    if len(human):
        X, ok = pair_matrix(human, ids, e)
        p = (model.predict_proba(X)[:, 1] +
             model.predict_proba(np.concatenate((X[:, 50:], X[:, :50]), axis=1))[:, 1]) / 2
        scores = np.full(len(human), np.nan, dtype=float)
        scores[ok] = p
        target.loc[human.index, "score"] = scores
    out = slb.write(target, "slxgo2026__go_pca", split)
    (out.with_suffix(".provenance.json")).write_text(json.dumps({
        "method": "SLxGO GO-PCA branch; full ANN network branch omitted",
        "embeddings": "official BioBERT/hum_embedded.csv, PANTHER/GO provenance not evidence-filtered",
        "labels": f"{slb.BENCH.name}/train.parquet only",
        "model": "symmetric HistGradientBoostingClassifier; fixed hyperparameters",
        "leakage": "possibly; GO annotation evidence not audited",
    }, indent=2))
    slb.evaluate("slxgo2026__go_pca", split)


if __name__ == "__main__":
    main()
