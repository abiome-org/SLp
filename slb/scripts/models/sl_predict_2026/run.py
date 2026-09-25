"""Frozen SL-Predict MAE branch with a clean SLB-trained pair classifier.

This uses the authors' 256-dimensional gene encoder and symmetric sum/absolute
difference pair features. The released SynLethDB classifier is not used.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import slb  # noqa: E402

EMB = slb.RAW / "sl_predict_2026/gene_embeddings.parquet"
PROFILE = slb.RAW / "sl_predict_2026/dependency_zscore.npy"
CHECKPOINT = slb.ROOT / "external/models/sl_predict_2026/_slb" / slb.BENCH.name / "mae_pair_lgbm.joblib"
TRAIN_CONFIG = dict(n_estimators=600, learning_rate=.05, num_leaves=63,
                    min_child_samples=20, subsample=.9, colsample_bytree=.9,
                    random_state=42, verbosity=-1, class_weight="balanced")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def vectors() -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    d = pd.read_parquet(EMB)
    z = d.filter(like="dim_").to_numpy(dtype=np.float32)
    if z.shape[1] != 256:
        raise ValueError("expected 256-dimensional author MAE vectors")
    resolve = slb.symbol_resolver()
    names = [resolve(x) or x for x in d.gene]
    counts = pd.Series(names).value_counts()
    index = {n: i for i, n in enumerate(names) if counts[n] == 1}
    profile = np.load(PROFILE, mmap_mode="r")
    if profile.shape != (len(z), 1206):
        raise ValueError(f"dependency profile misaligned with vectors: {profile.shape}")
    return index, z, profile


def pair_features(d: pd.DataFrame, index: dict[str, int], z: np.ndarray,
                  profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ai = d.gene_a.map(index).fillna(-1).to_numpy(dtype=np.int32)
    bi = d.gene_b.map(index).fillna(-1).to_numpy(dtype=np.int32)
    ok = (ai >= 0) & (bi >= 0)
    a, b = z[ai[ok]], z[bi[ok]]
    # Symmetric MAE features and the authors' raw Chronos coessentiality.
    coess = np.empty(len(a), dtype=np.float32)
    valid_a, valid_b = ai[ok], bi[ok]
    for start in range(0, len(a), 4096):
        end = min(start + 4096, len(a))
        coess[start:end] = (profile[valid_a[start:end]] * profile[valid_b[start:end]]).mean(axis=1)
    out = np.column_stack([a + b, np.abs(a - b), coess]).astype(np.float32)
    return out, ok


def train_model(d: pd.DataFrame, x: np.ndarray, y: np.ndarray) -> lgb.LGBMClassifier:
    rng = np.random.default_rng(42)
    genes = np.array(sorted(set(d.gene_a) | set(d.gene_b)))
    held = set(rng.choice(genes, max(1, int(.15 * len(genes))), replace=False))
    a = d.gene_a.isin(held).to_numpy()
    b = d.gene_b.isin(held).to_numpy()
    tr, va = ~a & ~b, a & b
    kwargs = {**TRAIN_CONFIG, "n_jobs": int(os.environ.get("SLB_THREADS", "8"))}
    rounds = 600
    if va.sum() >= 100 and 0 < y[va].sum() < va.sum() and y[tr].sum() > 0:
        trial = lgb.LGBMClassifier(**kwargs)
        trial.fit(x[tr], y[tr], eval_X=x[va], eval_y=y[va], eval_metric="auc",
                  callbacks=[lgb.early_stopping(75, verbose=False)])
        rounds = max(50, trial.best_iteration_)
        print(f"internal gene-holdout AUROC={roc_auc_score(y[va], trial.predict_proba(x[va])[:, 1]):.4f}; rounds={rounds}")
    kwargs["n_estimators"] = rounds
    model = lgb.LGBMClassifier(**kwargs)
    model.fit(x, y)
    return model


def load_or_train(checkpoint: Path, variant: str, train: pd.DataFrame,
                  x: np.ndarray, y: np.ndarray) -> lgb.LGBMClassifier:
    """Reuse a classifier only with identical inputs, settings and source code."""
    meta = checkpoint.with_suffix(".json")
    fingerprint = {
        "variant": variant,
        "benchmark_manifest_sha256": digest(slb.BENCH / "manifest.json"),
        "train_parquet_sha256": digest(slb.BENCH / "train.parquet"),
        "hgnc_sha256": digest(slb.RAW / "ids/hgnc_complete_set.txt"),
        "extracted_features_metadata_sha256": digest(EMB.with_suffix(".json")),
        "train_feature_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "train_label_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
        "run_source_sha256": digest(Path(__file__)),
        "training_config": {**TRAIN_CONFIG, "n_jobs": int(os.environ.get("SLB_THREADS", "8")),
                            "holdout_fraction": .15, "early_stopping_patience": 75,
                            "min_final_rounds": 50, "selection_metrics": "LightGBM default binary_logloss and auc"},
    }
    if checkpoint.exists() and meta.exists():
        saved = json.loads(meta.read_text())
        if (all(saved.get(k) == v for k, v in fingerprint.items())
                and saved.get("classifier_sha256") == digest(checkpoint)):
            print(f"verified cached classifier {checkpoint}")
            return joblib.load(checkpoint)
    classifier = train_model(train, x, y)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, checkpoint)
    meta.write_text(json.dumps({**fingerprint, "classifier_sha256": digest(checkpoint)}, indent=2))
    print(f"trained on {len(y):,} SLB human train pairs ({y.sum():,} positives)")
    return classifier


def main() -> None:
    split = slb.SPLIT
    index, z, profile = vectors()
    train = slb.load("train")
    train = train[train.species == "human"].reset_index(drop=True)
    x, ok = pair_features(train, index, z, profile)
    y = train.label.to_numpy(dtype=np.int8)[ok]
    model = load_or_train(CHECKPOINT, "mae_plus_coessentiality",
                          train.loc[ok].reset_index(drop=True), x, y)
    target = slb.load(split).copy()
    target["score"] = np.nan
    human = target[target.species == "human"]
    x, ok = pair_features(human, index, z, profile)
    scores = np.full(len(human), np.nan)
    scores[ok] = model.predict_proba(x)[:, 1]
    target.loc[human.index, "score"] = scores
    out = slb.write(target, "sl_predict_2026__mae", split)
    out.with_suffix(".provenance.json").write_text(json.dumps({
        "source": "j8ckfi/sl-predict @ 312974f",
        "encoder": "released leak-repaired 256-dimensional DepMap 26Q1 MAE",
        "classifier": "LightGBM refitted on SLB train pairs; symmetric MAE and Chronos coessentiality features",
        "labels": f"{slb.BENCH.name}/train.parquet only",
        "scope": "MAE branch adaptation; not the full source feature set",
    }, indent=2))
    slb.evaluate("sl_predict_2026__mae", split)


if __name__ == "__main__":
    main()
