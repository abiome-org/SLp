"""Frozen two-phase MuSL readout for baseline and world-augmented features."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

SEEDS = (42, 432)
FOLDS = range(5)
BASELINE_BLOCKS = ("embed_pairs.npy", "pair_summary.npy", "public_relations.npy")
BASELINE_WIDTHS = (1032, 40, 9)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest_has_receipt(value, filename: str, digest: str) -> bool:
    if isinstance(value, dict):
        path = str(value.get("path", value.get("source", "")))
        claimed = str(value.get("sha256", value.get("digest", "")))
        if Path(path).name == filename and claimed == digest:
            return True
        for key, child in value.items():
            if Path(str(key)).name == filename:
                if str(child) == digest:
                    return True
                if isinstance(child, dict) and str(child.get("sha256", child.get("digest", ""))) == digest:
                    return True
            if _manifest_has_receipt(child, filename, digest):
                return True
    elif isinstance(value, list):
        return any(_manifest_has_receipt(x, filename, digest) for x in value)
    return False


class BaselineFeatures:
    """Memory-mapped feature blocks, concatenated only for requested fold rows."""
    def __init__(self, path: Path, rows: int):
        self.path = Path(path)
        if self.path.is_dir():
            self.paths = [self.path / name for name in BASELINE_BLOCKS]
        else:
            self.paths = [self.path]
        self.blocks = [np.load(p, mmap_mode="r") for p in self.paths]
        if any(x.ndim != 2 or x.shape[0] != rows for x in self.blocks):
            raise ValueError("baseline rows do not match canonical roster")
        if self.path.is_dir() and tuple(x.shape[1] for x in self.blocks) != BASELINE_WIDTHS:
            raise ValueError(f"expected baseline block widths {BASELINE_WIDTHS}, found "
                             f"{tuple(x.shape[1] for x in self.blocks)}")
        self.shape = (rows, sum(x.shape[1] for x in self.blocks))

    def __getitem__(self, rows):
        return np.concatenate([np.asarray(x[rows], np.float32) for x in self.blocks], axis=1)


def load_features(baseline_path: Path, world_path: Path, roster: Path):
    pair_ids = np.load(roster / "pairs.npz", allow_pickle=False)["pair_ids"].astype(str)
    baseline = BaselineFeatures(baseline_path, len(pair_ids))
    with np.load(world_path, allow_pickle=False) as pack:
        required = {"features", "pair_ids", "source_pair_roster_sha256", "source_start", "source_stop"}
        if not required.issubset(pack.files):
            raise ValueError(f"world NPZ missing alignment fields: {sorted(required-set(pack.files))}")
        world = np.asarray(pack["features"], np.float32)
        world_ids = pack["pair_ids"].astype(str)
        roster_digest = str(pack["source_pair_roster_sha256"].item())
        source_start = int(pack["source_start"].item())
        source_stop = int(pack["source_stop"].item())
    if baseline.shape[1] != 1081:
        raise ValueError(f"expected 1081 baseline features, found {baseline.shape[1]}")
    if world.ndim != 2 or world.shape[1] != 1054:
        raise ValueError(f"expected 1054 world features, found {world.shape}")
    if world.shape[0] != len(pair_ids) or world.ndim != 2:
        raise ValueError("world rows do not match canonical roster")
    if world_ids.shape != pair_ids.shape or not np.array_equal(world_ids, pair_ids):
        raise ValueError("world pair_ids are not exactly aligned to canonical roster")
    if roster_digest != sha256(roster / "pairs.npz") or source_start != 0 or source_stop != len(pair_ids):
        raise ValueError("world source roster digest/range does not match canonical roster")
    if baseline_path.is_dir():
        baseline_manifest = json.loads((baseline_path / "manifest.json").read_text(encoding="utf-8"))
        declared_width = baseline_manifest.get("featureDim", baseline_manifest.get("feature_dim"))
        if declared_width is not None and int(declared_width) != 1081:
            raise ValueError(f"baseline manifest feature dimension is {declared_width}, expected 1081")
        expected = sha256(roster / "manifest.json")
        if not _manifest_has_receipt(baseline_manifest.get("inputs", {}), "manifest.json", expected):
            raise ValueError("baseline manifest does not receipt the canonical roster manifest")
    if not np.isfinite(world).all():
        raise ValueError("world features contain non-finite values")
    return pair_ids, baseline, world


def label_vector(path: Path, fold: int) -> np.ndarray:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, (list, tuple)) or len(value) != 5:
        raise ValueError(f"expected five label folds in {path}")
    labels = np.asarray(value[fold]).reshape(-1)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary")
    return labels.astype(np.uint8)


def inner_split(pair_ids: np.ndarray, seed: int, fold: int):
    genes = np.unique(pair_ids.reshape(-1))
    held = {g for g in genes if hashlib.sha256(f"{seed}:{fold}:{g}".encode()).digest()[0] < 51}
    a = np.fromiter((x in held for x in pair_ids[:, 0]), bool, len(pair_ids))
    b = np.fromiter((x in held for x in pair_ids[:, 1]), bool, len(pair_ids))
    return ~(a | b), a & b, held


def assert_outer_gene_disjoint(pair_ids: np.ndarray, train_indices: np.ndarray,
                               test_indices: np.ndarray):
    train_genes = set(pair_ids[train_indices].reshape(-1))
    test_genes = set(pair_ids[test_indices].reshape(-1))
    overlap = train_genes & test_genes
    if overlap:
        raise ValueError(f"outer split is not stable-gene-disjoint ({len(overlap)} overlaps)")
    return len(train_genes), len(test_genes)


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    y = np.asarray(y, np.float64)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log1p(-p))))


def trapezoidal_pr_auc(y: np.ndarray, prediction: np.ndarray) -> float:
    from sklearn.metrics import auc, precision_recall_curve
    precision, recall, _ = precision_recall_curve(y, prediction)
    return float(auc(recall, precision))


def model_params() -> dict:
    return dict(objective="binary", num_leaves=31, learning_rate=.03,
                feature_fraction=.7, lambda_l2=1., min_child_samples=30,
                deterministic=True, force_col_wise=True, seed=123,
                num_threads=4, verbosity=-1)


def _fit_lgb(X_fit, y_fit, X_val, y_val):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(n_estimators=1000, **model_params())
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model, int(model.best_iteration_)


def _refit_lgb(X, y, trees: int):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(n_estimators=trees, **model_params())
    model.fit(X, y)
    return model


def fit(args) -> None:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    pair_ids, baseline, world = load_features(args.baseline, args.world, args.roster)
    source_dir = Path(__file__).resolve().parent
    baseline_files = baseline.paths + ([args.baseline / "manifest.json"] if args.baseline.is_dir() else [])
    input_files = [*baseline_files, args.world, args.roster / "pairs.npz",
                   args.roster / "manifest.json", source_dir / "readout.py",
                   source_dir / "requirements.lock", source_dir / "CONTRACT.md"]
    runs = []
    for seed in SEEDS:
        train_labels_path = args.labels / f"train_labels_seed{seed}.pkl"
        input_files.append(train_labels_path)
        for fold in FOLDS:
            split_path = args.roster / f"seed{seed}-fold{fold}.npz"
            input_files.append(split_path)
            with np.load(split_path, allow_pickle=False) as split:
                indices = split["pair_indices"].astype(np.int64)
                part = split["partition"]
                source = split["source_row"].astype(np.int64)
            train = part == 0
            test = part == 1
            train_idx = indices[train]
            test_idx = indices[test]
            train_gene_count, test_gene_count = assert_outer_gene_disjoint(
                pair_ids, train_idx, test_idx)
            labels_all = label_vector(train_labels_path, fold)
            train_source = source[train]
            if len(labels_all) <= int(train_source.max(initial=-1)):
                raise ValueError("training source_row exceeds label vector")
            y = labels_all[train_source]
            fit_mask, val_mask, held = inner_split(pair_ids[train_idx], seed, fold)
            if fit_mask.sum() == 0 or val_mask.sum() == 0:
                raise ValueError("empty gene-disjoint inner split")
            if len(np.unique(y[fit_mask])) < 2 or len(np.unique(y[val_mask])) < 2:
                raise ValueError("inner split lacks both classes")
            xb = baseline[train_idx]
            xw = np.concatenate((np.asarray(xb, np.float32), world[train_idx]), axis=1)
            mb, tb = _fit_lgb(xb[fit_mask], y[fit_mask], xb[val_mask], y[val_mask])
            ma, ta = _fit_lgb(xw[fit_mask], y[fit_mask], xw[val_mask], y[val_mask])
            pb = mb.predict_proba(xb[val_mask])[:, 1]
            pa = ma.predict_proba(xw[val_mask])[:, 1]
            weights = np.linspace(0., 1., 21)
            losses = [logloss(y[val_mask], (1-w)*pb + w*pa) for w in weights]
            weight = float(weights[int(np.argmin(losses))])
            mb = _refit_lgb(xb, y, tb)
            ma = _refit_lgb(xw, y, ta)
            prefix = f"seed{seed}-fold{fold}"
            mb.booster_.save_model(str(output / f"{prefix}-baseline.txt"))
            ma.booster_.save_model(str(output / f"{prefix}-augmented.txt"))
            test_b = baseline[test_idx]
            test_a = np.concatenate((np.asarray(test_b, np.float32), world[test_idx]), axis=1)
            pred_b = mb.predict_proba(test_b)[:, 1].astype(np.float64)
            pred_a = ma.predict_proba(test_a)[:, 1].astype(np.float64)
            np.savez_compressed(output / f"{prefix}-predictions.npz",
                pair_indices=test_idx, source_row=source[test], pair_ids=pair_ids[test_idx],
                baseline=pred_b, augmented=pred_a,
                blend=(1-weight)*pred_b + weight*pred_a)
            runs.append(dict(seed=seed, fold=fold, outer_train=int(train.sum()),
                outer_test=int(test.sum()), inner_fit=int(fit_mask.sum()),
                inner_validation=int(val_mask.sum()), inner_held_genes=len(held),
                outer_train_genes=train_gene_count, outer_test_genes=test_gene_count,
                baseline_trees=tb, augmented_trees=ta, augmented_blend_weight=weight,
                inner_logloss=float(min(losses))))
            print(json.dumps(dict(event="fit_fold", seed=seed, fold=fold,
                baseline_trees=tb, augmented_trees=ta, blend=weight,
                inner_logloss=float(min(losses)), ntrain=int(train.sum()),
                ntest=int(test.sum())), sort_keys=True), flush=True)
    params = dict(schema="slp.sl-readout-fit/v1", model="LightGBM", params=model_params(),
                  early_stopping_rounds=50, max_trees=1000,
                  inner_split="sha256(seed:fold:stable_gene_id)[0] < 51; fit=both nonheld; validation=both held",
                  blend_grid=[float(x) for x in np.linspace(0., 1., 21)], runs=runs)
    _json(output / "params.json", params)
    generated = sorted(output.glob("*.txt")) + sorted(output.glob("*.npz")) + [output / "params.json"]
    manifest = dict(schema="slp.sl-readout-lock/v1", phase="fit-before-test-label-access",
        inputs={str(p.resolve()): sha256(p) for p in sorted(set(input_files))},
        artifacts={p.name: sha256(p) for p in generated}, test_labels_accessed=False)
    _json(output / "fit-manifest.json", manifest)


def score(args) -> None:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    output = args.output.resolve()
    manifest_path = output / "fit-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["artifacts"].items():
        if sha256(output / name) != expected:
            raise ValueError(f"locked fit artifact changed: {name}")
    score_path = output / "scores.json"
    if score_path.exists():
        raise FileExistsError(score_path)
    rows = []
    for seed in SEEDS:
        test_path = args.labels / f"test_labels_seed{seed}.pkl"
        for fold in FOLDS:
            pred_path = output / f"seed{seed}-fold{fold}-predictions.npz"
            with np.load(pred_path, allow_pickle=False) as p:
                source = p["source_row"].astype(np.int64)
                forecasts = {k: p[k].astype(np.float64) for k in ("baseline", "augmented", "blend")}
            all_labels = label_vector(test_path, fold)
            if len(all_labels) <= int(source.max(initial=-1)):
                raise ValueError("test source_row exceeds label vector")
            y = all_labels[source]
            prevalence = float(y.mean())
            positive_count = int(y.sum())
            for arm, prediction in forecasts.items():
                prevalence_call = np.zeros(len(y), dtype=bool)
                if positive_count:
                    chosen = np.argsort(-prediction, kind="stable")[:positive_count]
                    prevalence_call[chosen] = True
                rows.append(dict(seed=seed, fold=fold, arm=arm, n=len(y), positives=int(y.sum()),
                    auroc=float(roc_auc_score(y, prediction)), ap=float(average_precision_score(y, prediction)),
                    trapezoidal_pr_auc=trapezoidal_pr_auc(y, prediction),
                    f1_at_0_5=float(f1_score(y, prediction >= .5)),
                    f1_at_prevalence=float(f1_score(y, prevalence_call)),
                    prevalence=prevalence))
    macro = []
    for arm in ("baseline", "augmented", "blend"):
        selected = [r for r in rows if r["arm"] == arm]
        macro.append(dict(arm=arm, folds=len(selected),
            auroc=float(np.mean([r["auroc"] for r in selected])),
            ap=float(np.mean([r["ap"] for r in selected])),
            trapezoidal_pr_auc=float(np.mean([r["trapezoidal_pr_auc"] for r in selected])),
            f1_at_0_5=float(np.mean([r["f1_at_0_5"] for r in selected])),
            f1_at_prevalence=float(np.mean([r["f1_at_prevalence"] for r in selected]))))
    _json(score_path, dict(schema="slp.sl-readout-score/v1", fit_manifest_sha256=sha256(manifest_path),
        test_label_receipts={str((args.labels / f'test_labels_seed{s}.pkl').resolve()):
                             sha256(args.labels / f'test_labels_seed{s}.pkl') for s in SEEDS},
        rows=rows, macro_across_ten_folds=macro))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("fit", "score"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--world", type=Path)
    parser.add_argument("--roster", type=Path, default=root / "data/derived/slp11-musl-world-pair-roster-v1")
    parser.add_argument("--labels", type=Path, default=root / "data/models/MuSL/processed_data/data/CV3_bins_32/fold_data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "fit":
        if args.baseline is None or args.world is None:
            parser.error("fit requires --baseline and --world")
        fit(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
