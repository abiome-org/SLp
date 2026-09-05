"""Local research inference for a fitted native-panel rank-response model.

This predicts signed log1p molecular profiles.  It is not a count generator,
and it has no default or inferred context mixture.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_module(path: Path, expected: str):
    if _sha256(path) != expected:
        raise ValueError("numerical source checksum mismatch")
    name = "slp_reduced_rank_runtime_" + expected[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalize_weights(weights, rows: int, contexts: int) -> np.ndarray:
    """Validate and privately normalize caller-supplied GEM weights."""
    value = np.array(weights, dtype=np.float64, copy=True)
    if (
        value.shape != (rows, contexts)
        or not np.isfinite(value).all()
        or np.any(value < 0)
        or np.any(value.sum(1) <= 0)
    ):
        raise ValueError("GEM weights must be finite nonnegative [N,C] with positive rows")
    value /= value.sum(1, keepdims=True)
    return value


class ResearchPredictor:
    """Compose panel-native control anchors and feature-linear residuals."""

    def __init__(self, bundle: str | Path, source: str):
        root = Path(bundle)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != "slp.rank32-local-research-inference-bundle/v1":
            raise ValueError("unsupported local research inference bundle schema")
        if source not in manifest["sources"]:
            raise ValueError("unknown native source")
        self.root = root
        self.source = source
        hashes = manifest["sha256"]
        entry = manifest["sources"][source]
        for name in (entry["model"], entry["reference"], "static-actions.npz"):
            if _sha256(root / name) != hashes[name]:
                raise ValueError(f"bundle checksum mismatch: {name}")
        model_source = "source/response_model.py"
        response = _load_module(root / model_source, hashes[model_source])
        self.model = response.load(root / entry["model"])
        with np.load(root / entry["reference"], allow_pickle=False) as archive:
            self.reference = {name: np.asarray(archive[name]) for name in archive.files}
        with np.load(root / "static-actions.npz", allow_pickle=False) as archive:
            if str(archive["schema"]) != "slp.rank32-local-static-action-cache/v1":
                raise ValueError("unsupported static-action cache schema")
            self.action_ids = archive["entity_id"].astype(str)
            self.action_features = np.asarray(archive["feature_values"], np.float32)
            entity_taxon = np.asarray(archive["entity_taxon"])
        if entity_taxon.shape != (len(self.action_ids),) or np.any(entity_taxon != 9606):
            raise ValueError("static-action cache taxon axis is invalid")
        self._action_lookup = {value: row for row, value in enumerate(self.action_ids)}
        self.query_ids = self.reference["query_ids"].astype(str)
        self.context_ids = self.reference["context_ids"].astype(str)
        self.gem_group_ids = np.asarray(self.reference["gem_group_ids"])
        self._validate()

    def _validate(self) -> None:
        ref = self.reference
        q = len(self.query_ids)
        c = len(self.context_ids)
        if (
            str(ref.get("schema", ""))
            != "slp.rank32-local-native-control-reference/v1"
            or str(ref["source_id"]) != self.source
            or ref["basal_rate"].shape != (c, q)
            or len(self.gem_group_ids) != c
            or self.action_features.shape != (len(self.action_ids), len(self.model.feature_mean))
            or len(set(self.action_ids.tolist())) != len(self.action_ids)
            or len(set(self.query_ids.tolist())) != q
            or len(set(self.context_ids.tolist())) != c
            or len(set(self.gem_group_ids.tolist())) != c
            or not np.isfinite(ref["basal_rate"]).all()
            or np.any(ref["basal_rate"] <= 0)
            or not np.isfinite(self.action_features).all()
        ):
            raise ValueError("invalid native-panel inference reference")
        model_path = self.root / json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )["sources"][self.source]["model"]
        with np.load(model_path, allow_pickle=False) as archive:
            if (
                str(archive["source_id"]) != self.source
                or not np.array_equal(archive["query_ids"].astype(str), self.query_ids)
            ):
                raise ValueError("model and reference query identities differ")

    def _queries(self, query_indices):
        selected = (
            np.arange(len(self.query_ids), dtype=np.int64)
            if query_indices is None
            else np.asarray(query_indices)
        )
        if (
            selected.ndim != 1
            or not np.issubdtype(selected.dtype, np.integer)
            or np.any(selected < 0)
            or np.any(selected >= len(self.query_ids))
        ):
            raise ValueError("query_indices must be valid one-dimensional integers")
        return selected.astype(np.int64, copy=False)

    def control(self, gem_weights, *, query_indices=None):
        """Return the explicit rate-mixture control anchor in log1p units."""
        weights = np.asarray(gem_weights)
        if weights.ndim != 2:
            raise ValueError("GEM weights must be two-dimensional")
        weights = normalize_weights(weights, len(weights), len(self.context_ids))
        selected = self._queries(query_indices)
        rate = weights @ self.reference["basal_rate"][:, selected].astype(np.float64)
        return {
            "query_ids": self.query_ids[selected].copy(),
            "control_rate_cp10k": rate,
            "control_log1p_cp10k": np.log1p(rate),
        }

    def predict_features(self, raw_features, gem_weights, *, query_indices=None):
        """Add an unmodified signed residual to the explicit control anchor."""
        features = np.asarray(raw_features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError("raw_features must be [N,F]")
        selected = self._queries(query_indices)
        control = self.control(gem_weights, query_indices=selected)
        if len(features) != len(control["control_log1p_cp10k"]):
            raise ValueError("feature and GEM-weight rows differ")
        residual = self.model.predict(features, selected)
        prediction = control["control_log1p_cp10k"] + residual
        if not np.isfinite(prediction).all():
            raise ValueError("molecular prediction is nonfinite")
        return {
            **control,
            "residual_log1p_profile": residual,
            "mean_log1p_cp10k": prediction,
        }

    def predict_genes(self, gene_ids, gem_weights, *, query_indices=None):
        """Convenience lookup through the frozen static-action cache."""
        genes = np.asarray(gene_ids).astype(str)
        if genes.ndim != 1 or len(set(genes.tolist())) != len(genes):
            raise ValueError("gene_ids must be unique one-dimensional stable IDs")
        missing = [gene for gene in genes if gene not in self._action_lookup]
        if missing:
            raise KeyError(f"stable gene absent from frozen action cache: {missing[0]}")
        rows = np.asarray([self._action_lookup[gene] for gene in genes], np.int64)
        result = self.predict_features(
            self.action_features[rows], gem_weights, query_indices=query_indices
        )
        result["gene_ids"] = genes.copy()
        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict a local research-only signed log1p molecular profile. This is not count generation."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source", choices=("k562", "rpe1"), required=True)
    parser.add_argument("--gene", required=True, help="Stable ENSG ID present in the frozen action cache")
    parser.add_argument(
        "--gem-weights",
        required=True,
        help="Comma-separated nonnegative weights in the bundle's printed GEM order; no default is inferred",
    )
    parser.add_argument("--query-index", type=int, action="append")
    args = parser.parse_args()
    predictor = ResearchPredictor(args.bundle, args.source)
    weights = np.asarray([[float(value) for value in args.gem_weights.split(",")]])
    result = predictor.predict_genes(
        np.asarray([args.gene]), weights, query_indices=args.query_index
    )
    print(
        json.dumps(
            {
                "source": args.source,
                "gene": args.gene,
                "gemGroupOrder": predictor.gem_group_ids.tolist(),
                "queryIds": result["query_ids"].tolist(),
                "meanLog1pCp10k": result["mean_log1p_cp10k"][0].tolist(),
                "warning": "signed molecular profile; not counts or a new-context forecast",
            }
        )
    )


if __name__ == "__main__":
    main()
