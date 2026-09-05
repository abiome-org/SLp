"""Inference for the Norman observed-background compositional-state pilot.

Importing this module disables PyTorch's global fused multi-head-attention
fastpath.  PyTorch 2.11 produced materially different CPU and CUDA results for
the fused evaluation path; the unfused path is used for artifact replay.

This interface accepts observed single-intervention endpoints on the exact
stored query axis.  Values are control-standardized RNA pseudobulk endpoints,
not raw counts.  It makes no claim about sequential time-course dynamics.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


torch.backends.mha.set_fastpath_enabled(False)


def _load_core(path: Path):
    name = "slp11_compositional_operator_inference_core"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Predictor:
    """A frozen fold/seed observed-background molecular predictor."""

    def __init__(self, model, basis, zscale, decoder, feature_mean, feature_scale, query_ids):
        self.model = model
        self.basis = basis
        self.zscale = zscale
        self.decoder = decoder
        self.feature_mean = feature_mean
        self.feature_scale = feature_scale
        self.query_ids = query_ids

    def _endpoint(self, value, name):
        result = np.asarray(value, dtype=np.float32)
        if result.shape != (self.basis.shape[1],) or not np.isfinite(result).all():
            raise ValueError(f"{name} must be one finite endpoint on the stored query axis")
        return result

    def _feature(self, value, name):
        # Match the runner exactly: normalize in float64, then cast the action
        # tensor to float32 at the model boundary.
        result = np.asarray(value, dtype=np.float64)
        if result.shape != self.feature_mean.shape or not np.isfinite(result).all():
            raise ValueError(f"{name} must be one finite raw 577-dimensional static feature vector")
        return (result - self.feature_mean) / self.feature_scale

    def predict(self, y_a, y_b, raw_features_a, raw_features_b):
        """Predict a double endpoint from two observed single endpoints.

        ``y_a`` and ``y_b`` are control-standardized RNA pseudobulk vectors in
        exactly ``query_ids`` order.  Static features are raw with respect to
        the fold's stored feature normalization (the caller does not scale them).
        """
        ya, yb = self._endpoint(y_a, "y_a"), self._endpoint(y_b, "y_b")
        fa = self._feature(raw_features_a, "raw_features_a")
        fb = self._feature(raw_features_b, "raw_features_b")
        za = torch.from_numpy(((ya @ self.basis.T) / self.zscale)[None].astype(np.float32))
        zb = torch.from_numpy(((yb @ self.basis.T) / self.zscale)[None].astype(np.float32))
        zero = torch.zeros_like(za)
        mask = torch.tensor([[True, False]])

        def action(feature):
            result = torch.zeros((1, 2, self.feature_mean.size), dtype=torch.float32)
            result[0, 0] = torch.from_numpy(feature.astype(np.float32, copy=False))
            return result

        aa, ab = action(fa), action(fb)
        with torch.no_grad():
            increment = 0.5 * (
                (self.model(za, ab, mask) - za) - self.model(zero, ab, mask)
                + (self.model(zb, aa, mask) - zb) - self.model(zero, aa, mask)
            )
        return ya.astype(np.float64) + yb.astype(np.float64) + increment.numpy()[0] @ self.decoder


def load(run_dir: str | Path, fold: int, seed: int) -> Predictor:
    """Load one frozen ``observed_operator`` checkpoint on CPU."""
    run_dir = Path(run_dir)
    protocol_path = run_dir / "protocol.json"
    basis_path = run_dir / f"fold{fold}-basis.npz"
    checkpoint_path = run_dir / f"fold{fold}-observed_operator-seed{seed}.safetensors"
    if not protocol_path.is_file() or not basis_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("protocol, fold basis, or observed-operator checkpoint is missing")
    protocol = json.loads(protocol_path.read_text())
    if seed not in protocol.get("seeds", []):
        raise ValueError("seed is not declared by the run protocol")
    core = _load_core(Path(__file__).with_name("operator.py"))
    allowed = {field.name for field in fields(core.Config)}
    supplied = protocol.get("config", {})
    if set(supplied) - allowed:
        raise ValueError("protocol contains unsupported operator configuration")
    config = core.Config(**supplied)
    model = core.CompositionalStateOperator(config)
    model.load_state_dict(load_file(str(checkpoint_path), device="cpu"))
    model.eval()
    with np.load(basis_path, allow_pickle=False) as archive:
        required = {"basis", "zscale", "feature_mean", "feature_scale", "query_ids"}
        if not required.issubset(archive.files):
            raise ValueError("fold basis artifact is incomplete")
        basis = archive["basis"].astype(np.float64)
        zscale = archive["zscale"].astype(np.float64)
        feature_mean = archive["feature_mean"].astype(np.float64)
        feature_scale = archive["feature_scale"].astype(np.float64)
        query_ids = archive["query_ids"].astype(str)
    if basis.shape != (config.state_dim, len(query_ids)) or zscale.shape != (config.state_dim,):
        raise ValueError("fold state coordinates disagree with the operator config")
    if feature_mean.shape != (config.action_dim,) or feature_scale.shape != feature_mean.shape:
        raise ValueError("fold feature coordinates disagree with the operator config")
    if not all(np.isfinite(x).all() for x in (basis, zscale, feature_mean, feature_scale)):
        raise ValueError("fold coordinates must be finite")
    if np.any(zscale <= 0) or np.any(feature_scale <= 0):
        raise ValueError("fold scales must be positive")
    decoder = zscale[:, None] * basis
    return Predictor(model, basis, zscale, decoder, feature_mean, feature_scale, query_ids)
