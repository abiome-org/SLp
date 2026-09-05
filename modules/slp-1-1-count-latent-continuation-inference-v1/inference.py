"""Portable inference for paired continuations of a count-latent checkpoint.

The two registered arms share an unchanged numerical core and reference.  Cell
counts, library sizes, and fitting aggregate targets are deliberately absent
from this target-free forecast interface.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_source(path: Path, expected: str):
    actual = _sha256(path)
    if actual != expected:
        raise ValueError("numerical source checksum mismatch")
    name = "slp_count_latent_continuation_runtime_" + actual[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Predictor:
    """Reload one frozen continuation arm and its registered query axis."""

    def __init__(
        self,
        artifact: str | Path,
        arm: str,
        device: str = "cpu",
    ):
        root = Path(artifact)
        protocol_path = root / "protocol.json"
        manifest_path = root / "artifact-manifest.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _sha256(protocol_path) != manifest["protocolSha256"]:
            raise ValueError("protocol checksum mismatch")
        if arm not in manifest["arms"]:
            raise ValueError(f"unknown continuation arm: {arm}")
        self.arm = arm
        model_path = root / manifest["arms"][arm]["modelPath"]
        reference_path = root / "reference.npz"
        source_path = root / "source/count_latent_state.py"
        hashes = manifest["sha256"]
        for relative, path in (
            (manifest["arms"][arm]["modelPath"], model_path),
            ("reference.npz", reference_path),
            ("source/count_latent_state.py", source_path),
        ):
            if _sha256(path) != hashes[relative]:
                raise ValueError(f"artifact checksum mismatch: {relative}")
        core = _load_source(source_path, hashes["source/count_latent_state.py"])
        self.device = torch.device(device)
        self.model = core.CountLatentState(core.Config(**protocol["modelConfig"]))
        self.model.load_state_dict(load_file(str(model_path)))
        self.model.to(self.device).eval()
        with np.load(reference_path, allow_pickle=False) as values:
            self.reference = {
                key: np.array(values[key], copy=True) for key in values.files
            }
        self.query_ids = self.reference["query_ids"].astype(str)
        self.gem_group_ids = np.asarray(self.reference["gem_group_ids"])
        self.feature_clip = float(self.reference["feature_clip"])
        self._validate_reference()

    def _validate_reference(self) -> None:
        ref = self.reference
        q, f = ref["query_features"].shape
        g = len(ref["gem_group_ids"])
        invalid = (
            q != len(ref["query_ids"])
            or f != self.model.config.feature_dim
            or ref["feature_mean"].shape != (f,)
            or ref["feature_scale"].shape != (f,)
            or ref["basal_rate"].shape != (g, q)
            or ref["basal_observed"].shape != (g, q)
            or ref["basal_observed"].dtype != np.bool_
            or not np.isfinite(ref["query_features"]).all()
            or not np.isfinite(ref["feature_mean"]).all()
            or not np.isfinite(ref["feature_scale"]).all()
            or np.any(ref["feature_scale"] <= 0)
            or not np.isfinite(ref["basal_rate"]).all()
            or np.any(ref["basal_rate"] <= 0)
            or not ref["basal_observed"].any(1).all()
            or not np.isfinite(self.feature_clip)
            or self.feature_clip <= 0
            or len(set(self.query_ids.tolist())) != q
            or len(set(self.gem_group_ids.tolist())) != g
        )
        if invalid:
            raise ValueError("invalid registered count-latent reference")

    @torch.no_grad()
    def predict(
        self,
        raw_action_features,
        gem_group_weights,
        *,
        action_mask=None,
        query_indices=None,
        chunk_size: int = 1024,
    ) -> dict[str, np.ndarray]:
        """Return expected CP10k and log1p(CP10k) for a GEM mixture."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        actions = np.asarray(raw_action_features, dtype=np.float32)
        if actions.ndim == 2:
            actions = actions[:, None, :]
        if actions.ndim != 3 or actions.shape[2] != self.model.config.feature_dim:
            raise ValueError("raw action features must be [B,F] or [B,A,F]")
        batch = len(actions)
        mask = (
            np.ones(actions.shape[:2], dtype=np.bool_)
            if action_mask is None
            else np.asarray(action_mask)
        )
        if mask.shape != actions.shape[:2] or mask.dtype != np.bool_:
            raise ValueError("action_mask must be Boolean [B,A]")
        safe = np.where(mask[..., None], actions, 0.0)
        if not np.isfinite(safe).all():
            raise ValueError("unmasked action features must be finite")
        safe = np.clip(
            (safe - self.reference["feature_mean"])
            / self.reference["feature_scale"],
            -self.feature_clip,
            self.feature_clip,
        ).astype(np.float32)
        if not np.isfinite(safe).all():
            raise ValueError("normalized action features must be finite")
        weights = np.asarray(gem_group_weights, dtype=np.float64)
        groups = len(self.gem_group_ids)
        if (
            weights.shape != (batch, groups)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or np.any(weights.sum(1) <= 0)
        ):
            raise ValueError("gem_group_weights must be finite nonnegative [B,G]")
        weights = weights / weights.sum(1, keepdims=True)
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
        selected = selected.astype(np.int64, copy=False)

        def tensor(values):
            return torch.as_tensor(values, device=self.device)

        query = tensor(self.reference["query_features"])
        basal = tensor(self.reference["basal_rate"])
        basal_mask = tensor(self.reference["basal_observed"])
        context = self.model.encode_context(query, basal, basal_mask)
        prior = self.model.prior_from_context(
            tensor(np.repeat(safe, groups, axis=0)),
            tensor(np.repeat(mask, groups, axis=0)),
            context.repeat(batch, 1),
        )
        pieces = []
        for left in range(0, len(selected), chunk_size):
            index = selected[left : left + chunk_size]
            local_basal = np.broadcast_to(
                self.reference["basal_rate"][:, index],
                (batch, groups, len(index)),
            )
            local = self.model.population_mean(
                prior,
                query[tensor(index)],
                tensor(local_basal.reshape(batch * groups, len(index)).copy()),
            ).reshape(batch, groups, len(index))
            pieces.append((local * tensor(weights[..., None])).sum(1).cpu().numpy())
        cp10k = (
            np.concatenate(pieces, axis=1)
            if pieces
            else np.empty((batch, 0), dtype=np.float32)
        )
        return {
            "query_ids": self.query_ids[selected].copy(),
            "mean_cp10k": cp10k,
            "mean_log1p_cp10k": np.log1p(cp10k),
            "prior_mean": prior["mean"].reshape(batch, groups, -1).cpu().numpy(),
            "prior_logvar": prior["logvar"].reshape(batch, groups, -1).cpu().numpy(),
        }
