"""Portable prior-mean inference for a multi-panel count world artifact.

Each panel retains its native query axis, control contexts, and library
denominator.  Cell counts and library size are deliberately absent from this
API: measurement exposure cannot change the forecast molecular mean.
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
    name = "slp_count_world_runtime_" + actual[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalize_actions(raw_action_features, feature_mean, feature_scale, feature_clip):
    """Apply the persisted shared fitting-action normalization."""
    raw = np.asarray(raw_action_features, dtype=np.float64)
    mean = np.asarray(feature_mean, dtype=np.float64)
    scale = np.asarray(feature_scale, dtype=np.float64)
    clip = float(feature_clip)
    if raw.ndim == 2:
        raw = raw[:, None, :]
    if (
        raw.ndim != 3
        or raw.shape[2] != len(mean)
        or scale.shape != mean.shape
        or not np.isfinite(raw).all()
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
        or not np.isfinite(clip)
        or clip <= 0
    ):
        raise ValueError("invalid raw action features or fitted normalizer")
    result = np.clip((raw - mean) / scale, -clip, clip).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("normalized action features are nonfinite")
    return result


class Predictor:
    """Load one arm and forecast on one registered source-native panel."""

    def __init__(
        self, artifact: str | Path, arm: str, panel: str, device: str = "cpu"
    ):
        root = Path(artifact)
        protocol_path = root / "protocol.json"
        manifest_path = root / "artifact-manifest.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _sha256(protocol_path) != manifest["protocolSha256"]:
            raise ValueError("protocol checksum mismatch")
        if arm not in manifest["arms"] or panel not in manifest["panels"]:
            raise ValueError("unknown arm or panel")
        hashes = manifest["sha256"]
        model_name = manifest["arms"][arm]["modelPath"]
        reference_name = manifest["panels"][panel]["referencePath"]
        for name in (model_name, reference_name):
            if _sha256(root / name) != hashes[name]:
                raise ValueError(f"artifact checksum mismatch: {name}")
        source_name = "source/count_latent_state.py"
        core = _load_source(root / source_name, hashes[source_name])
        self.device = torch.device(device)
        self.model = core.CountLatentState(core.Config(**protocol["modelConfig"]))
        self.model.load_state_dict(load_file(str(root / model_name)))
        self.model.to(self.device).eval()
        with np.load(root / reference_name, allow_pickle=False) as archive:
            self.reference = {key: np.array(archive[key], copy=True) for key in archive.files}
        self.panel = panel
        self.query_ids = self.reference["query_ids"].astype(str)
        self.context_ids = self.reference["context_ids"].astype(str)
        self._validate_reference()

    def _validate_reference(self) -> None:
        ref = self.reference
        q, f = ref["query_features"].shape
        c = len(ref["context_ids"])
        if (
            q != len(ref["query_ids"])
            or f != self.model.config.feature_dim
            or ref["feature_mean"].shape != (f,)
            or ref["feature_scale"].shape != (f,)
            or ref["basal_rate"].shape != (c, q)
            or not np.isfinite(ref["query_features"]).all()
            or not np.isfinite(ref["feature_mean"]).all()
            or not np.isfinite(ref["feature_scale"]).all()
            or np.any(ref["feature_scale"] <= 0)
            or not np.isfinite(ref["basal_rate"]).all()
            or np.any(ref["basal_rate"] <= 0)
            or len(set(self.query_ids.tolist())) != q
            or len(set(self.context_ids.tolist())) != c
        ):
            raise ValueError("invalid registered panel reference")

    @torch.no_grad()
    def predict(
        self,
        raw_action_features,
        context_weights,
        *,
        action_mask=None,
        query_indices=None,
        chunk_size: int = 1024,
    ) -> dict[str, np.ndarray]:
        """Return the prior expected CP10k mixture and its ln1p transform."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        actions = normalize_actions(
            raw_action_features,
            self.reference["feature_mean"],
            self.reference["feature_scale"],
            self.reference["feature_clip"],
        )
        batch = len(actions)
        if action_mask is None:
            mask = np.ones(actions.shape[:2], dtype=np.bool_)
        else:
            mask = np.asarray(action_mask)
        if mask.shape != actions.shape[:2] or mask.dtype != np.bool_:
            raise ValueError("action_mask must be Boolean [B,A]")
        actions = np.where(mask[..., None], actions, 0).astype(np.float32)
        # Always own the normalization buffer. ``np.asarray`` can alias a
        # caller's float64 array and an in-place division would mutate it.
        weights = np.array(context_weights, dtype=np.float64, copy=True)
        contexts = len(self.context_ids)
        if (
            weights.shape != (batch, contexts)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or np.any(weights.sum(1) <= 0)
        ):
            raise ValueError("context_weights must be finite nonnegative [B,C]")
        weights /= weights.sum(1, keepdims=True)
        if query_indices is None:
            selected = np.arange(len(self.query_ids), dtype=np.int64)
        else:
            selected = np.asarray(query_indices)
        if (
            selected.ndim != 1
            or not np.issubdtype(selected.dtype, np.integer)
            or np.any(selected < 0)
            or np.any(selected >= len(self.query_ids))
        ):
            raise ValueError("query_indices must be valid one-dimensional integers")
        selected = selected.astype(np.int64, copy=False)

        def tensor(value):
            return torch.as_tensor(value, device=self.device)

        query = tensor(self.reference["query_features"])
        basal = tensor(self.reference["basal_rate"])
        basal_mask = torch.ones_like(basal, dtype=torch.bool)
        context = self.model.encode_context(query, basal, basal_mask)
        prior = self.model.prior_from_context(
            tensor(np.repeat(actions, contexts, axis=0)),
            tensor(np.repeat(mask, contexts, axis=0)),
            context.repeat(batch, 1),
        )
        pieces = []
        for left in range(0, len(selected), chunk_size):
            index = selected[left : left + chunk_size]
            local = self.reference["basal_rate"][:, index]
            local = np.broadcast_to(local, (batch, contexts, len(index)))
            rate = self.model.population_mean(
                prior,
                query[tensor(index)],
                tensor(local.reshape(batch * contexts, len(index)).copy()),
            ).reshape(batch, contexts, len(index))
            pieces.append((rate * tensor(weights[..., None])).sum(1).cpu().numpy())
        mean = (
            np.concatenate(pieces, axis=1)
            if pieces
            else np.empty((batch, 0), dtype=np.float32)
        )
        return {
            "query_ids": self.query_ids[selected].copy(),
            "mean_cp10k": mean,
            "mean_log1p_cp10k": np.log1p(mean),
            "prior_mean": prior["mean"].reshape(batch, contexts, -1).cpu().numpy(),
            "prior_logvar": prior["logvar"].reshape(batch, contexts, -1).cpu().numpy(),
        }


__all__ = ["Predictor", "normalize_actions"]
