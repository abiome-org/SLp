"""Portable native-panel inference for matched response-query count models.

Callers provide static577 action features and explicit native GEM weights. The
adapter appends the 33 exact zero action coordinates used during training.
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
    if _sha256(path) != expected:
        raise ValueError("numerical source checksum mismatch")
    name = "slp_count_response_query_runtime_" + expected[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalize_static_actions(raw_static577, feature_mean610, feature_scale610, clip):
    """Append exact zero response coordinates and apply the saved normalizer."""
    raw = np.asarray(raw_static577, np.float64)
    mean = np.asarray(feature_mean610, np.float64)
    scale = np.asarray(feature_scale610, np.float64)
    if raw.ndim == 2:
        raw = raw[:, None, :]
    if (
        raw.ndim != 3
        or raw.shape[-1] != 577
        or mean.shape != (610,)
        or scale.shape != (610,)
        or not np.isfinite(raw).all()
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
        or not np.isfinite(clip)
        or float(clip) <= 0
        or not np.array_equal(mean[577:], np.zeros(33))
        or not np.array_equal(scale[577:], np.ones(33))
    ):
        raise ValueError("invalid static577 actions or persisted 610-wide normalizer")
    padded = np.concatenate((raw, np.zeros((*raw.shape[:-1], 33))), axis=-1)
    return np.clip((padded - mean) / scale, -float(clip), float(clip)).astype(np.float32)


class Predictor:
    """Load one matched arm and predict one registered native molecular panel."""

    def __init__(self, artifact: str | Path, arm: str, panel: str, device: str = "cpu"):
        root = Path(artifact)
        protocol_path = root / "protocol.json"
        manifest = json.loads((root / "artifact-manifest.json").read_text(encoding="utf-8"))
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if _sha256(protocol_path) != manifest["protocolSha256"]:
            raise ValueError("protocol checksum mismatch")
        if arm not in manifest["arms"] or panel not in manifest["arms"][arm]["panels"]:
            raise ValueError("unknown arm or native panel")
        hashes = manifest["sha256"]
        model_name = manifest["arms"][arm]["modelPath"]
        reference_name = manifest["arms"][arm]["panels"][panel]["referencePath"]
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
            self.reference = {name: np.array(archive[name], copy=True) for name in archive.files}
        self.panel = panel
        self.query_ids = self.reference["query_ids"].astype(str)
        self.context_ids = self.reference["context_ids"].astype(str)
        self._validate_reference(arm)

    def _validate_reference(self, arm: str) -> None:
        ref = self.reference
        q, f = ref["query_features"].shape
        c = len(ref["context_ids"])
        expected_mode = "static-zero33" if arm == "static-zero33" else "response33"
        if (
            str(ref["schema"]) != "slp.count-world-response-query-reference/v1"
            or str(ref["source_id"]) != self.panel
            or str(ref["feature_mode"]) != expected_mode
            or q != len(ref["query_ids"])
            or f != 610
            or f != self.model.config.feature_dim
            or ref["feature_mean"].shape != (f,)
            or ref["feature_scale"].shape != (f,)
            or ref["basal_rate"].shape != (c, q)
            or not all(
                np.isfinite(ref[name]).all()
                for name in ("query_features", "feature_mean", "feature_scale", "basal_rate")
            )
            or np.any(ref["feature_scale"] <= 0)
            or np.any(ref["basal_rate"] <= 0)
            or not np.array_equal(ref["feature_mean"][577:], np.zeros(33))
            or not np.array_equal(ref["feature_scale"][577:], np.ones(33))
            or len(set(self.query_ids.tolist())) != q
            or len(set(self.context_ids.tolist())) != c
        ):
            raise ValueError("invalid response-query native-panel reference")

    @torch.no_grad()
    def predict(
        self,
        raw_static577,
        context_weights,
        *,
        action_mask=None,
        query_indices=None,
        chunk_size: int = 1024,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        actions = normalize_static_actions(
            raw_static577,
            self.reference["feature_mean"],
            self.reference["feature_scale"],
            self.reference["feature_clip"],
        )
        batch = len(actions)
        mask = np.ones(actions.shape[:2], np.bool_) if action_mask is None else np.asarray(action_mask)
        if mask.shape != actions.shape[:2] or mask.dtype != np.bool_:
            raise ValueError("action_mask must be Boolean [B,A]")
        actions = np.where(mask[..., None], actions, 0).astype(np.float32)
        weights = np.array(context_weights, np.float64, copy=True)
        contexts = len(self.context_ids)
        if (
            weights.shape != (batch, contexts)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or np.any(weights.sum(1) <= 0)
        ):
            raise ValueError("context_weights must be finite nonnegative [B,C]")
        weights /= weights.sum(1, keepdims=True)
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

        def tensor(value):
            return torch.as_tensor(value, device=self.device)

        query = tensor(self.reference["query_features"])
        basal = tensor(self.reference["basal_rate"])
        context = self.model.encode_context(query, basal, torch.ones_like(basal, dtype=torch.bool))
        prior = self.model.prior_from_context(
            tensor(np.repeat(actions, contexts, axis=0)),
            tensor(np.repeat(mask, contexts, axis=0)),
            context.repeat(batch, 1),
        )
        pieces = []
        for left in range(0, len(selected), chunk_size):
            index = selected[left : left + chunk_size]
            local = np.broadcast_to(
                self.reference["basal_rate"][:, index], (batch, contexts, len(index))
            )
            rate = self.model.population_mean(
                prior,
                query[tensor(index)],
                tensor(local.reshape(batch * contexts, len(index)).copy()),
            ).reshape(batch, contexts, len(index))
            pieces.append((rate * tensor(weights[..., None])).sum(1).cpu().numpy())
        mean = np.concatenate(pieces, axis=1) if pieces else np.empty((batch, 0), np.float32)
        return {
            "query_ids": self.query_ids[selected].copy(),
            "mean_cp10k": mean,
            "mean_log1p_cp10k": np.log1p(mean),
            "prior_mean": prior["mean"].reshape(batch, contexts, -1).cpu().numpy(),
            "prior_logvar": prior["logvar"].reshape(batch, contexts, -1).cpu().numpy(),
        }


__all__ = ["Predictor", "normalize_static_actions"]
