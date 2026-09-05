"""Portable prior-only inference with a caller-supplied molecular context.

The adapter loads a frozen CountLatentState checkpoint, but replaces its
registered query/control axis with caller-supplied raw static query features,
positive control rates and masks.  Both query and action features use the
checkpoint's saved training transform.  No perturbed outcomes or libraries are
accepted by this API.
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


def _load_core(path: Path, expected: str):
    if _sha256(path) != expected:
        raise ValueError("local numerical core checksum mismatch")
    name = "slp11_count_prior_context_adapter_core"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ContextPriorPredictor:
    """Load one frozen prior and evaluate it against external control inputs."""

    def __init__(
        self,
        artifact: str | Path,
        *,
        freeze_receipt: str | Path,
        device: str = "cpu",
    ):
        root = Path(artifact)
        protocol_path = root / "protocol.json"
        manifest_path = root / "artifact-manifest.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _sha256(protocol_path) != manifest.get("protocolSha256"):
            raise ValueError("artifact protocol checksum mismatch")
        hashes = manifest.get("sha256", {})
        for relative in ("model.safetensors", "reference.npz", "source/count_latent_state.py"):
            if relative not in hashes or _sha256(root / relative) != hashes[relative]:
                raise ValueError(f"artifact member checksum mismatch: {relative}")
        freeze = json.loads(Path(freeze_receipt).read_text(encoding="utf-8"))
        if (
            freeze.get("modelSha256") != hashes["model.safetensors"]
            or freeze.get("referenceSha256") != hashes["reference.npz"]
            or freeze.get("originalProtocolSha256", freeze.get("protocolSha256"))
            != manifest["protocolSha256"]
            or freeze.get("developmentCountMembersOpened") is not False
            or freeze.get("testOpened") is not False
        ):
            raise ValueError("authoritative freeze receipt does not match artifact")
        local_core = Path(__file__).resolve().with_name("count_latent_state.py")
        core = _load_core(local_core, hashes["source/count_latent_state.py"])
        self.device = torch.device(device)
        self.model = core.CountLatentState(core.Config(**protocol["modelConfig"]))
        self.model.load_state_dict(load_file(str(root / "model.safetensors")))
        self.model.to(self.device).eval()
        with np.load(root / "reference.npz", allow_pickle=False) as archive:
            reference = {name: np.asarray(archive[name]) for name in archive.files}
        dimension = self.model.config.feature_dim
        self.feature_mean = np.asarray(reference["feature_mean"], dtype=np.float32)
        self.feature_scale = np.asarray(reference["feature_scale"], dtype=np.float32)
        self.feature_clip = float(reference["feature_clip"])
        if (
            self.feature_mean.shape != (dimension,)
            or self.feature_scale.shape != (dimension,)
            or not np.isfinite(self.feature_mean).all()
            or not np.isfinite(self.feature_scale).all()
            or np.any(self.feature_scale <= 0)
            or not np.isfinite(self.feature_clip)
            or self.feature_clip <= 0
        ):
            raise ValueError("saved fitting feature transform is invalid")
        self.model_sha256 = hashes["model.safetensors"]
        self.reference_sha256 = hashes["reference.npz"]
        self.protocol_sha256 = manifest["protocolSha256"]

    def _normalize(self, raw: np.ndarray) -> np.ndarray:
        """Apply the checkpoint's exact saved float32 affine/clip transform."""
        values = np.asarray(raw, dtype=np.float32)
        result = np.clip(
            (values - self.feature_mean) / self.feature_scale,
            -self.feature_clip,
            self.feature_clip,
        ).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("feature transform produced nonfinite values")
        return result

    @torch.no_grad()
    def predict(
        self,
        raw_action_features,
        raw_query_features,
        query_ids,
        basal_rate,
        basal_mask,
        context_weights,
        *,
        action_mask=None,
        chunk_size: int = 1024,
    ) -> dict[str, np.ndarray]:
        """Return prior marginal means on the caller's explicit query axis."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        raw_actions = np.asarray(raw_action_features, dtype=np.float32)
        if raw_actions.ndim == 2:
            raw_actions = raw_actions[:, None, :]
        if raw_actions.ndim != 3 or raw_actions.shape[2] != self.model.config.feature_dim:
            raise ValueError("raw action features must be [B,F] or [B,A,F]")
        batch = len(raw_actions)
        if action_mask is None:
            mask = np.ones(raw_actions.shape[:2], dtype=np.bool_)
        else:
            mask = np.asarray(action_mask)
        if mask.shape != raw_actions.shape[:2] or mask.dtype != np.bool_:
            raise ValueError("action mask must be Boolean [B,A]")
        safe_actions = np.where(mask[..., None], raw_actions, 0.0)
        if not np.isfinite(safe_actions).all():
            raise ValueError("unmasked action features must be finite")
        actions = self._normalize(safe_actions)

        raw_queries = np.asarray(raw_query_features, dtype=np.float32)
        ids = np.asarray(query_ids).astype(str)
        if (
            raw_queries.ndim != 2
            or raw_queries.shape[1] != self.model.config.feature_dim
            or ids.shape != (len(raw_queries),)
            or len(set(ids.tolist())) != len(ids)
            or not np.isfinite(raw_queries).all()
        ):
            raise ValueError("query features and unique query IDs must align")
        query_features = self._normalize(raw_queries)
        basal = np.asarray(basal_rate, dtype=np.float32)
        observed = np.asarray(basal_mask)
        if (
            basal.ndim != 2
            or basal.shape[1] != len(ids)
            or observed.shape != basal.shape
            or observed.dtype != np.bool_
            or not observed.any(1).all()
            or not np.isfinite(basal).all()
            or np.any(basal <= 0)
        ):
            raise ValueError("positive basal rates and Boolean support must be [C,Q]")
        contexts = len(basal)
        weights = np.asarray(context_weights, dtype=np.float64)
        if (
            weights.shape != (batch, contexts)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or np.any(weights.sum(1) <= 0)
        ):
            raise ValueError("context weights must be finite nonnegative [B,C]")
        weights = weights / weights.sum(1, keepdims=True)

        def tensor(values):
            return torch.as_tensor(values, device=self.device)

        query = tensor(query_features)
        basal_tensor = tensor(basal)
        observed_tensor = tensor(observed)
        context = self.model.encode_context(query, basal_tensor, observed_tensor)
        prior = self.model.prior_from_context(
            tensor(np.repeat(actions, contexts, axis=0)),
            tensor(np.repeat(mask, contexts, axis=0)),
            context.repeat(batch, 1),
        )
        pieces: list[np.ndarray] = []
        for left in range(0, len(query), chunk_size):
            stop = min(left + chunk_size, len(query))
            local_basal = basal_tensor[:, left:stop]
            local_basal = local_basal.unsqueeze(0).expand(batch, -1, -1).reshape(
                batch * contexts, stop - left
            )
            mean = self.model.population_mean(
                prior, query[left:stop], local_basal
            ).reshape(batch, contexts, stop - left)
            mixture = (
                mean.double() * tensor(weights[..., None])
            ).sum(1).cpu().numpy()
            pieces.append(mixture)
        cp10k = np.concatenate(pieces, axis=1) if pieces else np.empty((batch, 0), np.float64)
        empty = ~mask.any(1)
        if empty.any():
            # Preserve the public empty-intervention contract bit-for-bit at
            # the externally supplied context-mixture level as well as within
            # each numerical-core context.
            cp10k[empty] = weights[empty] @ basal.astype(np.float64)
        if not np.isfinite(cp10k).all() or np.any(cp10k <= 0):
            raise FloatingPointError("prior prediction is not finite and positive")
        support = (weights[:, :, None] == 0) | observed[None]
        supported = support.all(1)
        return {
            "query_ids": ids.copy(),
            "mean_cp10k": cp10k,
            "mean_log1p_cp10k": np.log1p(cp10k),
            "query_supported": supported,
            "prior_mean": prior["mean"].reshape(batch, contexts, -1).cpu().numpy(),
            "prior_logvar": prior["logvar"].reshape(batch, contexts, -1).cpu().numpy(),
        }
