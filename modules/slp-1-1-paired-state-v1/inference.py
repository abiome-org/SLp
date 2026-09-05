"""Local paired-state artifact inference from explicit control measurements."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


class PairedEndpointPredictor:
    """Reload numerical model, transformations and registered assay descriptors.

    Raw action features are supplied directly. Stable query IDs index saved
    measurement descriptors; they never index trainable intervention weights.
    Controls are caller-supplied means on the registered RNA/antibody axes.
    """

    def __init__(self, artifact: str | Path, device: str = "cpu"):
        root = Path(artifact)
        protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
        report = json.loads((root / "report.json").read_text(encoding="utf-8"))
        for filename in ("protocol.json", "reference.npz", "model.safetensors"):
            with (root / filename).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != report["artifacts"][filename]:
                raise ValueError(f"artifact checksum mismatch: {filename}")
        source = root / "source/paired_model.py"
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if (
            source_hash
            != protocol["sources"]["modules/slp-1-1-paired-state-v1/paired_model.py"]
        ):
            raise ValueError("numerical source checksum mismatch")
        name = "slp_paired_runtime_" + source_hash[:16]
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        self.device = torch.device(device)
        self.model = (
            module.PairedStateModel(module.Config(**protocol["config"]))
            .to(self.device)
            .eval()
        )
        self.model.load_state_dict(
            load_file(str(root / "model.safetensors"), device=device)
        )
        self.feature_clip = protocol["settings"]["feature_clip"]
        with np.load(root / "reference.npz", allow_pickle=False) as reference:
            self.reference = {
                key: np.array(reference[key], copy=True) for key in reference.files
            }
        self.query_ids = {
            name: self.reference[f"{name}_query_ids"].astype(str)
            for name in ("rna", "protein")
        }

    @torch.no_grad()
    def predict(
        self,
        raw_actions,
        controls,
        *,
        action_mask=None,
        query_indices=None,
        chunk_size=1024,
    ):
        """Predict both assay means given controls in the fitted observation spaces.

        ``raw_actions`` is [B,A,1156] or [B,1156]. ``controls`` maps each
        modality to finite [B,Q] arrays in the saved assay order. These are
        quantitative control means, never perturbed outcomes. Optional query
        indices select output measurements without changing basal context.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        actions = np.asarray(raw_actions, dtype=np.float32)
        if actions.ndim == 2:
            actions = actions[:, None]
        if actions.ndim != 3 or actions.shape[-1] != self.model.config.action_features:
            raise ValueError("raw actions have the wrong feature shape")
        batch = len(actions)
        if action_mask is None:
            mask = np.ones(actions.shape[:2], dtype=bool)
        else:
            mask = np.asarray(action_mask)
            if mask.shape != actions.shape[:2] or mask.dtype != bool:
                raise ValueError("action_mask must be Boolean [B,A]")
        safe = np.where(mask[..., None], actions, 0.0)
        if not np.isfinite(safe).all():
            raise ValueError("unmasked action features must be finite")
        normalized = np.clip(
            (safe - self.reference["feature_mean"]) / self.reference["feature_scale"],
            -self.feature_clip,
            self.feature_clip,
        ).astype(np.float32)

        def tensor(array):
            return torch.as_tensor(array, device=self.device)

        basal, query, control_arrays = {}, {}, {}
        for name, ids in self.query_ids.items():
            values = np.asarray(controls[name], dtype=np.float32)
            if values.shape != (batch, len(ids)) or not np.isfinite(values).all():
                raise ValueError(
                    f"{name} controls must be finite [B,Q] on the saved query axis"
                )
            control_arrays[name] = values
            index = self.reference[f"{name}_basal_indices"]
            mean, scale = self.reference[f"{name}_basal_stats"]
            query[name] = tensor(self.reference[f"{name}_query_features"])
            basal[name] = {
                "features": query[name][tensor(index)],
                "values": tensor(
                    ((values[:, index] - mean) / scale).astype(np.float32)
                ),
                "observed": tensor(np.ones((batch, len(index)), dtype=bool)),
            }
        encoded = self.model.encode(tensor(normalized), tensor(mask), basal)
        result = {}
        for name, ids in self.query_ids.items():
            indices = (
                np.arange(len(ids))
                if query_indices is None
                else np.asarray(query_indices[name])
            )
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("query indices must be one-dimensional integers")
            if np.any(indices < 0) or np.any(indices >= len(ids)):
                raise ValueError("query index outside registered assay axis")
            parts = []
            for start in range(0, len(indices), chunk_size):
                selected = indices[start : start + chunk_size]
                output = self.model.observe(
                    encoded,
                    name,
                    query[name][tensor(selected)],
                    tensor(control_arrays[name][:, selected]),
                    tensor(self.reference[f"{name}_amplitude"][selected]),
                )
                parts.append(output["mean"].cpu().numpy())
            result[name] = {
                "query_ids": ids[indices].copy(),
                "mean": np.concatenate(parts, 1)
                if parts
                else np.empty((batch, 0), dtype=np.float32),
            }
        result["state"] = encoded["state"].cpu().numpy()
        return result
