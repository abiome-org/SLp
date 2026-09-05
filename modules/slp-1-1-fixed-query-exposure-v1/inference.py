"""Portable point inference for a frozen fixed-query transition artifact.

Exposure is deliberately absent from this API because it changed fitting loss
weights only and must not affect molecular means or states.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


class Predictor:
    """Load source, reference, and weights from one immutable artifact."""

    def __init__(self, artifact: str | Path):
        root = Path(artifact).resolve(strict=True)
        model_path = root / "source/transition_model.py"
        spec = importlib.util.spec_from_file_location(
            "portable_fixed_query_exposure_core", model_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {model_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        with np.load(root / "reference.npz", allow_pickle=False) as archive:
            self.reference = {name: archive[name] for name in archive.files}
        self.model = module.FixedQueryTransition(
            module.Config(
                len(self.reference["feature_mean"]),
                len(self.reference["query_feature_mean"]),
                state_dim=int(self.reference["state_dim"]),
                hidden_dim=int(self.reference["hidden_dim"]),
                dropout=float(self.reference["dropout"]),
            )
        )
        self.model.load_state_dict(load_file(root / "model.safetensors"))
        self.model.eval()

    def predict(
        self, raw_action_features: np.ndarray, context_index: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Predict molecular mean and state from target-free inputs."""

        raw = np.asarray(raw_action_features, dtype=np.float32)
        contexts = np.asarray(context_index)
        if (
            raw.ndim not in (2, 3)
            or raw.shape[0] == 0
            or raw.shape[-1] != len(self.reference["feature_mean"])
            or not np.isfinite(raw).all()
            or contexts.shape != (raw.shape[0],)
            or contexts.dtype.kind not in "iu"
            or np.any(contexts < 0)
            or np.any(contexts >= len(self.reference["context_ids"]))
        ):
            raise ValueError("raw action features and contexts do not align")
        normalized = (raw - self.reference["feature_mean"]) / self.reference[
            "feature_std"
        ]
        query = (
            self.reference["query_features"] - self.reference["query_feature_mean"]
        ) / self.reference["query_feature_std"]
        selected = self.reference["context_query_indices"]
        with torch.no_grad():
            result = self.model(
                torch.as_tensor(normalized),
                torch.as_tensor(self.reference["fixed_query_coordinates"]),
                torch.as_tensor(self.reference["control_mean"][contexts]),
                torch.as_tensor(self.reference["delta_amplitude"]),
                torch.as_tensor(self.reference["objective_query_scale"][contexts]),
                torch.as_tensor(query[selected]),
                torch.as_tensor(self.reference["context_values"][contexts]),
                torch.as_tensor(
                    self.reference["context_mask"][contexts], dtype=torch.bool
                ),
            )
        return {name: value.numpy() for name, value in result.items()}
