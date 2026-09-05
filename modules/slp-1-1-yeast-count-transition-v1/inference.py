"""Portable inference for the yeast count-moment transition experiment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def _load_core(path: Path):
    spec = importlib.util.spec_from_file_location("slp11_yeast_transition_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load transition core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Predictor:
    """Predict absolute aggregate RNA means from static action and WT state."""

    def __init__(self, artifact: str | Path, *, device: str = "cpu") -> None:
        root = Path(artifact)
        self.device = torch.device(device)
        self.core = _load_core(root / "source/control_transition_model.py")
        with np.load(root / "reference.npz", allow_pickle=False) as archive:
            self.reference = {name: archive[name] for name in archive.files}
        config = self.core.Config(
            action_feature_dim=577,
            query_feature_dim=577,
            hidden_dim=128,
            state_dim=128,
            dropout=0.2,
        )
        self.model = self.core.MinimalControlTransition(config).to(self.device).eval()
        self.model.load_state_dict(
            load_file(str(root / "model.safetensors"), device=str(self.device))
        )

    def _forward(self, actions: np.ndarray, batches: np.ndarray, *, empty: bool):
        batch = np.asarray(batches)
        if batch.ndim != 1 or batch.dtype.kind not in "iu":
            raise ValueError("batch indices must be an integer vector")
        count = len(self.reference["batch_ids"])
        if np.any(batch < 0) or np.any(batch >= count):
            raise ValueError("batch index out of range")
        if empty:
            action_tensor = torch.empty((len(batch), 0, 577), device=self.device)
            action_mask = torch.empty(
                (len(batch), 0), dtype=torch.bool, device=self.device
            )
        else:
            raw = np.asarray(actions, dtype=np.float32)
            if raw.shape != (len(batch), 577) or not np.isfinite(raw).all():
                raise ValueError("action features must be finite [B,577]")
            normalized = (raw - self.reference["feature_mean"]) / self.reference[
                "feature_std"
            ]
            action_tensor = torch.as_tensor(normalized, device=self.device)
            action_mask = None
        index = torch.as_tensor(batch, dtype=torch.int64, device=self.device)
        query = torch.as_tensor(
            self.reference["query_features_normalized"], device=self.device
        )
        selected = self.reference["basal_query_indices"]
        with torch.no_grad():
            return self.model(
                action_tensor,
                query,
                torch.as_tensor(self.reference["control_mean"], device=self.device)[
                    index
                ],
                torch.as_tensor(self.reference["delta_amplitude"], device=self.device),
                torch.as_tensor(
                    self.reference["objective_query_scale"], device=self.device
                )[
                    torch.as_tensor(
                        self.reference["batch_context_index"][batch],
                        dtype=torch.int64,
                        device=self.device,
                    )
                ],
                query[torch.as_tensor(selected, dtype=torch.int64, device=self.device)],
                torch.as_tensor(
                    self.reference["basal_values_normalized"], device=self.device
                )[index],
                torch.as_tensor(
                    self.reference["basal_mask"], dtype=torch.bool, device=self.device
                )[index],
                action_mask=action_mask,
            )

    def predict(
        self, action_features: np.ndarray, batch_indices: np.ndarray
    ) -> np.ndarray:
        """Return absolute aggregate means [B,Q]; cell count never enters the mean."""
        return (
            self._forward(action_features, batch_indices, empty=False)["mean"]
            .cpu()
            .numpy()
        )

    def predict_empty(self, batch_indices: np.ndarray) -> np.ndarray:
        """Return the exact supplied WT control mean for each batch."""
        batch = np.asarray(batch_indices)
        return (
            self._forward(np.empty((len(batch), 577)), batch, empty=True)["mean"]
            .cpu()
            .numpy()
        )
