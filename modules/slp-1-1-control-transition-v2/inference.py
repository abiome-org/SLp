"""Portable inference for the minimal control-anchored transition.

The runtime consumes raw static action features and a control-only context on
the checkpoint's exact query panel.  It contains all preprocessing needed by
the fitted numerical model and deliberately has no data-loader dependency.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

try:
    from .transition_model import Config, MinimalControlTransition
except ImportError:  # Support a copied module directory placed on sys.path.
    _MODEL_PATH = Path(__file__).with_name("transition_model.py").resolve()
    _MODEL_KEY = "_slp11_minimal_control_transition_v2_" + hashlib.sha256(
        str(_MODEL_PATH).encode("utf-8")
    ).hexdigest()[:16]
    if _MODEL_KEY in sys.modules:
        _MODEL = sys.modules[_MODEL_KEY]
    else:
        _SPEC = importlib.util.spec_from_file_location(_MODEL_KEY, _MODEL_PATH)
        if _SPEC is None or _SPEC.loader is None:
            raise ImportError(f"cannot load sibling transition model at {_MODEL_PATH}")
        _MODEL = importlib.util.module_from_spec(_SPEC)
        sys.modules[_MODEL_KEY] = _MODEL
        _SPEC.loader.exec_module(_MODEL)
    Config = _MODEL.Config
    MinimalControlTransition = _MODEL.MinimalControlTransition


class PortableMinimalControl:
    """Load one frozen minimal-control checkpoint and its preprocessing state."""

    def __init__(self, package: str | Path, *, device: str = "cpu") -> None:
        package = Path(package)
        with (package / "model-config.json").open(encoding="utf-8") as stream:
            config = Config(**json.load(stream))
        with np.load(package / "runtime-reference.npz", allow_pickle=False) as archive:
            reference = {name: archive[name] for name in archive.files}
        required = {
            "feature_mean",
            "feature_std",
            "query_feature_mean",
            "query_feature_std",
            "query_features",
            "delta_amplitude",
            "query_ids",
            "context_query_indices",
            "context_panel_mask",
            "context_value_space",
        }
        if required - reference.keys():
            raise ValueError(
                "runtime reference is missing "
                + ", ".join(sorted(required - reference.keys()))
            )
        self.device = torch.device(device)
        self.model = MinimalControlTransition(config).to(self.device)
        self.model.load_state_dict(load_file(package / "model.safetensors", device=device))
        self.model.eval()
        self.query_ids = np.asarray(reference["query_ids"])
        self.context_value_space = str(reference["context_value_space"].item())
        self.feature_mean = np.asarray(reference["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(reference["feature_std"], dtype=np.float32)
        self.query_feature_mean = np.asarray(
            reference["query_feature_mean"], dtype=np.float32
        )
        self.query_feature_std = np.asarray(
            reference["query_feature_std"], dtype=np.float32
        )
        self.raw_query_features = np.asarray(
            reference["query_features"], dtype=np.float32
        )
        self.delta_amplitude = np.asarray(
            reference["delta_amplitude"], dtype=np.float32
        )
        self.context_query_indices = np.asarray(
            reference["context_query_indices"], dtype=np.int64
        )
        self.context_panel_mask = np.asarray(
            reference["context_panel_mask"], dtype=np.bool_
        )
        self._validate_reference()
        self.query_features = (
            (self.raw_query_features - self.query_feature_mean)
            / self.query_feature_std
        ).astype(np.float32)

    def _validate_reference(self) -> None:
        q = len(self.query_ids)
        if (
            self.query_ids.ndim != 1
            or len(set(self.query_ids.tolist())) != q
            or self.raw_query_features.shape
            != (q, self.model.config.query_feature_dim)
            or self.delta_amplitude.shape != (q,)
            or self.context_panel_mask.shape != (q,)
            or self.context_query_indices.ndim != 1
            or self.feature_mean.shape != (self.model.config.action_feature_dim,)
            or self.feature_std.shape != self.feature_mean.shape
            or self.query_feature_mean.shape != (self.model.config.query_feature_dim,)
            or self.query_feature_std.shape != self.query_feature_mean.shape
        ):
            raise ValueError("runtime reference shape or identity contract mismatch")
        if (
            not np.all(np.diff(self.context_query_indices) != 0)
            or np.any(self.context_query_indices < 0)
            or np.any(self.context_query_indices >= q)
            or not self.context_panel_mask[self.context_query_indices].all()
            or not np.isfinite(self.raw_query_features).all()
            or not np.isfinite(self.delta_amplitude).all()
            or not (self.delta_amplitude > 0).all()
            or not np.isfinite(self.feature_mean).all()
            or not np.isfinite(self.feature_std).all()
            or not (self.feature_std > 0).all()
            or not np.isfinite(self.query_feature_mean).all()
            or not np.isfinite(self.query_feature_std).all()
            or not (self.query_feature_std > 0).all()
        ):
            raise ValueError("runtime reference contains invalid numerical values")

    @staticmethod
    def _batch_matrix(values: np.ndarray, batch: int, width: int, label: str) -> np.ndarray:
        values = np.asarray(values)
        if values.shape == (width,):
            values = np.broadcast_to(values, (batch, width))
        if values.shape != (batch, width):
            raise ValueError(f"{label} must be [Q] or [B,Q]")
        return values

    def _prepare_context(
        self,
        values: np.ndarray,
        mask: np.ndarray,
        batch: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        q = len(self.query_ids)
        values = self._batch_matrix(values, batch, q, "control_context")
        mask = self._batch_matrix(mask, batch, q, "control_context_mask")
        if mask.dtype != np.bool_:
            raise ValueError("control_context_mask must be Boolean")
        if not np.all(mask == self.context_panel_mask[None, :]):
            raise ValueError("control context must use the frozen fixed-panel mask")
        safe = np.where(mask, values, 0.0).astype(np.float32)
        if not np.isfinite(safe).all():
            raise ValueError("observed control-context values must be finite")
        # Preserve NumPy's float32 mean/std operations used by the frozen
        # launcher.  A vectorized integer-count division promotes to float64
        # and changes a few last bits of the selected context tokens.
        normalized = np.zeros_like(safe)
        standard_deviations = []
        for row in range(batch):
            panel = safe[row, mask[row]]
            mean = panel.mean()
            std = panel.std()
            if not np.isfinite(std) or std < 1e-5:
                raise ValueError("control context has zero or nonfinite panel variance")
            standard_deviations.append(std)
            normalized[row, mask[row]] = (panel - mean) / std
        if not np.isfinite(standard_deviations).all() or np.any(
            np.asarray(standard_deviations) < 1e-5
        ):
            raise ValueError("control context has zero or nonfinite panel variance")
        selected = self.context_query_indices
        return normalized[:, selected].astype(np.float32), mask[:, selected]

    def predict(
        self,
        action_features: np.ndarray,
        control_context: np.ndarray,
        control_context_mask: np.ndarray,
        control_mean: np.ndarray,
        *,
        query_ids: np.ndarray,
        action_mask: np.ndarray | None = None,
        query_indices: np.ndarray | None = None,
        measurement_scale: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Forecast a supplied action set around an explicit control mean.

        With no ``measurement_scale``, no uncertainty is returned or implied.
        ``query_ids`` must equal the packaged ordered identity roster exactly.
        """
        if not np.array_equal(np.asarray(query_ids), self.query_ids):
            raise ValueError("query IDs do not equal the packaged ordered roster")
        actions = np.asarray(action_features, dtype=np.float32)
        if actions.ndim == 2:
            actions = actions[:, None, :]
        if actions.ndim != 3 or actions.shape[2] != self.model.config.action_feature_dim:
            raise ValueError("action_features must be [B,F] or [B,A,F]")
        batch = actions.shape[0]
        if action_mask is None:
            action_mask = np.ones(actions.shape[:2], dtype=np.bool_)
        action_mask = np.asarray(action_mask)
        if action_mask.shape != actions.shape[:2] or action_mask.dtype != np.bool_:
            raise ValueError("action_mask must be Boolean [B,A]")
        safe_actions = np.where(action_mask[..., None], actions, 0.0)
        if not np.isfinite(safe_actions).all():
            raise ValueError("unmasked action features must be finite")
        normalized_actions = np.where(
            action_mask[..., None],
            (safe_actions - self.feature_mean) / self.feature_std,
            0.0,
        ).astype(np.float32)
        basal_values, basal_mask = self._prepare_context(
            control_context, control_context_mask, batch
        )
        q = len(self.query_ids)
        control_mean = self._batch_matrix(control_mean, batch, q, "control_mean")
        if not np.isfinite(control_mean).all():
            raise ValueError("control_mean must be finite")
        if query_indices is None:
            query_indices = np.arange(q, dtype=np.int64)
        query_indices = np.asarray(query_indices, dtype=np.int64)
        if (
            query_indices.ndim != 1
            or np.any(query_indices < 0)
            or np.any(query_indices >= q)
            or len(np.unique(query_indices)) != len(query_indices)
        ):
            raise ValueError("query_indices must be unique in-range indices")
        if measurement_scale is None:
            scale = np.ones((batch, q), dtype=np.float32)
            calibrated = False
        else:
            scale = self._batch_matrix(measurement_scale, batch, q, "measurement_scale")
            if not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError("measurement_scale must be finite and positive")
            scale = scale.astype(np.float32)
            calibrated = True

        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(value, device=self.device)

        with torch.no_grad():
            output = self.model(
                tensor(normalized_actions),
                tensor(self.query_features[query_indices]),
                tensor(control_mean[:, query_indices].astype(np.float32)),
                tensor(self.delta_amplitude[query_indices]),
                tensor(scale[:, query_indices]),
                tensor(self.query_features[self.context_query_indices]),
                tensor(basal_values),
                tensor(basal_mask),
                action_mask=tensor(action_mask),
            )
        result: dict[str, Any] = {
            name: value.cpu().numpy()
            for name, value in output.items()
            if name != "scale"
        }
        result["query_ids"] = self.query_ids[query_indices].copy()
        result["query_indices"] = query_indices.copy()
        result["uncertainty_calibrated"] = calibrated
        if calibrated:
            result["scale"] = output["scale"].cpu().numpy()
        return result
