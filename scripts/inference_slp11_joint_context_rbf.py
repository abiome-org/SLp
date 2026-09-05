#!/usr/bin/env python3
"""Target-free runtime for the joint context-conditioned RBF mean baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.square(left).sum(axis=1, dtype=np.float64)
    right_norm = np.square(right).sum(axis=1, dtype=np.float64)
    return np.maximum(
        left_norm[:, None] + right_norm[None, :] - 2.0 * (left @ right.T), 0.0
    )


def prediction_or_identity(
    control_mean: np.ndarray, prediction: np.ndarray | None
) -> np.ndarray:
    if prediction is None:
        return control_mean.copy()
    return prediction.astype(np.float32, copy=True)


class JointContextRbfRuntime:
    """Single-action mean predictor; it does not expose uncertainty or combinations."""

    def __init__(self, model_path: Path, feature_path: Path) -> None:
        with np.load(model_path, allow_pickle=False) as model:
            self.model = {key: model[key] for key in model.files}
        with np.load(feature_path, allow_pickle=False) as features:
            if np.any(features["entity_taxon"] != 9606):
                raise ValueError("physical feature taxonomy mismatch")
            ids = features["entity_id"].astype(str)
            values = features["feature_values"].astype(np.float32)
        if len(set(ids)) != len(ids) or values.shape[1] != 1156:
            raise ValueError("physical feature identity/dimension mismatch")
        self.feature_by_id = dict(zip(ids, values, strict=True))
        self.query_ids = self.model["query_ids"].astype(str)

    def _context_basis(
        self, basal_values: np.ndarray, basal_observed: np.ndarray
    ) -> np.ndarray:
        values = np.asarray(basal_values, dtype=np.float32)
        observed = np.asarray(basal_observed, dtype=bool)
        common = self.model["context_common_mask"].astype(bool)
        if values.shape != self.query_ids.shape or observed.shape != self.query_ids.shape:
            raise ValueError("caller basal descriptor must use the exact stored query axis")
        if not np.all(observed[common]) or int(common.sum()) != 6789:
            raise ValueError("caller must observe every stored fixed-panel control value")
        selected = values[common]
        scale = max(float(selected.std(dtype=np.float64)), 1e-5)
        normalized = ((selected - float(selected.mean(dtype=np.float64))) / scale).astype(
            np.float32
        )[None, :]
        distances = squared_distances(
            normalized, self.model["context_normalized_anchors"]
        )
        kernel = np.exp(
            -distances / (2.0 * float(self.model["context_bandwidth"]) ** 2)
        ).astype(np.float32)
        return kernel @ self.model["context_kernel_basis"]

    def _action_basis(self, entity_id: str) -> np.ndarray:
        if entity_id not in self.feature_by_id:
            raise KeyError(f"no exact taxonomy9606/Ensembl feature for {entity_id}")
        value = self.feature_by_id[entity_id][None, :]
        standardized = (
            value - self.model["action_feature_mean"]
        ) / self.model["action_feature_scale"]
        distances = squared_distances(
            standardized, self.model["action_standardized_landmarks"]
        )
        kernel = np.exp(
            -distances / (2.0 * float(self.model["action_bandwidth"]) ** 2)
        ).astype(np.float32)
        mapped = kernel @ self.model["action_kernel_basis"]
        if mapped.shape[1] > 512:
            raise ValueError("stored action kernel exceeds fixed width")
        return np.pad(mapped, ((0, 0), (0, 512 - mapped.shape[1]))).astype(np.float32)

    def predict(
        self,
        action_entity_ids: tuple[str, ...],
        basal_values: np.ndarray,
        basal_observed: np.ndarray,
        control_mean: np.ndarray,
    ) -> np.ndarray:
        control = np.asarray(control_mean, dtype=np.float32)
        if control.shape != self.query_ids.shape or not np.isfinite(control).all():
            raise ValueError("control mean must be finite and use the exact query axis")
        if len(action_entity_ids) == 0:
            return prediction_or_identity(control, None)
        if len(action_entity_ids) != 1:
            raise ValueError("this fitted single-action baseline does not support combinations")
        context = self._context_basis(basal_values, basal_observed)
        action = self._action_basis(action_entity_ids[0])
        interaction = np.einsum("ni,nj->nij", context, action).reshape(1, 1536)
        design = np.concatenate((context, interaction), axis=1).astype(np.float32)
        rotated = (
            design - self.model["ridge_feature_mean"]
        ) @ self.model["ridge_eigenvectors"]
        prediction = self.model["target_mean"] + (
            rotated
            / (
                self.model["ridge_eigenvalues"]
                + float(str(self.model["selected_alpha"]))
            )
        ) @ self.model["ridge_rhs"]
        return prediction_or_identity(control, prediction[0])
