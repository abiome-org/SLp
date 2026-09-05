"""Portable mean ensemble for compatible transition-world tensor artifacts.

The ensemble averages molecular means only. Member latent states are returned
on a separate member axis because independently trained latent coordinates are
not aligned. The calibrated Gaussian scale describes observation uncertainty
around the ensemble mean; it is not full Bayesian epistemic uncertainty.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from inference import Predictor


def combine_member_outputs(
    outputs: list[dict[str, np.ndarray]], measurement_scale: np.ndarray
) -> dict[str, np.ndarray]:
    """Average member means while preserving each incompatible latent state."""

    if len(outputs) < 2:
        raise ValueError("an ensemble requires at least two member outputs")
    means = np.stack([np.asarray(output["mean"], dtype=np.float32) for output in outputs])
    states = np.stack([np.asarray(output["state"], dtype=np.float32) for output in outputs])
    scale = np.asarray(measurement_scale, dtype=np.float32)
    if any(output["mean"].shape != outputs[0]["mean"].shape for output in outputs):
        raise ValueError("member molecular mean shapes differ")
    if scale.shape != means.shape[1:] or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("ensemble observation scale must align with member means")
    if not np.isfinite(means).all() or not np.isfinite(states).all():
        raise ValueError("member outputs must be finite")
    return {
        "mean": means.mean(axis=0),
        "scale": scale.copy(),
        "marginal_scale": scale.copy(),
        "member_means": means,
        "member_states": states,
    }


class EnsemblePredictor:
    """Load a frozen ensemble artifact and predict its calibrated mean Gaussian."""

    def __init__(self, artifact: str | Path, device: str = "cpu"):
        self.artifact = Path(artifact)
        manifest = json.loads((self.artifact / "ensemble-manifest.json").read_text())
        members = manifest.get("members")
        if not isinstance(members, list) or len(members) < 2:
            raise ValueError("ensemble manifest requires at least two members")
        self.predictors = []
        self.references = []
        expected_queries = None
        expected_contexts = None
        for member in members:
            relative = Path(member["artifactPath"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("ensemble member path must stay inside the artifact")
            root = self.artifact / relative
            self.predictors.append(Predictor(root, device=device))
            with np.load(root / "reference.npz", allow_pickle=False) as archive:
                payload = {name: archive[name] for name in archive.files}
            if expected_queries is None:
                expected_queries = payload["query_ids"]
                expected_contexts = payload["context_ids"]
            elif not np.array_equal(payload["query_ids"], expected_queries) or not np.array_equal(
                payload["context_ids"], expected_contexts
            ):
                raise ValueError("ensemble member query or context identities differ")
            self.references.append(payload)
        self.query_ids = expected_queries
        self.context_ids = expected_contexts
        with np.load(self.artifact / "ensemble-exposure-uncertainty.npz", allow_pickle=False) as archive:
            self.biological = archive["ensemble_biological_variance"].astype(np.float64)
            self.sampling = archive["ensemble_sampling_variance"].astype(np.float64)
            self.scale_floor = float(archive["ensemble_scale_floor"])
            if not np.array_equal(archive["ensemble_query_ids"], self.query_ids):
                raise ValueError("ensemble exposure query identities differ")
            if not np.array_equal(archive["ensemble_context_ids"], self.context_ids):
                raise ValueError("ensemble exposure context identities differ")

    def measurement_scales(
        self, num_cells: np.ndarray, context_index: np.ndarray, query_indices: np.ndarray
    ) -> np.ndarray:
        counts = np.asarray(num_cells, dtype=np.float64)
        contexts = np.asarray(context_index)
        queries = np.asarray(query_indices)
        if counts.ndim != 1 or contexts.shape != counts.shape or contexts.dtype.kind not in "iu":
            raise ValueError("one integer context is required per positive exposure")
        if not np.isfinite(counts).all() or np.any(counts <= 0):
            raise ValueError("exposures must be finite and positive")
        if queries.ndim != 1 or queries.dtype.kind not in "iu":
            raise ValueError("query indices must be an integer vector")
        if (
            np.any(contexts < 0)
            or np.any(contexts >= self.biological.shape[0])
            or np.any(queries < 0)
            or np.any(queries >= self.biological.shape[1])
        ):
            raise ValueError("context or query index is out of range")
        variance = (
            self.biological[contexts[:, None], queries[None, :]]
            + self.sampling[contexts[:, None], queries[None, :]] / counts[:, None]
        )
        return np.sqrt(np.maximum(variance, self.scale_floor**2)).astype(np.float32)

    def predict(
        self,
        action_features: np.ndarray,
        num_cells: np.ndarray,
        context_index: np.ndarray,
        query_indices: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        actions = np.asarray(action_features, dtype=np.float32)
        contexts = np.asarray(context_index, dtype=np.int64)
        if actions.ndim != 2 or contexts.shape != (len(actions),):
            raise ValueError("actions and contexts must contain one row per prediction")
        if query_indices is None:
            query_indices = np.arange(len(self.query_ids), dtype=np.int64)
        query_indices = np.asarray(query_indices, dtype=np.int64)
        outputs = []
        for predictor, reference in zip(self.predictors, self.references):
            outputs.append(
                predictor.predict(
                    actions,
                    reference["query_features"][query_indices],
                    reference["reference"][contexts][:, query_indices],
                    reference["reference_scale"][contexts][:, query_indices],
                    context_features=np.broadcast_to(
                        reference["context_features"],
                        (len(actions), *reference["context_features"].shape),
                    ),
                    context_values=reference["context_values"][contexts],
                    context_mask=np.ones(
                        (len(actions), reference["context_values"].shape[1]), dtype=np.bool_
                    ),
                )
            )
        scale = self.measurement_scales(num_cells, contexts, query_indices)
        return combine_member_outputs(outputs, scale)
