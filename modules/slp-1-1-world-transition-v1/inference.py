"""Local inference from portable tensor files, without corpus or OMF access.

Input references are measured/fitting molecular quantities in the same fixed
value space as training. This runtime provides conditional predictions, not a release
approval or evidence that generated samples match a biological population.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transition_model import Config, TransitionWorld


class Predictor:
    def __init__(self, artifact: str | Path, device: str = "cpu"):
        root = Path(artifact)
        self.artifact = root
        config = Config(**json.loads((root / "model-config.json").read_text()))
        self.device = torch.device(device)
        self.model = TransitionWorld(config).to(self.device)
        self.model.load_state_dict(load_file(str(root / "model.safetensors"), device=device), strict=True)
        self.model.eval()
        with np.load(root / "reference.npz", allow_pickle=False) as archive:
            self.feature_mean = archive["feature_mean"].copy()
            self.feature_std = archive["feature_std"].copy()
            self.query_feature_mean = archive["query_feature_mean"].copy() if "query_feature_mean" in archive else self.feature_mean.copy()
            self.query_feature_std = archive["query_feature_std"].copy() if "query_feature_std" in archive else self.feature_std.copy()
            self.reference = archive["reference"].copy() if "reference" in archive else None
            self.reference_scale = archive["reference_scale"].copy() if "reference_scale" in archive else None
        if self.feature_mean.shape != (config.feature_dim,) or self.feature_std.shape != (config.feature_dim,):
            raise ValueError("checkpoint feature normalization shape mismatch")
        if not np.isfinite(self.feature_mean).all() or not np.isfinite(self.feature_std).all() or (self.feature_std <= 0).any():
            raise ValueError("invalid checkpoint feature normalization")
        query_width = config.query_feature_dim or config.feature_dim
        if (self.query_feature_mean.shape != (query_width,) or self.query_feature_std.shape != (query_width,)
                or not np.isfinite(self.query_feature_mean).all() or not np.isfinite(self.query_feature_std).all()
                or (self.query_feature_std <= 0).any()):
            raise ValueError("invalid checkpoint query normalization")

    def predict(self, action_features, query_features, reference, reference_scale,
                *, measurement_scale=None, context_features=None, context_values=None,
                context_mask=None):
        actions = np.asarray(action_features, dtype=np.float32)
        queries = np.asarray(query_features, dtype=np.float32)
        reference = np.asarray(reference, dtype=np.float32)
        scale = np.asarray(reference_scale, dtype=np.float32)
        if actions.ndim not in (2, 3) or queries.ndim != 2:
            raise ValueError("actions [B,F] or [B,A,F], queries [Q,F] required")
        if queries.shape[1] != self.query_feature_mean.size or actions.shape[-1] != self.feature_mean.size:
            raise ValueError("input feature width differs from checkpoint")
        if reference.shape not in ((len(queries),), (len(actions), len(queries))) or scale.shape != reference.shape or (scale <= 0).any():
            raise ValueError("aligned query reference and positive scale required")
        if not all(np.isfinite(x).all() for x in (actions, queries, reference, scale)):
            raise ValueError("prediction inputs must be finite")
        measurement = None
        if measurement_scale is not None:
            measurement = np.asarray(measurement_scale, dtype=np.float32)
            if measurement.shape not in (
                (len(queries),),
                (len(actions), len(queries)),
            ):
                raise ValueError("measurement_scale must be [Q] or [B,Q]")
            if not np.isfinite(measurement).all() or (measurement <= 0).any():
                raise ValueError("measurement_scale must be finite and positive")
        def tensor(value):
            return torch.as_tensor(value, device=self.device, dtype=torch.float32)
        context = {}
        if context_features is not None:
            features = np.asarray(context_features, dtype=np.float32)
            values = np.asarray(context_values, dtype=np.float32)
            mask = np.asarray(context_mask, dtype=bool)
            if (features.ndim != 3 or features.shape[0] != len(actions)
                    or features.shape[-1] != self.query_feature_mean.size
                    or features.shape[:2] != values.shape or values.shape != mask.shape):
                raise ValueError("context requires aligned [B,C,F] features and [B,C] values/mask")
            if not np.isfinite(features[mask]).all() or not np.isfinite(values[mask]).all():
                raise ValueError("observed context must be finite")
            context = {"context_features": tensor((features-self.query_feature_mean)/self.query_feature_std),
                       "context_values": tensor(values),
                       "context_mask": torch.as_tensor(mask, device=self.device)}
        elif context_values is not None or context_mask is not None:
            raise ValueError("context values/mask require features")
        with torch.no_grad():
            result = self.model(
                tensor((actions - self.feature_mean) / self.feature_std),
                tensor((queries - self.query_feature_mean) / self.query_feature_std),
                tensor(reference), tensor(scale), **context,
            )
        output = {key: value.cpu().numpy() for key, value in result.items()}
        # Exposure describes measurement noise only. It is deliberately applied
        # after model forward and therefore cannot alter the mean or latent state.
        if measurement is not None:
            output["scale"] = np.broadcast_to(
                measurement, (len(actions), len(queries))
            ).copy()
        output["marginal_scale"] = np.sqrt(output["scale"] ** 2 + (
            (output["factor"] ** 2).sum(-1) if "factor" in output else 0
        ))
        return output

    def fitted_reference(self, action_features, context_index, query_indices):
        """Return a target-free fitted reference and its stored base scale.

        A saved context-specific feature-linear ridge is used when present.
        Otherwise this selects the saved context mean. Query subsets are
        dynamic and retain the order and duplicates requested by the caller.
        """

        actions = np.asarray(action_features, dtype=np.float64)
        contexts = np.asarray(context_index)
        queries = np.asarray(query_indices)
        if actions.ndim != 2 or actions.shape[1] != self.feature_mean.size:
            raise ValueError("action_features must be [B,F] with checkpoint feature width")
        if not np.isfinite(actions).all():
            raise ValueError("action_features must be finite")
        if contexts.shape != (len(actions),) or contexts.dtype.kind not in "iu":
            raise ValueError("context_index must be one integer per action row")
        if queries.ndim != 1 or not len(queries) or queries.dtype.kind not in "iu":
            raise ValueError("query_indices must be a nonempty integer vector")
        if self.reference is None or self.reference_scale is None:
            raise ValueError("reference artifact lacks saved reference and reference_scale")
        if (
            self.reference.ndim != 2
            or self.reference.shape != self.reference_scale.shape
            or not np.isfinite(self.reference).all()
            or not np.isfinite(self.reference_scale).all()
            or (self.reference_scale <= 0).any()
        ):
            raise ValueError("saved reference arrays are invalid")
        if (
            (contexts < 0).any()
            or (contexts >= self.reference.shape[0]).any()
            or (queries < 0).any()
            or (queries >= self.reference.shape[1]).any()
        ):
            raise ValueError("context or query index is out of range")

        base_scale = self.reference_scale[contexts[:, None], queries[None, :]].copy()
        linear_path = self.artifact / "linear-reference.npz"
        if not linear_path.exists():
            fitted = self.reference[contexts[:, None], queries[None, :]].copy()
            return fitted, base_scale
        with np.load(linear_path, allow_pickle=False) as archive:
            required = {"coefficient", "feature_mean", "feature_std", "intercept"}
            if set(archive.files) != required:
                raise ValueError("linear reference artifact member contract mismatch")
            coefficient = archive["coefficient"].astype(np.float64, copy=False)
            feature_mean = archive["feature_mean"].astype(np.float64, copy=False)
            feature_std = archive["feature_std"].astype(np.float64, copy=False)
            intercept = archive["intercept"].astype(np.float64, copy=False)
        expected_contexts, expected_queries = self.reference.shape
        expected_features = self.feature_mean.size
        if (
            coefficient.shape != (expected_contexts, expected_features, expected_queries)
            or feature_mean.shape != (expected_contexts, expected_features)
            or feature_std.shape != feature_mean.shape
            or intercept.shape != self.reference.shape
            or not all(
                np.isfinite(value).all()
                for value in (coefficient, feature_mean, feature_std, intercept)
            )
            or (feature_std <= 0).any()
        ):
            raise ValueError("linear reference arrays are invalid")
        standardized = (actions - feature_mean[contexts]) / feature_std[contexts]
        selected_coefficient = coefficient[contexts][:, :, queries]
        fitted = intercept[contexts[:, None], queries[None, :]] + np.einsum(
            "bf,bfq->bq", standardized, selected_coefficient
        )
        if not np.isfinite(fitted).all():
            raise ValueError("linear fitted reference produced non-finite values")
        return fitted, base_scale

    def measurement_scales(self, num_cells, context_index, query_indices):
        """Load mean-baseline exposure components and return selected scales.

        Cell counts enter only this explicit uncertainty method. They are never
        forwarded to the molecular model or used to construct its mean inputs.
        """

        counts = np.asarray(num_cells, dtype=np.float64)
        contexts = np.asarray(context_index)
        queries = np.asarray(query_indices)
        if counts.ndim != 1 or not len(counts):
            raise ValueError("num_cells must be a nonempty vector")
        if not np.isfinite(counts).all() or (counts <= 0).any():
            raise ValueError("num_cells must be finite and positive")
        if contexts.shape != counts.shape or contexts.dtype.kind not in "iu":
            raise ValueError("context_index must be one integer per exposure")
        if queries.ndim != 1 or not len(queries) or queries.dtype.kind not in "iu":
            raise ValueError("query_indices must be a nonempty integer vector")
        path = self.artifact / "exposure-uncertainty.npz"
        with np.load(path, allow_pickle=False) as archive:
            world = {
                "world_biological_variance",
                "world_sampling_variance",
            }
            mean = {"mean_biological_variance", "mean_sampling_variance"}
            if world & set(archive.files) and not world.issubset(archive.files):
                raise ValueError("exposure artifact has incomplete world variance components")
            prefix = "world" if world.issubset(archive.files) else "mean"
            if prefix == "mean" and not mean.issubset(archive.files):
                raise ValueError("exposure artifact lacks mean variance components")
            biological = archive[f"{prefix}_biological_variance"].astype(
                np.float64, copy=False
            )
            sampling = archive[f"{prefix}_sampling_variance"].astype(
                np.float64, copy=False
            )
        if (
            biological.ndim != 2
            or biological.shape != sampling.shape
            or not np.isfinite(biological).all()
            or not np.isfinite(sampling).all()
            or (biological < 0).any()
            or (sampling < 0).any()
        ):
            raise ValueError("exposure variance components are invalid")
        if (
            (contexts < 0).any()
            or (contexts >= biological.shape[0]).any()
            or (queries < 0).any()
            or (queries >= biological.shape[1]).any()
        ):
            raise ValueError("context or query index is out of range")
        variance = (
            biological[contexts[:, None], queries[None, :]]
            + sampling[contexts[:, None], queries[None, :]] / counts[:, None]
        )
        return np.sqrt(np.maximum(variance, 0.05**2)).astype(np.float32)

    @staticmethod
    def sample(prediction, draws: int = 1, seed: int = 731):
        if not isinstance(draws, int) or draws <= 0:
            raise ValueError("draws must be positive")
        rng = np.random.default_rng(seed)
        mean = prediction["mean"]
        samples = mean[None] + prediction["scale"][None] * rng.standard_normal((draws, *mean.shape))
        if "factor" in prediction:
            factor = prediction["factor"]
            state_noise = rng.standard_normal((draws, factor.shape[0], factor.shape[-1]))
            samples += np.einsum("dbk,bqk->dbq", state_noise, factor)
        return samples.astype(np.float32)
