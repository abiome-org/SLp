"""Typed, sparse-query world-model architecture for the SLp-1.1 candidate.

The module consumes only numerical features, explicit presence masks, and small
ontology type indices. Stable entity and query identifiers remain corpus
provenance and never enter ``WorldBatch`` or the parameterization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math

import torch
from torch import nn


MODEL_FORMAT = "slp.world-sparse/v1.1"
GAUSSIAN = 0
NEGATIVE_BINOMIAL = 1
LIKELIHOODS = ("gaussian", "negative-binomial")


@dataclass(frozen=True)
class WorldConfig:
    entity_feature_dim: int
    species_feature_dim: int
    entity_types: int
    context_types: int
    action_types: int
    readout_types: int
    record_covariate_dim: int = 0
    context_covariate_dim: int = 0
    action_covariate_dim: int = 0
    observation_covariate_dim: int = 0
    d_model: int = 256
    nhead: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 2
    ffn_multiplier: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        positive = (
            self.entity_feature_dim,
            self.species_feature_dim,
            self.entity_types,
            self.context_types,
            self.action_types,
            self.readout_types,
            self.d_model,
            self.nhead,
            self.encoder_layers,
            self.decoder_layers,
            self.ffn_multiplier,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in positive
        ):
            raise ValueError("model and ontology dimensions must be positive")
        covariate_dims = (
            self.record_covariate_dim,
            self.context_covariate_dim,
            self.action_covariate_dim,
            self.observation_covariate_dim,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in covariate_dims
        ):
            raise ValueError("covariate dimensions cannot be negative")
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead")
        if (
            not isinstance(self.dropout, (int, float))
            or isinstance(self.dropout, bool)
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout <= 0.5
        ):
            raise ValueError("dropout must be between 0 and 0.5")

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class WorldBatch:
    context_features: torch.Tensor
    context_feature_present: torch.Tensor
    context_entity_type: torch.Tensor
    context_type: torch.Tensor
    context_covariates: torch.Tensor
    context_covariate_present: torch.Tensor
    context_mask: torch.Tensor
    action_features: torch.Tensor
    action_feature_present: torch.Tensor
    action_entity_type: torch.Tensor
    action_type: torch.Tensor
    action_covariates: torch.Tensor
    action_covariate_present: torch.Tensor
    action_mask: torch.Tensor
    query_features: torch.Tensor
    query_feature_present: torch.Tensor
    query_entity_type: torch.Tensor
    readout_type: torch.Tensor
    likelihood_type: torch.Tensor
    query_mask: torch.Tensor
    species_features: torch.Tensor
    species_feature_present: torch.Tensor
    record_covariates: torch.Tensor
    record_covariate_present: torch.Tensor
    observation_covariates: torch.Tensor
    observation_covariate_present: torch.Tensor

    def __post_init__(self) -> None:
        forbidden = {"entity_id", "query_id", "gene_id", "dictionary_index"}
        present = {field.name for field in fields(self)}
        if present & forbidden:  # Defensive if the dataclass is extended later.
            raise TypeError("stable identifiers are forbidden in WorldBatch")


@dataclass(frozen=True)
class WorldPrediction:
    parameters: torch.Tensor
    likelihood_type: torch.Tensor
    query_mask: torch.Tensor

    @property
    def first_parameter(self) -> torch.Tensor:
        """Gaussian mean or NB log mean, selected by likelihood type."""

        return self.parameters[..., 0]

    @property
    def second_parameter(self) -> torch.Tensor:
        """Gaussian log scale or NB log inverse dispersion."""

        return self.parameters[..., 1]


class MaskedValueProjection(nn.Module):
    """Project values and their missingness as two distinct signals."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if input_dim:
            self.value = nn.Linear(input_dim, output_dim, bias=False)
            self.presence = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.value = None
            self.presence = None

    def forward(self, value: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        if value.shape != present.shape:
            raise ValueError("value and presence tensors must have identical shape")
        if value.shape[-1] != self.input_dim:
            raise ValueError("masked projection input dimension mismatch")
        if self.input_dim == 0:
            return value.new_zeros((*value.shape[:-1], self.output_dim))
        safe_value = torch.where(present, value, torch.zeros_like(value))
        return self.value(safe_value) + self.presence(present.to(value.dtype))


class CrossAttentionQueryBlock(nn.Module):
    """Update each query only from shared memory, never from other queries."""

    def __init__(self, config: WorldConfig):
        super().__init__()
        d_model = config.d_model
        hidden = config.ffn_multiplier * d_model
        self.cross_attention_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=config.nhead,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_attention_dropout = nn.Dropout(config.dropout)
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, d_model),
        )
        self.feedforward_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.cross_attention(
            query=self.cross_attention_norm(queries),
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        queries = queries + self.cross_attention_dropout(attended)
        transformed = self.feedforward(self.feedforward_norm(queries))
        return queries + self.feedforward_dropout(transformed)


class IndependentQueryDecoder(nn.Module):
    """A composition of pointwise and cross-attention-only query operations."""

    def __init__(self, config: WorldConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            CrossAttentionQueryBlock(config) for _ in range(config.decoder_layers)
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Keep the numerical kernel's query axis fixed at one. Although batched
        # cross-attention is algebraically independent across queries, BLAS/SDPA
        # kernels may choose Q-dependent reduction paths and lose bitwise
        # chunk equivalence. The explicit loop makes the marginal contract exact.
        decoded: list[torch.Tensor] = []
        for query in queries.unbind(dim=1):
            marginal = query.unsqueeze(1)
            for layer in self.layers:
                marginal = layer(marginal, memory, memory_key_padding_mask)
            decoded.append(self.norm(marginal))
        if not decoded:
            raise ValueError("the independent decoder requires at least one query")
        return torch.cat(decoded, dim=1)


class SparseTypedWorldModel(nn.Module):
    """Predict typed scalar molecular readouts without learned entity IDs."""

    def __init__(self, config: WorldConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        self.entity_features = MaskedValueProjection(config.entity_feature_dim, d_model)
        self.species_features = MaskedValueProjection(config.species_feature_dim, d_model)
        self.record_covariates = MaskedValueProjection(
            config.record_covariate_dim, d_model
        )
        self.context_covariates = MaskedValueProjection(
            config.context_covariate_dim, d_model
        )
        self.action_covariates = MaskedValueProjection(
            config.action_covariate_dim, d_model
        )
        self.observation_covariates = MaskedValueProjection(
            config.observation_covariate_dim, d_model
        )
        self.entity_kind = nn.Embedding(config.entity_types, d_model)
        self.context_kind = nn.Embedding(config.context_types, d_model)
        self.action_kind = nn.Embedding(config.action_types, d_model)
        self.readout_kind = nn.Embedding(config.readout_types, d_model)
        self.likelihood_kind = nn.Embedding(len(LIKELIHOODS), d_model)
        self.memory_kind = nn.Embedding(2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.nhead,
            dim_feedforward=config.ffn_multiplier * d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.memory_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.query_decoder = IndependentQueryDecoder(config)
        self.output_norm = nn.LayerNorm(d_model)
        self.gaussian_head = nn.Linear(d_model, 2)
        self.negative_binomial_head = nn.Linear(d_model, 2)

    def forward(self, batch: WorldBatch) -> WorldPrediction:
        self._validate_batch(batch)
        global_state = (
            self.species_features(
                batch.species_features, batch.species_feature_present
            )
            + self.record_covariates(
                batch.record_covariates, batch.record_covariate_present
            )
            + self.observation_covariates(
                batch.observation_covariates,
                batch.observation_covariate_present,
            )
        )

        context_entity_type = _safe_type(batch.context_entity_type, batch.context_mask)
        context = (
            self.entity_features(
                batch.context_features, batch.context_feature_present
            )
            + self.context_covariates(
                batch.context_covariates, batch.context_covariate_present
            )
            + self.entity_kind(context_entity_type)
            + self.context_kind(_safe_type(batch.context_type, batch.context_mask))
            + self.memory_kind.weight[0]
            + global_state[:, None, :]
        )
        action_entity_type = _safe_type(batch.action_entity_type, batch.action_mask)
        actions = (
            self.entity_features(batch.action_features, batch.action_feature_present)
            + self.action_covariates(
                batch.action_covariates, batch.action_covariate_present
            )
            + self.entity_kind(action_entity_type)
            + self.action_kind(_safe_type(batch.action_type, batch.action_mask))
            + self.memory_kind.weight[1]
            + global_state[:, None, :]
        )
        memory = torch.cat((context, actions), dim=1)
        memory_mask = torch.cat((batch.context_mask, batch.action_mask), dim=1)
        encoded = self.memory_encoder(memory, src_key_padding_mask=~memory_mask)

        parameter_chunks: list[torch.Tensor] = []
        likelihood_chunks: list[torch.Tensor] = []
        for query_index in range(batch.query_features.shape[1]):
            query_slice = slice(query_index, query_index + 1)
            query_mask = batch.query_mask[:, query_slice]
            safe_query_type = _safe_type(
                batch.query_entity_type[:, query_slice], query_mask
            )
            safe_readout_type = _safe_type(
                batch.readout_type[:, query_slice], query_mask
            )
            safe_likelihood_type = _safe_type(
                batch.likelihood_type[:, query_slice], query_mask
            )
            query = (
                self.entity_features(
                    batch.query_features[:, query_slice],
                    batch.query_feature_present[:, query_slice],
                )
                + self.entity_kind(safe_query_type)
                + self.readout_kind(safe_readout_type)
                + self.likelihood_kind(safe_likelihood_type)
                + global_state[:, None, :]
            )
            decoded = self.output_norm(
                self.query_decoder(query, encoded, memory_key_padding_mask=~memory_mask)
            )
            gaussian = self.gaussian_head(decoded)
            gaussian = torch.stack(
                (gaussian[..., 0], gaussian[..., 1].clamp(-7.0, 4.0)), dim=-1
            )
            negative_binomial = self.negative_binomial_head(decoded)
            negative_binomial = torch.stack(
                (
                    negative_binomial[..., 0].clamp(-12.0, 12.0),
                    negative_binomial[..., 1].clamp(-8.0, 8.0),
                ),
                dim=-1,
            )
            parameters = torch.where(
                (safe_likelihood_type == NEGATIVE_BINOMIAL).unsqueeze(-1),
                negative_binomial,
                gaussian,
            )
            parameter_chunks.append(
                torch.where(query_mask.unsqueeze(-1), parameters, torch.zeros_like(parameters))
            )
            likelihood_chunks.append(
                torch.where(
                    query_mask,
                    safe_likelihood_type,
                    torch.full_like(safe_likelihood_type, -1),
                )
            )
        parameters = torch.cat(parameter_chunks, dim=1)
        output_likelihood_type = torch.cat(likelihood_chunks, dim=1)
        return WorldPrediction(
            parameters=parameters,
            likelihood_type=output_likelihood_type,
            query_mask=batch.query_mask,
        )

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate_batch(self, batch: WorldBatch) -> None:
        if batch.context_features.ndim != 3:
            raise ValueError("context features must have shape [batch, tokens, features]")
        if batch.action_features.ndim != 3 or batch.query_features.ndim != 3:
            raise ValueError("action and query features must be rank three")
        batch_size = batch.context_features.shape[0]
        tensors = tuple(getattr(batch, field.name) for field in fields(batch))
        if any(tensor.shape[0] != batch_size for tensor in tensors):
            raise ValueError("all tensors must share the same batch dimension")
        self._validate_token_group(
            "context",
            batch.context_features,
            batch.context_feature_present,
            batch.context_entity_type,
            batch.context_mask,
            self.config.entity_feature_dim,
            self.config.entity_types,
        )
        self._validate_token_group(
            "action",
            batch.action_features,
            batch.action_feature_present,
            batch.action_entity_type,
            batch.action_mask,
            self.config.entity_feature_dim,
            self.config.entity_types,
        )
        self._validate_token_group(
            "query",
            batch.query_features,
            batch.query_feature_present,
            batch.query_entity_type,
            batch.query_mask,
            self.config.entity_feature_dim,
            self.config.entity_types,
        )
        _validate_typed(batch.context_type, batch.context_mask, self.config.context_types, "context")
        _validate_typed(batch.action_type, batch.action_mask, self.config.action_types, "action")
        _validate_typed(batch.readout_type, batch.query_mask, self.config.readout_types, "readout")
        _validate_typed(batch.likelihood_type, batch.query_mask, len(LIKELIHOODS), "likelihood")
        self._validate_covariates(
            "context",
            batch.context_covariates,
            batch.context_covariate_present,
            batch.context_mask,
            self.config.context_covariate_dim,
        )
        self._validate_covariates(
            "action",
            batch.action_covariates,
            batch.action_covariate_present,
            batch.action_mask,
            self.config.action_covariate_dim,
        )
        self._validate_vector(
            "species",
            batch.species_features,
            batch.species_feature_present,
            self.config.species_feature_dim,
        )
        self._validate_vector(
            "record covariate",
            batch.record_covariates,
            batch.record_covariate_present,
            self.config.record_covariate_dim,
        )
        self._validate_vector(
            "observation covariate",
            batch.observation_covariates,
            batch.observation_covariate_present,
            self.config.observation_covariate_dim,
        )
        if not batch.query_mask.any(dim=1).all():
            raise ValueError("every record requires at least one query")
        if not (batch.context_mask.any(dim=1) | batch.action_mask.any(dim=1)).all():
            raise ValueError("every record requires at least one context or action token")

    @staticmethod
    def _validate_token_group(
        name: str,
        value: torch.Tensor,
        present: torch.Tensor,
        entity_type: torch.Tensor,
        mask: torch.Tensor,
        feature_dim: int,
        type_count: int,
    ) -> None:
        if value.shape[-1] != feature_dim or value.shape != present.shape:
            raise ValueError(f"{name} feature shape mismatch")
        if mask.shape != value.shape[:2] or entity_type.shape != mask.shape:
            raise ValueError(f"{name} token shape mismatch")
        if present.dtype != torch.bool or mask.dtype != torch.bool:
            raise ValueError(f"{name} masks must be boolean")
        if (present & ~mask.unsqueeze(-1)).any():
            raise ValueError(f"padded {name} tokens cannot contain present features")
        _validate_observed_finite(value, present, f"{name} feature")
        _validate_typed(entity_type, mask, type_count, f"{name} entity")

    @staticmethod
    def _validate_covariates(
        name: str,
        value: torch.Tensor,
        present: torch.Tensor,
        token_mask: torch.Tensor,
        dimension: int,
    ) -> None:
        if value.shape != present.shape or value.shape[:2] != token_mask.shape:
            raise ValueError(f"{name} covariate shape mismatch")
        if value.shape[-1] != dimension:
            raise ValueError(f"{name} covariate dimension mismatch")
        if present.dtype != torch.bool:
            raise ValueError(f"{name} covariate presence must be boolean")
        if (present & ~token_mask.unsqueeze(-1)).any():
            raise ValueError(f"padded {name} tokens cannot contain covariates")
        _validate_observed_finite(value, present, f"{name} covariate")

    @staticmethod
    def _validate_vector(
        name: str,
        value: torch.Tensor,
        present: torch.Tensor,
        dimension: int,
    ) -> None:
        if value.ndim != 2 or value.shape != present.shape or value.shape[-1] != dimension:
            raise ValueError(f"{name} shape mismatch")
        if present.dtype != torch.bool:
            raise ValueError(f"{name} presence must be boolean")
        _validate_observed_finite(value, present, name)


def _safe_type(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, values, torch.zeros_like(values))


def _validate_typed(
    values: torch.Tensor, mask: torch.Tensor, type_count: int, name: str
) -> None:
    if values.shape != mask.shape or values.dtype != torch.long:
        raise ValueError(f"{name} type indices must be int64 and match the token mask")
    active = values[mask]
    if active.numel() and (active.min().item() < 0 or active.max().item() >= type_count):
        raise ValueError(f"{name} type index is out of range")


def _validate_observed_finite(
    values: torch.Tensor, present: torch.Tensor, name: str
) -> None:
    if not torch.is_floating_point(values):
        raise ValueError(f"{name} values must be floating point")
    if present.any() and not torch.isfinite(values[present]).all():
        raise ValueError(f"observed {name} values must be finite")


def negative_log_likelihood(
    prediction: WorldPrediction,
    target: torch.Tensor,
    target_observed: torch.Tensor,
    loss_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the weighted mean typed scalar negative log likelihood."""

    per_target = negative_log_likelihood_terms(
        prediction, target, target_observed
    )
    if loss_weight is None:
        weight = torch.ones_like(target)
    else:
        if loss_weight.shape != target.shape or not torch.is_floating_point(loss_weight):
            raise ValueError("loss_weight must be a floating target-shaped tensor")
        if not torch.isfinite(loss_weight).all() or (loss_weight < 0).any():
            raise ValueError("loss weights must be finite and non-negative")
        weight = loss_weight
    effective_weight = weight * target_observed.to(weight.dtype)
    denominator = effective_weight.sum()
    if denominator.item() <= 0:
        raise ValueError("observed targets must have positive total loss weight")
    return (per_target * effective_weight).sum() / denominator


def negative_log_likelihood_terms(
    prediction: WorldPrediction,
    target: torch.Tensor,
    target_observed: torch.Tensor,
) -> torch.Tensor:
    """Return target-shaped typed NLL terms, with unobserved entries set to zero."""

    if target.shape != prediction.query_mask.shape:
        raise ValueError("target shape must match prediction query shape")
    if target_observed.shape != target.shape or target_observed.dtype != torch.bool:
        raise ValueError("target_observed must be a boolean target-shaped tensor")
    if (target_observed & ~prediction.query_mask).any():
        raise ValueError("a padded query cannot have an observed target")
    if target_observed.any() and not torch.isfinite(target[target_observed]).all():
        raise ValueError("observed targets must be finite")
    nb_observed = target_observed & (prediction.likelihood_type == NEGATIVE_BINOMIAL)
    if nb_observed.any():
        nb_target = target[nb_observed]
        if (nb_target < 0).any() or not torch.allclose(nb_target, nb_target.round()):
            raise ValueError("negative-binomial targets must be non-negative counts")
    if not target_observed.any():
        raise ValueError("at least one target must be observed")

    gaussian_target = torch.where(target_observed, target, torch.zeros_like(target))
    gaussian_mean = prediction.parameters[..., 0]
    gaussian_log_scale = prediction.parameters[..., 1]
    gaussian_nll = 0.5 * math.log(2.0 * math.pi) + gaussian_log_scale
    gaussian_nll = gaussian_nll + 0.5 * (
        (gaussian_target - gaussian_mean) * torch.exp(-gaussian_log_scale)
    ).square()

    safe_nb_target = torch.where(nb_observed, target, torch.zeros_like(target))
    log_mean = prediction.parameters[..., 0]
    log_inverse_dispersion = prediction.parameters[..., 1]
    inverse_dispersion = torch.exp(log_inverse_dispersion)
    log_total = torch.logaddexp(log_inverse_dispersion, log_mean)
    nb_log_probability = (
        torch.lgamma(safe_nb_target + inverse_dispersion)
        - torch.lgamma(inverse_dispersion)
        - torch.lgamma(safe_nb_target + 1.0)
        + inverse_dispersion * (log_inverse_dispersion - log_total)
        + safe_nb_target * (log_mean - log_total)
    )
    nb_nll = -nb_log_probability
    per_target = torch.where(
        prediction.likelihood_type == NEGATIVE_BINOMIAL, nb_nll, gaussian_nll
    )
    return torch.where(target_observed, per_target, torch.zeros_like(per_target))
