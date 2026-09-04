"""Species-aware, vocabulary-free intervention world model for SLp-1.1."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


MODEL_FORMAT = "slp.world/v1.1"


@dataclass(frozen=True)
class WorldConfig:
    entity_feature_dim: int
    species_feature_dim: int
    readout_types: int
    action_covariate_dim: int = 4
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
            self.readout_types,
            self.action_covariate_dim,
            self.d_model,
            self.nhead,
            self.encoder_layers,
            self.decoder_layers,
            self.ffn_multiplier,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("all world-model dimensions must be positive")
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead")
        if not 0 <= self.dropout <= 0.5:
            raise ValueError("dropout must be between 0 and 0.5")

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class WorldBatch:
    context_features: torch.Tensor
    context_mask: torch.Tensor
    action_features: torch.Tensor
    action_covariates: torch.Tensor
    action_mask: torch.Tensor
    query_features: torch.Tensor
    query_mask: torch.Tensor
    readout_type: torch.Tensor
    species_features: torch.Tensor


@dataclass(frozen=True)
class WorldPrediction:
    mean: torch.Tensor
    log_scale: torch.Tensor
    query_mask: torch.Tensor

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp()


class SpeciesAwareWorldModel(nn.Module):
    """Predict sparse molecular readouts from context and intervention sets.

    Gene identifiers never enter this module. Every entity is represented by
    versioned molecular features supplied by the corpus. With no positional
    encoding, context and intervention memory is permutation equivariant and
    the query outputs are invariant to memory order.
    """

    def __init__(self, config: WorldConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.context_projection = nn.Linear(config.entity_feature_dim, d)
        self.action_projection = nn.Linear(config.entity_feature_dim, d)
        self.action_covariate_projection = nn.Linear(config.action_covariate_dim, d)
        self.query_projection = nn.Linear(config.entity_feature_dim, d)
        self.species_projection = nn.Linear(config.species_feature_dim, d, bias=False)
        self.memory_kind = nn.Embedding(2, d)
        self.readout_kind = nn.Embedding(config.readout_types, d)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.nhead,
            dim_feedforward=config.ffn_multiplier * d,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=config.nhead,
            dim_feedforward=config.ffn_multiplier * d,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.memory_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(d),
        )
        self.query_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(d),
        )
        self.output_norm = nn.LayerNorm(d)
        self.mean_head = nn.Linear(d, 1)
        self.scale_head = nn.Linear(d, 1)

    def forward(self, batch: WorldBatch) -> WorldPrediction:
        self._validate_batch(batch)
        species = self.species_projection(batch.species_features)[:, None, :]
        context = (
            self.context_projection(batch.context_features)
            + self.memory_kind.weight[0]
            + species
        )
        actions = (
            self.action_projection(batch.action_features)
            + self.action_covariate_projection(batch.action_covariates)
            + self.memory_kind.weight[1]
            + species
        )
        memory = torch.cat((context, actions), dim=1)
        memory_mask = torch.cat((batch.context_mask, batch.action_mask), dim=1)
        encoded = self.memory_encoder(memory, src_key_padding_mask=~memory_mask)

        queries = (
            self.query_projection(batch.query_features)
            + self.readout_kind(batch.readout_type)
            + species
        )
        decoded = self.query_decoder(
            tgt=queries,
            memory=encoded,
            tgt_key_padding_mask=~batch.query_mask,
            memory_key_padding_mask=~memory_mask,
        )
        decoded = self.output_norm(decoded)
        mean = self.mean_head(decoded).squeeze(-1)
        log_scale = self.scale_head(decoded).squeeze(-1).clamp(-7.0, 4.0)
        zeros = torch.zeros((), dtype=mean.dtype, device=mean.device)
        return WorldPrediction(
            mean=torch.where(batch.query_mask, mean, zeros),
            log_scale=torch.where(batch.query_mask, log_scale, zeros),
            query_mask=batch.query_mask,
        )

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate_batch(self, batch: WorldBatch) -> None:
        if batch.context_features.ndim != 3 or batch.action_features.ndim != 3:
            raise ValueError("context and action features must have shape [batch, tokens, features]")
        if batch.query_features.ndim != 3:
            raise ValueError("query features must have shape [batch, queries, features]")
        batch_size = batch.context_features.shape[0]
        if any(
            tensor.shape[0] != batch_size
            for tensor in (
                batch.context_mask,
                batch.action_features,
                batch.action_covariates,
                batch.action_mask,
                batch.query_features,
                batch.query_mask,
                batch.readout_type,
                batch.species_features,
            )
        ):
            raise ValueError("all tensors must have the same batch dimension")
        if batch.context_features.shape[-1] != self.config.entity_feature_dim:
            raise ValueError("context feature dimension does not match the model config")
        if batch.action_features.shape[-1] != self.config.entity_feature_dim:
            raise ValueError("action feature dimension does not match the model config")
        if batch.query_features.shape[-1] != self.config.entity_feature_dim:
            raise ValueError("query feature dimension does not match the model config")
        if batch.action_covariates.shape[-1] != self.config.action_covariate_dim:
            raise ValueError("action covariate dimension does not match the model config")
        if batch.species_features.shape[-1] != self.config.species_feature_dim:
            raise ValueError("species feature dimension does not match the model config")
        if batch.context_mask.shape != batch.context_features.shape[:2]:
            raise ValueError("context mask shape mismatch")
        if batch.action_mask.shape != batch.action_features.shape[:2]:
            raise ValueError("action mask shape mismatch")
        if batch.action_covariates.shape[:2] != batch.action_features.shape[:2]:
            raise ValueError("action covariate shape mismatch")
        if batch.query_mask.shape != batch.query_features.shape[:2]:
            raise ValueError("query mask shape mismatch")
        if batch.readout_type.shape != batch.query_features.shape[:2]:
            raise ValueError("readout type shape mismatch")
        if (
            not batch.context_mask.any(dim=1).all()
            or not batch.action_mask.any(dim=1).all()
            or not batch.query_mask.any(dim=1).all()
        ):
            raise ValueError("every example requires at least one context, action, and query token")
        valid_types = batch.readout_type[batch.query_mask]
        if valid_types.numel() and (
            valid_types.min().item() < 0 or valid_types.max().item() >= self.config.readout_types
        ):
            raise ValueError("readout type is outside the configured vocabulary")
