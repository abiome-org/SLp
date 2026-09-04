"""Shard-streaming maximum-likelihood and molecular-reward training."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath
from typing import Iterator

import numpy as np
import torch
from torch import nn

from architecture import SpeciesAwareWorldModel, WorldBatch, WorldConfig


REQUIRED_ARRAYS = {
    "context_features",
    "context_mask",
    "action_features",
    "action_covariates",
    "action_mask",
    "query_features",
    "query_mask",
    "readout_type",
    "species_features",
    "species_taxon",
    "target",
    "target_mask",
}


@dataclass(frozen=True)
class CorpusIndex:
    root: Path
    role: str
    entity_feature_dim: int
    species_feature_dim: int
    readout_types: tuple[str, ...]
    shards: tuple[dict[str, object], ...]

    @classmethod
    def load(cls, root: str | Path, role: str) -> "CorpusIndex":
        root = Path(root).resolve()
        manifest = json.loads((root / "corpus.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != "slp.corpus/v1" or manifest.get("role") != role:
            raise ValueError(f"expected an slp.corpus/v1 {role!r} snapshot")
        if manifest.get("benchmarkLabelsPresent") is not False:
            raise ValueError("benchmark labels are forbidden in a world-model corpus")
        feature_dim = manifest.get("entityFeatureDim")
        species_dim = manifest.get("speciesFeatureDim")
        readouts = manifest.get("readoutTypes")
        if not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("entityFeatureDim must be positive")
        if not isinstance(species_dim, int) or species_dim <= 0:
            raise ValueError("speciesFeatureDim must be positive")
        if not isinstance(readouts, list) or not readouts or any(not item for item in readouts):
            raise ValueError("readoutTypes must be a non-empty string list")
        return cls(
            root=root,
            role=role,
            entity_feature_dim=feature_dim,
            species_feature_dim=species_dim,
            readout_types=tuple(readouts),
            shards=tuple(manifest["shards"]),
        )

    def shard_path(self, shard: dict[str, object]) -> Path:
        relative = PurePosixPath(str(shard["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe shard path")
        return self.root.joinpath(*relative.parts)


def _validate_compatible(*corpora: CorpusIndex) -> None:
    first = corpora[0]
    for corpus in corpora[1:]:
        if (
            corpus.entity_feature_dim != first.entity_feature_dim
            or corpus.species_feature_dim != first.species_feature_dim
            or corpus.readout_types != first.readout_types
        ):
            raise ValueError("corpus feature and readout contracts do not match")


def _load_shard(corpus: CorpusIndex, shard: dict[str, object]) -> dict[str, np.ndarray]:
    with np.load(corpus.shard_path(shard), allow_pickle=False) as source:
        missing = sorted(REQUIRED_ARRAYS - set(source.files))
        if missing:
            raise ValueError(f"shard is missing arrays: {', '.join(missing)}")
        arrays = {name: source[name] for name in REQUIRED_ARRAYS}
    records = int(shard["records"])
    if any(array.shape[0] != records for array in arrays.values()):
        raise ValueError("shard record count does not match corpus.json")
    return arrays


def iter_batches(
    corpus: CorpusIndex,
    batch_size: int,
    *,
    seed: int,
    shuffle: bool,
) -> Iterator[tuple[WorldBatch, torch.Tensor, torch.Tensor, torch.Tensor]]:
    generator = np.random.default_rng(seed)
    shard_order = generator.permutation(len(corpus.shards)) if shuffle else np.arange(len(corpus.shards))
    for shard_id in shard_order:
        arrays = _load_shard(corpus, corpus.shards[int(shard_id)])
        records = len(arrays["target"])
        order = generator.permutation(records) if shuffle else np.arange(records)
        for start in range(0, records, batch_size):
            index = order[start : start + batch_size]
            batch = WorldBatch(
                context_features=torch.as_tensor(arrays["context_features"][index], dtype=torch.float32),
                context_mask=torch.as_tensor(arrays["context_mask"][index], dtype=torch.bool),
                action_features=torch.as_tensor(arrays["action_features"][index], dtype=torch.float32),
                action_covariates=torch.as_tensor(
                    arrays["action_covariates"][index], dtype=torch.float32
                ),
                action_mask=torch.as_tensor(arrays["action_mask"][index], dtype=torch.bool),
                query_features=torch.as_tensor(arrays["query_features"][index], dtype=torch.float32),
                query_mask=torch.as_tensor(arrays["query_mask"][index], dtype=torch.bool),
                readout_type=torch.as_tensor(arrays["readout_type"][index], dtype=torch.long),
                species_features=torch.as_tensor(
                    arrays["species_features"][index], dtype=torch.float32
                ),
            )
            yield (
                batch,
                torch.as_tensor(arrays["target"][index], dtype=torch.float32),
                torch.as_tensor(arrays["target_mask"][index], dtype=torch.bool),
                torch.as_tensor(arrays["species_taxon"][index], dtype=torch.long),
            )


def _to_device(batch: WorldBatch, device: torch.device) -> WorldBatch:
    return WorldBatch(**{name: value.to(device) for name, value in batch.__dict__.items()})


def _gaussian_nll(
    mean: torch.Tensor,
    log_scale: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * ((target - mean) * torch.exp(-log_scale)).square() + log_scale


def _linear_features(batch: WorldBatch, readout_types: int) -> torch.Tensor:
    context_weight = batch.context_mask.to(batch.context_features.dtype).unsqueeze(-1)
    action_weight = batch.action_mask.to(batch.action_features.dtype).unsqueeze(-1)
    context = (batch.context_features * context_weight).sum(1) / context_weight.sum(1).clamp_min(1)
    actions = (batch.action_features * action_weight).sum(1) / action_weight.sum(1).clamp_min(1)
    covariates = (batch.action_covariates * action_weight).sum(1) / action_weight.sum(1).clamp_min(1)
    fixed = torch.cat((context, actions, covariates, batch.species_features), dim=-1)
    fixed = fixed[:, None, :].expand(-1, batch.query_features.shape[1], -1)
    readout = torch.nn.functional.one_hot(
        batch.readout_type.clamp(0, readout_types - 1), num_classes=readout_types
    ).to(batch.query_features.dtype)
    intercept = torch.ones(
        (*batch.query_features.shape[:2], 1),
        dtype=batch.query_features.dtype,
        device=batch.query_features.device,
    )
    return torch.cat((fixed, batch.query_features, readout, intercept), dim=-1)


@dataclass(frozen=True)
class Baselines:
    mean: torch.Tensor
    mean_log_scale: torch.Tensor
    linear_weight: torch.Tensor
    linear_log_scale: torch.Tensor


def fit_baselines(corpus: CorpusIndex, batch_size: int, ridge: float = 1.0) -> Baselines:
    count = torch.zeros(len(corpus.readout_types), dtype=torch.float64)
    total = torch.zeros_like(count)
    square = torch.zeros_like(count)
    linear_xtx: torch.Tensor | None = None
    linear_xty: torch.Tensor | None = None
    for batch, target, target_mask, _species in iter_batches(
        corpus, batch_size, seed=0, shuffle=False
    ):
        mask = target_mask & batch.query_mask
        types = batch.readout_type[mask]
        values = target[mask].double()
        count.scatter_add_(0, types, torch.ones_like(values))
        total.scatter_add_(0, types, values)
        square.scatter_add_(0, types, values.square())
        features = _linear_features(batch, len(corpus.readout_types))[mask].double()
        if linear_xtx is None:
            linear_xtx = torch.zeros(
                (features.shape[1], features.shape[1]), dtype=torch.float64
            )
            linear_xty = torch.zeros(features.shape[1], dtype=torch.float64)
        linear_xtx.add_(features.T @ features)
        linear_xty.add_(features.T @ values)
    if (count == 0).any():
        raise ValueError("every declared readout type needs at least one pretraining target")
    mean = total / count
    variance = (square / count - mean.square()).clamp_min(1e-4)
    assert linear_xtx is not None and linear_xty is not None
    penalty = torch.eye(linear_xtx.shape[0], dtype=torch.float64) * ridge
    penalty[-1, -1] = 0.0
    linear_weight = torch.linalg.solve(linear_xtx + penalty, linear_xty)
    linear_square = torch.zeros_like(count)
    linear_count = torch.zeros_like(count)
    for batch, target, target_mask, _species in iter_batches(
        corpus, batch_size, seed=0, shuffle=False
    ):
        mask = target_mask & batch.query_mask
        types = batch.readout_type[mask]
        prediction = _linear_features(batch, len(corpus.readout_types))[mask].double() @ linear_weight
        residual_square = (target[mask].double() - prediction).square()
        linear_count.scatter_add_(0, types, torch.ones_like(residual_square))
        linear_square.scatter_add_(0, types, residual_square)
    linear_variance = (linear_square / linear_count).clamp_min(1e-4)
    return Baselines(
        mean=mean.float(),
        mean_log_scale=variance.sqrt().log().float(),
        linear_weight=linear_weight.float(),
        linear_log_scale=linear_variance.sqrt().log().float(),
    )


@dataclass
class Moments:
    count: int = 0
    sum_prediction: float = 0.0
    sum_target: float = 0.0
    sum_prediction_square: float = 0.0
    sum_target_square: float = 0.0
    sum_product: float = 0.0
    model_nll: float = 0.0
    mean_baseline_nll: float = 0.0
    linear_baseline_nll: float = 0.0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        model_nll: torch.Tensor,
        mean_baseline_nll: torch.Tensor,
        linear_baseline_nll: torch.Tensor,
    ) -> None:
        x = prediction.double()
        y = target.double()
        self.count += x.numel()
        self.sum_prediction += x.sum().item()
        self.sum_target += y.sum().item()
        self.sum_prediction_square += x.square().sum().item()
        self.sum_target_square += y.square().sum().item()
        self.sum_product += (x * y).sum().item()
        self.model_nll += model_nll.double().sum().item()
        self.mean_baseline_nll += mean_baseline_nll.double().sum().item()
        self.linear_baseline_nll += linear_baseline_nll.double().sum().item()

    @property
    def pearson(self) -> float:
        numerator = self.count * self.sum_product - self.sum_prediction * self.sum_target
        left = self.count * self.sum_prediction_square - self.sum_prediction**2
        right = self.count * self.sum_target_square - self.sum_target**2
        denominator = math.sqrt(max(left * right, 0.0))
        return numerator / denominator if denominator else 0.0

    @property
    def improvement(self) -> float:
        if not self.count:
            return 0.0
        model = self.model_nll / self.count
        baseline = min(self.mean_baseline_nll, self.linear_baseline_nll) / self.count
        return (baseline - model) / max(abs(baseline), 1e-8)

    @property
    def strongest_baseline_nll(self) -> float:
        return min(self.mean_baseline_nll, self.linear_baseline_nll) / self.count


@torch.no_grad()
def evaluate(
    model: SpeciesAwareWorldModel,
    corpus: CorpusIndex,
    baselines: Baselines,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    overall = Moments()
    by_species: dict[int, Moments] = {}
    for batch, target, target_mask, taxon in iter_batches(
        corpus, batch_size, seed=0, shuffle=False
    ):
        batch = _to_device(batch, device)
        target = target.to(device)
        mask = target_mask.to(device) & batch.query_mask
        prediction = model(batch)
        model_loss = _gaussian_nll(prediction.mean, prediction.log_scale, target)
        means = baselines.mean.to(device)[batch.readout_type]
        scales = baselines.mean_log_scale.to(device)[batch.readout_type]
        mean_baseline_loss = _gaussian_nll(means, scales, target)
        linear_mean = _linear_features(batch, len(corpus.readout_types)) @ baselines.linear_weight.to(device)
        linear_scale = baselines.linear_log_scale.to(device)[batch.readout_type]
        linear_baseline_loss = _gaussian_nll(linear_mean, linear_scale, target)
        overall.update(
            prediction.mean[mask].cpu(),
            target[mask].cpu(),
            model_loss[mask].cpu(),
            mean_baseline_loss[mask].cpu(),
            linear_baseline_loss[mask].cpu(),
        )
        for species in taxon.unique().tolist():
            row_mask = taxon == species
            cell_mask = mask.cpu() & row_mask[:, None]
            by_species.setdefault(int(species), Moments()).update(
                prediction.mean.cpu()[cell_mask],
                target.cpu()[cell_mask],
                model_loss.cpu()[cell_mask],
                mean_baseline_loss.cpu()[cell_mask],
                linear_baseline_loss.cpu()[cell_mask],
            )
    if not overall.count:
        raise ValueError("molecular validation corpus has no observed targets")
    return {
        "nll": overall.model_nll / overall.count,
        "baselineNll": overall.strongest_baseline_nll,
        "meanBaselineNll": overall.mean_baseline_nll / overall.count,
        "linearBaselineNll": overall.linear_baseline_nll / overall.count,
        "nllImprovement": overall.improvement,
        "effectPearson": overall.pearson,
        "minimumSpeciesNllImprovement": min(item.improvement for item in by_species.values()),
        "species": {
            str(taxon): {
                "targets": item.count,
                "nll": item.model_nll / item.count,
                "baselineNll": item.strongest_baseline_nll,
                "meanBaselineNll": item.mean_baseline_nll / item.count,
                "linearBaselineNll": item.linear_baseline_nll / item.count,
                "nllImprovement": item.improvement,
                "effectPearson": item.pearson,
            }
            for taxon, item in sorted(by_species.items())
        },
    }


def train_world(
    roots: dict[str, str | Path],
    config: dict[str, object],
) -> tuple[SpeciesAwareWorldModel, dict[str, object], Baselines]:
    seed = int(config.get("seed", 111))
    torch.manual_seed(seed)
    np.random.seed(seed)
    pretrain = CorpusIndex.load(roots["pretrain"], "pretrain")
    validation = CorpusIndex.load(roots["molecularValidation"], "molecular-validation")
    reward = CorpusIndex.load(roots["molecularReward"], "molecular-reward")
    _validate_compatible(pretrain, validation, reward)
    model_config = WorldConfig(
        entity_feature_dim=pretrain.entity_feature_dim,
        species_feature_dim=pretrain.species_feature_dim,
        readout_types=len(pretrain.readout_types),
        d_model=int(config.get("dModel", 256)),
        nhead=int(config.get("nhead", 8)),
        encoder_layers=int(config.get("encoderLayers", 4)),
        decoder_layers=int(config.get("decoderLayers", 2)),
        ffn_multiplier=int(config.get("ffnMultiplier", 4)),
        dropout=float(config.get("dropout", 0.1)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeciesAwareWorldModel(model_config).to(device)
    batch_size = int(config.get("batchSize", 32))
    baselines = fit_baselines(pretrain, batch_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learningRate", 3e-4)),
        weight_decay=1e-2,
    )
    history: list[dict[str, object]] = []
    initial_metrics = evaluate(model, validation, baselines, batch_size, device)
    history.append({"phase": "initial", "epoch": 0, **initial_metrics})
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, object] | None = None
    for epoch in range(int(config.get("epochs", 4))):
        model.train()
        for batch, target, target_mask, _species in iter_batches(
            pretrain, batch_size, seed=seed + epoch, shuffle=True
        ):
            batch = _to_device(batch, device)
            target = target.to(device)
            mask = target_mask.to(device) & batch.query_mask
            prediction = model(batch)
            loss = _gaussian_nll(prediction.mean, prediction.log_scale, target)[mask].mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metrics = evaluate(model, validation, baselines, batch_size, device)
        history.append({"phase": "pretrain", "epoch": epoch + 1, **metrics})
        if best_metrics is None or float(metrics["nll"]) < float(best_metrics["nll"]):
            best_metrics = metrics
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state)

    pre_rl_metrics = best_metrics
    pre_rl_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    rl_retained = False
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("reinforcementLearningRate", 2e-5)),
        weight_decay=1e-2,
    )
    anchor_weight = float(config.get("reinforcementAnchorWeight", 0.1))
    for epoch in range(int(config.get("reinforcementEpochs", 0))):
        model.train()
        for batch, target, target_mask, _species in iter_batches(
            reward, batch_size, seed=seed + 1000 + epoch, shuffle=True
        ):
            batch = _to_device(batch, device)
            target = target.to(device)
            mask = target_mask.to(device) & batch.query_mask
            prediction = model(batch)
            distribution = torch.distributions.Normal(prediction.mean, prediction.scale)
            sample = distribution.sample()
            advantage = (
                -(sample - target).abs() + (prediction.mean.detach() - target).abs()
            ).detach()
            policy_loss = -(advantage * distribution.log_prob(sample))[mask].mean()
            anchor = _gaussian_nll(prediction.mean, prediction.log_scale, target)[mask].mean()
            loss = policy_loss + anchor_weight * anchor
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metrics = evaluate(model, validation, baselines, batch_size, device)
        history.append({"phase": "reinforcement", "epoch": epoch + 1, **metrics})
        if (
            float(metrics["nll"]) < float(best_metrics["nll"])
            and float(metrics["minimumSpeciesNllImprovement"])
            >= float(pre_rl_metrics["minimumSpeciesNllImprovement"])
        ):
            best_metrics = metrics
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            rl_retained = True
    if rl_retained:
        model.load_state_dict(best_state)
    else:
        model.load_state_dict(pre_rl_state)
        best_metrics = pre_rl_metrics
    report = {
        "schema": "slp.training-report/v1.1",
        "seed": seed,
        "modelConfig": model.config.as_dict(),
        "readoutTypes": list(pretrain.readout_types),
        "parameterCount": model.count_parameters(),
        "reinforcementRetained": rl_retained,
        "selected": best_metrics,
        "history": history,
    }
    return model.cpu(), report, baselines
