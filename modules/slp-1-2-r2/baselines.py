"""Matched inductive SL feature baselines; no quantitative pretraining."""

from dataclasses import asdict
import torch
from torch import nn
from torch.nn import functional as F


class PairBaseline(nn.Module):
    def __init__(self, config, kind="feature_mlp"):
        super().__init__()
        if kind not in ("feature_mlp", "sequence_similarity"):
            raise ValueError("Unknown direct baseline")
        self.config, self.kind = config, kind
        dimension = config.sequence_dim + config.annotation_dim + 2
        if kind == "feature_mlp":
            self.gene = nn.Sequential(
                nn.LayerNorm(dimension), nn.Linear(dimension, 512), nn.GELU()
            )
            self.score = nn.Sequential(
                nn.LayerNorm(1536 + config.context_dim + 2),
                nn.Linear(1536 + config.context_dim + 2, 1024),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Linear(512, 1),
            )
        else:
            self.score = nn.Linear(4, 1)

    def configuration(self):
        return {"baseline": self.kind, "model": asdict(self.config)}

    def forward(self, batch):
        if batch["action_mask"].shape[1] != 2 or not batch["action_mask"].all():
            raise ValueError("Direct baseline requires two resolved genes")
        if not ((batch["query_kind"] == 4) | ~batch["query_mask"]).all():
            raise ValueError("Direct baseline accepts SL labels only")
        sequence = batch["action_sequence"]
        known = batch["action_known"]
        similarity = F.cosine_similarity(
            sequence[:, 0].float(), sequence[:, 1].float(), dim=-1
        )
        if self.kind == "feature_mlp":
            features = torch.cat(
                (sequence, batch["action_annotation"], known.to(sequence.dtype)), dim=-1
            )
            a, b = self.gene(features).unbind(1)
            pair = torch.cat(
                (
                    a + b,
                    (a - b).abs(),
                    a * b,
                    batch["context"],
                    batch["context_known"][:, None].to(a.dtype),
                    similarity[:, None],
                ),
                dim=-1,
            )
        else:
            both = known[:, :, 0].all(1).to(similarity.dtype)
            pair = torch.stack(
                (
                    similarity * both,
                    both,
                    known[:, :, 0].sum(1).to(similarity.dtype),
                    known[:, :, 1].sum(1).to(similarity.dtype),
                ),
                dim=-1,
            )
        logit = self.score(pair).expand(-1, batch["query_mask"].shape[1])
        return {
            "sl_logit": logit,
            "location": torch.zeros_like(logit),
            "log_scale": torch.zeros_like(logit),
        }
