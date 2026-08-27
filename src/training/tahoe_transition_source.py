import hashlib, json, math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaModel

ROOT = Path(__file__).resolve().parents[2]
SE = ROOT / "data/models/weights/SE-600M"
ST = ROOT / "data/models/weights/ST-SE-Tahoe/zeroshot/state_generalization_zeroshot_X_state"
EXPR = ROOT / "data/depmap24q2/OmicsExpressionProteinCodingGenesTPMLogp1.selected.csv.gz"
OUT = ROOT / "results/sl_predict/tahoe_transition_source_smoke.json"


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__(); d = 2048
        self.qkv_proj = nn.Linear(d, 3*d); self.out_proj = nn.Linear(d, d)
        self.norm1 = nn.LayerNorm(d); self.norm2 = nn.LayerNorm(d)
        self.linear1 = nn.Linear(d, d); self.linear2 = nn.Linear(d, d)

    def forward(self, x):
        q, k, v = self.qkv_proj(x).chunk(3, -1); shape = (x.shape[0], x.shape[1], 16, 128)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape_as(x)
        x = self.norm1(x + self.out_proj(a))
        return self.norm2(x + self.linear2(F.gelu(self.linear1(x))))


class StateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe = nn.Embedding(19790, 5120); self.cls = nn.Parameter(torch.empty(1, 5120))
        self.dataset = nn.Parameter(torch.empty(1, 5120))
        self.encoder = nn.Sequential(nn.Linear(5120, 2048), nn.LayerNorm(2048), nn.SiLU())
        self.count = nn.Sequential(nn.Linear(1, 512), nn.LeakyReLU(), nn.Linear(512, 10))
        self.bins = nn.Embedding(10, 2048); self.layers = nn.ModuleList([EncoderLayer() for _ in range(16)])
        self.skip_norm = nn.LayerNorm(2048); self.skip_up = nn.Linear(2048, 4096)
        self.skip_down = nn.Linear(4096, 2048); self.decode = nn.Linear(2048, 2048)
        self.dataset_embed = nn.Linear(2048, 10)

    def forward(self, tokens, counts):
        x = F.normalize(self.pe(tokens), dim=2); x[:, 0] = self.cls
        x = torch.cat((x, self.dataset.expand(x.shape[0], 1, -1)), 1)
        x = self.encoder(x) * math.sqrt(2048)
        w = F.softmax(self.count(counts[..., None]), -1)
        c = w @ self.bins.weight
        x = x + torch.cat((c, torch.zeros_like(c[:, :1])), 1)
        for layer in self.layers: x = layer(x)
        y = self.skip_norm(x + self.skip_down(F.relu(self.skip_up(x))))
        y = self.decode(y)
        return torch.cat((F.normalize(y[:, 0], dim=1), self.dataset_embed(y[:, -1])), 1)


class NoRoPE(nn.Module):
    def __init__(self, head_dim): super().__init__(); self.head_dim = head_dim
    def forward(self, x, position_ids):
        shape = (*position_ids.shape, self.head_dim)
        return x.new_ones(shape), x.new_zeros(shape)


class BidirectionalLlama(LlamaModel):
    def __init__(self, cfg):
        super().__init__(cfg); self.rotary_emb = NoRoPE(cfg.head_dim); self.config.is_causal = False
        for layer in self.layers: layer.self_attn.is_causal = False
    def _update_causal_mask(self, *args, **kwargs): return None


class Transition(nn.Module):
    def __init__(self):
        super().__init__()
        self.pert = nn.Linear(1138, 768); self.basal = nn.Linear(2058, 768)
        cfg = LlamaConfig(hidden_size=768, intermediate_size=3072, num_hidden_layers=8,
                          num_attention_heads=12, num_key_value_heads=12, head_dim=64,
                          max_position_embeddings=256, vocab_size=32000, use_cache=False,
                          attention_dropout=0., hidden_dropout=0., rms_norm_eps=1e-6)
        self.backbone = BidirectionalLlama(cfg); self.out = nn.Linear(768, 2058)

    def forward(self, basal, action):
        x = self.basal(basal) + self.pert(action)
        return self.out(self.backbone(inputs_embeds=x).last_hidden_state + self.basal(basal))


def load_se(model):
    rename = {"pe.weight":"pe_embedding.weight", "cls":"cls_token", "dataset":"dataset_token",
              "count.0":"count_encoder.0", "count.2":"count_encoder.2", "bins":"bin_encoder",
              "skip_norm":"decoder.0.layer_norm", "skip_up":"decoder.0.intermediate_dense",
              "skip_down":"decoder.0.dense", "decode":"decoder.1", "dataset_embed":"dataset_embedder"}
    with safe_open(str(SE / "model.safetensors"), framework="pt", device="cpu") as f:
        for name, p in model.named_parameters():
            key = name
            if name.startswith("layers."): key = "transformer_encoder." + name
            else:
                for a, b in rename.items():
                    if name == a or name.startswith(a + "."): key = b + name[len(a):]; break
            p.data.copy_(f.get_tensor(key))


def load_transition(model):
    state = torch.load(ST / "checkpoints/best.ckpt", map_location="cpu", weights_only=True)["state_dict"]
    mapped = {}
    for k, v in state.items():
        if k.startswith("pert_encoder.0."): mapped["pert." + k.rsplit(".", 1)[1]] = v
        elif k.startswith("basal_encoder.0."): mapped["basal." + k.rsplit(".", 1)[1]] = v
        elif k.startswith("transformer_backbone."): mapped["backbone." + k[21:]] = v
        elif k.startswith("project_out.0."): mapped["out." + k.rsplit(".", 1)[1]] = v
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    assert not unexpected and missing == []


def action_map():
    torch.serialization.add_safe_globals([np._core.multiarray.scalar, np.dtype, np.dtypes.StrDType])
    return torch.load(ST / "pert_onehot_map.pt", map_location="cpu", weights_only=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest().upper()


def main():
    torch.manual_seed(731); torch.set_grad_enabled(False)
    frame = pd.read_csv(EXPR, nrows=2, index_col=0)
    symbols = [x.rsplit(" (", 1)[0] for x in frame.columns]
    vocab = list(torch.load(SE / "protein_embeddings.pt", map_location="cpu", weights_only=True))
    index = {g:i for i,g in enumerate(vocab)}; keep = [i for i,g in enumerate(symbols) if g in index]
    values = frame.iloc[:, keep].to_numpy("float32") * np.float32(math.log(2))
    global_ids = np.asarray([index[symbols[i]] for i in keep])
    top = np.argsort(-values, axis=1, kind="stable")[:, :2047]
    tokens = np.concatenate((np.full((2,1), index[symbols[keep[3]]]), global_ids[top]), 1)
    weights = values / values.sum(1, keepdims=True)
    local_counts = np.concatenate((weights[:, 3:4], np.take_along_axis(weights, top, 1)), 1) * 100
    device = torch.device("cuda"); dtype = torch.bfloat16
    encoder = StateEncoder(); load_se(encoder); encoder.to(device, dtype=dtype).eval()
    t = torch.as_tensor(tokens, device=device); c = torch.as_tensor(local_counts, device=device, dtype=dtype)
    basal1 = encoder(t, c).float().cpu(); basal2 = encoder(t, c).float().cpu()
    del encoder; torch.cuda.empty_cache()
    transition = Transition(); load_transition(transition); transition.to(device, dtype=dtype).eval()
    amap = action_map(); keys = list(amap)
    selected = [next(k for k in keys if "DMSO_TF', 0.0" in str(k)), keys[0], keys[3]]
    def predict():
        out = []
        b = basal1.to(device, dtype=dtype)
        for key in selected:
            a = amap[key].to(device, dtype=dtype).expand(2, 256, -1)
            x = b[:, None].expand(-1, 256, -1)
            out.append(transition(x, a).mean(1).float().cpu())
        return torch.stack(out)
    pred1, pred2 = predict(), predict()
    control = pred1[0]; effects = pred1[1:] - control
    result = {
        "sources": {"expression_sha256": sha(EXPR), "se_sha256": sha(SE/"model.safetensors"),
                    "transition_sha256": sha(ST/"checkpoints/best.ckpt")},
        "cells": frame.index.tolist(), "expression_genes": len(symbols), "overlap_genes": len(keep),
        "basal_shape": list(basal1.shape), "prediction_shape": list(pred1.shape),
        "actions": [str(x) for x in selected],
        "finite": bool(torch.isfinite(basal1).all() and torch.isfinite(pred1).all()),
        "basal_repeat_max_abs": float((basal1-basal2).abs().max()),
        "prediction_repeat_max_abs": float((pred1-pred2).abs().max()),
        "action_effect_l2": effects.norm(dim=2).tolist(),
        "context_effect_difference_l2": (effects[:,0]-effects[:,1]).norm(dim=1).tolist()
    }
    result["admitted"] = bool(result["finite"] and result["basal_shape"] == [2,2058]
        and min(sum(result["action_effect_l2"], [])) > 0 and min(result["context_effect_difference_l2"]) > 0
        and result["prediction_repeat_max_abs"] <= 1e-5)
    OUT.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
