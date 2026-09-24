"""GiGCN binary architecture adaptation on SLB human measured pairs.

Source: wqz2469/GIGCN (2026). This retains its factorised signed graph
convolution and pair-correlation decoder. SLB has SL and measured neutral
labels, so the third (synthetic-viable) channel is empty and the head is binary.
Only SLB train labels form graph edges. GO features come from the filtered SLB
bundle; the authors' GO similarity matrix was not released.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import bundle_io as B  # noqa: E402
import slb  # noqa: E402

DIM = 64
FACTORS = 8
LAYERS = 2
SEED = 42
CONFIG = {"dim": DIM, "factors": FACTORS, "layers": LAYERS, "seed": SEED,
          "lr": .005, "weight_decay": .005, "lambda_disc": .05,
          "max_epochs": 300, "patience": 30, "validation_gene_fraction": .15,
          "aggregator": "mean", "go": "filtered propagated annotations; TF-IDF and 64D SVD"}
CHECKPOINT = slb.ROOT / "external/models/gigcn/_slb" / slb.BENCH.name / "binary_go.pt"


def digest(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def features() -> tuple[list[str], np.ndarray]:
    names: set[str] = set()
    for split in ("train", "dev", "test"):
        d = slb.load(split)
        d = d[d.species == "human"]
        names.update(d.gene_a)
        names.update(d.gene_b)
    genes = sorted(names)
    go = B.go_propagated("human")
    terms = sorted(set().union(*(go.get(g, set()) for g in genes)))
    term_index = {t: j for j, t in enumerate(terms)}
    rows, cols = [], []
    for i, gene in enumerate(genes):
        ts = go.get(gene, ())
        rows.extend([i] * len(ts))
        cols.extend(term_index[t] for t in ts)
    mat = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                     shape=(len(genes), len(terms)), dtype=np.float32)
    tfidf = TfidfTransformer(norm="l2").fit_transform(mat)
    x = TruncatedSVD(n_components=DIM, n_iter=7, random_state=SEED).fit_transform(tfidf)
    x = StandardScaler().fit_transform(x).astype(np.float32)
    print(f"GO features: {len(genes):,} genes, {len(terms):,} terms; "
          f"{sum(g in go for g in genes):,} genes annotated", flush=True)
    return genes, x


class GiGCN(nn.Module):
    """The source DINES encoder with its supported mean aggregation and binary head."""

    def __init__(self) -> None:
        super().__init__()
        self.init_weight = nn.Parameter(torch.empty(FACTORS, DIM, DIM // FACTORS))
        self.init_bias = nn.Parameter(torch.zeros(1, FACTORS, DIM // FACTORS))
        nn.init.xavier_uniform_(self.init_weight)
        self.update_weight = nn.ParameterList([
            nn.Parameter(torch.empty(FACTORS, 4 * DIM // FACTORS, DIM // FACTORS))
            for _ in range(LAYERS)])
        self.update_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(1, FACTORS, DIM // FACTORS)) for _ in range(LAYERS)])
        for w in self.update_weight:
            nn.init.xavier_uniform_(w)
        self.sign_predictor = nn.Linear(FACTORS**2, 2, bias=False)
        nn.init.xavier_uniform_(self.sign_predictor.weight)
        self.factor_discriminator = nn.Linear(DIM // FACTORS, FACTORS)
        nn.init.xavier_uniform_(self.factor_discriminator.weight)
        nn.init.zeros_(self.factor_discriminator.bias)

    def encode(self, x: torch.Tensor, edges: list[torch.Tensor]) -> torch.Tensor:
        # Matches source InitDisenLayer: initial normalisation is over factors.
        z = F.normalize(torch.tanh(torch.einsum("ij,kjl->ikl", x, self.init_weight)
                                   + self.init_bias))
        for w, bias in zip(self.update_weight, self.update_bias):
            aggregates = [z]
            for edge in edges:  # SV (empty), SL, measured neutral
                agg = torch.zeros_like(z)
                if edge.numel():
                    src, dst = edge[:, 0], edge[:, 1]
                    agg.index_add_(0, src, z[dst])
                    count = torch.bincount(src, minlength=len(z)).clamp_min(1)
                    agg = agg / count[:, None, None]
                aggregates.append(agg)
            z = F.normalize(torch.tanh(torch.einsum(
                "ijk,jkl->ijl", torch.cat(aggregates, dim=2), w) + bias), dim=2)
        return z

    def pair_logits(self, z: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        a, b = z[edge[:, 0]], z[edge[:, 1]]
        ab = torch.bmm(a, b.transpose(1, 2)).flatten(1)
        ba = torch.bmm(b, a.transpose(1, 2)).flatten(1)
        return (self.sign_predictor(ab) + self.sign_predictor(ba)) / 2

    def factor_loss(self, z: torch.Tensor) -> torch.Tensor:
        n = len(z)
        logits = self.factor_discriminator(z.reshape(-1, DIM // FACTORS))
        # The source applies Softmax before cross-entropy here; retain it.
        prob = F.softmax(logits, dim=1)
        targets = torch.arange(FACTORS, device=z.device).repeat(n)
        return F.cross_entropy(prob, targets)


def tensor_edges(d: pd.DataFrame, node: dict[str, int], device: torch.device) -> torch.Tensor:
    return torch.tensor(np.column_stack((d.gene_a.map(node).to_numpy(dtype=np.int64),
                                         d.gene_b.map(node).to_numpy(dtype=np.int64))),
                        dtype=torch.long, device=device)


def graph_edges(edges: torch.Tensor, labels: torch.Tensor) -> list[torch.Tensor]:
    empty = torch.empty((0, 2), dtype=torch.long, device=edges.device)
    out = [empty]
    for value in (1, 0):
        e = edges[labels == value]
        out.append(torch.cat((e, e[:, [1, 0]]), dim=0))
    return out


def fit(x: torch.Tensor, train_edge: torch.Tensor, train_y: torch.Tensor,
        valid_edge: torch.Tensor | None, valid_y: torch.Tensor | None,
        epochs: int) -> tuple[GiGCN, int, float]:
    torch.manual_seed(SEED)
    model = GiGCN().to(x.device)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"],
                           weight_decay=CONFIG["weight_decay"])
    graph = graph_edges(train_edge, train_y)
    pos = int(train_y.sum())
    pos_weight = torch.tensor((len(train_y) - pos) / max(pos, 1), device=x.device)
    best_auc, best_epoch, stale = -1.0, 1, 0
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        z = model.encode(x, graph)
        logits = model.pair_logits(z, train_edge)
        margin = logits[:, 1] - logits[:, 0]
        loss = F.binary_cross_entropy_with_logits(margin, train_y.float(),
                                                  pos_weight=pos_weight)
        loss = loss + CONFIG["lambda_disc"] * model.factor_loss(z)
        loss.backward()
        opt.step()
        if valid_edge is None:
            if epoch == epochs or epoch % 10 == 0:
                print(f"full fit epoch {epoch}: loss={loss.item():.4f}", flush=True)
            continue
        model.eval()
        with torch.no_grad():
            z = model.encode(x, graph)
            v = model.pair_logits(z, valid_edge)
            margin = (v[:, 1] - v[:, 0]).cpu().numpy()
        auc = roc_auc_score(valid_y.cpu().numpy(), margin)
        if auc > best_auc + 1e-4:
            best_auc, best_epoch, stale = auc, epoch, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or stale >= CONFIG["patience"]:
            print(f"cold-gene epoch {epoch}: AUROC={auc:.4f}; best={best_auc:.4f} "
                  f"at {best_epoch}", flush=True)
        if stale >= CONFIG["patience"]:
            break
    return model, best_epoch, best_auc


def main() -> None:
    split = slb.SPLIT
    genes, x_np = features()
    node = {g: i for i, g in enumerate(genes)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(x_np, device=device)
    train = slb.load("train")
    train = train[train.species == "human"].reset_index(drop=True)
    edges = tensor_edges(train, node, device)
    y = torch.tensor(train.label.to_numpy(dtype=np.int64), device=device)
    meta_path = CHECKPOINT.with_suffix(".json")
    fingerprint = {
        "benchmark_manifest_sha256": digest(slb.BENCH / "manifest.json"),
        "train_sha256": digest(slb.BENCH / "train.parquet"),
        "go_sha256": digest(B.BUNDLE / "human/go.parquet"),
        "go_edges_sha256": digest(B.BUNDLE / "_go/edges.parquet"),
        "features_sha256": hashlib.sha256(x_np.tobytes()).hexdigest(),
        "source_sha256": digest(Path(__file__)),
        "config": CONFIG,
    }
    model = GiGCN().to(device)
    cache_ok = False
    if CHECKPOINT.exists() and meta_path.exists():
        saved = json.loads(meta_path.read_text())
        cache_ok = (all(saved.get(k) == v for k, v in fingerprint.items())
                    and saved.get("checkpoint_sha256") == digest(CHECKPOINT))
        if cache_ok:
            model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
            print(f"verified cached {CHECKPOINT}", flush=True)
    if not cache_ok:
        rng = np.random.default_rng(SEED)
        seen_genes = np.array(sorted(set(train.gene_a) | set(train.gene_b)))
        held = set(rng.choice(seen_genes, int(CONFIG["validation_gene_fraction"] * len(seen_genes)),
                              replace=False))
        a, b = train.gene_a.isin(held).to_numpy(), train.gene_b.isin(held).to_numpy()
        tr, va = ~a & ~b, a & b
        if va.sum() < 100 or train.label.to_numpy()[va].sum() < 5:
            raise ValueError("internal gene-holdout has too few validation positives")
        print(f"internal cold-gene holdout: {tr.sum():,} fit, {va.sum():,} validation pairs; "
              f"{train.label.to_numpy()[va].sum()} validation positives", flush=True)
        _, best_epoch, best_auc = fit(x, edges[tr], y[tr], edges[va], y[va],
                                      CONFIG["max_epochs"])
        print(f"selected {best_epoch} epochs at internal AUROC {best_auc:.4f}", flush=True)
        model, _, _ = fit(x, edges, y, None, None, best_epoch)
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), CHECKPOINT)
        meta_path.write_text(json.dumps({**fingerprint, "checkpoint_sha256": digest(CHECKPOINT),
                                         "selected_epochs": best_epoch,
                                         "internal_cold_gene_auc": best_auc}, indent=2))
    model.eval()
    with torch.no_grad():
        z = model.encode(x, graph_edges(edges, y))
    target = slb.load(split).copy()
    target["score"] = np.nan
    human = target[target.species == "human"]
    query = tensor_edges(human, node, device)
    preds = []
    with torch.no_grad():
        for batch in query.split(4096):
            logits = model.pair_logits(z, batch)
            preds.append(torch.sigmoid(logits[:, 1] - logits[:, 0]).cpu().numpy())
    target.loc[human.index, "score"] = np.concatenate(preds)
    out = slb.write(target, "gigcn__binary_go", split)
    out.with_suffix(".provenance.json").write_text(json.dumps({
        "source": "wqz2469/GIGCN @ fdd7dd055a684a8fac4d6715b8ad8c449d21bf56",
        "adaptation": "binary SL vs measured neutral; empty SV channel; source DINES factors and pair-correlation decoder",
        "labels": f"{slb.BENCH.name}/train.parquet only",
        "features": "GO annotations from SLB bundle, IGI/ND excluded, propagated TF-IDF SVD",
        "checkpoint_sha256": digest(CHECKPOINT),
        "native_human_coverage": len(human),
    }, indent=2))
    slb.evaluate("gigcn__binary_go", split)


if __name__ == "__main__":
    main()
