"""PAGAN genes-to-pairs adaptation on the SLB-clean GO/PPI bundle.

The published approach trains on single-gene essentiality, inserts each
candidate pair as a zero-feature gene node, and connects that node to the
union of its members' PPI/paralog neighbors and GO terms. The architecture
here follows the authors' two-layer HeteroConv/SAGE, with fixed 16 hidden
units, sum aggregation, tanh, 0.5 dropout and 0.01 Adam. It reads no SL
pair labels. We replace the authors' KG with the SLB evidence-filtered bundle
and use their released single-gene feature tables and essentiality labels.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import slb  # noqa: E402


BUNDLE = slb.ROOT / "data/interim/bundle"
AUTHORS = slb.ROOT / "external/models/pagan/data"
CKPTS = slb.ROOT / "external/models/pagan/_slb" / slb.BENCH.name / "clean_bundle_v2"
FEATURES_HUMAN = ["zscore_mis", "zscore_syn", "f_parameter", "GDI"]
FEATURES_SCER = [
    "dn_ds", "chemical_compound_accumulation", "chronological_lifespan",
    "competitive_fitness", "desiccation_resistance", "haploinsufficient",
    "heat_sensitivity", "metal_resistance", "oxidative_stress_resistance",
    "replicative_lifespan", "resistance_to_chemicals", "respiratory_growth",
    "stress_resistance", "toxin_resistance", "utilization_of_nitrogen_source",
    "vacuolar_morphology", "vegetative_growth",
]
ASPECT = {"P": "BP", "F": "MF", "C": "CC"}


def edge(src: np.ndarray, dst: np.ndarray) -> torch.Tensor:
    if len(src) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(np.stack((src, dst)), dtype=torch.long)


def gene_features(sp: str, genes: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Authors' single-gene features and essentiality labels, canonicalized."""
    if sp == "human":
        h = slb.hgnc().dropna(subset=["ensembl_gene_id", "symbol"])
        ens_to_symbol = dict(zip(h.ensembl_gene_id, h.symbol))
        df = pd.read_csv(AUTHORS / "processed/human_genes_features_20230414.tsv", sep="\t")
        df["gene"] = df.ENSG_ID.map(ens_to_symbol)
        essential = set(pd.read_csv(AUTHORS / "raw/essentials_K562.tsv", sep="\t").ENSG_ID)
        df["y"] = df.ENSG_ID.isin(essential).astype(np.float32)
        cols = FEATURES_HUMAN
    else:
        df = pd.read_csv(AUTHORS / "processed/yeast_features_1000_20240717.tsv", sep="\t")
        df["gene"] = df.gene_id.astype(str)
        et = pd.read_csv(AUTHORS / "processed/yeast_essential_nonessential_20240612.tsv", sep="\t")
        labels = dict(zip(et.gene_id, et.type))
        df["y"] = df.gene.map(labels).map({"essential": 1., "non_essential": 0.})
        cols = FEATURES_SCER
    df = df.dropna(subset=["gene"]).drop_duplicates("gene").set_index("gene")
    raw = df.reindex(genes)
    X = raw[cols].apply(pd.to_numeric, errors="coerce")
    med = X.median().fillna(0)
    X = X.fillna(med)
    mu = X.mean()
    sd = X.std().replace(0, 1).fillna(1)
    X = ((X - mu) / sd).clip(-8, 8).to_numpy(dtype=np.float32)
    y = raw.y.fillna(0).to_numpy(dtype=np.float32)
    mask = raw.y.notna().to_numpy()
    return X, y, mask


def gene_universe(sp: str) -> set[str]:
    """Fix the unlabeled node set across dev and test, including isolated genes."""
    genes: set[str] = set()
    for name in ("train", "dev", "test_inputs"):
        d = pd.read_parquet(slb.BENCH / f"{name}.parquet",
                            columns=["species", "gene_a", "gene_b"])
        d = d.loc[d.species == sp]
        genes.update(d.gene_a)
        genes.update(d.gene_b)
    return genes


def graph(sp: str, targets: pd.DataFrame):
    go = pd.read_parquet(BUNDLE / sp / "go.parquet", columns=["gene", "term", "aspect"])
    ppi = pd.read_parquet(BUNDLE / sp / "ppi.parquet")
    ppi = ppi.loc[(ppi.biogrid_phys >= 2) | (ppi.string_database >= 700), ["gene_a", "gene_b"]]
    par = pd.read_parquet(slb.ROOT / "data/interim/orthology_extra/diamond_self_paralogs_all.parquet")
    prefix = sp + ":"
    par = par.loc[(par.identity >= .30) & par.u.str.startswith(prefix) & par.v.str.startswith(prefix)]
    par = pd.DataFrame({"gene_a": par.u.str.removeprefix(prefix),
                        "gene_b": par.v.str.removeprefix(prefix)})
    genes = sorted(set(go.gene) | set(ppi.gene_a) | set(ppi.gene_b) |
                   set(par.gene_a) | set(par.gene_b) |
                   gene_universe(sp))
    gid = {g: i for i, g in enumerate(genes)}
    X, y, trainable = gene_features(sp, genes)
    data = HeteroData()
    data["GENE"].x = torch.from_numpy(X)
    data["GENE"].y = torch.from_numpy(y.copy())
    data["GENE"].trainable = torch.from_numpy(trainable.copy())
    neighbors: dict[tuple[str, str, str], list[list[int]]] = {}

    for rel, pairs in [("PPI", ppi), ("paralog", par)]:
        aa = pairs.gene_a.map(gid).dropna().to_numpy(dtype=np.int64)
        bb = pairs.gene_b.map(gid).dropna().to_numpy(dtype=np.int64)
        if len(aa) != len(bb):
            raise ValueError(f"Unmapped {rel} endpoint")
        a = np.concatenate((aa, bb))
        b = np.concatenate((bb, aa))
        key = ("GENE", rel, "GENE")
        data[key].edge_index = edge(a, b)
        adj = [set() for _ in genes]
        for src, dst in zip(a, b):
            adj[int(src)].add(int(dst))
        neighbors[key] = [sorted(v)[:30] for v in adj]

    hierarchy = pd.read_parquet(BUNDLE / "_go/edges.parquet")
    for aspect, node in ASPECT.items():
        d = go.loc[go.aspect == aspect, ["gene", "term"]].drop_duplicates()
        terms = sorted(set(d.term))
        tid = {t: i for i, t in enumerate(terms)}
        gd = d.gene.map(gid).to_numpy(dtype=np.int64)
        td = d.term.map(tid).to_numpy(dtype=np.int64)
        deg = np.bincount(td, minlength=len(terms))
        # Degree features stand in for the authors' OneHotDegree GO features.
        onehot = np.eye(33, dtype=np.float32)[np.minimum(deg, 32)]
        data[node].x = torch.from_numpy(onehot)
        data[("GENE", f"gene_to_{node}", node)].edge_index = edge(gd, td)
        rev = (node, f"rev_gene_to_{node}", "GENE")
        data[rev].edge_index = edge(td, gd)
        adj = [[] for _ in genes]
        for g, t in zip(gd, td):
            adj[int(g)].append(int(t))
        neighbors[rev] = [sorted(set(v))[:32] for v in adj]
        h = hierarchy.loc[hierarchy.child.isin(tid) & hierarchy.parent.isin(tid)]
        c = h.child.map(tid).to_numpy(dtype=np.int64)
        p = h.parent.map(tid).to_numpy(dtype=np.int64)
        data[(node, node.lower() + "_hierarchy", node)].edge_index = edge(c, p)
        data[(node, "rev_" + node.lower() + "_hierarchy", node)].edge_index = edge(p, c)
        print(f"{sp} {node}: {len(terms):,} terms, {len(d):,} annotations, {len(h):,} hierarchy edges")
    print(f"{sp}: {len(genes):,} genes, {trainable.sum():,} single-gene labels, "
          f"{len(ppi):,} PPI, {len(par):,} paralogs")
    return data, gid, neighbors


class PaganN2P(torch.nn.Module):
    def __init__(self, relations: list[tuple[str, str, str]]):
        super().__init__()
        self.conv1 = HeteroConv({
            r: SAGEConv((-1, -1), 16, aggr="sum", normalize=True,
                        root_weight=r[0] == r[2] and r[0] != "GENE") for r in relations
        }, aggr="sum")
        self.conv2 = HeteroConv({
            r: SAGEConv((-1, -1), 16, aggr="sum", normalize=True,
                        root_weight=True) for r in relations
        }, aggr="sum")
        self.out = torch.nn.Linear(16, 1)

    def forward(self, data: HeteroData) -> torch.Tensor:
        h = self.conv1(data.x_dict, data.edge_index_dict)
        h = {k: F.dropout(torch.tanh(v), p=.5, training=self.training) for k, v in h.items()}
        h = self.conv2(h, data.edge_index_dict)
        return self.out(F.dropout(torch.tanh(h["GENE"]), p=.5, training=self.training)).squeeze(-1)


def train_model(sp: str, data: HeteroData, device: torch.device) -> PaganN2P:
    torch.manual_seed(31415)
    np.random.seed(31415)
    model = PaganN2P(data.edge_types).to(device)
    d = data.to(device)
    # Initialize lazy SAGE layers before optimizer construction.
    with torch.no_grad():
        model.eval()(d)
    path = CKPTS / f"{sp}.pt"
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return model.eval()
    indices = torch.where(d["GENE"].trainable)[0]
    order = indices[torch.randperm(len(indices), device=device)]
    nval = max(1, len(order) // 10)
    val, tr = order[:nval], order[nval:]
    opt = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=5e-4)
    best_loss, wait, best_state = float("inf"), 0, None
    for epoch in range(1, 81):
        model.train()
        opt.zero_grad()
        z = model(d)
        loss = F.binary_cross_entropy_with_logits(z[tr], d["GENE"].y[tr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(d)
            vl = F.binary_cross_entropy_with_logits(pred[val], d["GENE"].y[val]).item()
            auc = roc_auc_score(d["GENE"].y[val].cpu(), pred[val].cpu())
        print(f"{sp} epoch {epoch:02d} train={loss.item():.4f} gene-val={vl:.4f} AUROC={auc:.3f}", flush=True)
        if vl < best_loss - 1e-4:
            best_loss, wait = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 8:
                break
    model.load_state_dict(best_state)
    CKPTS.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, path)
    return model.eval()


@torch.no_grad()
def predict(model: PaganN2P, data: HeteroData, gid: dict[str, int],
            neighbors: dict, target: pd.DataFrame, device: torch.device) -> np.ndarray:
    d = data.to(device)
    model.eval()
    h = model.conv1(d.x_dict, d.edge_index_dict)
    h = {k: torch.tanh(v) for k, v in h.items()}
    pair_ix = target.gene_a.map(gid).fillna(-1).to_numpy(dtype=np.int32)
    pair_jx = target.gene_b.map(gid).fillna(-1).to_numpy(dtype=np.int32)
    score = np.full(len(target), np.nan, dtype=float)
    pair_rels = [r for r in data.edge_types if r[2] == "GENE" and r in neighbors]
    for start in range(0, len(target), 2048):
        stop = min(start + 2048, len(target))
        ia, ib = pair_ix[start:stop], pair_jx[start:stop]
        valid = np.where((ia >= 0) & (ib >= 0))[0]
        if not len(valid):
            continue
        pair_x = torch.zeros((len(valid), d["GENE"].x.shape[1]), device=device)
        pair_h = None
        links = {}
        for rel in pair_rels:
            adj = neighbors[rel]
            sources, destinations = [], []
            for local, row in enumerate(valid):
                a, b = int(ia[row]), int(ib[row])
                members = sorted((set(adj[a]) | set(adj[b])) - ({a, b} if rel[0] == "GENE" else set()))
                sources.extend(members)
                destinations.extend([local] * len(members))
            links[rel] = edge(np.array(sources), np.array(destinations)).to(device)
            conv = model.conv1.convs[rel]
            src = d[rel[0]].x
            v = conv((src, pair_x), links[rel], size=(len(src), len(valid)))
            pair_h = v if pair_h is None else pair_h + v
        pair_h = torch.tanh(pair_h)
        out_h = None
        for rel in pair_rels:
            conv = model.conv2.convs[rel]
            src = h[rel[0]]
            v = conv((src, pair_h), links[rel], size=(len(src), len(valid)))
            out_h = v if out_h is None else out_h + v
        p = torch.sigmoid(model.out(torch.tanh(out_h)).squeeze(-1)).cpu().numpy()
        score[start + valid] = p
        if start % 20000 == 0:
            print(f"pair inference {start:,}/{len(target):,}", flush=True)
    return score


def main() -> None:
    target = slb.load(slb.SPLIT).copy()
    target["score"] = np.nan
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for sp in ("human", "scer"):
        rows = target.loc[target.species == sp]
        if rows.empty:
            continue
        data, gid, neighbors = graph(sp, rows)
        model = train_model(sp, data, device)
        target.loc[rows.index, "score"] = predict(model, data, gid, neighbors, rows, device)
        del model, data
        torch.cuda.empty_cache()
    out = slb.write(target, "pagan__clean_n2p", slb.SPLIT)
    out.with_suffix(".provenance.json").write_text(json.dumps({
        "method": "PAGAN genes-to-pairs, two-layer hetero GraphSAGE adaptation",
        "pair_supervision": "none; single-gene essentiality only",
        "human_single_gene_labels": "PAGAN essentials_K562.tsv",
        "scer_single_gene_labels": "PAGAN yeast_essential_nonessential_20240612.tsv",
        "gene_features": "PAGAN strict human and 17 yeast single-gene features",
        "graph": "SLB bundle: GO minus IGI, physical/curated PPI, DIAMOND paralogs",
        "coverage": "human and budding yeast; fission yeast tied",
        "adaptation": "full-batch training, one fixed 16-unit setting, capped pair neighborhoods",
    }, indent=2))
    slb.evaluate("pagan__clean_n2p", slb.SPLIT)


if __name__ == "__main__":
    main()
