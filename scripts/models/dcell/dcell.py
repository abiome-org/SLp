"""DCell-style visible neural network (Ma et al. 2018), re-implemented in PyTorch and retrained on SLB train.

Architecture follows DCell / DrugCell (idekerlab/DrugCell code/drugcell_NN.py, MIT): one small subsystem per
GO term, fed by the states of its child terms plus the genes directly annotated to it, tanh + batchnorm,
with an auxiliary 1-unit head per term (aux loss weight 0.2) and a final head on the root.
Input genotype: binary gene-disruption vector (the two genes of the pair = 1).
Variants: faithful (per-gene input weights, as DCell) and --tied (each term receives only the number of
its directly annotated genes that are disrupted, i.e. an ontotype input). Under SLB's held-out gene
families the faithful variant cannot generalise: the input weights of dev/test genes are never trained.
Output: logit of P(SL). Loss: binary cross-entropy on SLB train labels (the original DCell regressed
Costanzo growth / GI scores; the published weights are therefore leaky for scer and are not used).

Ontology per species: GO (go-basic, is_a + part_of, all three namespaces; IGI annotations removed),
annotations propagated to ancestors, terms kept if they cover >= MIN_GENES genes of the species'
gene universe; a term whose gene set equals that of its only kept child is collapsed (DCell's
redundancy filter); a synthetic ROOT joins the three namespaces.

    python dcell.py --species scer --out x.parquet [--tied] [--device cuda]   (see run.sh; GO from the bundle)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import bundle_io as B  # noqa: E402


# ------------------------------------------------------------------ ontology
def build_ontology(species: str, min_genes: int = 6, max_frac: float = 0.3):
    parents, _ = B.go_dag()
    prop = B.go_propagated(species)
    genes_of = defaultdict(set)
    for g, ts in prop.items():
        for t in ts:
            genes_of[t].add(g)
    ann_genes = set(prop)
    ngenes = len(ann_genes)
    keep = {t for t, s in genes_of.items() if min_genes <= len(s) <= max_frac * ngenes}
    # kept-term DAG: parent of a kept term = nearest kept ancestors
    kparents: dict[str, set] = {}

    def kp(t):
        out = set()
        for p in parents.get(t, ()):
            out |= {p} if p in keep else kp(p)
        return out

    for t in keep:
        kparents[t] = kp(t)
    # collapse redundant: term with identical gene set to one of its kept parents -> drop the child
    changed = True
    while changed:
        changed = False
        for t in list(keep):
            for p in kparents[t]:
                if genes_of[p] == genes_of[t]:
                    # remove t: its children now point to t's parents
                    for c in keep:
                        if t in kparents[c]:
                            kparents[c] = (kparents[c] - {t}) | kparents[t]
                    keep.discard(t)
                    del kparents[t]
                    changed = True
                    break
    children = defaultdict(set)
    for t in keep:
        ps = kparents[t] or {"ROOT"}
        for p in ps:
            children[p].add(t)
    # transitive reduction is not needed for the VNN; direct genes = genes of t not covered by any child
    direct = {}
    for t in list(keep) + ["ROOT"]:
        gs = genes_of[t] if t != "ROOT" else ann_genes
        covered = set().union(*[genes_of[c] for c in children.get(t, ())]) if children.get(t) else set()
        direct[t] = sorted(gs - covered)
    # topological order: leaves first
    order, seen = [], set()

    def visit(t):
        if t in seen:
            return
        seen.add(t)
        for c in children.get(t, ()):
            visit(c)
        order.append(t)

    visit("ROOT")
    return order, {t: sorted(children.get(t, ())) for t in order}, direct


# ------------------------------------------------------------------ model
class VNN(nn.Module):
    def __init__(self, order, children, direct, gene_index, hidden=6, tied=False):
        super().__init__()
        self.tied = tied  # tied: a term sees only the COUNT of its disrupted direct genes (gene-agnostic)
        self.order, self.children_of = order, children
        self.hidden = hidden
        self.direct_idx = {t: torch.tensor([gene_index[g] for g in direct[t]], dtype=torch.long) for t in order}
        self.lin = nn.ModuleDict()
        self.bn = nn.ModuleDict()
        self.aux = nn.ModuleDict()
        for t in order:
            k = t.replace(":", "_").replace(".", "_")
            n_in = hidden * len(children[t]) + (min(len(direct[t]), 1) if tied else len(direct[t]))
            self.lin[k] = nn.Linear(max(n_in, 1), hidden)
            self.bn[k] = nn.BatchNorm1d(hidden)
            self.aux[k] = nn.Sequential(nn.Linear(hidden, 1), nn.Tanh(), nn.Linear(1, 1))
        self.keys = {t: t.replace(":", "_").replace(".", "_") for t in order}
        self.final = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def to(self, *a, **kw):
        m = super().to(*a, **kw)
        dev = next(self.parameters()).device
        self.direct_idx = {t: v.to(dev) for t, v in self.direct_idx.items()}
        return m

    def forward(self, x):
        state, aux = {}, []
        for t in self.order:
            parts = [state[c] for c in self.children_of[t]]
            di = self.direct_idx[t]
            if len(di):
                xi = x.index_select(1, di)
                parts.append(xi.sum(1, keepdim=True) if self.tied else xi)
            inp = torch.cat(parts, 1) if parts else x.new_zeros(x.shape[0], 1)
            k = self.keys[t]
            h = self.bn[k](torch.tanh(self.lin[k](inp)))
            state[t] = h
            aux.append(self.aux[k](h))
        return self.final(state["ROOT"]).squeeze(1), torch.cat(aux, 1)


# ------------------------------------------------------------------ training
def encode(pairs: pd.DataFrame, gene_index: dict, n: int) -> torch.Tensor:
    x = torch.zeros(len(pairs), n)
    for col in ("gene_a", "gene_b"):
        idx = pairs[col].map(gene_index)
        ok = idx.notna().to_numpy()
        x[np.flatnonzero(ok), idx[ok].astype(int).to_numpy()] = 1.0
    return x


def predict(model, x, device, bs=16384):
    model.eval()
    out = [torch.zeros(0)]
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(model(x[i:i + bs].to(device))[0].float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--split", default=B.SPLIT)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--neg_ratio", type=float, default=10.0, help="negatives per positive, resampled per epoch")
    ap.add_argument("--hidden", type=int, default=6)
    ap.add_argument("--min_genes", type=int, default=6)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tied", action="store_true",
                    help="gene-agnostic input (count of disrupted genes per term) so unseen genes generalise")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    torch.set_num_threads(int(os.environ.get("SLB_THREADS", 8)))

    tr = B.load_train()
    tr = tr[tr.species == a.species]
    ev = B.load_split(a.split)
    ev = ev[ev.species == a.species]
    t0 = time.time()
    order, children, direct = build_ontology(a.species, a.min_genes)
    genes = sorted(set().union(*map(set, direct.values())))
    gi = {g: i for i, g in enumerate(genes)}
    print(f"{a.species}: {len(order)} terms, {len(genes)} genes, depth-ordered; built in {time.time() - t0:.0f}s",
          flush=True)

    # internal validation: hold out 15% of train genes (pairs with both genes held out), no dev labels used
    tg = np.array(sorted(set(tr.gene_a) | set(tr.gene_b)))
    val_genes = set(rng.choice(tg, int(0.15 * len(tg)), replace=False))
    va_m = tr.gene_a.isin(val_genes) & tr.gene_b.isin(val_genes)
    tr_m = ~tr.gene_a.isin(val_genes) & ~tr.gene_b.isin(val_genes)
    trn, val = tr[tr_m], tr[va_m]
    if val.label.nunique() < 2:  # small species: gene-disjoint holdout empty/single-class -> random 15% pairs
        va_idx = rng.random(len(tr)) < 0.15
        trn, val = tr[~va_idx], tr[va_idx]
    print(f"train {len(trn):,} ({int(trn.label.sum()):,} pos), internal val {len(val):,} ({int(val.label.sum()):,} pos)")
    pos, neg = trn[trn.label == 1], trn[trn.label == 0]
    xval, yval = encode(val, gi, len(genes)), val.label.to_numpy()

    model = VNN(order, children, direct, gi, a.hidden, a.tied).to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.99), eps=1e-5)
    bce = nn.BCEWithLogitsLoss()
    best, best_state, bad = -1.0, None, 0
    for ep in range(a.epochs):
        t = time.time()
        model.train()
        nn_ = min(len(neg), int(a.neg_ratio * len(pos)))
        batch = pd.concat([pos, neg.sample(nn_, random_state=int(rng.integers(1e9)))]).sample(frac=1.0,
                                                                                             random_state=ep)
        x, y = encode(batch, gi, len(genes)), torch.tensor(batch.label.to_numpy(), dtype=torch.float32)
        tot = 0.0
        for i in range(0, len(x), a.bs):
            xb, yb = x[i:i + a.bs].to(a.device), y[i:i + a.bs].to(a.device)
            if len(xb) < 2:
                continue
            out, aux = model(xb)
            loss = bce(out, yb) + 0.2 * bce(aux, yb[:, None].expand_as(aux))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        pv = predict(model, xval, a.device)
        auc = roc_auc_score(yval, pv) if 0 < yval.sum() < len(yval) else 0.5 + 1e-6 * ep  # no val: keep last
        print(f"epoch {ep} loss {tot / len(x):.4f} val_auc {auc:.4f} {time.time() - t:.0f}s", flush=True)
        if auc > best:
            best, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    s = predict(model, encode(ev, gi, len(genes)), a.device)
    cov = (ev.gene_a.isin(gi) & ev.gene_b.isin(gi)).to_numpy()
    out = pd.DataFrame({"example_id": ev.example_id.to_numpy(), "score": s, "in_model": cov})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out)
    print(f"best internal val AUROC {best:.4f}; wrote {a.out} ({cov.mean():.1%} of {a.split} pairs with both genes in ontology)")


if __name__ == "__main__":
    main()
