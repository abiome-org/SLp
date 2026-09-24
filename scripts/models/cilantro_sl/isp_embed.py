"""Cilantro-SL stage 0: Geneformer in-silico knockout (delete) delta embeddings for (SLB cell line x SLB gene).

usage: python isp_embed.py <out.parquet>

Faithful to Cilantro-SL notebooks 1_tokenizer / 2_perturber + isp/viability_perturber.py (+ kaileyhh/geneformer
InSilicoPerturber(perturb_type="delete", emb_mode="cell", cell_emb_style="mean_pool", emb_layer=-1) and
EmbExtractor(emb_layer=-1)), re-implemented as one batched loop (the original re-embeds all 1479 DepMap lines once
per gene through temp datasets on disk):
  * expression: DepMap 24Q4 OmicsExpressionProteinCodingGenesTPMLogp1 (log2 TPM+1) used directly as 'counts',
    genes = top-500 HVGs over all DepMap lines (scanpy seurat flavour, as in notebook 1) + SL genes
  * tokenisation = Geneformer V1 rank-value encoding: x / n_counts * 1e4 / gene_median, non-zero genes ranked
    descending, truncated to 2048, no <cls>/<eos> (special_token=False). (n_counts is a per-cell constant and does
    not change the ranking.)
  * model: gf-12L-30M-i2048 with ITS OWN gc30M token/median dictionaries (Cilantro's fork points to the gc95M
    dictionaries, which do not match this model's vocabulary - see notes)
  * cell embedding = mean over tokens of hidden_states[11] (layer 12 + emb_layer -1); delta = emb(original) -
    emb(gene token deleted); viability label = DepMap CRISPR gene effect of that (line, gene).
SLB adaptations (documented in notes/models/cilantro_sl.md):
  * cells: only the human SLB contexts (contexts.parquet depmap_id; hTERT-RPE1 -> mean profile of the DepMap
    RPE1-ss* clones; contexts with no DepMap expression are skipped)
  * 'SL genes' = the SLB genes that occur in that context (train/dev/test inputs; no labels used), so the
    2048-token window is spent on genes the benchmark asks about; only those genes are knocked out.
A (context, gene) delta exists only if the gene is expressed and inside that line's 2048-token window.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "esm4sl"))
import slb  # noqa: E402
from cellfeat import context_models  # noqa: E402

torch.set_num_threads(8)
GF = slb.ROOT / "external/models/cilantro_sl/gf_weights"
INPUT = 2048
BS = int(os.environ.get("GF_BATCH", 24))


def depmap(name):
    m = pd.read_csv(slb.RAW / "depmap" / name, index_col=0)
    ent = [c.split(" (")[1].rstrip(")") if " (" in c else None for c in m.columns]
    h = slb.hgnc().dropna(subset=["entrez_id", "ensembl_gene_id"])
    e2ens = dict(zip(h.entrez_id.astype(int).astype(str), h.ensembl_gene_id))
    m.columns = [e2ens.get(e) for e in ent]
    return m.loc[:, [c is not None for c in m.columns]].T.groupby(level=0).first().T


def main(out):
    import scanpy as sc
    import anndata as ad
    from transformers import BertForMaskedLM

    tok = pickle.load(open(GF / "dicts/token_dictionary_gc30M.pkl", "rb"))
    med = pickle.load(open(GF / "dicts/gene_median_dictionary_gc30M.pkl", "rb"))
    expr = depmap("OmicsExpressionProteinCodingGenesTPMLogp1_24Q4.csv")
    eff = depmap("CRISPRGeneEffect_24Q4.csv")
    expr = expr.loc[:, [g in tok and g in med for g in expr.columns]]
    a = ad.AnnData(expr.to_numpy(np.float32))
    a.var_names = expr.columns
    sc.pp.highly_variable_genes(a, n_top_genes=500, inplace=True)
    hvg = set(expr.columns[a.var["highly_variable"].to_numpy()])
    print(f"expression {expr.shape}, HVG {len(hvg)}", file=sys.stderr)

    sym2ens = dict(slb.hgnc().dropna(subset=["ensembl_gene_id"])[["symbol", "ensembl_gene_id"]].values)
    ctx_genes: dict[str, set] = {}
    for split in ["train", "dev", "test"]:
        try:
            d = slb.load(split)
        except FileNotFoundError:
            continue
        d = d[d.species == "human"]
        for c, g in pd.concat([d[["context_id", "gene_a"]].set_axis(["c", "g"], axis=1),
                               d[["context_id", "gene_b"]].set_axis(["c", "g"], axis=1)]).drop_duplicates().values:
            ctx_genes.setdefault(c, set()).add(g)

    model = BertForMaskedLM.from_pretrained(GF / "gf-12L-30M-i2048", output_hidden_states=True)
    dev = os.environ.get("GF_DEVICE", "cuda")
    model = (model.half() if dev == "cuda" else model).to(dev).eval()
    layer = 12 - 1

    def cell_emb(batch_ids):
        L = max(len(x) for x in batch_ids)
        ids = torch.zeros((len(batch_ids), L), dtype=torch.long)
        mask = torch.zeros((len(batch_ids), L), dtype=torch.long)
        for i, x in enumerate(batch_ids):
            ids[i, :len(x)] = torch.as_tensor(x)
            mask[i, :len(x)] = 1
        ids, mask = ids.to(dev), mask.to(dev)
        with torch.no_grad():
            h = model(input_ids=ids, attention_mask=mask).hidden_states[layer].float()
        m = mask.unsqueeze(2).float()
        return ((h * m).sum(1) / m.sum(1)).cpu().numpy()

    rows, t0 = [], time.time()
    only = os.environ.get("GF_CONTEXTS")   # debugging: comma-separated context_ids
    for ctx, ids in context_models().items():
        if only and ctx not in only.split(","):
            continue
        ids = [i for i in ids if i in expr.index]
        if not ids or ctx not in ctx_genes:
            print(f"{ctx}: no DepMap expression -> skipped", file=sys.stderr)
            continue
        slb_ens = {sym2ens[g]: g for g in ctx_genes[ctx] if g in sym2ens and sym2ens[g] in expr.columns}
        genes = [g for g in expr.columns if g in hvg or g in slb_ens]
        x = expr.loc[ids, genes].mean().to_numpy()
        norm = x / np.array([med[g] for g in genes])
        nz = np.nonzero(x > 0)[0]
        order = nz[np.argsort(-norm[nz], kind="stable")][:INPUT]
        toks = [tok[genes[i]] for i in order]
        orig = cell_emb([toks])[0]
        pos = [(k, genes[i]) for k, i in enumerate(order) if genes[i] in slb_ens]
        ge = eff.loc[[i for i in ids if i in eff.index]].mean() if any(i in eff.index for i in ids) else None
        if os.environ.get("GF_MAXPOS"):
            pos = pos[:int(os.environ["GF_MAXPOS"])]
        for s in range(0, len(pos), BS):
            chunk = pos[s:s + BS]
            pert = cell_emb([toks[:k] + toks[k + 1:] for k, _ in chunk])
            for (k, g), p in zip(chunk, pert):
                v = float(ge[g]) if ge is not None and g in ge.index and pd.notna(ge[g]) else np.nan
                rows.append((ctx, slb_ens[g], g, k, len(toks), v, *(orig - p)))
        print(f"{ctx}: {len(toks)} tokens, {len(pos)}/{len(slb_ens)} SLB genes knocked out, "
              f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    cols = ["context_id", "gene", "ensembl", "rank", "n_tokens", "viability"] + [f"d{i}" for i in range(512)]
    df = pd.DataFrame(rows, columns=cols)
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {len(df)} (context, gene) deltas", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
