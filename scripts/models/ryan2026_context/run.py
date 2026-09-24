"""Context-specific paralog SL random forest (Kebabci, ..., Ryan; bioRxiv 2026, doi 10.64898/2026.01.19.700065;
github cancergenetics/context_specific_paralog_SL) re-built for SLB rows (cell line x pair).

Per (cell line, pair) features, as in scripts/rf_contextualized.py (16 'contextualised' features):
  rMaxExp_A1A2 / rMinExp_A1A2      max/min of the two genes' expression rank within the line (DepMap 24Q4 TPM)
  max_ranked_A1A2 / min_ranked_A1A2  max/min of the genes' essentiality rank within the line (Chronos 24Q4, rank 1 = most essential)
  max_cn / min_cn                   relative copy number (OmicsCNGene 24Q4)
  Protein_Altering / Damaging       either gene carries a damaging mutation (24Q4 damaging matrix; the hotspot /
                                    protein-altering split of the original is approximated by the same flag)
  min_sequence_identity             Ens111 paralog table (Zenodo 14973633); 0 for non-paralogs
  prediction_score                  De Kegel 2021 released RF score (DepMap-derived labels); 0 for non-paralogs
  weighted_PPI_essentiality/expression  shared interactors' mean essentiality rank / expression rank in the line,
                                    weighted by interaction confidence. DEVIATION: the paper used STRING combined
                                    scores; we use the shared bundle (data/interim/bundle/human/ppi.parquet: STRING
                                    without the experimental + text-mining channels, which import GI data, plus
                                    BioGRID physical). weight = 1 - prod(1 - s/1000) over kept channels, BioGRID physical
                                    >= 0.7; shared-interactor weight = mean of the two genes' weights.
  smallest_BP/CC_GO_essentiality, smallest_BP_GO_expression, go_CC_expression
                                    among GO BP/CC terms shared by both genes (bundle GO, IGI evidence dropped; term
                                    gene sets restricted to Ens111 paralog genes, 5 < size < 100 as in the notebooks),
                                    the smallest term's mean essentiality rank / expression in the line (max over ties)
Missing values as scripts/05_process_chunks.py: expression-type NaN -> 0, essentiality-type NaN -> 18000.
Variants:
  ryan2026_context__released_ctx   released contextualised_model.pickle (trained on GEMINI-scored Ito/Parrish/
                                   Klingbeil/Harle/CHyMErA labels => LEAKY; Harle and Parrish are SLB sources)
  ryan2026_context__slbtrain_ctx   same 16 features + RF hyper-parameters, refit on SLB train rows
  ryan2026_context__slbtrain_full  + the 22 De Kegel features ('full' model), refit on SLB train
Human only (needs DepMap cell-line omics). Lines absent from DepMap (hTERT-RPE1, C092) are left unscored.
"""
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dekegel2021"))
import omics  # noqa: E402
import slb  # noqa: E402
from features import FEATURES as DK22, feature_table as dk_table  # noqa: E402

CTX = ['rMaxExp_A1A2', 'rMinExp_A1A2', 'max_ranked_A1A2', 'min_ranked_A1A2', 'max_cn', 'min_cn', 'Protein_Altering',
       'Damaging', 'min_sequence_identity', 'prediction_score', 'weighted_PPI_essentiality',
       'weighted_PPI_expression', 'smallest_BP_GO_essentiality', 'smallest_CC_GO_essentiality',
       'smallest_BP_GO_expression', 'go_CC_expression']
EXPR_LIKE = ['weighted_PPI_expression', 'smallest_BP_GO_expression', 'go_CC_expression']
ESS_LIKE = ['weighted_PPI_essentiality', 'smallest_BP_GO_essentiality', 'smallest_CC_GO_essentiality']
REPO = slb.ROOT / "external/models/ryan_context_paralog_sl"
DK_DIR = slb.ROOT / "external/models/dekegel_paralog_sl/_slb" / slb.BENCH.name


def line_matrices(lines):
    ex = omics.ccle_expr().reindex(lines)
    es = omics.crispr().reindex(lines)
    cn = omics.ccle_cn().reindex(lines)
    mu = omics.ccle_mut().reindex(lines)
    ex_rank = ex.rank(axis=1, ascending=True)  # higher = more expressed
    es_rank = es.rank(axis=1, ascending=True)  # 1 = most essential
    return ex_rank, es_rank, cn, mu


def ppi_shared(pairs: pd.DataFrame):
    """Sparse (pair x interactor) weights of shared interactors."""
    p = pd.read_parquet(slb.ROOT / "data/interim/bundle/human/ppi.parquet")
    ch = [c for c in p.columns if c.startswith("string_")]
    w = 1 - np.prod([1 - p[c].to_numpy(float) / 1000 for c in ch], axis=0)
    w = np.where(p.biogrid_phys.to_numpy() > 0, np.maximum(w, 0.7), w)
    e = pd.DataFrame({"a": p.gene_a, "b": p.gene_b, "w": w})
    e = e[e.w > 0.15]
    e = pd.concat([e, e.rename(columns={"a": "b", "b": "a"})], ignore_index=True)
    nb = e.groupby("a").apply(lambda g: dict(zip(g.b, g.w)), include_groups=False).to_dict()
    rows = []
    for i, (a, b) in enumerate(zip(pairs.a, pairs.b)):
        na, nb_ = nb.get(a, {}), nb.get(b, {})
        for g in set(na) & set(nb_):
            if g not in (a, b):
                rows.append((i, g, (na[g] + nb_[g]) / 2))
    return pd.DataFrame(rows, columns=["i", "g", "w"])


def go_smallest(pairs: pd.DataFrame, universe: set):
    go = pd.read_parquet(slb.ROOT / "data/interim/bundle/human/go.parquet")
    out = {}
    for asp, name in [("P", "BP"), ("C", "CC")]:
        g = go[(go.aspect == asp) & go.gene.isin(universe)]
        sets = g.groupby("term").gene.apply(set)
        sets = sets[(sets.str.len() > 5) & (sets.str.len() < 100)]
        g2t = {}
        for t, s in sets.items():
            for x in s:
                g2t.setdefault(x, set()).add(t)
        best = []
        for a, b in zip(pairs.a, pairs.b):
            sh = g2t.get(a, set()) & g2t.get(b, set())
            if not sh:
                best.append([])
                continue
            m = min(len(sets[t]) for t in sh)
            best.append([t for t in sh if len(sets[t]) == m])
        out[name] = (best, sets)
    return out


def build(split: str) -> pd.DataFrame:
    d = slb.load(split)
    d = d[d.species == "human"].copy()
    ctx = slb.contexts().set_index("context_id").depmap_id
    d["line"] = d.context_id.map(ctx)
    d["key"] = slb.pair_key(d.gene_a, d.gene_b).values
    lines = sorted(d.line.dropna().unique())
    ex_rank, es_rank, cn, mu = line_matrices(lines)
    # paralog table features + De Kegel score
    dk = dk_table()
    pre = pd.read_csv(DK_DIR / f"{split}_pretrained.csv").set_index("example_id").score
    d["prediction_score"] = d.example_id.map(pre).fillna(0.0).to_numpy()
    x = dk.reindex(d.key.values)
    for c in DK22:
        d[c] = x[c].to_numpy()
    d["min_sequence_identity"] = d["min_sequence_identity"].fillna(0.0)

    def gv(mat, genes, lines_):
        cols = {g: i for i, g in enumerate(mat.columns)}
        rix = {l: i for i, l in enumerate(mat.index)}
        M = mat.to_numpy(float)
        out = np.full(len(genes), np.nan)
        for k, (g, l) in enumerate(zip(genes, lines_)):
            if l == l and g in cols and l in rix:
                out[k] = M[rix[l], cols[g]]
        return out

    for nm, mat in [("Exp", ex_rank), ("ess", es_rank), ("cn", cn), ("mut", mu)]:
        va, vb = gv(mat, d.gene_a, d.line), gv(mat, d.gene_b, d.line)
        d[f"{nm}_a"], d[f"{nm}_b"] = va, vb
    d["rMaxExp_A1A2"] = d[["Exp_a", "Exp_b"]].max(axis=1)
    d["rMinExp_A1A2"] = d[["Exp_a", "Exp_b"]].min(axis=1)
    d["max_ranked_A1A2"] = d[["ess_a", "ess_b"]].max(axis=1)
    d["min_ranked_A1A2"] = d[["ess_a", "ess_b"]].min(axis=1)
    d["max_cn"] = d[["cn_a", "cn_b"]].max(axis=1)
    d["min_cn"] = d[["cn_a", "cn_b"]].min(axis=1)
    d["Damaging"] = (d[["mut_a", "mut_b"]].fillna(0).max(axis=1) > 0).astype(float)
    d["Protein_Altering"] = d["Damaging"]
    # neighbourhood features, per unique pair then gathered per line
    up = d[["key", "gene_a", "gene_b"]].drop_duplicates("key").reset_index(drop=True).rename(
        columns={"gene_a": "a", "gene_b": "b"})
    kidx = {k: i for i, k in enumerate(up.key)}
    li = {l: i for i, l in enumerate(lines)}
    ri = d.key.map(kidx).to_numpy()
    lj = d.line.map(li)
    ok = lj.notna().to_numpy()
    lj = lj.fillna(0).astype(int).to_numpy()
    EXr = ex_rank.to_numpy(float)
    ESr = es_rank.to_numpy(float)
    gcol = {g: i for i, g in enumerate(ex_rank.columns)}
    ecol = {g: i for i, g in enumerate(es_rank.columns)}
    sh = ppi_shared(up)
    for nm, M, cols in [("weighted_PPI_expression", EXr, gcol), ("weighted_PPI_essentiality", ESr, ecol)]:
        s = sh[sh.g.isin(cols)]
        num = np.zeros((len(up), len(lines)))
        den = np.zeros(len(up))
        gi = s.g.map(cols).to_numpy()
        vals = np.nan_to_num(M[:, gi].T)  # (edges, lines)
        np.add.at(num, s.i.to_numpy(), vals * s.w.to_numpy()[:, None])
        np.add.at(den, s.i.to_numpy(), s.w.to_numpy())
        with np.errstate(invalid="ignore", divide="ignore"):
            F = num / den[:, None]
        v = np.where(ok, F[ri, lj], np.nan)
        d[nm] = v
    universe = {g for k in dk_table().index for g in k.split("|")}
    gos = go_smallest(up, universe)
    for name, (best, sets) in gos.items():
        for feat, M, cols in [("expression", EXr, gcol), ("essentiality", ESr, ecol)]:
            F = np.full((len(up), len(lines)), np.nan)
            cache = {}
            for i, terms in enumerate(best):
                if not terms:
                    continue
                vs = []
                for t in terms:
                    if t not in cache:
                        idx = [cols[g] for g in sets[t] if g in cols]
                        cache[t] = np.nanmean(M[:, idx], axis=1) if len(idx) >= 2 else np.full(len(lines), np.nan)
                    vs.append(cache[t])
                F[i] = np.nanmax(np.vstack(vs), axis=0)
            d[f"smallest_{name}_GO_{feat}"] = np.where(ok, F[ri, lj], np.nan)
    d["go_CC_expression"] = d["smallest_CC_GO_expression"]
    for c in EXPR_LIKE:
        d[c] = d[c].fillna(0)
    for c in ESS_LIKE:
        d[c] = d[c].fillna(18000)
    d["has_ctx"] = d[["rMaxExp_A1A2", "max_ranked_A1A2"]].notna().all(axis=1)
    return d


def cached(split):
    p = omics.CACHE / f"ryan2026ctx_{slb.BENCH.name}_{split}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    d = build(split)
    d.to_parquet(p)
    return d


def main(split):
    ev = cached(split)
    tr = cached("train")
    tr = tr[tr.has_ctx]
    m = ev.has_ctx.to_numpy()
    full = slb.load(split)[["example_id"]]
    outs = {}
    rel = pickle.load(gzip.open(REPO / "data/output/models/contextualised_model.pickle.gz"))
    s = np.full(len(ev), np.nan)
    s[m] = rel.predict_proba(ev.loc[m, list(rel.feature_names_in_)])[:, 1]
    outs["ryan2026_context__released_ctx"] = s
    for name, feats in [("ryan2026_context__slbtrain_ctx", CTX), ("ryan2026_context__slbtrain_full", CTX + DK22)]:
        rf = RandomForestClassifier(n_estimators=600, max_depth=20, max_features=0.2, min_samples_leaf=4,
                                    random_state=8, n_jobs=8)
        rf.fit(tr[feats].fillna(0).to_numpy(), tr.label.to_numpy())
        s = np.full(len(ev), np.nan)
        s[m] = rf.predict_proba(ev.loc[m, feats].fillna(0).to_numpy())[:, 1]
        outs[name] = s
        imp = sorted(zip(rf.feature_importances_, feats), reverse=True)[:6]
        print(name, "top features:", ", ".join(f"{n}={v:.3f}" for v, n in imp), file=sys.stderr)
    print(f"train rows {len(tr):,} ({int(tr.label.sum())} SL); {split} human rows with line data {m.sum():,}/{len(ev):,}",
          file=sys.stderr)
    for name, s in outs.items():
        out = full.merge(pd.DataFrame({"example_id": ev.example_id.values, "score": s}), on="example_id", how="left")
        slb.write(out, name, split)
        slb.evaluate(name, split)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else slb.SPLIT)
