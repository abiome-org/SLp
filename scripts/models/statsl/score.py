"""Scores for the unsupervised statistical SL-inference methods, from scripts/models/_common/statsl.py pair stats.

No SL labels are used anywhere (DepMap / DEMETER2 / CCLE / TCGA single-gene data + ISLE's phylogenetic profiles).
Human only: these methods need cancer cell-line and tumour omics, which exist only for human.

  daisy   Jerby-Arnon et al. 2014 (Cell): a pair is SL if (1) co-inactivation is under-represented in tumours
          (SoF, expression or copy number), (2) one gene is more essential when the other is inactive in
          cell-line screens (functional examination, shRNA/CRISPR) and (3) the genes are co-expressed.
          Score = number of the three tests passed at p<0.05, tie-broken by the mean percentile of the three
          statistics.
  isle    Lee et al. 2018 (Nat Commun): cascade (i) in-vitro functional examination -> (ii) under-represented
          co-inactivation in BOTH mRNA and SCNA -> (iii) co-inactivation associated with better patient
          survival (mRNA or SCNA) -> (iv) phylogenetic distance below the median. Score = number of consecutive
          steps passed (BH-FDR 0.2 as in isle.r for i-iii), tie-broken by the mean percentile of the step statistics.
  statsl__<component>  each single statistic as its own ranking (diagnostic variants).
usage: uv run python scripts/models/statsl/score.py [split]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import slb  # noqa: E402
import statsl  # noqa: E402

ESS = ["ess_crispr_expr", "ess_crispr_cn", "ess_crispr_mut", "ess_rnai_expr", "ess_rnai_cn", "ess_rnai_mut"]


def pct(x):
    return pd.Series(x).rank(pct=True).fillna(0.5).to_numpy()


def p_of(z):
    return stats.norm.sf(np.asarray(z, float))


def fdr(p):
    p = np.asarray(p, float)
    out = np.ones_like(p)
    ok = ~np.isnan(p)
    if ok.any():
        out[ok] = stats.false_discovery_control(p[ok], method="bh")
    return out


def scores(s: pd.DataFrame) -> dict[str, np.ndarray]:
    z_ess = s[ESS].max(axis=1).to_numpy()
    n_ess = s[ESS].notna().sum(axis=1).clip(lower=1).to_numpy()
    p_ess = np.minimum(1, p_of(z_ess) * n_ess)  # Bonferroni over the screens x alterations tried
    z_sof = s[["sof_expr", "sof_cna"]].max(axis=1).to_numpy()
    p_sof = np.minimum(1, 2 * p_of(z_sof))
    r = s["coexp_tcga"].to_numpy()
    n_t = 9000
    p_co = stats.norm.sf(np.arctanh(np.clip(r, -0.999, 0.999)) * np.sqrt(n_t - 3))
    daisy = (np.nan_to_num(p_sof, nan=1) < .05).astype(float) + (np.nan_to_num(p_ess, nan=1) < .05) + \
        (np.nan_to_num(p_co, nan=1) < .05)
    daisy = daisy + (pct(z_sof) + pct(z_ess) + pct(r)) / 3 * 0.999

    # ISLE cascade (FDR 0.2 as in isle.r)
    st1 = fdr(p_ess) <= 0.2
    st2 = (fdr(p_of(s.sof_expr)) < 0.2) & (fdr(p_of(s.sof_cna)) < 0.2)
    st3 = (fdr(p_of(s.surv_expr)) < 0.2) | (fdr(p_of(s.surv_cna)) < 0.2)
    ph = s.phylo_dist.to_numpy()
    st4 = ph < np.nanmedian(ph)
    steps = st1.astype(float)
    steps += st1 & st2
    steps += st1 & st2 & st3
    steps += st1 & st2 & st3 & st4
    z_surv = s[["surv_expr", "surv_cna"]].max(axis=1).to_numpy()
    tie = (pct(z_ess) + pct(s[["sof_expr", "sof_cna"]].min(axis=1)) + pct(z_surv) + pct(-ph)) / 4
    isle = steps + 0.999 * tie
    out = {"daisy": daisy, "isle": isle}
    for c in ESS + ["sof_expr", "sof_cna", "coexp_tcga", "coexp_ccle", "surv_expr", "surv_cna"]:
        out[f"statsl__{c}"] = s[c].to_numpy()
    out["statsl__phylo_neg_dist"] = -ph
    return out


def main(split):
    d = slb.load(split)
    h = d[d.species == "human"].copy()
    h["key"] = slb.pair_key(h.gene_a, h.gene_b).values
    s = statsl.pair_stats((split,)).drop_duplicates("key").set_index("key").reindex(h.key.unique())
    sc = scores(s.reset_index())
    for name, v in sc.items():
        m = pd.Series(v, index=s.index)
        out = d[["example_id"]].copy()
        out["score"] = np.nan
        out.loc[h.index, "score"] = m.reindex(h.key).to_numpy()
        slb.write(out, name, split)
        slb.evaluate(name, split)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else slb.SPLIT)
