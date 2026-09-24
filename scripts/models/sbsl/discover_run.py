"""Compute SBSL's discover_mutex feature for new (gene_a, gene_b, cancer) keys with DISCOVER (R, docker slb/sbsl).
Cached in external/models/sbsl/_slb/cache/keys_discover.parquet.  usage: python discover_run.py <split>"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "elisl"))
import common as C  # noqa: E402
import omics_data as O  # noqa: E402

KEY = ["gene_a", "gene_b", "cancer"]


def mutex_pvalues(bg, ev, gi, g1, g2, chunk=2000, kmax=30):
    """DISCOVER one-sided mutual-exclusivity p = P(X <= k_obs), X ~ PoissonBinomial(bg[g1, j] * bg[g2, j]).
    Exact DP over samples when k_obs <= kmax (validated equal to R pairwise.discover.test); for k_obs > kmax the
    refined normal approximation (Volkova 1996; skewness-corrected) is used to keep PANCAN (~9k samples) tractable.
    Genes without events -> 1 (SBSL default)."""
    from scipy.stats import norm
    i1 = np.array([gi.get(g, -1) for g in g1])
    i2 = np.array([gi.get(g, -1) for g in g2])
    p = np.ones(len(i1))
    ok = np.where((i1 >= 0) & (i2 >= 0) & (i1 != i2))[0]
    for s in range(0, len(ok), chunk):
        b = ok[s:s + chunk]
        q = bg[i1[b]] * bg[i2[b]]                                     # pairs x samples
        k = (ev[:, i1[b]] & ev[:, i2[b]]).sum(0).astype(int)          # observed co-occurrence
        small = k <= kmax
        if small.any():
            qs, ks = q[small], k[small]
            K = ks.max() + 1
            dist = np.zeros((len(ks), K + 1))
            dist[:, 0] = 1.0
            for j in range(qs.shape[1]):
                qj = qs[:, j:j + 1]
                new = dist * (1 - qj)
                new[:, 1:] += dist[:, :-1] * qj
                new[:, K] += dist[:, K] * qj[:, 0]                    # absorb overflow (> K) in the last state
                dist = new
            cdf = np.cumsum(dist[:, :K], axis=1)
            p[b[small]] = np.minimum(1.0, cdf[np.arange(len(ks)), ks])
        if (~small).any():
            ql, kl = q[~small], k[~small]
            mu = ql.sum(1)
            sd = np.sqrt((ql * (1 - ql)).sum(1))
            g = (ql * (1 - ql) * (1 - 2 * ql)).sum(1) / sd ** 3
            x = (kl + 0.5 - mu) / sd
            p[b[~small]] = np.clip(norm.cdf(x) + g * (1 - x ** 2) * norm.pdf(x) / 6, 0, 1)
    return p


WORK = C.ROOT / "external/models/sbsl/_slb"


def main(split):
    rows = C.pairs_with_type(C.human(split))
    keys = rows[KEY].drop_duplicates()
    p = WORK / "cache/keys_discover.parquet"
    have = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=KEY + ["discover_mutex"])
    todo = keys.merge(have[KEY], on=KEY, how="left", indicator=True).query("_merge == 'left_only'")[KEY]
    C.log(f"discover: {len(keys):,} keys, computing {len(todo):,}")
    if not len(todo):
        return
    CN, MU, ptype = O.tcga_cna(), O.tcga_mut(), O.patient_type()
    tmp = WORK / "discover_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    new = []
    for cancer, kk in todo.groupby("cancer"):
        genes = sorted(set(kk.gene_a) | set(kk.gene_b))
        mu_idx = set(MU.index)
        s = [x for x in CN.index if x in mu_idx and O.is_tumour([x])[0] and ptype.get(x[:12]) is not None
             and (cancer == C.PANCAN or ptype.get(x[:12]) == cancer)]
        ev = ((CN.loc[s].reindex(columns=genes).abs() == 2) | (MU.loc[s].reindex(columns=genes).fillna(0) > 0))
        if len(s) < 10:  # e.g. LAML: no samples with both GISTIC and MC3 calls -> SBSL default p = 1
            o = kk[KEY].copy()
            o["discover_mutex"] = 1.0
            new.append(o)
            C.log(f"discover {cancer}: {len(s)} samples -> p = 1 for {len(kk):,} pairs")
            continue
        long = ev.stack()
        long = long[long].reset_index()
        long.columns = ["sample", "gene", "v"]
        long[["sample", "gene"]].to_csv(tmp / "ev.csv", index=False)
        pd.DataFrame({"sample": s, "stratum": [ptype.get(x[:12]) for x in s]}).to_csv(tmp / "samples.csv", index=False)
        subprocess.run(["docker", "run", "--rm", "--cpus", "2", "-v", f"{tmp}:/w", "-v", f"{HERE}:/code:ro", "slb/sbsl",
                        "Rscript", "/code/discover.R", "/w/ev.csv", "/w/samples.csv", "/w/out"],
                       check=True, stdout=subprocess.DEVNULL)
        bg_genes = open(tmp / "out_genes.txt").read().split()
        bg = np.fromfile(tmp / "out_bg.f64", dtype=np.float64).reshape(len(bg_genes), len(s))
        evm = ev.reindex(columns=bg_genes).to_numpy()
        o = kk[["gene_a", "gene_b"]].copy()
        o["discover_mutex"] = mutex_pvalues(bg, evm, {g: i for i, g in enumerate(bg_genes)}, kk.gene_a, kk.gene_b)
        o["cancer"] = cancer
        new.append(o[KEY + ["discover_mutex"]])
        C.log(f"discover {cancer}: {len(s)} samples, {len(kk):,} pairs")
    have = pd.concat([have] + new, ignore_index=True)
    have.to_parquet(p)


if __name__ == "__main__":
    main(sys.argv[1])
