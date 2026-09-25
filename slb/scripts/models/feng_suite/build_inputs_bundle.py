"""`allspecies` inputs for the Feng et al. 2024 SL_benchmark models that only need a gene graph + gene
similarity matrices (SL2MF, GRSMF, CMF-W, DDGCN, GCATSL, SLMGAE). One node universe per species = all genes of
that species appearing in any SLB split (labels of dev/test never read). Inputs come from the shared bundle
data/interim/bundle/<species>/ (built by models-mechanistic; GO without IGI/ND/NOT, BioGRID physical + STRING
without experimental/textmining channels):
  PPI graph      bundle ppi.parquet: biogrid_phys > 0, or any kept STRING channel score >= SLB_STRING_MIN (400)
                 (species such as spne have almost no BioGRID physical edges, so STRING channels are needed)
  GO similarity  Wang + BMA on bundle go.parquet annotations (DAG and is_a/part_of types from go-basic.obo)
  topo sim       cosine of PPI adjacency rows (as for the human run)
Usage: SLB_SPECIES=<sp> build_inputs_bundle.py   -> SLB_WORK/feng_<sp>/data   (same file names as the human run)
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_inputs import parse_obo, wang_sv, ROOT, BENCH, WORK  # noqa: E402

SP = os.environ["SLB_SPECIES"]
OUT = WORK / f"feng_{SP}" / "data"
PP = OUT / "preprocessed_data"
BUNDLE = ROOT / "data/interim/bundle" / SP
STRING_MIN = int(os.environ.get("SLB_STRING_MIN", "400"))


def genes_of_species():
    genes = set()
    for f in ["train.parquet", "dev_inputs.parquet", "dev_semi_inputs.parquet", "test_inputs.parquet", "test_semi_inputs.parquet"]:
        p = BENCH / f
        if p.exists():
            d = pd.read_parquet(p, columns=["species", "gene_a", "gene_b"])
            d = d[d.species == SP]
            genes |= set(d.gene_a) | set(d.gene_b)
    return sorted(genes)


def go_sim_from_ann(ann, N, ns):
    import numba
    terms = parse_obo(ROOT / "data/raw/go/go-basic.obo")
    ann = ann[ann.term.isin([t for t, v in terms.items() if not v["obsolete"] and v.get("ns") == ns])]
    ann = ann.drop_duplicates(["u", "term"])
    if len(ann) == 0:
        return np.zeros((N, N))
    at = sorted(ann.term.unique()); tix = {t: i for i, t in enumerate(at)}
    sv = [wang_sv(terms, t) for t in at]
    anc = sorted({a for s in sv for a in s}); aix = {a: i for i, a in enumerate(anc)}
    r, c, v = [], [], []
    for i, s in enumerate(sv):
        for a, x in s.items():
            r.append(i); c.append(aix[a]); v.append(x)
    S = sp.csr_matrix((np.array(v), (r, c)), shape=(len(at), len(anc)))
    I = S.copy(); I.data[:] = 1.0
    tot = np.asarray(S.sum(1)).ravel()
    TS = ((S @ I.T + I @ S.T).toarray() / (tot[:, None] + tot[None, :])).astype(np.float32)
    gl = ann.groupby("u").term.apply(lambda x: np.array(sorted(tix[t] for t in x), dtype=np.int32))
    indptr = np.zeros(N + 1, dtype=np.int64)
    for u, a in gl.items():
        indptr[u + 1] = len(a)
    indptr = np.cumsum(indptr); idx = np.zeros(indptr[-1], dtype=np.int32)
    for u, a in gl.items():
        idx[indptr[u]:indptr[u + 1]] = a

    @numba.njit(parallel=True, fastmath=True)
    def bma(TS, indptr, idx, N):
        out = np.zeros((N, N), dtype=np.float32)
        for i in numba.prange(N):
            a0, a1 = indptr[i], indptr[i + 1]
            if a1 == a0:
                continue
            for j in range(i, N):
                b0, b1 = indptr[j], indptr[j + 1]
                if b1 == b0:
                    continue
                s1 = 0.0
                for x in range(a0, a1):
                    m = 0.0
                    for y in range(b0, b1):
                        t = TS[idx[x], idx[y]]
                        if t > m:
                            m = t
                    s1 += m
                s2 = 0.0
                for y in range(b0, b1):
                    m = 0.0
                    for x in range(a0, a1):
                        t = TS[idx[x], idx[y]]
                        if t > m:
                            m = t
                    s2 += m
                val = (s1 + s2) / ((a1 - a0) + (b1 - b0))
                out[i, j] = val; out[j, i] = val
        return out
    numba.set_num_threads(int(os.environ.get("SLB_THREADS", "32")))
    out = np.round(bma(TS, indptr, idx, N).astype(np.float64), 3)
    has = np.diff(indptr) > 0
    out[np.arange(N)[has], np.arange(N)[has]] = 1.0
    print(f"GO {ns}: {len(at)} terms, {has.sum()}/{N} genes annotated")
    return out


def main():
    PP.mkdir(parents=True, exist_ok=True); (OUT / "data_split").mkdir(parents=True, exist_ok=True); (PP / "wo_compt").mkdir(exist_ok=True)
    genes = genes_of_species(); N = len(genes); gm = {g: i for i, g in enumerate(genes)}
    pd.DataFrame({"unified_id": np.arange(N), "symbol": genes, "entrez_id": -1, "kg_uid": -1}).to_parquet(OUT / "universe.parquet", index=False)
    pd.DataFrame({"gene": genes, "unified_id": np.arange(N)}).to_parquet(OUT / "slb_gene_index.parquet", index=False)
    pd.DataFrame({"unified_id": np.arange(N), "symbol": genes, "entrez_id": -1, "kg_id": -1}).to_csv(PP / "meta_table_9845.csv", index=False)
    # split (fit / valid by gene family, from slb_pairs.py output for this species)
    tr = pd.read_parquet(WORK / f"{SP}_train_pairs.parquet")
    tr["a"], tr["b"] = tr.gene_a.map(gm), tr.gene_b.map(gm)
    tr = tr[tr.a != tr.b]
    tr["lo"], tr["hi"] = np.minimum(tr.a, tr.b), np.maximum(tr.a, tr.b)
    arr = lambda d: d[["lo", "hi"]].values.astype(np.int64)
    fit, val = tr[tr.split == "fit"], tr[tr.split == "valid"]
    P = [arr(fit[fit.label == 1]), arr(val[val.label == 1])]; Q = [arr(fit[fit.label == 0]), arr(val[val.label == 0])]
    np.savez(OUT / "data_split/slb_split_arrays.npz", fit_pos=P[0], fit_neg=Q[0], val_pos=P[1], val_neg=Q[1], N=np.array([N]))
    sl = pd.DataFrame({"unified_id_A": P[0][:, 0], "unified_id_B": P[0][:, 1]}); sl["kg_id_A"], sl["kg_id_B"] = sl.unified_id_A, sl.unified_id_B
    sl.to_csv(PP / "human_sl_9845.csv", index=False); sl.to_csv(PP / "human_sl_6460.csv", index=False)
    print(f"{SP}: N={N}; fit {len(P[0])} pos / {len(Q[0])} neg; valid {len(P[1])} pos / {len(Q[1])} neg")
    # PPI
    ppi = pd.read_parquet(BUNDLE / "ppi.parquet")
    sc = [c for c in ppi.columns if c.startswith("string_")]
    keep = (ppi.biogrid_phys > 0) | (ppi[sc].max(axis=1) >= STRING_MIN)
    ppi = ppi[keep]
    a, b = ppi.gene_a.map(gm), ppi.gene_b.map(gm)
    ok = a.notna() & b.notna()
    a, b = a[ok].astype(int).values, b[ok].astype(int).values
    lo, hi = np.minimum(a, b), np.maximum(a, b); k = lo != hi
    M = sp.csr_matrix((np.ones(k.sum()), (lo[k], hi[k])), shape=(N, N)); M.data[:] = 1
    sp.save_npz(PP / "ppi_sparse_upper_matrix_without_sl_relation_9845.npz", M.astype(np.int64))
    print(f"PPI edges {M.nnz} (biogrid_phys or STRING>={STRING_MIN}), genes with edges {len(set(lo[k]) | set(hi[k]))}")
    A = (M + M.T + sp.eye(N)).tocsr().astype(np.float64)
    nrm = np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel()); Dn = sp.diags(1.0 / nrm)
    (PP / "sl2mf_data").mkdir(exist_ok=True)
    np.save(PP / "sl2mf_data/ppi_topo_sim_matrix.npy", (Dn @ A @ A.T @ Dn).toarray())
    go = pd.read_parquet(BUNDLE / "go.parquet")
    go = go[go.evidence != "IGI"]
    go["u"] = go.gene.map(gm); go = go[go.u.notna()]; go["u"] = go.u.astype(int)
    for ont, ns in [("bp", "biological_process"), ("cc", "cellular_component"), ("mf", "molecular_function")]:
        np.save(PP / f"final_gosim_{ont}_from_r_9845.npy", go_sim_from_ann(go[["u", "term"]], N, ns))
    print("done", OUT)


if __name__ == "__main__":
    main()
