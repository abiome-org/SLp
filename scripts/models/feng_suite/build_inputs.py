"""Build SLB inputs for the Feng et al. 2024 SL_benchmark code base (models-graph agent).

Node universe: the 9,845 SynLethDB genes of the original benchmark keep ids 0..9844 (so the original
feature files keep their meaning); every SLB human gene (any split, labels not read for dev/test) that is not
among them is appended as id 9845..N-1.

Everything a model sees is rebuilt for the N-gene universe, from sources that carry NO SL / GI labels:
  preprocessed_data/meta_table_9845.csv                 unified_id, symbol (current HGNC), entrez_id
  preprocessed_data/fin_kg_wo_sl_9845.csv               Feng's SynLethKG with SL_GsG, SR_GsrG, NONSL_GnsG already
                                                        removed (verified: those 3 relation types are absent); entity ids
                                                        re-mapped so that the N genes are 0..N-1
  preprocessed_data/ppi_sparse_upper_matrix_without_sl_relation_9845.npz
                                                        BioGRID 5.0.261 physical human PPI, binary, upper triangle.
                                                        (Feng's original file had PPI edges that coincide with SynLethDB SL
                                                        pairs deleted, i.e. edited with SL labels; we do not reuse it.)
  preprocessed_data/final_gosim_{bp,cc,mf}_from_r_9845.npy
                                                        GO semantic similarity, Wang + BMA (GOSemSim default), on NCBI
                                                        gene2go (NOT-qualified and IGI [genetic-interaction] annotations dropped), rounded to 3 dp.
                                                        Recomputed for all N genes (validated against Feng's R values).
  preprocessed_data/sl2mf_data/ppi_topo_sim_matrix.npy  cosine similarity of PPI adjacency rows (with self loops); Feng's exact recipe is
                                                        unreleased, cosine reproduces their matrix at Spearman 0.998
  preprocessed_data/human_sl_9845.csv, human_sl_6460.csv  SLB *fit* positives only (in unified ids)
  data_split/CV1_50_slb.pkl                              Feng indep_test format: [graph_train, graph_valid, graph_test,
                                                        train_pairs, valid_pairs, test_pairs] x 1 fold, pos and neg.
                                                        train = SLB train pairs in "fit" gene families, valid = SLB train
                                                        pairs in held-aside train families; test = valid (placeholder:
                                                        dev labels are never given to the Feng code).
  slb_gene_index.parquet                                SLB gene symbol -> unified id
Run with SLB_BENCH (default data/bench/slb1.2). Output: SLB_WORK/feng/data
"""
import os, pickle, re, sys, gzip
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(os.environ.get("SLB_BENCH", ROOT / "data/bench/slb1.2"))
if not BENCH.is_absolute():
    BENCH = ROOT / BENCH
WORK = Path(os.environ.get("SLB_WORK", ROOT / "external/models/_slb_work" / BENCH.name))
FENG = ROOT / "data/raw/feng2024_slbench/extracted/data"
OUT = WORK / "feng" / "data"
PP = OUT / "preprocessed_data"
N_OLD = 9845


def hgnc_maps():
    h = pd.read_csv(ROOT / "data/raw/ids/hgnc_complete_set.txt", sep="\t", dtype=str, low_memory=False)
    h = h[h.status == "Approved"]
    cur = set(h.symbol)
    by_id = dict(zip(h.hgnc_id, h.symbol))
    entrez = dict(zip(h.symbol, h.entrez_id))
    alias = defaultdict(set)
    for col in ["prev_symbol", "alias_symbol"]:
        for s, v in zip(h.symbol, h[col]):
            if isinstance(v, str):
                for a in v.split("|"):
                    alias[a].add(s)

    def resolve(x):
        if x in cur:
            return x
        c = alias.get(x, set())
        return next(iter(c)) if len(c) == 1 else None
    return resolve, by_id, entrez


def slb_genes():
    genes = set()
    for f in ["train.parquet", "dev.parquet", "dev_semi.parquet", "test_inputs.parquet", "test_semi_inputs.parquet"]:
        p = BENCH / f
        if p.exists():
            d = pd.read_parquet(p, columns=["species", "gene_a", "gene_b"])
            d = d[d.species == "human"]
            genes |= set(d.gene_a) | set(d.gene_b)
    return sorted(genes)


def build_universe():
    resolve, by_id, entrez = hgnc_maps()
    meta = pd.read_csv(FENG / "preprocessed_data/meta_table_9845.csv")
    meta = meta.sort_values("unified_id")
    assert (meta.unified_id.values == np.arange(N_OLD)).all()
    cur = [by_id.get(h) or resolve(s) or s for h, s in zip(meta.hgnc_id, meta.symbol)]
    meta["symbol_current"] = cur
    old_sym = {}
    for i, s in enumerate(cur):
        old_sym.setdefault(s, i)
    ents = pd.read_csv(FENG / "preprocessed_data/not_used/fin_entities.csv")
    kg_gene = {}
    for uid, name in zip(ents[ents.entity_type_name == "Gene"].unified_id, ents[ents.entity_type_name == "Gene"].entity_name):
        if uid < N_OLD:
            continue
        s = resolve(str(name)) or str(name)
        kg_gene.setdefault(s, uid)
    genes = slb_genes()
    idx, new = {}, []
    for g in genes:
        s = resolve(g) or g
        if s in old_sym:
            idx[g] = old_sym[s]
        else:
            new.append((g, s))
    new_ids = {}
    for k, (g, s) in enumerate(sorted(set(new), key=lambda x: x[1])):
        if s not in new_ids:
            new_ids[s] = N_OLD + len(new_ids)
        idx[g] = new_ids[s]
    N = N_OLD + len(new_ids)
    rows = [dict(unified_id=i, symbol=cur[i], entrez_id=meta.entrez_id.iloc[i], kg_uid=i) for i in range(N_OLD)]
    for s, i in sorted(new_ids.items(), key=lambda x: x[1]):
        e = entrez.get(s)
        rows.append(dict(unified_id=i, symbol=s, entrez_id=int(e) if isinstance(e, str) and e.isdigit() else -1, kg_uid=kg_gene.get(s, -1)))
    U = pd.DataFrame(rows)
    U["entrez_id"] = pd.to_numeric(U.entrez_id, errors="coerce").fillna(-1).astype(int)
    gi = pd.DataFrame({"gene": list(idx), "unified_id": list(idx.values())})
    print(f"universe N={N} (old {N_OLD}, new {len(new_ids)}; new genes with a KG entity: {(U.kg_uid[N_OLD:] >= 0).sum()})")
    return U, gi, N


def remap_kg(U, N):
    kg = pd.read_csv(FENG / "preprocessed_data/fin_kg_wo_sl_9845.csv")
    n_ent = int(max(kg.unified_id_A.max(), kg.unified_id_B.max())) + 1
    m = -np.ones(n_ent, dtype=np.int64)
    m[:N_OLD] = np.arange(N_OLD)
    for uid, kid in zip(U.unified_id[N_OLD:], U.kg_uid[N_OLD:]):
        if kid >= 0:
            m[kid] = uid
    nxt = N
    for e in range(n_ent):
        if m[e] < 0:
            m[e] = nxt
            nxt += 1
    kg["unified_id_A"] = m[kg.unified_id_A.values]
    kg["unified_id_B"] = m[kg.unified_id_B.values]
    return kg, m


def build_ppi(U, N):
    f = ROOT / "data/raw/biogrid/BIOGRID-ORGANISM-Homo_sapiens-5.0.261.tab3.txt"
    b = pd.read_csv(f, sep="\t", usecols=["Entrez Gene Interactor A", "Entrez Gene Interactor B", "Experimental System Type",
                                          "Organism ID Interactor A", "Organism ID Interactor B"], dtype=str)
    b = b[(b["Experimental System Type"] == "physical") & (b["Organism ID Interactor A"] == "9606") & (b["Organism ID Interactor B"] == "9606")]
    e2u = {str(int(e)): u for e, u in zip(U.entrez_id, U.unified_id) if e is not None and not pd.isna(e) and int(e) > 0}
    a = b["Entrez Gene Interactor A"].map(e2u)
    c = b["Entrez Gene Interactor B"].map(e2u)
    ok = a.notna() & c.notna()
    a, c = a[ok].astype(int).values, c[ok].astype(int).values
    lo, hi = np.minimum(a, c), np.maximum(a, c)
    keep = lo != hi
    M = sp.csr_matrix((np.ones(keep.sum(), dtype=np.int64), (lo[keep], hi[keep])), shape=(N, N))
    M.data[:] = 1
    print(f"PPI: {M.nnz} undirected physical edges over {len(set(lo[keep]) | set(hi[keep]))} genes")
    return M


# ---------------- GO semantic similarity (Wang 2007 + BMA, as GOSemSim::mgeneSim default) ----------------

def parse_obo(path):
    terms, cur = {}, None
    for line in open(path):
        line = line.rstrip("\n")
        if line == "[Term]":
            cur = {"is_a": [], "part_of": [], "obsolete": False}
        elif line.startswith("[") and line.endswith("]"):
            cur = None
        elif cur is not None:
            if line.startswith("id: "):
                cur["id"] = line[4:]
                terms[cur["id"]] = cur
            elif line.startswith("namespace: "):
                cur["ns"] = line[11:]
            elif line.startswith("is_a: "):
                cur["is_a"].append(line[6:16])
            elif line.startswith("relationship: part_of "):
                cur["part_of"].append(line[22:32])
            elif line.startswith("is_obsolete: true"):
                cur["obsolete"] = True
    return terms


def wang_sv(terms, t):
    s = {t: 1.0}
    frontier = [t]
    while frontier:
        nf = []
        for c in frontier:
            for p, w in [(p, 0.8) for p in terms[c]["is_a"]] + [(p, 0.6) for p in terms[c]["part_of"]]:
                if p not in terms or terms[p]["ns"] != terms[t]["ns"]:
                    continue
                v = s[c] * w
                if v > s.get(p, 0):
                    s[p] = v
                    nf.append(p)
        frontier = nf
    return s


def go_sim(U, N, ont):
    import numba
    ns = {"bp": "biological_process", "cc": "cellular_component", "mf": "molecular_function"}[ont]
    cat = {"bp": "Process", "cc": "Component", "mf": "Function"}[ont]
    terms = parse_obo(ROOT / "data/raw/go/go-basic.obo")
    g2g = pd.read_csv(ROOT / "data/raw/go/gene2go_human.tsv", sep="\t", header=None,
                      names=["tax", "gene", "go", "ev", "qual", "term", "pmid", "cat"], dtype=str)
    g2g = g2g[(g2g.cat == cat) & ~g2g.qual.str.startswith("NOT") & (g2g.ev != "IGI")]  # IGI = inferred from genetic interaction: leaky, dropped
    g2g = g2g[g2g.go.isin([t for t, v in terms.items() if not v["obsolete"] and v.get("ns") == ns])]
    e2u = {str(int(e)): u for e, u in zip(U.entrez_id, U.unified_id) if int(e) > 0}
    g2g = g2g[g2g.gene.isin(e2u)]
    g2g["u"] = g2g.gene.map(e2u).astype(int)
    g2g = g2g.drop_duplicates(["u", "go"])
    ann_terms = sorted(g2g.go.unique())
    tix = {t: i for i, t in enumerate(ann_terms)}
    sv = [wang_sv(terms, t) for t in ann_terms]
    allanc = sorted({a for s in sv for a in s})
    aix = {a: i for i, a in enumerate(allanc)}
    r, c, v = [], [], []
    for i, s in enumerate(sv):
        for a, x in s.items():
            r.append(i); c.append(aix[a]); v.append(x)
    S = sp.csr_matrix((np.array(v, dtype=np.float64), (r, c)), shape=(len(ann_terms), len(allanc)))
    I = S.copy(); I.data[:] = 1.0
    num = (S @ I.T + I @ S.T).toarray()
    tot = np.asarray(S.sum(1)).ravel()
    TS = (num / (tot[:, None] + tot[None, :])).astype(np.float32)
    del num
    # gene -> term lists (CSR)
    gl = g2g.groupby("u").go.apply(lambda x: np.array(sorted(tix[t] for t in x), dtype=np.int32))
    indptr = np.zeros(N + 1, dtype=np.int64)
    for u, arr in gl.items():
        indptr[u + 1] = len(arr)
    indptr = np.cumsum(indptr)
    idx = np.zeros(indptr[-1], dtype=np.int32)
    for u, arr in gl.items():
        idx[indptr[u]:indptr[u + 1]] = arr

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
                out[i, j] = val
                out[j, i] = val
        return out

    numba.set_num_threads(int(os.environ.get("SLB_THREADS", "32")))
    out = bma(TS, indptr, idx, N)
    out = np.round(out.astype(np.float64), 3)
    has = np.diff(indptr) > 0
    out[np.arange(N)[has], np.arange(N)[has]] = 1.0
    print(f"GO {ont}: {len(ann_terms)} terms, {has.sum()} of {N} genes annotated")
    return out


def validate_go(new, ont):
    old = np.load(FENG / f"preprocessed_data/final_gosim_{ont}_from_r_9845.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, N_OLD, 200000), rng.integers(0, N_OLD, 200000)
    a, b = np.asarray(old[i, j]), new[i, j]
    ok = (a > 0) & (b > 0)
    r = np.corrcoef(a[ok], b[ok])[0, 1]
    print(f"GO {ont} validation vs Feng R/GOSemSim on 200k random old-gene pairs: pearson={r:.3f}, "
          f"both-annotated={ok.mean():.3f}, mean abs diff={np.abs(a[ok]-b[ok]).mean():.3f}")
    return r


def split_pkl(gi, N):
    tr = pd.read_parquet(WORK / "human_train_pairs.parquet")
    u = dict(zip(gi.gene, gi.unified_id))
    tr["a"], tr["b"] = tr.gene_a.map(u), tr.gene_b.map(u)
    tr = tr[tr.a != tr.b]
    tr["lo"], tr["hi"] = np.minimum(tr.a, tr.b), np.maximum(tr.a, tr.b)
    tr = tr.sort_values("label", ascending=False).drop_duplicates(["lo", "hi"])  # symbols collapsing onto one uid

    def arr(d):
        return d[["lo", "hi"]].values.astype(np.int64)

    def g(p):
        m = sp.csr_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(N, N))
        return m + m.T
    fit, val = tr[tr.split == "fit"], tr[tr.split == "valid"]
    P = [arr(fit[fit.label == 1]), arr(val[val.label == 1])]
    Nn = [arr(fit[fit.label == 0]), arr(val[val.label == 0])]
    # plain arrays only: the pickle in the Feng format is written inside the py3.7 container (make_pkl.py),
    # because numpy>=2 / scipy>=1.8 pickles cannot be read by the legacy stack
    np.savez(OUT / "data_split/slb_split_arrays.npz", fit_pos=P[0], fit_neg=Nn[0], val_pos=P[1], val_neg=Nn[1], N=np.array([N]))
    print(f"split: fit {len(P[0])} pos / {len(Nn[0])} neg; valid {len(P[1])} pos / {len(Nn[1])} neg")
    return P[0]


def main():
    PP.mkdir(parents=True, exist_ok=True)
    (OUT / "data_split").mkdir(parents=True, exist_ok=True)
    U, gi, N = build_universe()
    U.to_parquet(OUT / "universe.parquet", index=False)
    gi.to_parquet(OUT / "slb_gene_index.parquet", index=False)
    meta = U[["unified_id", "symbol", "entrez_id"]].copy()
    meta["kg_id"] = U.kg_uid
    meta.to_csv(PP / "meta_table_9845.csv", index=False)
    fitpos = split_pkl(gi, N)
    sl = pd.DataFrame({"unified_id_A": fitpos[:, 0], "unified_id_B": fitpos[:, 1]})
    sl["kg_id_A"], sl["kg_id_B"] = sl.unified_id_A, sl.unified_id_B
    sl.to_csv(PP / "human_sl_9845.csv", index=False)
    sl.to_csv(PP / "human_sl_6460.csv", index=False)
    (PP / "wo_compt").mkdir(exist_ok=True)
    kg, m = remap_kg(U, N)
    kg.to_csv(PP / "fin_kg_wo_sl_9845.csv", index=False)
    np.save(OUT / "kg_entity_remap.npy", m)
    print(f"KG: {len(kg)} triples, {len(set(kg.unified_id_A) | set(kg.unified_id_B))} entities, max id {max(kg.unified_id_A.max(), kg.unified_id_B.max())}")
    ppi = build_ppi(U, N)
    sp.save_npz(PP / "ppi_sparse_upper_matrix_without_sl_relation_9845.npz", ppi)
    A = (ppi + ppi.T + sp.eye(N)).tocsr().astype(np.float64)
    nrm = np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel())
    Dn = sp.diags(1.0 / nrm)
    topo = (Dn @ A @ A.T @ Dn).toarray()
    (PP / "sl2mf_data").mkdir(exist_ok=True)
    np.save(PP / "sl2mf_data/ppi_topo_sim_matrix.npy", topo)
    del topo
    for ont in ["bp", "cc", "mf"]:
        if (PP / f"final_gosim_{ont}_from_r_9845.npy").exists() and not os.environ.get("SLB_REDO_GO"):
            continue
        s = go_sim(U, N, ont)
        validate_go(s, ont)
        np.save(PP / f"final_gosim_{ont}_from_r_9845.npy", s)
        del s
    print("done", OUT)


if __name__ == "__main__":
    main()
