"""Build KR4SL (Zhang et al., Bioinformatics 2023, ISMB) inductive-mode inputs for SLB (human).

Output dir: SLB_WORK/kr4sl/{SLB, SLB_ind, all_entities.txt}
Knowledge graph (rebuilt so that every SLB gene can be placed in it, and without genetic-interaction evidence):
  - GO-GO relations (is_a, part_of, regulates, negatively_regulates, positively_regulates, has_part, occurs_in,
    happens_during, ends_during) from go-basic.obo (2026-09-23); entity names = GO term names with spaces -> '_'
    (KR4SL's convention)
  - gene-GO relations from goa_human.gaf (2026-09-23): relation = GAF qualifier (incl. NOT|... qualifiers, as in
    KR4SL's kg.txt); IGI (inferred from genetic interaction) annotations dropped
  - gene-pathway PARTICIPATES_GpPW edges from KR4SL's own kg.txt (SynLethKG pathways)
  The shipped kg.txt has no SL relation; we still rebuild gene-GO to drop IGI and to cover SLB genes absent from it.
SL facts / queries (from SLB train only; KR4SL learns from positives only, measured negatives are unused):
  - SLB/train.txt   = KG + half of the `fit` SL pairs as SL_GsG facts (graph)
  - SLB/valid.txt   = the other half of the `fit` SL pairs (KR4SL trains on the trans 'valid' queries)
  - SLB/test.txt    = `valid`-family SL pairs (KR4SL selects the model on trans 'test' queries = our validation)
  - SLB_ind/train.txt = KG + all `fit` SL pairs as facts; SLB_ind/{valid,test}.txt = the validation queries again
    (placeholders so the loader runs; dev/test pairs are scored separately by predict.py, never used as queries)
"""
import os, gzip, random
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(os.environ.get("SLB_BENCH", ROOT / "data/bench/slb1.2"))
if not BENCH.is_absolute():
    BENCH = ROOT / BENCH
WORK = Path(os.environ.get("SLB_WORK", ROOT / "external/models/_slb_work" / BENCH.name))
KR = ROOT / "external/models/KR4SL/data"
OUT = WORK / "kr4sl"
random.seed(0)


def nm(s):
    return s.strip().replace(" ", "_")


def main():
    (OUT / "SLB").mkdir(parents=True, exist_ok=True); (OUT / "SLB_ind").mkdir(parents=True, exist_ok=True)
    # GO DAG
    terms, cur = {}, None
    for line in open(ROOT / "data/raw/go/go-basic.obo"):
        line = line.rstrip("\n")
        if line == "[Term]":
            cur = {"rel": [], "obs": False}
        elif line.startswith("[") and line.endswith("]"):
            cur = None
        elif cur is not None:
            if line.startswith("id: "):
                terms[line[4:]] = cur
            elif line.startswith("name: "):
                cur["name"] = nm(line[6:])
            elif line.startswith("namespace: "):
                cur["ns"] = {"biological_process": "BP", "cellular_component": "CC", "molecular_function": "MF"}[line[11:]]
            elif line.startswith("is_a: "):
                cur["rel"].append(("is_a", line[6:16]))
            elif line.startswith("relationship: "):
                r, t = line[14:].split()[:2]
                cur["rel"].append((r, t))
            elif line.startswith("is_obsolete: true"):
                cur["obs"] = True
    terms = {k: v for k, v in terms.items() if not v["obs"] and "name" in v}
    trip = []
    for t, v in terms.items():
        for r, p in v["rel"]:
            if p in terms:
                trip.append((v["name"], r, terms[p]["name"]))
    # gene - GO (GAF), IGI dropped
    gaf = pd.read_csv(ROOT / "data/raw/go/goa_human.gaf.gz", sep="\t", comment="!", header=None, dtype=str,
                      usecols=[0, 2, 3, 4, 6, 11])
    gaf.columns = ["db", "sym", "qual", "go", "ev", "type"]
    gaf = gaf[(gaf.db == "UniProtKB") & (gaf.type == "protein") & (gaf.ev != "IGI") & gaf.go.isin(terms)]
    gaf["rel"] = gaf.qual.str.replace("NOT|", "NOT|", regex=False)
    for s, r, g in zip(gaf.sym, gaf.rel, gaf.go):
        trip.append((s, r, terms[g]["name"]))
    # gene - pathway from KR4SL's kg.txt
    kg = pd.read_csv(KR / "kg.txt", sep=" ", header=None, names=["h", "r", "t"], dtype=str, quoting=3)
    pw = kg[kg.r == "PARTICIPATES_GpPW"]
    trip += list(zip(pw.h, pw.r, pw.t))
    kgdf = pd.DataFrame(trip, columns=["h", "r", "t"]).drop_duplicates()
    kgdf = kgdf[~kgdf.h.str.contains(" ") & ~kgdf.t.str.contains(" ")]
    # SLB genes
    tr = pd.read_parquet(WORK / "human_train_pairs.parquet")
    genes = set()
    for f in ["train.parquet", "dev.parquet", "dev_semi.parquet", "test_inputs.parquet", "test_semi_inputs.parquet"]:
        p = BENCH / f
        if p.exists():
            d = pd.read_parquet(p, columns=["species", "gene_a", "gene_b"]); d = d[d.species == "human"]
            genes |= set(d.gene_a) | set(d.gene_b)
    gene_set = sorted(genes | set(gaf.sym) | set(pw.h))
    etype = {}
    for v in terms.values():
        etype[v["name"]] = v["ns"]
    for p in set(pw.t):
        etype.setdefault(p, "Pathway")
    for g in gene_set:
        etype[g] = "Gene"
    ents = sorted(set(kgdf.h) | set(kgdf.t) | set(gene_set))
    ents = [e for e in ents if e in etype]
    kgdf = kgdf[kgdf.h.isin(etype) & kgdf.t.isin(etype)]
    rels = sorted(set(kgdf.r) | {"SL_GsG"})
    fit = tr[(tr.split == "fit") & (tr.label == 1)][["gene_a", "gene_b"]].values.tolist()
    val = tr[(tr.split == "valid") & (tr.label == 1)][["gene_a", "gene_b"]].values.tolist()
    random.shuffle(fit)
    facts, queries = fit[: len(fit) // 2], fit[len(fit) // 2:]
    kg_lines = [f"{h} {r} {t}" for h, r, t in zip(kgdf.h, kgdf.r, kgdf.t)]
    sl = lambda ps: [f"{a} SL_GsG {b}" for a, b in ps]
    for d in ["SLB", "SLB_ind"]:
        with open(OUT / d / "entities.txt", "w") as f:
            for i, e in enumerate(ents):
                f.write(f"{e} {i} {etype[e]}\n")
    with open(OUT / "SLB/Gene_set.txt", "w") as f:
        f.write("\n".join(gene_set) + "\n")
    with open(OUT / "SLB/entitype.txt", "w") as f:
        for i, t in enumerate(["BP", "CC", "Gene", "MF", "Pathway"]):
            f.write(f"{t} {i}\n")
    with open(OUT / "SLB/relations.txt", "w") as f:
        for i, r in enumerate(rels):
            f.write(f"{r} {i}\n")
    open(OUT / "SLB/train.txt", "w").write("\n".join(kg_lines + sl(facts)) + "\n")
    open(OUT / "SLB/valid.txt", "w").write("\n".join(sl(queries)) + "\n")
    open(OUT / "SLB/test.txt", "w").write("\n".join(sl(val)) + "\n")
    open(OUT / "SLB_ind/train.txt", "w").write("\n".join(kg_lines + sl(fit)) + "\n")
    open(OUT / "SLB_ind/valid.txt", "w").write("\n".join(sl(val)) + "\n")
    open(OUT / "SLB_ind/test.txt", "w").write("\n".join(sl(val)) + "\n")
    open(OUT / "all_entities.txt", "w").write("\n".join(ents) + "\n")
    print(f"entities {len(ents)} (genes {len(gene_set)}), relations {len(rels)}, KG triples {len(kg_lines)}, "
          f"SL facts {len(facts)}, train queries {len(queries)}, validation queries {len(val)}")


if __name__ == "__main__":
    main()
