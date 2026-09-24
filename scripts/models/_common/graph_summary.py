"""Summary table of models-graph results. Usage: graph_summary.py [bench=slb1.3]
Reads results/models/[<bench>/]<name>_dev.txt for every (name, leakage, species) below."""
import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
BENCH = sys.argv[1] if len(sys.argv) > 1 else "slb1.3"
D = ROOT / "results/models" / ("" if BENCH == "slb1.2" else BENCH)
ROWS = [  # name, leakage, species covered
    ("slmgae", "clean", "human"), ("slmgae__allspecies", "clean", "all bundle species"),
    ("sl2mf", "clean", "human"), ("sl2mf__allspecies", "clean", "all bundle species"),
    ("grsmf", "clean", "human"), ("cmfw", "clean", "human"), ("ddgcn", "clean", "human"),
    ("gcatsl", "clean", "human"), ("kg4sl", "clean", "human"), ("slgnn", "clean", "human"),
    ("nsf4sl", "clean", "human"), ("ptgnn", "clean", "human"), ("pilsl", "clean", "human"),
    ("mvgcn_isl", "clean", "human"), ("mvgcn_isl__allspecies", "clean", "all bundle species"),
    ("msgt_sl", "clean", "human"), ("mlec_isl", "clean", "human"), ("kr4sl", "clean", "human"),
    ("lukepi", "possibly leaky (PrimeKG PPI)", "human"), ("struct2sl", "clean", "human"), ("struct2sl__released", "LEAKY", "human"),
]
SP = ["H. sapiens", "S. cerevisiae", "S. pombe", "S. pneumoniae", "B. subtilis", "C. elegans", "D. melanogaster", "M. musculus"]
print(f"| model | leakage | species | SLB ({BENCH} dev) | " + " | ".join(SP[:3] + (SP[3:4] if BENCH == "slb1.2" else SP[4:])) + " |")
print("|---" * (4 + (4 if BENCH == "slb1.2" else 7)) + "|")
for n, lk, sp in ROWS:
    f = D / f"{n}_dev.txt"
    if not f.exists():
        print(f"| {n} | {lk} | {sp} | not scored | " + " | ".join(["-"] * (4 if BENCH == "slb1.2" else 7)) + " |"); continue
    t = f.read_text()
    m = re.search(r"SLB score: ([0-9.]+)", t)
    vals = dict(re.findall(r"([A-Z]\. [a-z]+)=([0-9.]+|n/a)", t))
    cols = SP[:3] + (SP[3:4] if BENCH == "slb1.2" else SP[4:])
    print(f"| {n} | {lk} | {sp} | {m.group(1) if m else '?'} | " + " | ".join(vals.get(c, "-") for c in cols) + " |")
