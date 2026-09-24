"""Summary table of results/models/slb1.3/<name>_dev.txt: headline, per-species (headline + auxiliary strata),
paralog stratum, native coverage. usage: uv run python scripts/models/_common/summarize13.py pattern..."""
import glob, json, re, sys
from pathlib import Path
R = Path(__file__).resolve().parents[3] / "results/models/slb1.3"


def parse(p):
    t = p.read_text()
    g = lambda pat: (re.search(pat, t).group(1) if re.search(pat, t) else "")
    row = {"model": p.name[:-8], "SLB": g(r"SLB score: ([\d.]+)"), "human": g(r"H\. sapiens=([\d.]+)"),
           "scer": g(r"S\. cerevisiae=([\d.]+)"), "spom": g(r"S\. pombe=([\d.]+)")}
    for sp in ["bsub", "cele", "dmel"]:
        row[sp] = g(rf"species={sp}\s+[\d,]+\s+\d+\s+([\d.na]+)")
    row["paralog"] = g(r"same family \(paralogs\)\s+[\d,]+\s+[\d,]+\s+([\d.]+)")
    cj = p.with_name(p.name.replace(".txt", ".coverage.json"))
    if cj.exists():
        c = json.load(open(cj))
        row["native"] = ",".join(k for k, v in c.items() if v["scored"] > 0)
    else:
        row["native"] = "?"
    return row


files = []
for a in sys.argv[1:]:
    files += sorted(glob.glob(str(R / f"{a}_dev.txt")))
cols = ["model", "SLB", "human", "scer", "spom", "bsub", "cele", "dmel", "paralog", "native"]
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for f in files:
    r = parse(Path(f))
    print("| " + " | ".join(r[c] for c in cols) + " |")
