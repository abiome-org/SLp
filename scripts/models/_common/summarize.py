"""Summarise results/models/<name>_dev.txt eval reports: SLB, species scores, human ancestry, paralog stratum,
missing-filled count. usage: uv run python scripts/models/_common/summarize.py name1 name2 ... (or prefixes with *)"""
import glob
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[3] / "results/models"


def parse(p):
    t = p.read_text()
    g = lambda pat: (re.search(pat, t).group(1) if re.search(pat, t) else "")
    row = {"model": p.name[:-8], "SLB": g(r"SLB score: ([\d.]+)"), "human": g(r"H\. sapiens=([\d.]+)"),
           "scer": g(r"S\. cerevisiae=([\d.]+)"), "spom": g(r"S\. pombe=([\d.]+)"), "spne": g(r"S\. pneumoniae=([\d.]+)"),
           "missing": g(r"WARNING: ([\d,]+) missing") or "0"}
    for k, pat in [("AFR", r"ancestry=AFR\s+[\d,]+\s+\d+\s+([\d.]+)"), ("EAS", r"ancestry=EAS\s+[\d,]+\s+\d+\s+([\d.]+)"),
                   ("EUR", r"ancestry=EUR\s+[\d,]+\s+\d+\s+([\d.]+)"),
                   ("paralog", r"same family \(paralogs\)\s+[\d,]+\s+[\d,]+\s+([\d.]+)"),
                   ("human_flat", r"species=human\s+[\d,]+\s+[\d,]+\s+([\d.]+)")]:
        row[k] = g(pat)
    return row


if __name__ == "__main__":
    files = []
    for a in sys.argv[1:]:
        files += sorted(glob.glob(str(R / f"{a}_dev.txt")))
    rows = [parse(Path(f)) for f in files]
    cols = ["model", "SLB", "human", "scer", "spom", "spne", "AFR", "EAS", "EUR", "human_flat", "paralog", "missing"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(r[c] for c in cols) + " |")
