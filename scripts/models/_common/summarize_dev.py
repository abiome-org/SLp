"""Print a markdown table of dev results for the given model result names (reads results/models/<name>_dev.txt)."""
import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
print("| model | SLB (dev) | human | scer | spom | spne | human rows scored | missing filled |")
print("|---|---|---|---|---|---|---|---|")
for n in sys.argv[1:]:
    f = ROOT / "results/models" / f"{n}_dev.txt"
    if not f.exists():
        print(f"| {n} | - | - | - | - | - | - | - |"); continue
    t = f.read_text()
    slb = re.search(r"SLB score: ([0-9.]+)", t); sp = dict(re.findall(r"(H\. sapiens|S\. cerevisiae|S\. pombe|S\. pneumoniae)=([0-9.]+|nan)", t))
    miss = re.search(r"([0-9,]+) missing scores", t)
    import pandas as pd
    p = pd.read_parquet(ROOT / "results/models" / f"{n}_dev.parquet")
    hs = p.example_id.str.startswith("human|").sum()
    print(f"| {n} | {slb.group(1) if slb else '-'} | {sp.get('H. sapiens','-')} | {sp.get('S. cerevisiae','-')} | {sp.get('S. pombe','-')} | {sp.get('S. pneumoniae','-')} | {hs} | {miss.group(1) if miss else 0} |")
