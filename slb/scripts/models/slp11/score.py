"""Score SLB human pairs with the frozen SLp-1.1 world model (abiome-org/slp, HF potteryrage/SLp).

Run with the old repo's environment:
    external/models/slp11/.venv/bin/python scripts/models/slp11/score.py IN.tsv OUT.tsv
IN.tsv: gene_a<TAB>gene_b (HGNC symbols). OUT.tsv adds sl_score and label_free (excess fitness loss);
pairs whose genes are outside SLp's descriptor vocabulary are written with empty scores.
"""
import sys
from pathlib import Path

import numpy as np

BUNDLE = Path("external/models/slp11/results/slp11-transition/cellular-genomic-world-predictor-v2").resolve()
sys.path.insert(0, str(BUNDLE))
from functional_predict import SLpWorld  # noqa: E402

world = SLpWorld(BUNDLE, device="cpu")
pairs = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
ok = [i for i, (a, b) in enumerate(pairs) if a in world.lookup and b in world.lookup and world.lookup[a] != world.lookup[b]]
sl = np.full(len(pairs), np.nan)
lf = np.full(len(pairs), np.nan)
for s in range(0, len(ok), 2048):
    idx = ok[s:s + 2048]
    r = world.predict_pairs([pairs[i] for i in idx])
    sl[idx] = r["sl_score"]
    lf[idx] = r["label_free_excess_fitness_loss"]
    print(f"{s + len(idx)}/{len(ok)}", file=sys.stderr, flush=True)
with open(sys.argv[2], "w") as f:
    for (a, b), x, y in zip(pairs, sl, lf):
        f.write(f"{a}\t{b}\t{'' if np.isnan(x) else x}\t{'' if np.isnan(y) else y}\n")
print(f"scored {len(ok)}/{len(pairs)} pairs", file=sys.stderr)
