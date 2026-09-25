"""Export GEM gene -> canonical SLB gene maps for GEMs whose IDs need slbench resolvers (run in project env).
Writes external/models/fba/annot/<species>_gem_map.tsv (gem_gene, gene)."""
import logging
import re
from pathlib import Path

from slbench import ids_extra

E = Path("external/models/fba")
GEMS = {"mmus": "Mouse-GEM/model/Mouse-GEM.xml", "dmel": "Fruitfly-GEM/model/Fruitfly-GEM.xml",
        "cele": "Worm-GEM/model/Worm-GEM.xml"}
logging.getLogger().setLevel(logging.ERROR)
for sp, f in GEMS.items():
    p = E / "gems" / f
    if not p.exists():
        continue
    genes = sorted(set(re.findall(r'<fbc:geneProduct [^>]*fbc:label="([^"]+)"', p.read_text())))
    r = ids_extra.resolver(sp)
    rows = []
    for g in genes:
        cand = [g, g.replace("__", "."), g.replace("_", "."), g.replace("__45__", "-"), re.sub(r"__(\d+)__", lambda m: chr(int(m.group(1))), g)]
        hit = next((r(c) for c in cand if r(c)), None)
        if hit:
            rows.append((g, hit))
    (E / "annot" / f"{sp}_gem_map.tsv").write_text("gem_gene\tgene\n" + "".join(f"{a}\t{b}\n" for a, b in rows))
    print(sp, len(genes), "genes,", len(rows), "mapped")
