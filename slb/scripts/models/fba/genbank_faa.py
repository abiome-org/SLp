"""Extract CDS proteins (locus_tag -> translation) and gene names from the D39V / R6 GenBank records."""
import re
from pathlib import Path

A = Path(__file__).resolve().parents[3] / "external/models/fba/annot"
for fn, name in (("D39V_CP027540.gb", "D39V"), ("R6_AE007317.gb", "R6")):
    txt = (A / fn).read_text().split("ORIGIN")[0]
    rows = []
    for f in txt.split("\n     CDS ")[1:]:
        f = f.split("\n     gene ")[0]
        lt, g = re.search(r'/locus_tag="([^"]+)"', f), re.search(r'/gene="([^"]+)"', f)
        tr = re.search(r'/translation="([^"]+)"', f, re.S)
        if lt and tr:
            rows.append((lt.group(1), g.group(1) if g else "", re.sub(r"\s", "", tr.group(1))))
    (A / f"{name}.faa").write_text("".join(f">{lt}\n{s}\n" for lt, _, s in rows))
    (A / f"{name}_genes.tsv").write_text("locus_tag\tgene\told_locus_tag\n" + "".join(f"{lt}\t{g}\t\n" for lt, g, _ in rows))
