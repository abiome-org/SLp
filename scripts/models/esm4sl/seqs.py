"""Build the protein-sequence table used for ESM-2 embedding (one protein per gene).

usage: python seqs.py <out.tsv>

Sources (no species hard-coded; every species with data/interim/orthology_extra/fasta/<sp>.fa is included):
  * data/interim/orthology_extra/fasta/<sp>.fa  (headers '<sp>:<canonical SLB gene id>', longest isoform per gene;
    built by data-orthology)
  * human override: UniProt reviewed canonical isoform (data/raw/uniprot_human/UP000005640_reviewed_canonical.fasta)
    via HGNC uniprot_ids, as in the original ESM4SL preprocessing (Entrez -> UniProt Entry -> Sequence).
    Falls back to the orthology_extra (Ensembl longest) sequence.
Also adds any SLB gene (train/dev/test_inputs, all species) missing from these files with seq = NA, so coverage is
reported honestly.
Output columns: species, gene, source, seq
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
import slb  # noqa: E402

FASTA_DIR = slb.ROOT / "data/interim/orthology_extra/fasta"
UNIPROT_HUMAN = slb.RAW / "uniprot_human/UP000005640_reviewed_canonical.fasta"


def read_fasta(path):
    name, seq = None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq)
                name, seq = line[1:].split()[0], []
            else:
                seq.append(line.strip())
    if name is not None:
        yield name, "".join(seq)


def bench_genes() -> pd.DataFrame:
    parts = []
    for split in ["train", "dev", "test"]:
        try:
            d = slb.load(split)
        except FileNotFoundError:
            continue
        for c in ["gene_a", "gene_b"]:
            parts.append(d[["species", c]].rename(columns={c: "gene"}).drop_duplicates())
    return pd.concat(parts).drop_duplicates().astype(str)


def main(out: str) -> None:
    rows = {}
    for fa in sorted(FASTA_DIR.glob("*.fa")):
        if fa.stem == "all":
            continue
        for h, s in read_fasta(fa):
            sp, g = h.split(":", 1)
            rows[(sp, g)] = ("orthology_extra", s)
    if UNIPROT_HUMAN.exists():
        up = {h.split("|")[1]: s for h, s in read_fasta(UNIPROT_HUMAN)}
        for sym, ids in zip(*slb.hgnc()[["symbol", "uniprot_ids"]].values.T):
            if isinstance(ids, str):
                for acc in ids.split("|"):
                    if acc in up:
                        rows[("human", sym)] = (f"uniprot:{acc}", up[acc])
                        break
    for sp, g in bench_genes().itertuples(index=False):
        rows.setdefault((sp, g), (None, None))
    df = pd.DataFrame([(sp, g, src, s) for (sp, g), (src, s) in rows.items()],
                      columns=["species", "gene", "source", "seq"])
    df["seq"] = df.seq.str.upper().str.replace(r"[^A-Z]", "", regex=True)
    df.loc[df.seq == "", "seq"] = None
    df.sort_values(["species", "gene"]).to_csv(out, sep="\t", index=False)
    bg = bench_genes()
    m = bg.merge(df, on=["species", "gene"], how="left")
    for sp, g in m.groupby("species"):
        print(f"{sp}: bench genes with sequence {g.seq.notna().sum()}/{len(g)}", file=sys.stderr)
    print(f"total proteins {df.seq.notna().sum()}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
