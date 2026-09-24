"""PT-GNN inputs for the SLB node universe: protein sequences encoded as overlapping amino-acid 3-mer word ids
(Feng et al.'s PTGNN_pre.ipynb recipe: sequence cut/padded with 'Z' to 802, 800 words, vocabulary = 3-mers over
the 21-letter alphabet ACDEFGHIKLMNPQRSTUVWY plus Z-padded words; 9,724 words). The 9,845 original genes keep
Feng's encodings (UniProt 2022); appended SLB genes are encoded from UniProt reviewed canonical sequences
(data/raw/uniprot_human, fetched by models-features-fm). Genes without a protein get all-padding words.
The word-embedding init (trained_word_embedding.npy) is Feng's (a sequence-only embedding, no SL labels)."""
import os, shutil
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(os.environ.get("SLB_BENCH", ROOT / "data/bench/slb1.2"))
if not BENCH.is_absolute():
    BENCH = ROOT / BENCH
WORK = Path(os.environ.get("SLB_WORK", ROOT / "external/models/_slb_work" / BENCH.name))
FENG = ROOT / "data/raw/feng2024_slbench/extracted/data/preprocessed_data/ptgnn_data"
OUT = WORK / "feng/data/preprocessed_data/ptgnn_data"
OUT.mkdir(parents=True, exist_ok=True)
acids = sorted("ACDEFGHIKLMNPQRSTUVWY")
words = [a + b + c for a in acids for b in acids for c in acids] + [a + b + "Z" for a in acids for b in acids] + [a + "ZZ" for a in acids] + ["ZZZ"]
wd = {w: i for i, w in enumerate(words)}
assert len(words) == 9724
seqs, cur = {}, None
for line in open(ROOT / "data/raw/uniprot_human/UP000005640_reviewed_canonical.fasta"):
    if line.startswith(">"):
        cur = line.split("|")[1]; seqs[cur] = []
    else:
        seqs[cur].append(line.strip())
seqs = {k: "".join(v) for k, v in seqs.items()}
g = pd.read_csv(ROOT / "data/raw/uniprot_human/uniprot_reviewed_9606_genes.tsv", sep="\t")
sym2acc = {}
for acc, gp in zip(g.iloc[:, 0], g["Gene Names (primary)"] if "Gene Names (primary)" in g else g.iloc[:, 2]):
    if isinstance(gp, str) and acc in seqs:
        for s in gp.split(";"):
            sym2acc.setdefault(s.strip(), acc)


def encode(seq):
    seq = "".join(ch if ch in acids else "Z" for ch in seq)[:802].ljust(802, "Z")
    return [wd.get(seq[j:j + 3], wd["ZZZ"]) for j in range(800)]


U = pd.read_parquet(WORK / "feng/data/universe.parquet")
old = np.load(FENG / "ptgnn_encod_by_word_sl_9845_800.npy")
enc = np.full((len(U), 800), wd["ZZZ"], dtype=np.int64)
enc[:len(old)] = old
miss = 0
for i, s in zip(U.unified_id[len(old):], U.symbol[len(old):]):
    acc = sym2acc.get(s)
    if acc is None:
        miss += 1; continue
    enc[i] = encode(seqs[acc])
np.save(OUT / "ptgnn_encod_by_word_sl_9845_800.npy", enc)
shutil.copy(FENG / "trained_word_embedding.npy", OUT / "trained_word_embedding.npy")
print(f"encoded {len(U) - len(old)} new genes, {miss} without a UniProt reviewed protein; total {len(U)}")
