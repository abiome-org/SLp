# PAGAN (genes-to-pairs, 2026)

**Source:** [Rausell Lab preprint](https://www.medrxiv.org/content/10.64898/2026.01.27.26344931v1) and
[official code](https://github.com/RausellLab/PAGAN). Acquired at
`external/models/pagan` (ignored by git). The authors release human and
budding-yeast single-gene features, essentiality labels, pair datasets and
KGs. The genes-to-pairs method learns *single-gene essentiality* and predicts
pair nodes without pairwise SL training labels.

**SLB adapter:** `scripts/models/pagan/run.sh` runs a fixed 16-hidden-unit,
two-layer heterogeneous GraphSAGE variant with the authors' sum aggregation,
tanh, 0.5 dropout, Adam 0.01 and zero-feature candidate pair nodes. Each
pair node receives the union of its members' PPI/paralog neighbors and GO
terms; direct interactions between the two members are excluded. The two
layers of pair inference are computed in batches. A three-pair check against
an explicitly augmented PyG graph agreed to 1.2e-8 in predicted probability.

The adapter uses PAGAN's K-562 essentiality labels, yeast essentiality labels,
four human strict gene features and 17 yeast single-gene features. Its KG is
rebuilt from the SLB bundle: GO annotations with IGI evidence removed,
BioGRID physical edges with at least two observations or STRING database
confidence >=700, and DIAMOND paralogs with identity >=30%. Term hierarchy
is retained. This changes the authors' KG intentionally to satisfy the SLB
leakage contract. Pair neighborhoods are capped at 30 PPI/paralog neighbors
and 32 GO terms per member for reproducible, bounded inference. One K-562
essentiality model is used for all human contexts; fission yeast is tied.
Validation and early stopping use only a random 10% of **single-gene** labels;
no SLB dev/test pair labels select the checkpoint. A stable unlabeled gene
universe is built from train/dev/test inputs, never their labels. The model
is an SLB adaptation of PAGAN's genes-to-pairs approach, not an exact
replication of all settings in the preprint.

| SLB-1.3 split | SLB | Human | Budding yeast | Fission yeast |
|---|---:|---:|---:|---:|
| Dev | 0.553 | 0.609 | 0.550 | 0.500 tied |
| Test | 0.520 (95% CI 0.490–0.555) | 0.591 | 0.469 | 0.500 tied |

The test prediction file and result JSON are in `results/models/slb1.3/`
as `pagan__clean_n2p_test.*`; all human and budding-yeast rows are natively
scored. The adapter saves single-gene checkpoints in the ignored authors'
repo under `_slb/slb1.3/clean_bundle_v2/`.
