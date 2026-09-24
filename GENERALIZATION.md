# Gene exposure and synthetic lethality: what the benchmark says

## CV1 is a gene-exposure split

In CV1, the **tested pair's label is withheld**, but both genes appear in
other training pairs. CV2 withholds one gene from training pairs; CV3
withholds both. Released model weights or knowledge graphs may separately
contain outside SL labels, which requires a provenance audit. The distinction
matters: CV1 can reward learning gene-specific SL degree or graph placement
without requiring a rule that transfers to a new gene.

In a [controlled 2024 comparison of 12 methods](https://www.nature.com/articles/s41467-024-52900-7),
the median F1 falls from **0.861** in CV1 to **0.669** in CV3. With that study's
1:1 random negatives, always predicting positive gives F1 **0.667**. Median
NDCG@10 falls from **0.239** to **0.002**. [Figure 1](figures/01_gene_exposure.svg)
plots every method from its Table 3, so this pattern is independent of our
benchmark design. These are F1 and NDCG on SynLethDB, not SLB AUROC.

## What survives SLB's harder holdout

SLB-1.3 holds out whole paralog/ortholog families from SL pair-label training,
uses experimentally measured negatives, and balances single-gene fitness
within context and screen. [Figure 2](figures/02_slb_held_out_test.svg) shows
the held-out test score beside the human component. The three-species score
is **0.645** for SLp Fusion, **0.640** for Ontotype and **0.625** for GO/PPI GBM.
The top confidence intervals overlap. [Figure 3](figures/03_species_signal.svg)
shows that even these broad methods vary sharply by species.

The newly audited 2026 methods sharpen the conclusion. A clean SLB refit of
the [SL-Predict](https://github.com/j8ckfi/sl-predict) frozen DepMap MAE branch
scores **0.724** on the human component, and Ryan's clean paralog refit scores
**0.712**. They score only human, so their three-species scores are **0.575**
and **0.571**; unscored yeasts are tied at 0.500. The SL-Predict human score
averages ancestry/context strata; pooled human AUROC is lower (0.668).
The frozen MAE vectors alone score 0.688 on the human component under the same
classifier, versus 0.543 for raw coessentiality alone. Thus cold-family
transfer is possible, and a single-gene representation carries much of this
model's signal. [The dated 2026 audit](notes/models/2026_landscape.md) explains
which branches were actually scored and why other releases cannot be ranked
without contaminated inputs or missing mappings.

## Reading the comparison

Our Fusion was selected on this benchmark's dev split and covers all three
headline species. That gives it a real advantage on the three-species metric
over human-only methods. The same family holdout, measured negatives and
scorer apply to every ranked adapter, and the independent CV1→CV3 study
shows the gene-exposure effect without our model. The result supports a
specific opportunity for SLp: transferable gene and context representations
that retain performance on new families and across species. It does not
establish that no current model learns relationships.

Nor is every gene in a body literally unseen. A production query may involve
genes already measured in other pair screens, or genes with no pair labels but
available sequence, ontology, interaction and single-gene dependency data.
SLB's family holdout is a deliberate stress test of the latter situation;
it should be reported alongside seen-gene performance when judging a model
for a particular deployment mix. In SLB's human training pairs, **5,021**
distinct genes are observed; the test contains **1,231** genes from held-out
families. Those counts describe this benchmark, not the fraction of future
clinical queries that will be cold start.

The exact test scores, coverage and result hashes are in
[LEADERBOARD.md](LEADERBOARD.md); the frozen grader and model-selection ledger
are in [CLIMB.md](CLIMB.md).
