# ortholog_gi_transfer: cross-species GI transfer through orthologs (LEAKY diagnostic)

battery: ortholog_gi_transfer; species: every species with orthologs in the family graph (human, scer, spom,
dmel; + cele/mmus/bacteria via slpbench.families_extra when present); needs: data/interim/measurements/*.parquet,
slpbench.families.edges() (+ families_extra.extra_edges()).

## What it is
The idea behind Slorth-style cross-species SL prediction and "conserved genetic interaction" transfer
(Dixon et al. 2008, Roguev et al. 2008, Ryan et al. 2012, Srivas et al. 2016): a pair is predicted SL if its
orthologous pair in another species has a strong negative GI. Score = max over ortholog pairs in other
species of (-score / SD of that source); sources = every table in data/interim/measurements (Costanzo 2016
epsilon, Ryan 2012 S, the human zdLFC / GEMINI screens, dual CRISPRi-seq, fly screens). Pairs with no
measured ortholog pair: 0 (`in_model=False`).

## Leakage: LEAKY, not rankable
SLB families join orthologs across species, so the ortholog pair of a dev pair lies in dev families and its
measured GI comes from the combinatorial screens behind SLB (and from excluded sources). This model is a
diagnostic of how well GIs are conserved, not a benchmark entry. No non-leaky version exists by
construction: the family split removes every cross-species ortholog of a held-out gene from train.

## Results (dev)
Full split: SLB 0.500 (human 0.456, scer 0.504, spom 0.541, spne 0.500).
Covered subset (pairs with a measured ortholog pair): human 2,132 pairs / 58 pos: 0.294; scer 6,431 / 73: 0.537;
spom 12,771 / 141: 0.556; spne 18 / 1: n/a.
- Yeast <-> yeast GI conservation is weak but positive (0.54-0.56), in line with the literature
  (Dixon 2008 / Roguev 2008: ~ a few-fold enrichment, most GIs not conserved).
- Human is ANTI-correlated (0.29). Cause: human paralog pairs from large families (histones H3, septins,
  BCAT1/2, OCRL/SYNJ1) whose yeast orthologs are 1:1 duplicates with very strong negative GI are neutral in
  human screens (more family members buffer), while the human SL positives (e.g. UBR1/UBR2, TIA1/TIAL1)
  map to yeast pairs with no or positive GI. Yeast GI strength does not transfer to human paralog SL.
Report: results/models/ortholog_gi_transfer_dev.txt. Runtime: ~2 min.

## Slorth (Benstead-Hume et al. 2019, PLoS Comput Biol)
The Slorth database of cross-species SL predictions (http://slorth.biochem.sussex.ac.uk) is offline (HTTP 404
/ 403 on 2026-09-24, no Wayback snapshot, no code repository found), so its predictions could not be scored.
They would be leaky in any case: the underlying SLant random forests are trained on BioGRID SL/GI data
from all five species, including Costanzo and pombe screens. SLant (the classifier itself) is covered by
models-features.

## Results (dev, SLB-1.3)
Full split SLB 0.531 (human 0.553, scer 0.507, spom 0.535; aux bsub 0.499, cele 0.471, dmel 0.591).
Covered subset (46,108 pairs, 689 pos): human 4,252 / 106 pos 0.568, scer 17,947 / 246 0.547, spom 23,812 / 331
0.547. The human anti-correlation seen on SLB-1.2 (0.29 on 2,132 covered pairs) is gone on SLB-1.3, whose human
dev set has different families and more non-paralog pairs; the 1.2 figure was a property of a small
paralog-heavy subset. Still LEAKY and not rankable. Reports: results/models/slb1.3/ortholog_gi_transfer_dev.txt.
