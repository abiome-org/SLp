# desjardins2026 — Desjardins, Bowlan et al. 2026 Cell Rep (Repare)

Parser: `src/slbench/sources/staging/desjardins2026.py`. This module also holds the shared scoring engine
and the gate checks that `feng2022.py` and `deweirdt2020.py` import.

**Data.** Mendeley 10.17632/k6wm46g4tw v1 (CC BY 4.0). The `/files?folder_id=root` listing returns `[]`,
but the dataset record (`/public-api/datasets/k6wm46g4tw`) has a `files` array with download URLs and
sha256 (all 17 verified). The record is saved as `data/raw/desjardins2026/dataset_record.json`.

**Design.** Each file has parental T0/T18A/T18B and altered-clone T0/T18A/T18B. The replicates are
technical: same clone, same T0.

**Arms kept.** Only "-/-" knockouts:
- FBXW7, CDK12, STK11, TET2 in RPE1 (hTERT, Cas9, TP53-/-)
- ARID1A, KMT2D in BEAS-2B

**Arms excluded:**
- CCNE1: overexpression
- IDH1 R132H, DNMT3A R882H, SF3B1 K666N, SRSF2 P95H, U2AF1 Q157R: knock-ins
- Chr18q and Chr13q: arm losses
- KEAP1 in A549: A549 is KEAP1-mutant, and the altered arm is a KEAP1 *rescue* (re-expression), not a KO
- The two PKMYT1i files: drug screens

The data also contain CDK12, KMT2D and TET2 knockouts and a DNMT3A R882H knock-in, which the brief's list
did not name.

**Score.** Residual z of the KO-arm LFC against a running median over genes ordered by WT-arm LFC, scaled
by a running MAD. This absorbs clone growth-rate differences and the essential-gene floor.

**Positive requirements.**
- z <= -4 in the pooled score, and z < 0 in both replicates.
- LFC_KO < -0.5. Without this, tumour suppressors that give a growth gain in WT but not in the KO
  (TSC2, NF1, PTEN, LATS2) dominate the "negative GI" list.
- The gene is not a hit for >= 3 queries.

**Why -4.** A null that scores parental arm vs parental arm from different screens of the same line gives
56% of the query hit rate at z <= -3 and 32% at -4.

**Results.**
- 106,561 pairs, 245 pos (0.23%), 72,384 neg.
- Per query 0.11–0.43%; no hubs.
- Technical split-half 0.96/0.91. Fitness AUROC 0.67. 33 of 17,808 symbols unresolved.
- Positive controls: CDK12–CDK13 z = −13, ARID1A–ARID1B z = −6.6.

**Cross-study.** There is no same-line overlap. Against feng2022 (HEK293-A; STK11, ARID1A) the result is at
chance: 0.46 with 98 pos (our labels), 0.51 with 27 pos (theirs).

**Verdict.** Include, but borderline. The only evidence is technical replication plus the null
calibration. Downgrade to measurements-only if independent replication is required.
