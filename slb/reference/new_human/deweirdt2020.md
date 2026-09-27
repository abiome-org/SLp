# deweirdt2020 — DeWeirdt, Sangree et al. 2020 Nat Commun (Doench lab), anchor screens

Parser: `src/slbench/sources/staging/deweirdt2020.py`. It uses the engine in `desjardins2026.py`.

**Data.** MOESM4 (counts; Brunello and HAP1 sheets) and MOESM5 (the authors' z).

**Design.** SaCas9 anchor guide plus the SpCas9 Brunello library. Only dropout arms are used. The
reference is pDNA, and the control-guide arm serves as WT.

**Screens kept:**
- MEL-JUSO: MCL1, BCL2L1
- OVCAR-8: MCL1, BCL2L1, PARP1 (2 anchor guides)
- A375: PARP1 (2 anchor guides)
- HAP1: PARP1 single-cell KO clone vs parental

**Screens not used.** Drug arms, the Gattinara library and the secondary targeted screen. Guides are mapped
through "Brunello reference"; guides that map to more than one gene and control guides are dropped.

**Agreement with the authors.** Our z matches MOESM5 at Spearman 0.90–0.96 in all 9 genetic screens. The
authors' score is the same quantity, but recomputing gives per-replicate values.

**Label rule.**
- Positive: z <= -3 in the pooled score, z < 0 in both replicates, LFC_anchor < -0.5, and no hit for
  >= 3 distinct anchors. The last condition drops OTUD5 and H2BC13, which look like SaCas9-cutting
  artifacts.
- Negative: |z| < 1.
- MCL1/MEL-JUSO and PARP1/A375 carry no labels because their own split-half is 0.59–0.60.
- The replicates are technical (A/B), plus gi_anchor1/2 (two independent PARP1 SaCas9 guides).

**Replication.**
- Split-half over the labelled screens: 0.715/0.713.
- Anchor-guide split: 0.76/0.85 pooled; OVCAR-8 0.94/0.92.

**Biology recovered.**
- BCL2L1↔MCL1 in both directions
- MARCHF5 and WSB2 with BCL2L1/MCL1
- PARP1 with PARP2, MUS81, EME1 and GEN1 in HAP1

**Results.**
- 127,055 pairs, 255 pos (0.20%), 61,662 neg.
- Per labelled screen 0.12–0.77%. HAP1 PARP1 is the largest at 140 pos; it is a single-cell clone
  (possible clone effects).
- DepMap fitness AUROC 0.55. 50 symbols unresolved.

**Cross-study.** No usable overlap.

**Verdict.** Include.
