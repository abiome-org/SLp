# Candidate SLB build with new human screens (`data/slb_next`)

This is a side-by-side build: `data/slb`, its pinned results and the leaderboard are unchanged. Each
candidate source is a parser in `src/slbench/sources/staging/<key>.py`. Its gate checks are in
`data/interim/new_human/<key>.{json,md}`, following the brief in `data/interim/new_human/BRIEF.md` and
the same inclusion rule as SLB. `stage_measurements.py` writes the included sources to
`data/interim/measurements_next/`, and `build_next.sh` builds with them
(`SLB_OUT`/`SLB_WORK`/`SLB_EXTRA_MEASUREMENTS`).

```bash
bash scripts/next/build_next.sh                                   # stage + examples + splits -> data/slb_next
TRANSFER_BENCH=data/slb_next uv run python scripts/transfer/crossfit.py build --scope all   # transfer track on it
```

The project is non-commercial, so NC-licensed sources are included; a public bundle built from this must
be marked non-commercial. Hayward 2026 and Nebenfuehr 2026 carry NoDerivatives terms, and Fong 2025 /
Najm 2018 are under journal terms. Check those before any public release.

## Sources (surveyed 2026-09-25)

| key | lines | licence | positives | verdict | main evidence |
|---|---|---|---|---|---|
| billmann2026 | HAP1 (query × genome, 127 queries, rich medium) | CC BY | 8,261 | include | independent re-screens of the same query 0.87; unrelated query 0.49 |
| herken2026 | K562 (CRISPRi, 313 genes) | CC BY | 872 | include | two independent maps 0.96 / 0.97; vs Horlbeck 0.80 / 0.84 (same platform) |
| simpson2023 | K562 (CRISPRi, preprint) | CC BY | 499 | include | two screens 0.98 / 0.94; vs Horlbeck 0.73–0.88 |
| najm2023 | THP-1, Reh | CC BY | 341 | include | independent libraries in the same lines 0.66–0.82 |
| deweirdt2020 | MEL-JUSO, OVCAR-8, A375, HAP1 (anchor × genome) | CC BY | 255 | include | independent anchor guides 0.76 / 0.85; 2 screens unlabelled (fail) |
| hayward2026 | MCF10A, RPE1 (233 DDR genes, preprint) | CC BY-NC-ND | 146 | include | replicates 0.88–1.00 with per-replicate T0; SPIDR recovers RPE1 labels 0.68 |
| lenoir2021 | MOLM-13, NOMO-1 | CC BY | 68 | include | independent infections 0.93–0.99 |
| burgold2025 | HT-29 (40 anchors × 444) | CC BY | 58 | include (borderline) | leave-one-replicate-out 0.68–0.71 with independent references |
| li2022 | IPC-298, PK-1, MEL-JUSO | CC BY | 27 | include | architectures/orientations 0.74–1.00; 19 of 27 positives are planted controls |
| wolf2025 | HCT 116 (Cas12a arm; Cas9 unlabelled) | CC BY | 26 | include | Cas9 recovers Cas12a labels 0.73–0.77; replicates 0.71–0.89 |
| najm2018 | 5 lines (786-O unlabelled) | journal terms | 17 | include | Sp/Sa orientation 0.80; same-line included studies 0.81 / 0.83 |
| desjardins2026 | RPE1, BEAS-2B (query × genome) | CC BY | — | measurements-only | technical replicates only; WT-vs-WT null gives about 32% of the hit rate |
| feng2022 | HEK293-A (query × genome) | CC BY-NC | — | measurements-only | WT-vs-WT null gives 51–83% of the hit rate |
| fong2025 | 7 lines | journal terms | — | measurements-only | replicates don't agree on GI (0.45–0.60); single-gene fitness predicts calls at 0.80 |
| ford2023 | 3 TNBC lines | CC BY | — | measurements-only | 7 positives, fitness predicts at 0.92 |
| kim2025 | MDA-MB-231 | CC BY | — | exclude | no replicate data; hits in unexpressed genes |
| nebenfuehr2026 | eHAP | CC BY-NC-ND | — | exclude | replicates don't agree (Spearman ≈ 0), no depletion |

Not obtainable: Lin 2025 (Hart lab, 8 lines, RTK/DDR all-by-all; data repo 404, author request needed).

Measurements-only sources are not staged yet. When this build is promoted, they go into `EXCLUDED_SOURCES`
alongside their reasons, like the existing excluded sources.

## Effect on human SLB (`data/slb_next` vs `data/slb`)

| human | slb | slb_next |
|---|---|---|
| train / dev / test positives | 3,486 / 351 / 561 | 8,222 / 541 / 907 |
| dev+test unique positive pairs | 353 | 872 |
| dev+test gene families with a positive | 232 | 527 |
| dev+test non-paralog positive pairs | 90 | 603 |
| semi (one gene held out) positives | 1,017 | 5,447 |
| contexts | 50 | 60 |

Human test positives by ancestry: EUR 503, EAS 100, AFR 32, and 272 in lines without an ancestry
estimate (HAP1, MCF10A, RPE1). The ancestry-averaged human score leaves those lines out.

Build changes needed: `families.MAX_FAMILY` was raised from 400 to 1000. The genome-wide HAP1 screen
brings in the C2H2 zinc-finger family (474 genes, train-only by size), and the current build's largest
family is 309, so this does not change it.
