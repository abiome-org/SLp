# exp2sl: EXP2SL (Wan et al. 2020, Front Pharmacol 11:112)

battery: exp2sl__cellline, exp2sl__pancell; species: human; needs: LINCS 2020 level-5 trt_sh signatures (data/raw/lincs2020, 11.7 GB gctx + siginfo/geneinfo, sha256 in SOURCES.tsv), CPU torch venv external/models/exp2sl/.venv

PPI/KG: none (L1000 shRNA signatures only).

- Paper: Wan F, Li S, Tian T, Lei Y, Zhao D, Zeng J. EXP2SL: a machine learning framework for cell-line-specific synthetic lethality prediction. Front Pharmacol 11:112 (2020). doi:10.3389/fphar.2020.00112
- Repo: https://github.com/FangpingWan/EXP2SL (commit be7c75d, no license). The L1000 feature pickle (Google Drive 1lBJdTRSHtw16FIN45mmoEW-dJp53D2O4) now returns 404, so features were rebuilt from LINCS 2020 (trt_sh.cgs consensus gene signatures, 96 h preferred, 978 landmark genes).
- Released labels in the repo (GEMINI calls from Shen 2017 / Zhao 2018 / Big-Papi for A375, A549, HT29, HEK) are combinatorial-screen labels; no weights are released. Not used.
- Model ported to device-agnostic torch (the original hard-codes .cuda()); same architecture/loss (MSE to +-1 plus semi-supervised BPR over unknown pairs), Adam 1e-3, 1000 epochs, grad clip 5. Hyper-parameters fixed a priori (paper only gives grids): BPR weight 32, l2 0.01, 1 hidden layer, d 128. Not tuned on dev.
- Variants: __cellline = one model per SLB context that is an L1000 line (A-375, A-549, HT-29), trained on that context's SLB train rows, as in the paper; __pancell (extension) = signatures averaged over all L1000 lines, one model for all human contexts.
- Leakage: SLB train labels only; LINCS shRNA signatures are single-gene perturbation readouts => **not leaky**. Human only.
- Status: acquired / features rebuilt / ported / dev-scored.

## Dev (SLB-1.2)
| variant | native human coverage | SLB | H. sapiens |
|---|---|---|---|
| __cellline | 280/15,048 (A-375 55, A-549 120, HT-29 105 rows with both genes profiled) | 0.4997 | 0.4990 |
| __pancell | 4,361/15,048 | 0.4877 | 0.4508 |
Per-context (cellline): A-375 0.517, A-549 0.482, HT-29 0.432 (15/9/15 positives). Uninformative on SLB; limited by L1000 shRNA gene coverage (~3.3-4.3k genes). Runtime ~6 min CPU (after a one-off 10 min GCTX extraction cached in external/models/_statsl_cache/).

## SLB-1.3 dev results (results/models/slb1.3/; SLB_BENCH=data/bench/slb1.3)
Headline = mean of human/scer/spom; auxiliary columns are the per-species strata (bsub/cele/dmel have < 20 dev positives, not in the headline). Species a model does not score get a constant (AUROC 0.5); `native` lists the species actually scored (from <name>_dev.coverage.json).

| model | SLB | human | scer | spom | bsub | cele | dmel | paralog | native |
|---|---|---|---|---|---|---|---|---|---|
| exp2sl__cellline | 0.4994 | 0.4981 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.4942 | human |
| exp2sl__pancell | 0.5157 | 0.5470 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5619 | human |

Reference on SLB-1.3 dev: lgbm 0.5315 (human 0.558), paralog_identity 0.5392 (human 0.608), codependency 0.5149, random 0.4854.
