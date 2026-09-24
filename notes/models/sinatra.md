# sinatra: SINaTRA (Jacunski, Dixon & Tatonetti 2015, PLoS Comput Biol 11:e1004506)

battery: sinatra__released; species: human; needs: released predictions (figshare)

PPI/KG: SINaTRA's human network topology features come from its own PPI compilation (paper: BioGRID/HPRD-era physical PPI); cannot be changed (predictions only) - moot, the model is already leaky (yeast SL labels).

- Method: "connectivity homology": RF-type classifier on PPI-network topology features of gene pairs, trained on YEAST SL labels, applied to the human PPI network.
- Code: not released (no repository found; the paper points to figshare predictions only). Predictions: figshare 1501103 / 1501105 / 1501115 (CC BY 4.0), 109,358,780 Entrez-ID pairs with scores; data/raw/sinatra (sha256 in SOURCES.tsv).
- Leakage: trained on yeast SL labels, including pairs from SLB held-out (cross-species ortholog) families => **leaky**. No retrain possible without code.
- Changes: Entrez -> HGNC symbol via HGNC; max score over duplicate symbol pairs. Human only.
- Status: acquired (predictions) / dev-scored. Coverage 13,564/15,048 human dev rows (1,484 median-filled + 117,674 non-human).
- Dev: SLB 0.5042; H. sapiens 0.5169 (AFR 0.570, EAS 0.477, EUR 0.504); human flat 0.482; paralog 0.490. Runtime 1.5 min.

## SLB-1.3 dev results (results/models/slb1.3/; SLB_BENCH=data/bench/slb1.3)
Headline = mean of human/scer/spom; auxiliary columns are the per-species strata (bsub/cele/dmel have < 20 dev positives, not in the headline). Species a model does not score get a constant (AUROC 0.5); `native` lists the species actually scored (from <name>_dev.coverage.json).

| model | SLB | human | scer | spom | bsub | cele | dmel | paralog | native |
|---|---|---|---|---|---|---|---|---|---|
| sinatra__released | 0.5203 | 0.5609 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5466 | human |

Reference on SLB-1.3 dev: lgbm 0.5315 (human 0.558), paralog_identity 0.5392 (human 0.608), codependency 0.5149, random 0.4854.
