# MGE4SL (Multi-Graph Ensemble for SL)

battery: none (not adapted); species: -; needs: per-pair KEGG / CORUM / STRING / Reactome / GO co-annotation
features over a fixed labelled pair table

- Paper: Lai M et al., "Predicting Synthetic Lethality in Human Cancers via Multi-Graph Ensemble Neural Network",
  IEEE EMBC 2021, doi:10.1109/EMBC46164.2021.9630716.
- Repo: https://github.com/JieZheng-ShanghaiTech/MGE4SL @ 8a7ae8150de0401b6905aa862a144c1c1b053d70, MIT. No weights.
  Also implemented in the Feng et al. 2024 code base (train_mge4sl.py; not part of the paper's 11-model comparison).
- Original data: SynLethDB (data/human_sl_encoder.csv: 35,906 SL + 3,588 other pairs incl. text-mined and
  computational pairs, 9,872 genes). Feng's version reads preprocessed_data/mge4sl_processed_data.csv (pair table with
  kegg, corum, STRING combined_score, reactome, go_P/C/F and the `sl` label).
- Status: **acquired; not adapted.** Every input graph (PPI, Reactome, CORUM, GO-P/C/F, KEGG) is built only from rows
  of the *labelled pair table* (`construct_kg_sldb(data)`), i.e. an edge can only exist between two genes that form a
  labelled pair, and the released table includes test positives (leaky by construction). A faithful SLB version would
  need a new pair-feature table (KEGG, CORUM, STRING, Reactome, GO) for every SLB train/dev pair; this is a different
  data pipeline (KEGG/CORUM/Reactome downloads) and the model was not in the Feng 2024 ranking, so it was deprioritised.
- Leakage status: released data leaky; not scored.
