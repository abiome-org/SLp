# EFOL-SL (interpretable complex-knowledge multi-hop reasoning for SL)

battery: none (not runnable as released); species: -; needs: SynLethKG + BetaE-style logical query files

- Paper: "An Interpretable Complex Knowledge Multi-Hop Reasoning Model for Predicting Synthetic Lethality in Human
  Cancers", IEEE (record 11219295). Repo: https://github.com/Cheng0829/EFOL-SL @ 83ca22462bfe4deced63cc4354214729146e0163
  (license: none stated). torch 1.7-1.9 + PyG. Weights only on Google Drive (folder 1fBAwWtJiq7RPufBPvhPEhFsyHIB3oIm9).
- Original data: SynLethDB v1.0 / v2.0 (KG4SL-style sl2id, 36,402 pos + 36,402 sampled neg) + SynLethKG; the shipped
  sl_gene_data.txt contains SL_GsG / SR_GsrG / NONSL_GnsG edges (removable by relation name).
- Status: **acquired; not adapted.** The released code is a generic KG query-reasoning template: configs/configs.json
  (called by the README) is missing, the remaining configs are drug-task configs with F:/kg-datasets paths, metric.py
  hard-codes Windows DrugBank pickles, and nothing converts SL pairs into the BetaE *-queries.pkl / *-answers.pkl
  files the trainer reads. Blocker: missing SL-to-query pipeline and configs.
