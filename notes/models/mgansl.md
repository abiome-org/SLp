# MGANSL (multi-network representation generation with a GAN for SL)

battery: none (not adapted); species: -; needs: precomputed per-gene PPI / GO / "gauss" 128-d features

- Paper: BMC Bioinformatics (2025), doi:10.1186/s12859-025-06345-4. Repo: https://github.com/lijinxinchina/MGANSL @
  caa5c642e378f46ce0a3ee526c13a585b10ad2ae (no README, no license). torch 1.0 / python 3.6. No weights.
- Original data: SynLethDB v1 (19,667 pairs, 6,375 genes; SLDB/). Features feature_ppi_128, feature_go_128,
  feature_gauss_128 are shipped for those 6,375 genes only; the code that generates them is not released. The "gauss"
  view is most likely a Gaussian interaction-profile kernel of the full SL matrix (label-derived: leaky, and undefined
  for held-out genes).
- Status: **acquired; not adapted.** Not a graph model at training time (MLP + GAN over precomputed network
  embeddings); feature generation code missing, so SLB genes outside the 6,375 cannot be featurised faithfully.
