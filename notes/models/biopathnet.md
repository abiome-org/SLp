# BioPathNet (NBFNet-style path-based link prediction on biomedical KGs; SL task)

battery: none yet; species: -; needs: KG triples (background graph) + SL supervision edges; TorchDrug with a compiled
CUDA extension

- Paper: Nature Biomedical Engineering (2025), doi:10.1038/s41551-025-01598-z; Zenodo 10.5281/zenodo.16944505.
- Repo: https://github.com/emyyue/BioPathNet @ b051a9003a1363b5655cdb7dfdad60609c0e4fe3, MIT. The only checkpoint in
  the repo (experiments/.../model_epoch_12.pth) is a 134-byte Git-LFS pointer for a mock-data demo; no SL weights.
- Original SL setup (reproduce/synleth/): KR4SL's KG (GO + pathways) as background graph, SynLethDB pairs with
  statistic_score >= 0.90 as supervision (train2/valid/test). KR4SL's transductive facts.txt adds SL_GsG edges to the
  background graph (would have to be removed / restricted to training pairs). Positives only + negative sampling.
- Status: **acquired; not run.** The SL-free KR4SL-style KG for SLB genes already exists (scripts/models/kr4sl/build.py),
  so the data side is ready; the blocker is compute/environment: BioPathNet needs TorchDrug built from source with its
  rspmm CUDA extension (JIT-compiled; this host has no nvcc, so it needs a CUDA-devel container) and a GPU slot, which was
  queued behind four other agents during this session.
