# StanX (breast-cancer attention variant of KG4SL)

battery: none (covered by kg4sl); species: -; needs: SynLethKG (KG4SL interface)

- Repo: https://github.com/bishesh4562r/StanX @ f5dac1d793543ebb790a2a8e73fcf6f453f89bf8 (no license, no paper;
  student modification of KG4SL, TF1 + dgl 0.5). No weights; KG4SL data files missing (../data).
- Its preprocessing already removes SL_GsG / SR_GsrG / NONSL relations and KG triples between SL pairs.
- Status: **acquired; not run separately.** Architecturally KG4SL with a breast-cancer attention tweak restricted to
  breast-related nodes; SLB has no breast-cancer-only context, so the pooled SLB run is the `kg4sl` result
  (notes/models/kg4sl.md).
