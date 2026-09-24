# LukePi (self-supervised universal KG embedding, PrimeKG HGT)

battery: lukepi (human); species: human (PrimeKG is human); needs: PrimeKG HeteroData + pre-trained HGT checkpoint

- Paper: Tao et al., bioRxiv 2025 / IEEE (record 11189877).
- Repo: https://github.com/JieZheng-ShanghaiTech/LukePi @ f4ee6d790ba64ac19cc409cd63e8689325574b94 (MIT).
- Released weights: pre-trained HGT Primekg_HGT_0.2_0.001 + kgdata.pkl from the authors' Google Drive (self-supervised on PrimeKG, no SL labels; sha256 in MANIFEST).
- Original training data: SynLethDB-derived C1/C2/C3 splits (for the fine-tuning head only).
- Leakage status: **clean** for the SLB-trained variants (fit on SLB human train only; no SL/GI edges or GI-derived inputs; dev labels never read), except where noted below. PPI / KG sources and how GI evidence was removed: PrimeKG (authors' pre-built HeteroData), not filterable. POSSIBLY LEAKY: PrimeKG's protein_protein edges integrate several PPI resources (the PrimeKG paper lists STRING among them) and its GO edges are not evidence-filtered, so genetic-interaction-derived edges (STRING experimental channel, GO IGI) may be present; the self-supervised encoder was pre-trained on that graph by the authors and cannot be re-trained here. Treat the lukepi score as 'clean labels, possibly leaky inputs'.
- What we changed:
  - Fine-tuning head trained on SLB fit pairs (measured labels) exactly as src/finetune_LukePi.py + test_LukePi.sh (frozen encoder, 50 epochs, lr1 0.003, last epoch kept); validation AUROC only logged.
  - Released src/model.py does not import (IndentationError at `class GIN`); the HGT class is taken from the file text. The checkpoint needs the pre-2.3 PyG HGTConv (torch 1.13.1 + PyG 2.2.0).
  - SLB genes mapped to PrimeKG gene/protein nodes by symbol (PrimeKG node table from the MiT4SL data release).
- Adapter: scripts/models/lukepi/run.sh; env SLB_BENCH, SLB_SPLIT.

## Results (rows a model does not score get its median score before `slpbench eval`, as `--allow-missing` does)
### SLB1.3 dev
- lukepi: **SLB 0.4856**; H. sapiens 0.4568, S. cerevisiae 0.5000, S. pombe 0.5000, B. subtilis n/a, C. elegans n/a, D. melanogaster n/a, M. musculus n/a. Human ancestry: AFR 0.5061, EAS 0.4117, EUR 0.4526, unknown 0.6503. Rows scored by the model: 18,890 (the rest get the model's median score).
### SLB1.2 dev
- lukepi: **SLB 0.4688**; H. sapiens 0.3752, S. cerevisiae 0.5000, S. pombe 0.5000, S. pneumoniae 0.5000. Human ancestry: AFR 0.3095, EAS 0.3510, EUR 0.4650, unknown 0.4957. Rows scored by the model: 14,783 (the rest get the model's median score).
- Comment: Validation AUROC (held-aside train families, pairs pooled over contexts, unadjusted) reached 0.65 at epoch 50, yet the dev ranking is inverted (human 0.375, unadjusted within-stratum AUROC 0.30-0.47): what the head learns on pooled pairs does not transfer to within-screen, fitness-balanced ranking of held-out genes.
- Status: acquired, env built, adapted, dev-scored.
