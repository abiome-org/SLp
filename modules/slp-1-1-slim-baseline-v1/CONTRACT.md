# SLIM native-panel comparator

This module copies the bilinear closed-form algebra of SLIM commit
`5a7e9ade5d0a6b6331e6dbc81181450605047bcc`. It uses SLIM's default PCA
basis, training-perturbation mean bias, unstandardized intervention embeddings,
`K=10`, and ridge value `0.1`.

The adaptation supplies SLp's static577 intervention descriptors in place of
published STRING embeddings and fits control-anchored native-panel molecular
residuals in place of GEARS-normalized expression. It is therefore a matched
feature/native-panel comparator, not a reproduction of SLIM's published
canonical benchmark scores. It has no gene-ID parameters. Quantitative outcomes
from held development interventions are excluded from fitting.

The optional `K=32` arm is a declared development diagnostic. It is not used to
select or relabel the fixed `K=10` primary comparator.

`scripts/run_slp11_slim_cv.py` is the stronger matched-feature arm. It
standardizes every static577 column using fitting genes only and selects rank
and ridge strength by deterministic three-fold fitting-gene CV. Each fold fits
its own normalizer and PCA basis. The selected model is frozen before the
existing development arrays are loaded.

`scripts/prepare_slp11_string_features.py` prepares an optional independent
STRING64 feature source from the exact embedding file tracked by the pinned
SLIM repository. It maps stable Ensembl IDs through the source Replogle GTF,
retains explicit missingness, and contains no molecular outcome values.
