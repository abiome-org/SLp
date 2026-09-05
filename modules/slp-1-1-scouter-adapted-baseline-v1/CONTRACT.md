# Scouter adapted baseline v1

This module adapts the compressor-generator architecture introduced by Ouyang
Zhu and Jun Li in *Scouter predicts transcriptional responses to genetic
perturbations with large language model embeddings*, Nature Computational
Science (2025), DOI `10.1038/s43588-025-00912-8`. The author implementation at
`PancakeZoy/scouter`, pinned separately by the experiment protocol, is MIT
licensed. Its copyright and permission notice are reproduced in `LICENSE`.

The control compressor has hidden widths 2048 and 512 and a 64-dimensional
bottleneck. The generator has one 2048-wide hidden layer and emits one value
for every member of a fixed molecular readout panel. Hidden layers use batch
normalization followed by SELU. AlphaDropout is supported and disabled for the
frozen comparison. Action vectors are summed before concatenation with the
control state, matching Scouter's combination rule.

This is an adapted baseline, not a reproduction of the paper's numerical
results. It substitutes versioned ESM/GO/physical features for GenePT text
embeddings, stable Ensembl identities for display symbols, pooled pseudobulk
control profiles for sampled single control cells, and a fixed exposure-aware
Gaussian loss for the autofocus direction-aware loss. It trains a separate
model per source context, as the paper trains separate models per dataset.

The fixed output panel is acceptable for this baseline comparison but cannot
query unseen molecular readouts. The model predicts an endpoint aggregate and
does not expose a reusable world state, calibrated new-context uncertainty,
time dynamics, or single-cell generation. Summing action features permits
multiple inputs structurally but does not identify genetic interactions from
singleton training. No learned gene identifier or mutable gene vocabulary is
used.
