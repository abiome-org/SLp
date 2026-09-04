# SLp-1.1 molecular evaluation

This application-neutral OMF module evaluates molecular prediction profiles. It
does not load synthetic-lethality labels or choose checkpoints. Both inputs are
immutable snapshot directories containing `evaluation.json` plus checksum-pinned
`.jsonl` or `.jsonl.gz` shards.

The reference input has role `molecular-reference` and contains fitting-only
perturbation profiles. The prediction input has role
`molecular-validation-predictions`, names the exact model checkpoint, and pins
the SHA-256 of the reference manifest. Both manifests declare
`labelClass: molecular`, `benchmarkLabelsPresent: false`, NCBI taxonomy IDs,
stable source IDs, their originating snapshot digest, value space, and shards.

Each JSONL record has this shape:

```json
{
  "speciesTaxon": 4932,
  "sourceId": "costanzo:2016",
  "centeringGroup": "condition:rich-media",
  "perturbationId": "SGD:S000000001",
  "interventionIds": ["SGD:S000000001"],
  "readoutIds": ["RNA:feature-1", "RNA:feature-2"],
  "target": [0.2, -0.1],
  "predictionMean": [0.18, -0.08],
  "predictionLogScale": [-1.0, -1.0]
}
```

The last two fields occur only in prediction records. Repeated records for the
same perturbation are averaged into a centroid. Individual intervention IDs are
audited across the reference and validation inputs, so a held gene nested in a
combination is still leakage.

For each species, source, centering group, and readout, the module averages the
training perturbation centroids with equal perturbation weight. It reports
ordinary error/correlation plus Pearson and cosine metrics after subtracting
this training perturbed-centroid reference. It also reports strict Euclidean
centroid accuracy over the common observed readout panel. This is a sparse,
multi-modal adaptation of Systema (DOI
`10.1038/s41587-025-02777-8`), not a claim of exact reproduction of its
single-cell benchmark. Undefined per-profile reference metrics contribute zero
to macro means and are counted explicitly.
