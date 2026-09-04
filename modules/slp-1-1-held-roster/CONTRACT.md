# Held-intervention roster contract

This module constructs an outcome-blind yeast intervention split from two or
more separately admitted identity-inventory artifacts. It never reads a
quantitative molecular value. Inputs retain `NCBITaxon:4932` and canonical
`SGD:S#########` identities; symbols and orthology substitutions are invalid.

Every inventory is a separate top-level stage input. At runtime the module
accepts only OMF's materialized copied `DatasetSnapshot` object with exactly
`resource`, `mode`, `path`, and `manifestDigest`. The resource URI must name the
`datasetsnapshot` kind and end in a literal SHA-256 revision; the artifact
manifest is independently SHA-256 pinned. The materialized directory must be
`.../inputs/<input-name>/<dataset-name>` consistently with that URI. Bare
paths, mutable revisions, alternate kinds, mounted data, extra fields, missing
paths, and inconsistent materialization paths fail.

Each input directory contains an `inventory.json` object with exactly:

```json
{
  "schema": "slp.intervention-identity-inventory/v1",
  "sourceId": "repository:immutable-accession",
  "sourceRelease": "immutable-version",
  "ncbiTaxon": 4932,
  "stableIdNamespace": "SGD",
  "identityMappingId": "sgd:immutable-mapping-release",
  "identityMappingSha256": "<lowercase SHA-256>",
  "inventoryFormat": "slp.intervention-identity-record/v1",
  "files": [
    {"path": "inventory-000.jsonl", "sha256": "<lowercase SHA-256>", "records": 1}
  ]
}
```

File entries are unique and path-sorted. Paths are canonical relative POSIX
`.jsonl` paths resolving to regular non-symlink files inside the artifact.
Every parent path component is also checked for symlinks. Digests and
record counts are recomputed. Every JSONL line has exactly:

```json
{"schema":"slp.intervention-identity-record/v1","interventionId":"SGD:S000000001","ncbiTaxon":4932,"qcPassing":true}
```

No outcome, score, label, abundance, expression, fitness, effect, p-value or
other quantitative readout field is permitted. Repeated identical records are
reported and collapsed; repeated identities with conflicting QC status fail.
The required identity-mapping ID and digest attest the pinned SGD mapping used
by the source adapter. Raw ORF spellings, including valid suffixed systematic
names, are mapped there; this roster module accepts only resulting SGD CURIEs
and does not implement a narrower ORF-name regex. Every protected inventory
must declare the exact same mapping ID and digest; otherwise canonicalization
drift could alter the intersection and roster construction fails.

The candidate set is the intersection of QC-passing identities across every
protected source. For each sorted CURIE, compute lowercase SHA-256 of the exact
bytes `slp-1.1-yeast-global-held-v1\x00<SGD-CURIE>`. Interpret the first 16 hex
digits as an unsigned integer and reduce modulo 100. Buckets 0–9 are
`molecular-final`, 10–29 are `molecular-validation`, and 30–99 are `pretrain`.
An empty or configured-undersized intersection fails; there is no reroll.

The output `held-intervention-roster.tsv` has no header and contains exactly
`SGD-CURIE<TAB>role<TAB>hash`, path-sorted by CURIE with LF line endings.
`coverage.json` attests the algorithm, roster SHA-256, source manifests,
coverage, QC failures, and source-specific exclusions. The module bounds source
count, files, records, manifest bytes, and JSONL line bytes before processing.
