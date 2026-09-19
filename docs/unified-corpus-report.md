# Unified SLp corpus lineage registry

## Scope and evidence boundary

This report unifies the supplied 136-row fleet master registry with the SLp bucket census/reconciliation and the Round 1 source-object manifests. It is a **lineage and provenance index**, not a training-admission list: a `materialized_fleet_evidence` or `materialized_or_censused_SLP_raw` status means data/evidence was downloaded or reported by the bucket census, not that its rights, evidence semantics, or evaluation eligibility were approved.

Evidence was limited to these supplied files, read without object-body access:

- `/home/user/.capy/work/corpus-assembly/registry/master-registry.csv`
- `/home/user/.capy/work/corpus-assembly/round1/round1-source-objects.json`
- `/home/user/.capy/work/corpus-assembly/round1/round1-unpinned.json`
- `/home/user/.capy/work/abiome-bucket-scan/unified-corpus-reconciliation.md`
- `/home/user/.capy/work/abiome-bucket-scan/reconciliation.md`
- `storage/source-objects.json` and `rights/fleet-*.yaml`
- the deployed worker `/status` result for manifest version `2026-09-18-fleet-round1`

Bucket writes used only the token-gated, manifest-allowlisted worker. The final `/status` sweep on 2026-09-19 verified all 249 live objects by stored size and checksum; no claim below is a new license adjudication or measurement of training examples.

## Method

1. Started from all **136 fleet entries** and preserved every `entry_id` in `registry_entry_ids`.
2. Collapsed resource rows into one canonical row by their recorded `lineage_key`. Aggregator rows use the first recorded `aggregator:*` identity, because the rest of their lineage text describes mirrors rather than a new experiment.
3. Applied only explicit reconciliation overlap decisions: SCP1184→GSE157977 (Jin); SCP2554→GSE264667 (Nadig); `parse-10m-cytokines`→`parse10m_pbmc_90cytokines`; and SLp raw Costanzo, Harle, and DepMap associations. The registry retains the original IDs and aliases behind those merges.
4. Added the eight SLp raw source lineages from the bucket census: Costanzo, DepMap 24Q2, Feng, GO 2022-09-19, Harle, in4mer, SLKB, and SPIDR. Five are new to the master registry (Feng, GO auxiliary, in4mer, SLKB, SPIDR); Costanzo and Harle augment existing fleet lineages; DepMap 24Q2 is linked to the existing DepMap program metadata instead of counted again.
5. Attached each live source-object ID only where its object-key prefix can be explicitly mapped to a canonical lineage. Those object IDs document a concrete materialization candidate; they are not separate source lineages.
6. Kept prepared views and technical resamples as aliases/representation notes. They never generate another canonical experimental row. Rosetta is explicitly classified `technical_resample`; Parse pseudobulks are likewise described as an aggregate/resample representation within the Parse source row.

## Counts

| Measure | Count | Interpretation |
|---|---:|---|
| Fleet registry rows read | 136 | Resource rows in the master registry, not independent experiments. |
| Literal master `lineage_key` values | 115 | Before explicit cross-key reconciliation merges. |
| SLp raw source lineages added/augmented | 8 | The eight sources named in the bucket census. |
| Canonical registry rows | 116 | Includes experimental/source lineages, aggregators, one technical resample, and the GO auxiliary source. |
| Experimental lineages | 88 | Rows classified as a direct experimental lineage; excludes aggregators, auxiliary annotation, source database, and technical resample. |
| Source lineages | 13 | Repositories/programs/resources whose rows are not asserted to be one experiment. |
| Aggregators | 12 | Portals/benchmarks retained for provenance but excluded from experimental counts. |
| Technical resamples | 1 | Kept visible, but not treated as newly generated experiments. |
| Auxiliary annotations | 1 | GO, excluded from experimental counts. |
| Source databases | 1 | SLKB, separate from experimental-lineage counts. |
| Round 1 supplied manifest-snapshot objects | 221 | Objects in the supplied JSON snapshot. |
| Corrected live-manifest objects | 249 | Verified decomposition: 30 pre-existing base SLp objects + 219 live fleet objects. |
| Round 1 supplied manifest-snapshot bytes | 9,246,696,858 | Sum of bytes in the supplied fleet snapshot. |
| Corrected live-manifest bytes | 15,728,970,705 | Sum of all 249 pinned object sizes (14.649 GiB). |
| Live verified objects | 249 | Worker `/status`: 249 verified, 0 missing, 0 mismatch. |
| Round 1 supplied manifest-snapshot source prefixes | 24 | `slp/raw/<prefix>/...` groups in the supplied snapshot, not canonical lineages. |
| Unpinned records | 2 | Separate from live pinned objects. |

The reduction from 136 fleet rows plus eight SLp raw records to 116 canonical rows is intentional: multiple resource rows can represent the same lineage, and aggregator/prepared representations are not independent experiments.

## Readiness: admitted to registry, deferred, and blocked

“Admitted” here means admitted to the **registry index** under the evidence boundary above, not admitted for model training or redistribution.

| Registry readiness | Canonical rows | Meaning |
|---|---:|---|
| `blocked_acquisition` | 10 | Associated entries are acquisition-blocked without a materialized/metadata fallback. |
| `manual_import_deferred` | 1 | BioGRID ORCS has zero pinned live-manifest objects; two tarballs await manual import. |
| `materialized_fleet_evidence` | 61 | At least one associated fleet row has `downloaded_open`, excluding later-deferred ORCS and four storage-only Springer groups. |
| `materialized_or_censused_SLP_raw` | 8 | An SLp raw item was reported by the bucket census; some also have fleet evidence. |
| `metadata_or_manifest_only` | 32 | No associated `downloaded_open` row; metadata/manifest or not-attempted evidence exists. |
| `storage_only_rights_deferred` | 4 | Big Papi, Boettcher AXKO, Han CDKO, and Shen combinatorial have `trainingAllowed=false` and `redistributionAllowed=false`. |

Rows with rights caveats are **deferred from any generalized training/redistribution claim** even when their acquisition readiness is materialized:

- **Costanzo 2016:** `rights-unresolved`. The reconciliation says its own manifest calls for reconciliation before fitting. It is one lineage with FSL03, not a duplicate fleet/bucket count.
- **SLKB 2023:** `GPL-separate`. The reconciliation reports GPL v3.0 / GPL 3.0+ for data and pipeline. It is a database/assertion source rather than a single experiment; keep artifacts derived from it out of mixed non-GPL pools unless separately adjudicated.
- **DepMap:** the registry distinguishes historical 24Q2, which the reconciliation calls CC BY 4.0, from current releases. It does not generalize 24Q2 terms to 26Q1 or later.
- **Big Papi, Boettcher AXKO, Han CDKO, and Shen combinatorial:** verified correction classifies all four non-open Springer supplement groups as storage-only pending rights adjudication, with `trainingAllowed=false` and `redistributionAllowed=false`.

The master registry’s existing `blocked_budget`, `blocked_credentials`, and `blocked_terms` states are retained in the canonical rows through `acquisition_readiness`, entry IDs, and verification notes. No blocked source was silently converted into an acquired source.

## Rights pools

| Rights pool | Canonical rows | Basis and handling |
|---|---:|---|
| `GPL-separate` | 1 | SLKB is kept out of mixed non-GPL artifacts. |
| `as-recorded; no unified rights conclusion` | 75 | Recorded source terms/statuses aggregated conservatively; see row-level `rights_basis` and original entry IDs. |
| `manual-import deferred` | 1 | BioGRID ORCS live-manifest scope is zero pinned; two tarballs await manual import. |
| `open-attribution (per reconciliation)` | 6 | Based only on reconciliation statements, not a new adjudication. |
| `research-only/noncommercial` | 8 | Existing source terms/statuses retained. |
| `rights-unresolved` | 1 | Costanzo remains unresolved. |
| `storage-only rights-deferred` | 4 | Springer groups: trainingAllowed=false; redistributionAllowed=false. |
| `terms-gated` | 20 | Existing source terms/statuses retained. |

The reconciliation names CC0/CC BY/MIT material as an open-attribution pool, current/terms-gated resources as separate, noncommercial material as research-only, Costanzo as unresolved, and SLKB as GPL-separated. This report preserves those classifications but does not establish a blanket training or redistribution permission for any pool.

## Explicit overlap and representation decisions

| Canonical lineage | Decision |
|---|---|
| Costanzo 2016 | Fleet FSL03 and SLp `pairwise.zip` are one Costanzo 2016 yeast SGA lineage. `population-yeast` and `response-yeast` are prepared views. Rights remain unresolved. |
| DepMap | The 24Q2 SLp raw release is represented once and linked to fleet FSL08’s 26Q1 metadata reference. This is a program/release linkage, not a claim that versions have identical files or rights. |
| Harle 2025 | Fleet FSL15 and SLp DATA/METADATA tars are one lineage; reconciliation calls them the same Figshare objects. `harle-fitness` and `harle-gi` are prepared outcome views. |
| Replogle/GWPS | `population-*`/`response-*` GWPS, K562, and RPE1 views are retained as aliases of the Replogle/GWPS lineage, not independent studies. |
| Norman, Nadig, MCF10A | Norman prepared views map to GSE133344; Nadig SCP2554 merges into GSE264667; MCF10A d0/d6/TGFB1-d6 are views of the LINCS MCF10A source. |
| Parse 10M | The separate pseudobulk aggregate label maps to Parse10M and is explicitly a technical aggregate/resample, not a new experiment. |
| Rosetta cpg0003 | Classified as `technical_resample` because its master description says it is harmonized reprocessing and “not newly generated.” |
| Aggregators | scPerturb, PerturBase, PerturbDB, GEARS, scPertEval, pertpy, CZI VCP, CELLxGENE, and Arc VCA entries remain provenance rows only. Their stated upstream mirrors are not expanded into inferred experimental rows. |

## Round 1 manifest and verified correction handling

The supplied Round 1 JSON snapshot contains **221 objects totaling 9,246,696,858 bytes across 24 source prefixes**. The verified live-manifest count is **249 objects**, not 250, with explicit accounting: **30 pre-existing base SLp objects + 219 live fleet objects = 249**. The live fleet contribution is the 221-object Round 1 snapshot less the **two BioGRID ORCS tarballs** deferred to manual import. This makes the 28-object difference from the Round 1-only snapshot attributable to the pre-existing base, not an unattributable delta.

- **BioGRID ORCS:** the live-manifest scope is **zero pinned objects**. Two ORCS tarballs are deferred to manual import. The snapshot’s two ORCS object IDs are removed from the canonical row as superseded, while FSL01 remains as the source-lineage record.
- **TG-GATEs:** `tggates-2012-cel-file-attribute-zip` was a **13,429-byte publisher 404 page**, not corpus data. It is deferred/inadmissible and excluded from live-manifest and corpus counts. Its removal is a correction within the fleet composition, not an additional live-object delta: two real TG-GATEs archives were corrected and ingested, and their two object IDs remain associated with the source lineage.
- **NCI-ALMANAC:** `ComboDrugGrowth_Nov2017.zip` remains unpinned in `/home/user/.capy/work/corpus-assembly/round1/round1-unpinned.json` because its wiki attachment URL was unresolved. The source remains a registry row with no invented checksum or source URL.
- **Four Springer supplement groups:** Big Papi, Boettcher AXKO, Han CDKO, and Shen combinatorial are storage-only pending rights adjudication, with `trainingAllowed=false` and `redistributionAllowed=false`. They are excluded from any training/redistribution admission claim despite checksummed object records.
- **Rosetta:** the README checksum was corrected after identifying a basename-collision bug in the builder. The correction now matches full path suffixes. It strengthens object provenance only; Rosetta remains a technical resample, not a new experiment.

## Limitations and next controls

- The live worker status read R2 object metadata for every manifest entry and verified each stored size and checksum. Object bodies, prepared claim files, and release datasets were not read back during this reconciliation.
- The supplied reconciliation describes roughly 75 prepared directories. Their individual claim files were not audited here, so prepared aliases are included only where the reconciliation explicitly named them. The 24 new fleet rights records were validated structurally, but unresolved rights remain deferred as stated.
- `master-registry.csv` has source-level and resource-level entries mixed together. Classification in this report is deliberately conservative: `source_lineage` means a source/program identity, not a claim of one biological experiment.
- Evidence types are copied from the master registry or described exactly at the reconciliation’s level. The report does not re-label every source into the SLp 12-class evidence taxonomy.
- Counts are lineage/index counts, not counts of files, samples, gene pairs, observations, or training examples. They must not be used for coverage or statistical weighting without examining the underlying materials.

## Deliverables

- `storage/unified-lineage-registry.csv` — one canonical row per retained experimental/source lineage or explicitly nonexperimental provenance identity, with entry IDs, aliases, all 249 live source-object IDs, rights/readiness, and deduplication decisions.
- `docs/unified-corpus-report.md` — this report and its evidence boundary.
