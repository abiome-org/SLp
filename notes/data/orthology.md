# Extra homology edges for SLB gene families (data-orthology, 2026-09-24)

Goal: extend the family graph of `families.py` (paralogs >= 30% identity + cross-species orthologs;
held-out test genes must have no paralog or ortholog in train in any species) to the new species and to
S. pneumoniae, which had no homology edges at all in SLB-1.2.

Deliverables
- `data/interim/orthology_extra/edges.parquet`: columns u, v, kind (ortholog|paralog), weight (identity for
  paralogs, 1.0 for orthologs), source. 510,781 rows, u < v, nodes "species:canonical_id".
- `src/slpbench/families_extra.py`: `extra_edges()` returns (u, v, kind, weight) like `families.edges()`;
  `extra_edges_with_source()` keeps the source column; `all_edges()` = base + extra. Options:
  `base_pairs=False` drops DIAMOND RBH edges between two base species (human/scer/spom/dmel);
  `species={...}` restricts to edges between the listed species.
- `src/slpbench/ids_extra.py`: resolvers `mmus() cele() calb() spne() ecol() bsub() mtub() saur()`, plus
  `resolver(species)` / `resolve(species, series)` that also dispatch to `ids.py` for the base species.
- Scripts in `scripts/orthology_extra/`: `fetch.py` (downloads + SHA256SUMS + orthology.fetch.tsv),
  `build_edges.py` (proteomes, DIAMOND, edges), `validate.py` (-> validation.json),
  `simulate_families.py` (-> simulation*.json, simulated_families*.parquet).
- Rebuild: `uv run python scripts/orthology_extra/fetch.py && uv run python scripts/orthology_extra/build_edges.py`
  (about 5 min at 32 threads).

## Species and canonical IDs (agreed on the board with data-bacteria / data-eukaryotes)

| code | organism / proteome | canonical ID | proteins used |
|---|---|---|---|
| human | Ensembl 116 GRCh38 pep.all | HGNC symbol (ids.human) | 19,449 |
| mmus | Ensembl 116 GRCm39 pep.all | MGI symbol, e.g. Trp53 (MRK_List2) | 21,702 |
| cele | Ensembl 116 WBcel235 pep.all | WBGene ID (WormBase WS298) | 19,985 |
| dmel | Ensembl 116 BDGP6.54 pep.all | FBgn | 13,986 |
| scer | Ensembl 116 R64-1-1 pep.all | SGD ORF (ids.scer) | 6,600 |
| spom | PomBase peptide.fa | PomBase systematic ID | 5,126 |
| calb | CGD Assembly 22 default protein, haplotype A | ORF without allele suffix, C1_00060W | 6,212 |
| spne | GenBank CP027540.1 (D39V) | dualcrispri2025 author name, else GenBank gene name, else SPV_ tag | 1,993 |
| ecol | GenBank U00096.3 (K-12 MG1655) | b-number, b0002 | 4,290 |
| bsub | GenBank AL009126.3 (168) | BSU00010 (old_locus_tag; BSU_00010 also resolves) | 4,243 |
| mtub | GenBank AL123456.3 (H37Rv) | Rv number, Rv0001 | 4,018 |
| saur | GenBank CP000253.1 (NCTC 8325) | SAOUHSC_00001 | 2,892 |

One protein per gene: the longest isoform (Ensembl: protein_coding genes on the primary assembly only).
Bacterial resolver tables `data/raw/ids/{spne,ecol,bsub,mtub,saur}_genes.tsv` are derived from the GenBank
records (canonical, locus_tag, old_locus_tag, gene, synonyms, protein_id). Synonyms resolve only when
unambiguous, the same policy as `ids.Resolver`.

spne IDs: the benchmark uses the raw mmc4.csv names. 24 of the 530 SLB-1.2 spne genes differ from the
CP027540.1 gene name (e.g. SPV_0112 is `rtgR` in GenBank, `blpU` is a synonym of `blpK`). ids_extra matches
the author names to loci through the locus tag, gene name or synonym (unambiguous matches only), so 525/530
benchmark spne IDs are graph nodes. The 5 missing ones are `srf-09/14/15/20/23`, small RNA features with
no protein. `bacteria_extra.spne_table()` (data-bacteria) agrees with this on 2,255/2,266 loci. It differs on
11 loci where the author name is only a GenBank synonym (blpU/blpK, srf-06/shp144, vraTSR/liaFSR,
glnP4/glnQ4/artPQ, clyB/slgT, glpF/Pn-aqpB, SPV_0575/yesM). I posted this on the board and it is still open.

## Methods and thresholds

Sequence search: DIAMOND v2.2.8 (static binary `external/bin/diamond`, from
https://github.com/bbuchfink/diamond/releases/latest/download/diamond-linux64.tar.gz, tgz sha256
33a765d192ae4f5b388f15fe2e7c23d3aaea0f9e3f6872efe8c324aec261fbc7). All 114k proteins were searched against
each of the 12 proteomes separately with `blastp --more-sensitive --evalue 1e-5 --max-target-seqs 250`.
Searching per target proteome keeps a query's best hit in every species, including distant bacterial hits
that a single combined search would push out of the top 250. Hits were symmetrized because DIAMOND's
sensitivity is not symmetric: a pair found in either direction counts for both, and the best HSP per pair is
kept. Without this, spne rpsD <-> scer NAM9 was missed.

Edge sources:
1. `diamond_rbh` (ortholog, 124,955 edges, all 66 species pairs): reciprocal best hits by bitscore.
   Hits within 95% of the best bitscore count as best, which admits near-identical in-paralogs and
   co-orthologs such as ENO1/2/3. The alignment must cover >= 50% of the shorter protein, so single-domain
   matches of multidomain proteins are excluded. There is no identity threshold for orthologs, because
   bacteria <-> eukaryote orthologs often sit at 25-40% identity.
2. `alliance` (ortholog, 96,064): Alliance combined orthology (file already in SLB,
   `data/raw/orthology/ORTHOLOGY-ALLIANCE_COMBINED.tsv.gz`, Alliance 9.0.0, sha256
   977ad252878d25a6cf486a9054448feff0e308e83e2fe5d6e8b6ec31c45491d4). It is used only for pairs involving
   mouse (NCBITaxon:10090) or worm (6239); human/scer/dmel pairs are already in families.py. The filter is
   the same as families.py: reciprocal best in both directions, or >= 3 methods.
3. `ensembl_biomart` (paralog, 278,743): Ensembl 116 BioMart paralogs for mouse and worm. The query is the
   same as `slpbench.fetch.biomart_query`, and identity = max(%id, %id_r1)/100 >= 0.30, as in homology.py.
4. `diamond_self` (paralog, 11,019): within-species DIAMOND hits for species that have no Ensembl paralog
   table (calb, spne, ecol, bsub, mtub, saur). Identity = nident / min(qlen, slen), which is the analogue of
   Ensembl's max(%id query, %id target). Kept at >= 0.30, e-value <= 1e-5.
   Calibration against Ensembl paralogs (validate.py): Pearson 0.991 (scer), 0.993 (spom) and 0.962 (human)
   on shared pairs, with median difference 0.00. At the 30% cut DIAMOND finds 2,967 scer pairs vs Ensembl's
   2,388, with 2,253 shared, so it is slightly more inclusive (the leakage-safe direction).

Edge counts: DIAMOND RBH orthologs per species pair.

| RBH | human | mmus | cele | dmel | scer | spom | calb | spne | ecol | bsub | mtub | saur |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| human | - | 19410 | 6662 | 8048 | 2711 | 2881 | 2692 | 423 | 676 | 661 | 622 | 539 |
| mmus | 19410 | - | 6905 | 8082 | 2771 | 2941 | 2742 | 433 | 700 | 682 | 669 | 558 |
| cele | 6662 | 6905 | - | 6068 | 2229 | 2315 | 2182 | 350 | 620 | 587 | 585 | 452 |
| dmel | 8048 | 8082 | 6068 | - | 2332 | 2491 | 2341 | 346 | 631 | 576 | 546 | 474 |
| scer | 2711 | 2771 | 2229 | 2332 | - | 3167 | 3910 | 396 | 618 | 602 | 527 | 485 |
| spom | 2881 | 2941 | 2315 | 2491 | 3167 | - | 3019 | 363 | 600 | 563 | 512 | 459 |
| calb | 2692 | 2742 | 2182 | 2341 | 3910 | 3019 | - | 360 | 601 | 564 | 521 | 465 |
| spne | 423 | 433 | 350 | 346 | 396 | 363 | 360 | - | 844 | 1027 | 605 | 938 |
| ecol | 676 | 700 | 620 | 631 | 618 | 600 | 601 | 844 | - | 1359 | 1166 | 1070 |
| bsub | 661 | 682 | 587 | 576 | 602 | 563 | 564 | 1027 | 1359 | - | 1012 | 1485 |
| mtub | 622 | 669 | 585 | 546 | 527 | 512 | 521 | 605 | 1166 | 1012 | - | 784 |
| saur | 539 | 558 | 452 | 474 | 485 | 459 | 465 | 938 | 1070 | 1485 | 784 | - |

Alliance orthologs: cele-dmel 12,083; cele-human 15,460; cele-mmus 17,436; cele-scer 4,474; dmel-mmus 16,356;
human-mmus 24,525; mmus-scer 5,730. Paralogs: mmus 245,631, cele 33,112 (Ensembl); mtub 3,295, calb 2,225,
ecol 1,788, bsub 1,713, spne 1,079, saur 919 (DIAMOND).

## Validation (`data/interim/orthology_extra/validation.json`)

RBH vs curated orthology. "Precision" is the share of RBH pairs that also appear in the curated set.
"Component recall" is the share of curated pairs that end up in the same component using only
sequence-derived edges (RBH + DIAMOND paralogs >= 30%).

| pair | RBH | curated | precision | curated genes with an RBH | component recall |
|---|---|---|---|---|---|
| human-scer | 2711 | 5804 | 0.958 | 0.556 | 0.799 |
| human-dmel | 8048 | 15834 | 0.918 | 0.595 | 0.810 |
| dmel-scer | 2332 | 4855 | 0.963 | 0.581 | 0.820 |
| human-spom | 2881 | 5666 | 0.922 | 0.579 | 0.743 |
| scer-spom | 3167 | 5345 | 0.991 | 0.690 | 0.778 |
| cele-human (Alliance) | 6662 | 15460 | 0.946 | 0.557 | 0.797 |
| human-mmus (Alliance) | 19410 | 24525 | 0.937 | 0.875 | 0.864 |

RBH is precise (92-99%) but on its own recovers about 80% of curated pairs at the component level. Among
eukaryotes the curated resources are kept on top. For bacteria, and for bacteria <-> eukaryotes, RBH is the
only source (see caveats).

eggNOG 5.0 LUCA-level orthologous groups (validation only,
http://eggnog5.embl.de/download/eggnog_5.0/per_tax_level/1/1_members.tsv.gz, sha256
1557a48dabf0d152300902a2f497488190c9503f8e8b20745f394e3523b52393). For cross-kingdom RBH pairs where both
proteins are in eggNOG, the share that are in the same LUCA OG: ecol-human 0.894, ecol-scer 0.885,
bsub-human 0.878, bsub-scer 0.895, mtub-human 0.860, mtub-scer 0.869. The rest are mostly pairs whose two
proteins sit in different but related OGs (eggNOG splits large superfamilies).

Spot checks (ortholog neighbours):
- human POLA1: scer POL1 (YNL102W), spom pol1, dmel, cele, mmus Pola1, calb, and ecol polB (b0060).
  E. coli Pol II is a B-family polymerase, so a POLA1 <-> polB RBH is plausible, not an error.
  polB also links to POLD1 / scer POL3 (YDL102W).
- spne rplB (uL2): ecol b3317, bsub, mtub, saur, human MRPL2, mmus Mrpl2, scer RML2 (YEL050C), spom, calb,
  dmel, cele. spne rplL: MRPL12 / scer MNP1 (YGL068W). spne rpsD (uS4): scer NAM9 (YNL137C), spom, calb, and
  all bacteria. There is no human hit because human mitochondrial uS4 is not detectable by sequence.
- spne rpoB: POLR2B / scer RPB2 (YOR151C). gyrB: TOP2B, parE (spne paralog). eno: ENO1-3 and the scer
  enolases. ffh: SRP54. tuf: TUFM. pyrG: CTPS1. metK: MAT2A.
- Bacteria-only, as expected: dnaA, polC, dnaE, ftsZ, secA (no eukaryotic hit at e <= 1e-5).

## Effect on SLB-1.2 families (`simulate_families.py`, current benchmark genes, dev/test not read)

The baseline check passes: `families.edges()` with the reused DSU/_cap reproduces held_out_families for
16,312/16,312 genes.

spne (530 benchmark genes). Base graph: no edges. With all extra edges:
- 397 have at least one edge.
- 82 share a component with another benchmark spne gene (spne paralogs).
- 382 are in a cross-species component (any node).
- 185 are in a component with a benchmark gene of another species (human/scer/spom).
- Partner species, counting components: bsub 355, saur 347, ecol 306, mtub 259, human 184, mmus 184,
  spom 177, scer 175, calb 174, dmel 173, cele 169.

Family sizes, in benchmark genes per family. Before: 7,214 families, max 194. After: 6,469 families, max 261,
with 3 families of 101-400 and none above 400. `_cap` is never triggered.

Bucket changes. families.assign names a family after its lexicographically smallest node, so any
component that gains an `ecol:`/`bsub:`/`calb:`/`cele:` node gets renamed and re-hashed, even when its
membership among benchmark genes is unchanged.

| scenario | genes changing bucket | test genes leaving test | test genes before -> after |
|---|---|---|---|
| all extra edges, naive naming (current rule) | 6,473 | 1,911 | 3,120 -> 3,488 |
| all extra edges, stable naming | 1,178 (human 576, scer 287, spom 195, spne 120) | 436 | 3,120 -> 2,985 |
| no base-pair RBH, stable naming | 1,159 | 425 | 3,120 -> 2,991 |
| spne edges only (spne-spne, spne-base), either naming | 156 (spne 106) | 55 (spne 30) | 3,120 -> 3,116 |

"Stable naming" names a family after its smallest node from a species already in SLB-1.2 (human, scer,
spom, dmel, spne). The recommendation is below.

Recommendations for the lead:
- Adopt a stable family-naming rule when integrating, e.g. min node over the SLB-1.2 species. Otherwise most
  families get re-bucketed for a purely cosmetic reason, and SLB-1.2 comparability is lost.
- The minimal fix for the known spne gap is the "spne only" scenario: `extra_edges(base_pairs=False,
  species={'human','scer','spom','dmel','spne'})`. It moves only 156 genes.
- The full set also merges base-species families through mouse, worm and bacterial bridges: 745 merges.
  For example, two human genes joined via a mouse paralog pair at >= 30%. This follows the same transitive
  logic families.py already applies with dmel as a bridge species, and it is the conservative choice.

## Caveats

- Ortholog recall for bacteria is limited by RBH: about 80% component-level recall against curated
  eukaryotic orthology. A one-directional best-hit tier (e <= 1e-10, coverage >= 50%) was tested: it adds
  251k edges and creates a 920-gene component, above MAX_FAMILY, and `_cap` never splits ortholog edges. It
  was therefore rejected. Some real distant orthologs (e.g. uL10 rplJ <-> MRPL10) have no hit at e <= 1e-5
  and stay unlinked.
- RBH co-orthology ties at 95% bitscore can link a bacterial gene to several eukaryotic paralogs (e.g. ENO1/2/3),
  as intended for leakage.
- DIAMOND paralogs use --max-target-seqs 250 per proteome. That is fine for bacteria and calb (the only
  species where they are used), but not for human, where huge families are truncated. This is why
  diamond_self is not used for base species.
- calb: only haplotype-A proteins are used, and the B allele IDs resolve to the same canonical ID.
  data-eukaryotes will use calb only if a usable dataset appears.
- saur strain: NCTC 8325 (SAOUHSC_) was assumed. No agent has claimed a saur source yet. A USA300 or JE2
  (SAUSA300_) dataset would need that strain's GenBank and an extra BACTERIA entry in ids_extra.
- Newly added spne genes from other sources (crisprtnseq2024, dualtnseq2025) are covered, since every D39V
  protein is a node, provided their IDs use the ids_extra spne scheme (see the ID note above).
- The WormBase download server is behind Cloudflare, so the EBI mirror (WS298) is used.

## Files

`notes/data/orthology.fetch.tsv` lists filename<TAB>url for every download, and
`data/raw/orthology_extra/SHA256SUMS` holds their sha256. Key hashes:
- human.pep 9b43da92...7668
- mmus.pep 480d4a6e...a37
- cele.pep 78fb77ec...e8c
- dmel.pep a2175de6...7ee
- scer.pep 67ae76c7...5c
- spom peptide c72f4b0b...65
- calb protein 27958a98...5ab
- CP027540.1 cd83dc05...fa
- U00096.3 52285403...8d
- AL009126.3 42ef1564...e3
- AL123456.3 637bb2a5...7e
- CP000253.1 b6d8f2a9...57
- MRK_List2 12301e73...d
- WS298 geneIDs d87c69d1...46
- CGD features 7cc31d01...c
- BioMart mmus 84a04c1c...4e
- BioMart cele 05dede52...51
- eggNOG 1557a48d...93
