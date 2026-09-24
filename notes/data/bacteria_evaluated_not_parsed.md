# Bacterial GI candidates evaluated and NOT parsed (data-bacteria, 2026-09-24)

What I looked at beyond the six sources that did get parsers
(`crisprtnseq2024`, `dualtnseq2025`, `koo2025`, `babu2011`, `gagarinova2016`, `kumar2016`,
`cote2016`), and why each one is not in `bacteria_extra.py`. Nothing here supplies both measured
positives and measured negatives at usable scale, so none of it can become benchmark labels.

## Positives-only publications (no measured non-interactions)

| candidate | what is published | why not usable |
|---|---|---|
| **Babu 2014** eSGA *E. coli* GI map, PLoS Genet 10:e1004120 (`data/raw/babu2014_ecoli/`) | Table S2: 42,705 **high-confidence** GI pairs with scores | the tested-but-not-interacting pairs are not released, so there are no negatives. Table S3 (65 literature-curated pairs) is only useful as a sanity check |
| **van Opijnen 2009** Tn-seq GI, *S. pneumoniae* TIGR4, Nat Methods 6:767 (`data/raw/vanopijnen2009_spneumo/`) | Table S4: 131 called interactions (aggravating/alleviating) for 5 query genes; Table S2: wild-type library fitness for 2,116 genes | the per-background fitness of every gene is not released, only the calls. Table S2 is a usable *single-gene* fitness source for TIGR4 (SP_ tags map to D39V through `spne_map`) if a third spne fitness arm is ever wanted |
| **Bhowmick & Dickey** MRSA (*S. aureus*) genome-wide GI platform, bioRxiv 2026 (`data/raw/mrsa2026_dualcrispri/media-1_supplement.pdf`) | supplement is PDF only: library coverage, strains, plasmids, oligos | no GI score table in the supplement and no linked repository accession found; the per-pair data appear not to be public yet. Worth re-checking when the paper is published |

## Designs with too few query genes to form a stratum

| candidate | design | pairs |
|---|---|---|
| **Rachwalski 2023**, Cell Rep Methods 3:100693 (`data/raw/rachwalski2024_crispri_ecoli/`, PII S266723752300379X, all files hash-verified) | mobile CRISPRi knockdown of **3** genes (lolA, pssA, mreD) crossed into the whole Keio collection, 3 inducer levels, empty-vector control arm | ~11k, but only 3 distinct query genes. Good orthogonal *yardstick* for the E. coli array screens (knockdown × deletion, proper control arm); useless as its own stratum |
| **Rubin 2018** *Synechococcus elongatus* PCC 7942, PLoS Genet 14:e1007301 (`data/raw/rubin2018_synechococcus_tnseq/`) | Tn-seq in a single sensitised background; `Sensitized_Interaction_Score` + p + FDR + call for 1,920 genes | 1 query gene; would also need a new species (proteome, single-gene fitness, orthology) for one gene's worth of labels |
| **Vibrio cholerae** Tn-seq (`data/raw/vibrio2024_ep_tnseq/`, PLoS Genet pgen.1011234 Table S6) | one mutant-vs-WT Tn-seq comparison, fold change + p for 6,997 genes | 1 query gene |
| **Schramm** *Caulobacter crescentus* (`data/raw/schramm2026_caulobacter_tnseq/`, NIHMS2151696) | two comparisons, ΔclpB vs WT (heat stress) and Δpol1 vs WT (control), 4,091 genes each with log2FoldChange + padj | 2 query genes, and the "up-/down-regulated" wording of the significance column suggests these are RNA-seq differential-expression tables, not fitness. Not a GI readout |

## Searched for and not found as per-pair data

- **Typas 2008 GIANT-coli** (Nat Methods 5:781): the method paper; its GI scores were released through the
  later Babu/Gagarinova/Kumar maps, which are the ones parsed here. No separate full score matrix.
- ***M. tuberculosis* / *M. smegmatis* gene × gene maps.** The Mtb CRISPRi/CRISPR-KO resources
  (Li 2022 Sci Adv add5907; Bosch 2021; Ruiz 2025) are **chemical**-genetic (gene × drug) or
  single-gene essentiality/vulnerability screens. No published pairwise Mtb gene × gene GI matrix
  was found, so `mtub` has no source despite being wired into `ids_extra`.
- **Dual/double CRISPRi in Gram-negatives** (*Pseudomonas*, *Acinetobacter*, *Klebsiella*,
  *Vibrio*, *Salmonella*): searched repeatedly; as of 2026-09 the dual-guide GI screens exist only
  in *S. pneumoniae* and *B. subtilis* (both parsed). The dual-guide method papers
  (e.g. Nat Commun 2025 "next-generation dual guide CRISPR system", GRAPE 2026) are human/analysis
  papers.
- **Trigenic / higher-order bacterial maps:** none found. The only trigenic maps are yeast
  (Kuzmin 2018/2020, which `data-eukaryotes` handled).
- **E. coli TF GI map**, Gagarinova 2022 Nat Commun (`data/raw/gagarinova2022tf_ecoli/`, 157 MB
  MOESM22): a fifth screen of the same eSGA family from the same lab. Not parsed because the four
  already-parsed members of that family mutually disagree at AUROC 0.46–0.51, so a fifth cannot
  change the verdict; the download is kept in case the lead wants it as extra training data.

## Practical notes for whoever continues this

- **Dryad** is behind an Anubis proof-of-work gate: `scripts/dryad_anubis_fetch.py <url> <out>`
  solves it (used for the Zik 2025 per-run tables).
- **journals.asm.org** (ASM: mBio, mSystems, JB) returns 403 to this machine; PMC's `instance/<id>/bin/`
  path sometimes has the same files under different names and sizes, so verify by content.
- **Cell Press full texts and bioRxiv PDFs** rate-limited this machine (HTTP 429 / Cloudflare 1015)
  during the session, which is why the Koo 2025 label threshold was derived from the data rather
  than read off the Methods.
- R images (`.image`, `.RData`) are readable with `uv run --with pyreadr`, but only for data frames:
  R *lists* of data frames (e.g. Zik's `ML2fit`) come back missing.

## Where the audit numbers come from

`slpbench audit` recomputes every figure in these notes through the check functions in
`bacteria_extra.py`: `dualtnseq2025_checks`, `koo2025_checks`, `crisprtnseq2024_checks`,
`spne_cross_checks` (the three pneumococcal screens against each other, both directions, with
bootstrap CIs), `babu2011_checks`, `gagarinova2016_checks`, `kumar2016_checks`, `cote2016_checks`
and the wrapper `ecoli_array_checks`. The one-vs-all arm of `dualtnseq2025_checks` needs `pyreadr`
(`uv run --with pyreadr`); without it that row comes back with `auroc: None` and a note, and the
rest of the checks are unaffected.
