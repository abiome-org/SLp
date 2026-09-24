# Shared per-species feature bundle (owner: models-mechanistic)

Builder: `uv run python scripts/models/_common/bundle.py [--species ...] [--parts go ppi fitness]`
Loader helpers: `scripts/models/_common/bundle_io.py` (pandas only; `go_direct`, `go_propagated`, `go_dag`,
`ppi`, `fitness`, `load_split`, `load_train`, `out_path`). Thread caps: `source scripts/models/_common/threads.sh`.

Output: data/interim/bundle/
| file | columns | content |
|---|---|---|
| `<sp>/go.parquet` | gene, term, aspect, evidence | direct GO annotations; IGI, ND and NOT rows dropped |
| `<sp>/ppi.parquet` | gene_a < gene_b, biogrid_phys, string_{neighborhood,fusion,cooccurence,coexpression,database} | BioGRID 5.0.261 physical rows only (count of evidence rows); STRING v12 channels WITHOUT experimental (contains genetic assays) and textmining; max over protein pairs mapping to the same gene pair |
| `<sp>/fitness.parquet` | gene, effect, source | reference single-loss effect, negative = sicker: slpbench.fitness.gene_effects() for SLB-1.2 species (+dmel), else `<sp>_single()` from slpbench.sources.{eukaryotes,bacteria}_extra |
| `<sp>/esm2.parquet` | gene, e0..e1279 | ESM-2 650M mean embeddings, produced by models-features-fm in data/interim/esm2_650m/<sp>.parquet and linked here |
| `_go/terms.parquet`, `_go/edges.parquet` | term, namespace, name / child, parent | go-basic, is_a + part_of |
| `manifest.json` | | input files with sha256, per-species counts |

Gene IDs: canonical SLB IDs from `slpbench.ids_extra.resolver(species)` (ids.py for the SLB-1.2 species;
spne also accepts D39 SPD_ / R6 spr / TIGR4 SP_ tags via bacteria_extra.spne_resolver).

| species | GO source | STRING taxon | BioGRID organism | fitness source |
|---|---|---|---|---|
| human | goa_human.gaf | 9606 | Homo_sapiens | DepMap 24Q4 pan-line (slpbench.fitness) |
| scer | sgd.gaf | 4932 | S. cerevisiae S288c | SGA single-mutant fitness |
| spom | pombase.gaf | 284812 | S. pombe 972h | PomBase viability |
| spne | GOA proteome D39 (25749) | 171101 (R6) | ATCC BAA-255 (R6; 6 edges) | dual CRISPRi-seq single sgRNA |
| cele | wb.gaf | 6239 | C. elegans | WormBase phenotypes (cele_single) |
| dmel | fb.gaf | 7227 | D. melanogaster | Heigwer 2023 main effects (slpbench.fitness) |
| mmus | mgi.gaf | 10090 | M. musculus | DepMap of 1:1 human ortholog (mmus_single) |
| ecol | GOA proteome MG1655 (18) | 511145 | E. coli K12 MG1655 | Keio OD600 (ecol_single) |
| bsub | GOA proteome 168 (6) | 224308 | B. subtilis 168 (6 edges) | Koo 2025 marginal CRISPRi (bsub_single) |

Coverage of each species' proteome genes (data/interim/orthology_extra/fasta): GO 59% (cele) to 97%;
PPI 87-99%; fitness 25% (spne, bsub) to 96%.
Raw inputs: data/raw/go/, data/raw/string_v12/, data/raw/biogrid/ (sha256 in manifest.json).
Adding a species: one entry in `SPECIES` in bundle.py (GAF path(s), STRING taxon, BioGRID organism name) and
a resolver in slpbench.ids_extra.

Coverage of SLB-1.3 train+dev genes (go / fitness / esm2 / ppi): human 1.00/0.98/1.00/1.00; scer 0.91/0.91/1.00/0.99;
spom 0.99/0.99/0.99/0.99; bsub 0.98/0.99/0.99/1.00; cele 0.84/0.98/0.98/0.98; dmel 1.00/0.59/1.00/1.00;
mmus 1.00 for all. ESM-2 for cele/dmel/mmus currently covers only benchmark genes (703/93/126 genes); ecol has
no esm2 yet. Re-link after new embeddings: `uv run python scripts/models/_common/bundle.py --parts esm2`.
