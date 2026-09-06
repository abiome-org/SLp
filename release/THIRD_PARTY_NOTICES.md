# Data and third-party notices

Original SLp source code, documentation and trained SLp weights are distributed
under the MIT license. This does not relicense third-party observations,
annotations, benchmark data, software or pretrained feature models. Existing
third-party LICENSE files continue to apply.

The SLp-1.1 data release contains transformed, filtered research data. Its
`inventory.json` records exact files, sizes, hashes and roles. Historical input
manifests are retained byte-for-byte, including historical path strings; loaders
use the repository-relative paths described in the development guide.

| Source | Released material and attribution | Terms |
|---|---|---|
| Replogle et al. (2022), [Figshare 20029387 v1](https://doi.org/10.25452/figshare.plus.20029387.v1) | K562/RPE1 essential and K562 genome-wide CRISPRi counts, filtered populations and fitting indices | CC BY 4.0 |
| Frangieh et al. (2021), scPerturb [Zenodo 10044268](https://doi.org/10.5281/zenodo.10044268) | Processed public paired RNA/protein data, sparse training shards and population summaries; attribution also to the scPerturb authors | CC BY 4.0; controlled raw sequencing is excluded |
| Nadal-Ribelles et al. (2025), [Zenodo 14062629](https://doi.org/10.5281/zenodo.14062629) | Yeast Control/NaCl count-derived population means, metadata and reference panels from author-released Seurat objects | CC BY 4.0 |
| Broad DepMap, [24Q2](https://doi.org/10.25452/figshare.plus.25880521.v1) | Filtered human CRISPR gene effects, observed cell contexts and fitting/development partitions | CC BY 4.0 |
| Costanzo et al. (2016), [Dryad 4291s](https://doi.org/10.5061/dryad.4291s) | Species-native yeast single/double-mutant fitness, sampled and partitioned from the authors' raw pairwise data | CC0 1.0 |
| Norman et al. (2019), [GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344) | Processed molecular single/combination population responses | NCBI GEO public-data policy; submitter rights retained |
| Nadig et al. (2025), [GSE264667](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667) | Processed HepG2 molecular responses | NCBI GEO public-data policy; submitter rights retained |
| Zhao et al. (2021), [GSE164996](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164996) | MCF10A processed molecular population responses in three environments | NCBI GEO public-data policy; submitter rights retained |
| [Saccharomyces Genome Database](https://www.yeastgenome.org/) and [Gene Ontology](https://geneontology.org/docs/go-citation-policy/) | Stable identities, protein-sequence and shared MF/CC annotation-derived features | CC BY 4.0; source snapshots recorded in `rights/` and feature receipts |
| [Ensembl](https://www.ensembl.org/info/about/legal/disclaimer.html), [STRING](https://string-db.org/cgi/access.pl), [ESM](https://github.com/facebookresearch/esm) | Human identity/sequence, physical-network features and frozen ESM2-8M protein representations | Ensembl source terms, STRING CC BY 4.0, ESM MIT; original feature-model weights are not redistributed here |
| [MuSL](https://github.com/JieZheng-ShanghaiTech/MuSL) | Selected official CV folds and the associated gene roster; Jie Zheng (2022) | MIT; upstream LICENSE included with the selected files |

SLp adaptations include gene filtering, population aggregation, log1p(CP10K)
normalization where recorded, sparse/memory-mapped conversion, descriptor
projection, species-native partitioning, and frozen-model simulation features.
Derived files that combine sources retain all applicable component terms.
Original model predictions and fitted SLp decoder weights are MIT; observed
contexts and cached input descriptors accompanying them retain source terms.

For the three GEO sources, the redistribution basis is the
[GEO public-data policy](https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html)
and the authors' public processed-data deposits. NCBI imposes no distribution
restriction but does not transfer submitters' rights. These files are therefore
identified with that policy, not represented as CC BY or MIT. The original
conditional rights receipts remain intact. No additional source-specific
restriction was identified in the recorded public-data scopes.

The release excludes controlled Frangieh sequencing, patient-level TCGA data,
downloaded third-party model weights, credentials, OMF runtime records and the
separate SLAMR benchmark inputs whose redistribution terms are not established.
SLAMR aggregate evaluation statistics remain reported in the model card.

Historical input indices cite a yeast summary-only receipt. The accompanying
`rights/zenodo-14062629-cc-by-4.0.yaml` and
`rights/slp-1-1-nadal-ribelles-yeast-seus-split-cc-by-4.0.yaml` supply the
appropriate public Seurat-source scope for the count-derived fitting arrays;
the original receipt and index are not silently rewritten.
