# GiGCN (2026 binary SLB adaptation)

**Sources:** [Wang et al., *Briefings in Bioinformatics* 27:bbag470](https://doi.org/10.1093/bib/bbag470),
[official code and data](https://github.com/wqz2469/GIGCN) at commit
`fdd7dd055a684a8fac4d6715b8ad8c449d21bf56`.

The published model jointly predicts synthetic-lethal (SL), synthetic-viable
(SV), and neutral signed interactions. The repository includes 8,887 gene
symbols and its original signed graph, but no weights or the GO similarity
matrix used by its feature pipeline. Its trainer imports a missing `utils.py`
and its supplied edge splitter uses random edge folds. Those original graph
edges and external interaction labels are **not** used by the SLB adapter.

`scripts/models/gigcn/run.sh` implements the released DINES architecture:
64-dimensional gene input, eight factors, two signed graph convolution
layers, the source's factor discriminator, and the factor-correlation pair
decoder. The source's supported mean aggregator is used. Because SLB has
only measured SL and neutral outcomes, its SV graph channel is empty and
the three-class decoder becomes binary. Pair directions are averaged, and
positive examples are class-weighted to handle SLB's measured-pair ratio.
Train graph edges are added in both directions for message passing.
Human GO annotations are taken from the benchmark bundle (IGI and ND
evidence excluded), propagated through GO, TF-IDF weighted, and compressed
to 64 dimensions by SVD. This replaces the unreleased GO similarity/PCA
matrix; it is an architecture adaptation, not exact source reproduction.

The unsupervised GO reduction and factor-discrimination training include
gene nodes named in the public dev and test inputs, without using their pair
labels. This is a transductive use of single-gene annotations allowed by
SLB's contract; the family holdout applies to pair-label training rather
than hiding the gene identities or all unlabeled biological data.

Only SLB train pairs supply graph messages or supervised labels. An internal
15% gene holdout selects training length without using dev or test labels.
The final fit uses all human train pairs and the chosen 18 epochs. All 20,210
human test pairs are in the GO feature universe; unscored species tie at
0.500. The method is context-blind, as in the source: 900 distinct human
training pairs have conflicting labels across contexts, which limits this
adapter on a context-specific benchmark. Cache metadata hashes the benchmark,
GO features, source code, configuration and checkpoint.

| SLB-1.3 split | SLB score | Human component | Native human coverage |
|---|---:|---:|---:|
| Dev | 0.520 | 0.559 | 19,149 / 19,149 |
| Test | 0.523 (95% CI 0.492–0.549) | 0.569 | 20,210 / 20,210 |
