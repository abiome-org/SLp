# Dual-guide composition v1

This module maps exact 20-nt A/C/G/T dual-guide sequences to a fixed
71-dimensional descriptor. It contains no guide ID, genomic coordinate,
outcome, inferred efficacy, or target-gene embedding.

Each guide contributes 32 reverse-complement-canonical overlapping 3-mer
frequencies. The pair descriptor concatenates their mean and absolute
difference, GC mean and absolute difference, homopolymer minimum and maximum,
base-entropy mean and absolute difference, and the minimum normalized Hamming
distance between `(A,B)` and `(A,reverse_complement(B))`. It is invariant to
guide order and to independently reverse-complementing either guide.

`aggregate_gene_descriptors` averages exact pair descriptors using supplied
fitting-cell frequencies. Every cell pair and action must join exactly;
unsupported pairs or genes raise an error rather than receiving zero features.
