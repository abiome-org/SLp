# Yeast Seurat streaming contract

This numerical module inventories gzip RDX3/XDR serializations with pinned
`rdata==1.1.0` type codes. It retains small structure, symbols, S4 slots and
selected atomic ranges. Large atomic vectors are consumed in bounded chunks,
checksummed and discarded. A serialization reference points at the original
lightweight node and never causes a payload copy.

Only an S4 `dgCMatrix` reached through the RNA assay's `counts` slot is eligible
for later molecular processing. SCT, normalized `data`, and `scale.data` are
excluded. The module does not infer an assay from a truncated prefix.

Offsets refer to the uncompressed RDX3 byte stream. Gzip is replayed from byte
zero for pass two. CSC column selections are translated through the complete
`p` vector into ranges shared by `i` and `x`. No selected value range may be
chosen from an expression effect; eligibility must already follow from stable
intervention metadata, the global protected-gene partition, or control status.

The current 2 MiB source prefix ends within the first matrix's 181,083,366-entry
`i` payload. It cannot establish dimensions, dimnames, cell metadata, assay
identity, payload integrity, or a usable matrix. Full-file extraction remains
unexecuted and must first inventory these contracts.

Reference-bearing environments and external pointers preserve their reference
positions and lightweight structure. Unsupported weak references and bytecode
are rejected. This is deliberate: silently skipping one would shift the R
reference table and could corrupt every later symbol or slot interpretation.
