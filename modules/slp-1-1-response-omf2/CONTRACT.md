# SLp response model OMF 2.0 script contract

`train.py` fits independent K562 and RPE1 reduced-rank response maps from raw
static577 intervention descriptors to the supplied native measured panels. A
rank of zero selects the full numerically supported feature rank. The output is
a portable directory containing both models, inference source, numerical core,
and a manifest.

`evaluate.py` predicts the supplied development genes by adding each model's
residual response to the supplied GEM-weighted basal anchor. It reports the same
gene-profile MSE and independently query-centered residual correlation used by
the retained development comparison. Inputs must be prepared without protected
test outcomes.

This vertical slice manufactures the retained panel-specific feature-linear
baseline under OMF 2.0. It is not the proposed observation-encoded world model,
does not infer unmeasured queries, and does not identify molecular dynamics.

The dependency lock targets Linux x86_64 on CPython 3.11 or 3.12. Its NumPy
2.2.6 hashes are restricted to the PyPI-published
`cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64` and
`cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64` wheels, respectively.
Other operating systems, architectures, Python implementations, and Python
versions are outside this module contract.
