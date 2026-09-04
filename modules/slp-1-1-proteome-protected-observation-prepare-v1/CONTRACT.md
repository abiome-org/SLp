# Contract

Each run binds the same six exact source, identity, relation, roster, and SGD
inputs as the frozen pretraining preparation, and requires one exact role in
`config.role`. It emits one role-specific `observation-corpus.tar` and one
audit; it never emits both roles or a basal profile.

The production validation contract is 537 records, 529 stable intervention
genes, 1,850 readouts, 967,019 observed values, and 26,431 missing values. The
production final contract is 275 records, 268 genes, 1,850 readouts, 496,490
observed values, and 12,260 missing values. Role-specific trajectory, action
sequence, and raw-locator sequence digests are immutable constants.

All selected observations use finite, strictly positive MaxLFQ relative
intensities transformed with log2 and no pseudocount. Literal `NA` is omitted
from CSR. Every non-selected quantitative class is parsed only as part of the
CSV container and is not converted or semantically validated. The module has
no fitting or reward operation.
