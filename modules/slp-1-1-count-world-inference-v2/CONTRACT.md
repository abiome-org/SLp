# Count world inference v2

This self-contained adapter loads one frozen shared count-world arm and one
registered native measurement panel. It accepts raw static intervention
features and nonnegative weights over the saved control contexts, and returns
the prior expected CP10k mixture plus its `ln1p` transform.

Library size and perturbed counts are excluded from the forecast API. Native
query axes and control panels remain separate. Action normalization uses only
the persisted shared fitting-action statistics. Empty actions return the
corresponding weighted basal control mean within numerical precision.

Version 2 owns a private copy of caller-supplied context weights before
normalizing them. Prediction therefore cannot mutate an input array. Model,
reference, protocol, and embedded numerical-source checksums are validated
before inference.
