# Reduced-rank response local research inference v1

The caller selects one saved native source and supplies weights over that
source's exact GEM control contexts. The adapter mixes positive control rates
in CP10k units, applies `log1p`, and adds the fitted signed response residual
without clipping or renormalization.

The primary numerical API accepts raw static features. A convenience method
looks up stable ENSG IDs in a frozen static-action cache; gene identity is not
a model parameter. Native query IDs and order are returned with every result.

This bundle predicts panel-specific molecular profiles. It is not a count
generator, a new-context model, an OMF release, or evidence for unmeasured
query transfer.
