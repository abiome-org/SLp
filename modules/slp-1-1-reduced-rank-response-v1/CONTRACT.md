# Reduced-rank response model v1

This module fits a rank-constrained, regularized linear map from raw static
intervention features to a measured molecular response panel. The intercept is
unpenalized. Feature normalization and the response basis are fitted only from
the supplied fitting records.

The returned state has no learned intervention-ID vocabulary. Its query
loadings are quantitative, panel-specific fitted descriptors. They do not
define unmeasured-query inference, static biological priors, a cell generator,
or identified molecular dynamics.
