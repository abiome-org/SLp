# Fixed Query Transition v1

This application-neutral module predicts queried molecular deltas from static
intervention features and supplied basal measurements. The output query
coordinates are fixed numerical inputs and are not learned gene identifiers.

The module does not fit a response basis, load datasets or define splits. Empty
action sets return the supplied control mean exactly. A fitted basis must carry
its own quantitative-data provenance and cannot be described as a static prior.
