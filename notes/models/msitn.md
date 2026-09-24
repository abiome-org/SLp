# MSITN (2026)

[Publication](https://doi.org/10.1016/j.eswa.2026.132323) points to
[this source repository](https://github.com/dlmu-qxl/MSITN-main), acquired
at `external/models/msitn`. The repository README and code identify the
implementation as **IMSI**, rather than MSITN. It uses TensorFlow 1.13-era
SynLethKG triple files, with SL relations in the graph. There is no
verified MSITN checkpoint, SLB gene map, or script to remove SL-bearing
edges and refit on SLB train. Therefore a number attributed to MSITN would
be unsupported; no leaderboard row is created.
