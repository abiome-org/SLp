# misl: MiSL (Sinha et al. 2017, Nat Commun 8:15580)

battery: none (not scorable at SLB scale); species: human; needs: MiSL code + its preprocessed TCGA tables (Stanford Digital Repository), docker slb/r-stats

- Paper: Sinha S, ..., Dill DL. Systematic discovery of mutation-specific synthetic lethals by mining pan-cancer human primary tumor data. Nat Commun 8:15580 (2017). doi:10.1038/ncomms15580
- Code + data: https://purl.stanford.edu/ny450yx7231 (SL_code.final.tar.gz, 1.64 GB, CC BY-NC-SA 3.0; data/raw/misl, sha256 in SOURCES.tsv), unpacked to external/models/misl/SL_code.final. R scripts, no weights (unsupervised, TCGA mutation/CN/expression of 12 cancer types).
- MiSL is mutation-centric: for ONE recurrent mutation (e.g. KRAS_M) in ONE cancer type it returns a list of candidate SL partners (partners whose deletion is mutually exclusive with the mutation and whose expression/CN compensate). SLB pairs are mostly paralogs; the pair (A,B) can only be scored when A carries a MiSL-targetable recurrent mutation (>= 4 samples and > 2.5% in a cancer type).
- Status: acquired / env built (Docker R 4.3) / ran original (demo + batch driver src/slb_batch.R; KRAS_M in LUAD produced candidates; AP2B1_M in BLCA fails inside compute_diff_expr: "not enough 'x' observations") / NOT dev-scored.
- Blocker: dev genes carry 2,129 (mutation, cancer) combinations that pass the recurrence filter; each MiSL run takes minutes after a ~2 min table load, and many low-frequency mutations fail in the differential-expression step. Estimated > 80 CPU-hours at the 8-thread cap for a sparse, mutation-gated score that would cover few SLB pairs. Not run; its tumour co-alteration logic is covered by the statsl SoF statistics (which carry no signal on SLB dev: sof_cna 0.49, sof_expr 0.45 human).
- Leakage: none (no SL labels).
