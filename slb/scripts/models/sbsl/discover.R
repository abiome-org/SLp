# DISCOVER (Canisius et al. 2016) background model for SBSL's discover_mutex feature.
# Fits discover.matrix() on the event matrix (strata = TCGA type when several types are present) and writes the
# per-gene x per-sample background event probabilities; the pairwise mutual-exclusivity p-values
# (pairwise.discover.test, alternative "less": P(overlap <= observed) under the Poisson-binomial null with
# per-sample probabilities bg[g1, j] * bg[g2, j]) are then computed exactly in Python (discover_run.py), because
# calling pairwise.discover.test once per SLB pair runs at ~1 pair/s (114k training keys).
# usage: Rscript discover.R <events_long.csv: sample,gene> <samples.csv: sample,stratum> <out_prefix>
suppressPackageStartupMessages(library(discover))
a <- commandArgs(trailingOnly = TRUE)
ev <- read.csv(a[1]); sm <- read.csv(a[2])
genes <- sort(unique(ev$gene))
m <- matrix(0L, length(genes), nrow(sm), dimnames = list(genes, sm$sample))
m[cbind(match(ev$gene, genes), match(ev$sample, sm$sample))] <- 1L
dm <- if (length(unique(sm$stratum)) > 1) discover.matrix(m, strata = sm$stratum) else discover.matrix(m)
bg <- dm$bg[genes, sm$sample, drop = FALSE]
writeLines(genes, paste0(a[3], "_genes.txt"))
con <- file(paste0(a[3], "_bg.f64"), "wb"); writeBin(as.vector(t(bg)), con, size = 8); close(con)
