# SBSL models (r/utils/train-model.R of joanagoncalveslab/SBSL) retrained on SLB train, scoring an SLB split.
#   Elastic Net: caret::train(method = "glmnet", 10-fold CV, metric ROC, grid alpha = seq(0,1,len 20) x
#                lambda = seq(1e-4, 1, len 50))                                   -> sbsl__en
#   Random forest: caret::train(method = "rf", 10-fold CV, metric ROC, mtry 4:8)   -> sbsl__rf
# Preprocessing as in "1.1 Experiment.R": na.omit on training rows, class balancing by undersampling
# (train.balance_cancers, seed 124; SLB: per context, n_c = min(#pos_c, #neg_c), because SBSL's global
# n = min over all cancer x class cells would be 0 for SLB contexts without positives),
# caret::preProcess(center, scale, nzv). SLB addition: NA features of scored rows are imputed with the training
# medians (SBSL simply had no NA rows after na.omit).
# usage: Rscript train.R <train.csv> <test.csv> <out_prefix>
suppressPackageStartupMessages(library(caret))
args <- commandArgs(trailingOnly = TRUE)
tr <- read.csv(args[1], check.names = FALSE)
te <- read.csv(args[2], check.names = FALSE)
out <- args[3]

feat <- setdiff(colnames(tr), c("example_id", "context_id", "SL"))
tr <- na.omit(tr)
set.seed(124)
parts <- lapply(split(tr, tr$context_id), function(d) {
  n <- min(sum(d$SL == 1), sum(d$SL == 0))
  if (n == 0) return(NULL)
  rbind(d[sample(which(d$SL == 1), n), ], d[sample(which(d$SL == 0), n), ])
})
tr <- do.call(rbind, parts)
tr$SL <- factor(ifelse(tr$SL == 1, "Y", "N"), levels = c("N", "Y"))
cat(sprintf("balanced train: %d rows (%d pos), %d features\n", nrow(tr), sum(tr$SL == "Y"), length(feat)))

med <- sapply(tr[feat], median)
for (f in feat) te[[f]][is.na(te[[f]])] <- med[[f]]
pp <- preProcess(tr[feat], method = c("center", "scale", "nzv"))
Xtr <- predict(pp, tr[feat]); Xte <- predict(pp, te[feat])
cat("features kept after nzv:", ncol(Xtr), "\n")
dtr <- cbind(SL = tr$SL, Xtr)
f <- as.formula(paste("SL ~", paste(sprintf("`%s`", colnames(Xtr)), collapse = " + ")))

ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)
set.seed(124)
en <- train(f, data = dtr, method = "glmnet", metric = "ROC", trControl = ctrl,
            tuneGrid = expand.grid(alpha = seq(0, 1, length = 20), lambda = seq(0.0001, 1, length = 50)))
cat("EN best:", paste(names(en$bestTune), en$bestTune, collapse = " "), " CV ROC", max(en$results$ROC), "\n")
print(coef(en$finalModel, en$bestTune$lambda))
write.csv(data.frame(example_id = te$example_id, score = predict(en, Xte, type = "prob")[, "Y"]),
          paste0(out, "_en.csv"), row.names = FALSE)

set.seed(124)
rf <- train(f, data = dtr, method = "rf", metric = "ROC", trControl = ctrl, tuneGrid = expand.grid(mtry = 4:8))
cat("RF best mtry:", rf$bestTune$mtry, " CV ROC", max(rf$results$ROC), "\n")
write.csv(data.frame(example_id = te$example_id, score = predict(rf, Xte, type = "prob")[, "Y"]),
          paste0(out, "_rf.csv"), row.names = FALSE)
