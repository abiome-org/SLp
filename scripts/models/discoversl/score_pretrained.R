# Apply the released DiscoverSL random forest (regression, 700 trees, 4 p-value features; trained on
# published SL / non-SL pairs => LEAKY) to a feature table. Column order exactly as in predictSL():
# PValue (DiffExp), Mutex, correlation.pvalue, PvalPathway.
suppressMessages(library(randomForest))
args <- commandArgs(TRUE)
m <- readRDS("/d/export/model.rds")
x <- read.csv(args[1])
ok <- complete.cases(x[, c("PValue", "Mutex", "correlation.pvalue", "PvalPathway")])
x$score <- NA
X <- matrix(c(x$PValue[ok], x$Mutex[ok], x$correlation.pvalue[ok], x$PvalPathway[ok]), ncol = 4)
x$score[ok] <- predict(m, newdata = X, type = "response")
write.csv(x[, c("key", "g1", "g2", "score")], args[2], row.names = FALSE)
