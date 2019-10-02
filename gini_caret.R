library(ggplot)
library(MLmetrics)
library(caret)
library(doMC)
set.seed(346)
dat <- twoClassSim(10000)

giniMetric <- function(data, lev = NULL, model = NULL){
  gini <- MLmetrics:::Gini(y_pred = data[[lev[2]]], 
                           y_true = unclass(data$obs))
  c(Gini = gini)
}

registerDoMC(cores=8)
mod <- train(Class ~ ., data = dat,
             method = "rpart",
             tuneLength = 5,
             metric = "Gini",
             trControl = trainControl(summaryFunction = giniMetric,
                                      classProbs = TRUE))