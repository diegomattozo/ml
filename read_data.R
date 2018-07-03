library(tidyverse)
library(ggplot2)
library(ggthemes)
library(Boruta)

set.seed(1)

appl_train <- read_csv("data/application_train.csv")

TO_REMOVE <- c("name_type_suite", "name_family_status", "region_population_relative",
               "days_birth", "days_employed", "days_registration",
               "days_id_publish", "own_car_age", "occupation_type", "cnt_fam_members",
               "weekday_appr_process_start", "hour_appr_process_start", "organization_type",
               "fondkapremont_mode", "housetype_mode", "wallsmaterial_mode", "emergencystate_mode",
               "name_contract_type", "name_income_type")

CATEG_VARS <- c("name_education_type", "name_housing_type", "region_rating_client", 
                "region_rating_client_w_city", "code_gender", "flag_own_car", "flag_own_realty")

# convert colnames to lowercase
names(appl_train) <- tolower(names(appl_train))

# data partition
trainIndex <- caret::createDataPartition(appl_train$target, p=.8, 
                                         list=F, times = 1)

train.data <- appl_train[trainIndex, ]
valid.data <- appl_train[-trainIndex, ]

col.idx <- which(colnames(train.data) %in% TO_REMOVE)

train.data <- train.data[, -col.idx]
valid.data <- valid.data[, -col.idx]

# collect and remove train data
appl_train <- NULL
gc()

# get colnames with missing values
missing_colnames <- colnames(train.data)[colSums(is.na(train.data)) > 0]
missing_colnames

train.data[is.na(train.data)] <- -99

# converting to factor
for (col in CATEG_VARS) {
    train.data[, col] <- is.factor(train.data[, col])
}

boruta.train <- Boruta(target~., data = train.data, doTrace = 2)

plot(boruta.train, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i)
    boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)
