library(fasttime, quietly = T, warn.conflicts = F)
library(lubridate, quietly = T, warn.conflicts = F)
library(dplyr)
library(magrittr)
library(glmnet)
library(parallelMap)

# Reading tables
bike = read.csv("../data/train.csv")
test = read.csv("../data/test.csv")
dateTest = test$datetime

# Data prep
bike$log_count = log1p(bike$count)


bike %>% 
    mutate(datetime = fastPOSIXct(datetime, "GMT")) %>% 
    mutate(hour = hour(datetime),
           month = month(datetime),
           year = year(datetime),
           wday = wday(datetime)) -> bike

test %>% 
    mutate(datetime = fastPOSIXct(datetime, "GMT")) %>% 
    mutate(hour = hour(datetime),
           month = month(datetime),
           year = year(datetime),
           wday = wday(datetime)) -> test

bike$datetime <- bike$casual <- bike$registered <- NULL
test$datetime <- test$casual <- test$registered <- NULL

####################
#### Group by
bike %>% 
    group_by(hour, workingday) %>%
    summarise(hour_wday_median_count = median(count)) -> tbl_median_count1 

bike %>% 
    group_by(hour, wday) %>%
    summarise(hour_day_median_count = median(count)) -> tbl_median_count2 

bike %>% 
    group_by(hour, wday, season) %>%
    summarise(hour_season_median_count =  median(count)) -> tbl_median_count3 

####################
#### Mutations

bike %>% left_join(tbl_median_count1, by = c('hour', 'workingday')) %>% 
    mutate(hour_wday_median_count = log1p(hour_wday_median_count)) -> bike

bike %>% left_join(tbl_median_count2, by = c('hour', 'wday')) %>%
    mutate(hour_day_median_count = log1p(hour_day_median_count)) -> bike

bike %>% left_join(tbl_median_count3, by = c('hour', 'wday', 'season')) %>%
    mutate(hour_day_median_count = log1p(hour_day_median_count)) -> bike

test %>% left_join(tbl_median_count1, by = c('hour', 'workingday')) %>% 
    mutate(hour_wday_median_count = log1p(hour_wday_median_count)) -> test

test %>% left_join(tbl_median_count2, by = c('hour', 'wday')) %>%
    mutate(hour_day_median_count = log1p(hour_day_median_count)) -> test

test %>% left_join(tbl_median_count3, by = c('hour', 'wday', 'season')) %>%
    mutate(hour_day_median_count = log1p(hour_day_median_count)) -> test

bike$count <- bike$log_count
bike$log_count <- NULL

names_to_factorize <- c("hour", "month", "year", "wday", "season", "holiday", "workingday", "weather")
bike[,names_to_factorize] = lapply(bike[,names_to_factorize], factor)
test[,names_to_factorize] = lapply(test[,names_to_factorize], factor)


lasso_variable_selection <- function(df, y_name, folds, alpha = 1, parallel = F) {
    X <- model.matrix(as.formula(paste(y_name,"~.")), data = df)[, -1]
    fit <- glmnet::cv.glmnet(x=X, y = df[, y_name], type.measure = 'mse', 
                             nfolds = folds, parallel = parallel, alpha = alpha)
    coefs<- coef(fit, s = "lambda.1se")
    idx <- which(coefs != 0)
    variables <- row.names(coefs)[idx]
    variables<-variables[!(variables %in% '(Intercept)')]
    return(unique(gsub("[[:digit:]]", "", variables)))
}

parallelStartMulticore(4)
best_vars <- lasso_variable_selection(as.data.frame(bike), 'count',
                                      10, 1, T)
parallelStop()

fm <- as.formula(paste0("count ~", paste0(" ",best_vars, collapse = "+")))

fit <- lm(fm, data = bike)



