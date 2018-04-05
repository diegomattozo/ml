library(fasttime, quietly = T, warn.conflicts = F)
library(lubridate, quietly = T, warn.conflicts = F)
library(dplyr)
library(magrittr)
library(mlr)
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

# Create task and learner
trainTask = makeRegrTask(data = bike, target = "count")
lrn = makeLearner("regr.cubist")

# Define hyperparameter ranges you want to consider for tuning
ps = makeParamSet(
    makeDiscreteParam("committees", values = 1:100),
    makeDiscreteParam("neighbors", values = 0:9)
)

# Use 'maxit' iterations of random search for tuning (parallelize each iteration using 16 cores)
ctrl = makeTuneControlRandom(maxit = 48)
rdesc = makeResampleDesc("CV", iters = 4)
parallelStartMulticore(4)
(res = tuneParams(lrn, trainTask, rdesc, measures = rmse, par.set = ps, control = ctrl))
parallelStop()

# Train the model with best hyperparameters
mod = train(setHyperPars(lrn, par.vals = c(res$x)), trainTask)

# Make prediction (convert predictions back using the inverse of log(x+1))
pred = expm1(getPredictionResponse(predict(mod, newdata = test)))
submit = data.frame(datetime = dateTest, count = pred)
write.csv(submit, file = "script.csv", row.names = FALSE)

