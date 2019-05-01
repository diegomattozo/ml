library(Metrics)
library(dplyr)
library(magrittr)

# -- Model
model_data <- iris %>% filter(Species %in% c("virginica", "versicolor")) %>%
  mutate("Species" = ifelse(Species == "virginica", 0, 1))

lr <- glm(Species ~., family=binomial(link='logit'), data = model_data)

y_pred <- predict(lr, newdata = model_data, type = "response")
y_true <- model_data$Species


# -- Somers d
concordant_pairs <- function(pos_pred, neg_pred) {
  neg_pred <- neg_pred[order(neg_pred)]
  total <- 0
  for (neg_score in neg_pred){
    total <- total + sum(pos_pred > neg_score)
  }
  total
}

discordant_pairs <- function(pos_pred, neg_pred) {
  neg_pred <- neg_pred[order(neg_pred, decreasing = T)]
  total <- 0
  for (neg_score in neg_pred){
    total <- total + sum(neg_score > pos_pred)
  }
  total
}
somers_d <- function(y_true, y_pred){
  pos_pred  <- y_pred[y_true == 1]
  neg_pred  <- y_pred[y_true == 0]
  cc_pairs <- concordant_pairs(pos_pred, neg_pred)
  total_pairs <- length(pos_pred) * length(neg_pred)
  nc_pairs <- discordant_pairs(pos_pred, neg_pred)
  
  (cc_pairs - nc_pairs)/total_pairs 
}

# Probabilistic version of gini
prob_normalized_gini <- function(y_true, y_pred){
  pos_pred  <- y_pred[y_true == 1]
  neg_pred  <- y_pred[y_true == 0]
  cc_pairs <- 0
  total    <- 0
  for (pos_score in pos_pred){
    for (neg_score in neg_pred){
      if (pos_score > neg_score){
        cc_pairs <- cc_pairs + 1
      } 
      total <- total + 1
    }
  }
  cc_pairs / total
}

##  AUC Transformation
normalized_gini_from_auc <- function(y_true, y_pred){
  Metrics::auc(y_true, y_pred)*2-1
}


prob_normalized_gini(y_true, y_pred)
somers_d(y_true, y_pred)
gini_from_auc(y_true, y_pred)
