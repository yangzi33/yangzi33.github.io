#######################################
# Assignment 2 Functions
# STA314H1, Fall 2020
# Author: Ziyue Yang
# Student Number: 1004804759
# Contact: ziyue.yang@mail.utoronto.ca
#######################################

# Loading libraries and functions
library(tidymodels)
library(tidyverse)
source("functions.R")

# Registering new model PIRLS

PIRLS <- function(mode = "classification") {
  
  # "PIRLS" for Penalized Iterative Reweighted Least Squares
  new_model_spec("PIRLS",
                 args = NULL, 
                 mode = mode,
                 eng_args = NULL,
                 method = NULL, 
                 engine = NULL)
}
  
set_new_model("PIRLS")
set_model_mode(model = "PIRLS", mode = "classification")
set_model_engine("PIRLS",
  mode = "classification",
  eng = "fit_logistic_lasso"
)
set_dependency("PIRLS", eng = "fit_logistic_lasso", pkg = "base")

set_encoding(
  model = "PIRLS",
  eng = "fit_logistic_lasso",
  mode = "classification", 
  options = list(
    predictor_indicators = "traditional",
    compute_intercept = TRUE,
    remove_intercept = TRUE,        # Remove intercept because of penalty
    allow_sparse_x = FALSE
  )
)

show_model_info("PIRLS")

# Configuring PIRLS 

set_fit(
  model = "PIRLS",
  eng = "fit_logistic_lasso",
  mode = "classification",
  value = list(
    interface = "matrix",
    protect = c("x", "y"),
    func = c(fun = "fit_logistic_lasso"),
    defaults = list()
  )
)

set_pred(
  model = "PIRLS",
  eng = "fit_logistic_lasso",
  mode = "classification",
  type = "class",
  value = list(
    pre = NULL,
    post = NULL,
    func = c(fun = "predict_logistic_lasso"),
    args = list(
      fit = expr(object$fit),
      new_x = expr(as.matrix(new_data[, names(object$fit$data)]))
    )
  )
)

# Update function for function PIRLS

# update.PIRLS <- function(object, ...) {
#   new_model_spec("")
# }

n = 1000
dat <- tibble(x = seq(-3,3, length.out = n),
              w = 3*cos(3*seq(-pi,pi, length.out = n)),
              y = rbinom(n,size = 1, prob = 1/(1 + exp(-w+2*x)) )%>% as.numeric %>% factor,
              cat = sample(c("a","b","c"), n, replace = TRUE)
)
split <- initial_split(dat, strata = c("cat"))
train <- training(split)
test <- testing(split)
rec <- recipe(y ~ . , data = train) %>% step_dummy(all_nominal(), -y) %>% step_zv(all_outcomes()) %>% 
  step_normalize(all_numeric(), -y) %>% # don't normalize y! 
  step_intercept() ## This is always last!

spec <- PIRLS() %>% set_engine("fit_logistic_lasso")
  
fit <- workflow() %>% add_recipe(rec) %>% add_model(spec) %>% fit(train)
predict(fit, new_data = test) %>% bind_cols(test %>% select(y)) %>% conf_mat(truth = y, estimate = .pred_class)


