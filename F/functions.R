#######################################
# Assignment 2 Functions
# STA314H1, Fall 2020
# Author: Ziyue Yang
# Student Number: 1004804759
# Contact: ziyue.yang@mail.utoronto.ca
#######################################

library(tidyverse)
library(tidymodels)

fit_logistic_lasso <- function(x, y, lambda, beta0 = NULL, eps = 0.0001, iter_max = 100) {
  # Inputs:
  #         x: matrix of predictors (not including the intercept)
  #         y: vector of data
  #         beta0: initial guess
  #         eps: parameter for stopping criterion
  #         iter_max: maximum number number of iterations
  # Output:
  #         Returns a list containing the members `intercept`, `beta`, 
  #         and `lambda`.
  #
  ####################### Function Body begins here #######################

  if (is.null(beta0)) {
    beta0 <- rep(0, dim(as.matrix(x))[2])
  }
  beta <- beta0
  
  factor_levels <- levels(y)
  y <- as.numeric(y) - 1
  
  for (i in 1:iter_max) {
    xbeta <- as.numeric(as.matrix(x) %*% beta)
    p <- 1 / (1 + exp(-xbeta))
    w <- p * (1 - p)
    z <- xbeta + (y - p) / w
    beta_old <- beta
    beta <- coordinate_descent(x, w, z, y, lambda, beta, iter_max)
    
    # Converges
    if (sqrt(as.vector(Matrix::crossprod(beta - beta_old))) < eps) {
      break
    }
    
    return(list(intercept = lambda * sum(abs(beta)), beta = beta, lambda = lambda))
  }
}

# ret <- fit_logistic_lasso(x, y, lambda, beta0 = NULL, eps = 0.0001, iter_max = 1000)

predict_logistic_lasso <- function(object, new_x) {
  # Inputs:
  #         object: Output from fit_logistic_lasso
  #         new_x: Data to predict at (may be more than one point)
  # Output:
  #         Returns a list containing the intercept and beta.
  #
  ####################### Function Body begins here #######################
  
  pred <- as.numeric(as.matrix(new_x) %*% object$beta >= 0)
  return(list(
    intercept = object$intercept,
    beta = object$beta
  ))

}

### Helper functions

coordinate_descent = function(x, w, z, y, lambda, beta, iter_max) {
  # DOC TODO
  loss <- quadratic_approx(x, w, z, lambda, beta, iter_max)
  for(i in 1:iter_max) {
    beta_old <- beta
    loss_old <- loss
    beta <- update_coor(x, w, z, y, lambda, beta)
    loss <- quadratic_approx(x, w, z, lambda, beta, iter_max)
    if (loss >= loss_old) {
      beta <- beta_old
      break
    }
    if (i == iter_max && wls <= wls_old) {
      warning(paste("Coordinate did not converge in", iter_max, "iterations", 
                    sep = " "))
    }
  }
  return(beta)
}


quadratic_approx <- function(x, w, z, lambda, beta, iter_max) {
  # DOC TODO
  return(
    (length(x))**-1 / 2 * sum(w * (z - as.matrix(x) %*% beta) ** 2) + lambda * sum(abs(beta))
  )
}


update_coor <- function(x, w, z, y, lambda, beta) {
  new_beta <- beta
  for (j in 1:length(beta)) {
    rj <- y - new_beta %*% as.matrix(x)[,j] + new_beta[j] * as.matrix(x)[,j]
    new_beta[j,] <- sign(as.matrix(x)[,j] %*% rj) * max(abs(as.matrix(x)[,j] %*% rj) - lambda, 0)
  }
  return(new_beta)
}




# ret <- predict_logistic_lasso(object, new_x)
