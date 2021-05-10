data {
    int<lower=0> N;  // number of data items
    int<lower=0> K;  // number of predictors
    matrix[N, K] x;  // predictor matrix
    vector[N] y;  // outcome vector

    real alpha_mean;  // mean of alpha prior
    real<lower=0> alpha_sd;  //  sd of alpha prior
    real beta_mean;  // mean of beta prior
    real<lower=0> beta_sd;  // sd of beta priors
}

parameters {
    real alpha;
    vector[K] beta;
    real<lower=0> sigma;  // error scale
}

model {
    alpha ~ normal(alpha_mean, alpha_sd);
    for (k in 1:K)
        beta[k] ~ normal(beta_mean, beta_sd);
    y ~ normal(x * beta + alpha, sigma);  // likelihood
}
