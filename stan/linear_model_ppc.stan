data {
    int<lower=0> N_tilde;   // number of data items
    int<lower=0> K_tilde;   // number of predictors MUST BE EQUAL TO K
    matrix[N_tilde, K_tilde] x_tilde;   // predictor matrix
}

parameters {
    real alpha;           // intercept
    vector[K_tilde] beta; // coefficients for predictors
    real<lower=0> sigma;  // error scale
}

generated quantities {
    vector[N_tilde] y_tilde;
    for (n in 1:N_tilde)
        y_tilde[n] = normal_rng(x_tilde[n] * beta + alpha, sigma);
}
