import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import stan
from tqdm import tqdm


class BayesianLinearModel:
    def __init__(self, training_X, training_y, priors=None):
        self.model_code = """
            data {
            int<lower=0> N;   // number of data items
            int<lower=0> K;   // number of predictors
            matrix[N, K] x;   // predictor matrix
            vector[N] y;      // outcome vector
            }

            parameters {
            real alpha;           // intercept
            vector[K] beta;       // coefficients for predictors
            real<lower=0> sigma;  // error scale
            }

            model {
            y ~ normal(x * beta + alpha, sigma);  // likelihood
            }
        """
        self.features = training_X
        self.labels = training_y
        self.model_posterior = None
        self.model_fit = None
        self.fit_data = None

    def fit(self, num_chains=6, num_samples=10000):
        data = {
            "N": self.features.shape[0],
            "K": self.features.shape[1],
            "x": self.features,
            "y": self.labels,
        }
        self.model_posterior = stan.build(
            self.model_code, data=data, random_seed=1
        )
        self.model_fit = self.model_posterior.sample(
            num_chains=num_chains, num_samples=num_samples
        )
        self.fit_data = self.model_fit.to_frame()

    def predict(self):
        ...

    def plot_model_fit(self):
        # create location in which to save the summary plots
        now = datetime.datetime.today()
        now_timestamp = now.strftime("%b-%d-%Y-%H-%M-%S")
        plots_directory = f"bayes_linear_model_plots_{now_timestamp}"
        os.makedirs(plots_directory)

        betas = [f"beta.{i+1}" for i in range(self.features.shape[1])]
        params = ["alpha", "sigma"] + betas
        print(f"Plotting fitting statistics for {len(params)} parameters")
        for param_name in tqdm(params):
            # extract fitting statistics
            param = self.fit_data[param_name]
            mean = np.mean(param)
            median = np.median(param)
            CI_lower = np.percentile(param, 2.5)
            CI_upper = np.percentile(param, 97.5)

            plt.clf()

            # trace plot
            plt.subplot(2, 1, 1)
            plt.plot(param)
            plt.xlabel("samples")
            plt.ylabel(param_name)
            plt.axhline(mean, color="r", lw=2, linestyle="--")
            plt.axhline(median, color="c", lw=2, linestyle="--")
            plt.axhline(CI_lower, linestyle=":", color="k", alpha=0.2)
            plt.axhline(CI_upper, linestyle=":", color="k", alpha=0.2)

            # density plot
            plt.subplot(2, 1, 2)
            plt.hist(param, 30, density=True)
            sns.kdeplot(param, shade=True)
            plt.xlabel(param_name)
            plt.ylabel("density")
            plt.axvline(mean, color="r", lw=2, linestyle="--", label="mean")
            plt.axvline(
                median, color="c", lw=2, linestyle="--", label="median"
            )
            plt.axvline(
                CI_lower, linestyle=":", color="k", alpha=0.2, label="95% CI"
            )
            plt.axvline(CI_upper, linestyle=":", color="k", alpha=0.2)

            plt.title(f"Trace and Posterior Distribution for {param_name}")
            plt.gcf().tight_layout()
            plt.legend()
            plt.savefig(os.path.join(plots_directory, f"{param_name}.png"))
