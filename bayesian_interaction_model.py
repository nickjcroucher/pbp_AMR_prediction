import os
from uuid import uuid1

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cmdstanpy import CmdStanModel
from nptyping import NDArray
from pandas import DataFrame
from tqdm import tqdm


class BayesianLinearModel:
    def __init__(self, training_X: NDArray, training_y: NDArray):
        self.model = CmdStanModel(stan_file="stan/linear_model.stan")
        self.model_ppc = CmdStanModel(stan_file="stan/linear_model_ppc.stan")
        self.features = training_X
        self.labels = training_y
        self.model_fit = None
        self.data = None
        self.output_dir = os.path.join("bayesian_models", str(uuid1()))

    def fit(
        self,
        num_chains: int = 4,
        num_samples: int = 5000,
        alpha_mean: int = 0,
        alpha_sd: int = 1,
        beta_mean: int = 0,
        beta_sd: int = 1,
    ):
        self.data = {
            "N": self.features.shape[0],
            "K": self.features.shape[1],
            "x": self.features,
            "y": self.labels,
            "alpha_mean": alpha_mean,
            "alpha_sd": alpha_sd,
            "beta_mean": beta_mean,
            "beta_sd": beta_sd,
        }

        self.model_fit = self.model.sample(
            data=self.data,
            chains=num_chains,
            iter_sampling=num_samples,
            iter_warmup=int(num_samples / 5),
            show_progress=True,
            seed=1,
            output_dir=self.output_dir,
        )

    def predict(self, testing_features: NDArray) -> DataFrame:
        assert (
            self.model_fit is not None
        ), "Cannot generate predictions from an unfitted model"

        assert (
            testing_features.shape[1] == self.features.shape[1]
        ), "Number of features in the data is not equal to the number in the \
                data the model was trained on"

        data = {
            "N_tilde": testing_features.shape[0],
            "K_tilde": testing_features.shape[1],
            "x_tilde": testing_features,
        }
        new_quantities = self.model_ppc.generate_quantities(
            data=data, mcmc_sample=self.model_fit, seed=1
        )
        return new_quantities.generated_quantities_pd

    def plot_model_fit(self):
        # create location in which to save the summary plots
        plots_directory = os.path.join(self.output_dir, "plots")
        if not os.path.isdir(plots_directory):
            os.makedirs(plots_directory)

        betas = [f"beta[{i+1}]" for i in range(self.features.shape[1])]
        params = ["alpha", "sigma"] + betas
        posteriors = self.model_fit.draws_pd(inc_warmup=False)[params]
        print(f"Plotting fitting statistics for {len(params)} parameters")
        for param_name in tqdm(params):
            # extract fitting statistics
            param = posteriors[param_name]
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

            plt.clf()

        print(f"Plots saved under {plots_directory}")
