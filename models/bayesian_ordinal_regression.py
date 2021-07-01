from typing import Iterable

import matplotlib.pyplot as plt
import numpyro
from jax import numpy as jnp
from jax import random
from jax.interpreters.xla import _DeviceArray
from numpyro import sample
from numpyro.distributions import (
    ImproperUniform,
    Normal,
    OrderedLogistic,
    constraints,
)
from numpyro.infer import MCMC, NUTS, Predictive
import seaborn as sns


class BayesianOrdinalRegression:
    def __init__(
        self,
        X: Iterable,
        Y: Iterable,
        X_dim: int,
        n_classes: int,
        beta_prior_sd: float = 1.0,
    ):
        if not isinstance(X, _DeviceArray):
            X = jnp.array(X)
        if not isinstance(Y, _DeviceArray):
            Y = jnp.array(Y)
        self.X = X
        self.Y = Y
        self.X_dim = X_dim
        self.n_classes = n_classes
        self.beta_prior_sd = beta_prior_sd
        self.mcmc_key = random.PRNGKey(1234)

    def _model(self, X, Y=None):
        b_X_eta = sample(
            "b_X_eta", Normal(0, jnp.array([self.beta_prior_sd] * self.X_dim))
        )  # betas for each X are drawn from normal distribution
        c_y = sample(
            "c_y",
            ImproperUniform(
                support=constraints.ordered_vector,
                batch_shape=(),
                event_shape=(self.n_classes - 1,),
            ),
        )  # cutpoints for ordered logisitic model
        with numpyro.plate("obs", X.shape[0]):
            eta = X * b_X_eta
            eta = eta.sum(axis=1)  # summing across beta coefficients
            return sample("Y", OrderedLogistic(eta, c_y), obs=Y)

    def fit(self, num_warmup: int = 250, num_samples: int = 750):
        kernel = NUTS(self._model)
        self.mcmc = MCMC(
            kernel, num_warmup=num_warmup, num_samples=num_samples
        )
        self.mcmc.run(self.mcmc_key, self.X, self.Y)

    def predict(self, X: Iterable) -> _DeviceArray:
        if not isinstance(X, _DeviceArray):
            X = jnp.array(X)
        return Predictive(
            self._model, posterior_samples=self.mcmc.get_samples()
        )(self.mcmc_key, X)["Y"]

    def plot_posteriors(self, n: int, param_type: str):
        param_name = f"{param_type}_{n}"
        param = self.mcmc.get_samples()[param_type][:, n]  # posterior samples
        mean = jnp.mean(param)
        median = jnp.median(param)
        CI_lower = jnp.percentile(param, 2.5)
        CI_upper = jnp.percentile(param, 97.5)

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
        plt.axvline(median, color="c", lw=2, linestyle="--", label="median")
        plt.axvline(
            CI_lower, linestyle=":", color="k", alpha=0.2, label="95% CI"
        )
        plt.axvline(CI_upper, linestyle=":", color="k", alpha=0.2)

        plt.title(f"Trace and Posterior Distribution for {param_name}")
        plt.gcf().tight_layout()
        plt.legend()
        plt.show()

        plt.clf()
