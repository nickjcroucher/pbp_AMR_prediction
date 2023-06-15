import pickle
from multiprocessing import cpu_count
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpyro
import seaborn as sns
from jax import numpy as jnp
from jax import random, Array
#from jax.interpreters.xla import _DeviceArray
from numpyro import sample
from numpyro.diagnostics import gelman_rubin
from numpyro.distributions import (
    Normal,
    OrderedLogistic,
    TransformedDistribution,
    transforms,
)
from numpyro.infer import BarkerMH, init_to_value, MCMC, NUTS, Predictive

# this has to be set before calling any other numpyro functions
numpyro.set_host_device_count(cpu_count())


class BayesianOrdinalRegression:
    def __init__(
        self,
        X: Iterable,
        Y: Iterable,
        X_dim: int,
        n_classes: int,
        beta_prior_sd: float = 1.0,
        num_chains: int = 4,
    ):
#        if not isinstance(X, _DeviceArray):
        if not isinstance(X, Array):
            X = jnp.array(X)
#        if not isinstance(Y, _DeviceArray):
        if not isinstance(Y, Array):
            Y = jnp.array(Y)
        self.X = X
        self.Y = Y
        self.X_dim = X_dim
        self.n_classes = n_classes
        self.beta_prior_sd = beta_prior_sd
        self.mcmc_key = random.PRNGKey(1234)
        self.num_chains = num_chains
        self.posterior_samples = None
        self.last_mcmc_state = None

    def _model(self, X, Y=None):
        beta = sample(
            "beta",
            Normal(0, self.beta_prior_sd),
            sample_shape=(self.X_dim,),
        )  # betas for each X are drawn from normal distribution
        c = sample(
            "c",
            TransformedDistribution(
                Normal(0, 1).expand([self.n_classes - 1]),
                transforms.OrderedTransform(),
            ),
        )  # cutpoints for ordered logisitic model
        with numpyro.plate("obs", X.shape[0]):
            eta = X * beta
            eta = eta.sum(axis=1)  # summing across beta coefficients
            return sample("Y", OrderedLogistic(eta, c), obs=Y)

    def fit_with_NUTS(
        self,
        num_warmup: int = 2500,
        num_samples: int = 7500,
        step_size: float = 1.0,
        target_accept_prob: float = 0.8,
    ):
        if self.posterior_samples is None:
            kernel = NUTS(
                self._model,
                step_size=step_size,
                target_accept_prob=target_accept_prob,
            )
        else:
            kernel = NUTS(
                self._model,
                step_size=step_size,
                target_accept_prob=target_accept_prob,
                init_strategy=init_to_value(
                    values={
                        "beta": self.posterior_samples["beta"][-1, :],
                        "c": self.posterior_samples["c"][-1, :],
                    }
                ),
            )
        self._fit(kernel, num_warmup, num_samples)

    def fit_with_BarkerMH(
        self,
        num_warmup: int = 2500,
        num_samples: int = 7500,
        step_size: float = 1.0,
        target_accept_prob: float = 0.4,
    ):
        if self.posterior_samples is None:
            kernel = BarkerMH(
                self._model,
                step_size=step_size,
                target_accept_prob=target_accept_prob,
            )
        else:
            kernel = BarkerMH(
                self._model,
                step_size=step_size,
                target_accept_prob=target_accept_prob,
                init_strategy=init_to_value(
                    values={
                        "beta": self.posterior_samples["beta"][-1, :],
                        "c": self.posterior_samples["c"][-1, :],
                    }
                ),
            )
        self._fit(kernel, num_warmup, num_samples)

    def _fit(
        self,
        kernel: MCMC,
        num_warmup: int,
        num_samples: int,
    ):
        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=self.num_chains,
        )
        self.mcmc.run(self.mcmc_key, self.X, self.Y, init_params=self.last_mcmc_state)
        self.posterior_samples = self.mcmc.get_samples()
        self.last_mcmc_state = self.mcmc.last_state

#    def predict(self, X: Iterable, num_samples: Optional[int] = None) -> _DeviceArray:
    def predict(self, X: Iterable, num_samples: Optional[int] = None) -> Array:
#        if not isinstance(X, _DeviceArray):
        if not isinstance(X, Array):
            X = jnp.array(X)
        return Predictive(
            self._model,
            num_samples=num_samples,
            posterior_samples=self.posterior_samples,
            parallel=True,
        )(self.mcmc_key, X)["Y"]

    def plot_posteriors(self, n: int, param_type: str):
        param_name = f"{param_type}_{n}"
        param_chains = self.posterior_samples[param_type][:, n]  # type: ignore # noqa: E501
        param_chains = jnp.array_split(param_chains, self.num_chains)
        plt.clf()
        for n, chain in enumerate(param_chains):
            n += 1
            mean = jnp.mean(chain)
            median = jnp.median(chain)
            CI_lower = jnp.percentile(chain, 2.5)
            CI_upper = jnp.percentile(chain, 97.5)

            # trace plot
            plt.subplot(4, 2, (n * 2) - 1)
            plt.plot(chain)
            plt.axhline(mean, color="r", lw=2, linestyle="--")
            plt.axhline(median, color="c", lw=2, linestyle="--")
            plt.axhline(CI_lower, linestyle=":", color="k", alpha=0.2)
            plt.axhline(CI_upper, linestyle=":", color="k", alpha=0.2)

            # density plot
            plt.subplot(4, 2, n * 2)
            plt.hist(chain, 30, density=True)
            sns.kdeplot(chain, shade=True)
            plt.ylabel("")
            plt.axvline(mean, color="r", lw=2, linestyle="--", label="mean")
            plt.axvline(median, color="c", lw=2, linestyle="--", label="median")
            plt.axvline(CI_lower, linestyle=":", color="k", alpha=0.2, label="95% CI")
            plt.axvline(CI_upper, linestyle=":", color="k", alpha=0.2)

        plt.suptitle(f"Trace and Posterior Distribution for {param_name}")
        plt.gcf().tight_layout()
        plt.legend(loc="lower right")
        plt.show()

        plt.clf()

    def gelman_rubin_stats(self) -> Dict[str, float]:
        def param_gr_stats(param, i):
            chains = jnp.array_split(
                self.posterior_samples[param][:, i], self.num_chains
            )
            return gelman_rubin(jnp.stack(chains))

        gr_stats = {f"beta_{i}": param_gr_stats("beta", i) for i in range(self.X_dim)}
        gr_stats.update(
            {f"c_{i}": param_gr_stats("c", i) for i in range(self.n_classes)}
        )

        return gr_stats

    def test_convergence(self) -> float:
        """
        Returns the fraction of model parameters for which the gelman-rubin
        statistic is greater than 1.2
        """
        gr_stats = self.gelman_rubin_stats()
        return len([i for i in gr_stats.values() if i > 1.2]) / len(gr_stats)


# can't pickle mcmc.kernel so use this hack to save and reload models
def save_bayesian_ordinal_regression(model: BayesianOrdinalRegression, filename: str):
    attributes_dict = {
        "X": model.X,
        "Y": model.Y,
        "X_dim": model.X_dim,
        "n_classes": model.n_classes,
        "beta_prior_sd": model.beta_prior_sd,
        "mcmc_key": model.mcmc_key,
        "posterior_samples": model.posterior_samples,
        "num_chains": model.num_chains,
        "last_mcmc_state": model.last_mcmc_state,
    }
    with open(filename, "wb") as a:
        pickle.dump(attributes_dict, a)


def load_bayesian_ordinal_regression(
    filename: str,
) -> BayesianOrdinalRegression:
    with open(filename, "rb") as a:
        attributes_dict = pickle.load(a)

    model = BayesianOrdinalRegression(
        attributes_dict["X"],
        attributes_dict["Y"],
        attributes_dict["X_dim"],
        attributes_dict["n_classes"],
        attributes_dict["beta_prior_sd"],
        attributes_dict["num_chains"],
    )
    model.mcmc_key = attributes_dict["mcmc_key"]
    model.posterior_samples = attributes_dict["posterior_samples"]
    model.last_mcmc_state = attributes_dict["last_mcmc_state"]

    return model
