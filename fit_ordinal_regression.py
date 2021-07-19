import pickle
from functools import partial
from typing import Dict, List, Optional

from bayes_opt import BayesianOptimization
from numpy import mean

from fit_models import load_and_format_data
from models import bayesian_ordinal_regression
from utils import ordinal_regression_format


def train_evaluate(
    data: Dict,
    beta_prior_sd: float,
    num_warmup: int,
    num_samples: int,
    step_size: float,
    target_accept_prob: float,
) -> float:

    model = bayesian_ordinal_regression.BayesianOrdinalRegression(
        data["train"][0],
        data["train"][1],
        data["train"][0].shape[1],
        len(set(data["train"][1])),
        beta_prior_sd=beta_prior_sd,
        num_chains=2,
    )
    model.fit_with_NUTS(
        num_warmup=int(num_warmup),
        num_samples=int(num_samples),
        step_size=step_size,
        target_accept_prob=target_accept_prob,
    )
    gr_stats = model.gelman_rubin_stats()

    return -mean(list(gr_stats.values()))


def optimise_HMC_sampler(
    data: Dict, pbounds=Dict[str, List[float]], init_points=5, n_iter=15
):
    partial_fitting_function = partial(train_evaluate, data=data)
    optimizer = BayesianOptimization(
        f=partial_fitting_function, pbounds=pbounds, random_state=0
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    return optimizer


def main(
    train_data_population: str = "cdc",
    test_data_population_1: str = "pmen",
    test_data_population_2: Optional[str] = "maela",
    model_type: str = "random_forest",
    blosum_inference: bool = True,
    HMM_inference: bool = False,
    HMM_MIC_inference: bool = False,
    filter_unseen: bool = False,
    include_HMM_scores: bool = False,
    just_HMM_scores: bool = False,
    standardise_training_MIC: bool = True,
    standardise_test_and_val_MIC: bool = False,
    previous_rf_model: str = None,
    extended_sequences: bool = False,
    minor_variant_frequency: float = 0.05,
    beta_prior_sd: float = 10.0,
):

    data = load_and_format_data(
        train_data_population,
        test_data_population_1,
        test_data_population_2,
        blosum_inference=blosum_inference,
        HMM_inference=HMM_inference,
        HMM_MIC_inference=HMM_MIC_inference,
        filter_unseen=filter_unseen,
        include_HMM_scores=include_HMM_scores,
        just_HMM_scores=just_HMM_scores,
        standardise_training_MIC=standardise_training_MIC,
        standardise_test_and_val_MIC=standardise_test_and_val_MIC,
        extended_sequences=extended_sequences,
    )
    data = ordinal_regression_format(data, 0.05)
    pbounds = {
        "beta_prior_sd": [0.1, 10.0],
        "num_warmup": [10000, 100000],
        "num_samples": [10000, 100000],
        "step_size": [0.8, 1.2],
        "target_accept_prob": [0.5, 0.7],
    }
    optimizer = optimise_HMC_sampler(data, pbounds=pbounds)
    with open("hmc_sampler_optimizer.pkl", "wb") as f:
        pickle.dump(optimizer, f)


if __name__ == "__main__":
    main()
