from typing import Iterable, Optional

from fit_models import (
    check_new_file_path,
    load_and_format_data,
    filter_features_by_previous_model_fit,
)
from models import bayesian_ordinal_regression
from utils import ordinal_regression_format


def fit_ordinal_regression(
    x: Iterable,
    y: Iterable,
    beta_prior_sd: float,
    num_chains: int,
    *args,
    **kwargs
) -> bayesian_ordinal_regression.BayesianOrdinalRegression:
    model = bayesian_ordinal_regression.BayesianOrdinalRegression(
        x, y, x.shape[1], len(set(y)), beta_prior_sd, num_chains
    )
    model.fit_with_NUTS(**kwargs)
    return model


def main(
    train_data_population: str = "cdc",
    test_data_population_1: str = "pmen",
    test_data_population_2: Optional[str] = "maela",
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
    if previous_rf_model is not None:
        data = filter_features_by_previous_model_fit(previous_rf_model, data)
    data = ordinal_regression_format(data, 0.05)

    model = fit_ordinal_regression(
        data["train"][0],
        data["train"][1],
        beta_prior_sd,
        num_chains=2,
        step_size=2.0,
        target_accept_prob=0.6,
    )
    filename = "fitted_ord_reg.pkl"
    bayesian_ordinal_regression.save_bayesian_ordinal_regression(
        model, check_new_file_path(filename)
    )


if __name__ == "__main__":
    main()
