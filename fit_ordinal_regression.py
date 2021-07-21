import logging
import pickle
from typing import Iterable, Optional

import numpy as np
from sklearn.linear_model import Lasso
from scipy.sparse import csr_matrix

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
    **kwargs,
) -> bayesian_ordinal_regression.BayesianOrdinalRegression:
    model = bayesian_ordinal_regression.BayesianOrdinalRegression(
        x, y, x.shape[1], len(set(y)), beta_prior_sd, num_chains
    )
    model.fit_with_NUTS(**kwargs)
    return model


def filter_lasso_features(features: csr_matrix, model: Lasso) -> csr_matrix:
    features = np.array(features.todense())
    filtered_features = features[:, np.nonzero(model.coef_)[0]]
    return csr_matrix(filtered_features)


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
    interactions: bool = False,
):

    if interactions:
        with open(
            f"results/intermediates/{train_data_population}/paired_sf_p_values.pkl",  # noqa: E501
            "rb",
        ) as a:
            paired_sf_p_values = pickle.load(a)
        interactions = [i[0] for i in paired_sf_p_values if i[1] < 0.05]
        with open(
            "results/interaction_models/lasso_just_interactions.pkl", "rb"
        ) as a:
            # previously fitted lasso interaction model
            lasso_model = pickle.load(a).model

    logging.info("loading and formatting data")
    data = load_and_format_data(
        train_data_population,
        test_data_population_1,
        test_data_population_2,
        interactions=interactions,
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
    if interactions is not None:
        logging.info("filtering data by previously fitted interaction model")
        data = {
            k: [filter_lasso_features(v[0], lasso_model), v[1]]
            for k, v in data.items()
        }
    elif previous_rf_model is not None:
        logging.info("filtering data by previously fitted rf model")
        data = filter_features_by_previous_model_fit(previous_rf_model, data)

    data = ordinal_regression_format(data, minor_variant_frequency)

    logging.info("fitting ordinal regression model")
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
    logging.basicConfig(level=logging.INFO)
    main()
