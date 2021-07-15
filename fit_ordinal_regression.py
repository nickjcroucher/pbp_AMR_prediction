from typing import Optional

from fit_models import (
    filter_features_by_previous_model_fit,
    load_and_format_data,
)
from models import bayesian_ordinal_regression
from utils import ordinal_regression_format


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
    data = ordinal_regression_format(data, minor_variant_frequency)

    # filter features by things which have been used by previously fitted model
    if previous_rf_model is not None:
        data = filter_features_by_previous_model_fit(previous_rf_model, data)

    model = bayesian_ordinal_regression.BayesianOrdinalRegression(
        data["train"][0],
        data["train"][1],
        data["train"][0].shape[1],
        len(set(data["train"][1])),
        beta_prior_sd=1.0,
        num_chains=2,
    )
    model.fit(num_warmup=10000, num_samples=100000)
    print(model.test_convergence())

    bayesian_ordinal_regression.save_bayesian_ordinal_regression(
        model, "bayesian_ordinal_regression.pkl"
    )


if __name__ == "__main__":
    main()
