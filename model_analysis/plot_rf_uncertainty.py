import os
import pickle

import numpy as np
import pandas as pd

from fit_models import load_and_format_data
from models.supervised_models import (
    rf_PI_accuracy,
    rf_prediction_distribution,
    rf_prediction_intervals,
)


def load_results(
    train_data_population: str,
    just_HMM_scores: bool,
    include_HMM_scores: bool,
    blosum_inference: bool,
    filter_unseen: bool,
    HMM_inference: bool,
    HMM_MIC_inference: bool,
):
    outdir = "results/random_forest"
    if just_HMM_scores:
        outdir = os.path.join(outdir, "just_HMM_scores")
    elif include_HMM_scores:
        outdir = os.path.join(outdir, "include_HMM_scores")
    if blosum_inference:
        filename = f"train_pop_{train_data_population}_results_blosum_inferred_pbp_types.pkl"  # noqa: E501
    elif filter_unseen:
        filename = f"train_pop_{train_data_population}_results_filtered_pbp_types.pkl"  # noqa: E501
    elif HMM_inference:
        filename = f"train_pop_{train_data_population}_results_HMM_inferred_pbp_types.pkl"  # noqa: E501
    elif HMM_MIC_inference:
        filename = f"train_pop_{train_data_population}_results_HMM_MIC_inferred_pbp_types.pkl"  # noqa: E501

    with open(os.path.join(outdir, filename), "rb") as a:
        return pickle.load(a)


def main(
    train_data_population: str = "cdc",
    blosum_inference: bool = False,
    HMM_inference: bool = False,
    HMM_MIC_inference: bool = False,
    filter_unseen: bool = False,
    include_HMM_scores: bool = False,
    just_HMM_scores: bool = False,
):

    # load previously fitted random forest model
    results = load_results(
        train_data_population,
        just_HMM_scores,
        include_HMM_scores,
        blosum_inference,
        filter_unseen,
        HMM_inference,
        HMM_MIC_inference,
    )
    model = results.model

    # load data model was fitted with
    data = load_and_format_data(
        train_data_population,
        results.config["test_1_population"],
        results.config["test_2_population"],
        blosum_inference=False,
        HMM_inference=False,
        HMM_MIC_inference=False,
        filter_unseen=False,
        include_HMM_scores=include_HMM_scores,
        just_HMM_scores=just_HMM_scores,
        standardise_training_MIC=False,
        standardise_test_and_val_MIC=False,
    )

    pi_acc_dict = {}
    for d in ["train", "val", "test_1", "test_2"]:
        prediction_dist = rf_prediction_distribution(data[d][0], model)
        intervals = np.array(list(range(5, 100, 5))) / 100
        pi_accuracies = pd.DataFrame(
            {
                "Accuracy": [
                    rf_PI_accuracy(
                        data[d][1], rf_prediction_intervals(prediction_dist, i)
                    )
                    for i in intervals
                ],
                "Interval": intervals,
            }
        )
        pi_acc_dict[d] = pi_accuracies


if __name__ == "__main__":
    main()
