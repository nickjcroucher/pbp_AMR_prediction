#!/usr/bin/env python3

import argparse
import logging

from sklearn.metrics import mean_squared_error

from models.HMM_model import ProfileHMMPredictor
from utils import accuracy, mean_acc_per_bin, load_data, ResultsContainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit predictive model using profile HMM of penicilin MIC to PBP amino acid sequence"  # noqa: E501
    )
    parser.add_argument(
        "--train_pop",
        dest="train_data_population",
        type=str,
        help="Population which the model should be fitted to, either cdc, pmen or maela",  # noqa: E501
    )
    parser.add_argument(
        "--test_pop_1",
        dest="test_data_population_1",
        type=str,
        help="Either cdc, pmen, or maela",
    )
    parser.add_argument(
        "--test_pop_2",
        dest="test_data_population_2",
        type=str,
        help="Either cdc, pmen, or maela",
    )
    parser.add_argument(
        "--standardise_training_MIC",
        type=bool,
        default=True,
        help="Where multiple MICs are reported for same PBP type, sets all to mean of fitted normal distribution",  # noqa: E501
    )
    parser.add_argument(
        "--standardise_testing_MIC",
        dest="standardise_test_and_val_MIC",
        type=bool,
        default=False,
        help="Where multiple MICs are reported for same PBP type, sets all to mean of fitted normal distribution",  # noqa: E501
    )

    return vars(parser.parse_args())  # return as a dictionary


def main(
    train_data_population="cdc",
    test_data_population_1="pmen",
    test_data_population_2="maela",
    standardise_training_MIC=True,
    standardise_test_and_val_MIC=False,
):
    logging.info("Loading data")

    pbp_seqs = ["a1_seq", "b2_seq", "x2_seq"]

    train, test_1, test_2, val = load_data(
        train_data_population,
        test_data_population_1,
        test_data_population_2,
        standardise_training_MIC,
        standardise_test_and_val_MIC,
    )

    logging.info("Fitting model")
    p_HMM_predictor = ProfileHMMPredictor(
        train,
        pbp_seqs=pbp_seqs,
    )

    train_y = train.log2_mic.values
    test_1_y = test_1.log2_mic.values
    test_2_y = test_2.log2_mic.values
    val_y = val.log2_mic.values

    logging.info("Generating predictions")
    train_predictions = p_HMM_predictor.predict_phenotype(
        train[pbp_seqs].sum(axis=1)
    )
    test_predictions_1 = p_HMM_predictor.predict_phenotype(
        test_1[pbp_seqs].sum(axis=1)
    )
    test_predictions_2 = p_HMM_predictor.predict_phenotype(
        test_2[pbp_seqs].sum(axis=1)
    )
    validate_predictions = p_HMM_predictor.predict_phenotype(
        val[pbp_seqs].sum(axis=1)
    )

    results = ResultsContainer(
        training_predictions=train_predictions,
        validation_predictions=validate_predictions,
        testing_predictions_1=test_predictions_1,
        testing_predictions_2=test_predictions_2,
        training_MSE=mean_squared_error(train_y, train_predictions),
        validation_MSE=mean_squared_error(val_y, validate_predictions),
        testing_MSE_1=mean_squared_error(test_1_y, test_predictions_1),
        testing_MSE_2=mean_squared_error(test_2_y, test_predictions_2),
        training_accuracy=accuracy(train_predictions, train_y),
        validation_accuracy=accuracy(validate_predictions, val_y),
        testing_accuracy_1=accuracy(test_predictions_1, test_1_y),
        testing_accuracy_2=accuracy(test_predictions_2, test_2_y),
        training_mean_acc_per_bin=mean_acc_per_bin(train_predictions, train_y),
        validation_mean_acc_per_bin=mean_acc_per_bin(
            validate_predictions, val_y
        ),
        testing_mean_acc_per_bin_1=mean_acc_per_bin(
            test_predictions_1, test_1_y
        ),
        testing_mean_acc_per_bin_2=mean_acc_per_bin(
            test_predictions_2, test_2_y
        ),
        hyperparameters={},
        model_type="profile_HMM_phenotype_predictor",
        model=p_HMM_predictor,
        config={
            "train_data_population": train_data_population,
            "test_data_population_1": test_data_population_1,
            "test_data_population_2": test_data_population_2,
            "standardise_training_MIC": standardise_training_MIC,
            "standardise_test_and_val_MIC": standardise_test_and_val_MIC,
        },
    )

    print(results)


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)

    main(**parse_args())
