#!/usr/bin/env python3

import argparse
import logging

from utils import load_data


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

    train, test_1, test_2, val = load_data(
        train_data_population,
        test_data_population_1,
        test_data_population_2,
        standardise_training_MIC,
        standardise_test_and_val_MIC,
    )


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)

    main(**parse_args())
