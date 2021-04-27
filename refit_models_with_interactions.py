import logging
import pickle
from math import ceil
from random import choice
from typing import List, Tuple, Dict

import numpy as np
from nptyping import NDArray
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

from models import fit_model, load_data, optimise_hps, save_output
from parse_random_forest import valid_feature_pair
from utils import ResultsContainer, accuracy, mean_acc_per_bin


def map_loci(interacting_loci: NDArray) -> Dict[int, int]:
    loci = set(
        [i[0] for i in interacting_loci] + [i[1] for i in interacting_loci]
    )
    return {i: ceil((i + 1) / 20) for i in loci}


def check_interactions(model: Lasso, interactions: List[Tuple[int, int]]):
    non_zero_coef = np.where(model.coef_ != 0)[0]
    interactions_array = np.array(interactions)
    interacting_loci = interactions_array[non_zero_coef]

    return map_loci(interacting_loci)


def simulate_random_interactions(
    n: int, sequence_length: int
) -> List[Tuple[int, int]]:

    features = list(range(sequence_length))
    feature_pairs: List[Tuple[int, int]] = []
    while len(feature_pairs) < n:
        fp = (choice(features), choice(features))
        if valid_feature_pair(*fp):
            feature_pairs.append(fp)

    return feature_pairs


def random_interaction_model_fits(
    n: int, sequence_length: int, model_type: str = "lasso"
) -> float:
    interactions = simulate_random_interactions(n, sequence_length)

    train, test, _ = load_data(
        blosum_inference=True, interactions=tuple(interactions)
    )

    model = fit_model(train, model_type)
    test_predictions = model.predict(test[0])

    MSE = mean_squared_error(test[1], test_predictions)
    print(MSE)
    return MSE


def main():

    model_type = "lasso"
    pbounds = {"alpha": [0.05, 1.95]}

    logging.info("Loading inferred interaction data")
    with open("paired_sf_p_values.pkl", "rb") as a:
        paired_sf_p_values = pickle.load(a)

    # lowest p values are smaller than smallest 64 bit floating point number
    interactions = [i[0] for i in paired_sf_p_values if i[1] == 0]

    train, test, validate = load_data(
        blosum_inference=True, interactions=interactions
    )

    logging.info("Optimising the model for the test data accuracy")
    optimizer = optimise_hps(
        train, test, pbounds, model_type
    )  # select hps using GP

    logging.info(
        f"Fitting model with optimal hyperparameters: {optimizer.max['params']}"  # noqa: E501
    )
    model = fit_model(
        train, model_type=model_type, **optimizer.max["params"]
    )  # get best model fit

    train_predictions = model.predict(train[0])
    test_predictions = model.predict(test[0])
    validate_predictions = model.predict(validate[0])

    results = ResultsContainer(  # noqa: F841
        training_predictions=train_predictions,
        testing_predictions=test_predictions,
        validation_predictions=validate_predictions,
        training_MSE=mean_squared_error(train[1], train_predictions),
        testing_MSE=mean_squared_error(test[1], test_predictions),
        validation_MSE=mean_squared_error(validate[1], validate_predictions),
        training_accuracy=accuracy(train_predictions, train[1]),
        testing_accuracy=accuracy(test_predictions, test[1]),
        validation_accuracy=accuracy(validate_predictions, validate[1]),
        training_mean_acc_per_bin=mean_acc_per_bin(
            train_predictions, train[1]
        ),
        testing_mean_acc_per_bin=mean_acc_per_bin(test_predictions, test[1]),
        validation_mean_acc_per_bin=mean_acc_per_bin(
            validate_predictions, validate[1]
        ),
        hyperparameters=optimizer.max["params"],
        model_type=model_type,
        model=model,
    )


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)

    main()
