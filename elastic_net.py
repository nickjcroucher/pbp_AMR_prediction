from functools import partial
from typing import Tuple, Dict, Union

import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
from nptyping import NDArray
from scipy.sparse import csr_matrix

from parse_pbp_data import parse_cdc, parse_pmen, encode_sequences
from utils import accuracy, mean_acc_per_bin, ResultsContainer


def fit_model(
    train: Tuple[Union[NDArray, csr_matrix], NDArray],
    l1_ratio: float,
    alpha: float,
) -> ElasticNet:
    reg = ElasticNet(l1_ratio=l1_ratio, alpha=alpha, random_state=0)
    reg.fit(train[0], train[1])
    return reg


def train_evaluate(
    train: Tuple[Union[NDArray, csr_matrix], NDArray],
    test: Tuple[Union[NDArray, csr_matrix], NDArray],
    l1_ratio: float,
    alpha: float,
) -> float:
    reg = fit_model(train, l1_ratio, alpha)
    MSE = mean_squared_error(test[1], reg.predict(test[0]))
    return -MSE


def optimise_hps(
    train: Tuple[Union[NDArray, csr_matrix], NDArray],
    test: Tuple[Union[NDArray, csr_matrix], NDArray],
    pbounds: Dict[str, Tuple[float, float]],
) -> BayesianOptimization:
    partial_fitting_function = partial(train_evaluate, train=train, test=test)

    optimizer = BayesianOptimization(
        f=partial_fitting_function, pbounds=pbounds, random_state=0
    )
    optimizer.maximize(n_iter=10)

    return optimizer


def main():
    cdc = pd.read_csv("../data/pneumo_pbp/cdc_seqs_df.csv")
    pmen = pd.read_csv("../data/pneumo_pbp/pmen_pbp_profiles_extended.csv")

    pbp_patterns = ["a1", "b2", "x2"]

    cdc = parse_cdc(cdc, pbp_patterns)
    pmen = parse_pmen(pmen, cdc, pbp_patterns)

    cdc_encoded_sequences = encode_sequences(cdc, pbp_patterns)
    pmen_encoded_sequences = encode_sequences(pmen, pbp_patterns)

    X_train, X_test, y_train, y_test = train_test_split(
        cdc_encoded_sequences, cdc.log2_mic, test_size=0.33, random_state=0
    )

    pbounds = {"l1_ratio": [0.2, 0.8], "alpha": [0.02, 1.8]}

    # select hps using GP
    optimizer = optimise_hps((X_train, y_train), (X_test, y_test), pbounds)

    model = fit_model((X_train, y_train), **optimizer.max["params"])

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    validate_predictions = model.predict(pmen_encoded_sequences)

    results = ResultsContainer(
        training_predictions=train_predictions,
        testing_predictions=test_predictions,
        validation_predictions=validate_predictions,
        training_MSE=mean_squared_error(y_train, train_predictions),
        testing_MSE=mean_squared_error(y_test, test_predictions),
        validation_MSE=mean_squared_error(pmen.log2_mic, validate_predictions),
        training_accuracy=accuracy(train_predictions, y_train),
        testing_accuracy=accuracy(test_predictions, y_test),
        validation_accuracy=accuracy(validate_predictions, pmen.log2_mic),
        hyperparameters=optimizer.max["params"],
        model_type="elastic_net",
        model=model,
    )