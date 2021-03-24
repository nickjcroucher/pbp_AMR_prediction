import logging
from typing import List, Tuple, Dict
from functools import partial
from math import log10

import pandas as pd
from nptyping import NDArray
from Bio.SubsMat.MatrixInfo import blosum62
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization

from parse_pbp_data import parse_cdc, parse_pmen


logging.basicConfig()
logging.root.setLevel(logging.INFO)


def format_inputs(
    data: pd.DataFrame, pbp_seqs: List[str]
) -> Tuple[NDArray, NDArray]:
    X = data[pbp_seqs].to_numpy()
    Y = data.mic.to_numpy()
    return X, Y


def blosum_kernel(X, Y):
    n_features = X.shape[1]
    assert n_features == Y.shape[1], "Unequal number of features in inputs"

    def _blosum_distance(sequences, matrix=blosum62):
        seq_1, seq_2 = sequences.split("_")
        assert len(seq_1) == len(seq_2), "Sequences are of different lengths"

        def single_distance(i):
            if i[0] == i[1]:
                return 0
            try:
                return abs(blosum62[i])
            except KeyError:  # TODO verify this approach
                return abs(blosum62[(i[1], i[0])])  # try the other way around
            except KeyError:  # TODO verify this approach
                return 0

        if seq_1 == seq_2:
            return 0
        else:
            l1 = list(seq_1)
            l2 = list(seq_2)
            distances = map(single_distance, zip(l1, l2))
            return sum(distances)

    distance_matrices = []
    for i in range(n_features):
        x_ith_feature = X[:, i]
        y_ith_feature = Y[:, i]

        # dataframe of sequences of each pbp in X and Y delimited by _
        df = pd.DataFrame(
            {
                n: pd.Series(x_ith_feature)
                + "_"
                + pd.Series([y_ith_feature[n]] * X.shape[0])
                for n in range(Y.shape[0])
            }
        )

        # apply function to each cell of df
        distances = df.applymap(_blosum_distance)
        distance_matrices.append(distances.to_numpy())

    # metric is sum of blosum distances for each pbp
    return sum(distance_matrices)


def fit_model(train: Tuple[NDArray, NDArray], C: float, epsilon: float) -> SVR:
    model = SVR(C=C, epsilon=epsilon, kernel="precomputed")
    model.fit(train[0], train[1])
    return model


def train_evaluate(
    train: Tuple[NDArray, NDArray],
    test: Tuple[NDArray, NDArray],
    C: float,
    epsilon: float,
) -> float:
    # C and epsilon are sampled from log uniform distribution
    model = fit_model(train, C ** 10, epsilon ** 10)

    test_predictions = model.predict(test[0])
    return -mean_squared_error(
        test[1], test_predictions
    )  # negative error, optimiser will maximise


def optimise_hps(
    train: Tuple[NDArray, NDArray], test: Tuple[NDArray, NDArray]
) -> Dict[str, float]:
    partial_fitting_function = partial(train_evaluate, train=train, test=test)

    pbounds = {
        "C": (log10(1e-8), log10(1e-3)),
        "epsilon": (log10(1e-10), log10(1e-3)),
    }  # log uniform distribution

    optimizer = BayesianOptimization(
        f=partial_fitting_function, pbounds=pbounds, random_state=0
    )
    optimizer.maximize(n_iter=10)

    best_hyperparams = {
        "C": 10 ** optimizer.max["params"]["C"],
        "epsilon": 10 ** optimizer.max["params"]["epsilon"],
    }

    return best_hyperparams


def main():
    logging.info("Reading and formatting inputs")

    cdc = pd.read_csv("../data/pneumo_pbp/cdc_seqs_df.csv")
    pmen = pd.read_csv("../data/pneumo_pbp/pmen_pbp_profiles_extended.csv")

    pbp_patterns = ["a1", "b2", "x2"]
    pbp_seqs = list(map(lambda x: x + "_seq", pbp_patterns))

    cdc = parse_cdc(cdc, pbp_patterns)
    pmen = parse_pmen(pmen, cdc, pbp_patterns)

    cdc_X, cdc_y = format_inputs(cdc, pbp_seqs)
    X_train, X_test, y_train, y_test = train_test_split(
        cdc_X, cdc_y, test_size=0.33, random_state=0
    )
    X_validate, y_validate = format_inputs(pmen, pbp_seqs)

    logging.info(
        "Constructing gram matrices for train test and validation data"
    )
    gram_train = blosum_kernel(X_train, X_train)
    gram_test = blosum_kernel(X_test, X_train)
    gram_validate = blosum_kernel(X_validate, X_train)

    logging.info("Optimising SVR hyperparameters")
    best_hps = optimise_hps((gram_train, y_train), (gram_test, y_test))

    logging.info("Fitting model with optimised hyperparameters")
    model = fit_model((gram_train, y_train), **best_hps)


if __name__ == "__main__":
    main()
