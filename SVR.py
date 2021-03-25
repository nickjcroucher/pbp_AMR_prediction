import logging
import pickle
from typing import List, Tuple, Dict
from functools import partial
from math import log10

import pandas as pd
import numpy as np
from nptyping import NDArray, Object, Int
from Bio.SubsMat.MatrixInfo import blosum62
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import seaborn as sns

from parse_pbp_data import parse_cdc, parse_pmen
from utils import accuracy, ResultsContainer


logging.basicConfig()
logging.root.setLevel(logging.INFO)


def format_inputs(
    data: pd.DataFrame, pbp_seqs: List[str]
) -> Tuple[NDArray, NDArray]:
    X = data[pbp_seqs].to_numpy()
    Y = data.log2_mic.to_numpy()
    return X, Y


def blosum_kernel(X: NDArray[Object], Y: NDArray[Object]) -> NDArray[Int, Int]:
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
    model = fit_model(train, 10 ** C, 10 ** epsilon)

    test_predictions = model.predict(test[0])
    MSE = mean_squared_error(test[1], test_predictions)
    return -MSE  # negative error, optimiser will maximise


def optimise_hps(
    train: Tuple[NDArray, NDArray],
    test: Tuple[NDArray, NDArray],
    pbounds: Dict[str, Tuple[float, float]],
) -> Tuple[Dict[str, float], BayesianOptimization]:
    partial_fitting_function = partial(train_evaluate, train=train, test=test)

    optimizer = BayesianOptimization(
        f=partial_fitting_function, pbounds=pbounds, random_state=0
    )
    optimizer.maximize(n_iter=20)

    best_hyperparams = {
        "C": 10 ** optimizer.max["params"]["C"],
        "epsilon": 10 ** optimizer.max["params"]["epsilon"],
    }

    return best_hyperparams, optimizer


def plot_hps(
    optimizer: BayesianOptimization, pbounds: Dict[str, Tuple[float, float]]
):
    df = pd.DataFrame(
        {
            "target": [i["target"] for i in optimizer.res],
            "C": [i["params"]["C"] for i in optimizer.res],
            "epsilon": [i["params"]["epsilon"] for i in optimizer.res],
        }
    )

    # returns function which fits a 2d linear spline which estimates value of
    # target given C and epsilon
    interpolator = interp2d(df.C, df.epsilon, df.target, kind="linear")

    x_coords = np.arange(df.C.min(), df.C.max() + 1)
    y_coords = np.arange(df.epsilon.min(), df.epsilon.max() + 1)
    target_interpolated = pd.DataFrame(
        interpolator(x_coords, y_coords), columns=x_coords, index=y_coords
    )

    plt.clf()
    sns.heatmap(target_interpolated)
    plt.xlabel("C")
    plt.ylabel("epsilon")
    plt.show()


def main():
    logging.info("Reading and formatting inputs")

    cdc = pd.read_csv("../data/pneumo_pbp/cdc_seqs_df.csv")
    pmen = pd.read_csv("../data/pneumo_pbp/pmen_pbp_profiles_extended.csv")

    pbp_patterns = ["a1", "b2", "x2"]
    pbp_seqs = [i + "_seq" for i in pbp_patterns]

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
    # evaluating kernel is slowest part of training process so do it once here
    # and use gram matrix to fit model
    gram_train = blosum_kernel(X_train, X_train)
    gram_test = blosum_kernel(X_test, X_train)
    gram_validate = blosum_kernel(X_validate, X_train)

    logging.info("Optimising SVR hyperparameters")
    pbounds = {
        "C": (log10(1e-12), log10(1e-6)),
        "epsilon": (log10(1e-10), log10(1)),
    }  # log uniform distribution
    best_hps, optimizer = optimise_hps(
        (gram_train, y_train), (gram_test, y_test), pbounds
    )
    plot_hps(optimizer, pbounds)

    logging.info("Fitting model with optimised hyperparameters")
    model = fit_model((gram_train, y_train), **best_hps)

    train_predictions = model.predict(gram_train)
    test_predictions = model.predict(gram_test)
    validate_predictions = model.predict(gram_validate)

    results = ResultsContainer(
        training_predictions=train_predictions,
        testing_predictions=test_predictions,
        validation_predictions=validate_predictions,
        training_MSE=mean_squared_error(y_train, train_predictions),
        testing_MSE=mean_squared_error(y_test, test_predictions),
        validation_MSE=mean_squared_error(y_validate, validate_predictions),
        training_accuracy=accuracy(train_predictions, y_train),
        testing_accuracy=accuracy(test_predictions, y_test),
        validation_accuracy=accuracy(validate_predictions, y_validate),
        hyperparameters=best_hps,
        model_type="SVR",
        model=model,
    )

    with open("Results/SVR_results.pkl", "wb") as a:
        pickle.dump(results, a)


if __name__ == "__main__":
    main()
