import logging
import warnings
from functools import partial
from typing import Tuple, Dict, Union

import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
from bayes_opt import BayesianOptimization
from nptyping import NDArray
from scipy.sparse import csr_matrix

from parse_pbp_data import (
    parse_cdc,
    parse_pmen,
    encode_sequences,
    build_co_occurrence_graph,
)
from utils import accuracy, mean_acc_per_bin, ResultsContainer


logging.basicConfig()
logging.root.setLevel(logging.INFO)


def _fit_en(
    train: Tuple[Union[csr_matrix, NDArray], NDArray],
    max_iter: int,
    l1_ratio: float = 0.5,
    alpha: float = 1.0,
) -> ElasticNet:
    reg = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=0,
        max_iter=max_iter,
    )
    reg.fit(train[0], train[1])
    return reg


def _fit_lasso(
    train: Tuple[Union[csr_matrix, NDArray], NDArray],
    max_iter: int,
    alpha: float = 1.0,
) -> Lasso:
    reg = Lasso(alpha=alpha, random_state=0, max_iter=max_iter)
    reg.fit(train[0], train[1])
    return reg


def fit_model(
    train: Tuple[Union[csr_matrix, NDArray], NDArray],
    model_type: str,
    **kwargs,
) -> ElasticNet:
    max_iter = 100000
    fitted = False
    while not fitted:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            if model_type == "elastic_net":
                reg = _fit_en(train, max_iter, **kwargs)
            elif model_type == "lasso":
                reg = _fit_lasso(train, max_iter, **kwargs)
            else:
                raise NotImplementedError(model_type)

            if len(w) > 1:
                for warning in w:
                    logging.error(warning.category)
                raise Exception
            elif w and issubclass(w[0].category, ConvergenceWarning):
                logging.warning(
                    f"Failed to converge with max_iter = {max_iter}, "
                    + "adding 100000 more"
                )
                max_iter += 100000
            else:
                fitted = True

    return reg


def train_evaluate(
    train: Tuple[Union[NDArray, csr_matrix], NDArray],
    test: Tuple[Union[NDArray, csr_matrix], NDArray],
    model_type: str,
    **kwargs,
) -> float:
    reg = fit_model(train, model_type, **kwargs)
    MSE = mean_squared_error(test[1], reg.predict(test[0]))
    return -MSE


def optimise_hps(
    train: Tuple[Union[NDArray, csr_matrix], NDArray],
    test: Tuple[Union[NDArray, csr_matrix], NDArray],
    pbounds: Dict[str, Tuple[float, float]],
    model_type: str = "elastic_net",
) -> BayesianOptimization:
    partial_fitting_function = partial(
        train_evaluate, train=train, test=test, model_type=model_type
    )

    optimizer = BayesianOptimization(
        f=partial_fitting_function, pbounds=pbounds, random_state=0
    )
    optimizer.maximize(n_iter=10)

    return optimizer


def normed_laplacian(adj: csr_matrix, deg: csr_matrix) -> csr_matrix:
    deg_ = deg.power(-0.5)
    return deg_ * adj * deg_


def load_data(adj_convolution: bool, laplacian_convolution: bool):
    if adj_convolution is True and laplacian_convolution is True:
        raise ValueError(
            "Only one of adj_convolution or laplacian_convolution can be applied"
        )

    cdc = pd.read_csv("../data/pneumo_pbp/cdc_seqs_df.csv")
    pmen = pd.read_csv("../data/pneumo_pbp/pmen_pbp_profiles_extended.csv")

    pbp_patterns = ["a1", "b2", "x2"]

    cdc = parse_cdc(cdc, pbp_patterns)
    pmen = parse_pmen(pmen, cdc, pbp_patterns)

    cdc_encoded_sequences = encode_sequences(cdc, pbp_patterns)
    pmen_encoded_sequences = encode_sequences(pmen, pbp_patterns)

    if adj_convolution:
        logging.info("Applying graph convolution")
        cdc_adj = build_co_occurrence_graph(cdc, pbp_patterns)[0]
        pmen_adj = build_co_occurrence_graph(pmen, pbp_patterns)[0]

        cdc_convolved_sequences = cdc_adj * cdc_encoded_sequences
        pmen_convolved_sequences = pmen_adj * pmen_encoded_sequences

    elif laplacian_convolution:
        logging.info("Applying graph convolution")
        cdc_adj, cdc_deg = build_co_occurrence_graph(cdc, pbp_patterns)
        pmen_adj, pmen_deg = build_co_occurrence_graph(pmen, pbp_patterns)

        cdc_laplacian = normed_laplacian(cdc_adj, cdc_deg)
        pmen_laplacian = normed_laplacian(pmen_adj, pmen_deg)

        cdc_convolved_sequences = cdc_laplacian * cdc_encoded_sequences
        pmen_convolved_sequences = pmen_laplacian * pmen_encoded_sequences
        X_train, X_test, y_train, y_test = train_test_split(
            cdc_convolved_sequences,
            cdc.log2_mic,
            test_size=0.33,
            random_state=0,
        )
        X_validate, y_validate = pmen_convolved_sequences, pmen.log2_mic
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            cdc_encoded_sequences, cdc.log2_mic, test_size=0.33, random_state=0
        )
        X_validate, y_validate = pmen_encoded_sequences, pmen.log2_mic

    return (X_train, y_train), (X_test, y_test), (X_validate, y_validate)


def main():
    model_type = "elastic_net"

    logging.info("Loading data")
    train, test, validate = load_data(
        adj_convolution=False, laplacian_convolution=True
    )

    logging.info("Optimising the model for the test data accuracy")
    if model_type == "elastic_net":
        pbounds = {"l1_ratio": [0.3, 0.7], "alpha": [0.5, 1.5]}
    elif model_type == "lasso":
        pbounds = {"alpha": [0.5, 1.5]}
    else:
        raise NotImplementedError(model_type)
    optimizer = optimise_hps(train, test, pbounds, model_type)  # select hps using GP

    logging.info(
        f"Fitting model with optimal hyperparameters: {optimizer.max['params']}"
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
        model_type="elastic_net",
        model=model,
    )
