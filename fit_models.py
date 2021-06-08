#!/usr/bin/env python3

import argparse
import logging
import os
import pickle
import warnings
from functools import partial
from math import log10
from typing import Dict, Tuple, Union

import numpy as np
from bayes_opt import BayesianOptimization
from nptyping import NDArray
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_squared_error

from data_preprocessing.parse_pbp_data import encode_sequences
from model_analysis.parse_random_forest import DecisionTree_
from models.supervised_models import _fit_en, _fit_lasso, _fit_rf
from models.unsupervised_models import _fit_DBSCAN, _fit_DBSCAN_with_UMAP
from utils import (
    accuracy,
    load_data,
    mean_acc_per_bin,
    ResultsContainer,
)


def fit_model(
    train: Tuple[Union[csr_matrix, NDArray], NDArray],
    model_type: str,
    **kwargs,
) -> Union[ElasticNet, Lasso, RandomForestRegressor]:
    if model_type == "random_forest":
        reg = _fit_rf(train, **kwargs)

    elif model_type == "DBSCAN":
        reg = _fit_DBSCAN(train, **kwargs)

    elif model_type == "DBSCAN_with_UMAP":
        reg = _fit_DBSCAN_with_UMAP(train, **kwargs)

    else:
        # lasso and en models require iterative fitting
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
    return -MSE  # pylint: disable=invalid-unary-operand-type


def optimise_hps(
    train: Tuple[Union[NDArray, csr_matrix], NDArray],
    test: Tuple[Union[NDArray, csr_matrix], NDArray],
    pbounds: Dict[str, Tuple[float, float]],
    model_type: str,
    init_points: int = 5,
    n_iter: int = 10,
) -> BayesianOptimization:
    partial_fitting_function = partial(
        train_evaluate, train=train, test=test, model_type=model_type
    )

    optimizer = BayesianOptimization(
        f=partial_fitting_function, pbounds=pbounds, random_state=0
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    return optimizer


def filter_features_by_previous_model_fit(
    model_path: str,
    training_features: csr_matrix,
    validation_features: csr_matrix,
    testing_features_1: csr_matrix,
    testing_features_2: csr_matrix,
) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:

    with open(model_path, "rb") as a:
        original_model = pickle.load(a)
    if isinstance(original_model, ResultsContainer):
        original_model = original_model.model
    elif not isinstance(original_model, RandomForestRegressor):
        raise TypeError(f"Unknown input of type {type(original_model)}")

    # extract each decision tree from the rf
    trees = [DecisionTree_(dt) for dt in original_model.estimators_]

    # get all the features which were included in the model
    included_features = np.unique(
        np.concatenate([tree.internal_node_features for tree in trees])
    )

    filtered_features = []
    for features in [
        training_features,
        validation_features,
        testing_features_1,
        testing_features_2,
    ]:
        features = features.todense()
        filtered_features.append(csr_matrix(features[:, included_features]))

    return tuple(filtered_features)  # type: ignore


def load_and_format_data(
    train_data_population,
    test_data_population_1,
    test_data_population_2,
    *,
    interactions: Tuple[Tuple[int]] = None,
    blosum_inference: bool = False,
    filter_unseen: bool = True,
    standardise_training_MIC: bool = False,
    standardise_test_and_val_MIC: bool = False,
) -> Dict:

    pbp_patterns = ["a1", "b2", "x2"]

    train, test_1, test_2, val = load_data(
        train_data_population,
        test_data_population_1,
        test_data_population_2,
        interactions,
        blosum_inference,
        filter_unseen,
        standardise_training_MIC,
        standardise_test_and_val_MIC,
    )

    train_encoded_sequences = encode_sequences(train, pbp_patterns)
    test_1_encoded_sequences = encode_sequences(test_1, pbp_patterns)
    test_2_encoded_sequences = encode_sequences(test_2, pbp_patterns)
    val_encoded_sequences = encode_sequences(val, pbp_patterns)

    X_train, y_train = train_encoded_sequences, train.log2_mic
    X_test_1, y_test_1 = test_1_encoded_sequences, test_1.log2_mic
    X_test_2, y_test_2 = test_2_encoded_sequences, test_2.log2_mic
    X_validate, y_validate = val_encoded_sequences, val.log2_mic

    def interact(data, interacting_features):
        interacting_features = np.concatenate(
            [np.multiply(data[:, i[0]], data[:, i[1]]) for i in interactions],
            axis=1,
        )
        return csr_matrix(interacting_features)

    if interactions is not None:
        X_train = interact(X_train.todense(), interactions)
        X_test_1 = interact(X_test_1.todense(), interactions)
        X_test_2 = interact(X_test_2.todense(), interactions)
        X_validate = interact(X_validate.todense(), interactions)

    class DataList(list):
        def __init__(self, population: str):
            self.population = population

    data_dictionary = {
        "train": DataList(train_data_population),
        "val": DataList(train_data_population),
        "test_1": DataList(test_data_population_1),
        "test_2": DataList(test_data_population_2),
    }

    data_dictionary["train"].extend([X_train, y_train])
    data_dictionary["val"].extend([X_validate, y_validate])
    data_dictionary["test_1"].extend([X_test_1, y_test_1])
    data_dictionary["test_2"].extend([X_test_2, y_test_2])

    return data_dictionary


def save_output(results: ResultsContainer, filename: str, outdir: str):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # dont overwrite existing results file
    file_path = os.path.join(outdir, filename)
    i = 1
    while os.path.isfile(file_path):
        split_path = file_path.split(".")
        path_minus_ext = "".join(split_path[:-1])
        if i > 1:
            path_minus_ext = path_minus_ext[:-3]
        ext = split_path[-1]
        file_path = path_minus_ext + f"({i})." + ext
        i += 1

    with open(file_path, "wb") as a:
        pickle.dump(results, a)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit predictive model of penicilin MIC to PBP amino acid sequence"  # noqa: E501
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
        "--model_type",
        type=str,
        help="random_forest, elastic_net, lasso, DBSCAN, or DBSCAN_with_UMAP",
    )
    parser.add_argument(
        "--blosum_inference",
        type=bool,
        default=True,
        help="If blosum inference is not conducted on then PBP types not found in the train data will be filtered from the test data",  # noqa: E501
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
    parser.add_argument(
        "--previous_rf_model",
        type=str,
        default=None,
        help="Used to filter features",
    )

    return vars(parser.parse_args())  # return as a dictionary


def main(
    train_data_population="cdc",
    test_data_population_1="pmen",
    test_data_population_2="maela",
    model_type="random_forest",
    blosum_inference=True,
    standardise_training_MIC=True,
    standardise_test_and_val_MIC=False,
    previous_rf_model=None,
):

    logging.info("Loading data")
    data = load_and_format_data(
        train_data_population,
        test_data_population_1,
        test_data_population_2,
        blosum_inference=blosum_inference,
        filter_unseen=not blosum_inference,
        standardise_training_MIC=standardise_training_MIC,
        standardise_test_and_val_MIC=standardise_test_and_val_MIC,
    )

    # filter features by things which have been used by previously fitted model
    if previous_rf_model is not None:
        filtered_features = filter_features_by_previous_model_fit(
            previous_rf_model,
            data["train"][0],
            data["val"][0],
            data["test_1"][0],
            data["test_2"][0],
        )
        data["train"][0] = filtered_features[0]
        data["val"][0] = filtered_features[1]
        data["test_1"][0] = filtered_features[2]
        data["test_2"][0] = filtered_features[3]

    logging.info("Optimising the model for the test data accuracy")
    if model_type == "elastic_net":
        pbounds = {"l1_ratio": [0.05, 0.95], "alpha": [0.05, 1.95]}
    elif model_type == "lasso":
        pbounds = {"alpha": [0.05, 1.95]}
    elif model_type == "random_forest":
        pbounds = {
            "n_estimators": [1000, 10000],
            "max_depth": [2, 5],
            "min_samples_split": [2, 10],
            "min_samples_leaf": [2, 2],
        }
    elif model_type == "DBSCAN":
        pbounds = {
            "log_eps": [log10(0.0001), log10(0.1)],
            "min_samples": [2, 20],
        }
    elif model_type == "DBSCAN_with_UMAP":
        pbounds = {
            "log_eps": [log10(0.1), log10(10)],
            "min_samples": [2, 20],
            "umap_components": [2, 15],
        }
    else:
        raise NotImplementedError(model_type)

    optimizer = optimise_hps(
        data["train"], data["test_1"], pbounds, model_type
    )  # select hps using GP

    logging.info(
        f"Fitting model with optimal hyperparameters: {optimizer.max['params']}"  # noqa: E501
    )
    model = fit_model(
        data["train"], model_type=model_type, **optimizer.max["params"]
    )  # get best model fit

    train_predictions = model.predict(data["train"][0])
    validate_predictions = model.predict(data["val"][0])
    test_predictions_1 = model.predict(data["test_1"][0])
    test_predictions_2 = model.predict(data["test_2"][0])

    results = ResultsContainer(  # noqa: F841
        training_predictions=train_predictions,
        validation_predictions=validate_predictions,
        testing_predictions_1=test_predictions_1,
        testing_predictions_2=test_predictions_2,
        training_MSE=mean_squared_error(data["train"][1], train_predictions),
        validation_MSE=mean_squared_error(
            data["val"][1], validate_predictions
        ),
        testing_MSE_1=mean_squared_error(
            data["test_1"][1], test_predictions_1
        ),
        testing_MSE_2=mean_squared_error(
            data["test_2"][1], test_predictions_2
        ),
        training_accuracy=accuracy(train_predictions, data["train"][1]),
        validation_accuracy=accuracy(validate_predictions, data["val"][1]),
        testing_accuracy_1=accuracy(test_predictions_1, data["test_1"][1]),
        testing_accuracy_2=accuracy(test_predictions_2, data["test_2"][1]),
        training_mean_acc_per_bin=mean_acc_per_bin(
            train_predictions, data["train"][1]
        ),
        validation_mean_acc_per_bin=mean_acc_per_bin(
            validate_predictions, data["val"][1]
        ),
        testing_mean_acc_per_bin_1=mean_acc_per_bin(
            test_predictions_1, data["test_1"][1]
        ),
        testing_mean_acc_per_bin_2=mean_acc_per_bin(
            test_predictions_2, data["test_2"][1]
        ),
        hyperparameters=optimizer.max["params"],
        model_type=model_type,
        model=model,
        config={
            "blosum_inference": blosum_inference,
            "filter_unseen": not blosum_inference,
            "standardise_training_MIC": standardise_training_MIC,
            "standardise_test_and_val_MIC": standardise_test_and_val_MIC,
            "previous_rf_model": previous_rf_model,
            "train_val_population": data["train"].population,
            "test_1_population": data["test_1"].population,
            "test_2_population": data["test_2"].population,
        },
    )

    print(results)

    outdir = f"results/{model_type}"
    if blosum_inference:
        filename = f"train_pop_{train_data_population}_results_blosum_inferred_pbp_types.pkl"  # noqa: E501
    else:
        filename = f"train_pop_{train_data_population}_results_filtered_pbp_types.pkl"  # noqa: E501
    save_output(results, filename, outdir)


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)

    main(**parse_args())
