#!/usr/bin/env python3

import argparse
import logging
import os
import pickle
import random
import warnings
from functools import partial
from math import log10
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from nptyping import NDArray
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_squared_error

from data_preprocessing.parse_pbp_data import encode_sequences
from model_analysis.parse_random_forest import DecisionTree_
from models.supervised_models import _fit_en, _fit_lasso, _fit_rf, _fit_ord_reg
from models.unsupervised_models import _fit_DBSCAN, _fit_DBSCAN_with_UMAP
from models.HMM_model import get_HMM_scores, ProfileHMMPredictor
from utils import (
    accuracy,
    load_data,
    load_extended_sequence_data,
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

    elif model_type == "bayesian_ord_reg":
        reg = _fit_ord_reg(train, **kwargs)

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
    pbounds: Dict[str, List[float]],
    model_type: str,
    init_points: int = 5,
    n_iter: int = 5,
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
    all_data: Dict[str, csr_matrix],
) -> Dict[str, csr_matrix]:

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

    return {
        k: [csr_matrix(v[0].todense()[:, included_features]), v[1]]
        for k, v in all_data.items()
    }


def load_and_format_data(
    train_data_population: str,
    test_data_population_1: str,
    test_data_population_2: Optional[str] = None,
    interactions: Union[List, Tuple[Tuple[int]]] = None,
    blosum_inference: bool = False,
    HMM_inference: bool = False,
    HMM_MIC_inference: bool = False,
    filter_unseen: bool = True,
    include_HMM_scores: bool = False,
    just_HMM_scores: bool = False,
    standardise_training_MIC: bool = False,
    standardise_test_and_val_MIC: bool = False,
    extended_sequences: bool = False,
) -> Dict:

    pbp_patterns = ["a1", "b2", "x2"]

    if extended_sequences and sorted(
        [train_data_population, test_data_population_1]
    ) != ["maela", "pmen"]:
        raise ValueError("Extended sequence data is only available for pmen and maela")

    if extended_sequences:
        train, test_1, val = load_extended_sequence_data(
            train_data_population=train_data_population,
            blosum_inference=blosum_inference,
            HMM_inference=HMM_inference,
            HMM_MIC_inference=HMM_MIC_inference,
            filter_unseen=filter_unseen,
            standardise_training_MIC=standardise_training_MIC,
            standardise_test_and_val_MIC=standardise_test_and_val_MIC,
        )
        original_datasets = {"train": train, "test_1": test_1, "val": val}
    else:
        train, test_1, test_2, val = load_data(
            train_data_population=train_data_population,
            test_data_population_1=test_data_population_1,
            test_data_population_2=test_data_population_2,  # type: ignore
            blosum_inference=blosum_inference,
            HMM_inference=HMM_inference,
            HMM_MIC_inference=HMM_MIC_inference,
            filter_unseen=filter_unseen,
            standardise_training_MIC=standardise_training_MIC,
            standardise_test_and_val_MIC=standardise_test_and_val_MIC,
        )
        original_datasets = {
            "train": train,
            "test_1": test_1,
            "test_2": test_2,
            "val": val,
        }

    datasets = {
        k: encode_sequences(v, pbp_patterns) for k, v in original_datasets.items()
    }

    if include_HMM_scores or just_HMM_scores:
        datasets = {k: v.todense() for k, v in datasets.items()}

        if just_HMM_scores:
            datasets = {k: np.zeros((v.shape[0], 1)) for k, v in datasets.items()}

        for pbp in pbp_patterns:
            if extended_sequences:
                (train_scores, test_1_scores, val_scores,) = get_HMM_scores(
                    ProfileHMMPredictor(train, [f"{pbp}_seq"]),
                    [f"{pbp}_seq"],
                    train,
                    test_1,
                    val,
                )
                hmm_scores = {
                    "train": train_scores,
                    "test_1": test_1_scores,
                    "val": val_scores,
                }
            else:
                (
                    train_scores,
                    test_1_scores,
                    test_2_scores,
                    val_scores,
                ) = get_HMM_scores(
                    ProfileHMMPredictor(train, [f"{pbp}_seq"]),
                    [f"{pbp}_seq"],
                    train,
                    test_1,
                    test_2,
                    val,
                )
                hmm_scores = {
                    "train": train_scores,
                    "test_1": test_1_scores,
                    "test_2": test_2_scores,
                    "val": val_scores,
                }
            datasets = {
                k: np.concatenate((v, hmm_scores[k]), axis=1)
                for k, v in datasets.items()
            }

        if just_HMM_scores:  # remove empty first column
            datasets = {k: v[:, 1:] for k, v in datasets.items()}

        datasets = {k: csr_matrix(v) for k, v in datasets.items()}

    datasets = {k: (v, original_datasets[k]["log2_mic"]) for k, v in datasets.items()}

    def interact(data, interacting_features):
        interacting_features = np.concatenate(
            [np.multiply(data[:, i[0]], data[:, i[1]]) for i in interactions],
            axis=1,
        )
        return csr_matrix(interacting_features)

    if interactions is not None:
        datasets = {
            k: [interact(v[0].todense(), interactions), v[1]]
            for k, v in datasets.items()
        }

    class DataList(list):
        def __init__(self, population: str):
            self.population = population

    data_dictionary = {
        "train": DataList(train_data_population),
        "val": DataList(train_data_population),
        "test_1": DataList(test_data_population_1),
    }
    if test_data_population_2 is not None:
        data_dictionary["test_2"] = DataList(test_data_population_2)

    data_dictionary["train"].extend(datasets["train"])
    data_dictionary["val"].extend(datasets["val"])
    data_dictionary["test_1"].extend(datasets["test_1"])
    try:
        data_dictionary["test_2"].extend(datasets["test_2"])
    except KeyError:
        pass

    return data_dictionary


def randomise_training_data(data: Dict) -> Dict:
    # TODO make this the same as mixed populations in GNN code

    X = np.concatenate([i[0].todense() for i in data.values()])
    y = pd.concat([i[1] for i in data.values()], ignore_index=True)
    X_and_y = np.concatenate((np.expand_dims(y, 1), X), axis=1)

    train_n = int(0.5 * y.shape[0])

    random.seed(0)
    random.shuffle(X_and_y)

    def extract_samples(n):
        global X_and_y
        out = X_and_y[:n]
        X_and_y = X_and_y[n:]
        return out

    train_data = extract_samples(train_n)
    val_data = extract_samples(int(X_and_y.shape[0] / 2))
    test_data_1 = X_and_y
    test_data_2 = test_data_2


def check_new_file_path(file_path: str) -> str:
    i = 1
    while os.path.isfile(file_path):
        split_path = file_path.split(".")
        path_minus_ext = "".join(split_path[:-1])
        if i > 1:
            path_minus_ext = path_minus_ext[:-3]  # remove brackets and number
        ext = split_path[-1]
        file_path = path_minus_ext + f"({i})." + ext
        i += 1
    return file_path


def save_output(results: ResultsContainer, filename: str, outdir: str):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    file_path = os.path.join(outdir, filename)
    # dont overwrite existing results file
    file_path = check_new_file_path(file_path)

    with open(file_path, "wb") as a:
        pickle.dump(results, a)


def parse_args() -> Dict:
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
        default=False,
        action="store_true",
        help="Use blosum matrix to infer closest sequences to unseen sequences in the test data",  # noqa: E501
    )
    parser.add_argument(
        "--HMM_MIC_inference",
        default=False,
        action="store_true",
        help="Use HMM fitted to sequences of each MIC to infer closest sequences to unseen sequences in the test data",  # noqa: E501
    )
    parser.add_argument(
        "--HMM_inference",
        default=False,
        action="store_true",
        help="Use HMM fitted to all training data to infer closest sequences to unseen sequences in the test data",  # noqa: E501
    )
    parser.add_argument(
        "--filter_unseen",
        default=False,
        action="store_true",
        help="Filter out the unseen samples in the testing data",
    )
    parser.add_argument(
        "--include_HMM_scores",
        default=False,
        action="store_true",
        help="Use HMM scores as additional features in the model",
    )
    parser.add_argument(
        "--just_HMM_scores",
        default=False,
        action="store_true",
        help="Use HMM scores as ONLY features in the model",
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
    parser.add_argument(
        "--extended_sequences",
        type=bool,
        default=False,
        help="Fit model to extended sequences",
    )

    return vars(parser.parse_args())  # return as a dictionary


def main(
    train_data_population: str = "cdc",
    test_data_population_1: str = "pmen",
    test_data_population_2: Optional[str] = "maela",
    randomise_populations: bool = False,
    model_type: str = "random_forest",
    blosum_inference: bool = False,
    HMM_inference: bool = False,
    HMM_MIC_inference: bool = False,
    filter_unseen: bool = False,
    include_HMM_scores: bool = False,
    just_HMM_scores: bool = False,
    standardise_training_MIC: bool = True,
    standardise_test_and_val_MIC: bool = False,
    previous_rf_model: str = None,
    extended_sequences: bool = False,
):

    logging.info("Loading data")
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

    if randomise_populations:
        ...

    # filter features by things which have been used by previously fitted model
    if previous_rf_model is not None:
        data = filter_features_by_previous_model_fit(previous_rf_model, data)

    logging.info("Optimising the model for the test data accuracy")
    if model_type == "elastic_net":
        pbounds = {"l1_ratio": [0.05, 0.95], "alpha": [0.05, 1.95]}
    elif model_type == "lasso":
        pbounds = {"alpha": [0.05, 1.95]}
    elif model_type == "random_forest":
        if just_HMM_scores:
            pbounds = {
                "n_estimators": [50, 500],
                "max_depth": [1, 5],
                "min_samples_split": [2, 10],
                "min_samples_leaf": [2, 6],
            }
        else:
            pbounds = {
                "n_estimators": [1000, 10000],
                "max_depth": [1, 20],
                "min_samples_split": [2, 10],
                "min_samples_leaf": [2, 6],
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
    test_predictions_2 = model.predict(data["test_2"][0]) if "test_2" in data else None

    results = ResultsContainer(
        training_predictions=train_predictions,
        validation_predictions=validate_predictions,
        testing_predictions_1=test_predictions_1,
        testing_predictions_2=test_predictions_2,
        training_MSE=mean_squared_error(data["train"][1], train_predictions),
        validation_MSE=mean_squared_error(data["val"][1], validate_predictions),
        testing_MSE_1=mean_squared_error(data["test_1"][1], test_predictions_1),
        testing_MSE_2=mean_squared_error(data["test_2"][1], test_predictions_2)
        if "test_2" in data
        else None,
        training_accuracy=accuracy(train_predictions, data["train"][1]),
        validation_accuracy=accuracy(validate_predictions, data["val"][1]),
        testing_accuracy_1=accuracy(test_predictions_1, data["test_1"][1]),
        testing_accuracy_2=accuracy(test_predictions_2, data["test_2"][1])
        if "test_2" in data
        else None,
        training_mean_acc_per_bin=mean_acc_per_bin(train_predictions, data["train"][1]),
        validation_mean_acc_per_bin=mean_acc_per_bin(
            validate_predictions, data["val"][1]
        ),
        testing_mean_acc_per_bin_1=mean_acc_per_bin(
            test_predictions_1, data["test_1"][1]
        ),
        testing_mean_acc_per_bin_2=mean_acc_per_bin(
            test_predictions_2, data["test_2"][1]
        )
        if "test_2" in data
        else None,
        hyperparameters=optimizer.max["params"],
        model_type=model_type,
        model=model,
        config={
            "blosum_inference": blosum_inference,
            "filter_unseen": filter_unseen,
            "hmm_inference": HMM_inference,
            "hmm_mic_inference": HMM_MIC_inference,
            "hmm_scores_as_features": include_HMM_scores,
            "hmm_scores_as_only_features": just_HMM_scores,
            "standardise_training_MIC": standardise_training_MIC,
            "standardise_test_and_val_MIC": standardise_test_and_val_MIC,
            "previous_rf_model": previous_rf_model,
            "extended_sequences": extended_sequences,
            "train_val_population": data["train"].population,
            "test_1_population": data["test_1"].population,
            "test_2_population": data["test_2"].population
            if "test_2" in data
            else None,
        },
    )

    print(results)

    if extended_sequences:
        outdir = f"results/extended_sequences/{model_type}"
    else:
        outdir = f"results/{model_type}"
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
    else:
        filename = f"train_pop_{train_data_population}_results_no_inference_pbp_types.pkl"  # noqa: E501
    save_output(results, filename, outdir)


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)

    main(**parse_args())
