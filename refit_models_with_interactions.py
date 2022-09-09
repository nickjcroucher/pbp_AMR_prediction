import logging
import os
import pickle
from functools import lru_cache
from math import ceil
from random import choice
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nptyping import NDArray
from scipy.sparse import csr_matrix
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from bayesian_interaction_model import BayesianLinearModel
from fit_models import fit_model, load_and_format_data, optimise_hps
from model_analysis.interrogate_rf import load_model
from model_analysis.parse_random_forest import (
    DecisionTree_,
    valid_feature_pair,
)
from utils import ResultsContainer, accuracy, mean_acc_per_bin


def plot_interactions(model: Lasso, interactions: List[Tuple[int, int]]):
    # protein lengths for plotting
    a1_length = 277
    b2_length = 278

    non_zero_coef = np.where(model.coef_ != 0)[0]
    lasso_coefs = model.coef_[non_zero_coef]
    interactions_array = np.array(interactions)
    interacting_loci = interactions_array[non_zero_coef]
    interacting_loci = [sorted(i) for i in interacting_loci]
    loci = set([i[0] for i in interacting_loci] + [i[1] for i in interacting_loci])

    def get_protein_from_feature_loc(i):
        i = ceil((i + 1) / 20)
        if i <= a1_length:
            return 0
        elif i > a1_length + b2_length:
            return 2
        else:
            return 1

    loci_mapper = {i: get_protein_from_feature_loc(i) for i in loci}

    interaction_counts = np.zeros((3, 3))
    interaction_sum = np.zeros((3, 3))
    mask = np.zeros((3, 3))
    mask[0, 1:] = True
    mask[1, 2] = True
    for i, pair in enumerate(interacting_loci):
        n = loci_mapper[pair[0]]
        m = loci_mapper[pair[1]]
        interaction_counts[m, n] += 1
        interaction_sum[m, n] += abs(lasso_coefs[i])

    interaction_sum = pd.DataFrame(
        interaction_sum, columns=["a1", "b2", "x2"], index=["a1", "b2", "x2"]
    )
    interaction_counts = pd.DataFrame(
        interaction_counts,
        columns=["a1", "b2", "x2"],
        index=["a1", "b2", "x2"],
    )

    plt.clf()
    sns.heatmap(interaction_counts, mask=mask, cmap="crest", annot=True)
    plt.title("#Interactions")
    plt.ylabel("PBP type")
    plt.xlabel("PBP type")
    plt.tight_layout()
    plt.savefig("interaction_counts_heatmap.png")

    plt.clf()
    sns.heatmap(interaction_sum, mask=mask, cmap="crest", annot=True, fmt=".3")
    plt.title("Sum of interaction lasso coefficients")
    plt.ylabel("PBP type")
    plt.xlabel("PBP type")
    plt.tight_layout()
    plt.savefig("interaction_sum_heatmap.png")


@lru_cache(maxsize=1)
def get_included_features(**kwargs):
    model = load_model(**kwargs)

    # extract each decision tree from the rf
    trees = [DecisionTree_(dt) for dt in model.estimators_]

    # get all the features which were included in the model
    included_features = np.unique(
        np.concatenate([tree.internal_node_features for tree in trees])
    )

    return included_features


def simulate_random_interactions(n: int, **kwargs) -> List[Tuple[int, int]]:
    included_features = get_included_features(**kwargs)
    feature_pairs: List[Tuple[int, int]] = []
    while len(feature_pairs) < n:
        fp = (choice(included_features), choice(included_features))
        if valid_feature_pair(*fp):
            feature_pairs.append(fp)

    return feature_pairs


def random_interaction_model_fits(
    n: int,
    train: Tuple[csr_matrix, pd.Series],
    test: Tuple[csr_matrix, pd.Series],
    load_model_kwargs: Dict,
    load_data_kwargs: Dict,
) -> float:
    interactions = simulate_random_interactions(n, **load_model_kwargs)

    data = load_and_format_data(interactions=interactions, **load_data_kwargs)
    train, test = [data[k] for k in ["train", "val"]]

    model = LinearRegression().fit(train[0], train[1])
    test_predictions = model.predict(test[0])

    return mean_squared_error(test[1], test_predictions)


def plot_simulations(
    n_interactions: int,
    test_data_mse: int,
    train: Tuple[csr_matrix, pd.Series],
    test: Tuple[csr_matrix, pd.Series],
    load_model_kwargs: Dict,
    load_data_kwargs: Dict,
):
    random_interaction_MSEs = [
        random_interaction_model_fits(
            n_interactions,
            train,
            test,
            load_model_kwargs=load_model_kwargs,
            load_data_kwargs=load_data_kwargs,
        )
        for i in tqdm(range(100), desc="Running random interaction models")
    ]

    plt.clf()
    sns.displot(random_interaction_MSEs)
    plt.title("MSE Distribution of models fitted to random interactions")
    plt.xlabel("MSE of LASSO model")
    plt.axvline(test_data_mse, dashes=(1, 1))
    plt.tight_layout()
    plt.savefig("histogram_simulated_interactions.png")

    plt.clf()
    sns.displot(random_interaction_MSEs, kind="kde")
    plt.title("Kernel Density Estimation of the PDF")
    plt.xlabel("MSE of LASSO model")
    plt.axvline(test_data_mse, dashes=(1, 1))
    plt.tight_layout()
    plt.savefig("KDE_simulated_interactions.png")

    plt.clf()
    sns.displot(random_interaction_MSEs, kind="ecdf")
    plt.title("Empirical CDF")
    plt.xlabel("MSE of LASSO model")
    plt.axvline(test_data_mse, dashes=(1, 1))
    plt.tight_layout()
    plt.savefig("CDF_simulated_interactions.png")
    plt.clf()


def compare_interaction_model_with_rf(results: ResultsContainer):
    model = load_model()
    testing_data = load_data("pmen", interactions=None, blosum_inference=True)[1]
    rf_predictions = model.predict(testing_data[0])

    plt.clf()
    sns.kdeplot(testing_data[1], label="Testing Data")
    sns.kdeplot(rf_predictions, label="RF Predictions")
    sns.kdeplot(results.testing_predictions, label="Interaction Model Predictions")
    plt.legend()
    plt.xlabel("Log2(MIC)")
    plt.title("RF vs Lasso Interaction Model")
    plt.tight_layout()
    plt.savefig("RF_vs_lasso_interaction_model_predictions.png")


def filter_lasso_features(data: NDArray, model: Lasso) -> NDArray:
    non_zero_coef = np.where(model.coef_ != 0)[0]
    return data[:, non_zero_coef]


def bayesian_linear_model(
    training_features: csr_matrix, training_labels: pd.Series, **kwargs
):
    """
    **kwargs are passed to the fitting method of BayesianLinearModel
    """
    bayesian_lm = BayesianLinearModel(training_features, training_labels.values)
    bayesian_lm.fit(**kwargs)
    return bayesian_lm


def CI_accuracy(
    predictions: NDArray,
    labels: NDArray,
    CI: List[Union[int, float]] = [5, 95],
) -> pd.DataFrame:
    if isinstance(labels, pd.Series):
        labels = labels.values  # will break if given numpy array

    percentiles = predictions.apply(np.percentile, q=CI)
    df = pd.DataFrame(
        {
            "p_lower": percentiles.iloc[0],
            "p_upper": percentiles.iloc[1],
            "truth": labels,
        }
    )
    df["within_CI"] = (df.truth >= df.p_lower) & (df.truth <= df.p_upper)
    return df


def plot_CI_accuracies(
    train_bayes_predictions: pd.DataFrame,
    train_labels: NDArray,
    test_bayes_predictions: pd.DataFrame,
    test_labels: NDArray,
    pmen_bayes_predictions: pd.DataFrame,
    pmen_labels: NDArray,
    maela_bayes_predictions: pd.DataFrame,
    maela_labels: NDArray,
    outdir: str,
):
    """
    For each population plot the percentage of predictions which are correct
    within a range of credible intervals.
    """
    preds_and_labels = {
        "train": [train_bayes_predictions, train_labels],
        "test": [test_bayes_predictions, test_labels],
        "pmen": [pmen_bayes_predictions, pmen_labels],
        "maela": [maela_bayes_predictions, maela_labels],
    }

    # 95% to 5% CI
    CIs = [[0 + (i / 10), 100 - (i / 10)] for i in list(range(25, 500, 25))]

    CI_accuracies_per_pop = {}
    for k, v in preds_and_labels.items():
        accuracy_dfs = [CI_accuracy(*v, CI=CI) for CI in CIs]  # type: ignore
        accuracies = [
            len(df.loc[df["within_CI"]]) / v[1].shape[0] for df in accuracy_dfs
        ]
        CI_accuracies_per_pop[k] = accuracies

    df = pd.DataFrame(CI_accuracies_per_pop)
    df = pd.melt(df, value_vars=df.columns)
    df.value = df.value * 100
    CI_intervals = [np.diff(i)[0] for i in CIs]
    df["CI"] = CI_intervals * 4

    plt.clf()
    sns.set_style("whitegrid")
    sns.scatterplot(data=df, x="CI", y="value", hue="variable", legend=False)
    sns.lineplot(data=df, x="CI", y="value", hue="variable")
    plt.xlabel("Credible Interval")
    plt.ylabel("Percentile of Correct Predictions")
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.plot(CI_intervals, CI_intervals, "k--")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "model_prediction_accuracies.png"))


def combine_datasets(
    data_1: List[Union[csr_matrix, pd.Series]],
    data_2: List[Union[csr_matrix, pd.Series]],
) -> List[Union[csr_matrix, pd.Series]]:
    assert (data_1[1] == data_2[1]).all(), "labels do not match"

    features_1 = data_1[0].todense()
    features_2 = data_2[0].todense()
    combined_features = csr_matrix(np.concatenate((features_1, features_2), axis=1))

    return [combined_features, data_1[1]]


def main(
    train_data_population: str = "cdc",
    test_data_population_1: str = "pmen",
    test_data_population_2: str = "maela",
    blosum_inference: bool = False,
    HMM_MIC_inference: bool = False,
    standardise_training_MIC: bool = True,
    standardise_test_and_val_MIC: bool = False,
    retain_full_sequences: bool = False,
):

    model_type = "lasso"
    pbounds = {"alpha": [0.05, 1.95]}

    if blosum_inference:
        inference_method = "blosum_inferred"
    elif HMM_MIC_inference:
        inference_method = "HMM_MIC_inferred"
    else:
        inference_method = "no_inference"

    logging.info("Loading inferred interaction data")
    with open(
        f"results/intermediates/{train_data_population}/{inference_method}_paired_sf_p_values_test1_{test_data_population_1}.pkl",
        "rb",
    ) as a:
        paired_sf_p_values = pickle.load(a)

    interactions = [i[0] for i in paired_sf_p_values if i[1] < 0.05]

    data = load_and_format_data(
        train_data_population,
        test_data_population_1,
        test_data_population_2,
        interactions=interactions,
        blosum_inference=blosum_inference,
        HMM_inference=False,
        HMM_MIC_inference=HMM_MIC_inference,
        filter_unseen=False,
        include_HMM_scores=False,
        just_HMM_scores=False,
        standardise_training_MIC=standardise_training_MIC,
        standardise_test_and_val_MIC=standardise_test_and_val_MIC,
    )

    if retain_full_sequences:
        data2 = load_and_format_data(
            train_data_population,
            test_data_population_1,
            test_data_population_2,
            interactions=None,
            blosum_inference=blosum_inference,
            HMM_inference=False,
            HMM_MIC_inference=HMM_MIC_inference,
            filter_unseen=False,
            include_HMM_scores=False,
            just_HMM_scores=False,
            standardise_training_MIC=standardise_training_MIC,
            standardise_test_and_val_MIC=standardise_test_and_val_MIC,
        )
        for k in data.keys():
            data[k] = combine_datasets(data[k], data2[k])

    train = data["train"]
    test = data["test_1"]
    test_2 = data["test_2"]
    val = data["val"]

    logging.info("Optimising the model for the test data accuracy")
    optimizer = optimise_hps(train, test, pbounds, model_type)  # select hps using GP

    logging.info(
        f"Fitting model with optimal hyperparameters: {optimizer.max['params']}"  # noqa: E501
    )
    model = fit_model(
        train, model_type=model_type, **optimizer.max["params"]
    )  # get best model fit

    train_predictions = model.predict(train[0])
    test_predictions_1 = model.predict(test[0])
    test_predictions_2 = model.predict(test_2[0])
    validate_predictions = model.predict(val[0])

    results = ResultsContainer(  # noqa: F841
        training_predictions=train_predictions,
        validation_predictions=validate_predictions,
        testing_predictions_1=test_predictions_1,
        testing_predictions_2=test_predictions_2,
        training_MSE=mean_squared_error(data["train"][1], train_predictions),
        validation_MSE=mean_squared_error(data["val"][1], validate_predictions),
        testing_MSE_1=mean_squared_error(data["test_1"][1], test_predictions_1),
        testing_MSE_2=mean_squared_error(data["test_2"][1], test_predictions_2),
        training_accuracy=accuracy(train_predictions, data["train"][1]),
        validation_accuracy=accuracy(validate_predictions, data["val"][1]),
        testing_accuracy_1=accuracy(test_predictions_1, data["test_1"][1]),
        testing_accuracy_2=accuracy(test_predictions_2, data["test_2"][1]),
        training_mean_acc_per_bin=mean_acc_per_bin(train_predictions, data["train"][1]),
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
            "HMM-MIC_inference": HMM_MIC_inference,
            "filter_unseen": False,
            "standardise_training_MIC": True,
            "train_val_population": train_data_population,
            "test_1_population": test_data_population_1,
            "test_2_population": test_data_population_2,
        },
    )

    print(results)

    suffix = ""
    fname = f"results/interaction_models/train_pop_{train_data_population}_results_{inference_method}_pbp_types{suffix}.pkl"
    if os.path.isfile(fname):
        suffix = "(1)"
        fname = f"results/interaction_models/train_pop_{train_data_population}_results_{inference_method}_pbp_types{suffix}.pkl"
    if os.path.isfile(fname):
        raise Exception("Results file already exists")

    with open(fname, "wb") as a:
        pickle.dump(results, a)

    load_model_kwargs = {
        "train_pop": train_data_population,
        "test_1_pop": test_data_population_1,
        "inference_type": inference_method,
    }
    load_data_kwargs = {
        "train_data_population": train_data_population,
        "test_data_population_1": test_data_population_1,
        "test_data_population_2": test_data_population_2,
        "blosum_inference": blosum_inference,
        "HMM_inference": False,
        "HMM_MIC_inference": HMM_MIC_inference,
        "filter_unseen": False,
        "include_HMM_scores": False,
        "just_HMM_scores": False,
        "standardise_training_MIC": standardise_training_MIC,
        "standardise_test_and_val_MIC": standardise_test_and_val_MIC,
    }

    plot_simulations(
        results.model.sparse_coef_.count_nonzero(),
        results.testing_MSE_1,
        train,
        test,
        load_model_kwargs=load_model_kwargs,
        load_data_kwargs=load_data_kwargs,
    )
    # compare_interaction_model_with_rf(results)
    plot_interactions(results.model, interactions)

    # # filter the features to get only those which are used in the lasso model
    # training_features = filter_lasso_features(train[0].todense(), results.model)
    # val_features = filter_lasso_features(val[0].todense(), results.model)
    # test_1_features = filter_lasso_features(test[0].todense(), results.model)
    # test_2_features = filter_lasso_features(test_2[0].todense(), results.model)

    # # fit bayesian model
    # bayesian_lm = bayesian_linear_model(training_features, train[1])
    # bayesian_lm.plot_model_fit()

    # # distributional predictions for each dataset
    # train_bayes_predictions = bayesian_lm.predict(training_features)
    # val_bayes_predictions = bayesian_lm.predict(val_features)
    # test_1_bayes_predictions = bayesian_lm.predict(test_1_features)
    # test_2_bayes_predictions = bayesian_lm.predict(test_2_features)

    # plot_CI_accuracies(
    #     train_bayes_predictions,
    #     train[1].values,
    #     val_bayes_predictions,
    #     val[1].values,
    #     test_1_bayes_predictions,
    #     test[1].values,
    #     test_2_bayes_predictions,
    #     test_2[1].values,
    #     bayesian_lm.output_dir,
    # )


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)

    main()
