import logging
import os
import pickle
from itertools import combinations
from operator import itemgetter
from typing import List, Tuple

import numpy as np
import pandas as pd
import ray
from nptyping import Int, NDArray
from scipy.sparse import csr_matrix
from scipy.stats import fisher_exact
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.multitest import multipletests

from models import load_data
from parse_random_forest import DecisionTree_, co_occuring_feature_pairs


def paired_selection_frequency(
    trees: List[DecisionTree_],
    included_features: NDArray[Int],
    multiple_test_correction: bool = True,
) -> List[Tuple[Tuple[int, int], float]]:
    """
    tree_features: List of arrays which are the features on the internal nodes
    of each tree
    included_features: List of all the features which are present in any tree
    multiple_test_correction: Use Bonferroni method to correct for multiple
    testing

    Returns: list of tuples containing a pair of features and the p-value of
    the fisher exact test which gives the probability of their paired selection
    frequency under the null hypothesis
    """
    # df showing which features are in which tree
    feature_tree_matches = {i: [] for i in included_features}
    for tree in trees:
        for i in included_features:
            feature_tree_matches[i].append(i in tree.internal_node_features)
    tree_features_df = pd.DataFrame(feature_tree_matches)

    @ray.remote
    def paired_selection_frequency_(feature_pair, df):
        f_1 = feature_pair[0]
        f_2 = feature_pair[1]

        N_1 = len(df.loc[df[f_1]].loc[~df[f_2]])
        N_2 = len(df.loc[~df[f_1]].loc[df[f_2]])
        N_12 = len(df.loc[df[f_2]].loc[df[f_2]])
        N_neither = len(df.loc[~df[f_1]].loc[~df[f_2]])

        p_value = fisher_exact(
            [[N_12, N_1], [N_2, N_neither]], alternative="greater"
        )[1]

        return (f_1, f_2), p_value

    # calculates paired selection frequencies in parallel
    futures = [
        paired_selection_frequency_.remote(feature_pair, tree_features_df)
        for feature_pair in combinations(included_features, 2)
    ]
    fisher_test_p_values = ray.get(futures)

    if multiple_test_correction:
        corrected_p_values = multipletests(
            [i[1] for i in fisher_test_p_values], method="bonferroni"
        )[1]
        fisher_test_p_values = [
            (fisher_test_p_values[i][0], corrected_p_values[i])
            for i in range(len(fisher_test_p_values))
        ]

    fisher_test_p_values.sort(key=itemgetter(1))  # return in order
    return fisher_test_p_values


def split_asymmetry(
    model: RandomForestRegressor, X_train: csr_matrix, y_train: pd.Series
):
    y_train = y_train.values
    decision_paths = [i.decision_path(X_train) for i in model.estimators_]
    decision_paths = [
        np.array(i, dtype=bool) for i in decision_paths
    ]  # convert to boolean array to make downstream processing easier

    def mean_node_value(node, y_train):
        values = y_train[node]
        return sum(values) / len(values)

    node_values = [
        np.apply_along_axis(mean_node_value, 0, dp, y_train)
        for dp in decision_paths
    ]  # mean value of each node in the tree


def selection_asymmetry():
    ...


def load_model(
    *, blosum_inferred: bool = True, filtered: bool = False
) -> RandomForestRegressor:
    if blosum_inferred == filtered:
        raise ValueError("One of blosum_inferred or filtered must be true")

    if blosum_inferred:
        rf_file = "results_blosum_inferred_pbp_types.pkl"
    elif filtered:
        rf_file = "results_filtered_pbp_types.pkl"

    results_dir = "results/random_forest"

    with open(os.path.join(results_dir, rf_file), "rb") as a:
        results = pickle.load(a)

    return results.model


def main():
    model = load_model()
    X_train, y_train = load_data()[0]

    # extract each decision tree from the rf
    trees = [DecisionTree_(dt) for dt in model.estimators_]

    # get all the features which were included in the model
    included_features = np.unique(
        np.concatenate([tree.internal_node_features for tree in trees])
    )

    paired_sf_p_values = paired_selection_frequency(trees, included_features)

    linked_features = co_occuring_feature_pairs(trees, included_features)


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)

    ray.init()

    main()
