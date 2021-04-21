import pickle
import os
import logging
from itertools import combinations

import ray
import numpy as np
import pandas as pd
from nptyping import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import fisher_exact

ray.init()


def get_decision_tree_features(dt: DecisionTreeRegressor) -> NDArray:
    """
    Code taken from sklearn documentation:
    https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#tree-structure # noqa: E501
    """
    n_nodes = dt.tree_.node_count
    children_left = dt.tree_.children_left
    children_right = dt.tree_.children_right
    features = dt.tree_.feature

    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [0]  # start with the root node id (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id = stack.pop()

        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children to `stack`
        if is_split_node:
            stack.append(children_left[node_id])
            stack.append(children_right[node_id])
        else:
            is_leaves[node_id] = True

    return features[~is_leaves]


def paired_selection_frequency(
    model: RandomForestRegressor, multiple_test_correction: bool = True
):
    # features included in each tree in the rf
    tree_features = [get_decision_tree_features(i) for i in model.estimators_]

    # wont necessarily have every feature in the data in the model
    included_features = np.unique(np.concatenate(tree_features))

    # df showing which features are in which tree
    feature_tree_matches = {i: [] for i in included_features}
    for feats in tree_features:
        for i in included_features:
            feature_tree_matches[i].append(i in feats)
    tree_features_df = pd.DataFrame(feature_tree_matches)

    @ray.remote
    def paired_selection_frequency_(feature_pair, df):
        f_1 = feature_pair[0]
        f_2 = feature_pair[1]

        N_12 = len(df.loc[df[f_2]].loc[df[f_2]])
        if N_12 == 0:
            return (f_1, f_2), -1

        N_1 = len(df.loc[df[f_1]].loc[~df[f_2]])
        N_2 = len(df.loc[~df[f_1]].loc[df[f_2]])
        N_neither = len(df.loc[~df[f_1]].loc[~df[f_2]])

        p_value = fisher_exact(
            [[N_12, N_1], [N_2, N_neither]], alternative="greater"
        )[1]

        return (f_1, f_2), p_value

    futures = [
        paired_selection_frequency_.remote(feature_pair, tree_features_df)
        for feature_pair in combinations(included_features, 2)
    ]
    fisher_test_p_values = ray.get(futures)


def split_asymmetry(model: RandomForestRegressor):
    ...


def selection_asymmetry(model: RandomForestRegressor):
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


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)

    main()
