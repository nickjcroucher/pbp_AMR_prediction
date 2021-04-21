import pickle
import os
import logging
from typing import Dict, Tuple, List
from itertools import combinations

import ray
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import fisher_exact
from scipy.sparse import csr_matrix

from models import load_data

ray.init()


def parse_decision_tree(
    dt: DecisionTreeRegressor,
) -> Dict[int, List[int, int]]:
    """
    Code taken from sklearn documentation:
    https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#tree-structure # noqa: E501
    """
    children_left = dt.tree_.children_left
    children_right = dt.tree_.children_right
    features = dt.tree_.feature

    tree = {}
    stack = [0]  # start with the root node id (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id = stack.pop()

        parent_feature = features[node_id]

        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children to `stack`
        if is_split_node:
            stack.append(children_left[node_id])
            stack.append(children_right[node_id])
            child_left_feature = features[children_left[node_id]]
            child_right_feature = features[children_right[node_id]]
            tree.setdefault(parent_feature, []).append(child_left_feature)
            tree.setdefault(parent_feature, []).append(child_right_feature)

    return tree


def tree_feature_pairs(tree: Dict, feature_pair: Tuple[int, int]) -> bool:
    """
    Are two features linked in the decision path of a tree
    """
    internal_nodes = tree.keys()

    # check both nodes are actually in the tree
    if not all([i in internal_nodes for i in feature_pair]):
        return False

    def traverse_tree(feature_pair):
        def recursive_search(node):
            children = tree[node]
            if any([i == feature_pair[1] for i in children]):
                return True
            if all([i not in internal_nodes for i in children]):
                return False
            return any(
                [
                    recursive_search(child)
                    for child in children
                    if child in internal_nodes
                ]
            )

        return recursive_search(feature_pair[0])

    # start from first node and traverse down the tree
    same_path = traverse_tree(sorted(feature_pair, reverse=False))
    if same_path:
        return True

    # start from second node and traverse down the tree
    same_path = traverse_tree(sorted(feature_pair, reverse=True))
    if same_path:
        return True

    return False


def paired_selection_frequency(
    trees: List[Dict[int, List[int, int]]],
    multiple_test_correction: bool = True,
):
    # features used in each tree in the rf
    tree_features = [np.array(list(tree.keys())) for tree in trees]

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
            return (f_1, f_2), None

        N_1 = len(df.loc[df[f_1]].loc[~df[f_2]])
        N_2 = len(df.loc[~df[f_1]].loc[df[f_2]])
        N_neither = len(df.loc[~df[f_1]].loc[~df[f_2]])

        # TODO: check alternative hypothesis
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
    trees = [parse_decision_tree(dt) for dt in model.estimators_]

    X_train, y_train = load_data()[0]

    paired_selection_frequency(trees)


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)

    main()
