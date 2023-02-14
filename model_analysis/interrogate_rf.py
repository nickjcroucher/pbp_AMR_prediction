import logging
import os
import pickle
from itertools import combinations
from math import ceil
from operator import itemgetter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import ray
from nptyping import Int, NDArray
from scipy.stats import fisher_exact, ttest_ind
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.multitest import multipletests

from model_analysis.parse_random_forest import (
    DecisionTree_,
    co_occuring_feature_pairs,
    valid_feature_pair,
)


# TODO: come back to this
def plot_feature_importances(model: RandomForestRegressor):
    # protein lengths for plotting
    a1_length = 277
    b2_length = 278
    x2_length = 359

    feature_idx = np.where(model.feature_importances_ != 0)[0]
    feature_scores = model.feature_importances_[
        feature_idx
    ]  # total reduction in cost function for each feature

    # map each feature (SNP) to locus
    loci_mapper = {i: ceil((i + 1) / 20) for i in feature_idx}
    df = pd.DataFrame({"feature_idx": feature_idx, "feature_scores": feature_scores})
    df["locus"] = df.feature_idx.apply(lambda x: loci_mapper[x])

    all_feature_scores = np.zeros(sum((a1_length, b2_length, x2_length)))
    for i, d in df.groupby(df.locus):
        all_feature_scores[i] = d.feature_scores.mean()

    # non-zero scores all on positive logarithmic scale
    log_feature_scores = np.array(
        [np.log10(i) if i != 0 else i for i in all_feature_scores]
    )
    relative_feature_scores = [
        i + abs(log_feature_scores.min()) + 1 if i != 0 else i
        for i in log_feature_scores
    ]

    a1_feature_scores = relative_feature_scores[:a1_length]
    b2_feature_scores = relative_feature_scores[a1_length : (a1_length + b2_length)]
    x2_feature_scores = relative_feature_scores[(a1_length + b2_length) :]


def bonferroni_correction(
    p_values: List[Tuple[Tuple[int], float]]
) -> List[Tuple[Tuple[int], float]]:
    corrected_p_values = multipletests([i[1] for i in p_values], method="bonferroni")[1]
    p_values = [(p_values[i][0], corrected_p_values[i]) for i in range(len(p_values))]
    return p_values


def paired_selection_frequency(
    trees: List[DecisionTree_],
    included_features: NDArray[Int],
    feature_combinations: List[Tuple[int, int]],
    multiple_test_correction: bool = True,
) -> List[Tuple[Tuple[int], float]]:
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
    logging.info("Matching features to trees")
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
        N_12 = len(df.loc[df[f_1]].loc[df[f_2]])
        N_neither = len(df.loc[~df[f_1]].loc[~df[f_2]])

        p_value = fisher_exact([[N_12, N_1], [N_2, N_neither]], alternative="greater")[
            1
        ]

        return (f_1, f_2), p_value

    logging.info(
        "Performing fishers exact test for paired selection frequency of each pair of features"  # noqa: E501
    )
    # calculates paired selection frequencies in parallel
    futures = [
        paired_selection_frequency_.remote(feature_pair, tree_features_df)
        for feature_pair in feature_combinations
    ]
    fisher_test_p_values = ray.get(futures)

    if multiple_test_correction:
        fisher_test_p_values = bonferroni_correction(fisher_test_p_values)

    fisher_test_p_values.sort(key=itemgetter(1))  # return in order
    return fisher_test_p_values


def split_asymmetry(
    linked_features: Dict[Tuple[int], List[DecisionTree_]],
    multiple_test_correction: bool = True,
) -> List[Tuple[Tuple[int], float]]:
    # need at least five slopes to calculate t statistic
    candidate_fps = {k: v for k, v in linked_features.items() if len(v) >= 5}

    def get_slope(fp, tree):
        # check second feature appears at least twice
        if len(np.where(tree.features == fp[1])[0]) < 2:
            return None

        fp_1 = tree.get_feature_first_node_id(fp[0])
        left_child = tree.decision_tree.tree_.children_left[fp_1]
        right_child = tree.decision_tree.tree_.children_right[fp_1]
        if any((tree.leaf_idx[left_child], tree.leaf_idx[right_child])):
            return None

        # nodes split on second feature
        all_fp_2 = np.where(tree.features == fp[1])[0]

        # get nodes of second feature on left hand side of first feature
        left_fp_2_loc = []
        for fp_2 in all_fp_2:
            # TODO: remove _ from method
            if tree._traverse_tree(left_child, fp_2):
                left_fp_2_loc.append(fp_2)
        if len(left_fp_2_loc) == 0:
            return None

        # get nodes of second feature on right hand side of first feature
        all_fp_2 = [i for i in all_fp_2 if i not in left_fp_2_loc]
        right_fp_2_loc = []
        for fp_2 in all_fp_2:
            if tree._traverse_tree(right_child, fp_2):
                right_fp_2_loc.append(fp_2)
        if len(right_fp_2_loc) == 0:
            return None

        # TODO: move to class definition
        node_values = tree.decision_tree.tree_.value.squeeze()

        slope_1 = (
            node_values[tree.decision_tree.tree_.children_left[left_fp_2_loc[0]]]
            - node_values[tree.decision_tree.tree_.children_right[left_fp_2_loc[0]]]
        )
        slope_2 = (
            node_values[tree.decision_tree.tree_.children_left[right_fp_2_loc[0]]]
            - node_values[tree.decision_tree.tree_.children_right[right_fp_2_loc[0]]]
        )

        return slope_1, slope_2

    # get slopes for each feature pair where second feature appears on either
    # side of the first
    fp_slopes = {}
    for fp, linked_trees in candidate_fps.items():
        slopes = [get_slope(fp, tree) for tree in linked_trees]
        slopes = [i for i in slopes if i is not None]
        if len(slopes) >= 5:
            fp_slopes[fp] = slopes

    if len(fp_slopes) == 0:
        logging.warning(
            "Insufficient asymmetric splits to perform t test for any pair of features"  # noqa: E501
        )

    # ttest p value for each
    ttest_p_values = []
    for fp, slopes in fp_slopes.items():
        left_slopes = [i[0] for i in slopes]
        right_slopes = [i[1] for i in slopes]
        ttest_p_values.append(
            (fp, ttest_ind(left_slopes, right_slopes, equal_var=False)[1])
        )

    if multiple_test_correction and ttest_p_values:
        ttest_p_values = bonferroni_correction(ttest_p_values)

    ttest_p_values.sort(key=itemgetter(1))
    return ttest_p_values


def selection_asymmetry(linked_features: Dict[Tuple[int], List[DecisionTree_]]):
    ...


def ensemble_model():
    ...


def load_model(
    train_pop: str,
    test_1_pop: str,
    inference_type: str,
    results_dir: str = "results/random_forest",
) -> RandomForestRegressor:
    def _load_result(suffix):
        rf_file = (
            f"train_pop_{train_pop}_results_{inference_type}_pbp_types{suffix}.pkl"
        )
        with open(os.path.join(results_dir, rf_file), "rb") as a:
            return pickle.load(a)

    results = _load_result("")
    if results.config["test_1_population"] != test_1_pop:
        results = _load_result("(1)")

    return results.model


def main(
    train_pop: str = "cdc",
    test_1_pop: str = "pmen",
    inference_type: str = "no_inference",
):
    model = load_model(
        train_pop,
        test_1_pop,
        inference_type,
        "results/maela_updated_mic_rerun/random_forest/",
    )

    # extract each decision tree from the rf
    trees = [DecisionTree_(dt) for dt in model.estimators_]

    # get all the features which were included in the model
    included_features = np.unique(
        np.concatenate([tree.internal_node_features for tree in trees])
    )

    # extract potentially interacting pairs of features
    feature_pairs = list(combinations(included_features, 2))
    feature_pairs = [
        fp for fp in feature_pairs if valid_feature_pair(*fp, alphabet_size=20)
    ]

    # trees in which each pair of features are linked in the decision path
    linked_features = co_occuring_feature_pairs(trees, feature_pairs)

    paired_sf_p_values = paired_selection_frequency(
        trees, included_features, feature_pairs
    )
    with open(
        f"results/intermediates/maela_updated_mic_rerun/{train_pop}/{inference_type}_linked_features_test1_{test_1_pop}.pkl",
        "wb",
    ) as a:
        pickle.dump(linked_features, a)
    with open(
        f"results/intermediates/maela_updated_mic_rerun/{train_pop}/{inference_type}_paired_sf_p_values_test1_{test_1_pop}.pkl",
        "wb",
    ) as a:
        pickle.dump(paired_sf_p_values, a)


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)

    ray.init()

    main()
