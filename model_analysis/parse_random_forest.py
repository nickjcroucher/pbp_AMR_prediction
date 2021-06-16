import inspect
import os
import warnings
from itertools import compress
from math import ceil
from typing import Dict, List, Tuple, Union

import numpy as np
import ray
from psutil import cpu_count
from sklearn.tree import DecisionTreeRegressor


class DecisionTree_:
    def __init__(self, dt: DecisionTreeRegressor):
        self.decision_tree = dt
        self.n_nodes = dt.tree_.node_count
        self.features = dt.tree_.feature
        self.tree: Dict[int, List[int]] = {}
        self.leaf_idx = np.zeros(shape=self.n_nodes, dtype=bool)

        children_left = dt.tree_.children_left
        children_right = dt.tree_.children_right
        stack = [0]  # start with the root node id (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id = stack.pop()

            children_left_ids = children_left[node_id]
            children_right_ids = children_right[node_id]

            is_split_node = children_left_ids != children_right_ids
            # If a split node, append left and right children to `stack`
            if is_split_node:
                stack.append(children_left[node_id])
                stack.append(children_right[node_id])

                self.tree.setdefault(node_id, []).append(children_left_ids)
                self.tree.setdefault(node_id, []).append(children_right_ids)

            else:
                self.leaf_idx[node_id] = True

        self.internal_node_features = self.features[~self.leaf_idx]

    def get_feature_first_node_id(self, feature: int):
        try:
            return np.where(self.features == feature)[0][0]
        except IndexError:
            raise ValueError(f"{feature} not in tree")

    def _traverse_tree(self, feature_1, feature_2):
        def recursive_search(node):
            children = self.tree[node]
            if any([i == feature_2 for i in children]):
                return True
            if all([self.leaf_idx[i] for i in children]):
                return False
            return any(
                [
                    recursive_search(child)
                    for child in children
                    if not self.leaf_idx[child]
                ]
            )

        return recursive_search(feature_1)

    def linked_features(
        self,
        feature_pair: Union[Tuple, List],
        *,
        both_permutations: bool = False,
    ) -> bool:
        """
        Are two features linked in the decision path of the tree?
        feature_pair: Tuple or list containing two features in the tree
        both_permutations: If False (default) will start with the first item of
            feature_pair and traverse the tree looking for the second item
            only. If True will search for features appearing in the opposite
            order as well and will return True if it finds either permutation
        """
        if len(feature_pair) != 2:
            raise ValueError("feature pair must contain two valid node ids")

        # check both nodes are in the tree
        if not all([i in self.internal_node_features for i in feature_pair]):
            return False

        feature_1_id = self.get_feature_first_node_id(feature_pair[0])
        feature_2_id = self.get_feature_first_node_id(feature_pair[1])

        # start from first node and traverse down the tree
        same_path = self._traverse_tree(feature_1_id, feature_2_id)
        if same_path:
            return True

        if both_permutations:
            # start from second node and traverse down the tree
            same_path = self._traverse_tree(feature_2_id, feature_1_id)
            if same_path:
                return True

        return False


def valid_feature_pair(feature_1, feature_2, *, alphabet_size=20):
    """
    If features are a one hot encoding of a sequence, it does not
    make sense to ask if two features at the same locus are interacting.
    """
    return ceil((feature_1 + 1) / alphabet_size) != ceil(
        (feature_2 + 1) / alphabet_size
    )


def _co_occuring_feature_pairs(
    trees: List[DecisionTree_],
    feature_pairs: List[Tuple[int]],
) -> Dict[Tuple[int], List[DecisionTree_]]:
    def relevant_trees(fp):
        return [
            tree
            for tree in trees
            if tree.linked_features(fp, both_permutations=False)
        ]

    # checking for appearance of feature pair in each tree in each order
    fps_order_1 = [tuple(sorted(fp, reverse=True)) for fp in feature_pairs]
    fps_order_2 = [tuple(sorted(fp, reverse=False)) for fp in feature_pairs]
    linked_fps = {fp: relevant_trees(fp) for fp in fps_order_1}
    linked_fps.update({fp: relevant_trees(fp) for fp in fps_order_2})

    return {k: v for k, v in linked_fps.items() if v}


_co_occuring_feature_pairs_remote = ray.remote(_co_occuring_feature_pairs)


class JlApi:
    def __init__(self):
        from julia.core import UnsupportedPythonError

        try:
            from julia.api import Julia
            from julia import Main
        except UnsupportedPythonError:
            from julia.api import Julia

            Julia(compiled_modules=False)
            from julia import (  # noqa: E402, E501 # pylint: disable=no-name-in-module
                Main,
            )

        filename = inspect.getframeinfo(inspect.currentframe()).filename
        path = os.path.dirname(os.path.abspath(filename))
        julia_module = os.path.join(path, "co_occuring_feature_pairs.jl")

        self.jl_main = Main
        self.jl_main.eval(f'include("{julia_module}")')
        self.jl_main.eval("using .ParseRF")


JL_API = None


def _jl_co_occuring_feature_pairs(
    trees: List[DecisionTree_], feature_pairs: List[Tuple[int]]
) -> Dict[Tuple[int], List[DecisionTree_]]:

    global JL_API
    if JL_API is None:
        JL_API = JlApi()  # initialise pyjulia interface

    fps = [np.array(i) for i in feature_pairs]
    trees_ = [
        (i.features, i.tree, i.leaf_idx, i.internal_node_features)
        for i in trees
    ]
    linked_fps = JL_API.jl_main.ParseRF.co_occuring_feature_pairs(trees_, fps)
    linked_fps = {k: list(compress(trees, v)) for k, v in linked_fps.items()}
    return {k: v for k, v in linked_fps.items() if v}


def co_occuring_feature_pairs(
    trees: List[DecisionTree_],
    feature_pairs: List[Tuple[int]],
    *,
    use_julia: bool = True,
    parallel: bool = False,
) -> Dict[Tuple[int], List[DecisionTree_]]:
    """
    Returns dictionary mapping pairs of features to trees in which they are in
    the same decision path
    """
    if use_julia and parallel:
        warnings.warn(
            "use_julia and parallel are both set to True, use_julia will take preference"  # noqa: E501
        )

    if use_julia:
        return _jl_co_occuring_feature_pairs(trees, feature_pairs)

    if parallel:
        n_cpus = cpu_count()
        futures = [
            _co_occuring_feature_pairs_remote.remote(
                trees, feature_pairs[i : i + n_cpus]
            )
            for i in range(0, len(feature_pairs), n_cpus)
        ]
        results = ray.get(futures)

        all_results = results[0]
        for r in results[1:]:
            all_results.update(r)
        return all_results

    else:
        return _co_occuring_feature_pairs(trees, feature_pairs)
