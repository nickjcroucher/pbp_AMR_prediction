from typing import Union, Tuple, List, Dict
from itertools import combinations
from psutil import cpu_count

import ray
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from nptyping import NDArray, Int


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

    def linked_features(self, feature_pair: Union[Tuple, List]) -> bool:
        """
        Are two features linked in the decision path of the tree?
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

        # start from second node and traverse down the tree
        same_path = self._traverse_tree(feature_2_id, feature_1_id)
        if same_path:
            return True

        return False

    def __hash__(self):
        return hash((self.decision_tree, self.n_nodes))


def _co_occuring_feature_pairs(
    trees: List[DecisionTree_],
    feature_pairs: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], List[DecisionTree_]]:

    linked_fps = {
        fp: [tree for tree in trees if tree.linked_features(fp)]
        for fp in feature_pairs
    }
    return {k: v for k, v in linked_fps.items() if v}


_co_occuring_feature_pairs_remote = ray.remote(_co_occuring_feature_pairs)


def co_occuring_feature_pairs(
    trees: List[DecisionTree_],
    included_features: NDArray[Int],
    parallel: bool = True,
) -> Dict[Tuple[int, int], List[DecisionTree_]]:
    """
    Returns dictionary mapping pairs of features to trees in which they are in
    the same decision path
    """
    feature_pairs = list(combinations(included_features, 2))

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
