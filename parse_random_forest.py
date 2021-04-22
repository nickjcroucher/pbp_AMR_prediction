from typing import Union, Tuple, List, Dict
from itertools import combinations

import ray
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from nptyping import NDArray, Int


class DecisionTree_:
    def __init__(self, dt: DecisionTreeRegressor):
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

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False

        # check if attributes are the same length and then can safely check if
        # all are equal using the numpy functions
        if all(
            [
                len(self.tree) == len(other.tree),
                len(self.leaf_idx) == len(other.leaf_idx),
                len(self.internal_node_features)
                == len(other.internal_node_features),
            ]
        ):

            return all(
                [
                    self.tree == other.tree,
                    (self.leaf_idx == other.leaf_idx).all(),
                    (
                        self.internal_node_features
                        == other.internal_node_features
                    ).all(),
                ]
            )
        else:
            return False


def co_occuring_feature_pairs(
    trees: List[DecisionTree_], included_features: NDArray[Int]
) -> Dict[Tuple[int, int], List[DecisionTree_]]:
    """
    Returns dictionary mapping pairs of features to trees in which ther are in
    the same decision path
    """

    linked_fps = {}

    def get_trees(fp):
        fp_trees = [tree for tree in trees if tree.linked_features(fp)]
        if fp_trees:
            linked_fps[fp] = fp_trees

    feature_pairs = combinations(included_features, 2)
    for fp in feature_pairs:
        get_trees(fp)
    return linked_fps

    # linked_fps = {
    #     fp: [tree for tree in trees if tree.linked_features(fp)]
    #     for fp in feature_pairs
    # }
    # return {k: v for k, v in linked_fps.items() if v}
