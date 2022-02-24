from os import remove
from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from Bio import Phylo
from scipy.sparse import csr_matrix, vstack

from data_preprocessing.parse_pbp_data import (
    encode_sequences,
    parse_cdc,
    parse_pmen_and_maela,
)
from utils import _data_processing


def tree_to_graph(tree_file: str) -> Tuple[csr_matrix, List]:
    Tree = Phylo.read(tree_file, "newick")
    G = Phylo.to_networkx(Tree)
    adj = nx.adjacency_matrix(G, nodelist=G.nodes)
    return adj, list(G.nodes)


def load_features() -> Tuple[np.ndarray, np.ndarray, csr_matrix]:
    pbp_patterns = ["a1", "b2", "x2"]

    cdc = pd.read_csv("../data/pneumo_pbp/cdc_seqs_df.csv")
    pmen = pd.read_csv("../data/pneumo_pbp/pmen_pbp_profiles_extended.csv")
    maela = pd.read_csv("../data/pneumo_pbp/maela_aa_df.csv")
    cdc = parse_cdc(cdc, pbp_patterns)
    pmen = parse_pmen_and_maela(pmen, cdc, pbp_patterns)
    maela = parse_pmen_and_maela(maela, cdc, pbp_patterns)

    cdc, pmen, maela = _data_processing(
        pbp_patterns,
        standardise_training_MIC=True,
        standardise_test_and_val_MIC=True,
        blosum_inference=False,
        HMM_inference=False,
        HMM_MIC_inference=False,
        filter_unseen=False,
        train=cdc,
        test_1=pmen,
        test_2=maela,
    )

    node_features = vstack(
        [encode_sequences(i, pbp_patterns) for i in [cdc, pmen, maela]]
    )
    mics = np.concatenate([i.log2_mic for i in [cdc, pmen, maela]])

    pmen = pmen.assign(id="pmen_" + pmen.id)
    maela = maela.assign(id="maela_" + maela.id)
    ids = pd.concat([i.id for i in [cdc, pmen, maela]])
    ids = ids.str.replace("#", "_").values  # to match names in tree

    return ids, mics, node_features


def map_features_to_graph(
    nodes_list: List,
    ids: np.ndarray,
    mics: np.ndarray,
    node_features: csr_matrix,
):
    dense_features = np.array(node_features.todense())
    all_features = np.concatenate(
        (np.expand_dims(ids, 1), np.expand_dims(mics, 1), dense_features), 1
    )

    # name internal nodes
    i = 0
    for node in nodes_list:
        if node.name is None:
            node.name = f"internal_{i}"
            i += 1

    # add internal nodes
    node_names = [node.name for node in nodes_list]
    n_internal_nodes = sum([i.startswith("internal") for i in node_names])
    internal_features = np.zeros((n_internal_nodes, all_features.shape[1] - 1)).astype(
        "object"
    )
    internal_nodes = np.array(
        [f"internal_{n}" for n in range(n_internal_nodes)]
    ).astype("object")
    internal_features = np.concatenate(
        (np.expand_dims(internal_nodes, 1), internal_features), axis=1
    )

    # sort features by order in graph adjacency matrix
    all_features = np.concatenate((all_features, internal_features))
    indices = np.array([node_names.index(x[0]) for x in all_features])
    all_features = np.concatenate((np.expand_dims(indices, 1), all_features), axis=1)
    sorted_features = all_features[all_features[:, 0].argsort()]
    return sorted_features[:, 1:]


def remove_empty_features(sorted_features: np.ndarray) -> np.ndarray:
    return sorted_features[
        :, np.concatenate(([True] * 2, ~(np.sum(sorted_features[:, 2:], 0) == 0)))
    ]


if __name__ == "__main__":
    tree_file = "iqtree/PBP_alignment.fasta.treefile"
    adj_matrix, nodes_list = tree_to_graph(tree_file)
    ids, mics, node_features = load_features()
    sorted_features = map_features_to_graph(nodes_list, ids, mics, node_features)
    filtered_features = remove_empty_features(sorted_features)
