from typing import Dict, List, Optional, Tuple
import warnings

import math
import networkx as nx
import numpy as np
import pandas as pd
import torch
from Bio import Phylo
from scipy.sparse import coo_matrix, csr_matrix, identity, vstack

from data_preprocessing.parse_pbp_data import (
    encode_sequences,
    parse_cdc,
    parse_pmen_and_maela,
)
from utils import _data_processing


def tree_to_graph(tree_file: str) -> Tuple[coo_matrix, List]:
    Tree = Phylo.read(tree_file, "newick")
    G = Phylo.to_networkx(Tree)
    adj = nx.adjacency_matrix(G, nodelist=G.nodes, weight=None)
    adj = identity(adj.shape[0]) + adj
    adj = adj.tocoo()
    adj_tensor = torch.sparse_coo_tensor([adj.row, adj.col], adj.data)
    adj_tensor = adj_tensor.type(torch.FloatTensor).coalesce()
    return adj_tensor, list(G.nodes)


def load_features(
    drop_duplicates: bool = False, maela_correction: bool = False
) -> Tuple[np.ndarray, np.ndarray, csr_matrix]:
    pbp_patterns = ["a1", "b2", "x2"]

    cdc = pd.read_csv("./data/pneumo_pbp/cdc_seqs_df.csv")
    pmen = pd.read_csv("./data/pneumo_pbp/pmen_pbp_profiles_extended.csv")
    maela = pd.read_csv("./data/pneumo_pbp/maela_aa_df.csv")

    if maela_correction:
        maela.loc[maela.mic == 0.060, "mic"] = 0.03

    cdc = parse_cdc(cdc, pbp_patterns)
    pmen = parse_pmen_and_maela(pmen, cdc, pbp_patterns)
    maela = parse_pmen_and_maela(maela, cdc, pbp_patterns)

    cdc, pmen, maela = _data_processing(
        pbp_patterns,
        standardise_training_MIC=True,
        standardise_test_and_val_MIC=False,
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

    if drop_duplicates:
        dense_features = node_features.todense()
        df = pd.DataFrame(dense_features, index=ids)
        df = df.sample(frac=1, random_state=0)  # shuffles the rows
        df = df.drop_duplicates()

        id_mic_dict = {i: mic for i, mic in zip(ids, mics)}
        ids = df.index.values
        mics = np.array([id_mic_dict[i] for i in ids])
        node_features = csr_matrix(df.values)

    return ids, mics, node_features


def load_hamming_dist_network(
    parquet_path: str, ids: Optional[np.ndarray] = None, cutoff: float = 0.018
):
    dists = pd.read_parquet(parquet_path)
    if ids is not None:
        dists = dists.reindex(index=ids, columns=ids)  # ensure correct order
        return torch.Tensor(dists.values < cutoff).to_sparse()
    else:
        adj_tensor = torch.Tensor(dists.values < cutoff).to_sparse()
        return adj_tensor, dists.index


def select_items(
    all_indices: np.ndarray, n_selections: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(0)
    selected_idx = np.random.choice(all_indices, n_selections, replace=False)
    all_indices = np.setdiff1d(all_indices, selected_idx)
    return all_indices, selected_idx


def random_CV_split(
    sorted_features: np.ndarray,
    train_split: float,
    val_split: float,
    test_split: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # round to avoid floating point error

    assert round(sum([train_split, val_split, test_split]), 1) == 1

    terminal_nodes = np.vectorize(lambda x: not x.startswith("internal"))(
        sorted_features[:, 0]
    )
    terminal_node_indices = np.where(terminal_nodes)[0]
    n_terminal_nodes = len(terminal_node_indices)
    n_train = round(n_terminal_nodes * train_split)
    n_val = round(n_terminal_nodes * val_split)

    terminal_node_indices, train_idx = select_items(terminal_node_indices, n_train)
    test_idx, val_idx = select_items(terminal_node_indices, n_val)

    return train_idx, val_idx, test_idx


def population_CV_split(
    sorted_features: np.ndarray,
    train_population: str,
    test_population_1: str,
    val_split,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    test_population_2 = list(
        set(["cdc", "pmen", "maela"]) - set([train_population, test_population_1])
    )[0]
    test_1_idx = np.where(
        np.vectorize(lambda x: x.startswith(test_population_1))(sorted_features[:, 0])
    )[0]
    test_2_idx = np.where(
        np.vectorize(lambda x: x.startswith(test_population_2))(sorted_features[:, 0])
    )[0]
    train_idx = np.where(
        np.vectorize(lambda x: x.startswith(train_population))(sorted_features[:, 0])
    )[0]

    n_val = round(len(train_idx) * val_split)
    train_idx, val_idx = select_items(train_idx, n_val)

    return train_idx, val_idx, test_1_idx, test_2_idx


def get_CV_indices(
    sorted_features: np.ndarray,
    train_split: float = 0.5,
    val_split: float = 0.25,
    test_split: float = 0.25,
    train_population: Optional[str] = None,
    test_population_1: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_population is not None and test_population_1 is not None:
        return population_CV_split(
            sorted_features, train_population, test_population_1, val_split
        )
    else:
        return random_CV_split(sorted_features, train_split, val_split, test_split)


def map_features_to_graph(
    nodes_list: List,
    ids: np.ndarray,
    mics: np.ndarray,
    node_features: csr_matrix,
) -> np.ndarray:
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
    return sorted_features[:, 1:]  # remove index column


def approximate_internal_features(
    features: np.ndarray, adj: torch.Tensor
) -> np.ndarray:
    def get_unlabelled_node_indices(aa_features: np.ndarray) -> np.ndarray:
        return np.where(np.apply_along_axis(lambda x: (x == 0).all(), 1, aa_features))[
            0
        ]

    indices = np.array(adj.indices())
    amino_acid_features = features[:, 2:].astype(float)
    unlabelled_node_indices = get_unlabelled_node_indices(amino_acid_features)

    def mean_features(node_index: int) -> np.ndarray:
        neighbours = indices[1][indices[0] == node_index]
        try:
            # neighbours will include index node because of self loop in adj matrix
            return np.sum(amino_acid_features[neighbours], 0) / (
                neighbours.shape[0] - 1
            )
        except RuntimeWarning:
            return amino_acid_features[node_index]

    while unlabelled_node_indices.shape[0] > 0:
        with warnings.catch_warnings(record=True):
            approx_internal_features = np.array(
                [mean_features(i) for i in unlabelled_node_indices]
            )
        # TODO: figure out why this doesn't work
        amino_acid_features[unlabelled_node_indices] = approx_internal_features

        unlabelled_node_indices = get_unlabelled_node_indices(amino_acid_features)

    return np.concatenate((features[:, :2], amino_acid_features), 1)


def remove_non_variable_features(sorted_features: np.ndarray) -> np.ndarray:
    return sorted_features[
        :,
        np.concatenate(
            ([True] * 2, ~(np.std(sorted_features[:, 2:].astype(int), 0) == 0))
        ),
    ]


def convert_to_tensors(features: np.ndarray) -> Tuple:
    X = torch.Tensor(features[:, 2:].astype(float))
    y = torch.Tensor(features[:, 1].astype(float))
    y = torch.unsqueeze(y, 1)
    return X, y


def compute_graph_laplacian(adj_tensor: torch.Tensor):
    row_indices = adj_tensor.indices()[0].numpy()
    counts = np.unique(row_indices, return_counts=True)[1]
    normed_counts = np.array(
        [1 / math.sqrt(i) for i in counts]
    )  # equivalent to raising degree matrix to power -1/2
    D = torch.diag(torch.Tensor(normed_counts))
    D = D.type(adj_tensor.dtype)
    laplacian = torch.mm(D, torch.sparse.mm(adj_tensor, D))
    return laplacian.to_sparse().coalesce()


def load_data(
    filter_constant_features: bool = True,
    train_population: Optional[str] = None,
    test_population_1: Optional[str] = None,
    graph_laplacian: bool = True,
    tree: bool = False,
    n_duplicates_hd_network: bool = False,
    hamming_dist_network: bool = False,
    hamming_dist_tree: bool = True,
    hd_cuttoff: float = 0.005,
    drop_duplicates: bool = False,
    ranked_hamming_distance: bool = False,
    maela_correction: bool = False,
    n: int = 3,
) -> Dict:
    if (
        sum([tree, hamming_dist_network, hamming_dist_tree, n_duplicates_hd_network])
        != 1
    ):
        raise ValueError("One of tree and hamming_dist_network must be True")

    ids, mics, node_features = load_features(
        drop_duplicates=drop_duplicates, maela_correction=maela_correction
    )

    if tree:
        tree_file = "iqtree/PBP_alignment.fasta.treefile"
        adj_tensor, nodes_list = tree_to_graph(tree_file)
        sorted_features = map_features_to_graph(nodes_list, ids, mics, node_features)
    elif hamming_dist_tree:
        tree_file = "hamming_distance_network/ranked_hamming_distance_NJ_tree.nwk"
        adj_tensor, nodes_list = tree_to_graph(tree_file)
        sorted_features = map_features_to_graph(nodes_list, ids, mics, node_features)
    elif hamming_dist_network:
        if ranked_hamming_distance:
            parquet_path = "hamming_distance_network/ranked_hamming_dists.parquet"
            if drop_duplicates:
                raise ValueError(
                    "ranked hamming distances without duplicates is not yet available"
                )
        if drop_duplicates:
            parquet_path = "hamming_distance_network/unduplicated_hamming_dists.parquet"
        else:
            parquet_path = "hamming_distance_network/hamming_dists.parquet"
        adj_tensor = load_hamming_dist_network(parquet_path, ids, cutoff=hd_cuttoff)
        # TODO: add I to adj_tensor

        sorted_features = np.concatenate(
            (np.expand_dims(ids, 1), np.expand_dims(mics, 1), node_features.todense()),
            axis=1,
        )
        sorted_features = np.array(sorted_features)
    elif n_duplicates_hd_network:
        if ranked_hamming_distance:
            parquet_path = (
                f"hamming_distance_network/{n}_duplicates_ranked_hamming_dists.parquet"
            )
        else:
            parquet_path = (
                f"hamming_distance_network/{n}_duplicates_hamming_dists.parquet"
            )
        adj_tensor, samples_order = load_hamming_dist_network(
            parquet_path, ids=None, cutoff=hd_cuttoff
        )
        # TODO: add I to adj_tensor

        # order the samples
        samples_df = pd.DataFrame(
            np.concatenate((np.expand_dims(mics, 1), node_features.todense()), axis=1),
            index=ids,
        )
        samples_df = samples_df.loc[samples_df.index.isin(samples_order)]
        samples_df = (
            samples_df.assign(
                id=pd.Categorical(
                    samples_df.index, categories=samples_order, ordered=True
                )
            )
            .sort_values("id")
            .drop(columns="id")
        )

        sorted_features = samples_df.reset_index().values
        # ids = samples_df.index.values
        # mics = samples_df[0].values
        # node_features = csr_matrix(samples_df[samples_df.columns[1:]])

    CV_indices = get_CV_indices(
        sorted_features,
        train_population=train_population,
        test_population_1=test_population_1,
    )
    if filter_constant_features:
        sorted_features = remove_non_variable_features(sorted_features)
    sorted_features = approximate_internal_features(sorted_features, adj_tensor)
    X, y = convert_to_tensors(sorted_features)
    return {
        "X": X,
        "y": y,
        "adj": adj_tensor,
        "laplacian": compute_graph_laplacian(adj_tensor) if graph_laplacian else None,
        "node_names": sorted_features[:, 0],
        "CV_indices": CV_indices,
    }
