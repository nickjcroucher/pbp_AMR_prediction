import logging
import os
import pickle
import tempfile
from tqdm import tqdm
from typing import Dict, List

import numpy as np
import torch

from parse_GCN_data import load_data
from models.phylogeny_GNN_model import GCN
from model_analysis.GNN_explainer.simplified_explainer import (
    ExplainModule,
    load_neighborhoods,
)


def extract_neighborhood(node_idx, adj, feat, labels, neighborhoods):
    "Returns the neighborhood of a given node"

    neighbors_adj_row = neighborhoods[node_idx, :]
    # index of the query node in the new adj
    node_idx_new = sum(neighbors_adj_row[:node_idx])
    neighbors = np.nonzero(neighbors_adj_row)[0]
    sub_adj = adj[neighbors][:, neighbors]
    sub_feat = feat[neighbors]
    sub_label = labels[neighbors]

    return sub_adj, sub_feat, sub_label, node_idx_new, neighbors


def _build_explainer(x, adj, model, label) -> ExplainModule:
    optimiser_params = {
        "lr": 0.5,
        "opt_scheduler": "step",
        "opt_decay_step": 50,
        "opt_decay_rate": 0.5,
        "weight_decay": 0,
    }
    return ExplainModule(
        adj=adj,
        x=x,
        model=model,
        label=label,
        optimiser_params=optimiser_params,
        mask_activation_func="sigmoid",
        use_sigmoid=True,
        mask_bias=False,
    )


def train_explainer(
    x,
    neighbours_adj,
    model,
    label,
    node_idx_new: int,
    pred_label: float,
    num_epochs: int = 200,
    unconstrained: bool = False,
    verbose=False,
) -> ExplainModule:
    """
    Trains the explainer module to predict one node in the graph
    """

    explainer = _build_explainer(x, neighbours_adj, model, label)

    state_dict_file = tempfile.NamedTemporaryFile()
    explainer.train()
    losses = []
    for epoch in range(num_epochs):
        explainer.zero_grad()
        explainer.optimizer.zero_grad()
        ypred = explainer(node_idx_new, unconstrained=unconstrained, marginalize=True)
        loss = explainer.loss(ypred, pred_label, node_idx_new)
        loss.backward()

        explainer.optimizer.step()
        if explainer.scheduler is not None:
            explainer.scheduler.step()
        mask_density = explainer.mask_density()

        losses.append(loss)
        if losses[-1] == min(losses):
            torch.save(explainer.state_dict(), state_dict_file.name)

        if verbose:
            print(
                "epoch: ",
                epoch,
                "; loss: ",
                loss.item(),
                "; mask density: ",
                mask_density.item(),
                "; pred: ",
                ypred,
            )

    optimal_state = torch.load(state_dict_file.name)
    explainer.load_state_dict(optimal_state)

    return explainer


def save_explanations(node_explanations: Dict[int, np.ndarray]):
    ...


def load_model_and_data(
    filename: str, basepath: str = "results/phylogeny_GNN_model/hamming_dist_tree/"
):
    fpath = os.path.join(basepath, filename)
    with open(fpath, "rb") as a:
        result = pickle.load(a)

    laplacian = True
    hamming_dist_network = False
    tree = False
    n_duplicates_hd_network = False
    hamming_dist_tree = True
    drop_duplicates = False
    rank_hamming_distance = False
    hd_cuttoff = 5
    n = 3

    logging.warning("data being loaded with default arguments")

    data = load_data(
        filter_constant_features=True,
        train_population=result["train_population"],
        test_population_1=result["test_population_1"],
        graph_laplacian=laplacian,
        tree=tree,
        n_duplicates_hd_network=n_duplicates_hd_network,
        hamming_dist_tree=hamming_dist_tree,
        hamming_dist_network=hamming_dist_network,
        drop_duplicates=drop_duplicates,
        ranked_hamming_distance=rank_hamming_distance,
        hd_cuttoff=hd_cuttoff,
        n=n,
    )
    X = data["X"]
    y = data["y"]
    adj = data["laplacian"] if laplacian else data["adj"]
    idx_train = data["CV_indices"][0]

    adj = adj.to_dense()
    model = GCN(
        X.shape[1], X.shape[1], **result["model_params"], nclass=1, sparse_adj=False
    )
    model.load_state_dict(result["model_state_dict"])

    return X, y, adj, idx_train, model


def main(
    model,
    adj: torch.Tensor,
    feat: torch.Tensor,
    labels: torch.Tensor,
    num_gc_layers: int,
    node_indices: List[int],
):

    predictions = model(X, adj).detach()

    neighborhoods = load_neighborhoods(adj, n_hops=num_gc_layers)

    def extract_explanations(
        node_idx: int,
        adj: torch.Tensor,
        feat: torch.Tensor,
        labels: torch.Tensor,
        neighborhoods: np.ndarray,
    ) -> np.ndarray:
        neighbours_adj, x, label, node_idx_new, neighbors = extract_neighborhood(
            node_idx, adj, feat, labels, neighborhoods
        )
        pred_label = predictions[neighbors]
        explainer = train_explainer(
            x, neighbours_adj, model, label, node_idx_new, pred_label
        )
        feature_differences = (explainer.mask_features() - x).abs().mean(0)
        return feature_differences.detach().numpy()

    return {
        node_idx: extract_explanations(node_idx, adj, feat, labels, neighborhoods)
        for node_idx in tqdm(node_indices, "Getting explanatory features per node")
    }


if __name__ == "__main__":
    basepath = "results/phylogeny_GNN_model/hamming_dist_tree/"
    for filename in os.listdir(basepath):
        X, y, adj, idx_train, model = load_model_and_data(filename)
        node_explanations = main(
            model, adj, X, y, num_gc_layers=3, node_indices=idx_train
        )
