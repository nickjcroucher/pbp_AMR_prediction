from typing import List

import numpy as np
import torch

from .simplified_explainer import ExplainModule, load_neighborhoods


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
        "lr": 0.1,
        "opt_scheduler": None,
        "opt_decay_step": None,
        "opt_decay_rate": None,
        "weight_decay": 0.0,
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
    num_epochs: int = 100,
    unconstrained: bool = False,
) -> ExplainModule:
    """
    Trains the explainer module to predict one node in the graph
    """

    explainer = _build_explainer(x, neighbours_adj, model, label)

    explainer.train()
    for epoch in range(num_epochs):
        explainer.zero_grad()
        explainer.optimizer.zero_grad()
        ypred = explainer(node_idx_new, unconstrained=unconstrained)
        loss = explainer.loss(ypred, pred_label, node_idx_new)
        loss.backward()

        explainer.optimizer.step()
        if explainer.scheduler is not None:
            explainer.scheduler.step()

        mask_density = explainer.mask_density()
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

    return explainer


def save_explanations(explainer: ExplainModule, node_idx: int):
    print(f"Node {node_idx} feature_mask: ")
    print(explainer.feat_mask)
    print(f"Node {node_idx} adjacency mask: ")
    print(explainer.masked_adj)


def main(
    model,
    adj: torch.Tensor,
    feat: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    num_gc_layers: int,
    node_indices: List[int] = [],
):
    neighborhoods = load_neighborhoods(adj, n_hops=num_gc_layers)
    for node_idx in node_indices:
        neighbours_adj, x, label, node_idx_new, neighbors = extract_neighborhood(
            node_idx, adj, feat, labels, neighborhoods
        )
        pred_label = predictions[neighbors]
        explainer = train_explainer(
            x, neighbours_adj, model, label, node_idx_new, pred_label
        )
        save_explanations(explainer, node_idx)
