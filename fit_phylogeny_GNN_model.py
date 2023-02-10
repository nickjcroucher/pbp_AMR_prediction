#!/usr/bin/env python3

import os
import pickle
import time
import sys
from functools import partial
from multiprocessing import cpu_count
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from bayes_opt import BayesianOptimization

from models.phylogeny_GNN_model import GCN, SpGAT
from parse_GCN_data import load_data
from utils import accuracy, mean_acc_per_bin

torch.set_num_threads(cpu_count() - 2)

EPOCHS = 100
LAPLACIAN = True
HAMMING_DIST_NETWORK = False
TREE = False
N_DUPLICATES_HD_NETWORK = False
HAMMING_DIST_TREE = True
DROP_DUPLICATES = False
RANK_HAMMING_DISTANCE = False
HD_CUTTOFF = 5
N = 3
TRAIN_POPULATION = sys.argv[1]
TEST_POPULATION_1 = sys.argv[2]
MAELA_CORRECTION = True

data = load_data(
    filter_constant_features=True,
    train_population=TRAIN_POPULATION,
    test_population_1=TEST_POPULATION_1,
    graph_laplacian=LAPLACIAN,
    tree=TREE,
    n_duplicates_hd_network=N_DUPLICATES_HD_NETWORK,
    hamming_dist_tree=HAMMING_DIST_TREE,
    hamming_dist_network=HAMMING_DIST_NETWORK,
    drop_duplicates=DROP_DUPLICATES,
    ranked_hamming_distance=RANK_HAMMING_DISTANCE,
    hd_cuttoff=HD_CUTTOFF,
    n=N,
    maela_correction=MAELA_CORRECTION,
)
X = data["X"]
y = data["y"]
y_np = np.squeeze(y.numpy())
adj = data["laplacian"] if LAPLACIAN else data["adj"]
adj = adj.to_dense()
idx_train, idx_val = data["CV_indices"][:2]
if TEST_POPULATION_1 is not None:
    idx_test_1, idx_test_2 = data["CV_indices"][2:]
else:
    idx_test_1 = data["CV_indices"][-1]
    idx_test_2 = data["CV_indices"][-1]

metrics_dict = {
    "train_loss": [],
    "train_acc": [],
    "train_mean_bin_acc": [],
    "val_loss": [],
    "val_acc": [],
    "val_mean_bin_acc": [],
    "test_1_loss": [],
    "test_1_acc": [],
    "test_1_mean_bin_acc": [],
    "test_2_loss": [],
    "test_2_acc": [],
    "test_2_mean_bin_acc": [],
}


def compute_metrics(
    output: torch.Tensor, data_set: str, record_metrics: bool = False
) -> Tuple:
    if data_set == "train":
        idx = idx_train
    elif data_set == "test_1":
        idx = idx_test_1
    elif data_set == "test_2":
        idx = idx_test_2
    elif data_set == "val":
        idx = idx_val

    output_np = np.squeeze(output.detach().numpy())

    loss = F.mse_loss(output[idx], y[idx])
    acc = accuracy(output[idx], y[idx])
    mean_bin_acc = mean_acc_per_bin(output_np[idx], y_np[idx])

    if record_metrics:
        metrics_dict[f"{data_set}_loss"].append(loss.item())
        metrics_dict[f"{data_set}_acc"].append(acc)
        metrics_dict[f"{data_set}_mean_bin_acc"].append(mean_bin_acc)

    return loss, acc, mean_bin_acc


def train(
    epoch: int,
    model: GCN,
    optimizer: torch.optim.Adam,
    verbose: bool = False,
    record_metrics: bool = False,
) -> Tuple[GCN, torch.optim.Adam]:
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(X, adj)
    loss_val, acc_val, mean_acc_val = compute_metrics(
        output, "val", record_metrics=record_metrics
    )
    loss_val.backward()
    optimizer.step()
    if verbose:
        loss_train, acc_train, mean_acc_train = compute_metrics(
            output, "train", record_metrics=record_metrics
        )
        print(
            f"Epoch: {epoch + 1},",
            f"loss_train: {round(loss_train.item(), 2)},",
            f"acc_train: {round(acc_train, 2)},",
            f"mean_acc_train: {round(mean_acc_train, 2)},",
            f"loss_val: {round(loss_val.item(), 2)},",
            f"acc_val: {round(acc_val, 2)},",
            f"mean_acc_val: {round(mean_acc_val, 2)},",
            f"time: {round(time.time() - t, 2)}s\n",
        )
    return model, optimizer


def test(model: GCN, record_metrics: bool = False):
    model.eval()
    output = model(X, adj)
    loss_test_1, acc_test_1, mean_acc_test_1 = compute_metrics(
        output, "test_1", record_metrics=record_metrics
    )
    loss_test_2, acc_test_2, mean_acc_test_2 = compute_metrics(
        output, "test_2", record_metrics=record_metrics
    )
    print(
        "Test set results:",
        f"loss_test_1 = {round(loss_test_1.item(), 2)},",
        f"acc_test_1 = {round(acc_test_1, 2)},",
        f"mean_acc_test_1 = {round(mean_acc_test_1, 2)},",
        f"loss_test_2 = {round(loss_test_2.item(), 2)},",
        f"acc_test_2 = {round(acc_test_2, 2)},",
        f"mean_acc_test_2 = {round(mean_acc_test_2, 2)}\n\n",
    )


def plot_metrics(metrics_df: pd.DataFrame, metric: str):
    plt.clf()
    metrics_df[[f"{i}_{metric}" for i in ["train", "val", "test_1", "test_2"]]].plot()
    plt.savefig(f"{metric}.png")
    plt.clf()


def train_evaluate(lr: float, model_class: str, **kwargs):
    weight_decay = kwargs.pop("weight_decay")
    kwargs = {k: int(v) if k != "dropout" else v for k, v in kwargs.items()}

    torch.manual_seed(0)
    if model_class == "GCN":
        model = GCN(X.shape[1], X.shape[1], **kwargs, nclass=1, sparse_adj=False)
    elif model_class == "SpGAT":
        model = SpGAT(X.shape[1], **kwargs, nclass=1)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)
    for epoch in range(EPOCHS):
        model, optimizer = train(epoch, model, optimizer)
    output = model(X, adj)
    return -compute_metrics(output, "test_1")[0]


def optimise_hps(
    model_class: str,
    lr: float = 0.001,
    init_points: int = 10,
    n_iter: int = 15,
) -> BayesianOptimization:
    if model_class == "GCN":
        if DROP_DUPLICATES or N_DUPLICATES_HD_NETWORK:
            pbounds = {
                "nhid_2": [20, 500],
                "nhid_3": [10, 300],
                "nhid_4": [10, 300],
                "nhid_5": [5, 100],
                "dropout": [0.05, 0.4],
                "weight_decay": [0.01, 0.3],
            }
        else:
            pbounds = {
                "nhid_2": [200, 1500],
                "nhid_3": [100, 1000],
                "nhid_4": [100, 1000],
                "nhid_5": [50, 250],
                "dropout": [0.05, 0.4],
                "weight_decay": [0.01, 0.3],
            }
    elif model_class == "SpGAT":
        if DROP_DUPLICATES:
            raise NotImplementedError("DROP_DUPLICATES with SpGAT")
        pbounds = {
            "nhid": [2, 10],
            "nheads": [2, 10],
            "dropout": [0.05, 0.4],
            "weight_decay": [0.01, 0.3],
            "alpha": [0.01, 0.3],
        }
    partial_fitting_function = partial(train_evaluate, lr=lr, model_class=model_class)
    optimizer = BayesianOptimization(
        f=partial_fitting_function, pbounds=pbounds, random_state=0
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    return optimizer


def main(lr: float = 0.001, model_class: str = "GCN") -> Tuple[GCN, pd.DataFrame, Dict]:
    bayes_optimizer = optimise_hps(model_class)
    params = bayes_optimizer.max["params"]
    weight_decay = params.pop("weight_decay")
    params = {k: int(v) if k != "dropout" else v for k, v in params.items()}

    torch.manual_seed(0)
    if model_class == "GCN":
        model = GCN(X.shape[1], X.shape[1], **params, nclass=1, sparse_adj=False)
    elif model_class == "SpGAT":
        model = SpGAT(X.shape[1], **params, nclass=1)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)
    for epoch in range(EPOCHS):
        model, optimizer = train(
            epoch, model, optimizer, verbose=True, record_metrics=True
        )
        test(model, record_metrics=True)

    metrics_df = pd.DataFrame(metrics_dict)
    return model, metrics_df, params


def save_results(metrics_df: pd.DataFrame, model_params: Dict, model_state_dict: Dict):
    data = {
        "metrics_df": metrics_df,
        "train_population": TRAIN_POPULATION,
        "test_population_1": TEST_POPULATION_1,
        "model_params": model_params,
        "model_state_dict": model_state_dict,
        "adj": adj,
        "X": X,
        "y": y,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test_1": idx_test_1,
        "idx_test_2": idx_test_2,
    }
    if HAMMING_DIST_NETWORK:
        network = f"hamming_dist_network_cuttoff_{HD_CUTTOFF}"
        if RANK_HAMMING_DISTANCE:
            network = "ranked_" + network

    elif HAMMING_DIST_TREE:
        network = "hamming_dist_tree"
    elif TREE:
        network = "tree"
    elif N_DUPLICATES_HD_NETWORK:
        network = f"{N}_duplicates_network_cuttoff_{HD_CUTTOFF}"
        if RANK_HAMMING_DISTANCE:
            network = "ranked_" + network

    target_dir = f"./results/maela_updated_mic_rerun/phylogeny_GNN_model/{network}"
    os.makedirs(target_dir, exist_ok=True)

    dest = os.path.join(target_dir, f"GNN_{TRAIN_POPULATION}_{TEST_POPULATION_1}.pkl")
    with open(dest, "wb") as a:
        pickle.dump(data, a)


if __name__ == "__main__":
    model, metrics_df, model_params = main()
    save_results(metrics_df, model_params, model.state_dict())
