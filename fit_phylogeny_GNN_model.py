import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from models.phylogeny_GNN_model import GCN
from parse_GCN_data import load_data
from utils import accuracy, mean_acc_per_bin


EPOCHS = 100
LAPLACIAN = True

data = load_data(
    filter_constant_features=True,
    train_population="cdc",
    test_population_1="pmen",
    graph_laplacian=LAPLACIAN,
)
X = data["X"]
y = data["y"]
y_np = np.squeeze(y.numpy())
adj = data["laplacian"] if LAPLACIAN else data["adj"]
idx_train, idx_val = data["CV_indices"][:2]
idx_test = np.concatenate(data["CV_indices"][2:])

model = GCN(X.shape[1], X.shape[1], 500, 500, 500, 100, 1, 0.3)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.2)

metrics_dict = {
    "train_loss": [],
    "train_acc": [],
    "train_mean_bin_acc": [],
    "val_loss": [],
    "val_acc": [],
    "val_mean_bin_acc": [],
    "test_loss": [],
    "test_acc": [],
    "test_mean_bin_acc": [],
}


def compute_metrics(
    output: torch.Tensor, output_np: np.ndarray, data_set: str
) -> Tuple:
    if data_set == "train":
        idx = idx_train
    elif data_set == "test":
        idx = idx_test
    elif data_set == "val":
        idx = idx_val

    loss = F.mse_loss(output[idx], y[idx])
    acc = accuracy(output[idx], y[idx])
    mean_bin_acc = mean_acc_per_bin(output_np[idx], y_np[idx])

    metrics_dict[f"{data_set}_loss"].append(loss.item())
    metrics_dict[f"{data_set}_acc"].append(acc)
    metrics_dict[f"{data_set}_mean_bin_acc"].append(mean_bin_acc)

    return loss, acc, mean_bin_acc


def train(epoch: int):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(X, adj)
    output_np = np.squeeze(output.detach().numpy())
    loss_train, acc_train, mean_acc_train = compute_metrics(output, output_np, "train")
    loss_train.backward()
    optimizer.step()
    loss_val, acc_val, mean_acc_val = compute_metrics(output, output_np, "val")
    print(
        f"Epoch: {epoch + 1}",
        f"loss_train: {round(loss_train.item(), 2)}",
        f"acc_train: {round(acc_train, 2)}",
        f"mean_acc_train: {round(mean_acc_train, 2)}",
        f"loss_val: {round(loss_val.item(), 2)}",
        f"acc_val: {round(acc_val, 2)}",
        f"mean_acc_val: {round(mean_acc_val, 2)}",
        f"time: {round(time.time() - t, 2)}s\n",
    )


def test():
    model.eval()
    output = model(X, adj)
    output_np = np.squeeze(output.detach().numpy())
    loss_test, acc_test, mean_acc_test = compute_metrics(output, output_np, "test")
    print(
        "Test set results:",
        f"loss_test = {round(loss_test.item(), 2)}",
        f"acc_test = {round(acc_test, 2)}",
        f"mean_acc_test = {round(mean_acc_test, 2)}\n\n",
    )


def plot_metrics(metrics_df: pd.DataFrame, metric: str):
    metrics_df[f"train_{metric}"].plot()
    metrics_df[f"val_{metric}"].plot()
    metrics_df[f"test_{metric}"].plot()
    plt.savefig(f"{metric}.png")


if __name__ == "__main__":
    t_total = time.time()
    for epoch in range(EPOCHS):
        train(epoch)
        test()
    print("Optimization Finished!")
    print(f"Total time elapsed: {time.time() - t_total}s")

    metrics_df = pd.DataFrame(metrics_dict)
