import time
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models.phylogeny_GNN_model import GCN
from parse_GCN_data import load_data
from utils import accuracy, mean_acc_per_bin


EPOCHS = 150

data = load_data(filter_constant_features=True)
X = data["X"]
y = data["y"]
y_np = np.squeeze(y.numpy())
adj = data["adj"]
idx_train, idx_val, idx_test = data["CV_indices"]

model = GCN(X.shape[1], X.shape[1], 250, 1, 0.3)
optimizer = torch.optim.Adam(model.parameters())

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
        f"loss_train: {loss_train.item()}",
        f"acc_train: {acc_train}",
        f"mean_acc_train: {mean_acc_train}",
        f"loss_val: {loss_val.item()}",
        f"acc_val: {acc_val}",
        f"mean_acc_val: {mean_acc_val}",
        f"time: {time.time() - t}s\n",
    )


def test():
    model.eval()
    output = model(X, adj)
    output_np = np.squeeze(output.detach().numpy())
    loss_test, acc_test, mean_acc_test = compute_metrics(output, output_np, "test")
    print(
        "Test set results:",
        f"loss_test = {loss_test.item()}",
        f"acc_test = {acc_test}",
        f"mean_acc_test = {mean_acc_test}\n",
    )


if __name__ == "__main__":
    t_total = time.time()
    for epoch in range(EPOCHS):
        train(epoch)
        test()
    print("Optimization Finished!")
    print(f"Total time elapsed: {time.time() - t_total}s")
