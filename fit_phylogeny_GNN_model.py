import time

import torch.optim as optim
import torch.nn.functional as F

from models.phylogeny_GNN_model import GCN
from parse_GCN_data import load_data
from utils import accuracy


EPOCHS = 100


data = load_data(filter_constant_features=True)
X = data["X"]
y = data["y"]
adj = data["adj"]
idx_train, idx_val, idx_test = data["CV_indices"]

model = GCN(X.shape[1], 100, 1, 0.3)
optimizer = optim.Adam(model.parameters())


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(X, adj)
    loss_train = F.mse_loss(output[idx_train], y[idx_train])
    acc_train = accuracy(output[idx_train], y[idx_train])
    loss_train.backward()
    optimizer.step()

    loss_val = F.mse_loss(output[idx_val], y[idx_val])
    acc_val = accuracy(output[idx_val], y[idx_val])
    print(
        f"Epoch: {epoch + 1}",
        f"loss_train: {loss_train.item()}",
        f"acc_train: {acc_train}",
        f"loss_val: {loss_val.item()}",
        f"acc_val: {acc_val}",
        f"time: {time.time() - t}s",
    )


def test():
    model.eval()
    output = model(X, adj)
    loss_test = F.mse_loss(output[idx_test], y[idx_test])
    acc_test = accuracy(output[idx_test], y[idx_test])
    print(
        "Test set results:",
        f"loss= {loss_test.item()}",
        f"accuracy= {acc_test}",
    )


if __name__ == "__main__":
    t_total = time.time()
    for epoch in range(EPOCHS):
        train(epoch)
    print("Optimization Finished!")
    print(f"Total time elapsed: {time.time() - t_total}s")

    # Testing
    test()
