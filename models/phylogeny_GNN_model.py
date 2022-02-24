import torch

from torch.nn.modules.module import Module


class GCN_layer(Module):
    # arxiv.org/abs/1609.02907
    def __init__(self, in_features, out_features):
        super(GCN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lin = torch.nn.Linear(
            in_features, out_features
        )  # encodes weights and bias

    def forward(self, layer_input, adj):
        x = self.lin(layer_input)  # Y = WX^T + B
        x = torch.sparse.mm(adj, x)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.in_features}->{self.out_features})"
