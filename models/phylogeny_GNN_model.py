import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


class GCN_layer(Module):
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


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GCN_layer(nfeat, nhid)
        self.gc2 = GCN_layer(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
