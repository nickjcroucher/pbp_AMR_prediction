import torch
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
import torch.nn.functional as F
from torch.nn.modules.module import Module


class GCN_layer(Module):
    def __init__(self, in_features, out_features, bayesian=False):
        super(GCN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bayesian:
            self.lin = BayesianLinear(in_features, out_features)
        else:
            self.lin = torch.nn.Linear(in_features, out_features)

    def forward(self, layer_input, adj):
        x = self.lin(layer_input)  # Y = WX^T + B
        x = torch.sparse.mm(adj, x)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.in_features}->{self.out_features})"


class Perceptron(Module):
    def __init__(self, in_features, out_features, bayesian=False):
        super(Perceptron, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bayesian:
            self.lin = BayesianLinear(in_features, out_features)
        else:
            self.lin = torch.nn.Linear(in_features, out_features)

    def forward(self, layer_input):
        return self.lin(layer_input)

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.in_features}->{self.out_features})"


class GCN(Module):
    def __init__(
        self,
        nfeat,
        nhid_1,
        nhid_2,
        nhid_3,
        nhid_4,
        nhid_5,
        nclass,
        dropout,
    ):
        super(GCN, self).__init__()

        self.gc1 = GCN_layer(nfeat, nhid_1)
        self.gc2 = GCN_layer(nhid_1, nhid_2)
        self.gc3 = GCN_layer(nhid_2, nhid_3)
        self.perc_1 = Perceptron(nhid_3, nhid_4)
        self.perc_2 = Perceptron(nhid_4, nhid_5)
        self.perc_3 = Perceptron(nhid_5, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.perc_1(x))
        x = F.relu(self.perc_2(x))
        return self.perc_3(x)


@variational_estimator
class BayesianGCN(Module):
    def __init__(
        self,
        nfeat,
        nhid_1,
        nhid_2,
        nhid_3,
        nhid_4,
        nhid_5,
        nclass,
        dropout,
    ):
        super(BayesianGCN, self).__init__()

        self.gc1 = GCN_layer(nfeat, nhid_1, bayesian=True)
        self.gc2 = GCN_layer(nhid_1, nhid_2, bayesian=True)
        self.gc3 = GCN_layer(nhid_2, nhid_3, bayesian=True)
        self.perc_1 = Perceptron(nhid_3, nhid_4, bayesian=True)
        self.perc_2 = Perceptron(nhid_4, nhid_5, bayesian=True)
        self.perc_3 = Perceptron(nhid_5, nclass, bayesian=True)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.perc_1(x))
        x = F.relu(self.perc_2(x))
        return self.perc_3(x)
