import math

import torch
import torch.nn as nn
import torch.optim as optim


def load_neighborhoods(adj, n_hops, use_cuda=False):
    """Returns the n_hops degree adjacency matrix adj."""
    # adj = torch.tensor(adj, dtype=torch.float)
    if use_cuda:
        adj = adj.cuda()
    hop_adj = power_adj = adj
    for i in range(n_hops - 1):
        power_adj = power_adj @ adj
        hop_adj = hop_adj + power_adj
        hop_adj = (hop_adj > 0).float()
    return hop_adj.cpu().numpy().astype(int)


class ExplainModule(nn.Module):
    def __init__(
        self,
        adj,
        x,
        model,
        label,
        optimiser_params,
        mask_activation_func="sigmoid",
        use_sigmoid=True,
        mask_bias=False,
        gpu=False,
    ):
        super(ExplainModule, self).__init__()
        self.adj = adj
        self.x = x
        self.model = model
        self.label = label
        self.gpu = gpu
        self.mask_bias = mask_bias
        self.mask_act = mask_activation_func
        self.use_sigmoid = use_sigmoid

        init_strategy = "normal"
        num_nodes = adj.size()[1]
        self.mask, self.mask_bias = self.construct_edge_mask(
            num_nodes, init_strategy=init_strategy
        )

        self.feat_mask = self.construct_feat_mask(x.size(-1), init_strategy="constant")
        params = [self.mask, self.feat_mask]
        if self.mask_bias is not None:
            params.append(self.mask_bias)
        # For masking diagonal entries
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        if gpu:
            self.diag_mask = self.diag_mask.cuda()

        self.scheduler, self.optimizer = self.build_optimizer(
            params, **optimiser_params
        )

        self.coeffs = {
            "size": 0.005,
            "feat_size": 1.0,
            "ent": 1.0,
            "feat_ent": 0.1,
            "grad": 0,
            "lap": 1.0,
        }

    def build_optimizer(
        self,
        params,
        lr,
        opt_scheduler,
        opt_decay_step,
        opt_decay_rate,
        weight_decay=0.0,
    ):
        filter_fn = filter(lambda p: p.requires_grad, params)
        optimizer = optim.Adam(filter_fn, lr=lr, weight_decay=weight_decay)

        if opt_scheduler is None:
            return None, optimizer
        elif opt_scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=opt_decay_step, gamma=opt_decay_rate
            )
        return scheduler, optimizer

    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
                # mask[0] = 2
        return mask

    def construct_edge_mask(self, num_nodes, init_strategy="normal", const_val=1.0):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
                # mask.clamp_(0.0, 1.0)
        elif init_strategy == "const":
            nn.init.constant_(mask, const_val)

        if self.mask_bias:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias

    def _masked_adj(self):
        sym_mask = self.mask
        if self.mask_act == "sigmoid":
            sym_mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            sym_mask = nn.ReLU()(self.mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.adj.cuda() if self.gpu else self.adj
        masked_adj = adj * sym_mask
        if self.mask_bias:
            bias = (self.mask_bias + self.mask_bias.t()) / 2
            bias = nn.ReLU6()(bias * 6) / 6
            masked_adj += (bias + bias.t()) / 2
        return masked_adj * self.diag_mask

    def mask_density(self):
        mask_sum = torch.sum(self._masked_adj()).cpu()
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum

    def forward(
        self, node_idx, unconstrained=False, mask_features=True, marginalize=False
    ):

        if unconstrained:
            sym_mask = torch.sigmoid(self.mask) if self.use_sigmoid else self.mask
            self.masked_adj = (sym_mask + sym_mask.t()) / 2 * self.diag_mask
        else:
            self.masked_adj = self._masked_adj()
            if mask_features:
                x = self.mask_features()
        ypred = self.model(x, self.masked_adj)
        return ypred[node_idx]

    def mask_features(self):
        x = self.x.cuda() if self.gpu else self.x
        feat_mask = (
            torch.sigmoid(self.feat_mask) if self.use_sigmoid else self.feat_mask
        )
        std_tensor = torch.ones_like(x, dtype=torch.float) / 2
        mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
        z = torch.normal(mean=mean_tensor, std=std_tensor)
        return x + z * (1 - feat_mask)

    def loss(self, pred, pred_label, node_idx):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        pred_loss = torch.abs(pred - self.label[node_idx])

        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.ReLU()(self.mask)
        size_loss = self.coeffs["size"] * torch.sum(mask)

        feat_mask = (
            torch.sigmoid(self.feat_mask) if self.use_sigmoid else self.feat_mask
        )
        feat_size_loss = self.coeffs["feat_size"] * torch.mean(feat_mask)

        # entropy
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)

        # laplacian
        D = torch.diag(self.masked_adj)
        m_adj = self.masked_adj
        L = D - m_adj
        if self.gpu:
            pred_label = pred_label.cuda()
            L = L.cuda()
        lap_loss = (
            self.coeffs["lap"]
            * (pred_label.transpose(1, 0) @ L @ pred_label)
            / self.adj.numel()
        )

        loss = pred_loss + size_loss + lap_loss + mask_ent_loss + feat_size_loss
        return loss
