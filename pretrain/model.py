import torch.nn.functional as F
import torch
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool
import numpy as np


def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class RGCNLayer(nn.Module):
    def __init__(self, input_dim, out_dim, num_relations):
        super(RGCNLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.rel_weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(input_dim, out_dim))
            for _ in range(num_relations)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.rel_weights:
            nn.init.xavier_uniform_(weight)

    def forward(self, x, edge_index, edge_type):
        # edge_index: [2, E], edge_type: [E,]
        z = torch.zeros(x.size(0), self.out_dim, device=x.device)
        for i in range(self.num_relations):
            edges = edge_index[:, edge_type == i]
            rel_weight = self.rel_weights[i]
            node_indices = edges[0]
            neighbor_indices = edges[1]
            z[node_indices] += F.relu(x[neighbor_indices] @ rel_weight)
        return z


class Graph_RGSN(nn.Module):
    def __init__(self, id_dim, input_dim, hidden_dim, num_layers, num_relations):
        super(Graph_RGSN, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.id_dim = id_dim
        for i in range(num_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dim + id_dim
            self.layers.append(RGCNLayer(layer_input_dim, hidden_dim, num_relations))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, 300))

    def forward(self, x, edge_index, edge_type, batch):
        z = x
        zs = []
        for i, (layer, bn) in enumerate(zip(self.layers, self.batch_norms)):
            if i != 0:
                z = torch.cat((x[:, :self.id_dim], z), dim=1)
            z = layer(z, edge_index, edge_type)
            z = F.relu(bn(z))
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class Graph_GSN(nn.Module):
    def __init__(self, id_dim, input_dim, hidden_dim, num_layers):
        super(Graph_GSN, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.id_dim = id_dim

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim + id_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for i, (conv, bn) in enumerate(zip(self.layers, self.batch_norms)):
            if i != 0:
                z = torch.cat((x[:, :self.id_dim], z), dim=1)
            z = conv(z, edge_index)
            z = F.relu(bn(z))
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g



class GCNLayer(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, feat, adj):
        feat = self.fc(feat)
        out = torch.bmm(adj, feat)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, num_layers, device):
        super(GCN, self).__init__()
        n_h = out_ft
        self.layers = []
        self.num_layers = num_layers
        self.layers.append(GCNLayer(in_ft, n_h).to(device))
        for __ in range(num_layers - 1):
            self.layers.append(GCNLayer(n_h, n_h).to(device))

    def forward(self, feat, adj, mask):
        h_1 = self.layers[0](feat, adj)
        h_1g = torch.sum(h_1, 1)
        for idx in range(self.num_layers - 1):
            h_1 = self.layers[idx + 1](h_1, adj)
            h_1g = torch.cat((h_1g, torch.sum(h_1, 1)), -1)
        return h_1, h_1g
