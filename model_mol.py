import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch_geometric.nn import MessagePassing, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_mean_pool
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
# from conv_base import GNN_node_Virtualnode
from GOOD.networks.models.MolEncoders import AtomEncoder, BondEncoder
#from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from gcn_conv import GCNConv
import numpy as np


class GINEncoder(torch.nn.Module):
    def __init__(self, num_layer, in_dim, emb_dim):
        super(GINEncoder, self).__init__()

        self.num_layer = num_layer
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.dropout_rate = 0.5
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(num_layer - 1)])
        self.batch_norm1 = nn.BatchNorm1d(emb_dim)
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(emb_dim) for _ in range(num_layer - 1)])
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropouts = nn.ModuleList(
            [nn.Dropout(self.dropout_rate) for _ in range(num_layer - 1)])
        self.conv1 = GINConv(
            nn.Sequential(nn.Linear(in_dim, 2 * emb_dim),
                          nn.BatchNorm1d(2 * emb_dim), nn.ReLU(),
                          nn.Linear(2 * emb_dim, emb_dim)))
        self.convs = nn.ModuleList([
            GINConv(
                nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim),
                              nn.BatchNorm1d(2 * emb_dim), nn.ReLU(),
                              nn.Linear(2 * emb_dim, emb_dim)))
            for _ in range(num_layer - 1)
        ])

    def forward(self, batched_data):

        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        post_conv = self.dropout1(
            self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i != len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
        return post_conv


class Causal(torch.nn.Module):
    def __init__(self, hidden_in, hidden_out, hidden, num_layer, cls_layer=2):
        super(Causal, self).__init__()

        self.num_classes = hidden_out
        self.global_pool = global_add_pool
        self.gnn_node = GINEncoder(num_layer=num_layer,
                                   in_dim=hidden_in,
                                   emb_dim=hidden)

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)
        self.bno = BatchNorm1d(hidden)
        GConv = partial(GCNConv, edge_norm=True, gfn=False)
        self.objects_convs = GConv(hidden, hidden)

        self.object_mlp = torch.nn.Sequential(BatchNorm1d(hidden),
                                              Linear(hidden, hidden), ReLU(),
                                              BatchNorm1d(hidden),
                                              Linear(hidden, hidden_out))

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, causal):

        pred = self.objects_readout_layer(causal)

        return pred

    def forward_causal(self, batched_data):
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        row, col = edge_index
        x = self.gnn_node(batched_data)
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight = edge_att[:, 1]
        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        node_weight = node_att[:, 1]
        causal = node_weight.view(-1, 1) * x
        causal = F.relu(
            self.objects_convs(self.bno(causal), edge_index, edge_weight))
        causal = self.global_pool(causal, batch)
        return causal

    def objects_readout_layer(self, x):

        x = self.object_mlp(x)
      #  x = F.log_softmax(x, dim=-1)
        return x

    def eval_forward(self, batched_data):
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        row, col = edge_index
        x = self.gnn_node(batched_data)
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight = edge_att[:, 1]
        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        node_weight = node_att[:, 1]
        causal = node_weight.view(-1, 1) * x
        causal = F.relu(
            self.objects_convs(self.bno(causal), edge_index, edge_weight))
        causal = self.global_pool(causal, batch)
        pred = self.objects_readout_layer(causal)

        return pred
