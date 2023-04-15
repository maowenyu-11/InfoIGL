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

BCELoss = torch.nn.BCEWithLogitsLoss()


### GIN convolution along the graph structure
class DrugGINConv(MessagePassing):
    def __init__(self, emb_dim, edge_dim=3):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(DrugGINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim),
                                       torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        if edge_dim == 1:
            self.edge_encoder = BondEncoder(emb_dim=emb_dim)
        elif edge_dim > 0:
            self.edge_encoder = torch.nn.Linear(edge_dim, emb_dim)
        self.edge_dim = edge_dim

    def forward(self, x, edge_index, edge_attr):
        if self.edge_dim == -1:
            edge_embedding = edge_attr
        else:
            if self.edge_dim == 1:
                edge_attr = edge_attr
            edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp(
            (1 + self.eps) * x +
            self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        if self.edge_dim < 0:
            return F.relu(x_j)
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class HivGINConv(MessagePassing):
    """
    Graph Isomorphism Network message passing
    Input:
        - x (Tensor): node embedding
        - edge_index (Tensor): edge connectivity information
        - edge_attr (Tensor): edge feature
    Output:
        - prediction (Tensor): output node emebedding
    """
    def __init__(self, emb_dim):
        """
        Args:
            - emb_dim (int): node embedding dimensionality
        """

        super(HivGINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim),
                                       torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp(
            (1 + self.eps) * x +
            self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


# GINVirtual_node embedding for hiv,molbbbp dataset
class GINVirtual_node(torch.nn.Module):
    def __init__(self, num_layers, emb_dim, dropout=0.3, encode_node=True):

        super(GINVirtual_node, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.encode_node = encode_node
        # if self.num_layers < 2:
        #     raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layers):
            self.convs.append(HivGINConv(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layers - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))

    def forward(self, data):

        ### virtual node embeddings for graphs
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(
                edge_index.device))
        if self.encode_node:
            h_list = [self.atom_encoder(x)]
        else:
            h_list = [x]
        for layer in range(self.num_layers):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]
            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layers - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(
                    h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP
                virtualnode_embedding = F.dropout(
                    self.mlp_virtualnode_list[layer](
                        virtualnode_embedding_temp),
                    self.dropout,
                    training=self.training)

        node_embedding = h_list[-1]
        return node_embedding


class GCNMasker(nn.Module):
    def __init__(self, hidden_in, hidden, num_layer, cls_layer=2, mol=True):
        super(GCNMasker, self).__init__()

        self.global_pool = global_add_pool
        if hidden_in == 39:
            self.gnn_node = GNN_node_Virtualnode(num_layer=num_layer,
                                                 emb_dim=hidden,
                                                 input_dim=hidden_in)
        elif mol == True:
            self.gnn_node = GINVirtual_node(num_layers=num_layer,
                                            emb_dim=hidden)
        else:
            self.gnn_node = GINEncoder(num_layer, hidden_in, hidden)

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)
        self.bnc = BatchNorm1d(hidden)
        self.bno = BatchNorm1d(hidden)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, batched_data):

        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        row, col = edge_index
        x = self.gnn_node(batched_data)
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_o = edge_att[:, 1]
        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        node_weight_o = node_att[:, 1]

        return edge_weight_o, node_weight_o


class GCNNet(torch.nn.Module):
    def __init__(self,
                 hidden_in,
                 hidden,
                 num_layer,
                 num_classes,
                 cls_layer=2,
                 mol=True):
        super(GCNNet, self).__init__()
        self.global_pool = global_add_pool
        if hidden_in == 39:
            self.gnn_node = GNN_node_Virtualnode(num_layer=num_layer,
                                                 emb_dim=hidden,
                                                 input_dim=hidden_in)
        elif mol == True:
            self.gnn_node = GINVirtual_node(num_layers=num_layer,
                                            emb_dim=hidden)
        else:
            self.gnn_node = GINEncoder(num_layer, hidden_in, hidden)

        self.bno = BatchNorm1d(hidden)
        GConv = partial(GCNConv, edge_norm=True, gfn=False)
        self.objects_convs = GConv(hidden, hidden)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(hidden, 2 * hidden),
                                       torch.nn.BatchNorm1d(2 * hidden),
                                       torch.nn.ReLU(), torch.nn.Dropout())
        self.predictor = torch.nn.Linear(2 * hidden, num_classes)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, batched_data, edge_weight_o, node_weight_o):
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch

        x = self.gnn_node(batched_data)

        xo = node_weight_o.view(-1, 1) * x

        xo = F.relu(self.objects_convs(self.bno(xo), edge_index,
                                       edge_weight_o))

        xo = self.global_pool(xo, batch)
        xo = self.mlp(xo)

        return xo

    def predict(self, x):

        x = self.predictor(x)

        return x


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


### Virtual GNN to generate node embedding for DrugOOD dataset
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self,
                 num_layer,
                 emb_dim,
                 input_dim=1,
                 drop_ratio=0.5,
                 edge_dim=10):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

            # drugood
        self.node_encoder = torch.nn.Linear(
            input_dim, emb_dim)  # uniform input node embedding
        self.edge_dim = edge_dim

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):

            self.convs.append(DrugGINConv(emb_dim, edge_dim=self.edge_dim))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            # https://discuss.pytorch.org/t/batchnorm1d-cuda-error-an-illegal-memory-access-was-encountered/127641/5
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))

    def forward(self, batched_data):

        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(
                edge_index.device))
        h_list = [self.node_encoder(x)]
        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h),
                              self.drop_ratio,
                              training=self.training)

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(
                    h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                virtualnode_embedding = F.dropout(
                    self.mlp_virtualnode_list[layer](
                        virtualnode_embedding_temp),
                    self.drop_ratio,
                    training=self.training)

        ### Different implementations of Jk-concat

            node_representation = h_list[-1]

        return node_representation


class SSL(torch.nn.Module):
    def __init__(self, args, device):
        super(SSL, self).__init__()
        self.hidden_in = args.hidden_in
        self.num_classes = args.num_classes
        self.hidden = args.hidden
        self.num_layer = args.layers
        self.cls_layer = args.cls_layer
        self.mol = args.mol
        self.constraint = args.constraint
        self.global_ = args.global_
        self.local = args.local
        self.device = device
        # self.global_pool = global_add_pool
        self.att = GCNMasker(self.hidden_in, self.hidden, self.num_layer,
                             self.cls_layer, self.mol)

        self.subcausal = GCNNet(self.hidden_in, self.hidden, self.num_layer,
                                self.num_classes, self.cls_layer, self.mol)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden * 2, 2 * self.hidden),
            torch.nn.BatchNorm1d(2 * self.hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(self.hidden * 2, int(self.hidden / 2)),
            torch.nn.BatchNorm1d(int(self.hidden / 2)),
            torch.nn.ReLU(),
        )

        self.predictor = torch.nn.Linear(2 * self.hidden, self.num_classes)
        # self.memory_bank = torch.randn(self.num_classes, 10,
        #                                self.hidden_in).detach()

        self.register_buffer(
            "memory_bank",
            torch.randn(self.num_classes, 10, int(self.hidden / 2)))
        self.register_buffer(
            "prototype", torch.zeros(self.num_classes, int(self.hidden / 2)))

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, batched_data, criterion):
        edge_weight_o, node_weight_o = self.att(batched_data)
        # x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch

        xo = self.subcausal(batched_data, edge_weight_o, node_weight_o)
        pre = F.log_softmax(self.predictor(xo), dim=-1)
        pre_loss = self.pre_loss(pre, batched_data.y)
        # pred = self.subcausal.predict(xo)
        # one_hot_target = batched_data.y.view(-1).type(torch.int64)
        # pre_loss = F.nll_loss(pred, one_hot_target)

        if self.global_ == False and self.local == False:
            global_loss = 0
            local_loss = 0
        else:
            xp = self.projection_head(xo)
            class_causal, lack_class = self.class_split(batched_data.y, xp)
            self.prototype_update(class_causal, lack_class)
        if self.global_ == True:
            global_loss = self.global_ssl(class_causal, lack_class, criterion)
        else:
            global_loss = 0
        if self.local == True:
            local_loss = self.local_ssl(class_causal, lack_class, criterion)
        else:
            local_loss = 0
        return pre_loss, global_loss, local_loss

    def eval_forward(self, batched_data):
        edge_weight_o, node_weight_o = self.att(batched_data)
        # x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch

        xo = self.subcausal(batched_data, edge_weight_o, node_weight_o)
        pre = self.predictor(xo)

        return pre

    def pre_loss(self, pre, y):
        one_hot_target = y.view(-1).type(torch.int64)
        loss = F.nll_loss(pre, one_hot_target)
        return loss

    def class_split(self, y, causal_feature):
        class_causal = {}
        lack_class = []

        idx = 0
        for i in range(self.num_classes):
            k = np.where(y.view(-1).cpu() == i)
            if len(k[0]) == 0:
                class_causal[i] = torch.randn(1, int(self.hidden / 2)).cuda()
                lack_class.append(i)
            else:
                class_idx = torch.tensor(k).view(-1)
                class_causal_feature = causal_feature[class_idx]
                class_causal[i] = class_causal_feature

        return class_causal, lack_class

    def softmax_with_temperature(self, input, t=1, axis=-1):
        ex = torch.exp(input / t)
        sum = torch.sum(ex, axis=axis)
        return ex / sum

    #update prototypes
    def prototype_update(self, class_causal, lack_class):

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        if self.prototype.equal(
                torch.zeros(self.num_classes,
                            int(self.hidden / 2)).to(self.device)):
            prototype_ = [
                torch.mean(class_causal[key].to(torch.float32),
                           dim=0,
                           keepdim=True) for key in class_causal
            ]
            self.prototype = torch.cat(prototype_, dim=0).detach()

        else:
            for i in range(self.num_classes):

                if i in lack_class:
                    continue
                else:

                    cosine = cos(self.prototype[i].detach(),
                                 class_causal[i]).detach()
                    weights_proto = self.softmax_with_temperature(
                        cosine, t=5).reshape(1, -1).detach()
                    self.prototype[i] = torch.mm(weights_proto,
                                                 class_causal[i]).detach()
        # prototype = torch.cat(prototype, dim=0)

    def global_ssl(self, class_causal, lack_class, criterion):
        distance = None
        for i in range(self.num_classes):
            if i in lack_class:
                continue
            else:

                prototype_ = torch.cat((self.prototype[i:i + 1].detach(),
                                        self.prototype[0:i].detach(),
                                        self.prototype[i + 1:].detach()), 0)
                distance_ = torch.einsum('nc,kc->nk', [
                    nn.functional.normalize(class_causal[i], dim=1),
                    nn.functional.normalize(prototype_, dim=1)
                ])
                #distance_ /= 5
                if distance is None:
                    distance = F.softmax(distance_, dim=1)
                else:
                    distance = torch.cat(
                        (distance, F.softmax(distance_, dim=1)), 0)
        labels = torch.zeros(distance.shape[0], dtype=torch.long).cuda()

        loss = criterion(distance, labels)
        return loss

    def local_ssl(self, class_causal, lack_class, criterion):
        nce = None
        for key in class_causal:
            class_causal[key] = self.constraint * class_causal[key] + (
                1 - self.constraint) * self.prototype[key].detach()

        for i in lack_class:
            _ = class_causal.pop(i)

        for i in range(self.num_classes):
            if i in lack_class:
                continue
            else:
                pos = class_causal[i][0:1]
                class_causal_ = class_causal.copy()
                _ = class_causal_.pop(i)
                if class_causal_ == {}:
                    hard_neg = self.memory_bank[i].detach()
                else:
                    neg = torch.cat(list(class_causal_.values()))
                    # prototype_=self.prototype.clone().detach()
                    distance = F.softmax(torch.einsum(
                        'kc,nc->kn', [self.prototype[i:i + 1].detach(), neg]),
                                         dim=1)
                    dis, idx = torch.sort(distance)
                    if len(idx[0]) < 10:
                        hard_neg = torch.cat((self.memory_bank[i][0:(
                            10 - len(distance[0])), :].detach(), neg), 0)

                    else:
                        hard_neg = neg[idx[0][0:10]]

                    self.memory_bank[i] = hard_neg
                sample = torch.cat((pos, hard_neg), 0)
                nce_ = F.softmax(torch.einsum('nc,kc->nk',
                                              [class_causal[i], sample]),
                                 dim=1)
                if nce is None:
                    nce = nce_
                else:
                    nce = torch.cat((nce, nce_), 0)

        labels = torch.zeros(nce.shape[0], dtype=torch.long).cuda()
        loss = criterion(nce, labels)
        return loss
