import os
from torch import scatter
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import copy
import utils
import dgl.function as fn
msg_mask = fn.src_mul_edge('h', 'mask', 'm')
msg_orig = fn.copy_u('h', 'm')

class net_gcn(nn.Module):
    def __init__(self, embedding_dim, adj):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln + 1], bias=False) for ln in range(self.layer_num)])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.adj_nonzero = torch.nonzero(adj, as_tuple=False).shape[0]
        self.adj_mask1_train = nn.Parameter(self.generate_adj_mask(adj))
        self.adj_mask2_fixed = nn.Parameter(self.generate_adj_mask(adj), requires_grad=False)
        self.normalize = utils.torch_normalize_adj

    def forward(self, x, adj, val_test=False):

        adj = torch.mul(adj, self.adj_mask1_train)
        adj = torch.mul(adj, self.adj_mask2_fixed)
        adj = self.normalize(adj)
        # adj = torch.mul(adj, self.adj_mask2_fixed)
        for ln in range(self.layer_num):
            x = torch.mm(adj, x)
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)
        return x

    def generate_adj_mask(self, input_adj):

        sparse_adj = input_adj
        zeros = torch.zeros_like(sparse_adj)
        ones = torch.ones_like(sparse_adj)
        mask = torch.where(sparse_adj != 0, ones, zeros)
        return mask

class net_gcn_baseline(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln + 1], bias=False) for ln in range(self.layer_num)])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, adj, val_test=False):
        for ln in range(self.layer_num):
            x = torch.mm(adj, x)
            # x = torch.spmm(adj, x)
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)
        return x

class GINLayer(nn.Module):
    def __init__(self, apply_func, aggr_type, dropout, graph_norm, batch_norm, residual=False, init_eps=0, learn_eps=False):
        super().__init__()
        self.apply_func = apply_func
        if aggr_type == 'sum':
            self._reducer = fn.sum
        elif aggr_type == 'max':
            self._reducer = fn.max
        elif aggr_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggr_type))
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout
        in_dim = apply_func.mlp.input_dim
        out_dim = apply_func.mlp.output_dim
        if in_dim != out_dim:
            self.residual = False
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))
        self.bn_node_h = nn.BatchNorm1d(out_dim)

    def forward(self, g, h, snorm_n):
        h_in = h # for residual connection
        g = g.local_var()
        g.ndata['h'] = h
        # g.update_all(msg_orig, self._reducer('m', 'neigh'))
        ### pruning edges by cutting message passing process
        g.update_all(msg_mask, self._reducer('m', 'neigh'))
        h = (1 + self.eps) * h + g.ndata['neigh']
        if self.apply_func is not None:
            h = self.apply_func(h)
        if self.graph_norm:
            h = h * snorm_n # normalize activation w.r.t. graph size
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization
        h = F.relu(h) # non-linear activation
        if self.residual:
            h = h_in + h # residual connection
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class GINLayer_baseline(nn.Module):
    def __init__(self, apply_func, aggr_type, dropout, graph_norm, batch_norm, residual=False, init_eps=0, learn_eps=False):
        super().__init__()
        self.apply_func = apply_func
        if aggr_type == 'sum':
            self._reducer = fn.sum
        elif aggr_type == 'max':
            self._reducer = fn.max
        elif aggr_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggr_type))
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout
        in_dim = apply_func.mlp.input_dim
        out_dim = apply_func.mlp.output_dim
        if in_dim != out_dim:
            self.residual = False
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))
        self.bn_node_h = nn.BatchNorm1d(out_dim)

    def forward(self, g, h, snorm_n):
        h_in = h # for residual connection
        g = g.local_var()
        g.ndata['h'] = h
        g.update_all(msg_orig, self._reducer('m', 'neigh'))
        ### pruning edges by cutting message passing process
        # g.update_all(msg_mask, self._reducer('m', 'neigh'))
        h = (1 + self.eps) * h + g.ndata['neigh']
        if self.apply_func is not None:
            h = self.apply_func(h)
        if self.graph_norm:
            h = h * snorm_n # normalize activation w.r.t. graph size
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization
        h = F.relu(h) # non-linear activation
        if self.residual:
            h = h_in + h # residual connection
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class ApplyNodeFunc(nn.Module):
    """
        This class is used in class GINNet
        Update the node feature hv with MLP
    """
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, h):
        h = self.mlp(h)
        return h

class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        super().__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.input_dim = input_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim, bias=False)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.linears[i](h))
            return self.linears[-1](h)


class net_gin(nn.Module):
    def __init__(self, net_params, graph):
        super().__init__()
        in_dim = net_params[0]
        hidden_dim = net_params[1]
        n_classes = net_params[2]
        dropout = 0.5
        self.n_layers = 2
        self.edge_num = graph.all_edges()[0].numel()
        n_mlp_layers = 1  # GIN
        learn_eps = True  # GIN
        neighbor_aggr_type = 'mean'  # GIN
        graph_norm = False
        batch_norm = False
        residual = False
        self.n_classes = n_classes

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()

        for layer in range(self.n_layers):
            if layer == 0:
                mlp = MLP(n_mlp_layers, in_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, n_classes)

            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, graph_norm, batch_norm, residual, 0, learn_eps))
        self.linears_prediction = nn.Linear(hidden_dim, n_classes, bias=False)
        self.adj_mask1_train = nn.Parameter(torch.ones(self.edge_num, 1), requires_grad=True)
        self.adj_mask2_fixed = nn.Parameter(torch.ones(self.edge_num, 1), requires_grad=False)

    def forward(self, g, h, snorm_n, snorm_e):
        g.edata['mask'] = self.adj_mask1_train * self.adj_mask2_fixed
        hidden_rep = []

        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h, snorm_n)
            hidden_rep.append(h)

        score_over_layer = (self.linears_prediction(hidden_rep[0]) + hidden_rep[1]) / 2

        return score_over_layer


class net_gin_baseline(nn.Module):
    def __init__(self, net_params, graph):
        super().__init__()
        in_dim = net_params[0]
        hidden_dim = net_params[1]
        n_classes = net_params[2]
        dropout = 0.5
        self.n_layers = 2
        self.edge_num = graph.all_edges()[0].numel()
        n_mlp_layers = 1  # GIN
        learn_eps = True  # GIN
        neighbor_aggr_type = 'mean'  # GIN
        graph_norm = False
        batch_norm = False
        residual = False
        self.n_classes = n_classes

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()

        for layer in range(self.n_layers):
            if layer == 0:
                mlp = MLP(n_mlp_layers, in_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, n_classes)

            self.ginlayers.append(GINLayer_baseline(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, graph_norm, batch_norm, residual, 0, learn_eps))
        self.linears_prediction = nn.Linear(hidden_dim, n_classes, bias=False)

    def forward(self, g, h, snorm_n, snorm_e):
        hidden_rep = []
        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h, snorm_n)
            hidden_rep.append(h)
        score_over_layer = (self.linears_prediction(hidden_rep[0]) + hidden_rep[1]) / 2

        return score_over_layer


class GATHeadLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm, heads):
        super().__init__()
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.heads = heads

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}  # this dict all save in message box

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h, snorm_n, train_mask, fixed_mask):
        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)  # The function to generate new edge features
        ### pruning edges
        g.edata['e'] = g.edata['e'] * train_mask * fixed_mask
        g.update_all(self.message_func, self.reduce_func)

        h = g.ndata['h']
        if self.graph_norm:
            h = h * snorm_n
        if self.batch_norm:
            h = self.batchnorm_h(h)

        if not self.heads == 1:
            h = F.elu(h)
            h = F.dropout(h, self.dropout, training=self.training)
        return h

class GATHeadLayer_baseline(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm, heads):
        super().__init__()
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.heads = heads

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}  # this dict all save in message box

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h, snorm_n):
        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)  # The function to generate new edge features
        g.update_all(self.message_func, self.reduce_func)

        h = g.ndata['h']
        if self.graph_norm:
            h = h * snorm_n
        if self.batch_norm:
            h = self.batchnorm_h(h)

        if not self.heads == 1:
            h = F.elu(h)
            h = F.dropout(h, self.dropout, training=self.training)
        return h


class GATLayer(nn.Module):
    """
        Param: [in_dim, out_dim, n_heads]
    """

    def __init__(self, in_dim, out_dim, num_heads, dropout, graph_norm, batch_norm, residual=False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.residual = residual

        if in_dim != (out_dim * num_heads):
            self.residual = False

        self.heads = nn.ModuleList()
        for i in range(num_heads):  # 8
            self.heads.append(GATHeadLayer(in_dim, out_dim, dropout, graph_norm, batch_norm, num_heads))
        self.merge = 'cat'

    def forward(self, g, h, snorm_n, train_mask, fixed_mask):
        h_in = h  # for residual connection
        head_outs = [attn_head(g, h, snorm_n, train_mask, fixed_mask) for attn_head in self.heads]

        if self.merge == 'cat':
            h = torch.cat(head_outs, dim=1)
        else:
            h = torch.mean(torch.stack(head_outs))

        if self.residual:
            h = h_in + h  # residual connection
        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels, self.num_heads,
                                                                                   self.residual)

class GATLayer_baseline(nn.Module):
    """
        Param: [in_dim, out_dim, n_heads]
    """

    def __init__(self, in_dim, out_dim, num_heads, dropout, graph_norm, batch_norm, residual=False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.residual = residual
        if in_dim != (out_dim * num_heads):
            self.residual = False
        self.heads = nn.ModuleList()
        for i in range(num_heads):  # 8
            self.heads.append(GATHeadLayer_baseline(in_dim, out_dim, dropout, graph_norm, batch_norm, num_heads))
        self.merge = 'cat'

    def forward(self, g, h, snorm_n):
        h_in = h  # for residual connection
        head_outs = [attn_head(g, h, snorm_n) for attn_head in self.heads]
        if self.merge == 'cat':
            h = torch.cat(head_outs, dim=1)
        else:
            h = torch.mean(torch.stack(head_outs))

        if self.residual:
            h = h_in + h  # residual connection
        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels, self.num_heads,
                                                                                   self.residual)


class net_gat(nn.Module):

    def __init__(self, net_params, graph):
        super().__init__()

        in_dim_node = net_params[0]  # node_dim (feat is an integer)
        hidden_dim = net_params[1]
        out_dim = net_params[2]
        n_classes = net_params[2]
        num_heads = 8
        dropout = 0.6
        n_layers = 2
        self.edge_num = graph.number_of_edges()
        self.graph_norm = False
        self.batch_norm = False
        self.residual = False
        self.dropout = dropout
        self.n_classes = n_classes

        self.layers = nn.ModuleList([GATLayer(in_dim_node, hidden_dim, num_heads,
                                              dropout, self.graph_norm, self.batch_norm, self.residual) for _ in
                                     range(n_layers)])
        self.layers.append(
            GATLayer(hidden_dim * num_heads, out_dim, 1, 0, self.graph_norm, self.batch_norm, self.residual))
        self.classifier_ss = nn.Linear(hidden_dim * num_heads, n_classes, bias=False)
        self.adj_mask1_train = nn.Parameter(torch.ones(self.edge_num, 1), requires_grad=True)
        self.adj_mask2_fixed = nn.Parameter(torch.ones(self.edge_num, 1), requires_grad=False)


    def forward(self, g, h, snorm_n, snorm_e):
        # GAT
        for conv in self.layers:
            h_ss = h
            h = conv(g, h, snorm_n, self.adj_mask1_train, self.adj_mask2_fixed)
        h_ss = self.classifier_ss(h_ss)
        return h_ss


class net_gat_baseline(nn.Module):

    def __init__(self, net_params, graph):
        super().__init__()
        in_dim_node = net_params[0]  # node_dim (feat is an integer)
        hidden_dim = net_params[1]
        out_dim = net_params[2]
        n_classes = net_params[2]
        num_heads = 8
        dropout = 0.6
        n_layers = 2
        self.edge_num = graph.number_of_edges() + graph.number_of_nodes()
        self.graph_norm = False
        self.batch_norm = False
        self.residual = False
        self.dropout = dropout
        self.n_classes = n_classes
        self.layers = nn.ModuleList([GATLayer_baseline(in_dim_node, hidden_dim, num_heads,
                                              dropout, self.graph_norm, self.batch_norm, self.residual) for _ in
                                     range(n_layers)])
        self.layers.append(
            GATLayer_baseline(hidden_dim * num_heads, out_dim, 1, 0, self.graph_norm, self.batch_norm, self.residual))
        self.classifier_ss = nn.Linear(hidden_dim * num_heads, n_classes, bias=False)

    def forward(self, g, h, snorm_n, snorm_e):
        # GAT
        for conv in self.layers:
            h_ss = h
            h = conv(g, h, snorm_n)

        h_ss = self.classifier_ss(h_ss)

        return h_ss