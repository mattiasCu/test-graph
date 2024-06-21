from collections.abc import Sequence

from torchdrug import core, layers
from torchdrug.core import Registry as R
from torch_scatter import scatter_add
import torch.nn.functional as F
from torch import nn

import torch

from DGMGearnet.layer import Combinedlayer, rewireGeometricRelationalGraphConv


@R.register("models.DGMGearNet")
class DGMGeometryAwareRelationalGraphNeuralNetwork(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation, edge_input_dim=None, num_angle_bin=None,
                 short_cut=False, batch_norm=False, activation="relu", concat_hidden=False, readout="sum"):
        super(DGMGeometryAwareRelationalGraphNeuralNetwork, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.edge_dims = [edge_input_dim] + self.dims[:-1]
        self.num_relation = num_relation
        self.num_angle_bin = num_angle_bin
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.batch_norm = batch_norm

        self.layers = nn.ModuleList()
        self.combined_layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            
            self.combined_layers.append(Combinedlayer(self.dims[i], self.dims[i + 1], num_relation, edge_input_dim, 
                                             num_heads=8, attention_out_features=512))
            self.layers.append(rewireGeometricRelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                                                   None, batch_norm, activation))
        if num_angle_bin:
            self.spatial_line_graph = layers.SpatialLineGraph(num_angle_bin)
            self.edge_layers = nn.ModuleList()
            for i in range(len(self.edge_dims) - 1):
                self.edge_layers.append(layers.GeometricRelationalGraphConv(
                    self.edge_dims[i], self.edge_dims[i + 1], num_angle_bin, None, batch_norm, activation))

        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None, edge_list=None, edge_weight=None):
        """
        Compute the node representations and the graph representation(s).

        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        hiddens = []
        layer_input = input
        if self.num_angle_bin:
            line_graph = self.spatial_line_graph(graph)
            edge_input = line_graph.node_feature.float()

        for i in range(len(self.layers)):
            
            
            new_edge_list, new_edge_weight = self.combined_layers[i]( layer_input, graph, edge_list, edge_weight)
            hidden = self.layers[i](graph, layer_input, new_edge_list, new_edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            if self.num_angle_bin:
                edge_hidden = self.edge_layers[i](line_graph, edge_input)
                edge_weight = graph.edge_weight.unsqueeze(-1)
                if new_edge_list is None:
                    node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
                else:
                    node_out = new_edge_list[:, 1] * self.num_relation + new_edge_list[:, 2]
                if new_edge_weight is None:
                    update = scatter_add(edge_hidden * edge_weight, node_out, dim=0,
                                        dim_size=graph.num_node * self.num_relation)
                else:
                    update = scatter_add(edge_hidden * edge_weight, node_out, dim=0,
                                        dim_size=graph.num_node * self.num_relation)
                update = update.view(graph.num_node, self.num_relation * edge_hidden.shape[1])
                update = self.layers[i].linear(update)
                update = self.layers[i].activation(update)
                hidden = hidden + update
                edge_input = edge_hidden
            if self.batch_norm:
                hidden = self.batch_norms[i](hidden)
                
            hiddens.append(hidden)
            layer_input = hidden
            edge_list = new_edge_list
            edge_weight = new_edge_weight

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }