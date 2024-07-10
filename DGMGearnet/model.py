from collections.abc import Sequence

from torchdrug import core, layers
from torchdrug.core import Registry as R
from torch_scatter import scatter_add
import torch.nn.functional as F
from torch import nn
import time

import torch

from DGMGearnet.layer import relationalGraph, Rewirescorelayer, RewireGearnet


@R.register("models.DGMGearnet")
class DGMGearnet(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, score_in_dim, score_out_dim, num_relation, num_heads, window_size, k, edge_input_dim=None, num_angle_bin=None,
                 short_cut=False, batch_norm=False, activation="relu", concat_hidden=False, readout="sum"):
        super(DGMGearnet, self).__init__()

        #if not isinstance(hidden_dims, Sequence):
            #hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.score_in_dim = score_in_dim
        self.score_out_dim = score_out_dim    
        self.num_heads = num_heads
        self.window_size = window_size
        self.k = k
        self.edge_dims = [edge_input_dim] + self.dims[:-1]
        self.num_relation = num_relation
        self.num_angle_bin = num_angle_bin
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.batch_norm = batch_norm

        self.layers = nn.ModuleList()
        self.score_layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            if i == 0:
                self.score_layers.append(relationalGraph(self.dims[i], self.score_in_dim, num_relation, 
                                                        edge_input_dim=None, batch_norm=False, activation="relu")) 

            else:
                self.score_layers.append(relationalGraph(self.dims[i]*(num_relation+1), score_in_dim, num_relation, 
                                                        edge_input_dim=None, batch_norm=False, activation="relu")) 
                    
            self.score_layers.append(Rewirescorelayer(self.score_in_dim, self.score_out_dim, self.num_heads, self.window_size, 
                                            self.k, temperature=0.5))
            

            self.layers.append(RewireGearnet(self.dims[i], self.dims[i + 1], num_relation,
                                            edge_input_dim=None, batch_norm=False, activation="relu"))
        
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

    def forward(self, graph, input, edge_list=None, all_loss=None, metric=None):
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
        device = input.device
        node_in, node_out, relation = graph.edge_list.t().to(device)
        node_out = node_out * self.num_relation + relation
        adjacency = torch.sparse_coo_tensor(
            torch.stack([node_in, node_out]),
            graph.edge_weight.to(device),
            (graph.num_node, graph.num_node * graph.num_relation),
            device=device
        )
        adjacency = adjacency.to_dense()
        
        hiddens = []
        layer_input = input
        score_layer_input = input
        if self.num_angle_bin:
            line_graph = self.spatial_line_graph(graph)
            edge_input = line_graph.node_feature.float()
            
        for i in range(len(self.layers)):
            
            relational_output = self.score_layers[2*i](graph, score_layer_input, edge_list)
            new_edge_list = self.score_layers[2*i+1](graph, relational_output)
            new_edge_list = torch.max(adjacency, new_edge_list)
            hidden = self.layers[i](graph, layer_input, new_edge_list)
            
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            
            if self.num_angle_bin:
                edge_hidden = self.edge_layers[i](line_graph, edge_input)
                edge_weight = graph.edge_weight.unsqueeze(-1)
                if new_edge_list is None:
                    node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
                else:
                    node_out = new_edge_list[:, 1] * self.num_relation + new_edge_list[:, 2]
                
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
            score_layer_input = torch.cat([hidden, relational_output.view(hidden.size(0), self.num_relation*hidden.size(-1))], dim=-1)
            layer_input = hidden
            edge_list = new_edge_list

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }