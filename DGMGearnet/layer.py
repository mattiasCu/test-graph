from torch_scatter import  scatter_add
import torch.nn as nn
from torchdrug import utils, layers
from torch.utils import checkpoint
import torch 

from torchdrug.core import Registry as R
import torch.nn.functional as F




# 可rewire的关系图神经网络
class relationalGraph(layers. MessagePassingBase):
    eps = 1e-10

    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None, batch_norm=False, activation="relu"):
        super(relationalGraph, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.edge_input_dim = edge_input_dim

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        #self.self_loop = nn.Linear(input_dim, output_dim)
        self.linear = nn.Linear(input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def message_and_aggregate(self, graph, input, edge_list, edge_weight):
        device = input.device  # 获取设备
        assert graph.num_relation == self.num_relation

        if edge_list is None:
            node_in, node_out, relation = graph.edge_list.t()
        else:
            node_in, node_out, relation = edge_list.t()
        node_out = node_out * self.num_relation + relation
        if edge_weight is None:
            degree_out = scatter_add(graph.edge_weight, node_out, dim_size=graph.num_node * graph.num_relation)
            edge_weight = graph.edge_weight / (degree_out[node_out]+self.eps)
        else:
            degree_out = scatter_add(edge_weight, node_out, dim_size=graph.num_node * graph.num_relation)
            edge_weight = edge_weight / (degree_out[node_out]+self.eps)
        adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), edge_weight,
                                            (graph.num_node, graph.num_node * graph.num_relation)).to(device)
        update = torch.sparse.mm(adjacency.t(), input.to(device))
        
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_input = self.edge_linear(edge_input)
            edge_weight = edge_weight.unsqueeze(-1)
            edge_update = scatter_add(edge_input * edge_weight, node_out, dim=0,
                                      dim_size=graph.num_node * graph.num_relation)
            update += edge_update

        return update


    def combine(self, input, update):
        output = self.linear(update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
    
    def forward(self, graph, input, edge_list, edge_weight):
        
        if self.gradient_checkpoint:
            update = checkpoint.checkpoint(self._message_and_aggregate, *graph.to_tensors(), input)
        else:
            update = self.message_and_aggregate(graph, input, edge_list, edge_weight)
        output = self.combine(input, update)
        return output
    
    
# 计算得分
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_features, out_features, num_heads, temperature=0.5, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(in_features, out_features * num_heads)
        self.key = nn.Linear(in_features, out_features * num_heads)
        self.value = nn.Linear(in_features, out_features * num_heads)
        self.scale = torch.sqrt(torch.FloatTensor([out_features])).to(torch.device('cuda'))
        

    def forward(self, node_features, graph):
        num_nodes = node_features.size(0)
        device = node_features.device 
        
        self.scale = self.scale.to(device)

        # 计算查询、键和值，并为多头自注意力进行变形
        Q = self.query(node_features).view(num_nodes, self.num_heads, self.out_features)  # [num_nodes, num_heads, out_features]
        K = self.key(node_features).view(num_nodes, self.num_heads, self.out_features)    # [num_nodes, num_heads, out_features]
        #V = self.value(node_features).view(num_nodes, self.num_heads, self.out_features)  # [num_nodes, num_heads, out_features]

        # 计算相似性分数
        scores = torch.einsum("nhd,mhd->nhm", Q, K) / self.scale  # [num_heads, num_nodes, num_nodes]
        
        scores = scores / self.temperature

        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # [num_heads, num_nodes, num_nodes]
        
        # 应用 dropout
        attention_weights = self.dropout(attention_weights)

        # 多头结果合并
        attention_weights = attention_weights.mean(dim=-2)  # [num_nodes, num_nodes]

        return attention_weights  
    

#计算rewire后的边和边权重
class Combinedlayer(nn.Module):
    
    eps = 1e-10
    
    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim, 
                 num_heads = 8, attention_out_features = 64, temperature=0.5, dropout=0.1):
        super(Combinedlayer, self).__init__()
        
        self.relational_graph = relationalGraph(input_dim, output_dim, num_relation, edge_input_dim)
        self.attention = MultiHeadSelfAttention(output_dim, attention_out_features, num_heads, temperature, dropout)
    
    def split_matrix(self,matrix, num_relation):
        # 计算每份的行数
        rows_per_split = matrix.shape[0] // num_relation

        # 分割矩阵
        split_matrices = []
        for i in range(num_relation):
            start_idx = i * rows_per_split
            end_idx = start_idx + rows_per_split
            split_matrices.append(matrix[start_idx:end_idx])

        return split_matrices
    
    
    def merge_relation_matrices(self,matrix_list):
        merged_list = []

        # 遍历每个矩阵
        for idx, matrix in enumerate(matrix_list):
            num_rows = matrix.size(0)
            # 创建一个形状为 [num_rows, 1] 的张量，记录矩阵索引
            relation_idx = torch.full((num_rows, 1), idx, dtype=torch.long)
            # 将矩阵和索引列拼接在一起
            extended_matrix = torch.cat((matrix, relation_idx), dim=1)
            merged_list.append(extended_matrix)
        
        # 将所有扩展后的矩阵合并成一个矩阵
        merged_matrix = torch.cat(merged_list, dim=0)
        
        return merged_matrix

    def process_list(self, tensor_list):
        result_list = []
        edge_weights_list = []

        for matrix in tensor_list:
            num_nodes = matrix.size(0)
            indices_list = []
            values_list = []

            for i in range(num_nodes):
                row = matrix[i]
                # 找到前3个最大的数及其索引
                sorted_indices = torch.argsort(row, descending=True)  # 从大到小排序
                max_values = row[sorted_indices][:5]

                # 检查是否有超过3个相同的最大数
                count = (row == max_values[-1]).sum().item()

                if count > 3:
                    # 选择与当前行序号距离最近的列索引
                    relevant_indices = sorted_indices[:count]
                    distances = torch.abs(relevant_indices - i)
                    closest_indices = relevant_indices[torch.argsort(distances)][:3]
                    selected_indices = closest_indices
                else:
                    selected_indices = sorted_indices[:3]

                # 记录当前行号、所选列号和对应的值
                for j in selected_indices:
                    indices_list.append([i, j.item()])
                    values_list.append(row[j].item())

            result_matrix = torch.tensor(indices_list, dtype=torch.long)
            result_list.append(result_matrix)
            edge_weights_list.append(torch.tensor(values_list, dtype=torch.float))

        return self.merge_relation_matrices(result_list), torch.cat(edge_weights_list).view(-1, 1)

    
    def normalize_edge_weights(self,edge_weights):
        min_weight = edge_weights.min()
        max_weight = edge_weights.max()
        normalized_weights = (edge_weights - min_weight) / (max_weight - min_weight + self.eps)
        return normalized_weights

    
    def forward(self, node_features, graph, edge_list, edge_weight):
        # Apply relational graph layer
        relational_output = self.relational_graph(graph, node_features, edge_list, edge_weight)
        
        relational_output = self.split_matrix(relational_output, graph.num_relation)
        #print("relational_output: ", len(relational_output))
        
        attention_output = []
        for i in range(len(relational_output)):
            # Apply multi-head self attention
            output = self.attention(relational_output[i], graph)
            attention_output.append(output)
            
        #print("attention_output: ", attention_output)
        new_edge_list, new_edge_weight = self.process_list(attention_output)
        new_edge_weight = new_edge_weight.squeeze()  # 去掉多余的维度
        new_edge_weight = self.normalize_edge_weights(new_edge_weight)
        
        # 调试信息
        #print("new_edge_list: ", new_edge_list)
        print("new_edge_weight: ", new_edge_weight)
        
        
        return new_edge_list, new_edge_weight
    

# 可rewire的几何关系图卷积
class rewireGeometricRelationalGraphConv(layers.RelationalGraphConv):
    gradient_checkpoint = False

    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None, batch_norm=False, activation="relu"):
        super(rewireGeometricRelationalGraphConv, self).__init__(input_dim, output_dim, num_relation, edge_input_dim,
                                                           batch_norm, activation)

    def aggregate(self, graph, message, new_edge_list):
        assert graph.num_relation == self.num_relation

        if new_edge_list is None:
            node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
        else:
            node_out = new_edge_list[:, 1] * self.num_relation + new_edge_list[:, 2]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node * self.num_relation)
        update = update.view(graph.num_node, self.num_relation * self.input_dim)

        return update


    def message_and_aggregate(self, graph, input, new_edge_list, new_edge_weight):
        device = input.device  # 获取设备
        assert graph.num_relation == self.num_relation

        if new_edge_list is None:
            node_in, node_out, relation = graph.edge_list.t()
        else:
            node_in, node_out, relation = new_edge_list.t()
        node_out = node_out * self.num_relation + relation
        if new_edge_weight is None:
            adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), graph.edge_weight,
                                            (graph.num_node, graph.num_node * graph.num_relation)).to(device)
        else:
            adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), new_edge_weight,
                                            (graph.num_node, graph.num_node * graph.num_relation)).to(device)
        update = torch.sparse.mm(adjacency.t(), input.to(device))
        
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_input = self.edge_linear(edge_input)
            if new_edge_weight is None:
                edge_weight = graph.edge_weight.unsqueeze(-1)
            else:
                edge_weight = new_edge_weight.unsqueeze(-1)
            edge_update = scatter_add(edge_input * edge_weight, node_out, dim=0,
                                    dim_size=graph.num_node * graph.num_relation)
            update += edge_update
            
        return update.view(graph.num_node, self.num_relation * self.input_dim)
    
    def forward(self, graph, input, new_edge_list=None, new_edge_weight = None):
        """
        Perform message passing over the graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations of shape :math:`(|V|, ...)`
        """
        if self.gradient_checkpoint:
            update = checkpoint.checkpoint(self._message_and_aggregate, *graph.to_tensors(), input)
        else:
            update = self.message_and_aggregate(graph, input, new_edge_list, new_edge_weight)
        output = self.combine(input, update)
        return output