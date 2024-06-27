from torch_scatter import  scatter_add
import torch.nn as nn
from torchdrug import utils, layers
from torch.utils import checkpoint
import torch 

from torchdrug.core import Registry as R
import torch.nn.functional as F




# 可rewire的关系图神经网络
@R.register("layer.relationalGraph")
class relationalGraph(layers.MessagePassingBase):
    
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

        self.self_loop = nn.Linear(input_dim, output_dim)
        self.linear = nn.Linear(input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None
            
    def trans(self, A, graph):
    
        Degree_inv_sqrt = torch.diag(torch.pow(torch.sum(A, dim=1), -0.5))
        A_norm = torch.mm(torch.mm(Degree_inv_sqrt, A), Degree_inv_sqrt)
        
        n_rel = graph.num_relation
        n = A_norm.size(0)
        n_rel = n_rel.item()  # 将 n_rel 从 Tensor 转换为 int
        assert n % n_rel == 0, "n must be divisible by n_rel"
        
        block_size = n // n_rel
        
        # 初始化一个张量来存储累加结果
        accumulated = torch.zeros_like(A_norm[:block_size])
        
        # 将后面的所有块累加到第一块
        for i in range(n_rel):
            accumulated += A_norm[i * block_size: (i + 1) * block_size]
        
        # 用累加后的第一块替换原始矩阵的第一块
        A_trans = accumulated
    
        return A_trans

    def message_and_aggregate(self, graph, input, new_edge_list):
        assert graph.num_relation == self.num_relation
        
        
        if new_edge_list is None:
            node_in, node_out, relation = graph.edge_list.t()
            node_out = node_out * self.num_relation + relation
        
            edge_weight = torch.ones_like(node_out)
            degree_out = scatter_add(edge_weight, node_out, dim_size=graph.num_node * graph.num_relation)
            degree_out = degree_out
            edge_weight = edge_weight / degree_out[node_out]
            adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), edge_weight,
                                                (graph.num_node, graph.num_node * graph.num_relation))
            update = torch.sparse.mm(adjacency.t(), input)
        
        else:
            adjacency = self.trans(new_edge_list, graph)
            update = torch.mm(adjacency.t(), input)
        
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_input = self.edge_linear(edge_input)
            edge_weight = edge_weight.unsqueeze(-1)
            edge_update = scatter_add(edge_input * edge_weight, node_out, dim=0,
                                      dim_size=graph.num_node * graph.num_relation)
            update += edge_update

        return update

    def combine(self, input, update):
        # 自环特征
        device = input.device
        self.linear # Ensure the linear layers are on the correct device
        self.self_loop
        input = input.repeat(self.num_relation, 1)
        loop_update = self.self_loop(input)
        
        output = self.linear(update)+loop_update
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
    
    def forward(self, graph, input, new_edge_list=None):
        
        if self.gradient_checkpoint:
            update = checkpoint.checkpoint(self._message_and_aggregate, *graph.to_tensors(), input)
        else:
            update = self.message_and_aggregate(graph, input, new_edge_list)
        output = self.combine(input, update)
        return output
    
 
@R.register("layer.Rewirescorelayer")
class Rewirescorelayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, window_size, k, temperature=0.5, dropout=0.1):
        super(Rewirescorelayer, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.window_size = window_size
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.k = k
        
        self.query = nn.Linear(in_features, out_features * num_heads)
        self.key = nn.Linear(in_features, out_features * num_heads)
        self.value = nn.Linear(in_features, out_features * num_heads)
        self.scale = torch.sqrt(torch.FloatTensor([out_features])).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # get the start and end indices for each window of nodes
    def get_start_end(self, current, graph):
        segment = graph.num_nodes.repeat(graph.num_relation)
        index = torch.cumsum(segment, dim=0)
        
        # Use torch.searchsorted to find the appropriate segment
        pos = torch.searchsorted(index, current, right=True)

        if pos == 0:
            return (0, index[0].item())
        elif pos >= len(index):
            return (index[-1].item(), index[-1].item())
        else:
            return (index[pos-1].item(), index[pos].item())

    def gumbel_softmax_top_k(self, logits, tau=1.0, hard=False):
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / tau

        y_soft = F.softmax(gumbels, dim=-1)

        if hard:
            topk_indices = logits.topk(self.k, dim=-1)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, topk_indices, 1.0)
            y = (y_hard - y_soft).detach() + y_soft
        else:
            y = y_soft

        return y

    def forward(self, graph, node_features):
        device = node_features.device
        num_nodes = node_features.size(0)
    
        
         # calculate the start and end indices for each node
        half_window = self.window_size // 2
        start_end_indices = [self.get_start_end(i, graph) for i in range(num_nodes)]
        start_indices = torch.tensor([max(start_end_indices[i][0], i - half_window) for i in range(num_nodes)], device=device)
        end_indices = torch.tensor([min(start_end_indices[i][1], i + half_window) for i in range(num_nodes)], device=device)
        window_sizes = end_indices - start_indices

        Q = self.query(node_features).view(num_nodes, self.num_heads, self.out_features)     # [num_nodes, num_heads, out_features]
        K = self.key(node_features).view(num_nodes, self.num_heads, self.out_features)       # [num_nodes, num_heads, out_features]
        self.scale = self.scale.to(device)
        
        output = torch.zeros(num_nodes, num_nodes, device=device)
        all_scores = torch.zeros(num_nodes, self.num_heads, self.window_size, device=device)
        
        # 构建 K_windows 的批量张量并填充
        K_windows = torch.zeros((num_nodes, self.window_size, self.num_heads, self.out_features), device=device)
        for i in range(num_nodes):
            start = start_indices[i]
            end = end_indices[i]
            K_windows[i, :end-start] = K[start:end]

        # 扩展 Q 以便与 K_windows 进行批量矩阵乘法
        Q_expanded = Q.unsqueeze(2)  # [num_nodes, num_heads, 1, out_features]
        K_expanded = K_windows.permute(0, 2, 1, 3)  # [num_nodes, num_heads, max_window_size, out_features]

        # 批量计算 scores
        all_scores = torch.matmul(Q_expanded, K_expanded.transpose(-1, -2)) / self.scale  # [num_nodes, num_heads, 1, max_window_size]
        all_scores = all_scores.squeeze(2)  # [num_nodes, num_heads, max_window_size]
        
        # 掩码处理，确保仅计算有效部分
        mask = (torch.arange(self.window_size, device=device).expand(num_nodes, self.window_size) < window_sizes.unsqueeze(1))

        # 计算注意力权重
        all_scores = all_scores.masked_fill(~mask.unsqueeze(1), float('-inf'))  # [num_nodes, num_heads, max_window_size]
        attention_weights = F.softmax(all_scores / self.temperature, dim=-1)  # [num_nodes, num_heads, max_window_size]
        attention_weights = attention_weights.mean(dim=1) 
        
        # sample edges
        for i in range(num_nodes):
            start = start_indices[i]
            end = end_indices[i]
            output[i, start:end] = attention_weights[i, :end-start]

        edge_list = self.gumbel_softmax_top_k(output, self.temperature, self.k)

        return edge_list

# 可rewire的几何关系图卷积
@R.register("layer.RewireGearnet")
class RewireGearnet(nn.Module):
    gradient_checkpoint = False

    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None, batch_norm=False, activation="relu"):
        super(RewireGearnet, self).__init__()
        self.num_relation = num_relation
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(num_relation * input_dim, output_dim)
        self.self_loop = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim) if batch_norm else None
        self.activation = getattr(F, activation) if activation else None
        self.edge_linear = nn.Linear(edge_input_dim, output_dim) if edge_input_dim else None

    def trans(self, A, graph):
        n_rel = graph.num_relation
        n = A.size(0)
        n_rel = n_rel.item()  # 将 n_rel 从 Tensor 转换为 int
        assert n % n_rel == 0, "n must be divisible by n_rel"
        
        block_size = n // n_rel
        
        # 初始化一个张量来存储累加结果
        accumulated = torch.zeros_like(A[:block_size])
        
        # 将后面的所有块累加到第一块
        for i in range(n_rel):
            accumulated += A[i * block_size: (i + 1) * block_size]
        
        # 用累加后的第一块替换原始矩阵的第一块
        A_trans = accumulated
    
        return A_trans

    def message_and_aggregate(self, graph, input, new_edge_list=None):
        assert graph.num_relation == self.num_relation


        if new_edge_list is None:
            node_in, node_out, relation = graph.edge_list.t()
            node_out = node_out * self.num_relation + relation
            adjacency = torch.sparse_coo_tensor(
                torch.stack([node_in, node_out]),
                graph.edge_weight,
                (graph.num_node, graph.num_node * graph.num_relation)
            )
            update = torch.sparse.mm(adjacency.t(), input)
        else:
            adjacency = self.trans(new_edge_list, graph)
            update = torch.mm(adjacency.t(), input)
        
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_input = self.edge_linear(edge_input)
            edge_weight = graph.edge_weight.unsqueeze(-1)
            edge_update = scatter_add(
                edge_input * edge_weight, node_out, dim=0,
                dim_size=graph.num_node * graph.num_relation
            )
            update += edge_update
            
        return update.view(graph.num_node, self.num_relation * self.input_dim)

    def combine(self, input, update):
        self.linear  # Ensure the linear layers are on the correct device
        self.self_loop
        if self.batch_norm:
            self.batch_norm
        
        output = self.linear(update) + self.self_loop(input)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

    def forward(self, graph, input, new_edge_list=None, new_edge_weight=None):
        """
        Perform message passing over the graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations of shape :math:`(|V|, ...)`
        """
        if self.gradient_checkpoint:
            update = checkpoint.checkpoint(self.message_and_aggregate, graph, input)
        else:
            update = self.message_and_aggregate(graph, input, new_edge_list)
        output = self.combine(input, update)
        return output