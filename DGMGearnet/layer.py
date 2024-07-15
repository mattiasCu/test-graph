from torch_scatter import  scatter_add
import torch.nn as nn
from torchdrug import utils, layers
from torch.utils import checkpoint
import torch 

from torchdrug.core import Registry as R
import torch.nn.functional as F
from einops import rearrange, repeat, pack, unpack


class relationalGraphConv(layers.MessagePassingBase):
    
    eps = 1e-10

    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None, batch_norm=False, activation="relu"):
        super(relationalGraphConv, self).__init__()
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
        self.linear = nn.Linear(num_relation * input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None


    def message_and_aggregate(self, graph, input, new_edge_list):
        assert graph.num_relation == self.num_relation
        device = input.device  # Ensure device consistency
        
        if new_edge_list is None:
            node_in, node_out, relation = graph.edge_list.t().to(device)
            node_out = node_out * self.num_relation + relation
        
            degree_out = scatter_add(graph.edge_weight, node_out, dim_size=graph.num_node * graph.num_relation)
            edge_weight = graph.edge_weight / degree_out[node_out]
            adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), edge_weight,
                                                (graph.num_node, graph.num_node * graph.num_relation))
            update = torch.sparse.mm(adjacency.t().to(device), input.to(device))
        else:
            
            new_edge_list = new_edge_list.t().view(graph.num_relation, graph.num_node, graph.num_node).permute(1, 0, 2).reshape(graph.num_node*graph.num_relation, graph.num_node).t()
            row, col = new_edge_list.nonzero(as_tuple=True)
            new_edge_weight = torch.ones_like(col, dtype=torch.float32)
            degree_out = scatter_add(new_edge_weight, col, dim_size=graph.num_node * graph.num_relation)
            edge_weight = new_edge_weight / degree_out[col]
            adjacency = utils.sparse_coo_tensor(torch.stack([row, col]), edge_weight,
                                                        (graph.num_node, graph.num_node * graph.num_relation))
            update = torch.sparse.mm(adjacency.t().to(device), input.to(device))
        
        if self.edge_linear:
            edge_input = graph.edge_feature.float().to(device)
            edge_input = self.edge_linear(edge_input)
            edge_weight = edge_weight.unsqueeze(-1).to(device)
            edge_update = scatter_add(edge_input * edge_weight, node_out, dim=0,
                                      dim_size=graph.num_node * graph.num_relation)
            update += edge_update

        return update.view(input.size(0), self.num_relation * self.input_dim)                           


    def combine(self, input, update):
        device = input.device
        self.linear.to(device)  # Ensure the linear layers are on the correct device
        self.self_loop.to(device)
        output = self.linear(update) + self.self_loop(input)
        if self.batch_norm:
            self.batch_norm.to(device)
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
    
    def forward(self, graph, input, new_edge_list=None):
        device = input.device
        if self.gradient_checkpoint:
            update = checkpoint.checkpoint(self._message_and_aggregate, *graph.to_tensors(), input, new_edge_list)
        else:
            update = self.message_and_aggregate(graph.to(device), input, new_edge_list)
        output = self.combine(input, update)
        return output


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
        
    def message_and_aggregate(self, graph, input, new_edge_list):
        assert graph.num_relation == self.num_relation
        device = input.device  # Ensure device consistency
        
        if new_edge_list is None:
            node_in, node_out, relation = graph.edge_list.t().to(device)
            node_out = node_out * self.num_relation + relation
        
            degree_out = scatter_add(graph.edge_weight, node_out, dim_size=graph.num_node * graph.num_relation)
            edge_weight = graph.edge_weight / degree_out[node_out]
            adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), edge_weight,
                                                (graph.num_node, graph.num_node * graph.num_relation))
            update = torch.sparse.mm(adjacency.t().to(device), input.to(device))
        else:
            new_edge_list = new_edge_list.t().view(graph.num_relation, graph.num_node, graph.num_node).permute(1, 0, 2).reshape(graph.num_node*graph.num_relation, graph.num_node).t()
            row, col = new_edge_list.nonzero(as_tuple=True)
            new_edge_weight = torch.ones_like(col, dtype=torch.float32)
            degree_out = scatter_add(new_edge_weight, col, dim_size=graph.num_node * graph.num_relation)
            edge_weight = new_edge_weight / degree_out[col]
            adjacency = utils.sparse_coo_tensor(torch.stack([row, col]), edge_weight,
                                                        (graph.num_node, graph.num_node * graph.num_relation))
            update = torch.sparse.mm(adjacency.t().to(device), input.to(device))
        
        if self.edge_linear:
            edge_input = graph.edge_feature.float().to(device)
            edge_input = self.edge_linear(edge_input)
            edge_weight = edge_weight.unsqueeze(-1).to(device)
            edge_update = scatter_add(edge_input * edge_weight, node_out, dim=0,
                                      dim_size=graph.num_node * graph.num_relation)
            update += edge_update

        return update.view(input.size(0), self.num_relation, self.input_dim).permute(1, 0, 2).reshape(self.num_relation*input.size(0), self.input_dim)

    def combine(self, input, update):
        # 自环特征
        device = input.device
        self.linear.to(device)  # Ensure the linear layers are on the correct device
        self.self_loop.to(device)
        input = input.repeat(self.num_relation, 1).to(device)
        loop_update = self.self_loop(input).to(device)
        
        output = self.linear(update)+loop_update
        if self.batch_norm:
            self.batch_norm.to(device)
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
    
    def forward(self, graph, input, new_edge_list=None):
        device = input.device
        if self.gradient_checkpoint:
            update = checkpoint.checkpoint(self._message_and_aggregate, *graph.to_tensors(), input, new_edge_list)
        else:
            update = self.message_and_aggregate(graph.to(device), input, new_edge_list)
        output = self.combine(input, update)
        return output


R.register("layer.relationalGraphStack")
class relationalGraphStack(nn.Module):
    
    def __init__(self, dims, num_relation, edge_input_dim=None, batch_norm=True, activation="relu"):
        super(relationalGraphStack, self).__init__()
        self.num_layers = len(dims) - 1
        self.layers = nn.ModuleList()
        for i in range(self.num_layers-1):
            self.layers.append(relationalGraphConv(dims[i], dims[i + 1], num_relation, edge_input_dim, batch_norm, activation))
            
        self.layers.append(relationalGraph(dims[-2], dims[-1], num_relation, edge_input_dim, batch_norm, activation))
            

    def forward(self, graph, input, new_edge_list=None):
        device = input.device
        x = input
        for layer in self.layers:
            x = layer(graph.to(device), x, new_edge_list)         
        return x.reshape(graph.num_relation, input.size(0), -1)



#===================================================================================================================================
@R.register("layer.Rewirescorelayer")
class Rewirescorelayer(nn.Module):
    def __init__(self, in_features, out_features, num_relations,num_heads, window_size, k, temperature=0.5):
        super(Rewirescorelayer, self).__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        self.num_relations = num_relations
        self.num_heads = num_heads
        self.window_size = window_size
        self.k = k
        self.temperature = temperature
        
        self.query = nn.Linear(in_features, out_features * num_heads)
        self.key = nn.Linear(in_features, out_features * num_heads)
        self.scale = 1 / (out_features ** 0.5)
    
    
    
    class LocalAttention(nn.Module):
        def __init__(
            self,
            window_size,
            look_backward = 1,
            look_forward = None,
            dropout = 0.,
            dim = None,
            scale = None,
            pad_start_position = None
        ):
            super().__init__()

            self.scale = scale

            self.window_size = window_size

            self.look_backward = look_backward
            self.look_forward = look_forward
            
            self.dropout = nn.Dropout(dropout)
            self.pad_start_position = pad_start_position
            
            

        def exists(self,val):
            return val is not None

        # 如果value不存在，返回d
        def default(self,value, d):
            return d if not self.exists(value) else value

        def to(self, t):
            return {'device': t.device, 'dtype': t.dtype}

        def max_neg_value(self, tensor):
            return -torch.finfo(tensor.dtype).max  #返回给定张量数据类型的所能表示的最大负值

        def look_around(self, x, backward = 1, forward = 0, pad_value = -1, dim = 2):  #x = bk: (40, 32, 16, 64)
            t = x.shape[1]    #获取一共有多少个窗口，这里是32
            dims = (len(x.shape) - dim) * (0, 0)   #一个长度为 len(x.shape) - dim 的元组，每个元素为 (0, 0)；其中len(x.shape) = 4
            padded_x = F.pad(x, (*dims, backward, forward), value = pad_value)   #在第二维度上，前面加backward个元素，后面加forward个元素 -> (40, 33, 16, 64)
            tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)] #一个张量列表，每个张量的维度为(40, 32, 16, 64), len = 2
            return torch.cat(tensors, dim = dim) #在第二维度上拼接 -> (40, 32, 32, 64)
    
        def forward(
            self,
            q, k,
            mask = None,
            input_mask = None,
            window_size = None
        ):

            mask = self.default(mask, input_mask)
            assert not (self.exists(window_size) and not self.use_xpos), 'cannot perform window size extrapolation if xpos is not turned on'
            shape, pad_value, window_size, look_backward, look_forward = q.shape, -1, self.default(window_size, self.window_size), self.look_backward, self.look_forward
            (q, packed_shape), (k, _) = map(lambda t: pack([t], '* n d'), (q, k))  #打包成[5, 8, 512, 64] -> [40, 512, 64] 


            b, n, dim_head, device, dtype = *q.shape, q.device, q.dtype   # 40, 512, 64
            scale = self.default(self.scale, dim_head ** -0.5)
            assert (n % window_size) == 0, f'sequence length {n} must be divisible by window size {window_size} for local attention'

            windows = n // window_size  # 512 / 16 = 32

            seq = torch.arange(n, device = device)                  # 0, 1, 2, 3, ..., 511
            b_t = rearrange(seq, '(w n) -> 1 w n', w = windows, n = window_size)    # (1, 32, 16) 排序序列变形后的矩阵

            # bucketing

            bq, bk = map(lambda t: rearrange(t, 'b (w n) d -> b w n d', w = windows), (q, k)) #重构：（40，512，64）->（40, 32, 16, 64）

            bq = bq * scale    # (40, 32, 16, 64)

            look_around_kwargs = dict(
                backward =  look_backward,
                forward =  look_forward,
                pad_value = pad_value
            )

            bk = self.look_around(bk, **look_around_kwargs)      # (40, 32, 32, 64)
    

            # calculate positions for masking

            bq_t = b_t
            bq_k = self.look_around(b_t, **look_around_kwargs) # (1, 32, 32)

            bq_t = rearrange(bq_t, '... i -> ... i 1')      # (1, 32, 16, 1)
            bq_k = rearrange(bq_k, '... j -> ... 1 j')      # (1, 32, 1, 16)

            pad_mask = bq_k == pad_value

            sim = torch.einsum('b h i e, b h j e -> b h i j', bq, bk)  # (40, 32, 16, 64) * (40, 32, 32, 64) -> (40, 32, 16, 32)

            mask_value = self.max_neg_value(sim)

            sim = sim.masked_fill(pad_mask, mask_value)


            if self.exists(mask):
                batch = mask.shape[0]    # 5
                assert (b % batch) == 0

                h = b // mask.shape[0]  # 8

                mask = rearrange(mask, '... (w n) -> (...) w n', w = windows, n = window_size)
                mask = self.look_around(mask, **{**look_around_kwargs, 'pad_value': False})
                mask = rearrange(mask, '... j -> ... 1 j')
                mask = repeat(mask, 'b ... -> (b h) ...', h = h)

                sim = sim.masked_fill(~mask, mask_value)
                del mask
                
            indices = [self.pad_start_position[i] // window_size for i in range(len(self.pad_start_position)) if i % 2 != 0]
            all_indices = list(range(windows))
            remaining_indices = [idx for idx in all_indices if idx not in indices]
            
            # 使用剩余的索引选择元素
            rest_sim = sim[:, remaining_indices, :, :]

            # attention
            attn = rest_sim.softmax(dim = -1)
            attn = self.dropout(attn)
            
            return attn

    def insert_zero_rows(self, tensor, lengths, target_lengths):
        assert len(lengths) == len(target_lengths), "Lengths and target lengths must be of the same length."
        
        # 计算每个位置需要插入的零行数
        zero_rows = [target - length for length, target in zip(lengths, target_lengths)]
        
        # 初始化结果列表
        parts = []
        mask_parts = []
        start = 0
        
        for i, length in enumerate(lengths):
            end = start + length
            
            # 原始张量部分
            parts.append(tensor[:, start:end, :])
            mask_parts.append(torch.ones(tensor.size(0), length, dtype=torch.bool, device=tensor.device))
            
            # 插入零行
            if zero_rows[i] > 0:
                zero_padding = torch.zeros(tensor.size(0), zero_rows[i], tensor.size(2), device=tensor.device)
                mask_padding = torch.zeros(tensor.size(0), zero_rows[i], dtype=torch.bool, device=tensor.device)
                parts.append(zero_padding)
                mask_parts.append(mask_padding)
            
            start = end
        
        # 拼接所有部分
        padded_tensor = torch.cat(parts, dim=1)
        mask = torch.cat(mask_parts, dim=1)
        
        return padded_tensor, mask


    def round_up_to_nearest_k_and_a_window_size(self, lst, k):
        pad_start_position = []
        result_lst = [(x + k - 1) // k * k +k for x in lst]
        for i in range(len(lst)):
            pad_start_position.append(sum(result_lst[:i])-i*k + lst[i])
            pad_start_position.append(sum(result_lst[:i+1])-k)
        return result_lst, pad_start_position

    def gumbel_softmax_top_k(self, logits,  top_k,  hard=False):
            gumbels = -torch.empty_like(logits).exponential_().log()
            gumbels = (logits + gumbels) / self.temperature

            y_soft = F.softmax(gumbels, dim=-1)

            if hard:
                topk_indices = logits.topk(top_k, dim=-1)[1]
                y_hard = torch.zeros_like(logits).scatter_(-1, topk_indices, 1.0)
                y = (y_hard - y_soft).detach() + y_soft
            else:
                y = y_soft

            return y
        
    def displace_tensor_blocks_to_rectangle(self, tensor, displacement):
        batch_size, num_blocks, block_height, block_width = tensor.shape

        # 计算新矩阵的宽度和高度
        height = num_blocks * displacement
        width =  (2 + num_blocks) * displacement

        # 初始化新的大张量，确保其形状为 (batch_size, height, width)
        new_tensor = torch.zeros(batch_size, height, width, device=tensor.device, dtype=tensor.dtype)

        for i in range(num_blocks):
            start_pos_height = i * displacement
            start_pos_width = i * displacement
            end_pos_height = start_pos_height + block_height
            end_pos_width = start_pos_width + block_width

            new_tensor[:, start_pos_height:end_pos_height, start_pos_width:end_pos_width] = tensor[:, i, :, :]

        return new_tensor
    
    def forward(self, graph, node_features):
        
        device = node_features.device
        num_relation = self.num_relations
        index = graph.num_nodes.tolist()
        
        target_input, pad_start_position = self.round_up_to_nearest_k_and_a_window_size(index, self.window_size)
        padding_input, mask = self.insert_zero_rows(node_features, index, target_input)
        
        self.query = self.query.to(device)
        self.key = self.key.to(device)
        Q = self.query(padding_input).view(num_relation, padding_input.size(1), self.num_heads, self.output_dim).permute(0, 2, 1, 3)                           # (num_relations, num_nodes, num_heads, out_features
        K = self.key(padding_input).view(num_relation, padding_input.size(1), self.num_heads, self.output_dim).permute(0, 2, 1, 3)                             # (num_relations, num_nodes, num_heads, out_features)
        Q = Q.reshape(num_relation * self.num_heads, padding_input.size(1), self.output_dim)                                                  # (num_relations*num_heads, num_nodes, out_features)
        K = K.reshape(num_relation * self.num_heads, padding_input.size(1), self.output_dim) 
        
        attn = self.LocalAttention(
            dim = self.output_dim,                   # dimension of each head (you need to pass this in for relative positional encoding)
            window_size = self.window_size,          # window size. 512 is optimal, but 256 or 128 yields good enough results
            look_backward = 1,                  # each window looks at the window before
            look_forward = 1,                   # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
            dropout = 0.1,
            pad_start_position = pad_start_position
            
        ) 
        
        attn = attn(Q, K, mask = mask).view(num_relation, self.num_heads, -1, self.window_size, 3*self.window_size).mean(dim=1)  
        score = self.gumbel_softmax_top_k(attn, self.k, hard=True)
        
        result_tensor = self.displace_tensor_blocks_to_rectangle(score, self.window_size)
        result_tensor = result_tensor[:, :, self.window_size:-self.window_size]
        indice = [pad_start_position[i] for i in range(len(pad_start_position)) if i % 2 == 0]
        indices = []

        for num in indice:
            next_multiple_of_window_size = ((num + self.window_size-1) // self.window_size) * self.window_size  # 计算向上取10的倍数
            sequence = range(num, next_multiple_of_window_size)  # 生成序列
            indices.extend(sequence)  # 直接将序列中的元素添加到结果列表中
        all_indices = list(range(result_tensor.size(1)))
        remaining_indices = [idx for idx in all_indices if idx not in indices]
        
        result_tensor = result_tensor[:, remaining_indices, :]
        result_tensor = result_tensor[:, :, remaining_indices]

        
        return result_tensor.permute(1, 0, 2).contiguous().view(result_tensor.size(1), result_tensor.size(0)*result_tensor.size(2))
        

        

        


#============================================================================================================================
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

    def message_and_aggregate(self, graph, input, new_edge_list):
        assert graph.num_relation == self.num_relation

        device = input.device  # Ensure device consistency
        new_edge_list = new_edge_list.t().view(graph.num_relation, graph.num_node, graph.num_node).permute(1, 0, 2).reshape(graph.num_node*graph.num_relation, graph.num_node)
        update = torch.mm(new_edge_list, input.to(device))
        
        if self.edge_linear:
            edge_input = graph.edge_feature.float().to(device)
            edge_input = self.edge_linear(edge_input)
            edge_weight = graph.edge_weight.unsqueeze(-1).to(device)
            edge_update = scatter_add(
                edge_input * edge_weight, node_out, dim=0,
                dim_size=graph.num_node * graph.num_relation
            )
            update += edge_update
            
        return update.view(input.size(0), self.num_relation * self.input_dim).to(device)

    def combine(self, input, update):
        device = input.device
        self.linear.to(device)  # Ensure the linear layers are on the correct device
        self.self_loop.to(device)
        if self.batch_norm:
            self.batch_norm.to(device)
        
        output = self.linear(update) + self.self_loop(input)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

    def forward(self, graph, input, new_edge_list=None):
        
        if self.gradient_checkpoint:
            update = checkpoint.checkpoint(self.message_and_aggregate, graph, input)
        else:
            update = self.message_and_aggregate(graph, input, new_edge_list)
        output = self.combine(input, update)
        return output
    


class rewireGearNetstack(nn.Module):
    
    def __init__(self, dims, num_relation, edge_input_dim=None, batch_norm=True, activation="relu"):
        super(rewireGearNetstack, self).__init__()
        self.num_layers = len(dims) - 1
        self.layers = nn.ModuleList()
        
        for i in range(self.num_layers-1):
            self.layers.append(RewireGearnet(dims[i], dims[i+1], num_relation, edge_input_dim, batch_norm, activation))
        
 
            
    def forward(self, graph, input, new_edge_list=None):
        device = input.device
        x = input
        for layer in self.layers:
            x = layer(graph.to(device), x, new_edge_list)       
        return x