import math

import torch
from torch import nn, einsum
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, repeat, pack, unpack
import time

# constant

TOKEN_SELF_ATTN_VALUE = -5e4

# helper functions

def exists(val):
    return val is not None

# 如果value不存在，返回d
def default(value, d):
    return d if not exists(value) else value

def to(t):
    return {'device': t.device, 'dtype': t.dtype}

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max  #返回给定张量数据类型的所能表示的最大负值

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):  #x = bk: (40, 32, 16, 64)
    t = x.shape[1]    #获取一共有多少个窗口，这里是32
    dims = (len(x.shape) - dim) * (0, 0)   #一个长度为 len(x.shape) - dim 的元组，每个元素为 (0, 0)；其中len(x.shape) = 4
    padded_x = F.pad(x, (*dims, backward, forward), value = pad_value)   #在第二维度上，前面加backward个元素，后面加forward个元素 -> (40, 33, 16, 64)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)] #一个张量列表，每个张量的维度为(40, 32, 16, 64), len = 2
    return torch.cat(tensors, dim = dim) #在第二维度上拼接 -> (40, 32, 32, 64)

# main class

class LocalAttention(Module):
    def __init__(
        self,
        window_size,
        look_backward = 1,
        look_forward = None,
        dim = None,
        scale = None,
        pad_start_position = None
    ):
        super().__init__()

        self.scale = scale

        self.window_size = window_size

        self.look_backward = look_backward
        self.look_forward = look_forward
        
        self.pad_start_position = pad_start_position

    def forward(
        self,
        q, k,
        mask = None,
        input_mask = None,
        window_size = None
    ):

        mask = default(mask, input_mask)

        assert not (exists(window_size) and not self.use_xpos), 'cannot perform window size extrapolation if xpos is not turned on'

        shape, pad_value, window_size, look_backward, look_forward = q.shape, -1, default(window_size, self.window_size), self.look_backward, self.look_forward

        # https://github.com/arogozhnikov/einops/blob/master/docs/4-pack-and-unpack.ipynb
        (q, packed_shape), (k, _) = map(lambda t: pack([t], '* n d'), (q, k))  #打包成[5, 8, 512, 64] -> [40, 512, 64] 

        # auto padding


        b, n, dim_head, device, dtype = *q.shape, q.device, q.dtype   # 40, 512, 64

        scale = default(self.scale, dim_head ** -0.5)

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

        bk = look_around(bk, **look_around_kwargs)      # (40, 32, 32, 64)
 

        # calculate positions for masking

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs) # (1, 32, 32)

        bq_t = rearrange(bq_t, '... i -> ... i 1')      # (1, 32, 16, 1)
        bq_k = rearrange(bq_k, '... j -> ... 1 j')      # (1, 32, 1, 16)

        pad_mask = bq_k == pad_value

        sim = einsum('b h i e, b h j e -> b h i j', bq, bk)  # (40, 32, 16, 64) * (40, 32, 32, 64) -> (40, 32, 16, 32)

        mask_value = max_neg_value(sim)

        sim = sim.masked_fill(pad_mask, mask_value)


        if exists(mask):
            batch = mask.shape[0]    # 5
            assert (b % batch) == 0

            h = b // mask.shape[0]  # 8

            mask = rearrange(mask, '... (w n) -> (...) w n', w = windows, n = window_size)
            mask = look_around(mask, **{**look_around_kwargs, 'pad_value': False})
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

        return attn

def insert_zero_rows(tensor, lengths, target_lengths):
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


def round_up_to_nearest_k_and_a_window_size(lst, k):
    pad_start_position = []
    result_lst = [(x + k - 1) // k * k +k for x in lst]
    for i in range(len(lst)):
        pad_start_position.append(sum(result_lst[:i])+lst[i])
        pad_start_position.append(sum(result_lst[:i+1])-k)
    return result_lst, pad_start_position

def gumbel_softmax_top_k(logits,  top_k, tau=1.0,  hard=False):
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / tau

        y_soft = F.softmax(gumbels, dim=-1)

        if hard:
            topk_indices = logits.topk(top_k, dim=-1)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, topk_indices, 1.0)
            y = (y_hard - y_soft).detach() + y_soft
        else:
            y = y_soft

        return y




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    index = [185, 415]
    num_nodes = sum(index)
    input_dim = 128
    output_dim = 128
    num_relation = 7
    input = torch.randn(num_relation, num_nodes, input_dim).to(device)
    window_size = 10
    num_heads = 8
    
    
    target_input, pad_start_position = round_up_to_nearest_k_and_a_window_size(index, window_size)
    
    padding_input, mask = insert_zero_rows(input, index, target_input)

    query = nn.Linear(input_dim, output_dim * num_heads)
    key = nn.Linear(input_dim, output_dim * num_heads)
    
    start1 = time.time()
    Q = query(padding_input).view(num_relation, padding_input.size(1), num_heads, output_dim).permute(0, 2, 1, 3)
    K = key(padding_input).view(num_relation, padding_input.size(1), num_heads, output_dim).permute(0, 2, 1, 3)
    Q = Q.reshape(num_relation * num_heads, padding_input.size(1), output_dim)                                    # (num_relations*num_heads, num_nodes, out_features)
    K = K.reshape(num_relation * num_heads, padding_input.size(1), output_dim)                                    # (num_relations*num_heads, num_nodes, out_features)
    
    end1 = time.time()
    print(f"Time taken: {end1 - start1:.6f}s")
    
    
    start2 = time.time()
    attn = LocalAttention(
        dim = output_dim,                # dimension of each head (you need to pass this in for relative positional encoding)
        window_size = window_size,       # window size. 512 is optimal, but 256 or 128 yields good enough results
        look_backward = 1,       # each window looks at the window before
        look_forward = 1,        # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
        pad_start_position = pad_start_position
    )

    attn = attn(Q, K, mask = mask).view(num_relation, num_heads, -1, window_size, 3*window_size).mean(dim=1)  

    top_k = 5
    score = gumbel_softmax_top_k(attn, top_k, tau=1.0, hard=True)
    end2 = time.time()
    print(f"Time taken: {end2 - start2:.6f}s") 
    print(attn.shape)
    print(score.shape)
    
    sparse_tensor = score.to_sparse()
    print(sparse_tensor.indices().shape)