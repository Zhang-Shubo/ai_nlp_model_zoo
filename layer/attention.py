# time: 2021/8/22 0:20
# File: attention.py
# Author: zhangshubo
# Mail: supozhang@126.com
import torch
import torch.nn.functional as F


def attention(query, key, value, attention_mask):
    attn_weights = torch.matmul(query, torch.transpose(key, -1, -2))
    attn_weights = attn_weights / torch.sqrt(torch.tensor(key.shape[-1]))

    attention_mask = torch.tensor(1) - attention_mask
    if attention_mask is not None:
        attn_weights = attn_weights + (attention_mask * torch.tensor(-1e9)).unsqueeze(1)
    soft_attn_weights = F.softmax(attn_weights, -1)
    context = torch.matmul(soft_attn_weights, value)
    return context


if __name__ == '__main__':
    k = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]], dtype=torch.float)  # (1, 4, 3)
    v = torch.tensor([[[10, 0], [0, 10], [20, 5], [10, 5]]], dtype=torch.float)  # (1, 4, 2)

    q = torch.tensor([[[0, 0, 5]]], dtype=torch.float)  # (1, 1, 3)
    print(attention(q, k, v, None))