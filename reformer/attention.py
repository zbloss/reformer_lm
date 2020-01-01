import torch
import torch.nn as nn


class ComputeAttentionHeads(nn.Module):
    def __init__(self, n_heads=1, d_head=64):
        super(ComputeAttentionHeads, self).__init__()
        self.n_heads = n_heads
        self.d_head = d_head

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float)

        seqlen = x.shape[1]
        res = x

        # n_batch, seqlen, n_heads*d_head -> n_batch, seqlen, n_heads, d_head
        res = torch.reshape(res,
                            (x.shape[0], seqlen, self.n_heads, self.d_head))
        # n_batch, seqlen, n_heads, d_head -> n_batch, n_heads, seqlen, d_head
        res = torch.transpose(res, 1, 2)
        # n_batch, n_heads, seqlen, d_head -> n_batch*n_heads, seqlen, d_head
        res = torch.reshape(res, (-1, seqlen, self.d_head))
        res = nn.Linear(res.shape[-1], res.shape[-1])(res)
        return res


class ComputeAttentionOutput(nn.Module):
    def __init__(self, n_heads=1):
        super(ComputeAttentionOutput, self).__init__()
        self.n_heads = n_heads

    def forward(self, x):

        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)

        seqlen = x.shape[1]
        d_head = x.shape[2]

        x = torch.reshape(x, (-1, self.n_heads, seqlen, d_head))
        x = torch.transpose(x, 1, 2)
        x = torch.reshape(x, (-1, seqlen, self.n_heads * d_head))
        x = nn.Linear(x.shape[-1], x.shape[-1])(x)
        return x
