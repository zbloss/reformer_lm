import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from reformer_lm.attention import ComputeAttentionHeads, ComputeAttentionOutput
from reformer_lm.chunk import Unchunk
from reformer_lm.broadcasted_dropout import BroadcastedDropout


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        attn_k=64,
        attn_v=64,
        n_heads=1,
        n_chunks=2,
        share_qk=True,
        attn_type=None,
        dropout=None,
        ff_activation=None,
        ff_use_sru=None,
        mode="train",
    ):
        super(DecoderBlock, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.attn_k = attn_k
        self.attn_v = attn_v
        self.n_heads = n_heads
        self.n_chunks = n_chunks
        self.attn_type = attn_type
        self.dropout = dropout
        self.share_qk = share_qk
        self.ff_activation = ff_activation
        self.ff_use_sru = ff_use_sru
        self.mode = mode

    def pre_attention(self, x):

        x1, x2 = torch.chunk(x, self.n_chunks)
        k_layers = [
            ComputeAttentionHeads(self.n_heads, self.attn_k),
            nn.LayerNorm((x.shape[1], x.shape[2])),
        ]
        k_model = nn.Sequential(*k_layers)

        v_layers = [
            ComputeAttentionHeads(self.n_heads, self.attn_v),
            nn.LayerNorm((x.shape[1], x.shape[2])),
        ]
        v_model = nn.Sequential(*v_layers)

        k = k_model(x1)
        v = v_model(x2)

        if not self.share_qk:
            q_layers = k_layers
            q_model = nn.Sequential(*q_layers)
            q = q_model(x1)

            return (q, k, v)
        else:
            return (k, k, v)

    def attention(self, inputs):

        assert len(inputs) == 2 or len(inputs) == 3
        if len(inputs) == 2:
            k, v = inputs
            q = k
        else:
            q, k, v = inputs

        mask_size = q.shape[-2]
        mask = torch.tril(
            torch.ones((1, mask_size, mask_size), dtype=torch.bool), diagonal=0
        )

        attn = self.dotproductattention(q, k, v, mask)
        return attn

    def dotproductattention(self, q, k, v, mask, dropout=0.1):

        depth = q.shape[-1]
        dots = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(depth)
        dots = F.log_softmax(
            torch.where(mask, dots, torch.full_like(dots, -1e9)), dim=0
        )

        keep_prob = 1 - dropout
        keep = np.random.binomial(n=1, p=keep_prob, size=dots.shape)

        dots = torch.where(
            torch.tensor(keep, dtype=torch.bool),
            dots / torch.tensor(keep_prob),
            torch.zeros_like(dots),
        )
        attn = torch.matmul(dots, v)
        return attn

    def post_attention(self, x):

        cao = ComputeAttentionOutput()
        unchunk = Unchunk(n_sections=self.n_chunks, dim=-2)
        bd = BroadcastedDropout(rate=self.dropout)

        res = cao(x)
        # res = torch.cat((res, res), dim=-3)
        res = unchunk(res)
        res = bd(res)
        return res

    def forward(self, x):

        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)

        x = self.pre_attention(x)
        # x = tuple(torch.tensor(y) for y in x)
        x = self.attention(x)
        x = self.post_attention(x)
        return torch.cat((x, x))
