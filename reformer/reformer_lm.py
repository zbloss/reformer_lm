import torch.nn as nn
from .decoder import DecoderBlock
from .broadcasted_dropout import BroadcastedDropout
from .gelu import GeLU


class ReformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_in,
        d_out,
        attn_k=64,
        attn_v=64,
        n_layers=6,
        n_heads=1,
        dropout=0.1,
        max_len=2048,
        n_chunks=2,
        n_attention_chunks=2,
        share_qk=True,
        axial_pos_shape=(),
        d_axial_pos_embs=None,
        mode="train",
    ):
        super(ReformerLM, self).__init__()

        self.vocab_size = vocab_size
        self.d_in = d_in
        self.d_out = d_out
        self.attn_k = attn_k
        self.attn_v = attn_v
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.max_len = max_len
        self.n_chunks = n_chunks
        self.n_attention_chunks = n_attention_chunks
        self.share_qk = share_qk
        self.axial_pos_shape = axial_pos_shape
        self.d_axial_pos_embs = d_axial_pos_embs
        self.mode = mode

        self.layers = []
        self.layers.append(
            DecoderBlock(
                d_in=self.d_in,
                d_out=self.d_out,
                attn_k=self.attn_k,
                attn_v=self.attn_v,
                n_heads=self.n_heads,
                n_chunks=self.n_attention_chunks,
                share_qk=self.share_qk,
                attn_type=None,
                dropout=self.dropout,
            )
        )

        for layer in range(self.n_layers - 1):
            # self.layers.append(Chunk(n_sections=self.n_attention_chunks))
            self.layers.append(
                DecoderBlock(
                    d_in=self.d_out,
                    d_out=self.d_out,
                    attn_k=self.attn_k,
                    attn_v=self.attn_v,
                    n_heads=self.n_heads,
                    n_chunks=self.n_attention_chunks,
                    share_qk=self.share_qk,
                    attn_type=None,
                    dropout=self.dropout,
                )
            )

        self.ff_layers = [
            nn.LayerNorm((1, self.d_out * self.d_in)),
            nn.Linear(self.d_out * self.d_in, self.d_out * self.d_in),
            BroadcastedDropout(rate=self.dropout, mode=self.mode),
            GeLU(),
            nn.Linear(self.d_out * self.d_in, self.vocab_size),
            nn.LogSoftmax(dim=0),
        ]

        self.model = nn.Sequential(*self.layers)
        self.ff_model = nn.Sequential(*self.ff_layers)

    def forward(self, x):

        x = self.model(x)
        # Flattening
        x = x.view(x.shape[0], 1, -1)
        x = self.ff_model(x)

        return x
