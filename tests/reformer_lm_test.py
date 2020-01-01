import sys
print(sys.path)


import torch
from .reformer.reformer_lm import ReformerLM

test = torch.rand((4, 4, 64000))
model = ReformerLM(
    vocab_size=300000,
    d_in=test.shape[-2],
    d_out=test.shape[-1],
    n_layers=6,
    n_heads=1,
    attn_k=test.shape[-1],
    attn_v=test.shape[-1],
)

output = model(test)