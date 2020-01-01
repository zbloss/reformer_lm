import torch
import torch.nn as nn


class Chunk(nn.Module):
    def __init__(self, n_sections=2):
        super(Chunk, self).__init__()
        self.n_sections = n_sections

    def forward(self, x):
        assert x.shape[1] % self.n_sections == 0
        return torch.cat(
            torch.chunk(x, chunks=self.n_sections, dim=-2)
            )


class Unchunk(nn.Module):
    def __init__(self, n_sections=2, dim=-3):
        super(Unchunk, self).__init__()
        self.n_sections = n_sections
        self.dim = dim

    def forward(self, x):
        assert x.shape[0] % self.n_sections == 0
        return torch.cat(
            torch.chunk(x, chunks=self.n_sections, dim=self.dim), dim=-2
            )
