import torch
import torch.nn as nn
import numpy as np


class BroadcastedDropout(nn.Module):
    def __init__(self, rate=0.0, mode="train", broadcast_dims=(-2,)):
        super(BroadcastedDropout, self).__init__()

        self.rate = rate
        if self.rate >= 1.0:
            raise ValueError(f"Dropout rate ({self.rate}) must be < 1")
        elif self.rate < 0:
            raise ValueError(f"Dropout rate ({self.rate}) must be >= 0.0")

        self.broadcast_dims = broadcast_dims
        self.mode = mode

    def forward(self, x: torch.tensor, **kwargs):
        if self.mode == "train" and self.rate > 0.0:
            noise_shape = list(x.shape)

            for dim in self.broadcast_dims:
                noise_shape[dim] = 1

            keep_prob = 1 - self.rate
            keep = np.random.binomial(
                n=1, p=keep_prob, size=tuple(noise_shape)
                )
            keep = torch.tensor(keep)
            multiplier = keep / keep_prob
            return x * multiplier
        else:
            return x
