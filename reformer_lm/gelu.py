import torch
import torch.nn as nn
import math


class GeLU(nn.Module):
    """
    Implementing Gaussian Error Linear Unit
    """

    def forward(self, x):
        b1 = math.sqrt(2 / math.pi)
        b2 = x + 0.044715 * torch.pow(x, 3)
        w = 1 + torch.tanh(b1 * b2)
        x = 0.5 * x * w

        return x
