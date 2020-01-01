import torch
import torch.nn as nn
from reformer_lm.gelu import GeLU


class RevNetBlock(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.1, lol=[]):
        super(RevNetBlock, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.dropout = dropout

        layers = []
        if lol == list():
            layers.append(nn.LayerNorm((d_in, d_out)))
            layers.append(nn.Linear(d_in, d_out))
            layers.append(GeLU())
            layers.append(nn.Linear(d_in, d_out))
        else:
            for layer in lol:
                layers.append(layer)

        self.bottleneck_block = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.cat((x, x), dim=1)
        x1, x2 = self.split(x)
        Fx2 = self.bottleneck_block(x2)
        y1 = Fx2 + x1
        return (x2, y1)

    def inverse(self, x):
        x2, y1 = x[0], x[1]
        Fx2 = -self.bottleneck_block(x2)
        x1 = Fx2 + y1
        return (x1, x2)

    @staticmethod
    def split(x):
        n = int(x.size()[1] / 2)
        x1 = x[:, :n].contiguous()
        x2 = x[:, n:].contiguous()
        return (x1, x2)


class RevNetHalfAttnBlock(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.1, lol=[]):
        super(RevNetHalfAttnBlock, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.dropout = dropout

        layers = []
        if lol == list():
            layers.append(nn.LayerNorm((d_in, d_out)))
            layers.append(nn.Linear(d_out, d_out))
            layers.append(GeLU())
            layers.append(nn.Linear(d_out, d_out))
        else:
            for layer in lol:
                layers.append(layer)

        self.bottleneck_block = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.cat((x, x), dim=1)
        x1, x2 = self.split(x)
        Fx2 = self.bottleneck_block(x2)
        y1 = Fx2 + x1
        return (x2, y1)

    def inverse(self, x):
        x2, y1 = x[0], x[1]
        Fx2 = -self.bottleneck_block(x2)
        x1 = Fx2 + y1
        return (x1, x2)

    @staticmethod
    def split(x):
        n = int(x.size()[1] / 2)
        x1 = x[:, :n].contiguous()
        x2 = x[:, n:].contiguous()
        return (x1, x2)
