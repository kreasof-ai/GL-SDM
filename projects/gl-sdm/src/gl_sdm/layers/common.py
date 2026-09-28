"""ATMA normalization and gated squared-ReLU feed-forward layers."""
import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), eps=1e-6)


class MLP(nn.Module):
    def __init__(self, dim, intermediate_size=None):
        super().__init__()
        inner = 4 * dim if intermediate_size is None else intermediate_size
        if inner < 1:
            raise ValueError("intermediate_size must be positive")
        self.fc = nn.Linear(dim, 2 * inner)
        self.proj = nn.Linear(inner, dim)

    def forward(self, x):
        x, gate = self.fc(x).chunk(2, -1)
        return self.proj(gate * x.relu().square())
