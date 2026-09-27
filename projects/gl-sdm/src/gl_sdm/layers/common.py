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
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 8 * dim)
        self.proj = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x, gate = self.fc(x).chunk(2, -1)
        return self.proj(gate * x.relu().square())
