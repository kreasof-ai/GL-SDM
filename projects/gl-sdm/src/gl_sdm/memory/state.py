"""Frozen views, buffered proposals, request caches and the learned FP32 bank."""
from dataclasses import dataclass, field
import torch
from torch import nn


@dataclass(frozen=True)
class MemoryView:
    values: torch.Tensor  # [requests, heads, slots, value_dim], FP32
    version: int = 0
    lineage: object = field(default_factory=object)

    def __post_init__(self):
        if self.values.ndim != 4 or self.values.dtype != torch.float32 or min(self.values.shape) < 1 or self.version < 0:
            raise ValueError("memory views require nonempty [batch, heads, slots, dim] FP32 values and a nonnegative version")


@dataclass(frozen=True)
class WriteProposal:
    base_version: int
    addresses: torch.Tensor  # flattened request/head/slot addresses
    deltas: torch.Tensor     # [entries, value_dim], already weighted
    lineage: object
    sort_keys: torch.Tensor | None = None  # optional address/token/depth order


@dataclass(frozen=True)
class WriteBuffer:
    base_version: int
    proposals: tuple[WriteProposal, ...]
    lineage: object


@dataclass
class MemoryCache:
    view: MemoryView
    owner: object
    tokens: int = 0
    pending: tuple[WriteProposal, ...] = ()
    local: dict = field(default_factory=dict)


class MemoryBank(nn.Module):
    def __init__(self, heads, slots, dim):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(heads, slots, dim) * (heads * dim) ** -0.5)
        self.memory._sdm_memory_bank = True
        self.owner = object()

    def _apply(self, fn, recurse=True):
        # Reasoner weights may be BF16; transactional state and its learned
        # initializer remain FP32, preserving the original initializer values.
        values = self.memory.detach()
        gradient = None if self.memory.grad is None else self.memory.grad.detach()
        super()._apply(fn, recurse)
        self.memory.data = values.to(device=self.memory.device, dtype=torch.float32)
        if gradient is not None:
            self.memory.grad.data = gradient.to(device=self.memory.device, dtype=torch.float32)
        return self

    def snapshot(self, batch_size):
        return MemoryView(self.memory.unsqueeze(0).expand(batch_size, -1, -1, -1).clone())

    def new_cache(self, batch_size):
        return MemoryCache(self.snapshot(batch_size), self.owner)

    def validate_cache(self, cache, batch_size):
        expected = (batch_size, *self.memory.shape)
        if cache.owner is not self.owner or cache.view.values.shape != expected or cache.view.values.device != self.memory.device or cache.view.values.dtype != torch.float32:
            raise ValueError("memory cache belongs to another model, batch, device or dtype")
