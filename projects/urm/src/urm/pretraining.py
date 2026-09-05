"""Independently owned decoder pretraining harness for mixer comparisons."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from urm.compiler.anchors.sparse_memory import compile_sparse_memory_plan
from urm.compiler.semantic import DType, SDMExecutionMode, SparseMemoryMixerSpec

MixerBackend = Literal["upstream_sdm", "urm_native", "sdpa"]


@dataclass(frozen=True, slots=True)
class PretrainingConfig:
    vocab_size: int = 50_304
    sequence_length: int = 1_024
    layers: int = 12
    width: int = 768
    heads: int = 12
    value_dim: int = 64
    mlp_ratio: int = 4
    slots_per_partition: int = 4_096
    reads: int = 64
    writes: int = 64
    microbatch: int = 1
    gradient_accumulation: int = 4
    dropout: float = 0.0
    bias: bool = False

    def __post_init__(self) -> None:
        if self.width != self.heads * self.value_dim:
            raise ValueError("width must equal heads * value_dim")
        factor = round(self.slots_per_partition**0.5)
        if factor * factor != self.slots_per_partition:
            raise ValueError("slots_per_partition must be square")
        if self.reads > factor or self.writes > factor:
            raise ValueError("route widths must not exceed factor extent")
        if self.microbatch * self.heads > 16:
            raise ValueError("microbatch * heads exceeds native parallel envelope")

    @property
    def factor_extent(self) -> int:
        return round(self.slots_per_partition**0.5)

    @property
    def parallel(self) -> int:
        return self.microbatch * self.heads

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SemanticFlopLedger:
    embedding_projection: int
    normalization: int
    mlp: int
    logits_and_loss: int
    sparse_score_generation: int
    sparse_route_normalization: int
    sparse_state_update_read: int
    optimizer: int
    backward_included: bool
    uncredited_route_selection_comparisons: int

    @property
    def useful_total(self) -> int:
        return (
            self.embedding_projection
            + self.normalization
            + self.mlp
            + self.logits_and_loss
            + self.sparse_score_generation
            + self.sparse_route_normalization
            + self.sparse_state_update_read
            + self.optimizer
        )

    def to_dict(self) -> dict[str, int | bool]:
        return {**asdict(self), "useful_total": self.useful_total}


def semantic_training_flops(
    config: PretrainingConfig, backend: MixerBackend = "urm_native"
) -> SemanticFlopLedger:
    """Useful training FLOPs; sorting/padding/recomputation receive no credit."""
    b, t, c = config.microbatch, config.sequence_length, config.width
    l, h, d = config.layers, config.heads, config.value_dim
    f = config.factor_extent
    tokens = b * t * config.gradient_accumulation
    # A trained linear costs 2mnk forward and ~4mnk backward.
    linear_training = 6
    sparse_scores = l * linear_training * tokens * c * (h * 2 * f)
    sparse_values_gates = l * linear_training * tokens * c * (h * (d + 2))
    sparse_output = l * linear_training * tokens * c * c
    mlp = l * linear_training * tokens * (c * (4 * c) + (4 * c) * c)
    # Two LayerNorms/block plus final LayerNorm, forward+backward approximation.
    normalization = (2 * l + 1) * tokens * c * 24
    logits = linear_training * tokens * c * config.vocab_size
    loss = 10 * tokens * config.vocab_size
    route_norm = (
        l
        * b
        * h
        * t
        * config.gradient_accumulation
        * (config.reads + config.writes)
        * 5
    )
    # Useful selected-address recurrence only; top-k comparisons are uncredited.
    state_forward = (
        l
        * b
        * h
        * t
        * config.gradient_accumulation
        * (
            config.writes * d  # decay
            + 2 * config.writes * d  # retrieved reduction
            + 2 * d  # delta
            + 2 * config.writes * d  # scatter update
            + 2 * config.reads * d  # read reduction
        )
    )
    state_training = 3 * state_forward
    if backend == "sdpa":
        attention_projection = l * linear_training * tokens * 4 * c * c
        # QK^T and AV: 4*B*H*T*T*D forward; training is approximated as 3x.
        attention_state = 12 * l * b * h * t * t * d * config.gradient_accumulation
        sparse_projection = 0
        sparse_route = 0
        sparse_state = 0
        uncredited = 0
    else:
        attention_projection = 0
        attention_state = 0
        sparse_projection = sparse_scores + sparse_values_gates + sparse_output
        sparse_route = route_norm
        sparse_state = state_training
        uncredited = (
            l * b * h * t * config.gradient_accumulation * 2 * f * int(math.log2(f))
        )
    parameter_estimate = model_parameter_count(config, backend)
    optimizer = 10 * parameter_estimate
    return SemanticFlopLedger(
        embedding_projection=attention_projection,
        normalization=normalization,
        mlp=mlp,
        logits_and_loss=logits + loss,
        sparse_score_generation=sparse_projection,
        sparse_route_normalization=sparse_route,
        sparse_state_update_read=sparse_state + attention_state,
        optimizer=optimizer,
        backward_included=True,
        uncredited_route_selection_comparisons=uncredited,
    )


def model_parameter_count(
    config: PretrainingConfig, backend: MixerBackend = "urm_native"
) -> int:
    c, v, l, h, f = (
        config.width,
        config.vocab_size,
        config.layers,
        config.heads,
        config.factor_extent,
    )
    embeddings = v * c + config.sequence_length * c
    norms = (2 * l + 1) * 2 * c
    if backend == "sdpa":
        mixer = l * 4 * c * c
    else:
        score = l * (c * (h * 2 * f) + 2 * h * 2 * f)
        value_gate = l * c * (h * (config.value_dim + 2))
        output = l * c * c
        mixer = score + value_gate + output
    mlp = l * (c * (4 * c) + (4 * c) * c)
    return embeddings + norms + mixer + mlp


def tensor_bytes(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def model_memory_ledger(model: nn.Module, optimizer=None) -> dict[str, int]:
    parameters = list(model.parameters())
    gradients = [parameter.grad for parameter in parameters]
    optimizer_tensors = []
    if optimizer is not None:
        optimizer_tensors.extend(optimizer.state_tensors())
    persistent = [
        buffer
        for name, buffer in model.named_buffers()
        if name.endswith("persistent_memory")
    ]
    return {
        "parameter_bytes": tensor_bytes(parameters),
        "gradient_bytes": tensor_bytes(gradients),
        "optimizer_state_bytes": tensor_bytes(optimizer_tensors),
        "persistent_state_bytes": tensor_bytes(persistent),
    }


class FP32AdamW:
    """AdamW with FP32 master parameters and moment state for BF16 models."""

    def __init__(
        self,
        parameters,
        *,
        lr: float = 6e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ) -> None:
        items = list(parameters)
        if items and isinstance(items[0], tuple):
            named = [(str(name), parameter) for name, parameter in items]
        else:
            named = [
                (f"parameter_{index}", parameter)
                for index, parameter in enumerate(items)
            ]
        named = [
            (name, parameter) for name, parameter in named if parameter.requires_grad
        ]
        self.names = [name for name, _parameter in named]
        self.parameters = [parameter for _name, parameter in named]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.master = [
            parameter.detach().float().clone() for parameter in self.parameters
        ]
        self.exp_avg = [torch.zeros_like(parameter) for parameter in self.master]
        self.exp_avg_sq = [torch.zeros_like(parameter) for parameter in self.master]

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            parameter.grad = None if set_to_none else torch.zeros_like(parameter)

    @torch.no_grad()
    def step(self, *, record_updates: bool = False) -> dict[str, dict[str, float]]:
        self.step_count += 1
        correction1 = 1.0 - self.beta1**self.step_count
        correction2 = 1.0 - self.beta2**self.step_count
        update_sums: dict[str, torch.Tensor] = {}
        update_maxima: dict[str, torch.Tensor] = {}
        for name, parameter, master, mean, variance in zip(
            self.names,
            self.parameters,
            self.master,
            self.exp_avg,
            self.exp_avg_sq,
            strict=True,
        ):
            if parameter.grad is None:
                continue
            gradient = parameter.grad.float()
            mean.mul_(self.beta1).add_(gradient, alpha=1.0 - self.beta1)
            variance.mul_(self.beta2).addcmul_(
                gradient, gradient, value=1.0 - self.beta2
            )
            before = master.clone() if record_updates else None
            master.mul_(1.0 - self.lr * self.weight_decay)
            denominator = variance.sqrt().div_(math.sqrt(correction2)).add_(self.eps)
            master.addcdiv_(mean, denominator, value=-self.lr / correction1)
            parameter.copy_(master.to(parameter.dtype))
            if record_updates:
                delta = master - before
                group = parameter_group(name)
                squared = delta.square().sum()
                maximum = delta.abs().max()
                update_sums[group] = (
                    update_sums.get(group, squared.new_zeros(())) + squared
                )
                update_maxima[group] = torch.maximum(
                    update_maxima.get(group, maximum.new_zeros(())), maximum
                )
        return {
            group: {
                "l2": float(update_sums[group].sqrt().item()),
                "max_abs": float(update_maxima[group].item()),
            }
            for group in sorted(update_sums)
        }

    def state_tensors(self) -> tuple[torch.Tensor, ...]:
        return (*self.master, *self.exp_avg, *self.exp_avg_sq)


class SparseMemoryMixer(nn.Module):
    def __init__(self, config: PretrainingConfig, backend: MixerBackend) -> None:
        super().__init__()
        if backend not in {"upstream_sdm", "urm_native"}:
            raise ValueError(f"invalid sparse backend {backend}")
        self.config = config
        self.backend_name = backend
        c, h, f, d = config.width, config.heads, config.factor_extent, config.value_dim
        self.score = nn.Linear(c, h * 2 * f, bias=config.bias)
        self.read_score_bias = nn.Parameter(torch.zeros(h, 2 * f))
        self.write_score_bias = nn.Parameter(torch.empty(h, 2 * f))
        nn.init.normal_(self.write_score_bias, std=0.002)
        self.value_gate = nn.Linear(c, h * (d + 2), bias=config.bias)
        self.output = nn.Linear(c, c, bias=config.bias)
        self.register_buffer(
            "persistent_memory",
            torch.zeros(config.parallel, config.slots_per_partition, d),
            persistent=False,
        )
        self._pending_state = None
        self.profile_ranges = False
        spec = SparseMemoryMixerSpec(
            config.parallel,
            config.sequence_length,
            config.slots_per_partition,
            config.value_dim,
            config.writes,
            config.reads,
            DType.BFLOAT16,
            SDMExecutionMode.TRAINING,
        )
        object.__setattr__(self, "_spec", spec)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        try:
            if backend == "urm_native":
                object.__setattr__(self, "_executor", compile_sparse_memory_plan(spec))
            else:
                from urm.adapters.sparse_delta_memory import (
                    MODE_TRAINING,
                    UrmSparseDeltaMemoryAdapter,
                )

                adapter = UrmSparseDeltaMemoryAdapter(
                    slots_per_partition=config.slots_per_partition,
                    value_dim=config.value_dim,
                    num_writes=config.writes,
                    num_reads=config.reads,
                    chunk_size=16,
                    mode=MODE_TRAINING,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                object.__setattr__(self, "_executor", adapter)
                from urm.adapters.compiled_sparse_delta_memory import (
                    register_upstream_adapter,
                )

                object.__setattr__(
                    self,
                    "_compiled_upstream_handle",
                    register_upstream_adapter(adapter),
                )
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng:
                torch.cuda.set_rng_state_all(cuda_rng)

    def reset_state(self) -> None:
        self.persistent_memory.zero_()
        self._pending_state = None

    @torch.no_grad()
    def detach_state(self) -> None:
        if self._pending_state is not None:
            self.persistent_memory.copy_(self._pending_state.detach())
            self._pending_state = None

    def state_checksum(self) -> dict[str, float | int]:
        memory = self.persistent_memory.float()
        return {
            "elements": memory.numel(),
            "sum": float(memory.sum().item()),
            "mean": float(memory.mean().item()),
        }

    def _project(self, x):
        b, t, _ = x.shape
        h, f, d = self.config.heads, self.config.factor_extent, self.config.value_dim
        common = self.score(x).view(b, t, h, 2 * f).permute(0, 2, 1, 3)
        read_scores = (common + self.read_score_bias[None, :, None]).reshape(
            b * h, t, 2 * f
        )
        write_scores = (common + self.write_score_bias[None, :, None]).reshape(
            b * h, t, 2 * f
        )
        projected = self.value_gate(x).view(b, t, h, d + 2).permute(0, 2, 1, 3)
        values = projected[..., :d].reshape(b * h, t, d).contiguous()
        beta = torch.sigmoid(projected[..., d : d + 1]).reshape(b * h, t, 1)
        log_decay = -F.softplus(projected[..., d + 1 :]).reshape(b * h, t, 1)
        return (
            read_scores.contiguous(),
            write_scores.contiguous(),
            values,
            beta.contiguous(),
            log_decay.contiguous(),
        )

    def _profile(self, name: str):
        return (
            torch.autograd.profiler.record_function(name)
            if self.profile_ranges
            else nullcontext()
        )

    def forward(self, x):
        from urm.backends.sparse_state_mixer import SparseState

        b, t, _ = x.shape
        with self._profile("pretraining::sparse_memory::learned_projections"):
            read_scores, write_scores, values, beta, log_decay = self._project(x)
        memory = self.persistent_memory
        if self.backend_name == "urm_native":
            with self._profile("pretraining::sparse_memory::native_route"):
                prepared = self._executor.prepare(
                    read_scores,
                    write_scores=write_scores,
                    values=values,
                    beta=beta,
                    log_decay=log_decay,
                )
            with self._profile("pretraining::sparse_memory::native_state"):
                result = self._executor.execute(SparseState(memory), prepared)
            readings, final = result.readings, result.state.memory
        else:
            adapter = self._executor
            address = adapter.direct_calls["address"]
            with self._profile("pretraining::sparse_memory::upstream_route"):
                write_values, write_addresses = address(
                    write_scores, self.config.writes, self.config.factor_extent
                )
                read_values, read_addresses = address(
                    read_scores, self.config.reads, self.config.factor_extent
                )
                write_weights = adapter.layer.write_act(write_values)
                read_weights = adapter.layer.read_act(read_values)
            offsets = (
                torch.arange(
                    self.config.parallel, device=x.device, dtype=torch.int64
                ).view(-1, 1, 1)
                * self.config.slots_per_partition
            )
            flat_memory = memory.reshape(-1, self.config.value_dim)
            write_addresses = write_addresses + offsets
            read_addresses = read_addresses + offsets
            with self._profile("pretraining::sparse_memory::upstream_state"):
                if torch.compiler.is_compiling():
                    from urm.adapters.compiled_sparse_delta_memory import (
                        compiled_upstream_sdm_update,
                    )

                    readings, final = compiled_upstream_sdm_update(
                        flat_memory,
                        write_addresses,
                        write_weights,
                        values,
                        beta,
                        log_decay,
                        read_addresses,
                        read_weights,
                        self._compiled_upstream_handle,
                    )
                else:
                    readings, final = adapter.direct_calls["update"](
                        flat_memory + 0,
                        write_addresses,
                        write_weights,
                        values,
                        beta,
                        log_decay,
                        read_addresses,
                        read_weights,
                    )
                    # Upstream backward restores its mutable working memory.
                    final = final.clone()
            final = final.reshape_as(memory)
        self._pending_state = final
        readings = readings.view(b, self.config.heads, t, self.config.value_dim)
        readings = readings.permute(0, 2, 1, 3).reshape(b, t, self.config.width)
        return self.output(readings)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: PretrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=config.bias)
        self.output = nn.Linear(config.width, config.width, bias=config.bias)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        shape = (b, t, self.config.heads, self.config.value_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.output(y.transpose(1, 2).contiguous().view(b, t, c))


class MLP(nn.Module):
    def __init__(self, config: PretrainingConfig) -> None:
        super().__init__()
        hidden = config.mlp_ratio * config.width
        self.up = nn.Linear(config.width, hidden, bias=config.bias)
        self.down = nn.Linear(hidden, config.width, bias=config.bias)

    def forward(self, x):
        return self.down(F.gelu(self.up(x), approximate="tanh"))


class DecoderBlock(nn.Module):
    def __init__(self, config: PretrainingConfig, backend: MixerBackend) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.norm2 = nn.LayerNorm(config.width)
        self.mixer = (
            CausalSelfAttention(config)
            if backend == "sdpa"
            else SparseMemoryMixer(config, backend)
        )
        self.mlp = MLP(config)
        self.capture_mixer_input = False
        self.last_mixer_input = None

    def forward(self, x):
        mixer_input = self.norm1(x)
        if self.capture_mixer_input:
            mixer_input.retain_grad()
            self.last_mixer_input = mixer_input
        x = x + self.mixer(mixer_input)
        return x + self.mlp(self.norm2(x))


class URMDecoderLM(nn.Module):
    def __init__(self, config: PretrainingConfig, backend: MixerBackend) -> None:
        super().__init__()
        self.config = config
        self.backend_name = backend
        self.token = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.sequence_length, config.width)
        self.blocks = nn.ModuleList(
            DecoderBlock(config, backend) for _ in range(config.layers)
        )
        self.norm = nn.LayerNorm(config.width)
        self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.lm_head.weight = self.token.weight
        self.apply(self._initialize)
        self.profile_ranges = False

    def enable_profiling(self, enabled: bool = True) -> None:
        self.profile_ranges = enabled
        for mixer in self.sparse_mixers():
            mixer.profile_ranges = enabled

    @staticmethod
    def _initialize(module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def sparse_mixers(self):
        return [
            block.mixer
            for block in self.blocks
            if isinstance(block.mixer, SparseMemoryMixer)
        ]

    def reset_state(self) -> None:
        for mixer in self.sparse_mixers():
            mixer.reset_state()

    def detach_state(self) -> None:
        for mixer in self.sparse_mixers():
            mixer.detach_state()

    def discard_pending_state(self) -> None:
        """Discard functional final states from an evidence-only forward."""
        for mixer in self.sparse_mixers():
            mixer._pending_state = None

    def state_checksums(self) -> list[dict[str, float | int]]:
        return [mixer.state_checksum() for mixer in self.sparse_mixers()]

    def capture_mixer_input_gradients(self, enabled: bool) -> None:
        for block in self.blocks:
            block.capture_mixer_input = enabled
            if not enabled:
                block.last_mixer_input = None

    def mixer_input_gradients(self) -> list[torch.Tensor]:
        gradients = []
        for block in self.blocks:
            value = block.last_mixer_input
            if value is None or value.grad is None:
                raise RuntimeError("mixer input gradient was not captured")
            gradients.append(value.grad.detach())
        return gradients

    def forward(self, tokens, targets=None):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        context = (
            torch.autograd.profiler.record_function if self.profile_ranges else None
        )
        with context("pretraining::embeddings") if context else nullcontext():
            x = self.token(tokens) + self.position(positions)[None]
        for index, block in enumerate(self.blocks):
            with context(f"pretraining::block_{index}") if context else nullcontext():
                x = block(x)
        with context("pretraining::logits") if context else nullcontext():
            logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            with context("pretraining::loss") if context else nullcontext():
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)
                )
        return logits, loss


def parameter_group(name: str) -> str:
    if name.startswith(("token.", "position.", "lm_head.")):
        return "embeddings_and_logits"
    if name.startswith("blocks."):
        parts = name.split(".")
        return f"block_{parts[1]}_{parts[2]}"
    return "final_norm"


def gradient_norms(model: nn.Module) -> dict[str, float]:
    totals: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = parameter_group(name)
        squared = parameter.grad.float().square().sum()
        totals[group] = totals.get(group, squared.new_zeros(())) + squared
    return {
        group: float(value.sqrt().item()) for group, value in sorted(totals.items())
    }


__all__ = [
    "CausalSelfAttention",
    "FP32AdamW",
    "MixerBackend",
    "PretrainingConfig",
    "SemanticFlopLedger",
    "SparseMemoryMixer",
    "URMDecoderLM",
    "gradient_norms",
    "model_memory_ledger",
    "model_parameter_count",
    "parameter_group",
    "semantic_training_flops",
    "tensor_bytes",
]
