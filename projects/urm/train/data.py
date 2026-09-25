"""Shard data layer for the URM training harness (ATMA pattern).

Loads the finewebedu10B gpt2-tokenized shards produced by the ATMA
``train/data.py`` writer (256-int32 header + uint16/uint32 token body) and yields
contiguous ``(inputs, targets)`` windows. A synthetic random-token generator covers
self-contained correctness runs when no shard is present; the real-shard path is the
50%-MFU gate's data source.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


def get_data(fname: str, repo_id: str = "kjj0/finewebedu10B-gpt2", local_dir: str = "finewebedu10B") -> str:
    """Download one shard if absent; returns the local path."""
    from huggingface_hub import hf_hub_download

    path = os.path.join(local_dir, fname)
    if not os.path.exists(path):
        hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset", local_dir=local_dir)
    return path


def _load_data_shard(file: Path) -> torch.Tensor:
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    token_bytes = int(header[3]) or 2  # 0 = legacy file, default to uint16
    assert token_bytes in (2, 4), f"unsupported token width in header: {token_bytes}"
    torch_dtype = torch.uint16 if token_bytes == 2 else torch.int32
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch_dtype, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == token_bytes * num_tokens, "number of tokens read does not match header"
    return tokens


def shard_token_count(file: Path) -> int:
    """Read only the header; the shard body stays on disk."""
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520 and header[1] == 1
    return int(header[2])


def data_generator(filename_pattern: str, batch_size: int, seq_len: int, *, device: str = "cuda"):
    """Yield (inputs, targets) windows of shape (batch_size//seq_len, seq_len) forever."""
    files = sorted(Path.cwd().glob(filename_pattern))
    if not files:
        raise FileNotFoundError(f"no shards match {filename_pattern!r} under {Path.cwd()}")
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos: pos + batch_size + 1]
        inputs = buf[:-1].to(device=device, dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device=device, dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


def synthetic_generator(batch_size: int, seq_len: int, vocab_size: int, *,
                        device: str = "cuda", seed: int = 0):
    """Self-contained random-token stream (no dataset) for correctness plumbing runs.

    Fixed-seed uniform tokens: enough to drive the model/optimizer/checkpoint path
    deterministically without shipping a dataset. Not a learning signal.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    while True:
        buf = torch.randint(0, vocab_size, (batch_size + 1,), generator=gen)
        inputs = buf[:-1].to(device=device, dtype=torch.int32)
        targets = buf[1:].to(device=device, dtype=torch.int64)
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


__all__ = ["get_data", "data_generator", "synthetic_generator", "shard_token_count"]
