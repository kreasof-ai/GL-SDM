"""ATMA/nanoGPT shard format, with portable paths and explicit exhaustion."""
import glob
from pathlib import Path
import numpy as np
import torch


def load_shard(path):
    path = Path(path)
    header = np.fromfile(path, dtype=np.int32, count=256)
    if len(header) != 256 or header[0] != 20240520 or header[1] != 1:
        raise ValueError(f"invalid token shard header: {path}")
    width = int(header[3]) or 2
    if width not in (2, 4) or path.stat().st_size != 1024 + int(header[2]) * width:
        raise ValueError(f"invalid token shard size: {path}")
    return np.memmap(path, mode="r", offset=1024, dtype=np.uint16 if width == 2 else np.uint32, shape=(int(header[2]),))


def data_generator(pattern, batch_size, seq_len=1024, device="cuda", num_chunks=None):
    if batch_size % seq_len:
        raise ValueError("batch_size counts tokens and must be divisible by seq_len")
    files = sorted(glob.glob(str(pattern)))
    if num_chunks is not None:
        files = files[:num_chunks]
    if not files:
        raise FileNotFoundError(f"no token shards match {pattern}")
    for path in files:
        tokens = load_shard(path)
        for pos in range(0, len(tokens) - batch_size, batch_size):
            buf = torch.from_numpy(np.array(tokens[pos:pos + batch_size + 1], dtype=np.int64))
            buf = buf.to(device)
            yield buf[:-1].view(-1, seq_len), buf[1:].view(-1, seq_len)


def available_steps(pattern, batch_size, num_chunks):
    return sum((len(load_shard(p)) - 1) // batch_size for p in sorted(glob.glob(pattern))[:num_chunks])
