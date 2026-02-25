from __future__ import annotations

import torch


def bytes_to_mib(nbytes: int) -> float:
    return nbytes / (1024.0**2)


def estimate_kv_cache_bytes(
    seq_len: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
) -> int:
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    # 2 for keys and values
    return seq_len * n_layers * n_kv_heads * head_dim * 2 * bytes_per
