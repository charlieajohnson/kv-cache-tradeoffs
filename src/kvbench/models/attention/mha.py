from __future__ import annotations

from time import perf_counter

import torch
import torch.nn.functional as F
from torch import nn

from kvbench.models.attention.kv_cache import KVCacheState


class MHA(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head**-0.5

    def _causal_mask(self, q_len: int, kv_len: int, cache_len: int, device: torch.device) -> torch.Tensor:
        query_pos = torch.arange(q_len, device=device) + cache_len
        key_pos = torch.arange(kv_len, device=device)
        return key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)

    def forward_with_cache(self, x: torch.Tensor, cache: KVCacheState | None = None) -> tuple[torch.Tensor, KVCacheState]:
        y, cache_update = self.forward_with_cache_timed(x, cache)
        return y, cache_update

    def forward_with_cache_timed(
        self,
        x: torch.Tensor,
        cache: KVCacheState | None = None,
    ) -> tuple[torch.Tensor, KVCacheState, dict[str, float]]:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        cache_len = 0 if cache is None else cache.keys.size(2)
        cache_start = perf_counter()
        if cache is not None:
            k = torch.cat([cache.keys, k], dim=2)
            v = torch.cat([cache.values, v], dim=2)
        cache_update_ms = (perf_counter() - cache_start) * 1000

        kv_len = k.size(2)
        scores = torch.einsum("bhtd,bhsd->bhts", q, k) * self.scale
        mask = self._causal_mask(t, kv_len, cache_len, x.device)
        scores = scores.masked_fill(~mask[None, None, :, :], torch.finfo(scores.dtype).min)
        attention_start = perf_counter()
        attn = F.softmax(scores, dim=-1)
        y = torch.einsum("bhts,bhsd->bhtd", self.dropout(attn), v)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        attention_ms = (perf_counter() - attention_start) * 1000

        y = self.proj(y)
        return (
            y,
            KVCacheState(k, v),
            {"attention_ms": attention_ms, "cache_update_ms": cache_update_ms},
        )

    def forward(self, x: torch.Tensor, cache: KVCacheState | None = None) -> torch.Tensor:
        return self.forward_with_cache(x, cache)[0]
