from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn

from kvbench.models.attention import get_attention_class
from kvbench.models.attention.kv_cache import KVCacheState


@dataclass
class DecoderOnlyConfig:
    vocab_size: int = 50257
    layers: int = 6
    d_model: int = 384
    n_heads: int = 6
    n_kv_heads: int = 6
    seq_len: int = 1024
    n_kv_groups: int = 1
    attn_dropout: float = 0.0
    attn_variant: str = "mha"


class TinyGPTBlock(nn.Module):
    def __init__(self, cfg: DecoderOnlyConfig):
        super().__init__()
        attention_cls = get_attention_class(cfg.attn_variant)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = attention_cls(cfg.d_model, cfg.n_heads, cfg.n_kv_heads)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 4),
            nn.GELU(),
            nn.Linear(cfg.d_model * 4, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def forward_with_cache(
        self, x: torch.Tensor, cache: KVCacheState | None = None
    ) -> tuple[torch.Tensor, KVCacheState]:
        y, next_cache = self.attn(self.norm1(x), cache)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x, next_cache

    def forward_with_timing(
        self, x: torch.Tensor, cache: KVCacheState | None = None
    ) -> tuple[torch.Tensor, KVCacheState, dict[str, float]]:
        y, next_cache, timings = self.attn.forward_with_cache_timed(self.norm1(x), cache)
        x = x + y
        t1 = perf_counter()
        x = x + self.mlp(self.norm2(x))
        mlp_ms = perf_counter() - t1

        # Keep timing granularity stable for CLI/reporting without extra dependencies.
        # The model-level cache timing is supplied directly by attention.
        return (
            x,
            next_cache,
            {
                "attention_ms": float(timings["attention_ms"]),
                "mlp_ms": mlp_ms * 1000,
                "cache_update_ms": float(timings["cache_update_ms"]),
            },
        )


class SmallGPT(nn.Module):
    def __init__(self, cfg: DecoderOnlyConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([TinyGPTBlock(cfg) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(token_ids.size(1), device=token_ids.device)
        x = self.embed(token_ids) + self.pos(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        return logits

    def forward_with_cache(
        self,
        token_ids: torch.Tensor,
        caches: list[KVCacheState] | None = None,
    ) -> tuple[torch.Tensor, list[KVCacheState]]:
        if caches is None:
            caches = [None] * len(self.blocks)  # type: ignore[list-item]
        if len(caches) != len(self.blocks):
            raise ValueError("cache list length must match number of blocks")

        positions = torch.arange(token_ids.size(1), device=token_ids.device)
        x = self.embed(token_ids) + self.pos(positions)[None, :, :]

        next_caches: list[KVCacheState] = []
        for block, cache in zip(self.blocks, caches):
            x, next_cache = block.forward_with_cache(x, cache)
            next_caches.append(next_cache)
        logits = self.lm_head(self.norm(x))
        return logits, next_caches

    def forward_with_timing(
        self,
        token_ids: torch.Tensor,
        caches: list[KVCacheState] | None = None,
    ) -> tuple[torch.Tensor, list[KVCacheState], dict[str, float]]:
        t0 = perf_counter()
        if caches is None:
            caches = [None] * len(self.blocks)  # type: ignore[list-item]
        if len(caches) != len(self.blocks):
            raise ValueError("cache list length must match number of blocks")

        positions = torch.arange(token_ids.size(1), device=token_ids.device)
        x = self.embed(token_ids) + self.pos(positions)[None, :, :]

        total_attention_ms = 0.0
        total_mlp_ms = 0.0
        total_cache_ms = 0.0
        next_caches: list[KVCacheState] = []
        for block, cache in zip(self.blocks, caches):
            x, next_cache, timing = block.forward_with_timing(x, cache)
            total_attention_ms += timing["attention_ms"]
            total_mlp_ms += timing["mlp_ms"]
            total_cache_ms += timing["cache_update_ms"]
            next_caches.append(next_cache)

        logits = self.lm_head(self.norm(x))
        total_ms = (perf_counter() - t0) * 1000
        overhead_ms = max(
            0.0,
            total_ms
            - total_attention_ms
            - total_mlp_ms
            - total_cache_ms,
        )
        return (
            logits,
            next_caches,
            {
                "attention_ms": total_attention_ms,
                "mlp_ms": total_mlp_ms,
                "cache_update_ms": total_cache_ms,
                "overhead_ms": overhead_ms,
                "total_ms": total_ms,
            },
        )
