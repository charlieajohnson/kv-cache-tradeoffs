from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from kvbench.models.attention import get_attention_class


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
