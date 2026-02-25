from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GQA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int | None = None, dropout: float = 0.0):
        super().__init__()
        if n_kv_heads is None:
            n_kv_heads = max(1, n_heads // 2)
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.group_size = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q = self.q(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(x).view(b, t, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v(x).view(b, t, self.n_kv_heads, self.d_head).transpose(1, 2)
        kv_repeat = self.group_size
        k = k.repeat_interleave(kv_repeat, dim=1)
        v = v.repeat_interleave(kv_repeat, dim=1)
        scores = torch.einsum("bhtd,bhsd->bhts", q, k) * self.scale
        attn = F.softmax(scores, dim=-1)
        y = torch.einsum("bhts,bhsd->bhtd", self.dropout(attn), v)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)
