from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.einsum("bhtd,bhsd->bhts", q, k) * self.scale
        attn = F.softmax(scores, dim=-1)
        y = torch.einsum("bhts,bhsd->bhtd", self.dropout(attn), v)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)
