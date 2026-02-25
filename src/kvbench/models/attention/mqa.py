from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MQA(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.kv_heads = 1
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, self.d_head, bias=False)
        self.v = nn.Linear(d_model, self.d_head, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q = self.q(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(x).view(b, t, 1, self.d_head).transpose(1, 2)
        v = self.v(x).view(b, t, 1, self.d_head).transpose(1, 2)
        scores = torch.einsum("bhtd,bhsd->bhts", q, k.expand(-1, self.n_heads, -1, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        y = torch.einsum("bhts,bhsd->bhtd", self.dropout(attn), v.expand(-1, self.n_heads, -1, -1))
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)
