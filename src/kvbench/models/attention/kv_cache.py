from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class KVCacheState:
    keys: torch.Tensor
    values: torch.Tensor
