from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KVCacheState:
    keys: torch.Tensor
    values: torch.Tensor

    def num_bytes(self) -> int:
        return self.keys.numel() * self.keys.element_size() + self.values.numel() * self.values.element_size()
