from __future__ import annotations

import random


class OpenWebTextSubset:
    def __init__(self, size: int = 64, seed: int = 0):
        random.seed(seed)
        self.samples = [f"openweb-{i}" for i in range(size)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> str:
        return self.samples[idx]
