from __future__ import annotations

import random


class ShakespeareDataset:
    def __init__(self, size: int = 64, seed: int = 0):
        random.seed(seed)
        self.samples = ["To be or not to be."] * size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> str:
        return self.samples[idx]
