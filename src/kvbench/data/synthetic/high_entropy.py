from __future__ import annotations

import random
import string


class HighEntropyDataset:
    def __init__(self, size: int = 100, seq_len: int = 128, seed: int = 0):
        random.seed(seed)
        self.samples = [
            "".join(random.choice(string.ascii_uppercase) for _ in range(seq_len))
            for _ in range(size)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]
