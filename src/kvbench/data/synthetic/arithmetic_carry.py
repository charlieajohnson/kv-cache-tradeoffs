from __future__ import annotations


class ArithmeticCarryDataset:
    def __init__(self, size: int = 100, seq_len: int = 128):
        self.samples = [("12" * (seq_len // 2)) for _ in range(size)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]
