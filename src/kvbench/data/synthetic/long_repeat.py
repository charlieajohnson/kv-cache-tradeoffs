from __future__ import annotations


class LongRepeatDataset:
    def __init__(self, size: int = 100, seq_len: int = 128):
        self.samples = ["ABCD" * (seq_len // 4) for _ in range(size)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]
