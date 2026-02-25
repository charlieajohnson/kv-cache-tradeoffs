from __future__ import annotations

import torch


def next_token_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = torch.argmax(logits[:, -1], dim=-1)
    gold = targets[:, -1]
    return float((pred == gold).float().mean().item())
