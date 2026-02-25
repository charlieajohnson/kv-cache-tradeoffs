from __future__ import annotations

import torch


def perplexity(logits: torch.Tensor, targets: torch.Tensor) -> float:
    log_probs = torch.log_softmax(logits, dim=-1)
    n = targets.numel()
    target_log_prob = log_probs.view(-1, log_probs.size(-1)).gather(1, targets.view(-1, 1)).squeeze()
    return float(torch.exp(-target_log_prob.mean()).item())
