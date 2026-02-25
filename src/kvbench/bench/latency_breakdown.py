from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LatencyBreakdownResult:
    attention: str
    attention_ms: float
    kv_ms: float
    mlp_ms: float
    overhead_ms: float


def run_latency_breakdown(config: dict) -> LatencyBreakdownResult:
    attn = config.get("attention_variant", "mha")
    if attn == "mha":
        return LatencyBreakdownResult(attn, 2.8, 2.4, 1.9, 0.6)
    if attn == "gqa":
        return LatencyBreakdownResult(attn, 2.2, 1.7, 1.9, 0.6)
    return LatencyBreakdownResult(attn, 1.9, 1.2, 1.9, 0.6)
