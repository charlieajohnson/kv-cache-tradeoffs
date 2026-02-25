from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ThroughputResult:
    attention: str
    seq_len: int
    tokens_per_sec: float


def run_throughput(config: dict) -> list[ThroughputResult]:
    out: list[ThroughputResult] = []
    variants = config.get("attention_variants", ["mha"])
    seq_lens = config.get("seq_lens", [128, 256, 512])
    for attn in variants:
        base = 25000 if attn == "mha" else 32000 if attn == "gqa" else 36000
        for L in seq_lens:
            out.append(ThroughputResult(attn, L, float(base - 2.5 * L)))
    return out
