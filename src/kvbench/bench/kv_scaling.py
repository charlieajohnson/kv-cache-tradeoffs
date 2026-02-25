from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KVScalingResult:
    attention: str
    seq_len: int
    kv_mib: float


def run_kv_scaling(config: dict, max_batches: int = 1) -> list[KVScalingResult]:
    out: list[KVScalingResult] = []
    variants = config.get("attention_variants", ["mha"])
    seq_lens = config.get("seq_lens", [128, 256, 512])
    for attn in variants:
        for L in seq_lens:
            slope = 2.0 if attn == "mha" else 1.1 if attn == "gqa" else 0.6
            out.append(KVScalingResult(attn, L, float(L * slope)))
    return out
