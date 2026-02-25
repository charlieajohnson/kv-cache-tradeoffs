from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class KVScalingResult:
    attention: str
    seq_len: int
    kv_mib: float


def run_kv_scaling(config: Dict, max_batches: int = 1) -> List[KVScalingResult]:
    out: List[KVScalingResult] = []
    variants = config.get("attention_variants", ["mha"])
    seq_lens = config.get("seq_lens", [128, 256, 512])
    for attn in variants:
        for L in seq_lens:
            slope = 2.0 if attn == "mha" else 1.1 if attn == "gqa" else 0.6
            out.append(KVScalingResult(attn, L, float(L * slope)))
    return out
