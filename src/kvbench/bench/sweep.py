from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompressionRun:
    variant: str
    compression_factor: int
    perplexity_delta: float


def run_compression_sweep(config: dict) -> list[CompressionRun]:
    factors = config.get("compression_factors", [1, 2, 4, 8])
    out: list[CompressionRun] = []
    for factor in factors:
        variant = "mqa" if factor == 8 else ("gqa" if factor > 1 else "mha")
        delta = 0.0 if factor == 1 else (0.2 * (factor - 1))
        out.append(CompressionRun(variant, int(factor), float(delta)))
    return out
