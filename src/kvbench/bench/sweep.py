from __future__ import annotations

from dataclasses import dataclass

from kvbench.utils import runtime_report


@dataclass
class CompressionRun:
    variant: str
    compression_factor: int
    perplexity_delta: float | None
    notes: str
    runtime: dict[str, object]


def run_compression_sweep(config: dict) -> list[CompressionRun]:
    factors = config.get("compression_factors", [1, 2, 4, 8])
    out: list[CompressionRun] = []
    note = "Phase 2: quality runs are synthetic placeholders; only systems metrics are implemented."
    for factor in factors:
        variant = "mqa" if factor == 8 else ("gqa" if factor > 1 else "mha")
        if factor == 1:
            delta = 0.0
        else:
            delta = None
        out.append(
            CompressionRun(
                variant=variant,
                compression_factor=int(factor),
                perplexity_delta=delta,
                notes=note,
                runtime=runtime_report(),
            )
        )
    return out
