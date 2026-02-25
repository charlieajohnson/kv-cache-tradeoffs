# Project status

## Current state

- Stage: P0 implemented (synthetic benchmarks + real cache timing)
- Focus: Solidify minimal command ergonomics and synthetic quality story.

## Risk log

- No model checkpoints in repo by design (not versioned).
- GPU-specific memory metrics need hardware-specific calibration.
- Statistical reporting not yet automated end-to-end.
- CLI metadata is now attached to benchmark results, but file-level artifact persistence is still missing.
- Quality metrics are placeholders; current values are synthetic/phase-2 only.

## Next actions

1. Add explicit JSONL/CSV artifact writers for each benchmark run.
2. Add deterministic warmup/eval protocol before throughput measurement.
3. Add real quality/perplexity workflow or tighten Phase 2 scope in docs.
