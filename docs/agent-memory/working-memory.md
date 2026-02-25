# Working memory

## Session notes

- Core benchmark data plane is now implemented with synthetic generation and real KV-cache timing.
- Attention modules support incremental cache forwarding: `(x, cache) -> (y, new_cache)`.
- Model block and top-level decoders now expose cache-aware forward paths for prefill/decode loops.

## Assumptions to revisit

- Quality/perplexity results remain synthetic phase-0 placeholders.
- KV measurements are from tensor allocations and CUDA memory counters, not trained-model logits benchmarks.
- A future iteration should add explicit benchmark result schema + artifact writer.
