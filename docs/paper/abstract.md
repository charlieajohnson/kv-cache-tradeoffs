# Abstract draft

Transformer inference is memory-bound, with KV cache growth scaling with sequence length and head count.
We present a controlled benchmark suite for sub-100M decoders evaluating MHA, GQA, and MQA under identical checkpoints.
Across sequence-length sweeps, MQA and GQA substantially reduce KV memory at the cost of modest quality degradation.
We report confidence intervals, latency decomposition, and instability thresholds in aggressive grouping regimes.

This work demonstrates a repeatable frontier-style methodology for memory-aware inference tradeoff analysis in constrained settings.
