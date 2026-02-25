# Paper outline: KV-cache scaling and attention variants

## Title

Memory-Throughput Tradeoffs in Transformer Inference: A Controlled Study of KV Cache Scaling, MQA, and GQA in Sub-100M Models

## Abstract

- Introduce attention-memory bottleneck and scaling law.
- Present controlled MHA vs GQA vs MQA comparison across sequence length and batch size.
- Emphasize reproducibility via versioned configs, deterministic seeds, and artifact logging.

## Research questions

1. How does KV memory scale empirically across variants?
2. What is the latency-vs-memory-vs-quality trade-off?
3. Where is the grouping instability threshold?
4. Can small models tolerate aggressive compression?

## Method

- Identical backbone and checkpoints across variants.
- Identical eval harness and fixed seeds (three repetitions).
- Config-first sweeps for KV length, batch size, and grouping factor.

## Evaluation

- Peak memory, tokens/sec, latency breakdown
- Perplexity
- Next-token accuracy
- Confidence intervals + paired comparisons
