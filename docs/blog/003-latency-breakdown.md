# Blog: Latency breakdown in decoder inference

We separate inference latency into:

- attention projection + softmax
- KV cache memory access
- MLP forward pass
- Python overhead and data movement

This decomposition is required to interpret long-context throughput effects.
