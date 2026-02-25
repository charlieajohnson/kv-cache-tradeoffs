# Blog: KV cache scaling in small transformers

The first set of experiments profiles memory growth of attention variants as sequence length increases.
The linear relation `O(L * H * d_k)` remains clear even in sub-100M models.
