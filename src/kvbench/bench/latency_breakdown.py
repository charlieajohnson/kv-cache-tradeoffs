from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from kvbench.models import DecoderOnlyConfig, SmallGPT
from kvbench.utils import get_device, runtime_report, set_seed


@dataclass
class LatencyBreakdownResult:
    attention: str
    attention_ms: float
    kv_ms: float
    mlp_ms: float
    overhead_ms: float
    runtime: dict[str, object]


def _resolve_n_kv_heads(variant: str, n_heads: int, n_kv_heads: int | None) -> int:
    if n_kv_heads is not None:
        return n_kv_heads
    if variant == "mqa":
        return 1
    if variant == "gqa":
        return max(1, n_heads // 2)
    return n_heads


def _build_model(config: dict, attention: str, seq_len: int) -> tuple[SmallGPT, torch.device]:
    set_seed(int(config.get("seed", 123)))
    device = get_device(prefer_cuda=config.get("prefer_cuda", True))
    d_model = int(config.get("d_model", 256))
    n_heads = int(config.get("n_heads", 4))
    n_kv_heads = _resolve_n_kv_heads(attention, n_heads, config.get("n_kv_heads", None))
    cfg = DecoderOnlyConfig(
        layers=int(config.get("layers", 4)),
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        seq_len=seq_len,
        attn_variant=attention,
    )
    return SmallGPT(cfg).to(device), device


def run_latency_breakdown(config: dict) -> LatencyBreakdownResult:
    attn = config.get("attention_variant", "mha")
    seq_len = int(config.get("seq_lens", [256])[0])
    decode_steps = int(config.get("decode_tokens", 1))
    max_seq_len = seq_len + decode_steps + 1

    prompt_len = max(1, seq_len - 1)
    batch_size = int(config.get("batch_size", 1))
    model, device = _build_model(config, attn, max_seq_len)

    prompt = torch.randint(
        0,
        int(config.get("vocab_size", 50_257)),
        (batch_size, prompt_len),
        device=device,
        dtype=torch.long,
    )
    next_token = prompt[:, -1:]
    with torch.no_grad():
        _, cache = model.forward_with_cache(prompt)

    t0 = perf_counter()
    total_attention_ms = 0.0
    total_cache_ms = 0.0
    total_mlp_ms = 0.0
    for _ in range(decode_steps):
        with torch.no_grad():
            _, cache, timing = model.forward_with_timing(next_token, cache)
        total_attention_ms += timing["attention_ms"]
        total_cache_ms += timing["cache_update_ms"]
        total_mlp_ms += timing["mlp_ms"]
        next_token = torch.randint(
            0,
            int(config.get("vocab_size", 50_257)),
            (batch_size, 1),
            device=device,
            dtype=torch.long,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (perf_counter() - t0) * 1000

    attention_ms = total_attention_ms
    mlp_ms = total_mlp_ms
    kv_ms = total_cache_ms
    overhead_ms = max(0.0, elapsed_ms - attention_ms - mlp_ms - kv_ms)
    return LatencyBreakdownResult(
        attention=attn,
        attention_ms=(attention_ms / max(decode_steps, 1)),
        kv_ms=(kv_ms / max(decode_steps, 1)),
        mlp_ms=(mlp_ms / max(decode_steps, 1)),
        overhead_ms=(overhead_ms / max(decode_steps, 1)),
        runtime=runtime_report(),
    )
