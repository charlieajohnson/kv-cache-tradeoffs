from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from kvbench.models import DecoderOnlyConfig, SmallGPT
from kvbench.utils import bytes_to_mib, get_device, runtime_report, set_seed


@dataclass
class KVScalingResult:
    attention: str
    seq_len: int
    ttft_ms: float
    tpot_ms: float
    kv_mib: float
    peak_vram_mib: float
    decode_tokens: int
    runtime: dict[str, object]


def _resolve_n_kv_heads(variant: str, n_heads: int, n_kv_heads: int | None) -> int:
    if n_kv_heads is not None:
        return n_kv_heads
    if variant == "mqa":
        return 1
    if variant == "gqa":
        return max(1, n_heads // 2)
    return n_heads


def _build_model(
    variant: str, config: dict, max_seq_len: int
) -> tuple[SmallGPT, torch.device]:
    seed = config.get("seed", 123)
    set_seed(seed)
    device = get_device(prefer_cuda=config.get("prefer_cuda", True))

    d_model = int(config.get("d_model", 256))
    n_heads = int(config.get("n_heads", 4))
    n_kv_heads = _resolve_n_kv_heads(
        variant, n_heads, config.get("n_kv_heads", None)
    )
    cfg = DecoderOnlyConfig(
        layers=int(config.get("layers", 4)),
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        seq_len=max_seq_len,
        attn_variant=variant,
    )
    model = SmallGPT(cfg).to(device)
    return model, device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _prompt_tensors(
    config: dict,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    vocab_size = int(config.get("vocab_size", 50_257))
    decode_tokens = int(config.get("decode_tokens", 32))
    token_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )
    return token_ids, decode_tokens


def run_kv_scaling(config: dict, max_batches: int = 1) -> list[KVScalingResult]:
    max_batches = max(1, int(max_batches))
    out: list[KVScalingResult] = []
    variants = config.get("attention_variants", ["mha"])
    seq_lens = config.get("seq_lens", [128, 256, 512])
    batch_size = int(config.get("batch_size", 1))
    max_seq_len = int(max(seq_lens) + int(config.get("decode_tokens", 32)))
    for attn in variants:
        model, device = _build_model(attn, config, max_seq_len)
        for seq_len in seq_lens:
            token_ids, decode_tokens = _prompt_tensors(
                config,
                int(seq_len),
                batch_size,
                device,
            )

            ttft_total_ms = 0.0
            tpot_total_ms = 0.0
            peak_vram = 0
            kv_bytes = 0

            for _ in range(max_batches):
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.empty_cache()

                t0 = perf_counter()
                with torch.no_grad():
                    _, cache = model.forward_with_cache(token_ids)
                _synchronize(device)
                ttft_ms = (perf_counter() - t0) * 1000

                decode_input = token_ids[:, -1:]
                current_cache = cache
                decode_times: list[float] = []
                for _ in range(decode_tokens):
                    start = perf_counter()
                    with torch.no_grad():
                        _, current_cache = model.forward_with_cache(decode_input, current_cache)
                    _synchronize(device)
                    decode_times.append((perf_counter() - start) * 1000)
                    decode_input = torch.randint(
                        0,
                        int(config.get("vocab_size", 50_257)),
                        (batch_size, 1),
                        device=device,
                        dtype=torch.long,
                    )

                ttft_total_ms += ttft_ms
                if decode_times:
                    tpot_total_ms += sum(decode_times) / len(decode_times)
                kv_bytes = sum(entry.num_bytes() for entry in current_cache)
                if device.type == "cuda":
                    peak_vram = max(peak_vram, torch.cuda.max_memory_allocated(device))

            out.append(
                KVScalingResult(
                    attention=attn,
                    seq_len=int(seq_len),
                    ttft_ms=ttft_total_ms / max_batches,
                    tpot_ms=tpot_total_ms / max_batches,
                    kv_mib=bytes_to_mib(kv_bytes),
                    peak_vram_mib=bytes_to_mib(peak_vram),
                    decode_tokens=decode_tokens,
                    runtime=runtime_report(),
                )
            )
    return out
