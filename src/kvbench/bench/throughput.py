from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from kvbench.models import DecoderOnlyConfig, SmallGPT
from kvbench.utils import bytes_to_mib, get_device, runtime_report, set_seed


@dataclass
class ThroughputResult:
    attention: str
    seq_len: int
    tokens_per_sec: float
    runtime: dict[str, object]
    peak_vram_mib: float


def run_throughput(config: dict) -> list[ThroughputResult]:
    def _resolve_n_kv_heads(variant: str, n_heads: int, n_kv_heads: int | None) -> int:
        if n_kv_heads is not None:
            return n_kv_heads
        if variant == "mqa":
            return 1
        if variant == "gqa":
            return max(1, n_heads // 2)
        return n_heads

    def _build_model(variant: str, max_seq_len: int) -> tuple[SmallGPT, torch.device]:
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

    def _prompt(token_count: int, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(
            0,
            int(config.get("vocab_size", 50_257)),
            (batch_size, token_count),
            device=device,
            dtype=torch.long,
        )

    def _sync(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    out: list[ThroughputResult] = []
    variants = config.get("attention_variants", ["mha"])
    seq_lens = config.get("seq_lens", [128, 256, 512, 1024])
    batch_size = int(config.get("batch_size", 1))
    decode_tokens = int(config.get("decode_tokens", 128))
    max_seq_len = int(max(seq_lens) + decode_tokens + 1)
    max_batches = int(config.get("max_batches", 1))
    max_batches = max(1, max_batches)

    for attn in variants:
        model, device = _build_model(attn, max_seq_len)
        for L in seq_lens:
            prompt = _prompt(int(L), batch_size, device)
            warmup_input = prompt[:, -1:]

            with torch.no_grad():
                _, cache = model.forward_with_cache(prompt[:, :-1] if L > 1 else prompt)
            peak_vram = 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            decoded = 0
            elapsed = 0.0

            for _ in range(max_batches):
                decode_input = warmup_input
                _sync(device)
                start = perf_counter()
                for _ in range(decode_tokens):
                    with torch.no_grad():
                        _, cache = model.forward_with_cache(decode_input, cache)
                    _sync(device)
                    decode_input = torch.randint(
                        0,
                        int(config.get("vocab_size", 50_257)),
                        (batch_size, 1),
                        device=device,
                        dtype=torch.long,
                    )
                    decoded += 1
                _sync(device)
                elapsed += perf_counter() - start
                if device.type == "cuda":
                    peak_vram = max(peak_vram, torch.cuda.max_memory_allocated(device))
                with torch.no_grad():
                    _, cache = model.forward_with_cache(prompt[:, :-1] if L > 1 else prompt)
            tps = decoded / elapsed if elapsed > 0 else 0.0
            out.append(
                ThroughputResult(
                    attention=attn,
                    seq_len=L,
                    tokens_per_sec=float(tps),
                    runtime=runtime_report(),
                    peak_vram_mib=bytes_to_mib(peak_vram),
                )
            )
    return out
