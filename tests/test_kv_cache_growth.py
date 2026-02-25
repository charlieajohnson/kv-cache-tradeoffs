from __future__ import annotations

from kvbench.utils.memory import estimate_kv_cache_bytes


def test_kv_cache_growth_linear_l():
    a = estimate_kv_cache_bytes(128, 4, 8, 32)
    b = estimate_kv_cache_bytes(256, 4, 8, 32)
    assert b == a * 2
