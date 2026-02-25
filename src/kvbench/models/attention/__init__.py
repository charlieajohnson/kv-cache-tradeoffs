from .gqa import GQA
from .kv_cache import KVCacheState
from .mha import MHA
from .mqa import MQA


def get_attention_class(name: str):
    name = (name or "mha").lower()
    if name == "mha":
        return MHA
    if name == "mqa":
        return MQA
    if name == "gqa":
        return GQA
    raise ValueError(f"unknown attention variant: {name}")

__all__ = ["GQA", "MHA", "MQA", "KVCacheState", "get_attention_class"]
