from __future__ import annotations

import torch

from kvbench.models.attention import GQA, MHA, MQA


def _run_shape(attn_cls):
    m = attn_cls(64, 8, n_kv_heads=4)
    x = torch.randn(2, 16, 64)
    y = m(x)
    assert y.shape == x.shape


def test_mha_shape():
    _run_shape(MHA)


def test_gqa_shape():
    _run_shape(GQA)


def test_mqa_shape():
    _run_shape(MQA)
