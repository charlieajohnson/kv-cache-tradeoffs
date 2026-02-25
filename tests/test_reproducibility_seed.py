from __future__ import annotations

from kvbench.utils.seeding import set_seed
import torch


def test_deterministic_seed():
    set_seed(1234)
    a = torch.rand(4)
    set_seed(1234)
    b = torch.rand(4)
    assert torch.allclose(a, b)
