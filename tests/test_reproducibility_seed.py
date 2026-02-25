from __future__ import annotations

import torch

from kvbench.utils.seeding import set_seed


def test_deterministic_seed():
    set_seed(1234)
    a = torch.rand(4)
    set_seed(1234)
    b = torch.rand(4)
    assert torch.allclose(a, b)
