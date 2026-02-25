from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path: str | Path, strict: bool = True):
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return {"missing_keys": missing, "unexpected_keys": unexpected}
