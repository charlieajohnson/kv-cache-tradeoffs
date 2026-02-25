from __future__ import annotations

from kvbench.models import SmallGPT, DecoderOnlyConfig
from kvbench.models.checkpoints import save_checkpoint


def main():
    cfg = DecoderOnlyConfig()
    model = SmallGPT(cfg)
    save_checkpoint(model, "runs/example_model.pt")


if __name__ == "__main__":
    main()
