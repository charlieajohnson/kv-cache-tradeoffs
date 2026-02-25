from __future__ import annotations

import torch


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    print("cuda", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
