from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def fig_compression_threshold(results, out_path: str | Path) -> None:
    fig, ax = plt.subplots()
    x = [r["compression_factor"] for r in results]
    y = [r["perplexity_delta"] for r in results]
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Compression factor")
    ax.set_ylabel("Perplexity delta")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
