from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def fig_quality(results, out_path: str | Path) -> None:
    fig, ax = plt.subplots()
    names = [r["attention"] for r in results]
    values = [r["perplexity"] for r in results]
    ax.bar(names, values)
    ax.set_ylabel("Perplexity")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
