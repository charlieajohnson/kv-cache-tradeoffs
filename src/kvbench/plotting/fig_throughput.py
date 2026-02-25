from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def fig_throughput(results, out_path: str | Path) -> None:
    df = pd.DataFrame(results)
    fig, ax = plt.subplots()
    for name, group in df.groupby("attention"):
        ax.plot(group["seq_len"], group["tokens_per_sec"], label=name)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Tokens / sec")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)
