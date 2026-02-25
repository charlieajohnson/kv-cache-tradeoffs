from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def fig_kv_memory(results, out_path: str | Path) -> None:
    df = pd.DataFrame(results)
    fig, ax = plt.subplots()
    for name, group in df.groupby("attention"):
        ax.plot(group["seq_len"], group["kv_mib"], label=name)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("KV cache (MB)")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)
