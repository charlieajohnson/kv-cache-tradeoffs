from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def fig_latency_breakdown(results, out_path: str | Path) -> None:
    labels = ["attention", "kv", "mlp", "overhead"]
    vals = [results["attention_ms"], results["kv_ms"], results["mlp_ms"], results["overhead_ms"]]
    fig, ax = plt.subplots()
    ax.bar(labels, vals)
    ax.set_ylabel("ms")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
